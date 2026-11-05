# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""One-GPU remote-wait mechanism test, NOT DCP transport/numerical qualification.

Each worker imports real pinned CuTe code and uses one SM103 GPU. Positive workers
prepublish synthetic peer data. Negative workers leave exactly the selected kind
of publication missing; the native wait must fail-stop and poison that process's
CUDA context within the bound. Every negative worker is followed by a fresh
positive process. Never execute a negative case in a server/test-parent context.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
DEADLINE_NS = 50_000_000
LAUNCH_BOUND_SECONDS = 10.0
WORKER_BOUND_SECONDS = 240
KINDS = ("query", "lse", "partial")


def _save(path, report):
    Path(path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def _cuda_fault(error):
    text = str(error).lower()
    return any(
        phrase in text
        for phrase in (
            "illegal memory access",
            "illegal_address",
            "unspecified launch failure",
            "launch_failed",
            "device-side assert",
        )
    )


def _query_fixture(device, missing):
    import cutlass
    import cutlass.cute as cute

    from sgl_nvidia_mega.kimi_k3.cutedsl.common.host import stream, to_cute
    from sgl_nvidia_mega.kimi_k3.cutedsl.mla.query import MlaQuery

    class QueryWait:
        def __init__(self):
            self.query = MlaQuery(12, 1, dcp=8, deadline_ns=DEADLINE_NS)

        @cute.jit
        def __call__(self, mailbox, done, native_stream):
            self.kernel(mailbox, done).launch(
                grid=self.query.grid,
                block=256,
                stream=native_stream,
                use_pdl=False,
            )

        @cute.kernel
        def kernel(self, mailbox, done):
            tid, _, _ = cute.arch.thread_idx()
            # gathered reads only args[11]; all other entries are deliberately
            # unused. Invoke the production method, not a copied wait loop.
            args = (
                mailbox,
                mailbox,
                mailbox,
                mailbox,
                mailbox,
                mailbox,
                mailbox,
                mailbox,
                mailbox,
                mailbox,
                mailbox,
                mailbox,
            )
            self.query.gathered(args, tid)
            part, head, tile = cute.arch.block_idx()
            if tid == 0:
                done[(tile * 12 + head) * 8 + part] = cutlass.Int32(1)

    mailbox = torch.full(
        (64, 96, 576), 0xFF if missing else 0, dtype=torch.uint8, device=device
    )
    done = torch.zeros(96, dtype=torch.int32, device=device)
    args = to_cute(mailbox), to_cute(done), stream(device)
    compiled = cute.compile(QueryWait(), *args)
    compiled.to(device.index)

    def check():
        if not bool((done == 1).all()):
            raise AssertionError("Production query wait did not complete every CTA")
        if not bool((mailbox == 0).all()):
            raise AssertionError("Query wait mutated prepublished bytes")

    return lambda: compiled(*args), check


def _value_fixture(device, kind, missing):
    import cutlass

    from sgl_nvidia_mega.kimi_k3.cutedsl.common.host import layout, stream, to_cute
    from sgl_nvidia_mega.kimi_k3.cutedsl.mla.dcp import compile_value

    # Eight separate destination buffers on ONE device. Rank 0 writes only its
    # source plane; sources 1..7 are prepublished synthetic peers. No NCCL,
    # symmetric allocation, multicast or real cross-device transport is claimed.
    destinations, parities, sources, capacity, heads = 8, 2, 8, 1, 12
    partial = torch.empty(
        destinations,
        parities,
        sources,
        capacity,
        heads,
        512,
        dtype=torch.bfloat16,
        device=device,
    )
    lse = torch.empty(
        destinations,
        parities,
        sources,
        capacity,
        heads,
        dtype=torch.float32,
        device=device,
    )
    partial.view(torch.int32).fill_(-0x80000000)
    lse.view(torch.int32).fill_(-1)
    partial[:, 0, 1:].zero_()
    lse[:, 0, 1:].zero_()
    if missing and kind == "lse":
        lse[0, 0, 1].view(torch.int32).fill_(-1)
    if missing and kind == "partial":
        partial[0, 0, 1].view(torch.int32).fill_(-0x80000000)
    peers = torch.tensor(
        [partial[index].data_ptr() for index in range(destinations)],
        dtype=torch.int64,
        device=device,
    )
    lpeers = torch.tensor(
        [lse[index].data_ptr() for index in range(destinations)],
        dtype=torch.int64,
        device=device,
    )
    rows = torch.zeros(1, 1, 96, 512, dtype=torch.bfloat16, device=device)
    rows_lse = torch.zeros(1, 1, 96, dtype=torch.float32, device=device)
    lens = torch.ones(1, dtype=torch.int32, device=device)
    mailbox = torch.zeros(64, 96, 576, dtype=torch.uint8, device=device)
    uv = torch.zeros(12, 128, 512, dtype=torch.bfloat16, device=device).transpose(1, 2)
    projected = torch.zeros(1, 3648, dtype=torch.bfloat16, device=device)
    output = torch.full((1, 1536), float("nan"), dtype=torch.bfloat16, device=device)
    compiled = compile_value(
        12,
        8,
        1,
        capacity,
        layout(uv),
        3648,
        device.index,
        deadline_ns=DEADLINE_NS,
    )
    compiled.to(device.index)
    args = (
        to_cute(rows),
        to_cute(rows_lse, 4),
        to_cute(lens, 4),
        to_cute(partial[0, 0]),
        to_cute(lse[0, 0], 4),
        to_cute(peers, 8),
        to_cute(lpeers, 8),
        to_cute(mailbox[:capacity]),
        to_cute(uv),
        to_cute(projected[:, 2112:]),
        to_cute(lens, 4),
        to_cute(output),
        cutlass.Int32(0),
        cutlass.Int32(0),
        stream(device),
    )

    def check():
        if not bool((output == 0).all()):
            raise AssertionError("Production DcpValue positive result differs")
        if not bool((partial[0, 0].view(torch.int32) == -0x80000000).all()):
            raise AssertionError("Partial mailbox was not rearmed")
        if not bool((lse[0, 0].view(torch.int32) == -1).all()):
            raise AssertionError("LSE mailbox was not rearmed")
        if not bool((mailbox[0] == 0xFF).all()):
            raise AssertionError("Active query mailbox prefix was not rearmed")
        if not bool((mailbox[1:] == 0).all()):
            raise AssertionError("Inactive query mailbox tail changed")

    # Retain every destination allocation: pointer tables alone are not owners.
    owners = (
        partial,
        lse,
        peers,
        lpeers,
        rows,
        rows_lse,
        lens,
        mailbox,
        uv,
        projected,
        output,
    )

    def launch():
        if len(owners) != 11:
            raise AssertionError("Fixture ownership changed")
        compiled(*args)

    return launch, check


def worker(kind, missing, report_path, *, plugin_import_root=None):
    import_root = Path(plugin_import_root or ROOT).resolve()
    report = {
        "scope": "one_gpu_dcp_wait_mechanism_only",
        "kind": kind,
        "missing": missing,
        "pid": os.getpid(),
        "deadline_ns": DEADLINE_NS,
        "status": "FAIL",
        "distributed_transport": "NOT_RUN",
        "plugin_import_root": str(import_root),
    }
    _save(report_path, report)
    try:
        if sys.platform != "linux" or not torch.cuda.is_available():
            raise RuntimeError("Native Linux CUDA worker required")
        if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (
            10,
            3,
        ):
            raise RuntimeError("Exactly one visible SM103 GPU required")
        if any(
            name == "sgl_nvidia_mega" or name.startswith("sgl_nvidia_mega.")
            for name in sys.modules
        ):
            raise RuntimeError("Fresh plugin process required")
        sys.path.insert(0, str(import_root))
        torch.cuda.set_device(0)
        device = torch.device("cuda", 0)
        launch, check = (
            _query_fixture(device, missing)
            if kind == "query"
            else _value_fixture(device, kind, missing)
        )
        for name, module in tuple(sys.modules.items()):
            if name == "sgl_nvidia_mega" or name.startswith("sgl_nvidia_mega."):
                origin = getattr(module, "__file__", None)
                if origin is None or not Path(origin).resolve().is_relative_to(
                    import_root
                ):
                    raise RuntimeError(f"Plugin import escaped pinned source: {name}")
        torch.cuda.synchronize(device)
        report["stage"] = "launch"
        _save(report_path, report)
        started = time.monotonic()
        try:
            launch()
            torch.cuda.synchronize(device)
        except Exception as error:
            report["launch_seconds"] = time.monotonic() - started
            if not missing or not _cuda_fault(error):
                raise
            if not DEADLINE_NS / 1e9 <= report["launch_seconds"] < LAUNCH_BOUND_SECONDS:
                raise AssertionError(
                    "Fault occurred outside the native deadline window"
                ) from error
            report["cuda_fault"] = str(error)
            # The deliberate fault must poison THIS worker's context. Recovery
            # is tested only by a new subprocess, never by suppressing this error.
            try:
                torch.empty(1, device=device)
                torch.cuda.synchronize(device)
            except Exception as sticky:
                if not _cuda_fault(sticky):
                    raise
                report["sticky_cuda_fault"] = str(sticky)
            else:
                raise AssertionError("Expected a sticky poisoned CUDA context")
            report["status"] = "FAIL_STOP_OBSERVED"
        else:
            report["launch_seconds"] = time.monotonic() - started
            if missing:
                raise AssertionError("Missing publisher unexpectedly completed")
            check()
            report["status"] = "PASS"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    _save(report_path, report)
    return report


class TestDcpDeadlineFixture(unittest.TestCase):
    def test_immediate_sticky_fault_is_not_a_deadline_success(self):
        def bad_launch():
            raise RuntimeError("CUDA error: an illegal memory access was encountered")

        from contextlib import ExitStack

        with tempfile.TemporaryDirectory(prefix="kimi-dcp-deadline-host-") as temporary:
            with ExitStack() as stack:
                stack.enter_context(patch.object(sys, "platform", "linux"))
                stack.enter_context(patch.dict(sys.modules))
                for name in list(sys.modules):
                    if name == "sgl_nvidia_mega" or name.startswith("sgl_nvidia_mega."):
                        del sys.modules[name]
                stack.enter_context(
                    patch.object(torch.cuda, "is_available", return_value=True)
                )
                stack.enter_context(
                    patch.object(torch.cuda, "device_count", return_value=1)
                )
                stack.enter_context(
                    patch.object(
                        torch.cuda, "get_device_capability", return_value=(10, 3)
                    )
                )
                stack.enter_context(patch.object(torch.cuda, "set_device"))
                stack.enter_context(patch.object(torch.cuda, "synchronize"))
                stack.enter_context(
                    patch.object(time, "monotonic", side_effect=(10.0, 10.001))
                )
                stack.enter_context(
                    patch.dict(
                        worker.__globals__,
                        _query_fixture=lambda *_: (bad_launch, lambda: None),
                    )
                )
                report = worker("query", True, Path(temporary) / "report.json")
            self.assertEqual(report["status"], "FAIL")
            self.assertIn("outside the native deadline window", report["error"])
            self.assertNotIn("sticky_cuda_fault", report)

    def test_value_mailbox_view_matches_compiled_capacity(self):
        import ast
        import inspect
        import textwrap

        tree = ast.parse(textwrap.dedent(inspect.getsource(_value_fixture)))
        views = [
            node.args[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "to_cute"
            and node.args
        ]
        self.assertEqual(
            sum(ast.unparse(view) == "mailbox[:capacity]" for view in views), 1
        )
        self.assertFalse(
            any(isinstance(view, ast.Name) and view.id == "mailbox" for view in views)
        )

    def test_worker_refuses_non_native_execution_before_fixture_import(self):
        with tempfile.TemporaryDirectory(prefix="kimi-dcp-deadline-host-") as temporary:
            with patch.object(sys, "platform", "darwin"):
                report = worker("query", False, Path(temporary) / "report.json")
            self.assertEqual(report["status"], "FAIL")
            self.assertIn("Native Linux CUDA", report["error"])
            self.assertNotIn("stage", report)

    def test_unrelated_errors_cannot_count_as_expected_fail_stop(self):
        for message in (
            "timeout",
            "CUDA unavailable",
            "compilation failed",
            "out of memory",
        ):
            self.assertFalse(_cuda_fault(RuntimeError(message)))
        self.assertTrue(
            _cuda_fault(
                RuntimeError("CUDA error: an illegal memory access was encountered")
            )
        )


@unittest.skipUnless(torch.cuda.is_available(), "native CUDA qualification required")
class TestDcpDeadlineCuda(unittest.TestCase):
    def _case(self, kind):
        reports = []
        with tempfile.TemporaryDirectory(prefix="kimi-dcp-deadline-") as temporary:
            for ordinal, missing in enumerate((False, True, False)):
                path = Path(temporary) / f"{ordinal}.json"
                command = [
                    sys.executable,
                    "-B",
                    str(Path(__file__).resolve()),
                    "--worker",
                    kind,
                    "--report",
                    str(path),
                ]
                if missing:
                    command.append("--missing")
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=WORKER_BOUND_SECONDS,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                report = json.loads(path.read_text())
                self.assertEqual(
                    report["status"],
                    "FAIL_STOP_OBSERVED" if missing else "PASS",
                    report,
                )
                self.assertEqual(report["distributed_transport"], "NOT_RUN")
                reports.append(report)
        self.assertEqual(len({report["pid"] for report in reports}), 3)

    def test_query_wait_positive_missing_then_fresh_recovery(self):
        self._case("query")

    def test_lse_wait_positive_missing_then_fresh_recovery(self):
        self._case("lse")

    def test_partial_wait_positive_missing_then_fresh_recovery(self):
        self._case("partial")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=KINDS)
    parser.add_argument("--missing", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--plugin-import-root", type=Path)
    args, remaining = parser.parse_known_args()
    if args.worker:
        if args.report is None:
            parser.error("--worker requires --report")
        outcome = worker(
            args.worker,
            args.missing,
            args.report,
            plugin_import_root=args.plugin_import_root,
        )
        raise SystemExit(
            0 if outcome["status"] in ("PASS", "FAIL_STOP_OBSERVED") else 1
        )
    unittest.main(argv=[__file__, *remaining])

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K3.1 MK token-split indexer: the poll deadline (rule Q, fail closed). TRAP TESTS: these
deliberately fault the GPU (a device trap in a child process), so they run only on a dedicated node, in tasks whose name says
"trap". Split out of test_k31_tsi.py unchanged
(same child, same assertions); the shared helpers stay there.

- test_trap_control: the child with every source pushing completes with the exact output (positive control).
- test_deadline_traps[never | stale]: a source that never pushes (zero regions), or whose slots hold the previous
  forward's records (stale tag), must stop the reader at its deadline with a CUDA error, and nothing may be
  published: the parent's output and fallback buffers (CUDA IPC, written only by the faulting launch) stay poisoned.

Mutations (MUTATIONS of test_k31_tsi.py): stale_tag (any nonzero tag accepted) -> test_deadline_traps[stale].
Run: K31_PLUGIN=<tree> bash run_suite.sh OUT test/kernels/k31_rubin/test_k31_tsi_trap.py stale_tag
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

import pytest
import torch

from test_k31_tsi import (  # noqa: F401  (MUTATIONS: the shared mutation table; importing it loads the plugin)
    DEV, MUTATIONS, POISON, Q, T, W, Case, full_logits, needs_gpu, prepare_all, ranks, reference, run_layer,
)


# ------------------------------------------------------------------------------------------------ deadline traps
def _trap_child(scenario, out, fallback, marker):
    """Child process (spawned): emulate 16 ranks with a 0.3 s deadline; rank 0's final poll writes into the parent's
    ``out`` / ``fallback`` (CUDA IPC). Scenarios: never (source 1 never pushed: zero regions), stale (source 1's slots
    hold the previous forward's records), control (every source pushes)."""
    def mark(text):
        with open(marker, "a") as f:
            f.write(text + "\n")

    try:
        case = Case([3000], seed=55)
        _, states, _ = ranks(case.rows, case.width, deadline_s=0.3)
        local = [torch.empty(case.rows, T.TOPK, dtype=torch.int32, device=DEV) for _ in range(W)]
        ordinal = 4
        if scenario != "never":  # forward 1: every rank, complete
            prepare_all(states, case)
            run_layer(states, case, ordinal, local)
        prepare_all(states, case)  # forward 2: new epoch, same ordinal (same parity, new tag)
        case.new_query()
        for r, st in enumerate(states):
            if r != 1 or scenario == "control":
                st.select(ordinal, case.query, case.gates, case.cache, local[r], phases=T.PUSH)
        torch.cuda.synchronize()
        mark("READY")
        st0 = states[0]
        st0.fallback = fallback
        st0.select(ordinal, case.query, case.gates, case.cache, out, phases=T.POLL)
        torch.cuda.synchronize()
        mark("COMPLETED")
        if scenario == "control":
            ref, flags, _ = reference(case, full_logits(case))
            mark(f"MATCH {bool(torch.equal(out, ref))} FLAGS {bool(torch.equal(fallback.cpu(), flags))}")
    except BaseException as e:  # noqa: BLE001  (the trap surfaces as a CUDA error on the next sync)
        mark(f"ERROR {e!r}"[:2000])
        sys.stdout.flush()
        os._exit(3)
    sys.stdout.flush()
    os._exit(0)


def run_trap_child(scenario):
    import torch.multiprocessing as mp

    out = torch.full((Q, T.TOPK), POISON, dtype=torch.int32, device=DEV)
    fallback = torch.full((Q,), 7, dtype=torch.int32, device=DEV)
    torch.cuda.synchronize()
    fd, marker = tempfile.mkstemp(prefix="k31tsi_trap_")
    os.close(fd)
    t0 = time.time()
    proc = mp.get_context("spawn").Process(target=_trap_child, args=(scenario, out, fallback, marker))
    proc.start()
    proc.join(timeout=900)
    if proc.is_alive():
        proc.kill()
        proc.join()
    text = open(marker).read()
    os.unlink(marker)
    print("TRAP_CHILD", scenario, "exit", proc.exitcode, f"{time.time() - t0:.1f}s", text.replace("\n", " | ")[:600],
          flush=True)
    torch.cuda.synchronize()
    return proc.exitcode, text, out, fallback


@needs_gpu
def test_trap_control():
    """Positive control of the trap tests: the same child with every source pushing completes with the exact output."""
    code, text, out, fallback = run_trap_child("control")
    assert code == 0 and "COMPLETED" in text and "MATCH True FLAGS True" in text, text
    assert bool((out != POISON).all()) and bool((fallback == 0).all())


@needs_gpu
@pytest.mark.parametrize("scenario", ["never", "stale"])
def test_deadline_traps(scenario):
    """A source that never pushes (zero slots) or whose slots hold the previous forward's records (stale tag) must stop
    the reader at its deadline with a CUDA error, and nothing may be published: the parent's output and fallback
    buffers (written only by the trapping launch) stay poisoned."""
    code, text, out, fallback = run_trap_child(scenario)
    assert "READY" in text, f"child failed before the poll: {text}"
    assert "COMPLETED" not in text and code != 0, f"the poll accepted a missing/stale source: {text}"
    assert "CUDA" in text or "launch failure" in text or "illegal instruction" in text, text
    assert bool((out == POISON).all()) and bool((fallback == 7).all()), "a trapped merge published output"

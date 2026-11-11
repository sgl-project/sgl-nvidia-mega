# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Benchmark-only switches for served A/B timing (``benchmark/ab_step_times.py``). Both
are off unless their variable is set; neither changes a kernel of the step.

``SGLANG_NVIDIA_MEGAKERNEL_BENCH_FIXED_ROUTING`` (benchmark only: outputs are wrong).
Every covered MoE layer, target and draft, routes each row to the experts of a fixed
table instead of the router's choice, so two arms do the same expert GEMMs every step.
The router GEMM, top-k, route record, GEMM1 / GEMM2, finalize and hop B all run as
before on the real weights: on the fast-gate path (``gate`` and ``gate_route``) the
top-k kernel reads logits built from the table instead of the router's (the same
selection kernel picks the table's experts); on SGLang's top-k path (the hash layers)
the routed pair takes the table's ids and weights. Values:

- ``synth[:d=19,half0=9,draft_d=7,draft_half0=4,seed=0]``: per layer ``d`` distinct
  global experts (``half0`` of them in the first expert-parallel half, ids below
  experts / 2), dealt over the rows' top-k slots cyclically; a different random set
  per layer, the same on every rank.
- ``table:<file.npz>``: ``target_<layer>`` / ``draft_<layer>`` arrays [rows, top_k] of
  global ids (e.g. a recorded step, ``benchmark/ab_step_times.py table``); rows past
  the table's repeat it.

Refused unless simulated acceptance is on (``SGLANG_SIMULATE_ACC_LEN`` > 0): an
accuracy run never sets it.

``SGLANG_NVIDIA_MEGAKERNEL_BENCH_STEP_TIMES=<directory>``: each covered verify and draft
graph replay writes a %globaltimer stamp (and its first row's position) from a one-CTA
kernel (sm103/src/bench_stamp.cu) forked at the start of the forward and joined at its
end, into a pinned ring a
host thread drains to ``<directory>/events-r<rank>.bin`` (int64 [time ns, meta, position,
seq]). On TP rank 0 the same kernel also writes, per MoE layer of the graph's previous
replay, the number of distinct experts and how many lie in the first half
(``counts-r0.bin``); ``layout-r<rank>.json`` names the layers. With
``SGLANG_NVIDIA_MEGAKERNEL_BENCH_STEP_TIMES_ENDS=1`` a second stamp is issued inline at
the end of each forward (adds that kernel to the critical path); with
``SGLANG_NVIDIA_MEGAKERNEL_BENCH_STEP_TIMES_IDS=1`` rank 0 also writes the raw ids
(``ids-r0.bin``, for recording a routing table).
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import logging
import os
import threading
import time
from functools import cache
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

FIXED_ENV = "SGLANG_NVIDIA_MEGAKERNEL_BENCH_FIXED_ROUTING"
TIMES_ENV = "SGLANG_NVIDIA_MEGAKERNEL_BENCH_STEP_TIMES"
ENDS_ENV = "SGLANG_NVIDIA_MEGAKERNEL_BENCH_STEP_TIMES_ENDS"
IDS_ENV = "SGLANG_NVIDIA_MEGAKERNEL_BENCH_STEP_TIMES_IDS"

# event kinds (meta bits 0-7)
VERIFY, DRAFT, VERIFY_END, DRAFT_END = 0, 1, 2, 3
KIND = {"verify": VERIFY, "draft": DRAFT}

# fixed tables cover the widest routed step
FIXED_MAX_ROWS = 48
TARGET = (384, 6)  # (router experts, top-k) of the DSV4.1 target; else the draft

RING = 16384  # events; the drain keeps up at ~1 kHz with a 0.2 s poll
MAX_SLOTS = 1024  # captured or warm-up forwards with a table
MAX_ENTRIES = 64  # MoE layers per forward (bench_stamp.cu kMaxEntries)
IDS_STRIDE = 2048  # int16 ids per event in the raw ids ring
STAMP_THREADS = 256


def _flag(name: str) -> bool:
    return os.environ.get(name, "") not in ("", "0")


# --------------------------------------------------------------------------- fixed routing


def _parse_fixed(raw: str) -> dict | None:
    raw = raw.strip()
    if raw in ("", "0"):
        return None
    kind, _, rest = raw.partition(":")
    if kind == "table":
        if not rest:
            raise ValueError(f"{FIXED_ENV}=table:<file.npz> needs a file")
        return {"mode": "table", "path": rest}
    if kind != "synth":
        raise ValueError(
            f"{FIXED_ENV}: expected synth[:k=v,...] or table:<file>, got {raw!r}"
        )
    # defaults: the mean distinct experts per layer of a served TP4 run at forced
    # acceptance 3.51 (target 18.7 at 6 rows, about half in each EP half; draft 7.2)
    spec = {
        "mode": "synth",
        "d": 19,
        "half0": 9,
        "draft_d": 7,
        "draft_half0": 4,
        "seed": 0,
    }
    for item in filter(None, rest.split(",")):
        k, _, v = item.partition("=")
        if k not in spec or k == "mode":
            raise ValueError(f"{FIXED_ENV}: unknown key {k!r}")
        spec[k] = int(v)
    return spec


@cache
def fixed_spec() -> dict | None:
    """The fixed-routing request (None when off). Raises when it is set without
    simulated acceptance, so the server refuses to start rather than serve wrong
    tokens to an accuracy run."""
    spec = _parse_fixed(os.environ.get(FIXED_ENV, ""))
    if spec is None:
        return None
    try:
        acc = float(os.environ.get("SGLANG_SIMULATE_ACC_LEN", "-1"))
    except ValueError:
        acc = -1.0
    if acc <= 0:
        raise RuntimeError(
            f"{FIXED_ENV} is benchmark only (the MoE ignores the router, outputs are "
            "wrong) and needs simulated acceptance (SGLANG_SIMULATE_ACC_LEN > 0); unset "
            "it for any accuracy run"
        )
    logger.warning(
        "NVIDIA megakernel: %s=%s: BENCHMARK ONLY, every covered MoE layer routes to "
        "a fixed expert table, outputs are wrong",
        FIXED_ENV,
        os.environ.get(FIXED_ENV),
    )
    return spec


def _synth_ids(spec, kind: str, layer: int, experts: int, top_k: int) -> np.ndarray:
    d, h0 = (
        (spec["d"], spec["half0"])
        if kind == "target"
        else (spec["draft_d"], spec["draft_half0"])
    )
    half = experts // 2
    d = max(top_k, min(d, experts))
    h0 = max(0, min(h0, d, half))
    if d - h0 > experts - half:
        h0 = d - (experts - half)
    rng = np.random.default_rng([spec["seed"], 1 if kind == "draft" else 0, layer])
    chosen = np.concatenate(
        [
            rng.choice(half, h0, replace=False),
            half + rng.choice(experts - half, d - h0, replace=False),
        ]
    )
    rng.shuffle(chosen)
    slots = np.arange(FIXED_MAX_ROWS * top_k).reshape(FIXED_MAX_ROWS, top_k)
    return chosen[slots % d].astype(np.int64)


@cache
def _table(path: str) -> dict:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def _table_ids(spec, kind: str, layer: int, top_k: int) -> np.ndarray:
    table = _table(spec["path"])
    key = f"{kind}_{layer}"
    if key not in table:
        raise KeyError(
            f"{FIXED_ENV}: {spec['path']} has no {key} (has {sorted(table)})"
        )
    ids = np.asarray(table[key], dtype=np.int64).reshape(-1, top_k)
    return ids[np.arange(FIXED_MAX_ROWS) % ids.shape[0]]


def _emulate_topk(logits: torch.Tensor, bias: torch.Tensor, top_k: int) -> torch.Tensor:
    """gate_topk.cuh's selection on the CPU (enough to check the synthesized logits)."""
    sp = torch.where(logits > 20.0, logits, torch.log1p(torch.exp(logits)))
    return torch.topk(torch.sqrt(sp) + bias, top_k, dim=-1).indices


class FixedLayer:
    """One MoE layer's fixed routing: the table's ids / weights (SGLang's top-k path)
    and router logits whose top-k is the table (fast-gate path)."""

    def __init__(
        self, kind: str, layer: int, ids: np.ndarray, bias, scale: float, device
    ):
        self.kind, self.layer = kind, layer
        rows, top_k = ids.shape
        experts = TARGET[0] if kind == "target" else None
        ids_t = torch.from_numpy(ids)
        self.logits = None
        weights = torch.full((rows, top_k), scale / top_k, dtype=torch.float32)
        if bias is not None:
            b = bias.detach().float().cpu()
            experts = b.numel()
            logits = torch.full((rows, experts), -30.0, dtype=torch.float32)
            # chosen expert at rank r: biased score C_r = max(bias) + 5 + (top_k - r) / 4,
            # i.e. logit (C_r - b)^2 (> 20, so softplus is the identity)
            c = b.max() + 5.0 + (top_k - torch.arange(top_k, dtype=torch.float32)) / 4.0
            act = c.unsqueeze(0) - b[ids_t]
            logits.scatter_(1, ids_t, act * act)
            got = _emulate_topk(logits, b, top_k)
            if not torch.equal(got.sort(dim=-1).values, ids_t.sort(dim=-1).values):
                raise RuntimeError(
                    f"fixed routing: logits for {kind} layer {layer} do not select the table"
                )
            act = torch.sqrt(logits.gather(1, ids_t))
            weights = act / act.sum(dim=-1, keepdim=True) * scale
            self.logits = logits.to(device)
        self.ids32 = ids_t.to(torch.int32).to(device)
        self.ids64 = ids_t.to(device)
        self.weights = weights.to(device)
        self.distinct = len(np.unique(ids[: min(rows, 6)]))
        self.experts = experts

    def router_logits(self, real: torch.Tensor) -> torch.Tensor:
        return self.logits[: real.size(0)]

    def topk(self, ids: torch.Tensor, weights: torch.Tensor):
        rows = ids.size(0)
        table = self.ids64 if ids.dtype == torch.int64 else self.ids32
        return table[:rows], self.weights[:rows].to(weights.dtype)


_BOUND: list = []


def bind_fixed(mlp, gemm, bias, scale: float):
    """The layer's FixedLayer when fixed routing is on, else None (called at build)."""
    spec = fixed_spec()
    if spec is None:
        return None
    kind = "target" if (gemm.router_experts, gemm.top_k) == TARGET else "draft"
    layer = int(mlp.layer_id)
    if spec["mode"] == "synth":
        ids = _synth_ids(spec, kind, layer, gemm.router_experts, gemm.top_k)
    else:
        ids = _table_ids(spec, kind, layer, gemm.top_k)
    fixed = FixedLayer(kind, layer, ids, bias, scale, gemm.clamp.device)
    _BOUND.append(fixed)
    return fixed


def fixed_summary() -> str:
    """For the startup line: what fixed routing covers ("" when off)."""
    if fixed_spec() is None:
        return ""
    by = {}
    for f in _BOUND:
        by.setdefault(f.kind, []).append(f.distinct)
    parts = [
        f"{k} {len(v)} layers, distinct experts at 6 rows {min(v)}-{max(v)}"
        for k, v in sorted(by.items())
    ]
    return f"; FIXED ROUTING (benchmark only, outputs wrong): {', '.join(parts) or 'no layer'}"


# --------------------------------------------------------------------------- step times


@cache
def _stamp_kernel(device_index: int):
    from sgl_nvidia_mega.common.jit import compile_source, load

    source = (
        Path(__file__).resolve().parents[1] / "sm103" / "src" / "bench_stamp.cu"
    ).read_text()
    major, minor = torch.cuda.get_device_capability(device_index)
    with torch.cuda.device(device_index):
        cubin = compile_source(
            source,
            "bench_stamp",
            f"sm_{major}{minor}a",
            ("-std=c++17", "-default-device"),
        )
        return load(cubin, "dsv41_bench_stamp", (STAMP_THREADS, 1, 1), 0)


def _launch_plain(kernel, args, stream: int) -> None:
    """cuLaunchKernelEx without PDL: a stamp never launches early."""
    from sgl_nvidia_mega.common.jit import check, driver

    cu = driver()
    addresses = [ctypes.addressof(a) for a in args]
    array = (ctypes.c_void_p * len(addresses))(*addresses)
    config = cu.CUlaunchConfig()
    config.gridDimX = config.gridDimY = config.gridDimZ = 1
    config.blockDimX, config.blockDimY, config.blockDimZ = kernel.block
    config.sharedMemBytes = 0
    config.hStream = cu.CUstream(stream)
    config.numAttrs = 0
    check(
        cu.cuLaunchKernelEx(config, kernel.function, ctypes.addressof(array), 0),
        "bench stamp",
    )


class StepTimer:
    """The stamp kernel's rings and the host thread that drains them to files."""

    def __init__(self, directory: str, device: torch.device):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        dist = torch.distributed
        self.rank = (
            dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        )
        self.device = device
        self.kernel = _stamp_kernel(device.index)
        self.events = torch.zeros((RING, 4), dtype=torch.int64, pin_memory=True)
        lead = self.rank == 0
        self.counts = (
            torch.zeros((RING, 2 * MAX_ENTRIES), dtype=torch.int16, pin_memory=True)
            if lead
            else None
        )
        self.ids = (
            torch.zeros((RING, IDS_STRIDE), dtype=torch.int16, pin_memory=True)
            if lead and _flag(IDS_ENV)
            else None
        )
        self.table = torch.zeros(
            (MAX_SLOTS, 1 + 2 * MAX_ENTRIES), dtype=torch.int64, pin_memory=True
        )
        # the stamp's device copy of each slot's table row (made by its first replay)
        self.dev_table = torch.zeros(
            (MAX_SLOTS, 1 + 2 * MAX_ENTRIES), dtype=torch.int64, device=device
        )
        self.dev_valid = torch.zeros(MAX_SLOTS, dtype=torch.int32, device=device)
        self.counter = torch.zeros(1, dtype=torch.int32, device=device)
        self.side = torch.cuda.Stream(device)
        self.ends = _flag(ENDS_ENV)
        self.layouts: list[dict] = []
        self._held: list = []  # the id tensors the stamps read: never freed or reused
        self.current: dict | None = None
        self._next = 0
        self._files = {
            "events": open(self.dir / f"events-r{self.rank}.bin", "ab"),
            "counts": open(self.dir / f"counts-r{self.rank}.bin", "ab")
            if lead
            else None,
            "ids": open(self.dir / f"ids-r{self.rank}.bin", "ab")
            if self.ids is not None
            else None,
            # (host unix ns at the drain, the newest drained stamp): min(host - stamp)
            # maps stamps to the client's clock
            "clock": open(self.dir / f"clock-r{self.rank}.bin", "ab"),
        }
        threading.Thread(
            target=self._drain_loop, name="dsv41-bench-drain", daemon=True
        ).start()
        logger.warning(
            "NVIDIA megakernel: %s=%s: benchmark step stamps on (rank %d%s%s)",
            TIMES_ENV,
            directory,
            self.rank,
            ", end stamps" if self.ends else "",
            ", raw ids" if self.ids is not None else "",
        )

    # host side of a forward ------------------------------------------------------------

    def _stamp(self, stream, kind: int, positions, slot: int, with_table: bool) -> None:
        from sgl_nvidia_mega.common.jit import null_or_pointer, pointer

        rows = int(positions.shape[0]) if positions is not None else 0
        meta = kind | (min(rows, 255) << 8) | (slot << 16)
        with_table = with_table and slot < MAX_SLOTS
        row = slot if with_table else 0
        table = (
            ctypes.c_void_p(self.table[row].data_ptr())
            if with_table
            else ctypes.c_void_p(0)
        )
        args = [
            pointer(self.events),
            null_or_pointer(self.counts) if with_table else ctypes.c_void_p(0),
            null_or_pointer(self.ids) if with_table else ctypes.c_void_p(0),
            pointer(self.counter),
            table,
            ctypes.c_void_p(self.dev_table[row].data_ptr()),
            ctypes.c_void_p(self.dev_valid[row].data_ptr()),
            null_or_pointer(positions),
            ctypes.c_int32(positions is not None and positions.dtype == torch.int64),
            ctypes.c_uint32(RING),
            ctypes.c_int32(IDS_STRIDE),
            ctypes.c_uint64(meta),
        ]
        _launch_plain(self.kernel, args, stream.cuda_stream)

    @contextlib.contextmanager
    def forward(self, kind: str, positions):
        slot = len(self.layouts)
        self.layouts.append(
            {
                "slot": slot,
                "kind": kind,
                "rows": int(positions.shape[0]),
                "captured": torch.cuda.is_current_stream_capturing(),
                "entries": [],
            }
        )
        main = torch.cuda.current_stream()
        self.side.wait_stream(main)
        self._stamp(self.side, KIND[kind], positions, slot, True)
        self.current = self.layouts[-1]
        try:
            yield
        finally:
            self.current = None
            self._close(slot)
            if self.ends:
                self._stamp(main, KIND[kind] + 2, positions, slot, False)
            main.wait_stream(self.side)

    def note(self, ids: torch.Tensor, layer: int, experts: int) -> None:
        cur = self.current
        if cur is None or len(cur["entries"]) >= MAX_ENTRIES:
            return
        self._held.append(ids)
        cur["entries"].append(
            {
                "layer": layer,
                "n": int(ids.numel()),
                "k": int(ids.shape[-1]),
                "experts": int(experts),
                "ptr": ids.data_ptr(),
                "is64": ids.dtype == torch.int64,
            }
        )

    def _close(self, slot: int) -> None:
        layout = self.layouts[slot]
        if slot < MAX_SLOTS:
            row = self.table[slot].numpy()
            for j, e in enumerate(layout["entries"]):
                row[1 + 2 * j] = e["ptr"]
                row[2 + 2 * j] = (
                    e["n"] | (int(e["is64"]) << 32) | ((e["experts"] // 2) << 40)
                )
            row[0] = len(layout["entries"])
        with open(self.dir / f"layout-r{self.rank}.json", "w") as f:
            json.dump(
                [
                    {
                        **l,
                        "entries": [
                            {k: v for k, v in e.items() if k != "ptr"}
                            for e in l["entries"]
                        ],
                    }
                    for l in self.layouts
                ],
                f,
            )

    # drain ------------------------------------------------------------------------------

    def _drain_loop(self) -> None:
        events = self.events.numpy()
        counts = self.counts.numpy() if self.counts is not None else None
        ids = self.ids.numpy() if self.ids is not None else None
        while True:
            time.sleep(0.2)
            try:
                self._drain(events, counts, ids)
            except Exception as e:  # never take the server down
                logger.warning("NVIDIA megakernel: bench drain: %s", e)

    def _drain(self, events, counts, ids) -> None:
        out_e, out_c, out_i = [], [], []
        while True:
            idx = self._next
            slot = idx % RING
            if int(events[slot, 3]) != idx + 1:
                break
            e = events[slot].copy()
            c = counts[slot].copy() if counts is not None else None
            i = ids[slot].copy() if ids is not None else None
            if int(events[slot, 3]) != idx + 1:  # overwritten while read
                break
            out_e.append(e)
            if c is not None:
                out_c.append(
                    np.concatenate([np.array([idx], dtype=np.int64).view(np.int16), c])
                )
            if i is not None:
                out_i.append(
                    np.concatenate([np.array([idx], dtype=np.int64).view(np.int16), i])
                )
            self._next += 1
        if out_e:
            now = time.time_ns()
            f = self._files["events"]
            f.write(np.stack(out_e).tobytes())
            f.flush()
            self._files["clock"].write(
                np.array([now, out_e[-1][0]], dtype=np.int64).tobytes()
            )
            self._files["clock"].flush()
        for name, rows in (("counts", out_c), ("ids", out_i)):
            if rows and self._files[name] is not None:
                self._files[name].write(np.stack(rows).tobytes())
                self._files[name].flush()


_TIMER: StepTimer | None = None
_TIMER_INIT = False


def setup(device=None) -> None:
    """Build the step timer when requested (at build, outside capture); idempotent."""
    global _TIMER, _TIMER_INIT
    fixed_spec()  # refuse early
    if _TIMER_INIT:
        return
    _TIMER_INIT = True
    directory = os.environ.get(TIMES_ENV, "")
    if directory in ("", "0"):
        return
    device = device or torch.device("cuda", torch.cuda.current_device())
    _TIMER = StepTimer(directory, device)


def forward(kind: str, positions):
    """Context of one covered forward ("verify" / "draft"): the start stamp (and the end
    stamp), and the MoE ids it notes; a no-op without the timer."""
    if _TIMER is None:
        return contextlib.nullcontext()
    return _TIMER.forward(kind, positions)


def ids_buffer(rows: int, top_k: int, device) -> torch.Tensor | None:
    """A persistent int32 [rows, top_k] buffer for a layer's fast-gate ids while the
    step stamps are on (they read it after the replay; a graph-pool tensor can be
    reused by then), else None. At build, outside capture."""
    if _TIMER is None:
        return None
    return torch.full((rows, top_k), -1, dtype=torch.int32, device=device)


def note_ids(ids: torch.Tensor, layer: int, experts: int) -> None:
    """A MoE layer's top-k ids in the current forward, for the next replay's stamp."""
    if _TIMER is not None:
        _TIMER.note(ids, layer, experts)

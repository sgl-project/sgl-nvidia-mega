# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Host side of the small-M NVFP4 MoE engines: compile specs, scratch, launch.

Engine-neutral: every input is a torch tensor or a Python scalar; the serving engine's
glue extracts them from its MoE layer, compiles the specs through its own compile cache
and calls ``SmallMMoe``. A verify batch is ``rows_per_request`` (1..4 rows per request,
request-major, at most 4 requests):

* one request (M <= 4): the one-request engine (mode ``g1``);
* two requests (M <= 8): the two-request engine (mode ``g2``);
* three or four requests (M <= 16): the four-request engine (``engine16``).

``what``: bit 0 runs the engine (it writes ``partials`` / ``shared_partials`` and the
slot records), bit 1 the finalize (``out[M, 6144]`` BF16: the layer's un-reduced MoE
output). With bit 1 clear the scratch buffers are the output (the fold ABI: ``partials``
[4][M][8][6144] FP32, ``shared_partials`` [4][M][6144] FP32, ``slot_ids`` /
``slot_weights`` / ``slot_scales`` [M][8], row stride M, a slot id < 0 is skipped)."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from . import schedule as S
from .constants import HIDDEN, NUM_EXPERTS, TOP_K

MAX_ROWS = 16
MAX_REQUESTS = 4
REQUEST_ROWS = 4
SHARED_PARTS = 4
MIN_CLUSTERS_SHARED = 8
ENGINE_SMEM = {"g1": 227440, "g2": 229968, "g4": 223664}

# Opt-in forms of the four-request engine (mode g4, 9..16 rows), read at build; the
# defaults are the served form. Every form leaves each output bit unchanged.
#   ..._MOE16_XPIPE=1         per unit, push all groups' K-split partials before waiting
#   ..._MOE16_RECV_BUFS=n     exchange receive buffers (default 5)
#   ..._MOE16_LOOKAHEAD=n     FC2(k) after FC1(k + n) (default 2)
#   ..._MOE16_PRE=n           unit 0's first n FC1 chunks issue before the unit table
#                             is built (1..3, default 1)
#   ..._MOE16_FASTTAB=1       lane-parallel unit table (batched id loads, no union list)
#   ..._MOE16_XGATE=1         (with FASTTAB) unit 0's first chunks wait for the x loads' issue
#   ..._MOE16_SHARED_WEIGHT=w schedule weight of the shared quarter in units (default 4)
#   ..._MOE16_SHARED_FILL=1   the shared clusters also take up to cap - weight routed units
#   ..._MOE16_XTMA=n          (with FASTTAB, no XGATE) routed clusters stage x with bulk copies
#                             in ring slot 3 right after the id loads; n = 1: unrolled
#                             quantization, n = 2..4: rolled, n - 1 warp tiles per iteration
#   ..._MOE16_SNAKE=1         snake order of the routed experts over the routed clusters
MOE16_ENV = "SGLANG_NVIDIA_MEGAKERNEL_GLM_MOE16_"
# Opt-in form of the one- and two-request engines (modes g1 / g2), read at build; the
# default is the served form and the form leaves each output bit unchanged.
#   SGLANG_NVIDIA_MEGAKERNEL_GLM_MOE_PRE3=1  unit 0's three FC1 chunks issue as soon as
#                                            unit 0 is resolved (engine ``pre`` = 3); the
#                                            die-local g1 / g2 forms ignore it
MOE_PRE3_ENV = "SGLANG_NVIDIA_MEGAKERNEL_GLM_MOE_PRE3"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(MOE16_ENV + name, "").strip()
    return int(raw) if raw else default


def g4_knobs() -> dict:
    """The g4 engine's build-time choices from the environment (defaults: served)."""
    from .engine import LOOKAHEAD
    from .engine16 import RECV_BUFS

    return {
        "xpipe": _env_int("XPIPE", 0) != 0,
        "recv_bufs": _env_int("RECV_BUFS", RECV_BUFS),
        "lookahead": _env_int("LOOKAHEAD", LOOKAHEAD),
        "pre": _env_int("PRE", 1),
        "fasttab": _env_int("FASTTAB", 0) != 0,
        "xgate": _env_int("XGATE", 0) != 0,
        "xtma": _env_int("XTMA", 0),
    }


def g4_schedule_knobs() -> dict:
    """The g4 schedule's choices from the environment: shared-quarter weight, shared
    fill and snake order (``schedule.g4_table`` keywords)."""
    return {
        "shared_weight": _env_int("SHARED_WEIGHT", S.G4_SHARED_WEIGHT),
        "fill_shared": _env_int("SHARED_FILL", 0) != 0,
        "snake": _env_int("SNAKE", 0) != 0,
    }


def g12_pre3_requested() -> bool:
    """``SGLANG_NVIDIA_MEGAKERNEL_GLM_MOE_PRE3`` is 1 (0 or unset: no; else ValueError)."""
    raw = os.environ.get(MOE_PRE3_ENV, "").strip()
    if raw not in ("", "0", "1"):
        raise ValueError(f"{MOE_PRE3_ENV}={raw!r}: expected 0 or 1")
    return raw == "1"


def g12_knobs(localize: bool = False) -> dict:
    """The g1 / g2 engines' build-time choices from the environment (default: served):
    ``pre`` 3 with ``SGLANG_NVIDIA_MEGAKERNEL_GLM_MOE_PRE3=1``, else 1; the die-local
    forms (``localize``) keep 1 (their unit 0 comes from the claimed slot, after the
    table)."""
    return {"pre": 3 if g12_pre3_requested() and not localize else 1}


G12_SHARED_WEIGHT_ENV = "SGLANG_NVIDIA_MEGAKERNEL_GLM_MOE_SHARED_WEIGHT_G"


def g12_schedule_knobs(mode: str) -> dict:
    """The g1 / g2 schedule's shared-quarter weight in routed units from
    ``SGLANG_NVIDIA_MEGAKERNEL_GLM_MOE_SHARED_WEIGHT_G1`` / ``..._G2`` (default: served,
    ``schedule.G12_SHARED_WEIGHT``); only the unit-to-cluster assignment changes."""
    raw = os.environ.get(G12_SHARED_WEIGHT_ENV + mode[1:], "").strip()
    w = int(raw) if raw else S.G12_SHARED_WEIGHT
    if not 0 <= w <= 8:
        raise ValueError(f"{G12_SHARED_WEIGHT_ENV}{mode[1:]}={raw!r}: expected 0..8")
    return {"shared_weight": w}


@dataclass
class LayerTensors:
    """One layer's tensors as the engines read them (views of the engine's processed
    weights, plus the two derived forms)."""

    w13: (
        torch.Tensor
    )  # uint8 [256, 512, 3072]: processed FC1 weights (gated rows, block-32 shuffled)
    w13_scales: torch.Tensor  # uint8 [256, 512 * 384]: processed FC1 block scales
    fc2_records: torch.Tensor  # uint8 [256, 4, 4, 3 * 18432] (weights.fc2_records)
    # uint8 [4, 4, 18 * 32768] (weights.shared_pack) or None
    shared_pack: torch.Tensor | None
    g1_alpha: torch.Tensor  # FP32 [256]: FC1 input scale x per-expert weight scale 2
    g2_alpha: torch.Tensor  # FP32 [256]
    gs1: float  # 1 / FC1 input scale
    gs2: float  # 1 / FC2 input scale

    def check(self, quarters: int = 4) -> None:
        def need(ok: bool, what: str) -> None:
            if not ok:
                raise ValueError(f"small-M MoE: {what}")

        fc1_rows = 128 * quarters
        for name, t, shape in (
            ("w13", self.w13, (NUM_EXPERTS, fc1_rows, 3072)),
            ("w13_scales", self.w13_scales, (NUM_EXPERTS, fc1_rows * 384)),
            ("fc2_records", self.fc2_records, (NUM_EXPERTS, quarters, 4, 3 * 18432)),
        ):
            need(
                t.dtype == torch.uint8
                and tuple(t.shape) == shape
                and t.is_contiguous(),
                f"{name} must be contiguous uint8 {shape}, "
                f"got {t.dtype} {tuple(t.shape)}",
            )
            need(
                t.is_cuda and t.data_ptr() % 16 == 0,
                f"{name} must be a 16-byte aligned CUDA tensor",
            )
        if self.shared_pack is not None:
            sp = self.shared_pack
            need(
                sp.dtype == torch.uint8
                and sp.numel() == quarters * 4 * 18 * 32768
                and sp.is_contiguous()
                and sp.data_ptr() % 16 == 0,
                f"shared_pack must be contiguous 16-byte aligned uint8 [{quarters}, 4, 589824]",
            )
        for name, t in (("g1_alpha", self.g1_alpha), ("g2_alpha", self.g2_alpha)):
            need(
                t.dtype == torch.float32
                and tuple(t.shape) == (NUM_EXPERTS,)
                and t.is_contiguous(),
                f"{name} must be contiguous FP32 [{NUM_EXPERTS}]",
            )


class Scratch:
    """Engine outputs / finalize inputs for up to 16 rows (one set, reused in stream
    order)."""

    def __init__(self, device: torch.device, quarters: int = 4) -> None:
        f32, i32 = torch.float32, torch.int32
        self.partials = torch.empty(
            quarters * MAX_ROWS * TOP_K * HIDDEN, dtype=f32, device=device
        )
        self.shared_partials = torch.empty(
            quarters * MAX_ROWS * HIDDEN, dtype=f32, device=device
        )
        self.slot_ids = torch.empty(MAX_ROWS * TOP_K, dtype=i32, device=device)
        self.slot_weights = torch.empty(MAX_ROWS * TOP_K, dtype=f32, device=device)
        self.slot_scales = torch.empty(MAX_ROWS * TOP_K, dtype=f32, device=device)
        self.err = torch.zeros(1, dtype=i32, device=device)


def mode_for(rows_per_request: Sequence[int]) -> str:
    n = len(rows_per_request)
    if not 1 <= n <= MAX_REQUESTS or any(
        not 1 <= int(r) <= REQUEST_ROWS for r in rows_per_request
    ):
        raise ValueError(
            "small-M MoE: need 1..4 requests of 1..4 rows, "
            f"got {tuple(rows_per_request)}"
        )
    return "g1" if n == 1 else "g2" if n == 2 else "g4"


def pack_rows(rows_per_request: Sequence[int]) -> int:
    """The four-request engine's per-request row word: rows of request q in byte q."""
    v = 0
    for q, r in enumerate(rows_per_request):
        v |= int(r) << (8 * q)
    return v


def default_clusters(device: torch.device | None = None) -> int:
    """Co-resident 4-CTA clusters at one CTA per SM (the engines' occupancy: each uses
    nearly all of an SM's shared memory) on ``device`` (default: the current one)."""
    import cutlass.utils

    index = (
        torch.accelerator.current_device_index()
        if device is None
        else torch.device(device).index
    )
    # the device's primary context is current on this thread
    torch.accelerator.synchronize(index)
    return int(cutlass.utils.HardwareInfo(index).get_max_active_clusters(4))


def compile_specs(
    shared: bool = True,
    quarters: int = 4,
    include_g4: bool | None = None,
    pre3_default: bool = False,
) -> dict[str, tuple[object, tuple]]:
    """The served compile units: name -> (kernel object, example arguments for
    ``cute.compile``)."""
    from cutlass import Float32, Int32, Int64
    from cutlass.cute.runtime import make_fake_stream

    from .engine import Nvfp4MoeSmallM
    from .engine16 import Nvfp4MoeSmallM16
    from .finalize import MoeFinalize
    from .routing import MoeRouteTop8

    if include_g4 is None:
        include_g4 = quarters == 4

    z = Int64(0)
    one = Float32(1.0)

    def eng(m: int, split: int) -> tuple:
        return (
            z,
            z,
            z,
            z,
            z,
            z,
            z,
            one,
            one,
            Int32(m),
            Int32(split),
            z,
            z,
            z,
            z,
            z,
            z,
            z,
            z,
            z,
            Int32(33),
            make_fake_stream(),
        )

    k12 = (
        {"pre": 3}
        if pre3_default and not os.environ.get(MOE_PRE3_ENV, "").strip()
        else g12_knobs()
    )
    specs: dict[str, tuple[object, tuple]] = {
        "g1": (Nvfp4MoeSmallM("g1", shared, quarters=quarters, **k12), eng(4, 4)),
        "g2": (Nvfp4MoeSmallM("g2", shared, quarters=quarters, **k12), eng(8, 4)),
        "finalize": (
            MoeFinalize(shared, quarters=quarters),
            (z, z, z, z, z, z, Int32(1), make_fake_stream()),
        ),
        "route": (
            MoeRouteTop8(),
            (z, z, Float32(2.5), z, z, Int32(1), make_fake_stream()),
        ),
    }
    if include_g4:
        specs["g4"] = (
            Nvfp4MoeSmallM16(shared, **g4_knobs()),
            eng(16, pack_rows((4, 4, 4, 4))),
        )
    return specs


class SmallMMoe:
    """Launches the compiled engines on one device.

    ``compiled``: name -> compiled callable for the names of ``compile_specs``;
    ``nclusters``: the engine grid in 4-CTA clusters (at least 31 for quarters=4,
    at least 36 for quarters=8 g2; with fewer, the schedule tables do not fit and
    the constructor raises ``ValueError``)."""

    def __init__(
        self,
        compiled: dict[str, Callable],
        nclusters: int,
        device: torch.device,
        shared: bool = True,
        quarters: int = 4,
    ) -> None:
        min_sh = 2 * quarters
        if shared and nclusters < min_sh:
            raise ValueError(
                "small-M MoE: the in-engine shared expert needs "
                f">= {min_sh} clusters"
            )
        self.fn = compiled
        self.nclusters = nclusters
        self.shared = shared
        self.quarters = int(quarters)
        self.device = device
        self.scratch = Scratch(device, quarters=self.quarters)
        sk = g4_schedule_knobs()
        self.tables = {}
        modes = [("g1", S.g1_table), ("g2", S.g2_table)]
        if "g4" in compiled and self.quarters == 4:
            modes.append(("g4", S.g4_table))
        for mode, build in modes:
            kw = sk if mode == "g4" else {**g12_schedule_knobs(mode), "quarters": self.quarters}
            ncs = sorted({nclusters, min(nclusters, 32)}) if mode == "g1" else [nclusters]
            for nc in ncs:
                self.tables[(mode, nc)] = build(nc, shared, **kw).to(device)

    def clusters(self, mode: str, rows: int) -> int:
        return min(self.nclusters, 32) if mode == "g1" and rows == 1 else self.nclusters

    def __call__(
        self,
        layer: LayerTensors,
        x: torch.Tensor,
        ids: torch.Tensor,
        topw: torch.Tensor,
        rows_per_request: Sequence[int],
        out: torch.Tensor | None = None,
        what: int = 3,
    ) -> torch.Tensor | None:
        import cuda.bindings.driver as cuda
        from cutlass import Float32, Int32, Int64

        mode = mode_for(rows_per_request)
        m = sum(int(r) for r in rows_per_request)
        if (
            x.dtype != torch.bfloat16
            or tuple(x.shape) != (m, HIDDEN)
            or not x.is_contiguous()
            or ids.dtype != torch.int32
            or tuple(ids.shape) != (m, TOP_K)
            or not ids.is_contiguous()
            or topw.dtype != torch.float32
            or tuple(topw.shape) != (m, TOP_K)
            or not topw.is_contiguous()
        ):
            raise ValueError(
                "small-M MoE: x bf16 [M, 6144], ids int32 [M, 8], "
                "weights fp32 [M, 8], contiguous"
            )
        if self.shared and layer.shared_pack is None:
            raise ValueError(
                "small-M MoE: this launcher runs the shared expert; "
                "the layer has no shared pack"
            )
        if what & 2 and (
            out is None
            or out.dtype != torch.bfloat16
            or tuple(out.shape) != (m, HIDDEN)
            or not out.is_contiguous()
        ):
            raise ValueError("small-M MoE: out must be contiguous bf16 [M, 6144]")
        sc = self.scratch
        nc = self.clusters(mode, m)
        split = (
            m
            if mode == "g1"
            else int(rows_per_request[0])
            if mode == "g2"
            else pack_rows(rows_per_request)
        )
        stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)

        def p(t):
            return Int64(t.data_ptr() if t is not None else 0)

        if what & 1:
            self.fn[mode](
                p(layer.w13),
                p(x),
                p(ids),
                p(topw),
                p(layer.shared_pack if self.shared else None),
                p(layer.g1_alpha),
                p(layer.g2_alpha),
                Float32(layer.gs1),
                Float32(layer.gs2),
                Int32(m),
                Int32(split),
                p(sc.err),
                p(self.tables[(mode, nc)]),
                p(sc.shared_partials),
                p(sc.partials),
                p(sc.slot_ids),
                p(sc.slot_weights),
                p(sc.slot_scales),
                p(layer.w13_scales),
                p(layer.fc2_records),
                Int32(nc),
                stream,
            )
        if what & 2:
            self.fn["finalize"](
                p(sc.partials),
                p(sc.shared_partials if self.shared else None),
                p(sc.slot_ids),
                p(sc.slot_weights),
                p(sc.slot_scales),
                p(out),
                Int32(m),
                stream,
            )
        return out

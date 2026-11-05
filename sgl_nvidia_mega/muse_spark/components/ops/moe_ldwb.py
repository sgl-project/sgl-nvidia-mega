# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MoE-LD-W BF16 (``csrc/moe_ldwb.cuh``): routed BF16 experts of one expert-parallel rank for 1..512 token rows in one launch
of one CTA per SM, streaming each active local expert's weights once per pass of routed rows (development build, not wired
into any component).

``moe_ldwb(...)`` takes the BF16 routed-MoE arguments (packed top-k ids, BF16 token rows, the stock BF16 expert stacks in
their block-major-K layout) and returns the ``(gemm2_out, expert_weights, expanded_idx_to_permuted_idx)`` triple of the stock
call: FC2 rows in a compact expert-major order, the map pointing at them.

``Scratch`` (per device; the kernel keeps it consistent across launches ordered by completion): permuted token rows,
activation rows (the FC2 operand), two counter sets (alternating launches), start tickets, sticky error word.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import torch

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/jit/utils/__init__.py": ("load_jit",),
}


_CSRC = Path(__file__).resolve().parents[1] / "csrc"
_CUH = _CSRC / "moe_ldwb.cuh"
HIDDEN = 4096
INTER = 4096
TOP_K = 8
MAX_TOKENS = 512
MAX_SLOTS = MAX_TOKENS * TOP_K
MAX_POOL = 128
ROW_BYTES = HIDDEN * 2
ACT_WORDS = INTER // 4
FLAG_EVICT_FIRST = 2
FLAG_TAIL_TRIGGER = 32
STAMP_WORDS = 24
DEFAULT_FLAGS = FLAG_EVICT_FIRST | FLAG_TAIL_TRIGGER
RING4 = ("-DMUSE_VR_LDW_PASS_ROWS=32", "-DMUSE_VR_LDW_STAGES=4")
RING48 = ("-DMUSE_VR_LDW_PASS_ROWS=48", "-DMUSE_VR_LDW_STAGES=4")
# K-half relay form (development): each tile's two K halves run on CTAs of the two locality domains, the first half's exact
# fp32 accumulator handed to the second; the expert stacks placed by ``place_relay``
RELAY = ("-DMUSE_VR_LDW_RELAY=1",)


def _flashinfer_include_paths() -> list[str]:
    import flashinfer

    data = Path(flashinfer.__file__).resolve().parent / "data"
    paths = [
        data / "csrc" / "nv_internal",
        data / "csrc" / "nv_internal" / "include",
        data / "include",
        data / "cutlass" / "include",
    ]
    return [str(p) for p in paths if p.exists()]


@lru_cache(maxsize=None)
def _module(extra: tuple = ()):
    import hashlib

    from sglang.kernels.jit.utils import load_jit

    from .. import slot_claim

    abi = "v" + hashlib.sha256(" ".join(extra).encode()).hexdigest()[:12]
    extra = tuple(extra) + slot_claim.build_flags("moe_ldwb")
    return load_jit(
        "muse_vr_moe_ldwb",
        *[re.sub(r"[^A-Za-z0-9_]", "_", e) for e in extra],
        cuda_files=[str(_CUH)],
        cuda_wrappers=[
            ("run", "muse_vr_ldwb::MuseVrLdwb::run"),
            ("preload", "muse_vr_ldwb::MuseVrLdwb::preload"),
        ],
        extra_include_paths=[str(_CSRC)] + _flashinfer_include_paths(),
        extra_cuda_cflags=[
            "-O3",
            "-use_fast_math",
            "-DENABLE_BF16",
            "-lineinfo",
            f"-DMUSE_VR_LDW_ABI={abi}",
            *extra,
        ],
    )


def preload(extra: tuple = ()) -> None:
    _module(tuple(extra)).preload()


def relay_of(extra: tuple) -> bool:
    return "-DMUSE_VR_LDW_RELAY=1" in tuple(extra)


def place_relay(w: torch.Tensor) -> torch.Tensor:
    """Copy of a block-major-K expert stack [P, 64, R, 64]: K blocks 0..31 of expert e on locality domain e % 2, K blocks
    32..63 on the other domain (the relay form's placement)."""
    from ..ld_layout import alloc_pattern

    if not w.is_cuda or not w.is_contiguous() or w.dim() != 4 or w.shape[1] != 64:
        raise ValueError("place_relay needs a contiguous CUDA [P, 64, R, 64] stack")
    p = w.shape[0]
    chunk = w.numel() * w.element_size() // (2 * p)
    raw = alloc_pattern(
        w.device, chunk, [(e & 1) ^ h for e in range(p) for h in (0, 1)]
    )
    out = raw.view(w.dtype).view(w.shape)
    out.copy_(w)
    return out


def pass_rows_of(extra: tuple) -> int:
    for f in extra:
        if f.startswith("-DMUSE_VR_LDW_PASS_ROWS="):
            return int(f.split("=", 1)[1])
    return 128


class Scratch:
    """Per-device buffers of MoE-LD-W BF16 (launches that share it must be ordered by completion)."""

    def __init__(self, device: torch.device, pass_rows: int = 128, relay: bool = False):
        z = dict(device=device)
        passes = MAX_POOL + MAX_SLOTS // pass_rows
        self.pass_rows = pass_rows
        relay_slots = (
            passes * 96 if relay else 0
        )  # one per (pass, FC1 / FC2 tile); rows of pass_rows fp32 columns
        self.relay = (
            torch.empty(relay_slots * 128 * pass_rows, dtype=torch.float32, **z)
            if relay
            else None
        )
        self.hs_perm = torch.zeros(MAX_SLOTS, ROW_BYTES, dtype=torch.uint8, **z)
        self.act = torch.zeros(MAX_SLOTS * ACT_WORDS, dtype=torch.int64, **z)
        self.cnt = torch.zeros(
            2 * (2 * passes + 32 + relay_slots), dtype=torch.int32, **z
        )
        self.ctl = torch.zeros(32, dtype=torch.int32, **z)
        self.err = torch.zeros(1, dtype=torch.int32, **z)

    def error(self) -> int:
        return int(self.err.item())


def moe_ldwb(
    packed,
    hs,
    w13,
    w2,
    local_offset: int,
    scratch: Scratch,
    flags: int | None = None,
    dbg: int = 0,
    pdl: bool = False,
    extra: tuple = (),
    stamps=None,
):
    """``w13`` [P, 64, 8192, 64] / ``w2`` [P, 64, 4096, 64] BF16 (the stock block-major-K expert stacks), ``hs`` [M, 4096] BF16."""
    from ..moe_ld_ops import domain_slots

    if flags is None:
        flags = DEFAULT_FLAGS
    m = packed.shape[0]
    if not 1 <= m <= MAX_TOKENS:
        raise ValueError(f"num_tokens {m} outside 1..{MAX_TOKENS}")
    if scratch.pass_rows != pass_rows_of(tuple(extra)):
        raise ValueError(
            f"scratch for {scratch.pass_rows}-row passes, build with {pass_rows_of(tuple(extra))}-row passes"
        )
    dev = packed.device
    table, n0, n1 = domain_slots(
        dev.index if dev.index is not None else torch.cuda.current_device()
    )
    out = torch.empty(m * TOP_K, HIDDEN, dtype=torch.bfloat16, device=dev)
    ew = torch.empty(m, TOP_K, dtype=torch.bfloat16, device=dev)
    e2p = torch.empty(m * TOP_K, dtype=torch.int32, device=dev)
    s = scratch
    args = [
        packed,
        hs,
        w13,
        w2,
        table,
        out,
        ew.view(torch.int16),
        e2p,
        s.hs_perm,
        s.act,
        s.cnt,
        s.ctl,
        s.err,
        stamps,
        int(local_offset),
        n0,
        n1,
        int(flags),
        int(dbg),
        int(bool(pdl)),
    ]
    if relay_of(tuple(extra)):
        if s.relay is None:
            raise ValueError("relay build needs Scratch(relay=True)")
        args.append(s.relay)
    elif s.relay is not None:
        raise ValueError("Scratch(relay=True) with a build without the relay form")
    _module(tuple(extra)).run(*args)
    return out, ew, e2p

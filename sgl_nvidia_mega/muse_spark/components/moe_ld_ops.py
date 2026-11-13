# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MoE-LD kernel (``csrc/moe_ld.cuh``): JIT build, per-call-site workspace and call.

``moe_ld(...)`` takes the arguments SGLang passes to FlashInfer's ``trtllm_fp4_block_scale_routed_moe`` for the Muse
Spark expert layer (packed top-k, NVFP4 activations with linear block scales and per-token scales, shuffled NVFP4 expert
weights and scales, per-expert output scalars) for at most 16 token rows and returns the same
``(gemm2_out, expert_weights, expanded_idx_to_permuted_idx)`` triple that call returns with ``do_finalize=False``. The
FC2 rows are in a compact expert-major order; the returned map points at them, so the existing finalize consumes them
unchanged.

Every call site (one per MoE layer) owns a ``Site`` (control block + activation hand-off buffer): the kernel relies on
launches of one site being ordered by completion (stream order or a graph edge), which holds for one layer of a decode
step. Expert weights are best placed with ``ld_layout.split_rows_copy`` (each expert's row halves on the two locality
domains); the kernel is correct on any placement.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import torch

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/jit/utils/__init__.py": ("load_jit",),
}


_CSRC = Path(__file__).resolve().parent / "csrc"
_CUH = _CSRC / "moe_ld.cuh"
HIDDEN = 4096
INTER = 4096
TOP_K = 8
MAX_TOKENS = 16
MAX_ROWS = MAX_TOKENS * TOP_K
ACT_WORDS = INTER // 4  # 64-bit words of 4 bf16 per activation row
MAX_POOL = 128
# token rows -> (MMA N, weight stages of 36 KiB); shared memory per block decides the stage count
CONFIGS = {8: 5, 16: 4}
CONFIGS_FC2_HALF = {
    8: 4,
    16: 3,
}  # experiment (-DMUSE_VR_MOELD_FC2_HALF=1): 40 KiB stages
FLAG_IDS_EARLY = 1  # ids and weight stream before griddepcontrol.wait (needs a full graph dependency on the ids)
FLAG_EVICT_FIRST = (
    2  # L2 evict_first on the weight stream (each byte is read once per layer)
)
FLAG_EARLY_TRIGGER = 8  # launch_dependents after the prologue instead of at grid exit
FLAG_L2_PREFETCH = 16  # FC2 tiles: L2 prefetch of the stages that do not fit the ring
FLAG_TAIL_TRIGGER = 32  # launch_dependents when each CTA starts its last item (the finalize launches into freed SMs)
FLAG_IDS_BOARD = 64  # ids-first: ids read once their producer publishes this site's tag (see Site.board_*), weights early
# control block words (int32 view of Site.ctl): the start-ticket counter, the ids board tag, the board miss count
CTL_TICKET_WORD, CTL_BOARD_WORD, CTL_BOARD_MISS_WORD = 7, 9, 11
CTL_CLAIM_WORD = (
    7 + 14
)  # first word of the slot claim area (the kernel's kClaimWord; see slot_claim.py)
TICKET_CYCLE = 1 << 12  # launches per start-ticket cycle (the kernel's kStartCycle)
DEFAULT_FLAGS = FLAG_EVICT_FIRST | FLAG_TAIL_TRIGGER
GLOBAL_SCALE_INV = float(
    torch.tensor(1.0, dtype=torch.float32) / (torch.tensor(448.0) * torch.tensor(6.0))
)


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
def _module(n: int, slots: int, extra: tuple = ()):
    from sglang.kernels.jit.utils import load_jit

    from . import fi_guard, mma_k, slot_claim

    fi_guard.check("moe_ld_ops")
    extra = tuple(extra) + mma_k.build_flags(
        "moe_ld"
    )  # MMA-K form (MUSE_VR_MOE_MMA_K; no define for the default 64)
    extra = extra + slot_claim.build_flags(
        "moe_ld"
    )  # exact slot claim (default) or the UNSAFE form
    return load_jit(
        "muse_vr_moe_ld",
        str(n),
        str(slots),
        *[
            re.sub(r"[^A-Za-z0-9_]", "_", e) for e in extra
        ],  # build-key markers must be path-safe
        cuda_files=[str(_CUH)],
        cuda_wrappers=[("run", f"muse_vr_moeld::MuseVrMoeLd<{n}, {slots}>::run")],
        extra_include_paths=[str(_CSRC)] + _flashinfer_include_paths(),
        extra_cuda_cflags=[
            "-O3",
            "-use_fast_math",
            "-DENABLE_BF16",
            "-lineinfo",
            *extra,
        ],
    )


def config_for(num_tokens: int) -> tuple[int, int]:
    for n in sorted(CONFIGS):
        if num_tokens <= n:
            return n, int(
                os.environ.get(f"MUSE_VR_MOELD_SLOTS{n}", CONFIGS[n])
            )  # slot override: timing studies
    raise ValueError(f"num_tokens {num_tokens} above {MAX_TOKENS}")


@lru_cache(maxsize=None)
def domain_slots(device_index: int) -> tuple[torch.Tensor, int, int]:
    """(SM -> slot table, slots on domain 0, slots on domain 1); slot = locality domain << 7 | rank of the SM among the
    domain's SMs (ascending SM id)."""
    from . import ld_layout

    dev = torch.device("cuda", device_index)
    dom = [int(x) & 1 for x in ld_layout.sm_domain_table(dev).cpu().tolist()]
    n = [0, 0]
    slots = []
    for d in dom:
        slots.append((d << 7) | n[d])
        n[d] += 1
    if max(n) > 128 or len(dom) > 256:
        raise ValueError("slot table holds 128 ranks per domain and 256 SMs")
    table = torch.tensor(slots, dtype=torch.uint8).view(torch.int8).to(dev)
    return table, n[0], n[1]


class Site:
    """Per call site (MoE layer) scratch: control block, activation hand-off words (one copy per locality domain, backed
    by that domain's memory when the device has two), per-expert completion counters, sticky error word. The kernel
    keeps it consistent across launches of the site."""

    def __init__(self, device: torch.device, trace_ctas: int = 0):
        from . import ld_layout

        self.ctl = torch.zeros(32, dtype=torch.int32, device=device)
        half_bytes = (
            2 * MAX_ROWS * ACT_WORDS * 8
        )  # both launch-parity halves of one domain copy (2 MiB)
        try:
            if ld_layout.domain_count(device) != 2:
                raise RuntimeError("not two locality domains")
            self.act = ld_layout.alloc_pattern(device, half_bytes, [0, 1]).view(
                torch.int64
            )
        except Exception:  # noqa: BLE001 - placement only changes speed
            self.act = torch.empty(
                2 * half_bytes // 8, dtype=torch.int64, device=device
            )
        self.act.fill_(-1)  # all-ones = not written
        # counters (zero), then trace words for -DMUSE_VR_MOELD_TRACE builds
        self.aux = torch.zeros(
            MAX_POOL + 64 * trace_ctas, dtype=torch.int64, device=device
        )
        self.trace_ctas = trace_ctas
        self.err = torch.zeros(1, dtype=torch.int32, device=device)

    def board_next_tag(self, grid: int) -> int:
        """Host view (tests): the ids-board tag the next launch of this site expects (stream-ordered, no launch in flight)."""
        return (int(self.ctl[CTL_TICKET_WORD].item()) // grid) % TICKET_CYCLE + 1

    def board_misses(self) -> int:
        return int(self.ctl[CTL_BOARD_MISS_WORD].item())

    def slot_tally(self) -> dict:
        """Slot claim counters of this site (verified launches, hinted-slot misses; exact form only)."""
        from . import slot_claim

        return slot_claim.tally(self.ctl, CTL_CLAIM_WORD)

    def trace(self) -> torch.Tensor:
        """[ctas, 64] globaltimer stamps of the last launch (timing builds)."""
        return self.aux[MAX_POOL:].view(self.trace_ctas, 64)


def moe_ld(
    packed,
    hs,
    hs_sf,
    pts,
    w13,
    sf13,
    w2,
    sf2,
    gate_alpha,
    up_alpha,
    out_alpha,
    local_offset: int,
    site: Site,
    flags: int | None = None,
    dbg: int = 0,
    pdl: bool = False,
    extra: tuple = (),
):
    if flags is None:
        flags = DEFAULT_FLAGS
    m = packed.shape[0]
    n, slots = config_for(m)
    if "-DMUSE_VR_MOELD_FC2_HALF=1" in extra:
        slots = CONFIGS_FC2_HALF[n]
    dev = packed.device
    table, n0, n1 = domain_slots(
        dev.index if dev.index is not None else torch.cuda.current_device()
    )
    out = torch.empty(m * TOP_K, HIDDEN, dtype=torch.bfloat16, device=dev)
    ew = torch.empty(m, TOP_K, dtype=torch.bfloat16, device=dev)
    e2p = torch.empty(m * TOP_K, dtype=torch.int32, device=dev)
    _module(n, slots, tuple(extra)).run(
        packed,
        hs.view(torch.uint8),
        hs_sf.view(torch.uint8).reshape(m, -1),
        pts.reshape(m).float(),
        w13.view(torch.uint8),
        sf13.view(torch.uint8),
        w2.view(torch.uint8),
        sf2.view(torch.uint8),
        gate_alpha,
        up_alpha,
        out_alpha,
        table,
        out,
        ew.view(torch.int16),
        e2p,
        site.act,
        site.aux,
        site.ctl,
        site.err,
        int(local_offset),
        n0,
        n1,
        GLOBAL_SCALE_INV,
        int(flags),
        int(dbg),
        int(bool(pdl)),
    )
    return out, ew, e2p

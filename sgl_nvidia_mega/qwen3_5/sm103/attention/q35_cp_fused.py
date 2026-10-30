# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from TensorRT-LLM 4d09932ca7 + 7df45abe5c's plans above 32 requests (Apache-2.0):
# tensorrt_llm/_torch/attention_backend/fmha/q35_cp_fused.py; switches are constants here, the debug ring is dropped,
# and the kernel import points at this package.
"""Host side of the Qwen3.5 fused CP-lite partial decode attention.

One launch replaces the preparation, FMHA, LSE, and partial-output launches. Each CP
rank attends the pages it owns and writes the all-to-all merge's input layout.

The module owns the grid-barrier counters and partial workspace. Call
``q35_cp_fused_prepare`` before CUDA graph capture.
"""

import functools
import logging

import torch

logger = logging.getLogger(__name__)

PAGE = 64
HEAD_DIM = 256
MAX_ROWS = 128
# Batch bound of the plan: one request per lane of the plan warp up to 32; above that the wide-plan kernel variant
# (two requests per lane, q35_cp_partial_attn.Q35CpPartialAttnKernel._plan_wide) up to 64, and the quad-plan variant
# (four per lane, _plan_quad) up to 128. The equal split stays <= 32.
_QUAD_PLAN_MAX_BATCH = 128
# Above 64 requests the calls take the two-slot kernel (kernel_two / _plan_two, four requests per plan-warp lane): a
# CTA computes up to two requests, so no request is held to whole CTAs. The plans above give every request at least one
# of the SMs' CTAs; past ~SMs / 2 uniform long requests then get exactly one each and the slowest CTA runs a whole
# request (64k contexts, b96: 128 tiles where 81 would balance), which the two-slot plan's per-CTA share removes.
_TWO_SLOT_MAX_BATCH = 128
_WIDE_PLAN_MAX_BATCH = 64
_LANE_PLAN_MAX_BATCH = 32


# Keep the production selector on R11's unbalanced lane plan.  The wide and
# two-slot kernels remain available to explicit test hooks, but R11 did not
# select them when SGLANG_PLB_CP_BALANCED was disabled.
MAX_BATCH = _LANE_PLAN_MAX_BATCH
MAX_BATCH_Q1 = _LANE_PLAN_MAX_BATCH
# Pack every two-slot call; the unpacked form remains available as a test hook.
_TWO_SLOT_PACK = True


# Mirror of q35_cp_partial_attn.MAX_COMB_ROWS (rows one CTA's combine chunk may touch).
_MAX_COMB_ROWS = 32
LOG2E = 1.4426950408889634

# MTP may write the page immediately before the new tokens earlier in the step.
_FRESH_MARGIN = 1
# Request the prefix KV below that margin before the dependency wait (overlaps the q all-gather).
_KV_PRE_WAIT = True
# E4M3 P shift (log2 units), as TRTLLM_Q35_ATTN_FUSED_P_SHIFT: P = exp2(x - m + shift) before the FP8 conversion, so
# probabilities down to 2^-(6 + shift) of the row max stay E4M3 normals (0 = the unscaled P that flushes small
# probabilities: 5-13x the error of trtllm-gen with an attention sink). 5: trtllm-gen's error level (within ~5 % of 6)
# at no cost on flat scores and +0.5 us per call with very wide ones (6: +0.9 us).
_P_SHIFT = 5.0
_DBG_SLOTS = 0
# SMs left out of the grid (default 0: one CTA per SM). The in-kernel grid barrier needs every CTA resident; a reserve
# keeps the grid co-resident even when other kernels (e.g. a PDL neighbour still polling its peers) hold some SMs.
_SM_RESERVE = 0
_DBG_RING_CTAS = 256

# Declared alignment (bytes) of each pointer argument, used at compile and call time (make_ptr checks the address):
# q and the KV pool are TMA sources and the workspace takes 16-byte stores / bulk copies; tables, seq_lens and the
# scales are read element by element (views such as scales[2:3] are 4-byte aligned); out_o is 16 when its rows allow
# 8-element vector stores, else 4 (alltoall_merge reads it as int32).
_ALIGN = {
    "q": 16,
    "kv": 16,
    "tab": 4,
    "seq": 4,
    "bmm1": 4,
    "bmm2": 4,
    "lse": 4,
    "ws": 16,
    "cnt": 8,
    "dbg": 8,
}


def cp_partial_geometry(rows: int, n_splits: int) -> dict:
    """Mirror of q35_cp_partial_attn.cp_partial_geometry (kept import-free of cutlass; checked at compile time)."""
    total = rows * HEAD_DIM
    chunk = 8 * ((total + 8 * n_splits - 1) // (8 * n_splits))
    chunk_pad = chunk + (8 if (chunk // 8) % 2 == 0 else 0)
    n_chunks = (total + chunk - 1) // chunk
    lse_stride = 4 * ((n_splits + 3) // 4)
    ws_o_elems = n_chunks * n_splits * chunk_pad
    max_cov_rows = (chunk + HEAD_DIM - 2) // HEAD_DIM + 1
    return {
        "chunk": chunk,
        "ws_req_bytes": ws_o_elems * 2 + rows * lse_stride * 4,
        "max_cov_rows": max_cov_rows,
    }


def cp_partial_balanced_ws_bytes(rows: int, n_ctas: int, slots: int = 1) -> int:
    """Mirror of q35_cp_partial_attn.cp_partial_balanced_geometry()["ws_bytes"]: per CTA one O slot (the largest
    per-split share n_chunks(c) * chunk_pad(c) of a request's partials over c = 1..n_ctas, FP16) and 16 * rows bytes
    of LSE space; ``slots`` slots per CTA (2 for the two-slot plan)."""
    total = rows * HEAD_DIM
    slot_o = 0
    for c in range(1, n_ctas + 1):
        chunk = 8 * ((total + 8 * c - 1) // (8 * c))
        chunk_pad = chunk + (8 if (chunk // 8) % 2 == 0 else 0)
        slot_o = max(slot_o, ((total + chunk - 1) // chunk) * chunk_pad * 2)
    return slots * n_ctas * (16 * ((slot_o + 15) // 16) + 16 * rows)


def balanced_split_plan(seq_lens, cp_rank: int, cp_size: int, n_ctas: int):
    """Host mirror of the balanced split's in-kernel plan (q35_cp_partial_attn.Q35CpPartialAttnKernel._plan).

    Returns (T, c, ctas): T = the largest tile count of a CTA's block, c[r] = CTAs of request r, ctas[i] = (request,
    index in the request, first tile, tiles) of CTA i (tile k of a request = its owned pages 2k, 2k + 1 on this rank).
    """
    t = [
        ((((int(L) + PAGE - 1) // PAGE) - cp_rank + cp_size - 1) // cp_size + 1) // 2
        for L in seq_lens
    ]
    b = len(t)
    assert 1 <= b < n_ctas and b <= _QUAD_PLAN_MAX_BATCH
    total, tmax = sum(t), max(t)
    lo = max(-(-total // n_ctas), 1)
    hi = max(min(tmax, -(-total // (n_ctas - b))), lo)
    while lo < hi:
        mid = (lo + hi) // 2
        if sum(max(-(-x // mid), 1) for x in t) <= n_ctas:
            hi = mid
        else:
            lo = mid + 1
    c = [max(-(-x // lo), 1) for x in t]
    top = max(r for r in range(b) if t[r] == tmax)
    c[top] += n_ctas - sum(c)
    ctas = []
    for r in range(b):
        base, rem = divmod(t[r], c[r])
        for j in range(c[r]):
            ctas.append((r, j, j * base + min(j, rem), base + (1 if j < rem else 0)))
    return lo, c, ctas


def _two_slot_pack_starts(t, T: int):
    """Packed two-slot placement for T: request r starts at V_r; at a CTA boundary it takes max(t_r, 1) virtual
    tiles, at offset p inside a CTA's range max(t_r, T - p). Returns (starts, end)."""
    starts, V = [], 0
    for x in t:
        starts.append(V)
        p = V % T
        V += max(x, 1) if p == 0 else max(x, T - p)
    return starts, V


def _two_slot_pack_T(t, n_ctas: int, hi: int) -> int:
    """T of the packed plan (q35_cp_partial_attn.Q35CpPartialAttnKernel._plan_pack): feasible = end <= n_ctas * T,
    not monotone in T; hi (the unpacked plan's T) always is. The smallest feasible T in [lo, lo + 31], lo = ceil(sum
    max(t_r, 1) / n_ctas); else the smallest feasible of 32 candidates from lo + 32 to hi in steps of s, then the
    smallest feasible of the min(s - 1, 32) values below it (or it)."""
    lo = max(-(-sum(max(x, 1) for x in t) // n_ctas), 1)

    def ok(T):
        return _two_slot_pack_starts(t, T)[1] <= n_ctas * T

    for T in range(lo, lo + 32):
        if ok(T):
            return T
    lo2 = lo + 32
    s = max(-(-(hi - lo2) // 31), 1)
    c1 = next(T for T in (min(lo2 + k * s, hi) for k in range(32)) if ok(T))
    return next((T for T in range(c1 - min(s - 1, 32), c1) if ok(T)), c1)


def balanced_split_plan_two(
    seq_lens, cp_rank: int, cp_size: int, n_ctas: int, pack: bool = False
):
    """Host mirror of the two-slot plan (q35_cp_partial_attn.Q35CpPartialAttnKernel._plan_two).

    T = the smallest tile count with sum_r max(t_r, T) <= n_ctas * T; request r takes the virtual tile range [V_r, V_r
    + max(t_r, T)) (requests in order), CTA i the range [i * T, (i + 1) * T). Returns (T, slots): slots[i] = [slot 0,
    slot 1] of CTA i, each None or (request, index among the request's CTAs, its CTAs, first workspace slot, first
    tile, tiles, CTAs with work); slot 0 is the request holding the CTA's first position, slot 1 the one starting
    inside its range. With ``pack``, use the request positions and T from _two_slot_pack_starts / _two_slot_pack_T.
    """
    t = [
        ((((int(L) + PAGE - 1) // PAGE) - cp_rank + cp_size - 1) // cp_size + 1) // 2
        for L in seq_lens
    ]
    b = len(t)
    assert 1 <= b < n_ctas and b <= _TWO_SLOT_MAX_BATCH
    total, tmax = sum(t), max(t)
    lo = max(-(-total // n_ctas), 1)
    hi = max(tmax, lo)
    while lo < hi:
        mid = (lo + hi) // 2
        if sum(max(x, mid) for x in t) <= n_ctas * mid:
            hi = mid
        else:
            lo = mid + 1
    T = lo
    if pack:
        T = _two_slot_pack_T(t, n_ctas, T)
        starts, end = _two_slot_pack_starts(t, T)
    else:
        starts = [0]
        for x in t:
            starts.append(starts[-1] + max(x, T))
        end = starts.pop()
    slots = [[None, None] for _ in range(n_ctas)]
    s0 = 0
    for r, x in enumerate(t):
        V = starts[r]
        te = max(x, 1)
        f, last = V // T, (V + te - 1) // T
        c = last - f + 1
        for j in range(c):
            i = f + j
            lo_t, hi_t = max(V, i * T) - V, min(V + te, (i + 1) * T) - V
            slot = (
                r,
                j,
                c,
                s0,
                lo_t if x else 0,
                hi_t - lo_t if x else 0,
                c if x else 0,
            )
            k = 0 if V <= i * T else 1
            assert slots[i][k] is None
            slots[i][k] = slot
        s0 += c
    assert end <= n_ctas * T and s0 <= 2 * n_ctas
    return T, slots


def _num_splits(batch: int, sm_count: int) -> int:
    return max((sm_count - _SM_RESERVE) // batch, 1)


def _num_ctas(sm_count: int) -> int:
    """Grid of the balanced split (any batch)."""
    return max(sm_count - _SM_RESERVE, 1)


def q35_cp_fused_supported(
    q_len: int,
    cp_size: int,
    hq: int,
    batch: int,
    sm_count: int,
    max_batch: int | None = None,
) -> bool:
    """Shapes the kernel handles (D 256, FP8 q / KV and 64-token pages are checked by the caller). max_batch: the
    batch bound (default MAX_BATCH, or MAX_BATCH_Q1 at q_len 1; a caller that folds q_len into the heads passes its
    own)."""
    rows = q_len * cp_size * hq
    if max_batch is None:
        max_batch = MAX_BATCH if q_len > 1 else MAX_BATCH_Q1
    if not (
        1 <= rows <= MAX_ROWS
        and 1 <= batch <= min(max_batch, MAX_BATCH)
        and batch <= sm_count
    ):
        return False
    if batch > _LANE_PLAN_MAX_BATCH:
        return batch < _num_ctas(
            sm_count
        )  # the plan gives every request a CTA; it plans <= 128 requests
    return (
        cp_partial_geometry(rows, _num_splits(batch, sm_count))["max_cov_rows"]
        <= _MAX_COMB_ROWS
    )


class _DeviceState:
    """Per-device grid-barrier counters (int64 [MAX_BATCH + 1, MAX_BATCH], row b = batch size b, one epoch per
    request; zero-initialized once, never reset) and the partial workspace (sized for every supported call)."""

    def __init__(self, device: torch.device, sm_count: int):
        self.sm_count = sm_count
        need = 0
        for b in range(1, MAX_BATCH + 1):
            if b <= sm_count:
                need = max(
                    need,
                    b
                    * cp_partial_geometry(MAX_ROWS, _num_splits(b, sm_count))[
                        "ws_req_bytes"
                    ],
                )
        need = max(
            need,
            cp_partial_balanced_ws_bytes(MAX_ROWS, _num_ctas(sm_count), slots=2),
        )
        self.workspace = torch.empty(need // 4 + 64, dtype=torch.float32, device=device)
        self.ws_balanced = (
            None  # test calls that override the split mode (_balanced_workspace)
        )
        self.balanced_logged = (
            False  # runtime evidence: one INFO line at the first balanced call
        )
        self.wide_logged = (
            False  # ... and one at the first call above 32 requests (wide-plan variant)
        )
        self.quad_logged = False  # ... and one at the first call above 64 requests (two-slot / quad-plan variant)
        self.pack_logged = (
            False  # ... and one at the first two-slot call on the packed plan
        )
        self.counters = torch.zeros(
            MAX_BATCH + 1, MAX_BATCH, dtype=torch.int64, device=device
        )
        self.dbg_ring = None
        if _DBG_SLOTS > 0:
            self.dbg_ring = torch.zeros(
                _DBG_SLOTS * _DBG_RING_CTAS * 8, dtype=torch.int64, device=device
            )
            _start_dbg_dump(self.dbg_ring, device)


def _start_dbg_dump(ring: torch.Tensor, device: torch.device) -> None:
    raise NotImplementedError("the debug ring is not ported (_DBG_SLOTS = 0)")


_STATES: dict[int, _DeviceState] = {}


def _sm_count(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def q35_cp_fused_prepare(device: torch.device | None = None, sm_count: int = 0) -> None:
    """Allocate this device's counters / workspace (call before CUDA graph capture)."""
    device = (
        torch.device("cuda", torch.cuda.current_device()) if device is None else device
    )
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _STATES:
        _STATES[idx] = _DeviceState(
            torch.device("cuda", idx), sm_count or _sm_count(device)
        )


def _state(device: torch.device, sm_count: int) -> _DeviceState:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    st = _STATES.get(idx)
    if st is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "q35_cp_fused_partial: call q35_cp_fused_prepare() (or one eager call) before CUDA "
                "graph capture: the grid-barrier counters must be zero-initialized persistent memory"
            )
        q35_cp_fused_prepare(device, sm_count)
        st = _STATES[idx]
    return st


def _ptr(t: torch.Tensor, dtype, align: int):
    from cutlass import cute

    return cute.runtime.make_ptr(
        dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align
    )


def _balanced_workspace(state: "_DeviceState", need: int) -> torch.Tensor:
    """Workspace for a wide-plan call or a test that overrides the split mode."""
    if state.workspace.numel() * 4 >= need:
        return state.workspace
    if state.ws_balanced is None or state.ws_balanced.numel() * 4 < need:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "q35_cp_fused_partial(balanced=True): make one eager call before CUDA graph capture"
            )
        state.ws_balanced = torch.empty(
            need // 4 + 64, dtype=torch.float32, device=state.workspace.device
        )
    return state.ws_balanced


@functools.cache
def _compiled(
    q_len: int,
    gh: int,
    hql: int,
    cp_size: int,
    n_splits: int,
    bmm1_from_ptr: bool,
    bmm2_from_ptr: bool,
    o_vec16: bool,
    use_pdl: bool,
    kv_pre_wait: bool,
    fresh_margin: int,
    debug_ts: bool,
    unsafe_skip_wait: bool,
    p_shift: float = 5.0,
    debug_ring: int = 0,
    balanced: bool = False,
    rescale_threshold: float | None = None,
    l_from_p8: bool = False,
    len_sub: int = 0,
    early_trigger: bool = False,
    wide_plan: bool = False,
    quad_plan: bool = False,
    two_slot: bool = False,
    two_slot_pack: bool = False,
):
    import cuda.bindings.driver as cuda
    import cutlass
    from cutlass import cute

    from sgl_nvidia_mega.qwen3_5.sm103.attention.q35_cp_partial_attn import (
        Q35CpPartialAttnKernel,
    )

    if balanced:
        kernel = Q35CpPartialAttnKernel(
            q_len=q_len,
            gh=gh,
            hql=hql,
            cp_size=cp_size,
            n_splits=n_splits,
            bmm1_from_ptr=bmm1_from_ptr,
            bmm2_from_ptr=bmm2_from_ptr,
            o_vec16=o_vec16,
            use_pdl=use_pdl,
            kv_pre_wait=kv_pre_wait,
            fresh_margin_pages=fresh_margin,
            debug_ts=debug_ts,
            unsafe_skip_wait=unsafe_skip_wait,
            p_shift=p_shift,
            debug_ring=debug_ring,
            balanced=True,
            rescale_threshold=rescale_threshold,
            l_from_p8=l_from_p8,
            len_sub=len_sub,
            early_trigger=early_trigger,
            wide_plan=wide_plan,
            quad_plan=quad_plan,
            two_slot=two_slot,
            two_slot_pack=two_slot_pack,
        )
        assert kernel.ws_bytes == cp_partial_balanced_ws_bytes(
            q_len * gh, n_splits, 2 if two_slot else 1
        )
    else:
        kernel = Q35CpPartialAttnKernel(
            q_len=q_len,
            gh=gh,
            hql=hql,
            cp_size=cp_size,
            n_splits=n_splits,
            bmm1_from_ptr=bmm1_from_ptr,
            bmm2_from_ptr=bmm2_from_ptr,
            o_vec16=o_vec16,
            use_pdl=use_pdl,
            kv_pre_wait=kv_pre_wait,
            fresh_margin_pages=fresh_margin,
            debug_ts=debug_ts,
            unsafe_skip_wait=unsafe_skip_wait,
            p_shift=p_shift,
            debug_ring=debug_ring,
            rescale_threshold=rescale_threshold,
            l_from_p8=l_from_p8,
            len_sub=len_sub,
            early_trigger=early_trigger,
        )
        assert (
            kernel.ws_req_bytes
            == cp_partial_geometry(q_len * gh, n_splits)["ws_req_bytes"]
        )
    dev = torch.device("cuda")
    f32 = torch.empty(1024, dtype=torch.float32, device=dev)
    i32 = torch.zeros(64, dtype=torch.int32, device=dev)
    u8 = torch.empty(64 * HEAD_DIM, dtype=torch.uint8, device=dev)
    bf = torch.empty(1024, dtype=torch.bfloat16, device=dev)
    i64 = torch.zeros(16, dtype=torch.int64, device=dev)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    fp8 = cutlass.Float8E4M3FN
    a = _ALIGN
    I32 = cutlass.Int32
    return cute.compile(
        kernel,
        _ptr(u8, fp8, a["q"]),
        _ptr(u8, fp8, a["kv"]),
        _ptr(u8, fp8, a["kv"]),
        I32(64),
        _ptr(i32, I32, a["tab"]),
        I32(2),
        I32(1),
        _ptr(i32, I32, a["seq"]),
        I32(0),
        _ptr(f32, cutlass.Float32, a["bmm1"]),
        cutlass.Float32(1.0),
        _ptr(f32, cutlass.Float32, a["bmm2"]),
        cutlass.Float32(1.0),
        _ptr(bf, cutlass.BFloat16, 16 if o_vec16 else 4),
        I32(HEAD_DIM),
        I32(HEAD_DIM),
        I32(HEAD_DIM),
        _ptr(f32, cutlass.Float32, a["lse"]),
        I32(1),
        I32(1),
        I32(1),
        _ptr(f32, cutlass.Float32, a["ws"]),
        _ptr(i64, cutlass.Int64, a["cnt"]),
        _ptr(i64, cutlass.Int64, a["dbg"]),
        I32(1),
        stream,
    )


def q35_cp_fused_partial(
    q_all: torch.Tensor,
    kv_pool: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    q_len: int,
    cp_rank: int,
    cp_size: int,
    bmm1_scale: float | torch.Tensor,
    bmm2_scale: float | torch.Tensor,
    out_o: torch.Tensor,
    out_lse: torch.Tensor,
    sm_count: int = 0,
    enable_pdl: bool = False,
    *,
    debug: torch.Tensor | None = None,
    v_pool: torch.Tensor | None = None,
    unsafe_skip_wait: bool = False,
    n_splits: int | None = None,
    fresh_margin: int | None = None,
    counters: torch.Tensor | None = None,
    p_shift: float | None = None,
    balanced: bool | None = None,
    rescale_threshold: float | None = None,
    l_from_p8: bool = False,
    len_sub: int = 0,
    early_trigger: bool = False,
    plan: str | None = None,
    pack: bool | None = None,
) -> None:
    """This CP rank's partial attention of every group head over the pages it owns (q35_cp_fused_partial_ref).

    q_all [cp, rows, hq, D] FP8 (the all-gather output viewed member-major; the native layout, strides (hq * D,
    cp * hq * D, D, 1) at a 16-byte aligned address, is read in place, any other view through a copy); kv_pool [blocks,
    1, 64, D] FP8; block_tables [>= b, 2, max_pages] int32 (K / V block ids, unit stride on pages); seq_lens [>= b]
    int32 (L incl. the q_len new tokens); bmm1_scale float (natural-log domain) or a [>= 2] FP32 tensor (raw, raw *
    log2e); bmm2_scale float or a [>= 1] FP32 tensor. out_o [cp, rows, hq, D] bf16 (unit stride on D), out_lse [cp,
    rows, hq] FP32 (base 2). Test hooks: debug (int64 [b * splits, 16] %globaltimer stamps; [CTAs, 16] when balanced),
    unsafe_skip_wait (no dependency waits), n_splits (must come with its own ``counters``: every launch on a counter
    row must use the same split count; balanced: the grid's CTAs), fresh_margin, balanced (a test override of the
    split mode; the wide runtime path selects it automatically and an override needs its own ``counters``), plan
    ("lane", "wide", "quad" or "two": the wide split's plan variant instead of the one the batch selects; lane / wide
    / quad cover up to 32 / 64 / 128 requests and compute the same one-slot plan, "two" (up to 128) the two-slot plan,
    with [CTAs, 24] debug rows), pack (a test override of the packed two-slot plan).
    Numerics: p_shift (E4M3 P shift, default _P_SHIFT), rescale_threshold (lazy rescale threshold in log2 units,
    default min(8, 8.75 - p_shift); 0 = exact running max), l_from_p8 (row sums from the E4M3-rounded P).
    """
    import cuda.bindings.driver as cuda
    import cutlass

    cp, rows, hq, D = q_all.shape
    assert cp == cp_size and D == HEAD_DIM and q_all.dtype == torch.float8_e4m3fn
    assert rows % q_len == 0
    b = rows // q_len
    gh = cp * hq
    R = q_len * gh
    assert 1 <= R <= MAX_ROWS, f"q_len x group heads = {R} > {MAX_ROWS}"
    assert 0 <= cp_rank < cp_size
    assert (
        kv_pool.dim() == 4
        and kv_pool.shape[1] == 1
        and kv_pool.shape[2] == PAGE
        and kv_pool.shape[3] == HEAD_DIM
    )
    assert kv_pool.is_contiguous() and kv_pool.dtype in (
        torch.float8_e4m3fn,
        torch.uint8,
    )
    # v_pool: V in its own pool of kv_pool's shape; block_tables [b, P] then names the same page of both
    if v_pool is not None:
        assert (
            v_pool.shape == kv_pool.shape
            and v_pool.is_contiguous()
            and v_pool.dtype == kv_pool.dtype
        )
    if block_tables.dim() == 2:
        assert v_pool is not None, "a [b, P] page table needs v_pool"
        tab_sb, tab_skv = block_tables.stride(0), 0
    else:
        assert block_tables.dim() == 3 and block_tables.shape[1] == 2
        tab_sb, tab_skv = block_tables.stride(0), block_tables.stride(1)
    assert block_tables.shape[0] >= b
    assert block_tables.stride(-1) == 1 and block_tables.dtype == torch.int32
    assert (
        seq_lens.dtype == torch.int32
        and seq_lens.numel() >= b
        and seq_lens.stride(0) == 1
    )
    assert (
        tuple(out_o.shape) == (cp, rows, hq, D)
        and out_o.dtype == torch.bfloat16
        and out_o.stride(3) == 1
    )
    assert tuple(out_lse.shape) == (cp, rows, hq) and out_lse.dtype == torch.float32
    assert 1 <= b <= MAX_BATCH
    device = q_all.device
    sms = sm_count or _sm_count(device)
    state = _state(device, sms)
    bal = b > _LANE_PLAN_MAX_BATCH if balanced is None else bool(balanced)
    assert bal or b <= _LANE_PLAN_MAX_BATCH, "the equal split handles <= 32 requests"
    # plan variant: lane (<= 32 requests), wide (33..64), two-slot (65..128); `plan` overrides it in tests (also
    # "quad": the one-slot plan for 65..128)
    if plan is None:
        plan = (
            "two"
            if b > _WIDE_PLAN_MAX_BATCH
            else ("wide" if b > _LANE_PLAN_MAX_BATCH else "lane")
        )
    assert bal or plan == "lane", "plan variants are balanced-split plans"
    assert (
        b
        <= {
            "lane": _LANE_PLAN_MAX_BATCH,
            "wide": _WIDE_PLAN_MAX_BATCH,
            "quad": _QUAD_PLAN_MAX_BATCH,
            "two": _TWO_SLOT_MAX_BATCH,
        }[plan]
    )
    wide = bal and plan == "wide"
    quad = bal and plan == "quad"
    two = bal and plan == "two"
    two_pack = two and (_TWO_SLOT_PACK if pack is None else bool(pack))
    if two and debug is not None:
        assert debug.shape[-1] >= 24, (
            "the two-slot kernel's debug rows hold 24 words (both slots' plans)"
        )
    if bal:
        splits = n_splits if n_splits else _num_ctas(sms)
        assert b < splits <= sms, (
            "balanced split: more CTAs than requests, every CTA co-resident (one per SM)"
        )
    else:
        splits = n_splits if n_splits else _num_splits(b, sms)
        assert splits * b <= sms, (
            "the in-kernel grid barrier needs every CTA co-resident (one per SM)"
        )
    if n_splits or balanced is not None:
        assert counters is not None, (
            "a split-count or split-mode override needs its own counters tensor"
        )
    cnt = state.counters if counters is None else counters
    assert (
        cnt.dtype == torch.int64
        and cnt.dim() == 2
        and cnt.shape[0] > b
        and cnt.shape[1] >= b
    )
    assert cnt.is_contiguous()
    if bal:
        ws = _balanced_workspace(
            state, cp_partial_balanced_ws_bytes(R, splits, 2 if two else 1)
        )
        if balanced is None and not state.balanced_logged:
            state.balanced_logged = True
            logger.info(
                f"CP-lite first wide-plan call (q_len {q_len}, batch {b}, "
                f"cp_rank {cp_rank}, {splits} CTAs)"
            )
        if wide and balanced is None and not state.wide_logged:
            state.wide_logged = True
            logger.info(
                f"CP-lite first call above {_LANE_PLAN_MAX_BATCH} requests on the wide-plan kernel "
                f"(q_len {q_len}, batch {b}, cp_rank {cp_rank}, {splits} CTAs)"
            )
        if (quad or two) and balanced is None and not state.quad_logged:
            state.quad_logged = True
            logger.info(
                f"CP-lite first call above {_WIDE_PLAN_MAX_BATCH} requests on the "
                f"{'two-slot' if two else 'quad-plan'} kernel (q_len {q_len}, "
                f"batch {b}, cp_rank {cp_rank}, {splits} CTAs)"
            )
        if two_pack and balanced is None and not state.pack_logged:
            state.pack_logged = True
            logger.info(
                f"CP-lite first two-slot call on the packed plan (q_len {q_len}, "
                f"batch {b}, cp_rank {cp_rank}, {splits} CTAs)"
            )
    else:
        need = b * cp_partial_geometry(R, splits)["ws_req_bytes"]
        ws = state.workspace
        assert ws.numel() * 4 >= need, "CP partial workspace too small"
    # q: the native all-gather layout is one [b * rows_per_request, 256] matrix read by TMA in place
    native = (
        q_all.stride(3) == 1
        and q_all.stride(2) == D
        and q_all.stride(0) == hq * D
        and q_all.stride(1) == cp * hq * D
        and q_all.data_ptr() % 16 == 0
    )
    q_src = q_all if native else q_all.permute(1, 0, 2, 3).contiguous()
    b1_ptr = isinstance(bmm1_scale, torch.Tensor)
    b2_ptr = isinstance(bmm2_scale, torch.Tensor)
    if b1_ptr:
        assert (
            bmm1_scale.numel() >= 2
            and bmm1_scale.dtype == torch.float32
            and bmm1_scale.stride(0) == 1
        )
    if b2_ptr:
        assert bmm2_scale.numel() >= 1 and bmm2_scale.dtype == torch.float32
    o_vec16 = out_o.data_ptr() % 16 == 0 and all(
        out_o.stride(i) % 8 == 0 for i in range(3)
    )
    assert out_o.data_ptr() % 4 == 0 and all(
        out_o.stride(i) % 2 == 0 for i in range(3)
    ), "out_o: 4-byte aligned rows (alltoall_merge reads it as int32)"
    ring = state.dbg_ring if debug is None else None
    if bal:
        fn = _compiled(
            q_len,
            gh,
            hq,
            cp,
            splits,
            b1_ptr,
            b2_ptr,
            o_vec16,
            bool(enable_pdl),
            _KV_PRE_WAIT,
            _FRESH_MARGIN if fresh_margin is None else fresh_margin,
            debug is not None,
            unsafe_skip_wait,
            _P_SHIFT if p_shift is None else float(p_shift),
            _DBG_SLOTS if (ring is not None and not two) else 0,
            True,
            rescale_threshold,
            bool(l_from_p8),
            int(len_sub),
            bool(early_trigger),
            wide,
            quad,
            two,
            two_pack,
        )
    else:
        fn = _compiled(
            q_len,
            gh,
            hq,
            cp,
            splits,
            b1_ptr,
            b2_ptr,
            o_vec16,
            bool(enable_pdl),
            _KV_PRE_WAIT,
            _FRESH_MARGIN if fresh_margin is None else fresh_margin,
            debug is not None,
            unsafe_skip_wait,
            _P_SHIFT if p_shift is None else float(p_shift),
            _DBG_SLOTS if ring is not None else 0,
            False,
            rescale_threshold,
            bool(l_from_p8),
            int(len_sub),
            bool(early_trigger),
        )
    fp8 = cutlass.Float8E4M3FN
    f32 = cutlass.Float32
    I32 = cutlass.Int32
    a = _ALIGN
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    dummy = ws
    fn(
        _ptr(q_src, fp8, a["q"]),
        _ptr(kv_pool, fp8, a["kv"]),
        _ptr(kv_pool if v_pool is None else v_pool, fp8, a["kv"]),
        I32(kv_pool.shape[0] * PAGE),
        _ptr(block_tables, I32, a["tab"]),
        I32(tab_sb),
        I32(tab_skv),
        _ptr(seq_lens, I32, a["seq"]),
        I32(cp_rank),
        _ptr(bmm1_scale if b1_ptr else dummy, f32, a["bmm1"]),
        f32(0.0 if b1_ptr else float(bmm1_scale) * LOG2E),
        _ptr(bmm2_scale if b2_ptr else dummy, f32, a["bmm2"]),
        f32(0.0 if b2_ptr else float(bmm2_scale)),
        _ptr(out_o, cutlass.BFloat16, 16 if o_vec16 else 4),
        I32(out_o.stride(0)),
        I32(out_o.stride(1)),
        I32(out_o.stride(2)),
        _ptr(out_lse, f32, a["lse"]),
        I32(out_lse.stride(0)),
        I32(out_lse.stride(1)),
        I32(out_lse.stride(2)),
        _ptr(ws, f32, a["ws"]),
        _ptr(cnt[b], cutlass.Int64, a["cnt"]),
        _ptr(
            debug if debug is not None else (ring if ring is not None else cnt[b]),
            cutlass.Int64,
            a["dbg"],
        ),
        I32(b),
        stream,
    )

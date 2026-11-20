# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``spec_sample_split`` (env ``MUSE_VR_SPEC_SAMPLE_SPLIT=1``, default off).

Under non-greedy sampling, SGLang's DFLASH verify (the selector path, ``dspark_accept._accept_sampling_core``) runs
``chain_speculative_sampling_triton``: one Triton program per request walks the draft chain (accept/reject per
position) and then samples the final token from the residual distribution with two passes over the whole
vocabulary, a normalizer sum and a CDF search, each in tiles of 4096 entries. One program per request scans the full
vocabulary twice on its own, so the call takes about as long at one request as at many.

This component runs the same arithmetic split across the vocabulary tiles:

1. ``_split_sum_kernel`` (grid: requests x tiles) repeats the chain walk (scalar loads; tile 0 also performs every store
   of the stock walk) and writes each tile's sum, computed with the same ``tl.sum`` over the same 4096-entry tile.
2. ``_split_cdf_kernel`` (grid: requests x tiles) adds the tile sums in tile order, as the stock loop accumulates them,
   which gives both the normalizer and the running sum in front of its own tile, then applies the stock tile step
   (``tl.cumsum``, ``> u * normalizer``, first match). The smallest matching vocabulary index of the row is kept with
   an atomic minimum; with no match the stock fallback (last vocabulary index) remains. The program that completes
   the row (an acq_rel arrival counter at gpu scope) stores the sampled token.

The stock loop takes the first tile with a match and its first matching index, which is the smallest matching index
of the row, so the result is the same token for the same inputs. The block-verification variant, positional calls,
other dtypes or layouts, CPU tensors and CUDA-graph capture use the stock function. (SGLang's hook registry also
rebinds other modules' imports of the same function; they get the same treatment.) ``tests/test_spec_sample_split.py`` checks bitwise equality of every
output buffer against the stock kernel over the served shapes and times both.
"""

# The two Triton kernels below are derived from SGLang (https://github.com/sgl-project/sglang), Apache License 2.0;
# see include/LICENSE.sglang. Source: python/sglang/kernels/ops/speculative/reject_sampling.py
# (speculative_sampling_classic_kernel, non-block-verification branch) at revision a55ab7d.
# Modified by NVIDIA: the final-token pass is split across vocabulary tiles in two launches (tile sums, then the
# per-tile CDF step with an atomic minimum over matching indices); per-tile arithmetic is unchanged.

from __future__ import annotations

import logging
import threading

import torch
import triton
import triton.language as tl

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
}


logger = logging.getLogger(__name__)

NAME = "spec_sample_split"  # env MUSE_VR_SPEC_SAMPLE_SPLIT
TARGET = "sglang.kernels.ops.speculative.dspark.dspark_accept.chain_speculative_sampling_triton"
BLOCK_V = 4096  # the stock tile width (reject_sampling.chain_speculative_sampling_triton passes BLOCK_V=4096)
NUM_WARPS = 4  # the stock launch for the non-block-verification variant

# sha256 of the SGLang sources (relative to the directory holding the ``sglang`` package) this component relies on:
# the kernel it mirrors and the call site whose arguments it receives.
PINS = {
    "sglang/kernels/ops/speculative/reject_sampling.py": "62caac17755082e2f89b1c81a743f55c81c31b7ff0485b7533c0267e72e9262c",
    "sglang/kernels/ops/speculative/dspark/dspark_accept.py": (
        "6202a51f6d2f3a79fef3cbef2aee04296640087be2402d8b11a7f1b394015a9c",
        "4d87d25d0510a309e7f3484c0d2c6d806da39bd5557096c49b9ac2a92e5c3ddf",
    ),
}

# Triton releases the bitwise test (tests/test_spec_sample_split.py) passed with: the per-tile tl.sum / tl.cumsum must
# lower to the same reduction and scan order as in the stock kernel, which depends on the compiler.
VALIDATED_TRITON = ("3.8.0", "3.9.0")

_lock = threading.Lock()
_scratch: dict = {}


def enabled() -> bool:
    return bed.enabled(NAME)


def pins_match() -> tuple[bool, str]:
    for rel, want in PINS.items():
        if not bed.pinned(rel, want):
            return False, f"{rel} differs from the pinned revision"
    if triton.__version__ not in VALIDATED_TRITON:
        return (
            False,
            f"Triton {triton.__version__} not in the validated set {VALIDATED_TRITON}",
        )
    return True, "ok"


@triton.jit
def _chain_walk(
    Predicts,
    AcceptIndex,
    Candidates,
    RetriveIndex,
    UniformSamples,
    TargetProbs,
    DraftProbs,
    pid,
    do_store,
    stride_cand_b,
    stride_cand_s,
    stride_idx_b,
    stride_idx_s,
    stride_uni_b,
    stride_uni_s,
    stride_tp_b,
    stride_tp_s,
    stride_tp_v,
    stride_dp_b,
    stride_dp_s,
    stride_dp_v,
    NUM_SLOTS: tl.constexpr,
):
    cur_prob_row = 0
    cand_ptr_base = Candidates + pid * stride_cand_b
    idx_ptr_base = RetriveIndex + pid * stride_idx_b
    uni_ptr_base = UniformSamples + pid * stride_uni_b

    root_global_idx = tl.load(idx_ptr_base + 0 * stride_idx_s)
    if do_store:
        tl.store(AcceptIndex + pid * stride_idx_b + 0 * stride_idx_s, root_global_idx)
    last_accepted_global_idx = root_global_idx

    num_accept = 0
    step = 1
    continue_verifying = 1
    while (step < NUM_SLOTS) and (continue_verifying == 1):
        draft_token = tl.load(cand_ptr_base + step * stride_cand_s)
        p = tl.load(
            TargetProbs
            + pid * stride_tp_b
            + cur_prob_row * stride_tp_s
            + draft_token * stride_tp_v
        )
        q = tl.load(
            DraftProbs
            + pid * stride_dp_b
            + cur_prob_row * stride_dp_s
            + draft_token * stride_dp_v
        )
        coin = tl.load(uni_ptr_base + (step - 1) * stride_uni_s)
        q_is_prob = (q > 0.0) & (q <= 1.0)
        if q_is_prob & (coin * q < p):
            num_accept += 1
            cur_prob_row = step
            if do_store:
                tl.store(Predicts + last_accepted_global_idx, draft_token)
            last_accepted_global_idx = tl.load(idx_ptr_base + step * stride_idx_s)
            if do_store:
                tl.store(
                    AcceptIndex + pid * stride_idx_b + num_accept * stride_idx_s,
                    last_accepted_global_idx,
                )
            step += 1
        else:
            continue_verifying = 0
    return num_accept, cur_prob_row, last_accepted_global_idx


@triton.jit
def _tile_values(
    TargetProbs,
    DraftProbs,
    pid,
    cur_prob_row,
    all_drafts_accepted,
    v_start,
    stride_tp_b,
    stride_tp_s,
    stride_tp_v,
    stride_dp_b,
    stride_dp_s,
    stride_dp_v,
    VOCAB_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    residual_scale = 1.0
    tp_base_ptr = TargetProbs + (pid * stride_tp_b) + (cur_prob_row * stride_tp_s)
    dp_base_ptr_safe = DraftProbs + (pid * stride_dp_b) + (cur_prob_row * stride_dp_s)
    v_offsets = v_start + tl.arange(0, BLOCK)
    mask = v_offsets < VOCAB_SIZE
    p_ptr = tp_base_ptr + v_offsets * stride_tp_v
    p_val = tl.load(p_ptr, mask=mask, other=0.0)
    if all_drafts_accepted:
        val = p_val
    else:
        q_ptr = dp_base_ptr_safe + v_offsets * stride_dp_v
        q_val = tl.load(q_ptr, mask=mask, other=0.0)
        q_val = tl.where((q_val >= 0.0) & (q_val <= 1.0), q_val, 0.0)
        diff = residual_scale * p_val - q_val
        val = tl.where(diff > 0.0, diff, 0.0)
    return val


@triton.jit
def _split_sum_kernel(
    Predicts,
    AcceptIndex,
    AcceptTokenNum,
    Candidates,
    RetriveIndex,
    UniformSamples,
    TargetProbs,
    DraftProbs,
    TileSums,
    Result,
    stride_cand_b,
    stride_cand_s,
    stride_idx_b,
    stride_idx_s,
    stride_uni_b,
    stride_uni_s,
    stride_tp_b,
    stride_tp_s,
    stride_tp_v,
    stride_dp_b,
    stride_dp_s,
    stride_dp_v,
    NUM_SLOTS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    NUM_TILES: tl.constexpr,
):
    pid = tl.program_id(0)
    tile = tl.program_id(1)
    num_accept, cur_prob_row, last_idx = _chain_walk(
        Predicts,
        AcceptIndex,
        Candidates,
        RetriveIndex,
        UniformSamples,
        TargetProbs,
        DraftProbs,
        pid,
        tile == 0,
        stride_cand_b,
        stride_cand_s,
        stride_idx_b,
        stride_idx_s,
        stride_uni_b,
        stride_uni_s,
        stride_tp_b,
        stride_tp_s,
        stride_tp_v,
        stride_dp_b,
        stride_dp_s,
        stride_dp_v,
        NUM_SLOTS,
    )
    if tile == 0:
        tl.store(AcceptTokenNum + pid, num_accept)
        tl.store(Result + pid, VOCAB_SIZE - 1)
    all_drafts_accepted = num_accept == NUM_SLOTS - 1
    val = _tile_values(
        TargetProbs,
        DraftProbs,
        pid,
        cur_prob_row,
        all_drafts_accepted,
        tile * BLOCK_V,
        stride_tp_b,
        stride_tp_s,
        stride_tp_v,
        stride_dp_b,
        stride_dp_s,
        stride_dp_v,
        VOCAB_SIZE,
        BLOCK_V,
    )
    tl.store(TileSums + pid * NUM_TILES + tile, tl.sum(val))


@triton.jit
def _split_cdf_kernel(
    Predicts,
    Candidates,
    RetriveIndex,
    UniformSamples,
    UniformSamplesFinal,
    TargetProbs,
    DraftProbs,
    TileSums,
    Result,
    Arrived,
    stride_cand_b,
    stride_cand_s,
    stride_idx_b,
    stride_idx_s,
    stride_uni_b,
    stride_uni_s,
    stride_tp_b,
    stride_tp_s,
    stride_tp_v,
    stride_dp_b,
    stride_dp_s,
    stride_dp_v,
    NUM_SLOTS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    NUM_TILES: tl.constexpr,
):
    pid = tl.program_id(0)
    tile = tl.program_id(1)
    num_accept, cur_prob_row, last_idx = _chain_walk(
        Predicts,
        Predicts,
        Candidates,
        RetriveIndex,
        UniformSamples,
        TargetProbs,
        DraftProbs,
        pid,
        False,
        stride_cand_b,
        stride_cand_s,
        stride_idx_b,
        stride_idx_s,
        stride_uni_b,
        stride_uni_s,
        stride_tp_b,
        stride_tp_s,
        stride_tp_v,
        stride_dp_b,
        stride_dp_s,
        stride_dp_v,
        NUM_SLOTS,
    )
    all_drafts_accepted = num_accept == NUM_SLOTS - 1

    # Tile sums accumulated in tile order from 0.0: the running value in front of this tile and the full normalizer.
    norm_sum = 0.0
    cum_sum = 0.0
    for t in tl.static_range(NUM_TILES):
        if t == tile:
            cum_sum = norm_sum
        norm_sum += tl.load(TileSums + pid * NUM_TILES + t)

    coin_final = tl.load(UniformSamplesFinal + pid)
    target_u = coin_final * norm_sum
    v_start = tile * BLOCK_V
    val = _tile_values(
        TargetProbs,
        DraftProbs,
        pid,
        cur_prob_row,
        all_drafts_accepted,
        v_start,
        stride_tp_b,
        stride_tp_s,
        stride_tp_v,
        stride_dp_b,
        stride_dp_s,
        stride_dp_v,
        VOCAB_SIZE,
        BLOCK_V,
    )
    block_cumsum = tl.cumsum(val, axis=0)
    total_cumsum = cum_sum + block_cumsum
    candidates_mask = total_cumsum > target_u
    has_match = tl.max(candidates_mask, axis=0)
    if has_match:
        match_idx = tl.argmax(candidates_mask.to(tl.int32), axis=0)
        tl.atomic_min(Result + pid, v_start + match_idx, sem="relaxed", scope="gpu")
    # Arrival counter (acq_rel RMW, gpu scope): each program's atomic minimum above is ordered before its arrival
    # (CTA barrier, then the release of the RMW) and the last arrival acquires every earlier one through the RMW
    # chain on the counter, so its read of the minimum below sees all of them.
    tl.debug_barrier()
    arrived = tl.atomic_add(Arrived + pid, 1, sem="acq_rel", scope="gpu")
    if arrived == NUM_TILES - 1:
        final_token = tl.atomic_add(Result + pid, 0, sem="relaxed", scope="gpu")
        tl.store(Predicts + last_idx, final_token)
        tl.atomic_xchg(
            Arrived + pid, 0, sem="relaxed", scope="gpu"
        )  # ready for the next launch (stream order)


def _buffers(device: torch.device, bs: int, num_tiles: int):
    """Scratch for ``bs`` requests (whole buffers; the kernels index them by request only), one set per stream so
    calls queued on different streams never share tile sums, results or arrival counters."""
    key = (device.index, torch._C._cuda_getCurrentRawStream(device.index), num_tiles)
    s = _scratch.get(key)
    if s is None or s[0] < bs:
        with _lock:
            cap = max(bs, 0 if s is None else 2 * s[0])
            s = (
                cap,
                torch.empty(
                    (cap, num_tiles), dtype=torch.float32, device=device
                ),  # tile sums
                torch.empty(
                    (cap,), dtype=torch.int32, device=device
                ),  # sampled token (atomic minimum)
                torch.zeros(
                    (cap,), dtype=torch.int32, device=device
                ),  # arrival counters, left at zero
            )
            _scratch[key] = s
    return s[1], s[2], s[3]


def eligible(
    candidates,
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    draft_probs,
    block_verification,
) -> str | None:
    if block_verification:
        return "block verification"
    ts = (
        candidates,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
    )
    if not all(isinstance(t, torch.Tensor) and t.is_cuda for t in ts):
        return "not cuda"
    if target_probs.dtype != torch.float32 or draft_probs.dtype != torch.float32:
        return "dtype"
    if (
        candidates.dtype != torch.int64
        or uniform_samples.dtype != torch.float32
        or (uniform_samples_for_final_sampling.dtype != torch.float32)
    ):
        return "dtype"
    if target_probs.dim() != 3 or draft_probs.dim() != 3 or candidates.dim() != 2:
        return "layout"
    if torch.cuda.is_current_stream_capturing() and not _capture_allowed():
        return "graph capture"
    return None


def _capture_allowed() -> bool:
    """Capture is allowed only inside ``sampling_graph``'s capture of the sampling call (scratch then belongs to it).

    ``sampling_graph`` is a host-side runtime component and is not part of this package, so no capture is allowed here."""
    return False


_compiled: dict = {}


def _launch(kernel, grid, tensors, ints, consts):
    """``kernel[grid](*tensors, *ints, **consts)``. The compiled kernel of the first call is kept, keyed by what the
    launch specializes on apart from tensor dtypes, which ``eligible`` fixes (device, integer values, constexprs;
    tensors must be 16-byte aligned, else every call takes the regular path), and later calls launch it directly
    without ``JITFunction.run``'s per-call argument binding."""
    for t in tensors:
        if t.data_ptr() & 15:
            kernel[grid](*tensors, *ints, **consts, num_warps=NUM_WARPS)
            return
    dev = torch.cuda.current_device()
    cvals = tuple(consts.values())
    key = (kernel, dev, ints, cvals)
    ck = _compiled.get(key)
    if ck is None:
        _compiled[key] = kernel[grid](*tensors, *ints, **consts, num_warps=NUM_WARPS)
        return
    from triton import knobs

    stream = torch._C._cuda_getCurrentRawStream(dev)
    enter = knobs.runtime.launch_enter_hook
    meta = (
        None
        if enter is None
        else ck.launch_metadata(grid, stream, *tensors, *ints, *cvals)
    )
    ck.run(
        grid[0],
        grid[1],
        1,
        stream,
        ck.function,
        ck.packed_metadata,
        meta,
        enter,
        knobs.runtime.launch_exit_hook,
        *tensors,
        *ints,
        *cvals,
    )


def split_chain_speculative_sampling(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    draft_probs,
):
    """Same outputs as ``chain_speculative_sampling_triton`` (non-block-verification variant)."""
    batch_size, num_slots = candidates.shape
    vocab_size = target_probs.shape[-1]
    num_tiles = triton.cdiv(vocab_size, BLOCK_V)
    tile_sums, result, arrived = _buffers(target_probs.device, batch_size, num_tiles)
    strides = (
        candidates.stride(0),
        candidates.stride(1),
        retrive_index.stride(0),
        retrive_index.stride(1),
        uniform_samples.stride(0),
        uniform_samples.stride(1),
        target_probs.stride(0),
        target_probs.stride(1),
        target_probs.stride(2),
        draft_probs.stride(0),
        draft_probs.stride(1),
        draft_probs.stride(2),
    )
    consts = dict(
        NUM_SLOTS=num_slots, VOCAB_SIZE=vocab_size, BLOCK_V=BLOCK_V, NUM_TILES=num_tiles
    )
    grid = (batch_size, num_tiles)
    _launch(
        _split_sum_kernel,
        grid,
        (
            predicts,
            accept_index,
            accept_token_num,
            candidates,
            retrive_index,
            uniform_samples,
            target_probs,
            draft_probs,
            tile_sums,
            result,
        ),
        strides,
        consts,
    )
    _launch(
        _split_cdf_kernel,
        grid,
        (
            predicts,
            candidates,
            retrive_index,
            uniform_samples,
            uniform_samples_for_final_sampling,
            target_probs,
            draft_probs,
            tile_sums,
            result,
            arrived,
        ),
        strides,
        consts,
    )


def chain_speculative_sampling(original, *args, **kwargs):
    """``chain_speculative_sampling_triton`` with the split final-token pass when eligible (keyword call as the
    pinned call site makes it)."""
    if args:
        bed.count(NAME, "stock:positional call")
        return original(*args, **kwargs)
    reason = eligible(
        kwargs.get("candidates"),
        kwargs.get("uniform_samples"),
        kwargs.get("uniform_samples_for_final_sampling"),
        kwargs.get("target_probs"),
        kwargs.get("draft_probs"),
        bool(kwargs.get("block_verification", False)),
    )
    if reason is None:
        outs = (
            kwargs.get("predicts"),
            kwargs.get("accept_index"),
            kwargs.get("accept_token_num"),
        )
        ri = kwargs.get("retrive_index")
        if not all(
            isinstance(t, torch.Tensor) and t.dtype == torch.int32 for t in outs
        ) or not (isinstance(ri, torch.Tensor) and ri.dtype == torch.int64):
            reason = "dtype"
    if reason is not None:
        bed.count(NAME, "stock:" + reason)
        return original(*args, **kwargs)
    bs = kwargs["candidates"].shape[0]
    bed.count(NAME, f"split:bs={bs}")
    split_chain_speculative_sampling(
        kwargs["predicts"],
        kwargs["accept_index"],
        kwargs["accept_token_num"],
        kwargs["candidates"],
        kwargs["retrive_index"],
        kwargs["uniform_samples"],
        kwargs["uniform_samples_for_final_sampling"],
        kwargs["target_probs"],
        kwargs["draft_probs"],
    )
    return None


def register() -> bool:
    if not enabled():
        return False
    ok, why = pins_match()
    if not ok:
        bed.report(NAME, False, reason=why)
        return False
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(TARGET, chain_speculative_sampling, HookType.AROUND)
    bed.report(NAME, True)
    return True

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``verify_topk_sample`` (env ``MUSE_VR_VERIFY_TOPK_SAMPLE=1``, default off).

DFLASH target verification with temperature sampling and a uniform ``top_k`` (the checkpoint generation config of
Muse Spark Open sets ``top_k`` 64, which requests without sampling fields get) runs, in SGLang's selector path:

1. ``LogitsProcessor.finish_deferred_gather``: all-gather of every rank's fp32 vocabulary shard of the verify logits
   and the soft cap, giving full rows ``[bs * block, vocab]``;
2. ``build_speculative_verify_target_probs``: division by the temperature, ``torch.topk`` over the whole vocabulary,
   softmax over the ``k`` selected logits and a scatter into a dense zero matrix ``[bs, block, vocab]``;
3. the draft probabilities scattered into a dense ``[bs, block - 1, vocab]`` buffer, two ``torch.rand`` draws and
   ``chain_speculative_sampling_triton`` (accept/reject walk along the chain, then the final token drawn from the
   residual ``max(p - q, 0)`` of the last row with two passes over the vocabulary).

The sampled distribution only involves the ``k`` largest scaled logits of each row. This component computes them
without materializing full rows:

- each rank's shard pass (``_shard_prep_kernel``) soft-caps its shard with the arithmetic of the stock gather kernel
  and divides by the temperature (IEEE division, as the stock tensor division), keeping per 64-column bucket the
  largest key of ``torch.topk``'s radix order of fp32 values; ``csrc/verify_topk.cuh`` then reads only the ``k``
  buckets that can hold the shard's top ``k`` and writes it in ``torch.topk``'s order (value descending, equal values
  by ascending id);
- the ranks all-gather these records (``2k + 2`` words per row instead of the shard width) and every rank merges them
  into the row's global top ``k``. Shards are contiguous id ranges, so each local top ``k`` holds every global winner
  from its shard: the merged set, its values and their order equal ``torch.topk`` over the gathered row;
- ``torch.softmax`` runs on the same ``[rows, k]`` values as in the stock path, so the ``k`` probabilities are the same
  bits;
- the same two ``torch.rand`` draws (same shapes, same order) feed three Triton kernels derived from the stock
  sampling kernel: the chain walk looks the draft token up in the row's ``k`` (id, probability) pairs and in the
  selector's candidate list (the dense matrices hold exactly these values and zeros elsewhere); the residual of the
  final row is written at its ``k`` ids into a persistent zero row, which the stock tile code (4096-wide tiles,
  ``tl.sum`` / ``tl.cumsum`` over tiles loaded from memory, tile sums accumulated in tile order) then scans split
  across tiles, as ``spec_sample_split`` does; the last tile of a row stores the token and restores the zeros;
- the accept lengths and bonus tokens are returned as the stock path returns them; the stock tensor-parallel
  synchronization and commit run unchanged.

Under the synthetic-acceptance benchmark mode (``SGLANG_SIMULATE_ACC_LEN`` > 0) the sampling code is the same as with
real acceptance, and the mode's scaffolding runs completely and unchanged: the full-row verify-logits gather (the
wrapped stock gather), the worker's ``torch.argmax`` over those rows and ``apply_dflash_simulated_acceptance``. The
component never skips, feeds or replaces any of it (it does not read or write ``target_predict``), so the
synthetic-mode gain is the real-mode gain minus the full gather that only the scaffolding needs.

Every output (accept lengths, bonus tokens, ``predicts`` / ``accept_index`` buffers) is meant to be bit-identical to
the stock path for the same RNG state: ``tests/test_verify_topk_sample.py`` (one GPU, shards simulated) compares them
with the stock functions, ``tests/dist_verify_topk_sample.py`` across ranks.

Engaged only when every request samples with the same ``top_k`` in [2, 64], ``top_p`` 1, no penalties, logit bias,
custom logit processor, grammar, logprobs or sampling-mask capture, the selector path without block verification,
fp32 shard logits of 4096..65536 columns per rank on at most 8 ranks, outside CUDA-graph capture and without
``sampling_graph``; every rank decides the same way from host-side batch state. Otherwise every stock function runs
as is. Hooks: ``_can_skip_verify_logits`` (decision), ``finish_deferred_gather`` (shard records instead of the full
gather), ``_accept_block`` (full rows if the selector branch is not taken after all), ``_selector_sampling_accept``
(sampling from the top-k), ``deferred_argmax`` (the preload only, on a greedy first verify step; its result is
returned unchanged). If another plugin hook on ``finish_deferred_gather`` wraps this one, the component simply
does not engage.

Where the running PyTorch build's ``torch.topk`` is not deterministic at the verify shape (``VALIDATED_STACKS``; on the
newer build: k = 64 at row counts at or above the device's multiprocessor count, where the choice among exactly tied
logits at the k-th value varies from call to call), the step keeps the stock path (``stock:topk not deterministic``):
no implementation can be bit-identical to a reference that differs from itself. ``MUSE_VR_VERIFY_TOPK_SAMPLE_LOWTIE=1``
(default off; not bit-identical to stock at those steps) keeps the top-k path there, with the same deterministic choice
as at every other shape (exactly tied logits by ascending id). Before the first top-k step on a device, a self-check runs
``torch.topk`` at k = 64 over full-width rows of exactly tied logits at 1 and T - 1 rows (T = the multiprocessor count):
two calls must agree and select the lowest ids; otherwise the component keeps the stock path
(``stock:topk self-check``).

Preload: the CUDA module, that self-check and the Triton kernels' compiles are not left to the first top-k step (a
served request). Every verify step ends in ``finish_deferred_gather`` (a sampled batch) or ``deferred_argmax`` (a
greedy batch: the worker sets the selector sample only for sampled batches), both on the step's shard rows, so on
the first verify step (the server's own warm-up request, greedy, before it reports ready) and outside graph
capture, ``_preload`` builds and loads the module, runs the self-check, and compiles the kernels at a top-k step's
specializations: the shard prep on that step's shard rows, the chain walk and final-token passes at the checkpoint's
default ``top_k`` and the selector's ``top_k``. It draws nothing from any RNG and runs on a side stream with buffers
of its own, so no buffer the serving path reads is written. Counters: ``preload:missed`` (a top-k step found no
preload) and ``jit:serving`` (a top-k step still compiled a kernel).
"""

# The Triton kernels below are derived from SGLang (https://github.com/sgl-project/sglang), Apache License 2.0;
# see include/LICENSE.sglang. Sources at revision a55ab7d: python/sglang/kernels/ops/activation/softcap.py
# (_gathered_softcap_logits_kernel: the soft-cap arithmetic) and python/sglang/kernels/ops/speculative/
# reject_sampling.py (speculative_sampling_classic_kernel, non-block-verification branch).
# Modified by NVIDIA: the soft cap runs on one rank's shard; the chain walk reads the target and draft probabilities
# from sparse (id, probability) lists; the final-token pass reads the residual from a zero row holding only the
# selected entries and is split across vocabulary tiles in two launches (tile sums, then the per-tile CDF step with
# an atomic minimum over matching indices). Per-tile arithmetic is unchanged.

from __future__ import annotations

import functools
import logging
import threading
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/distributed/parallel_state.py": ("get_tp_group",),
    "srt/speculative/dflash_worker_v2.py": ("SIMULATE_ACC_LEN",),
    "kernels/jit/utils/__init__.py": ("load_jit",),
    "srt/speculative/dflash_utils.py": ("_get_or_create_chain_verify_buffers",),
    "srt/layers/logits_processor.py": ("LogitsProcessor",),
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
}


logger = logging.getLogger(__name__)

NAME = "verify_topk_sample"  # env MUSE_VR_VERIFY_TOPK_SAMPLE
T_SKIP = (
    "sglang.srt.speculative.dflash_worker_v2.DFlashWorkerV2._can_skip_verify_logits"
)
T_GATHER = "sglang.srt.layers.logits_processor.LogitsProcessor.finish_deferred_gather"
T_ACCEPT = "sglang.srt.speculative.dflash_worker_v2.DFlashWorkerV2._accept_block"
T_SAMPLE = (
    "sglang.srt.speculative.dflash_worker_v2.DFlashWorkerV2._selector_sampling_accept"
)
T_ARGMAX = "sglang.srt.layers.logits_processor.LogitsProcessor.deferred_argmax"
_CUH = Path(__file__).resolve().parent / "csrc" / "verify_topk.cuh"

BLOCK_V = 4096  # stock tile width of the final-token pass
NUM_WARPS = 4  # stock launch of the non-block-verification kernel
PREP_BLOCK = 4096  # columns per program of the shard prep pass
BUCKET = 64  # columns per bucket maximum (csrc/verify_topk.cuh kBucket)
MAX_K = 64
MIN_COLS, MAX_COLS, MAX_WORLD = MAX_K * BUCKET, 65536, 8

# sha256 of the SGLang sources (relative to the directory holding the ``sglang`` package) this component relies on:
# the worker (call sites, selector path, commit), the verify-probability builder and chain buffers, the selector
# accept path, the sampling kernel it mirrors, the logits processor (deferred shard, gather) and the soft-cap kernel.
PINS = {
    "sglang/srt/speculative/dflash_worker_v2.py": (
        "7d462f5f175828bcc3b4d47fcdfa33cb6af4b14669ef016eeb79b1a5191eec8c",
        "8f82de8d666f81f3c1aa43dc651006f4182ff58593fdfe4eea28c5ee2a14bfc0",
    ),
    "sglang/srt/speculative/dflash_utils.py": (
        "6e0423578bade4a8c58b921430df678840c6fa2de6ea411fe58eb77c98d5a820",
        "35e885f930e95ed82a1f47a0dfa2dab583225d170c6cabe54f6c7bb4e3abbf79",
    ),
    "sglang/kernels/ops/speculative/dspark/dspark_accept.py": (
        "6202a51f6d2f3a79fef3cbef2aee04296640087be2402d8b11a7f1b394015a9c",
        "4d87d25d0510a309e7f3484c0d2c6d806da39bd5557096c49b9ac2a92e5c3ddf",
    ),
    "sglang/kernels/ops/speculative/reject_sampling.py": "62caac17755082e2f89b1c81a743f55c81c31b7ff0485b7533c0267e72e9262c",
    "sglang/srt/layers/logits_processor.py": (
        "ecc0318937eee486414f6ae7165f2d1de17c01794a25aae83a99af61cbb0e3d0",
        "3558330bccf3ea0e46aabdaa0dee6ae7daf294c8e15b9a4e9744dab6883c4398",
    ),
    "sglang/kernels/ops/activation/softcap.py": (
        "deeb037d3d3e28f24a243719fc9e2ebc2aa065d00138fb9e22065c1d36943e90",
        "b6591c8605984960ff88fcab92c6e9e36658f89497507948777a53d41d310ee8",
    ),
    "sglang/srt/speculative/dflash_info_v2.py": "c66fd2a9629c5f0f36ad4eca34f2910dc8e8af509fd8c6b2fa3fd2722f18004f",
}
# (PyTorch build (torch.version.git_version), Triton release) pairs the bitwise test passed with (torch.topk selection,
# softmax, soft-cap codegen, tile tl.sum / tl.cumsum lowering), each with the verify shapes where that build's
# torch.topk is not deterministic (measured with tests/test_verify_topk_sample.py):
#   None: none;
#   "k64_rows": k = 64 at row counts at or above the device's multiprocessor count (runtime attribute).
VALIDATED_STACKS = {
    ("e0474491d782a73b73cf132f732a8513b3c9f49d", "3.8.0"): None,
    ("7bf52b3a9782d3440a17eb3f00983795f3b12af0", "3.9.0"): "k64_rows",
}
VALIDATED_TRITON = tuple(sorted({t for _, t in VALIDATED_STACKS}))
VALIDATED_TORCH = tuple(sorted({b for b, _ in VALIDATED_STACKS}))

_tls = threading.local()
_lock = threading.Lock()
_state: dict = {"scratch": {}, "pool": {}, "views": {}, "flags": None, "tp_group": None}


def _pooled(
    name: str, shape: tuple, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """A contiguous view of ``shape`` at the start of a per (device, stream, name, dtype) buffer reused across verify
    steps (valid until the next call with the same name). The buffer grows to the largest element count requested so
    far and is replaced, not kept, when it grows, so memory stays at one buffer per name of the largest shape served
    (not one per batch shape). Every use is stream-ordered after the previous step's last reader on the same stream;
    a replaced buffer goes back to the caching allocator, which reuses it in that stream's order. The view of each
    shape is kept (host objects only, dropped with their buffer), so a repeated shape costs one lookup."""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    key = (idx, torch._C._cuda_getCurrentRawStream(idx), name, dtype)
    views = _state["views"]
    v = views.get((key, shape))
    if v is not None:
        return v
    n = 1
    for d in shape:
        n *= int(d)
    with _lock:
        t = _state["pool"].get(key)
        if t is None or t.numel() < n:
            t = torch.empty((n,), dtype=dtype, device=device)
            _state["pool"][key] = t
            for vk in [
                vk for vk in views if vk[0] == key
            ]:  # views of the replaced buffer
                del views[vk]
        v = views[(key, shape)] = t[:n].view(shape)
    return v


def _flags() -> dict:
    f = _state["flags"]
    if f is None:
        f = {
            "sampling_graph": bed.enabled("sampling_graph"),
            "lowtie": bed.enabled(NAME + "_lowtie"),
        }
        _state["flags"] = f
    return f


def enabled() -> bool:
    return bed.enabled(NAME)


def pins_match() -> tuple[bool, str]:
    for rel, want in PINS.items():
        if not bed.pinned(rel, want):
            return False, f"{rel} differs from the pinned revision"
    if (torch.version.git_version, triton.__version__) not in VALIDATED_STACKS:
        return False, (
            f"PyTorch build {torch.version.git_version[:12]} with Triton {triton.__version__} not in the "
            "validated set"
        )
    return True, "ok"


def _sm_count(device) -> int:
    idx = (
        device.index
        if device is not None and device.index is not None
        else torch.cuda.current_device()
    )
    sms = _state.setdefault("sms", {})
    n = sms.get(idx)
    if n is None:
        n = sms[idx] = int(torch.cuda.get_device_properties(idx).multi_processor_count)
    return n


def _tie_rows(rows: int, vocab: int, device) -> torch.Tensor:
    """Rows of exactly tied logits: 1.0 at the ids c with (c + row) % 3 == 0, else 0.0."""
    c = torch.arange(vocab, device=device)
    r = torch.arange(rows, device=device).unsqueeze(1)
    return ((c + r) % 3 == 0).to(torch.float32)


def self_check(device, vocab: int, rows_list=None) -> str | None:
    """None when torch.topk at k = MAX_K over ``vocab``-wide rows of exactly tied logits returns the same selection on
    two calls and the lowest ids among the ties, at rows 1 and T - 1 (T = the device's multiprocessor count; on the
    newer build the largest row count the top-k path keeps at that k); else the first failing row count. Deterministic
    inputs and shapes, so every rank reaches the same answer. The inputs are freed before returning."""
    t = _sm_count(device)
    for rows in rows_list or sorted({1, max(1, t - 1)}):
        x = _tie_rows(rows, vocab, device)
        a = torch.topk(x, MAX_K, dim=-1).indices
        b = torch.topk(x, MAX_K, dim=-1).indices
        del x
        r = torch.arange(rows, device=device).unsqueeze(1)
        want = (3 - r % 3) % 3 + 3 * torch.arange(MAX_K, device=device)
        if not (torch.equal(a, b) and torch.equal(torch.sort(a, dim=-1).values, want)):
            return "rows " + (
                "T-1" if rows == t - 1 else "T" if rows == t else str(rows)
            )
    return None


def _checked(device, vocab: int) -> str | None:
    """``self_check`` once per (device, vocabulary width)."""
    chk = _state.setdefault("check", {})
    key = (device.index, int(vocab))
    if key not in chk:
        chk[key] = self_check(device, vocab)
        if chk[key] is not None:
            logger.warning(
                "%s: torch.topk self-check failed (%s); the stock path is kept",
                NAME,
                chk[key],
            )
    return chk[key]


def topk_nondeterministic(rows: int, k: int, device=None) -> bool:
    """True when the running build's torch.topk is not deterministic for ``rows`` verify rows at ``k``
    (``VALIDATED_STACKS``): the step keeps the stock path there unless MUSE_VR_VERIFY_TOPK_SAMPLE_LOWTIE=1."""
    if (
        VALIDATED_STACKS.get((torch.version.git_version, triton.__version__))
        != "k64_rows"
    ):
        return False
    return k >= 64 and rows >= _sm_count(device)


@functools.lru_cache(maxsize=None)
def module():
    from sglang.kernels.jit.utils import load_jit

    return load_jit(
        "muse_vr_verify_topk",
        cuda_files=[str(_CUH)],
        cuda_wrappers=[
            ("select", "muse_vr::verify_topk::VerifyTopK::select"),
            ("merge", "muse_vr::verify_topk::VerifyTopK::merge"),
        ],
    )


# ----------------------------------------------------------------------------------------------------------------
# Kernels


@triton.jit
def _shard_prep_kernel(
    logits_ptr,
    temp_ptr,
    s_ptr,
    bkey_ptr,
    amax_ptr,
    cols,
    in_row_stride,
    s_row_stride,
    nb,
    nblk,
    rows_per_req,
    softcapping_value,
    BLOCK_SIZE: tl.constexpr,
    BUCKET: tl.constexpr,
    WITH_ARGMAX: tl.constexpr,
):
    row = tl.program_id(1).to(tl.int64)
    blk = tl.program_id(0)
    offs = blk.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < cols
    x = tl.load(logits_ptr + row * in_row_stride + offs, mask=mask)
    # soft cap: the arithmetic of the stock gather kernel
    x = x / softcapping_value
    x = libdevice.tanh(x)
    x = x * softcapping_value
    # temperature: IEEE division, as the stock tensor division (x / 1 == x exactly; NaN payloads aside)
    t = tl.load(temp_ptr + row // rows_per_req)
    if t == 1.0:
        s = x
    else:
        s = tl.math.div_rn(x, t)
    tl.store(s_ptr + row * s_row_stride + offs, s, mask=mask)

    ALL = tl.full((), 0xFFFFFFFF, tl.uint32)
    SIGN = tl.full((), 0x80000000, tl.uint32)
    # bucket maximum of the radix key of torch.topk (NaN highest, -0 below +0); 0 marks columns past the shard
    sb = s.to(tl.uint32, bitcast=True)
    key = tl.where(s != s, ALL, tl.where((sb & SIGN) != 0, sb ^ ALL, sb | SIGN))
    key = tl.where(mask, key, 0)
    bk = tl.max(tl.reshape(key, (BLOCK_SIZE // BUCKET, BUCKET)), axis=1)
    bidx = blk * (BLOCK_SIZE // BUCKET) + tl.arange(0, BLOCK_SIZE // BUCKET)
    tl.store(bkey_ptr + row * nb + bidx, bk.to(tl.int32, bitcast=True), mask=bidx < nb)
    if WITH_ARGMAX:
        # argmax order of torch.argmax on the soft-capped values (NaN highest, -0 equal to +0), then ascending column
        inv = (0xFFFFFFFF - offs).to(tl.uint64)
        ub = x.to(tl.uint32, bitcast=True)
        key2 = tl.where(
            x != x,
            ALL,
            tl.where(
                (ub & ~SIGN) == 0, SIGN, tl.where((ub & SIGN) != 0, ub ^ ALL, ub | SIGN)
            ),
        )
        comp2 = tl.where(mask, (key2.to(tl.uint64) << 32) | inv, 0)
        tl.store(
            amax_ptr + row * nblk + blk,
            tl.max(comp2, axis=0).to(tl.int64, bitcast=True),
        )


@triton.jit
def _walk_kernel(
    Predicts,
    AcceptIndex,
    AcceptTokenNum,
    Candidates,
    RetriveIndex,
    UniformSamples,
    TopProbs,
    TopIds,
    CandIds,
    QRows,
    Zero,
    Meta,
    Result,
    stride_cand_b,
    stride_cand_s,
    stride_idx_b,
    stride_idx_s,
    stride_uni_b,
    stride_uni_s,
    stride_ci_b,
    stride_ci_s,
    stride_q_b,
    stride_q_s,
    stride_z,
    NUM_SLOTS: tl.constexpr,
    K: tl.constexpr,
    KP: tl.constexpr,
    KQ: tl.constexpr,
    KQP: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    kk = tl.arange(0, KP)
    kmask = kk < K
    qq = tl.arange(0, KQP)
    qmask = qq < KQ

    cand_ptr_base = Candidates + pid * stride_cand_b
    idx_ptr_base = RetriveIndex + pid * stride_idx_b
    uni_ptr_base = UniformSamples + pid * stride_uni_b

    root_global_idx = tl.load(idx_ptr_base + 0 * stride_idx_s)
    tl.store(AcceptIndex + pid * stride_idx_b + 0 * stride_idx_s, root_global_idx)
    last_accepted_global_idx = root_global_idx

    num_accept = 0
    cur_prob_row = 0
    step = 1
    continue_verifying = 1
    while (step < NUM_SLOTS) and (continue_verifying == 1):
        draft_token = tl.load(cand_ptr_base + step * stride_cand_s)
        trow = (pid * NUM_SLOTS + cur_prob_row).to(tl.int64)
        ids = tl.load(TopIds + trow * K + kk, mask=kmask, other=-1)
        pr = tl.load(TopProbs + trow * K + kk, mask=kmask, other=0.0)
        p = tl.sum(tl.where(ids == draft_token, pr, 0.0), axis=0)
        cids = tl.load(
            CandIds + pid * stride_ci_b + cur_prob_row * stride_ci_s + qq,
            mask=qmask,
            other=-1,
        )
        qv = tl.load(
            QRows + pid * stride_q_b + cur_prob_row * stride_q_s + qq,
            mask=qmask,
            other=0.0,
        )
        q = tl.sum(tl.where(cids == draft_token, qv.to(tl.float32), 0.0), axis=0)
        coin = tl.load(uni_ptr_base + (step - 1) * stride_uni_s)
        # A proposal token must have positive probability under its draft distribution.
        q_is_prob = (q > 0.0) & (q <= 1.0)
        if q_is_prob & (coin * q < p):
            num_accept += 1
            cur_prob_row = step
            tl.store(Predicts + last_accepted_global_idx, draft_token)
            last_accepted_global_idx = tl.load(idx_ptr_base + step * stride_idx_s)
            tl.store(
                AcceptIndex + pid * stride_idx_b + num_accept * stride_idx_s,
                last_accepted_global_idx,
            )
            step += 1
        else:
            continue_verifying = 0

    tl.store(AcceptTokenNum + pid, num_accept)
    tl.store(Result + pid, VOCAB_SIZE - 1)
    all_drafts_accepted = num_accept == NUM_SLOTS - 1
    tl.store(Meta + pid * 4 + 0, cur_prob_row)
    tl.store(Meta + pid * 4 + 1, last_accepted_global_idx)

    # Residual of the final row at its K ids (zero elsewhere, as in the dense matrices).
    trow = (pid * NUM_SLOTS + cur_prob_row).to(tl.int64)
    ids = tl.load(TopIds + trow * K + kk, mask=kmask, other=0)
    p_val = tl.load(TopProbs + trow * K + kk, mask=kmask, other=0.0)
    if all_drafts_accepted:
        val = p_val
    else:
        cids = tl.load(
            CandIds + pid * stride_ci_b + cur_prob_row * stride_ci_s + qq,
            mask=qmask,
            other=-1,
        )
        qv = tl.load(
            QRows + pid * stride_q_b + cur_prob_row * stride_q_s + qq,
            mask=qmask,
            other=0.0,
        )
        match = ids[:, None] == cids[None, :]
        q_val = tl.sum(tl.where(match, qv.to(tl.float32)[None, :], 0.0), axis=1)
        q_val = tl.where((q_val >= 0.0) & (q_val <= 1.0), q_val, 0.0)
        residual_scale = 1.0
        diff = residual_scale * p_val - q_val
        val = tl.where(diff > 0.0, diff, 0.0)
    tl.store(Zero + pid * stride_z + ids, val, mask=kmask)


@triton.jit
def _tile_sum_kernel(
    Zero,
    TileSums,
    stride_z,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    NUM_TILES: tl.constexpr,
):
    pid = tl.program_id(0)
    tile = tl.program_id(1)
    v_offsets = tile * BLOCK_V + tl.arange(0, BLOCK_V)
    mask = v_offsets < VOCAB_SIZE
    val = tl.load(Zero + pid * stride_z + v_offsets, mask=mask, other=0.0)
    tl.store(TileSums + pid * NUM_TILES + tile, tl.sum(val))


@triton.jit
def _tile_cdf_kernel(
    Predicts,
    UniformSamplesFinal,
    Zero,
    TileSums,
    Result,
    Arrived,
    Meta,
    TopIds,
    Bonus,
    stride_z,
    NUM_SLOTS: tl.constexpr,
    K: tl.constexpr,
    KP: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    NUM_TILES: tl.constexpr,
):
    pid = tl.program_id(0)
    tile = tl.program_id(1)

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
    v_offsets = v_start + tl.arange(0, BLOCK_V)
    mask = v_offsets < VOCAB_SIZE
    val = tl.load(Zero + pid * stride_z + v_offsets, mask=mask, other=0.0)
    block_cumsum = tl.cumsum(val, axis=0)
    total_cumsum = cum_sum + block_cumsum
    candidates_mask = total_cumsum > target_u
    has_match = tl.max(candidates_mask, axis=0)
    if has_match:
        match_idx = tl.argmax(candidates_mask.to(tl.int32), axis=0)
        tl.atomic_min(Result + pid, v_start + match_idx, sem="relaxed", scope="gpu")
    # Arrival counter (acq_rel RMW, gpu scope): every program's loads of the row and its atomic minimum are ordered
    # before its arrival (CTA barrier, then the release of the RMW); the last arrival acquires every earlier one
    # through the RMW chain, so its read of the minimum sees all of them and its stores restoring the zero row come
    # after every program's loads of it.
    tl.debug_barrier()
    arrived = tl.atomic_add(Arrived + pid, 1, sem="acq_rel", scope="gpu")
    if arrived == NUM_TILES - 1:
        final_token = tl.atomic_add(Result + pid, 0, sem="relaxed", scope="gpu")
        last_idx = tl.load(Meta + pid * 4 + 1)
        tl.store(Predicts + last_idx, final_token)
        # bonus token = predicts[accept_index[pid, accept_len]]: that entry is last_idx, just written
        tl.store(Bonus + pid, final_token.to(tl.int64))
        cur = tl.load(Meta + pid * 4 + 0)
        kk = tl.arange(0, KP)
        kmask = kk < K
        trow = (pid * NUM_SLOTS + cur).to(tl.int64)
        ids = tl.load(TopIds + trow * K + kk, mask=kmask, other=0)
        tl.store(Zero + pid * stride_z + ids, 0.0, mask=kmask)
        tl.atomic_xchg(
            Arrived + pid, 0, sem="relaxed", scope="gpu"
        )  # ready for the next launch (stream order)


# ----------------------------------------------------------------------------------------------------------------
# Host side


_compiled: dict = {}


def _launch(kernel, grid, tensors, scalars, consts, num_warps):
    """``kernel[grid](*tensors, *scalars, **consts)``. The compiled kernel of the first call is kept, keyed by what the
    launch specializes on apart from tensor dtypes and alignment (device, scalar values, constexprs; every tensor must
    be 16-byte aligned, else the call takes the regular path), and later calls launch it directly without
    ``JITFunction.run``'s per-call argument binding (the same scheme as ``spec_sample_split``)."""
    for t in tensors:
        if t.data_ptr() & 15:
            kernel[grid](*tensors, *scalars, **consts, num_warps=num_warps)
            return
    dev = torch.cuda.current_device()
    cvals = tuple(consts.values())
    key = (kernel, dev, tuple(t.dtype for t in tensors), scalars, cvals, num_warps)
    ck = _compiled.get(key)
    if ck is None:
        if not _state.get("in_preload"):
            bed.count(NAME, "jit:serving")
        _compiled[key] = kernel[grid](*tensors, *scalars, **consts, num_warps=num_warps)
        return
    from triton import knobs

    stream = torch._C._cuda_getCurrentRawStream(dev)
    enter = knobs.runtime.launch_enter_hook
    meta = (
        None
        if enter is None
        else ck.launch_metadata(grid, stream, *tensors, *scalars, *cvals)
    )
    g = tuple(grid) + (1,) * (3 - len(grid))
    ck.run(
        g[0],
        g[1],
        g[2],
        stream,
        ck.function,
        ck.packed_metadata,
        meta,
        enter,
        knobs.runtime.launch_exit_hook,
        *tensors,
        *scalars,
        *cvals,
    )


def prep_shard(
    logits: torch.Tensor,
    cap: float,
    temps: torch.Tensor,
    rows_per_req: int,
    cols: int,
    with_argmax: bool = False,
    pool: bool = False,
):
    """(scaled values [rows, cols], bucket maximum keys [rows, nb] (int32 bits), block argmax keys [rows, nblk] or
    None) of one rank's shard. ``pool``: outputs in reused buffers (valid until the next pooled call)."""
    rows = logits.shape[0]
    dev = logits.device
    nb, nblk = triton.cdiv(cols, BUCKET), triton.cdiv(cols, PREP_BLOCK)
    alloc = (
        (lambda n, sh, dt: _pooled(n, sh, dt, dev))
        if pool
        else (lambda n, sh, dt: torch.empty(sh, dtype=dt, device=dev))
    )
    s = alloc("s", (rows, cols), torch.float32)
    bkey = alloc("bkey", (rows, nb), torch.int32)
    amax = alloc("amax_blk", (rows, nblk), torch.int64) if with_argmax else bkey
    if rows:
        _launch(
            _shard_prep_kernel,
            (nblk, rows),
            (logits, temps, s, bkey, amax),
            (cols, logits.stride(0), s.stride(0), nb, nblk, rows_per_req, float(cap)),
            dict(BLOCK_SIZE=PREP_BLOCK, BUCKET=BUCKET, WITH_ARGMAX=bool(with_argmax)),
            8,
        )
    return s, bkey, (amax if with_argmax else None)


def local_records(
    logits: torch.Tensor,
    cap: float,
    temps: torch.Tensor,
    rows_per_req: int,
    k: int,
    col_offset: int,
    cols: int,
    with_argmax: bool = False,
    pool: bool = False,
) -> torch.Tensor:
    """This rank's record per row: k scaled values, k global ids (top-k order), argmax key and id (int32 words; the
    argmax only with ``with_argmax``, else zeros). ``pool``: reused buffers (valid until the next pooled call)."""
    s, bkey, amax = prep_shard(
        logits, cap, temps, rows_per_req, cols, with_argmax, pool
    )
    rows, dev = logits.shape[0], logits.device
    shape = (rows, 2 * k + 2)
    rec = (
        _pooled("rec", shape, torch.int32, dev)
        if pool
        else torch.empty(shape, dtype=torch.int32, device=dev)
    )
    if amax is None:
        # select's argument contract asks for an int64 [rows, blocks] tensor even when it reads no argmax keys
        # (has_amax 0): a dedicated [rows, 1] buffer, never read
        amax = (
            _pooled("amax_unused", (rows, 1), torch.int64, dev)
            if pool
            else torch.empty((rows, 1), dtype=torch.int64, device=dev)
        )
        has_amax = 0
    else:
        has_amax = 1
    module().select(s, bkey, amax, rec, int(col_offset), int(k), has_amax)
    return rec


def merge_records(gathered: torch.Tensor, k: int, pool: bool = False):
    """(top-k values fp32 [rows, k], ids int64 [rows, k], argmax int64 [rows]) from all ranks' records. ``pool``:
    reused buffers (valid until the next pooled call)."""
    world, rows, _ = gathered.shape
    dev = gathered.device
    alloc = (
        (lambda n, sh, dt: _pooled(n, sh, dt, dev))
        if pool
        else (lambda n, sh, dt: torch.empty(sh, dtype=dt, device=dev))
    )
    vals = alloc("vals", (rows, k), torch.float32)
    ids = alloc("ids", (rows, k), torch.int64)
    amax = alloc("amax", (rows,), torch.int64)
    module().merge(gathered, vals, ids, amax, int(k))
    return vals, ids, amax


def _scratch(device: torch.device, bs: int, vocab: int):
    """Per (device, stream): the zero rows (kept zero between calls), arrival counters (kept zero), tile sums, results
    and walk records. Calls on one stream are ordered, so they can share them."""
    num_tiles = triton.cdiv(vocab, BLOCK_V)
    idx = device.index if device.index is not None else torch.cuda.current_device()
    key = (idx, torch._C._cuda_getCurrentRawStream(idx), vocab)
    sc = _state["scratch"].get(key)
    if sc is None or sc["cap"] < bs:
        with _lock:
            cap = max(bs, 0 if sc is None else 2 * sc["cap"])
            sc = {
                "cap": cap,
                "zero": torch.zeros((cap, vocab), dtype=torch.float32, device=device),
                "arrived": torch.zeros((cap,), dtype=torch.int32, device=device),
                "tile_sums": torch.empty(
                    (cap, num_tiles), dtype=torch.float32, device=device
                ),
                "result": torch.empty((cap,), dtype=torch.int32, device=device),
                "meta": torch.empty((cap, 4), dtype=torch.int64, device=device),
            }
            _state["scratch"][key] = sc
    return sc


def sparse_chain_sample(
    *,
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    uniform_samples,
    uniform_samples_final,
    top_probs,
    top_ids,
    cand_ids,
    q_rows,
    vocab: int,
    bonus=None,
):
    """Same writes to predicts / accept_index / accept_token_num as ``chain_speculative_sampling_triton`` (non-block
    verification) on the dense matrices that hold top_probs at top_ids (target) and q_rows at cand_ids (draft); with
    ``bonus`` (int64 [bs]) also the bonus tokens (``predicts[accept_index[i, accept_token_num[i]]]``)."""
    bs, num_slots = candidates.shape
    k = top_ids.shape[1]
    kq = cand_ids.shape[-1]
    dev = candidates.device
    sc = _scratch(dev, bs, vocab)
    zero, arrived, tile_sums, result, meta = (
        sc["zero"],
        sc["arrived"],
        sc["tile_sums"],
        sc["result"],
        sc["meta"],
    )
    if bonus is None:
        bonus = torch.empty((bs,), dtype=torch.int64, device=dev)
    num_tiles = triton.cdiv(vocab, BLOCK_V)
    kp, kqp = triton.next_power_of_2(k), triton.next_power_of_2(kq)
    _launch(
        _walk_kernel,
        (bs,),
        (
            predicts,
            accept_index,
            accept_token_num,
            candidates,
            retrive_index,
            uniform_samples,
            top_probs,
            top_ids,
            cand_ids,
            q_rows,
            zero,
            meta,
            result,
        ),
        (
            candidates.stride(0),
            candidates.stride(1),
            retrive_index.stride(0),
            retrive_index.stride(1),
            uniform_samples.stride(0),
            uniform_samples.stride(1),
            cand_ids.stride(0),
            cand_ids.stride(1),
            q_rows.stride(0),
            q_rows.stride(1),
            zero.stride(0),
        ),
        dict(NUM_SLOTS=num_slots, K=k, KP=kp, KQ=kq, KQP=kqp, VOCAB_SIZE=vocab),
        NUM_WARPS,
    )
    grid = (bs, num_tiles)
    _launch(
        _tile_sum_kernel,
        grid,
        (zero, tile_sums),
        (zero.stride(0),),
        dict(VOCAB_SIZE=vocab, BLOCK_V=BLOCK_V, NUM_TILES=num_tiles),
        NUM_WARPS,
    )
    _launch(
        _tile_cdf_kernel,
        grid,
        (
            predicts,
            uniform_samples_final,
            zero,
            tile_sums,
            result,
            arrived,
            meta,
            top_ids,
            bonus,
        ),
        (zero.stride(0),),
        dict(
            NUM_SLOTS=num_slots,
            K=k,
            KP=kp,
            VOCAB_SIZE=vocab,
            BLOCK_V=BLOCK_V,
            NUM_TILES=num_tiles,
        ),
        NUM_WARPS,
    )
    return bonus


def draw_coins(bs: int, gamma: int, device):
    """The two draws of the stock selector accept path (``_accept_sampling_core``), same shapes, same order."""
    uniform_samples = torch.rand((bs, gamma), dtype=torch.float32, device=device)
    uniform_samples_final = torch.rand((bs,), dtype=torch.float32, device=device)
    return uniform_samples, uniform_samples_final


def accept_from_topk(
    *, candidates, top_probs, top_ids, cand_ids, q_rows, vocab: int, coins=None
):
    """``_selector_sampling_accept`` (pinned revision) from the rows' top-k (probabilities, ids): same RNG draws,
    same buffers, same returned tensors."""
    _get_or_create_chain_verify_buffers = _state.get("chain_buffers")
    if _get_or_create_chain_verify_buffers is None:
        from sglang.srt.speculative.dflash_utils import (
            _get_or_create_chain_verify_buffers,
        )

        _state["chain_buffers"] = _get_or_create_chain_verify_buffers
    bs, block = candidates.shape
    device = candidates.device
    (retrieve_index, _rnt, _rns, predicts, accept_index, accept_token_num) = (
        _get_or_create_chain_verify_buffers(bs=bs, draft_token_num=block, device=device)
    )
    uniform_samples, uniform_samples_final = (
        coins if coins is not None else draw_coins(bs, block - 1, device)
    )
    bonus = sparse_chain_sample(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrieve_index,
        uniform_samples=uniform_samples,
        uniform_samples_final=uniform_samples_final,
        top_probs=top_probs,
        top_ids=top_ids,
        cand_ids=cand_ids,
        q_rows=q_rows,
        vocab=vocab,
    )
    return accept_token_num, bonus, (uniform_samples, uniform_samples_final)


# ----------------------------------------------------------------------------------------------------------------
# Preload (first verify step, outside graph capture)


def _default_top_k(worker) -> int | None:
    """The checkpoint's default ``top_k`` (the server's sampling defaults; requests that set none use it) when it is a
    value the top-k path takes, else None."""
    try:
        k = int(
            worker.model_runner.model_config.get_default_sampling_params().get(
                "top_k", -1
            )
        )
    except Exception:  # noqa: BLE001 - no defaults on this worker: nothing k-specific is preloaded
        return None
    return k if 2 <= k <= MAX_K else None


def _selector_top_k(worker) -> int | None:
    try:
        kq = int(worker.selector.top_k)
    except Exception:  # noqa: BLE001 - no selector: the sampling passes are not preloaded
        return None
    return kq if 1 <= kq <= MAX_K else None


def _warm_sampler(device, block: int, k: int, kq: int, vocab: int) -> None:
    """The chain walk and final-token passes once, at a top-k step's specializations (the strides of the worker's
    chain buffers and of the selector's rows, which do not depend on the batch size), at batch size 1 on fixed
    inputs. Run on the caller's side stream; its scratch entry is dropped afterwards."""
    gamma = block - 1
    ar = torch.arange(block, dtype=torch.int64, device=device)
    sparse_chain_sample(
        predicts=torch.zeros((block,), dtype=torch.int32, device=device),
        accept_index=torch.full((1, block), -1, dtype=torch.int32, device=device),
        accept_token_num=torch.zeros((1,), dtype=torch.int32, device=device),
        candidates=(ar % vocab).view(1, block),
        retrive_index=ar.view(1, block).clone(),
        uniform_samples=torch.full((1, gamma), 0.5, dtype=torch.float32, device=device),
        uniform_samples_final=torch.full((1,), 0.5, dtype=torch.float32, device=device),
        top_probs=torch.full((block, k), 1.0 / k, dtype=torch.float32, device=device),
        top_ids=torch.arange(k, dtype=torch.int64, device=device).repeat(block, 1),
        cand_ids=torch.arange(kq, dtype=torch.int64, device=device).repeat(1, gamma, 1),
        q_rows=torch.full((1, gamma, kq), 1.0 / kq, dtype=torch.float32, device=device),
        vocab=vocab,
    )
    idx = device.index if device.index is not None else torch.cuda.current_device()
    _state["scratch"].pop((idx, torch._C._cuda_getCurrentRawStream(idx), vocab), None)


def _preload(worker, lp, logits: torch.Tensor, bs: int) -> None:
    """Once, on the first verify step (see the module docstring): module, self-check and kernel compiles. Any failure
    leaves the lazy path as it was (the first top-k step builds what is missing)."""
    _state["preload_done"] = True
    dev = logits.device
    info: dict = {}
    try:
        if (
            logits.dtype != torch.float32
            or logits.dim() != 2
            or logits.stride(1) != 1
            or logits.stride(0) % 4
            or logits.data_ptr() % 16
        ):
            bed.report(NAME + ":preload", False, reason="logits layout")
            return
        get_tp_group = _state["tp_group"]
        if get_tp_group is None:
            import sglang.srt.distributed.parallel_state as parallel_state

            get_tp_group = _state["tp_group"] = lambda: parallel_state.get_tp_group()
        group = get_tp_group()
        world, rank = group.world_size, group.rank_in_group
        rows, width = logits.shape
        vocab = int(lp.vocab_size)
        col_offset = rank * width
        cols = max(0, min(width, vocab - col_offset))
        block = rows // max(bs, 1)
        cap = lp.final_logit_softcapping
        if block * bs != rows or block < 2:
            reason = "rows"
        elif (
            not (MIN_COLS <= cols <= MAX_COLS)
            or world > MAX_WORLD
            or min(width * world, vocab) != vocab
        ):
            reason = "shard shape"
        elif not cap:
            reason = "no soft cap"
        else:
            reason = None
        if reason is not None:
            bed.report(NAME + ":preload", False, reason=reason)
            return
        k, kq = _default_top_k(worker), _selector_top_k(worker)
        info = dict(k=k, kq=kq, block=block, cols=cols)
        _state["in_preload"] = True
        module()
        info["check"] = _checked(dev, vocab) is None
        side = torch.cuda.Stream(dev)
        side.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(side):
            temps = torch.ones((bs,), dtype=torch.float32, device=dev)
            if k is None:
                prep_shard(logits, float(cap), temps, block, cols)
            else:
                rec = local_records(
                    logits,
                    float(cap),
                    temps,
                    block,
                    k,
                    col_offset,
                    cols,
                )
                merge_records(rec.unsqueeze(0), k)
                if kq is not None:
                    _warm_sampler(dev, block, k, kq, vocab)
        side.synchronize()
        bed.report(NAME + ":preload", True, **info)
    except Exception as e:  # noqa: BLE001 - the lazy path stays as it was
        logger.warning(
            "%s: preload failed (%s); the first top-k step builds what is missing",
            NAME,
            e,
        )
        bed.report(NAME + ":preload", False, reason=type(e).__name__, **info)
    finally:
        _state["in_preload"] = False


# ----------------------------------------------------------------------------------------------------------------
# Hooks (per-thread state: one verify step at a time per scheduler thread)


def _simulating() -> bool:
    import sglang.srt.speculative.dflash_worker_v2 as w

    return w.SIMULATE_ACC_LEN > 0


def eligible(worker, batch, sinfo) -> tuple[str | None, int]:
    """(None, k) when the verify step of this batch can take the top-k path; decided from host-side state only,
    which is the same on every rank."""
    if sinfo is None or sinfo.is_all_greedy:
        return "greedy", 0
    if worker._selector_sample is None:
        return "not the selector path", 0
    if getattr(worker, "_block_verification", False):
        return "block verification", 0
    if batch.has_grammar or batch.return_logprob:
        return "grammar or logprobs", 0
    penalizer = sinfo.penalizer_orchestrator
    if (
        sinfo.has_custom_logit_processor
        or getattr(sinfo, "acc_linear_penalties", None) is not None
        or (penalizer is not None and penalizer.is_required)
        or getattr(sinfo, "grammar_mask", None) is not None
        or getattr(sinfo, "logit_bias", None) is not None
        or sinfo.sampling_mask_batch_indices is not None
    ):
        return "logit adjustments", 0
    if not sinfo.need_top_k_sampling or sinfo.need_top_p_sampling:
        return "top-k / top-p", 0
    # The draft input's max_top_k / uniform_top_k_value (what the stock path passes on) are computed from the requests'
    # sampling params in prepare_for_decode; at this point batch.spec_info already holds the verify input, so the same
    # values are taken from the requests here.
    top_ks = {int(r.sampling_params.top_k) for r in batch.reqs}
    k = top_ks.pop() if len(top_ks) == 1 else None
    if k is None or not (2 <= k <= MAX_K):
        return "top-k not uniform or out of range", 0
    t = sinfo.temperatures
    if t.dtype != torch.float32 or t.numel() != len(batch.reqs):
        return "temperatures", 0
    return None, int(k)


def _h_skip(original, worker, batch, sampling_info):
    res = original(worker, batch, sampling_info)
    _tls.pending = None
    _tls.preload = None if _state.get("preload_done") else (worker, len(batch.reqs))
    if not res:
        why, k = eligible(worker, batch, sampling_info)
        if why is None:
            _tls.pending = (sampling_info, len(batch.reqs), k)
        else:
            bed.count(NAME, "stock:" + why)
    return res


def _h_gather(original, lp, logits):
    pending = getattr(_tls, "pending", None)
    pre = getattr(_tls, "preload", None)
    _tls.pending = None
    _tls.active = None
    _tls.preload = None
    if (
        pre is not None
        and not _state.get("preload_done")
        and not torch.cuda.is_current_stream_capturing()
    ):
        if pending is not None:
            bed.count(NAME, "preload:on a top-k step")
        _preload(pre[0], lp, logits, pre[1])
    if pending is None:
        return original(lp, logits)
    sinfo, bs, k = pending
    get_tp_group = _state["tp_group"]
    if get_tp_group is None:
        import sglang.srt.distributed.parallel_state as parallel_state

        get_tp_group = _state["tp_group"] = lambda: parallel_state.get_tp_group()
    group = get_tp_group()
    world, rank = group.world_size, group.rank_in_group
    reason = None
    if (
        logits.dtype != torch.float32
        or logits.dim() != 2
        or logits.stride(1) != 1
        or logits.stride(0) % 4
    ):
        reason = "logits layout"
    else:
        rows, width = logits.shape
        col_offset = rank * width
        cols = max(0, min(width, lp.vocab_size - col_offset))
        block = rows // max(bs, 1)
        if torch.cuda.is_current_stream_capturing():
            reason = "graph capture"
        elif block * bs != rows or block < 2:
            reason = "rows"
        elif (
            not (MIN_COLS <= cols <= MAX_COLS)
            or world > MAX_WORLD
            or min(width * world, lp.vocab_size) != lp.vocab_size
            or logits.data_ptr() % 16
        ):
            reason = "shard shape"
        elif not lp.final_logit_softcapping:
            reason = "no soft cap"
        elif _flags()["sampling_graph"]:
            reason = "sampling_graph enabled"
        elif topk_nondeterministic(rows, k, logits.device) and not _flags()["lowtie"]:
            reason = "topk not deterministic"
        elif _checked(logits.device, lp.vocab_size) is not None:
            reason = "topk self-check"
    if reason is not None:
        bed.count(NAME, "stock:" + reason)
        return original(lp, logits)
    if not _state.get("preload_done"):
        bed.count(NAME, "preload:missed")
    temps = sinfo.temperatures.reshape(-1)
    rec = local_records(
        logits,
        float(lp.final_logit_softcapping),
        temps,
        block,
        k,
        col_offset,
        cols,
        pool=True,
    )
    if world > 1:
        gathered = _pooled(
            "gathered", (world, rows, rec.shape[1]), torch.int32, rec.device
        )
        group.all_gather_into_tensor(gathered.view(-1, rec.shape[1]), rec)
    else:
        gathered = rec.unsqueeze(0)
    vals, ids, amax = merge_records(gathered, k, pool=True)
    probs = torch.softmax(vals, dim=-1)
    a = {
        "shard": logits,
        "probs": probs,
        "ids": ids,
        "amax": amax,
        "bs": bs,
        "block": block,
        "vocab": int(lp.vocab_size),
        "lp": lp,
        "out": logits,
    }
    if _simulating():
        # Synthetic-acceptance benchmark mode: after sampling, the worker takes torch.argmax over the full verify rows
        # and forces the acceptance. That scaffolding stays exactly stock: the full rows come from the stock gather
        # (the wrapped original) and are handed on as the step's logits; target_predict and the forcing are never
        # touched. The sampling itself runs the same top-k path as with real acceptance.
        a["out"] = original(lp, logits)
    _tls.active = a
    bed.count(
        NAME,
        f"topk:rows={rows}"
        + (":lowtie" if topk_nondeterministic(rows, k, logits.device) else ""),
    )
    return a["out"]


def _h_argmax(original, lp, logits):
    """A greedy verify step (the decision hook kept the gather off): the preload, if this is the first verify step;
    the result is the original's, unchanged."""
    pre = getattr(_tls, "preload", None)
    _tls.preload = None
    if (
        pre is not None
        and not _state.get("preload_done")
        and not torch.cuda.is_current_stream_capturing()
    ):
        _preload(pre[0], lp, logits, pre[1])
    return original(lp, logits)


def _full_logits(a):
    """Full rows for a consumer that needs them after all (the stock gather of the shard)."""
    if a["out"] is not a["shard"]:
        return a["out"]
    from sglang.srt.layers.logits_processor import LogitsProcessor

    return LogitsProcessor._gather_softcap_fused(
        a["lp"], a["shard"], None, use_buffer=False
    )


def _h_accept(original, worker, **kw):
    a = getattr(_tls, "active", None)
    if a is None or kw.get("next_token_logits") is not a["out"]:
        _tls.active = None
        return original(worker, **kw)
    try:
        if (
            worker._selector_sample is None
        ):  # not the selector branch after all: it reads full rows
            kw["next_token_logits"] = _full_logits(a)
            _tls.active = None
            bed.count(NAME, "fallback:full rows")
        return original(worker, **kw)
    finally:
        _tls.active = None


def _sample_layout_ok(cand, cids, q, a) -> bool:
    bs, block = a["bs"], a["block"]
    return (
        isinstance(cand, torch.Tensor)
        and cand.dtype == torch.int64
        and tuple(cand.shape) == (bs, block)
        and isinstance(cids, torch.Tensor)
        and cids.dtype == torch.int64
        and cids.dim() == 3
        and tuple(cids.shape[:2]) == (bs, block - 1)
        and cids.stride(-1) == 1
        and 1 <= cids.shape[-1] <= MAX_K
        and isinstance(q, torch.Tensor)
        and q.dtype in (torch.float32, torch.bfloat16, torch.float16)
        and tuple(q.shape) == tuple(cids.shape)
        and q.stride(-1) == 1
    )


def _h_sample(original, worker, **kw):
    a = getattr(_tls, "active", None)
    logits = kw.get("next_token_logits")
    if a is None or logits is not a["out"]:
        return original(worker, **kw)
    if not _sample_layout_ok(
        kw.get("candidates"), kw.get("candidate_ids"), kw.get("q_rows"), a
    ):
        bed.count(NAME, "fallback:full rows")
        kw["next_token_logits"] = _full_logits(a)
        return original(worker, **kw)

    cand = kw["candidates"]
    bs, block = cand.shape
    coins = draw_coins(bs, block - 1, cand.device)
    accept_len, bonus, _ = accept_from_topk(
        candidates=cand,
        top_probs=a["probs"],
        top_ids=a["ids"],
        cand_ids=kw["candidate_ids"],
        q_rows=kw["q_rows"],
        vocab=a["vocab"],
        coins=coins,
    )
    bed.count(NAME, f"sample:bs={bs}")
    return accept_len, bonus


def register() -> bool:
    if not enabled():
        return False
    ok, why = pins_match()
    if not ok:
        bed.report(NAME, False, reason=why)
        return False
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(T_SKIP, _h_skip, HookType.AROUND)
    HookRegistry.register(T_GATHER, _h_gather, HookType.AROUND)
    HookRegistry.register(T_ACCEPT, _h_accept, HookType.AROUND)
    HookRegistry.register(T_SAMPLE, _h_sample, HookType.AROUND)
    HookRegistry.register(T_ARGMAX, _h_argmax, HookType.AROUND)
    bed.report(
        NAME,
        True,
        lowtie=bed.enabled(NAME + "_lowtie"),
    )
    return True

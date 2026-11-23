# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``draft_boundary`` (env ``MUSE_VR_DRAFT_BOUNDARY=1``, default off).

Each DFlash draft decoder layer ends both of its sublayers the same way: the row-parallel projection (attention
``o_proj``, MLP ``down_proj``) is all-reduced (custom all-reduce, one-shot push), the output side of the layer's
grouped two-tap block convolution is applied (``DFlashGroupedConv.finish``, a compiled elementwise kernel), and the
residual add + RMSNorm of the next sublayer runs (FlashInfer fused add + RMSNorm; after the MLP of the last layer this
is the model's final norm). That is three kernels per boundary, ten boundaries per draft step.

This component runs the draft model's forward (an AROUND hook on ``DFlashDraftModel.forward``) with the projections'
all-reduce deferred, and replaces each triple with one kernel (``csrc/draft_ar_norm.cuh``, ``ops/draft_ar_norm``)
that performs the same one-shot push exchange (same plane, protocol and rank-order fp32 sum), the same convolution
arithmetic and residual rounding, and the norm. Every other part of the forward (input embedding norm, the input-side
convolution, attention, MLP projections) is SGLang's own code, called in the same order.

Numerics: the all-reduced values, the convolution output and the updated residual are bit-identical to the stock
sequence; the norm's sum of squares is accumulated in a different order (per thread, warp, then cluster), so the
normalized output can differ in the last bf16 place (tests/test_draft_ar_norm.py: about 1e-5 of the elements, at most
one bf16 unit, error against fp64 equal to stock's). The draft's dtype and work are unchanged.

``MUSE_VR_DRAFT_GEMV=1`` (default off; only with this component) also replaces, for row blocks of at most
``MUSE_VR_DRAFT_GEMV_MAX_M`` rows (default 16; tunable, tests/test_draft_gemv.py --bench), each block convolution's
coefficient projection and input-side convolution (nn.Linear + compiled convolution) by one GEMV with the convolution
fused (``ops/draft_gemv.conv_in``), and the MLP's merged gate/up projection + SiLU-and-multiply by one GEMV with the
activation fused (``ops/draft_gemv.swiglu``). Their tactics can be overridden with ``MUSE_VR_DRAFT_GEMV_TACTIC_CONV`` /
``MUSE_VR_DRAFT_GEMV_TACTIC_SWIGLU`` (default unset = the built-in defaults; tests/test_draft_gemv.py --bench), or for one token
tile only with the suffix ``_T<tile>`` (for example ``MUSE_VR_DRAFT_GEMV_TACTIC_CONV_T8``, which takes precedence). Rounding
points are the stock ones (bf16 operands, fp32 accumulation, bf16 coefficients / projections before the convolution /
activation); the projections' fp32 summation order differs
(tests/test_draft_gemv.py: error against fp64 equal to stock's). The attention's qkv and output projections keep
SGLang's dispatch (tests/test_draft_gemv.py also times the plain GEMV against the kernels the server selects for them).

``MUSE_VR_DRAFT_PUSH_EPI=1`` (default off; with ``MUSE_VR_DRAFT_GEMV=1`` and ``MUSE_VR_DRAFT_QK=1``) computes each layer's
attention output projection and MLP down projection with ``ops/draft_gemv.push`` (tactics ``MUSE_VR_DRAFT_PUSH_TACTIC_O`` /
``MUSE_VR_DRAFT_PUSH_TACTIC_DOWN``): the GEMV pushes its output vectors into the peers' one-shot push slots and the boundary
kernel runs consumer-only (same vectors in the same slots as the boundary would push; the projection's fp32 summation order is
the GEMV's, as for the other draft GEMVs; tests/dist_draft_push.py). Requires the boundary's whole-cluster layout (requests x
cluster CTAs within the plane's block counters). ``MUSE_VR_DRAFT_PUSH_SITES`` (default ``o,down``) limits it to one of the two
projections; the other boundary then keeps the stock projection and the boundary kernel's own exchange.
``MUSE_VR_DRAFT_PUSH_MC=1`` (default off; with ``MUSE_VR_DRAFT_PUSH_EPI=1`` and a push plane with a multicast address): the push
GEMV stores each output vector once through the plane's multicast address instead of once per peer (the per-vector store count
then does not grow with the group size, as with ``MUSE_VR_DRAFT_ARN_MC=1`` for the boundary's own push), and the consumer-only
boundary polls and scrubs this rank's own slot as that form does. Same vectors in the same slots: outputs bit for bit
(tests/dist_draft_push.py --mc).
``MUSE_VR_DRAFT_QK_STORE=1`` (default off; with ``MUSE_VR_DRAFT_QK=1``): the norm + RoPE kernel also writes the rotated key
heads and the value heads into the layer's KV cache rows (``ops/draft_qk.qk_norm_rope_qout_store``) and the attention backend
is called with ``save_kv_cache=False``, when its write would be a plain copy into a bf16 token-major pool (FlashAttention
backend, plain MHA pool, no scales); same rows, same values (tests/test_draft_qk.py).
``MUSE_VR_DRAFT_QKV_EPI=1`` (default off; with ``MUSE_VR_DRAFT_QK=1``): for blocks of at most 16 rows, each layer's QKV
projection runs as ``ops/draft_gemv.qkv_qk`` (tactic ``MUSE_VR_DRAFT_GEMV_TACTIC_QK``, 128-row head tiles): its epilogue applies
the QK norm + RoPE with the norm + RoPE kernel's exact float operations and writes the query heads and the K / V cache rows
(same eligibility as the KV store above); the QKV tensor is not materialized and the attention backend is called without
K / V and with ``save_kv_cache=False``. The projection's fp32 summation order is the GEMV's (tests/test_draft_qkv_epi.py:
bit-identical to the plain GEMV with the same tactic followed by the norm + RoPE kernel).
``MUSE_VR_DRAFT_GEMV_L2PF_{CONV,SWIGLU,PUSH}=<steps>`` (default 0 = off; tunable): the draft GEMVs' L2 lookahead
(``ops/draft_gemv``): each CTA requests its next weight steps beyond the ring into L2 before waiting on its predecessor, so the
weights stream while the boundary or attention runs. Same outputs, bit for bit.
``MUSE_VR_DRAFT_GEMV_CONV_PF=1`` / ``MUSE_VR_DRAFT_GEMV_QK_PF=1`` (default off; with ``MUSE_VR_DRAFT_GEMV=1`` / ``MUSE_VR_DRAFT_QKV_EPI=1``):
the conv GEMV's epilogue CTA reads the activation vectors its convolution needs, and the QKV-epilogue GEMV reads each token's
RoPE table entries and key-cache row, right after the grid dependency wait (before the MMA loop) instead of at the epilogue.
Same loads of the same values, outputs bit for bit (tests/test_draft_gemv_readahead.py, tests/test_draft_qkv_epi.py).
``MUSE_VR_DRAFT_QK=1`` (default off; only with this component) runs each layer's attention (``DFlashAttention.forward``,
pinned revision, table QK-norm + RoPE path) with ``ops/draft_qk.qk_norm_rope_qout`` in place of
``table_qk_norm_rope_``: the same per-head arithmetic (bit-identical QKV, tests/test_draft_qk.py), and the rotated query
heads are also written to a contiguous tensor that is handed to the attention layer, so the backend's copy of the
strided query view into a contiguous tensor is a no-op. Engaged when every layer uses that path with bf16 projections,
no value scale and no attention sinks.

``MUSE_VR_DRAFT_QK_WIDE=1`` (default off; only with this component) uses the same query/key call where this
component steps aside (for example above ``MUSE_VR_DRAFT_BOUNDARY_MAX_ROWS`` rows): SGLang's draft forward runs
unchanged except that each layer's attention is ``DFlashAttention.forward`` (pinned revision) with
``ops/draft_qk.qk_norm_rope_qout`` in place of ``table_qk_norm_rope_`` and the contiguous query handed to the
attention layer; the output projection keeps its all-reduce. Same eligibility as ``MUSE_VR_DRAFT_QK``; outputs are
bit-identical to the stock forward (tests/test_draft_qk.py ``--config``).

``MUSE_VR_DRAFT_L0_TABLE=1`` (opt-in; after TensorRT-LLM's
kernels/tt_table/drafter_l0.py): layer 0's attention input side from a per-token table. The draft block of one request is
[bonus, mask x 7] (dflash_worker_v2: the worker's block preparation), so layer 0's input norm (per row), the attention
convolution's coefficient projection (per row) and its two-tap input-side convolution (row r reads rows r and r - 1 of the
block) make QKV rows 0 / 1 and output-side coefficient row 0 functions of the bonus id alone, and the other rows constant.
The table (built when the DFLASH worker is constructed, before the KV pools are sized and before any graph capture) holds,
per id, QKV rows 0 / 1 and coefficient row 0 as the served calls compute them for the block [id, mask x 7]: the worker's
noise embedding, ``input_layernorm``, ``_prepare`` (``ops/draft_gemv.conv_in``) and the QKV GEMV in its plain form with the
served QK tactic (``ops/draft_gemv.plain``): the QKV-epilogue GEMV is bit-identical to that GEMV followed by
``ops/draft_qk.qk_norm_rope_qout_store`` (see ``MUSE_VR_DRAFT_QKV_EPI``). On a forward of exactly one block whose layer 0
would take the QKV epilogue, layer 0 runs: one gather (the bonus entry and the constant rows), the norm + RoPE + K / V cache
write kernel, the same attention call and the same boundary; its input norm, conv_in and QKV GEMV do not run. Checked at
install (random blocks at random positions, eagerly and over CUDA-graph replays: query heads, K / V rows and coefficients
bit for bit against the served calls); any difference leaves layer 0 served. Memory: (vocab + 1) x (2 x QKV width +
2 x groups) bf16 per rank.

Engaged per forward only for at most ``MUSE_VR_DRAFT_BOUNDARY_MAX_ROWS`` rows (default 64; tunable,
tests/dist_draft_boundary.py --bench) and only when every layer has both block convolutions and plain (bf16,
unfused) residual RMSNorms, the row count is a whole number of 8-token request blocks, the projections' stock
all-reduce would be the custom all-reduce's one-shot push on a 2/4/8-rank group, no sequence-parallel or reduce
override is active, the push slot holds the rows and the grid fits the plane's block counters. The decision depends only on facts every rank shares,
so all ranks take the same path. Kernels are built and loaded on the first eager call (warm-up before graph capture);
a capture that finds them unloaded runs the stock forward.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
import sys
import time

import torch

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/models/dflash.py": ("DFlashAttention", "_is_npu"),
    "srt/batch_invariant_ops/__init__.py": ("is_batch_invariant_mode_enabled",),
    "srt/layers/layernorm.py": ("RMSNorm",),
    "srt/layers/moe/utils.py": ("should_skip_mlp_all_reduce",),
    "srt/runtime_context.py": ("get_parallel", "get_forward"),
    "srt/distributed/*.py": ("get_tp_group",),
    "srt/distributed/device_communicators/custom_all_reduce_v2.py": (
        "CustomAllReduceV2",
    ),
    "srt/layers/quantization/unquant.py": ("UnquantizedLinearMethod",),
    "srt/layers/logits_processor.py": ("LogitsProcessorOutput",),
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
    "srt/layers/attention/flashattention_backend.py": ("FlashAttentionBackend",),
    "srt/layers/cp/utils.py": ("is_cp_active",),
    "srt/mem_cache/memory_pool.py": (
        "KVWriteLoc",
        "MHATokenToKVPool",
        "maybe_detect_oob",
        "unwrap_write_loc",
    ),
    "srt/model_executor/forward_context.py": (
        "get_attn_backend",
        "has_forward_context",
    ),
    "srt/environ.py": ("envs",),
}


logger = logging.getLogger(__name__)

NAME = "draft_boundary"  # env MUSE_VR_DRAFT_BOUNDARY
TARGET = "sglang.srt.models.dflash.DFlashDraftModel.forward"
BLOCK = 8

# sha256 of the SGLang sources (relative to the directory holding the ``sglang`` package) whose behaviour the fused
# forward reproduces: the draft model (forward, layers, convolution), the row-parallel linear and its all-reduce
# dispatch, the RMSNorm module, the custom all-reduce (plane, protocol, kernel), the convolution's compiled source and
# the draft QK-norm + RoPE kernel.
PINS = {
    "sglang/srt/models/dflash.py": (
        "59ae9eab87fd3416828604018c36ad6a57be07c4ff6d17f699bb9c0be8c719ff",
        "b7b541368cbbb072a679d08ba5bf0f952f4a43f02b6c078562b2ea7a9ddfd01b",
    ),
    "sglang/srt/layers/linear.py": "3ce31df0a8aab09ed8afa0ada8c0ce30ca207a748c02c3073d7f53648b9b7c3c",
    "sglang/srt/distributed/parallel_state.py": (
        "49f4ecb8627b469d1e6972ba6eba0c54412ea5eec756da587b3ad375f607a5c7",
        "b6c0cd98a81e884c7c4677bd783b3b0cb4eb1438c93540a520296595e8a2ff2d",
    ),
    "sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py": (
        "277da88e2837e585f3ebdf4122906d29ea7a96cc30b981f7a028f6f63b6cf001",
        "f7b23a55019734be7b2f1958e20cc1f78a31251a035f2d75885911aa4a46e6b5",
    ),
    "sglang/kernels/jit/csrc/distributed/custom_all_reduce.cuh": (
        "ec14bbf6b02c481daad053dce6246d684a957cc66aa58a464a15ef117a094258",
        "693fc6e5b73478bab41a6ea93c1636b180c0e2ddb733794d80f370efa9427c9a",
    ),
    "sglang/kernels/jit/include/sgl_kernel/distributed/communicator.cuh": "30d5c75e86e86da7ce99cd6eab85212fdbc71f83a8e3c715c4e9f2c6c1d8986f",
    "sglang/srt/layers/layernorm.py": (
        "98072d836e0c77312058521ee1015c80177b2c370da7408fd066eec61015ff21",
        "253ff80cef7def9af6caa4b4d700c1aaa7d133cf7c26f50da7d855a6a2ea0c21",
    ),
    "sglang/srt/layers/moe/utils.py": (
        "dd4bc85f40ee72622631b8dee52ed8a6368bf4dc257a4a78228901d283a63dd0",
        "a0c0cc2eaa112e3b3e79db07814735f99c099ab9d80c6fd5cfa10f8890d26ce6",
    ),
    "sglang/srt/speculative/dflash_utils.py": (
        "6e0423578bade4a8c58b921430df678840c6fa2de6ea411fe58eb77c98d5a820",
        "35e885f930e95ed82a1f47a0dfa2dab583225d170c6cabe54f6c7bb4e3abbf79",
    ),
    "sglang/srt/model_executor/forward_context.py": "63900e43eaf683bcc50991e56dac6d194566b9fcef2ed452ff8e5818e503d2d0",
}

_state: dict = {"loaded": set()}


def enabled() -> bool:
    return bed.enabled(NAME)


def pins_match() -> tuple[bool, str]:
    for rel, want in PINS.items():
        if not bed.pinned(rel, want):
            return False, f"{rel} differs from the pinned revision"
    return True, "ok"


@contextlib.contextmanager
def _deferred_reduce(linear):
    """Run ``linear`` (a RowParallelLinear) without its all-reduce for the duration of the block."""
    own = linear.__dict__.get("forward")
    linear.forward = functools.partial(linear.forward, skip_all_reduce=True)
    try:
        yield
    finally:
        if own is None:
            del linear.forward
        else:
            linear.forward = own


def _plain_residual_norm(norm) -> bool:
    from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
    from sglang.srt.layers.layernorm import RMSNorm

    return (
        type(norm) is RMSNorm
        and norm.has_weight
        and not norm.cast_x_before_out_mul
        and not getattr(norm, "fp32_residual", False)
        and norm.variance_size_override is None
        and getattr(norm, "_quant_linear", None) is None
        and norm.weight.dtype == torch.bfloat16
        and norm.weight.is_contiguous()
        and not is_batch_invariant_mode_enabled()
    )


def _reduce_group(linear):
    """The group RowParallelLinear.forward would all-reduce ``linear``'s output over (pinned revision), or None when
    it would not use a plain group all-reduce."""
    from sglang.srt.layers.moe.utils import should_skip_mlp_all_reduce
    from sglang.srt.runtime_context import get_parallel

    if (
        linear.use_decode_attn_tp
        or should_skip_mlp_all_reduce()
        or not linear.reduce_results
    ):
        return None
    if linear.use_dp_attention_reduce:
        return get_parallel().attn_tp_group
    if linear.parallel_group == "tp":
        return get_parallel().tp_group
    from sglang.srt.distributed import get_tp_group

    return get_tp_group()


def _model_eligible(model, hidden: torch.Tensor):
    """(reason, context) for one draft forward."""
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )
    from sglang.srt.runtime_context import get_forward

    from .ar_attn_boundary import _stock_ar_is_one_shot_push
    from .ops import draft_ar_norm as op

    m, hdim = hidden.shape
    if m == 0 or m % BLOCK:
        return "rows not whole request blocks", None
    if m > bed.env_int("MUSE_VR_DRAFT_BOUNDARY_MAX_ROWS", 64):
        return "rows above MUSE_VR_DRAFT_BOUNDARY_MAX_ROWS", None
    if hidden.dtype != torch.bfloat16 or not hidden.is_cuda:
        return "hidden dtype", None
    if get_forward().sp_active:
        return "sequence parallel", None
    layers = list(model.layers)
    norms = (
        [ly.input_layernorm for ly in layers]
        + [ly.post_attention_layernorm for ly in layers]
        + [model.norm]
    )
    if not all(_plain_residual_norm(n) for n in norms):
        return "norm configuration", None
    group = None
    for ly in layers:
        if ly.attention_conv is None or ly.mlp_conv is None:
            return "no block convolution", None
        for conv in (ly.attention_conv, ly.mlp_conv):
            if (
                conv.taps != 2
                or conv.group_size != 16
                or conv.block_size != BLOCK
                or conv.base_kernel.dtype != torch.bfloat16
            ):
                return "convolution shape", None
        for lin in (ly.self_attn.o_proj, ly.mlp.down_proj):
            g = _reduce_group(lin)
            if g is None or (group is not None and g is not group):
                return "reduce configuration", None
            group = g
    comm = getattr(group, "ca_comm", None)
    if (
        group.world_size not in (2, 4, 8)
        or not isinstance(comm, CustomAllReduceV2)
        or comm.disabled
    ):
        return "no custom all-reduce", None
    probe = torch.empty((m, hdim), dtype=torch.bfloat16, device=hidden.device)
    if not _stock_ar_is_one_shot_push(group, probe):
        return "stock all-reduce not one-shot push", None
    cluster = op.cluster_for(m)
    if (
        m * hdim * 2 > int(comm.max_push_size)
        or int(comm.config.num_push_blocks) < cluster
    ):
        return "rows above push plane capacity", None
    split = op.split_from_env() if not op.poll_all_from_env() else 0
    if split and m * hdim * 2 + (m // op.BLOCK) * (split - 1) * hdim * 2 > int(
        comm.max_push_size
    ):
        split = 0  # the previous-row copies would not fit the slot: the plain kernel (as op.run decides the same way)
    key = (
        hdim,
        group.world_size,
        op.use_pdl(),
        cluster,
        op.poll_all_from_env(),
        op.early_trigger_from_env(),
        op.mc_push_from_env(),
        split,
        op.mbarx_from_env(),
        hidden.device.index,
    )
    if torch.cuda.is_current_stream_capturing():
        if key not in _state["loaded"]:
            return "kernel not loaded before graph capture", None
    else:
        op.module(
            hdim,
            group.world_size,
            op.use_pdl(),
            cluster,
            op.poll_all_from_env(),
            op.early_trigger_from_env(),
            False,
            op.mc_push_from_env(),
            op.mbarx_from_env(),
        )
        if split:
            op.split_module(
                hdim,
                group.world_size,
                op.use_pdl(),
                cluster,
                split,
                op.early_trigger_from_env(),
                op.mc_push_from_env(),
            )
        _state["loaded"].add(key)
    return None, comm


def _coef_view(kern: torch.Tensor):
    """(base tensor, element offset, row stride) of the output-side coefficient view ``prepare`` returns."""
    base = kern._base if kern._base is not None else kern
    if (
        not base.is_contiguous()
        or kern.dim() != 3
        or kern.stride(2) != 1
        or kern.stride(1) != kern.shape[2]
    ):
        return None
    return base, kern.storage_offset() - base.storage_offset(), kern.stride(0)


def _boundary(comm, partial, coef, conv, residual, norm, consumer_only=False):
    """``coef``: (base tensor, element offset of the output-side coefficients, row stride). ``consumer_only``: the partial's
    vectors were pushed by the producing GEMV (``ops/draft_gemv.push``)."""
    from .ops import draft_ar_norm as op
    from .ops import draft_gemv

    base, off, stride = coef
    return op.run(
        comm,
        partial,
        base,
        off,
        stride,
        conv.base_kernel[1],
        residual,
        norm.weight,
        norm.variance_epsilon,
        consumer_only=consumer_only,
        producer_mc=consumer_only and draft_gemv.push_mc(),
    )


PUSH_TACTIC_O = "r64k128s8w4c1"  # tunable (tests/dist_draft_push.py --bench, tests/test_draft_gemv.py --bench)
PUSH_TACTIC_DOWN = "r64k128s8w4c1"  # measured on par with the stock down GEMM at 8 rows (r64k256s4w4c4 was ~1 us slower)


def _push_tactics():
    return (
        os.environ.get("MUSE_VR_DRAFT_PUSH_TACTIC_O") or PUSH_TACTIC_O,
        os.environ.get("MUSE_VR_DRAFT_PUSH_TACTIC_DOWN") or PUSH_TACTIC_DOWN,
    )


def _push_sites():
    """``MUSE_VR_DRAFT_PUSH_SITES`` (default ``o,down``): which row-parallel projections use the push epilogue when it is
    enabled; the other boundary keeps the stock projection and the full boundary exchange."""
    raw = os.environ.get("MUSE_VR_DRAFT_PUSH_SITES", "o,down")
    sites = {v.strip() for v in raw.split(",") if v.strip()}
    if not sites or sites - {"o", "down"}:
        raise ValueError(
            f"MUSE_VR_DRAFT_PUSH_SITES={raw!r}: expected a non-empty subset of o,down"
        )
    return "o" in sites, "down" in sites


def _push_eligible(model, m: int, comm) -> bool:
    """The push epilogue for both row-parallel projections of every layer (whole-cluster boundary layout, bf16 weights)."""
    if not bed.flag("draft_push_epi"):
        return False
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    from .ops import draft_ar_norm as op

    cl = op.cluster_for(m)
    if (
        m % BLOCK
        or (m // BLOCK) * cl > int(comm.config.num_push_blocks)
        or comm.world_size > 8
    ):
        return False
    for ly in model.layers:
        for lin in (ly.self_attn.o_proj, ly.mlp.down_proj):
            w = lin.weight
            if (
                not isinstance(lin.quant_method, UnquantizedLinearMethod)
                or w.dtype != torch.bfloat16
                or not w.is_contiguous()
                or getattr(lin, "bias", None) is not None
            ):
                return False
    return True


def _push_ready(m: int, device, world: int) -> bool:
    """Build and load the push GEMVs and the consumer-only boundary outside graph capture."""
    from .ops import draft_ar_norm as op
    from .ops import draft_gemv

    to, td = _push_tactics()
    pf = draft_gemv.l2pf(draft_gemv.EPI_PUSH)
    mc = draft_gemv.push_mc()
    key = (
        "push",
        draft_gemv.token_tile(m),
        to,
        td,
        world,
        draft_gemv.use_pdl(),
        pf,
        mc,
        device.index,
    )
    if key in _state["loaded"]:
        return True
    if torch.cuda.is_current_stream_capturing():
        return False
    for ts in (to, td):
        draft_gemv.module_push(
            draft_gemv.token_tile(m),
            draft_gemv.Tactic.parse(ts),
            world,
            draft_gemv.use_pdl(),
            draft_gemv.early_trigger(),
            pf,
            mc,
        )
    op.module(
        8192,
        world,
        op.use_pdl(),
        op.cluster_for(m),
        op.poll_all_from_env(),
        op.early_trigger_from_env(),
        True,
        mc,
        op.mbarx_from_env(),
    )
    _state["loaded"].add(key)
    return True


def _prepare(conv, h, use_gemv):
    """The input-side convolution of ``h`` and the output-side coefficients as (base, offset, row stride)."""
    if use_gemv:
        from .ops import draft_gemv

        h_conv, coef = draft_gemv.conv_in(
            h, conv.kernel_projection.weight, conv.base_kernel[0]
        )
        groups = coef.shape[1] // 4
        return h_conv, (coef, 2 * groups, 4 * groups)
    h_conv, kern = conv.prepare(h)
    cv = _coef_view(kern)
    if cv is None:
        # Not the in-place layout the boundary kernel reads (e.g. not a view of one contiguous projection output):
        # read a contiguous copy of the same values instead.
        bed.count(NAME, "coefficients copied")
        flat = kern.reshape(-1, kern.shape[-2], kern.shape[-1]).contiguous()
        cv = (flat, 0, flat.stride(0))
    return h_conv, cv


def _gemv_eligible(model, m: int) -> bool:
    if (
        not bed.flag("draft_gemv")
        or m > bed.env_int("MUSE_VR_DRAFT_GEMV_MAX_M", 16)
        or m % BLOCK
    ):
        return False
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    from .ops import draft_gemv

    # A token tile whose ring does not fit the device's per-block shared memory (227 KiB on sm_103)
    # keeps the stock projections instead of failing at the kernel's launch.
    if not all(
        draft_gemv.fits(draft_gemv.token_tile(m), epi, torch.cuda.current_device())
        for epi in (draft_gemv.EPI_CONV, draft_gemv.EPI_SWIGLU)
    ):
        bed.count(NAME, "draft_gemv: token tile exceeds the device's shared memory")
        return False
    for ly in model.layers:
        for conv in (ly.attention_conv, ly.mlp_conv):
            kp = conv.kernel_projection
            if (
                type(kp) is not torch.nn.Linear
                or kp.bias is not None
                or kp.weight.dtype != torch.bfloat16
            ):
                return False
        gu = ly.mlp.gate_up_proj
        if (
            not isinstance(gu.quant_method, UnquantizedLinearMethod)
            or gu.bias is not None
            or (gu.weight.dtype != torch.bfloat16)
            or not gu.weight.is_contiguous()
        ):
            return False
    return True


def _gemv_ready(m: int, device) -> bool:
    """Build and load the GEMV kernels outside graph capture; inside a capture only report whether they are."""
    from .ops import draft_gemv

    pfs = (draft_gemv.l2pf(draft_gemv.EPI_CONV), draft_gemv.l2pf(draft_gemv.EPI_SWIGLU))
    key = (
        "gemv",
        draft_gemv.token_tile(m),
        draft_gemv.use_pdl(),
        draft_gemv.early_trigger(),
        pfs,
        draft_gemv.conv_prefetch(),
        device.index,
    )
    if key in _state["loaded"]:
        return True
    if torch.cuda.is_current_stream_capturing():
        return False
    t = draft_gemv.token_tile(m)
    for epi, boxes in ((draft_gemv.EPI_CONV, 4), (draft_gemv.EPI_SWIGLU, 2)):
        draft_gemv.module(
            t,
            draft_gemv.Tactic.parse(draft_gemv.default_tactic(epi, t)),
            boxes,
            epi,
            draft_gemv.use_pdl(),
            draft_gemv.early_trigger(),
            draft_gemv.l2pf(epi),
            draft_gemv.conv_prefetch() and epi == draft_gemv.EPI_CONV,
        )
    _state["loaded"].add(key)
    return True


def _qk_eligible(model, flag_name: str = "draft_qk") -> bool:
    if not bed.flag(flag_name):
        return False
    import sglang.srt.models.dflash as d
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    if d._is_npu:
        return False
    for ly in model.layers:
        a = ly.self_attn
        if (
            type(a) is not d.DFlashAttention
            or not a.use_table_qk_norm_rope
            or a.v_scale is not None
            or (a.attention_sink_bias is not None)
        ):
            return False
        qp = a.qkv_proj
        if (
            not isinstance(qp.quant_method, UnquantizedLinearMethod)
            or qp.weight.dtype != torch.bfloat16
        ):
            return False
        for w in (a.q_norm.weight, a.k_norm.weight):
            if (
                w.dtype != torch.bfloat16
                or not w.is_contiguous()
                or w.numel() != a.head_dim
            ):
                return False
    return True


def _qk_ready(device) -> bool:
    """The Triton kernel is compiled on its first (eager) call; inside a capture only use it once that has happened."""
    key = ("qk", device.index)
    if key in _state["loaded"]:
        return True
    if torch.cuda.is_current_stream_capturing():
        return False
    _state["loaded"].add(key)
    return True


def _kv_store_target(
    attn, forward_batch, rows: int, require_flag: bool = True, why=None
):
    """(write slots, K buffer, V buffer) when the attention backend's KV write for this layer would be a plain copy of the
    bf16 K / V rows into a token-major bf16 pool (FlashAttention backend, plain MHA pool, no scales, no MLA / context
    parallelism / skipped cache); None otherwise (the backend writes as usual)."""
    if require_flag and not bed.flag("draft_qk_store"):
        return None
    from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
    from sglang.srt.layers.cp.utils import is_cp_active
    from sglang.srt.mem_cache.memory_pool import (
        KVWriteLoc,
        MHATokenToKVPool,
        maybe_detect_oob,
        unwrap_write_loc,
    )

    def no(reason):
        return why(reason) if why is not None else None

    from sglang.srt.model_executor.forward_context import (
        get_attn_backend,
        has_forward_context,
    )

    ra = attn.attn
    # the backend RadixAttention dispatches to (the active forward context's), else the batch's
    backend = (
        get_attn_backend()
        if has_forward_context()
        else getattr(forward_batch, "attn_backend", None)
    )
    if type(backend) is not FlashAttentionBackend:
        return no(f"backend {type(backend).__name__}")
    pool = backend.token_to_kv_pool  # the pool the backend's own KV write uses
    if type(pool) is not MHATokenToKVPool:
        return no(f"pool {type(pool).__name__}")
    if (
        backend.use_mla
        or backend.fa_skip_kv_cache
        or is_cp_active(forward_batch)
        or pool.is_quantized_kv_cache
        or pool.dtype != torch.bfloat16
        or pool.store_dtype != torch.bfloat16
        or pool.use_hnd
        or pool.kv_cache_layout != "nhd"
        or ra.is_cross_attention
        or ra.k_scale is not None
        or ra.v_scale is not None
    ):
        return no("kv write form")
    loc_info = KVWriteLoc.for_layer(
        forward_batch, ra, swa_loc=backend.forward_metadata.swa_out_cache_loc
    )
    pool._check_physical_write_loc(loc_info, "draft_boundary KV store")
    loc = unwrap_write_loc(loc_info)[0]
    if (
        loc.dtype != torch.int64
        or loc.dim() != 1
        or loc.shape[0] != rows
        or not loc.is_contiguous()
    ):
        return no(
            f"write slots {loc.dtype} {tuple(loc.shape)} contiguous={loc.is_contiguous()}"
        )
    maybe_detect_oob(loc, 0, pool.size + pool.page_size, "draft_boundary KV store")
    kb = pool.k_buffer[ra.layer_id - pool.start_layer]
    vb = pool.v_buffer[ra.layer_id - pool.start_layer]
    shape = (ra.tp_k_head_num, attn.head_dim)
    for b in (kb, vb):
        if (
            b.dim() != 3
            or tuple(b.shape[1:]) != shape
            or b.stride(2) != 1
            or b.stride(1) != attn.head_dim
            or b.stride(0) != kb.stride(0)
            or b.dtype != torch.bfloat16
        ):
            return no(f"cache buffer {tuple(b.shape)} {tuple(b.stride())}")
    if ra.tp_v_head_num != ra.tp_k_head_num or ra.v_head_dim != attn.head_dim:
        return no("head counts")
    return loc, kb, vb


def _qkv_epi_args(attn, positions, hidden_states, forward_batch):
    """Arguments of ops/draft_gemv.qkv_qk for this layer when ``MUSE_VR_DRAFT_QKV_EPI=1`` and the projection, the norm + RoPE
    inputs and the KV write qualify (see ``_kv_store_target``); None otherwise."""
    if not bed.flag("draft_qkv_epi"):
        return None
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    m = hidden_states.shape[0]
    qp = attn.qkv_proj
    w = qp.weight
    capturing = torch.cuda.is_current_stream_capturing()

    def no(reason):
        if not capturing:
            bed.count(NAME, "qkv stock:" + reason)
        return None

    if m > 16:
        return no("rows")
    if (
        not isinstance(qp.quant_method, UnquantizedLinearMethod)
        or getattr(qp, "bias", None) is not None
        or w.dtype != torch.bfloat16
        or not w.is_contiguous()
        or attn.head_dim != 128
        or w.shape[0] != (attn.num_heads + 2 * attn.num_kv_heads) * 128
    ):
        return no("projection")
    if hidden_states.dtype != torch.bfloat16 or not hidden_states.is_contiguous():
        return no("hidden layout")
    if (
        positions.dtype != torch.int64
        or positions.dim() != 1
        or positions.shape[0] != m
        or not positions.is_contiguous()
    ):
        return no(f"positions {positions.dtype} {tuple(positions.shape)}")
    cs = attn.rotary_emb.cos_sin_cache
    if (
        cs.dim() != 2
        or cs.shape[1] != 128
        or not cs.is_contiguous()
        or cs.dtype not in (torch.float32, torch.bfloat16)
    ):
        return no(f"rotary table {cs.dtype} {tuple(cs.shape)}")
    if not _qkv_ready(m, hidden_states.device):
        return no("not loaded before capture")
    tgt = _kv_store_target(attn, forward_batch, m, require_flag=False, why=no)
    if tgt is None:
        return None
    loc, kb, vb = tgt
    q = torch.empty(
        (m, attn.num_heads, 128), dtype=torch.bfloat16, device=hidden_states.device
    )
    return (
        hidden_states,
        w,
        q,
        kb,
        vb,
        loc,
        positions,
        cs,
        attn.q_norm.weight,
        attn.k_norm.weight,
        attn.q_norm.variance_epsilon,
    )


def _qkv_ready(m: int, device) -> bool:
    """Build and load the QKV-epilogue GEMV outside graph capture; inside a capture only report whether it is."""
    from .ops import draft_gemv

    t = draft_gemv.Tactic.parse(
        draft_gemv.default_tactic(draft_gemv.EPI_QK, draft_gemv.token_tile(m))
    )
    key = (
        "qkv",
        draft_gemv.token_tile(m),
        t,
        draft_gemv.use_pdl(),
        draft_gemv.early_trigger(),
        draft_gemv.l2pf(draft_gemv.EPI_QK),
        draft_gemv.qk_prefetch(),
        device.index,
    )
    if key in _state["loaded"]:
        return True
    if torch.cuda.is_current_stream_capturing() or t.rows != 128:
        return False
    draft_gemv.module(
        draft_gemv.token_tile(m),
        t,
        1,
        draft_gemv.EPI_QK,
        draft_gemv.use_pdl(),
        draft_gemv.early_trigger(),
        draft_gemv.l2pf(draft_gemv.EPI_QK),
        draft_gemv.qk_prefetch(),
    )
    _state["loaded"].add(key)
    return True


def _attention_qk(attn, positions, hidden_states, forward_batch, push_comm=None):
    """DFlashAttention.forward (pinned revision, table QK-norm + RoPE path, no value scale, no sinks) with the query
    heads handed to the attention layer as a contiguous tensor; o_proj without its all-reduce. With
    ``MUSE_VR_DRAFT_QK_STORE=1`` the norm + RoPE kernel also writes the K / V cache rows and the backend skips its write."""
    from .ops.draft_qk import qk_norm_rope_qout, qk_norm_rope_qout_store

    epi = _qkv_epi_args(attn, positions, hidden_states, forward_batch)
    if epi is not None:
        from .ops import draft_gemv

        q = draft_gemv.qkv_qk(*epi)
        if not torch.cuda.is_current_stream_capturing():
            bed.count(NAME, "qkv epilogue")
        out = attn.attn(
            q.view(q.shape[0], attn.q_size),
            None,
            None,
            forward_batch,
            save_kv_cache=False,
        )
        return _attention_out(attn, out, push_comm)
    qkv, _ = attn.qkv_proj(hidden_states)
    tgt = _kv_store_target(attn, forward_batch, qkv.shape[0])
    if tgt is not None:
        q = qk_norm_rope_qout_store(
            qkv,
            positions,
            attn.q_norm.weight,
            attn.k_norm.weight,
            attn.rotary_emb.cos_sin_cache,
            attn.num_heads,
            attn.num_kv_heads,
            attn.head_dim,
            attn.q_norm.variance_epsilon,
            *tgt,
        )
    else:
        q = qk_norm_rope_qout(
            qkv,
            positions,
            attn.q_norm.weight,
            attn.k_norm.weight,
            attn.rotary_emb.cos_sin_cache,
            attn.num_heads,
            attn.num_kv_heads,
            attn.head_dim,
            attn.q_norm.variance_epsilon,
        )
    _, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
    if tgt is not None:
        if not torch.cuda.is_current_stream_capturing():
            bed.count(NAME, "kv store fused")
        out = attn.attn(
            q.view(q.shape[0], attn.q_size), k, v, forward_batch, save_kv_cache=False
        )
    else:
        out = attn.attn(q.view(q.shape[0], attn.q_size), k, v, forward_batch)
    return _attention_out(attn, out, push_comm)


def _attention_out(attn, out, push_comm):
    """o_proj without its all-reduce (or the push-epilogue GEMV)."""
    if push_comm is not None:
        from .ops import draft_ar_norm as op
        from .ops import draft_gemv

        return draft_gemv.push(
            out.reshape(out.shape[0], -1),
            attn.o_proj.weight,
            push_comm,
            op.cluster_for(out.shape[0]),
            _push_tactics()[0],
        )
    out, _ = attn.o_proj(out, skip_all_reduce=True)
    return out


def _attention_qk_full(attn, positions, hidden_states, forward_batch):
    """DFlashAttention.forward (pinned revision, table QK-norm + RoPE path, no value scale, no sinks) with the query
    heads handed to the attention layer as a contiguous tensor; everything else as stock (o_proj with its all-reduce)."""
    from .ops.draft_qk import qk_norm_rope_qout

    qkv, _ = attn.qkv_proj(hidden_states)
    q = qk_norm_rope_qout(
        qkv,
        positions,
        attn.q_norm.weight,
        attn.k_norm.weight,
        attn.rotary_emb.cos_sin_cache,
        attn.num_heads,
        attn.num_kv_heads,
        attn.head_dim,
        attn.q_norm.variance_epsilon,
    )
    _, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
    out = attn.attn(q.view(q.shape[0], attn.q_size), k, v, forward_batch)
    out = attn.apply_attention_output(out, hidden_states)
    out, _ = attn.o_proj(out)
    return out


@contextlib.contextmanager
def _qk_wide_layers(model):
    """For the duration of one stock draft forward: each layer's attention module runs ``_attention_qk_full``
    (an instance attribute shadowing the class method; removed again on exit, also on an exception)."""
    patched = []
    try:
        for ly in model.layers:
            a = ly.self_attn

            def fwd(positions, hidden_states, forward_batch, _a=a):
                return _attention_qk_full(_a, positions, hidden_states, forward_batch)

            a.forward = fwd
            patched.append(a)
        yield
    finally:
        for a in patched:
            del a.forward


def _fused_forward(
    model,
    positions,
    hidden,
    forward_batch,
    comm,
    use_gemv,
    use_qk=False,
    use_push=False,
    input_ids=None,
):
    """DFlashDraftModel.forward (pinned revision) with the layer boundaries fused. ``use_push`` (with use_gemv and use_qk):
    the row-parallel projections push their outputs; the boundaries run consumer-only. ``input_ids``: the block ids
    (layer 0 from the drafter table when ``_l0_arm`` arms it)."""
    use_push = use_push and use_gemv and use_qk
    push_o, push_down = _push_sites() if use_push else (False, False)
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput

    layers = list(model.layers)
    residual = hidden
    tgt0 = _l0_arm(model, input_ids, positions, hidden, forward_batch, use_gemv, use_qk)
    h = layers[0].input_layernorm(hidden) if tgt0 is None else None
    for i, layer in enumerate(layers):
        if (
            i == 0 and tgt0 is not None
        ):  # layer 0's attention input side from the drafter table
            partial, coef = _attention_table(
                layer.self_attn,
                positions,
                input_ids,
                forward_batch,
                tgt0,
                comm if push_o else None,
            )
        elif use_qk:
            h_conv, coef = _prepare(layer.attention_conv, h, use_gemv)
            partial = _attention_qk(
                layer.self_attn,
                positions,
                h_conv,
                forward_batch,
                comm if push_o else None,
            )
        else:
            h_conv, coef = _prepare(layer.attention_conv, h, use_gemv)
            with _deferred_reduce(layer.self_attn.o_proj):
                partial = layer.self_attn(
                    positions=positions,
                    hidden_states=h_conv,
                    forward_batch=forward_batch,
                )
        h = _boundary(
            comm,
            partial,
            coef,
            layer.attention_conv,
            residual,
            layer.post_attention_layernorm,
            push_o,
        )
        h_conv, coef = _prepare(layer.mlp_conv, h, use_gemv)
        if use_gemv:
            from .ops import draft_gemv

            act = draft_gemv.swiglu(h_conv, layer.mlp.gate_up_proj.weight)
            if push_down:
                from .ops import draft_ar_norm as op

                partial = draft_gemv.push(
                    act,
                    layer.mlp.down_proj.weight,
                    comm,
                    op.cluster_for(act.shape[0]),
                    _push_tactics()[1],
                )
            else:
                partial, _ = layer.mlp.down_proj(act, skip_all_reduce=True)
        else:
            with _deferred_reduce(layer.mlp.down_proj):
                partial = layer.mlp(h_conv)
        nxt = layers[i + 1].input_layernorm if i + 1 < len(layers) else model.norm
        h = _boundary(comm, partial, coef, layer.mlp_conv, residual, nxt, push_down)
    return LogitsProcessorOutput(next_token_logits=None, hidden_states=h)


_ARGS = (
    "input_ids",
    "positions",
    "forward_batch",
    "input_embeds",
    "get_embedding",
    "pp_proxy_tensors",
)


def draft_forward(original, model, *args, **kwargs):
    bound = dict(zip(_ARGS, args))
    bound.update(kwargs)
    hidden = bound.get("input_embeds")
    if (
        set(bound) - set(_ARGS)
        or hidden is None
        or bound.get("get_embedding")
        or bound.get("pp_proxy_tensors") is not None
        or getattr(model, "embed_tokens", None) is not None
        or hidden.dim() != 2
    ):
        bed.count(NAME, "stock:call form")
        return original(model, *args, **kwargs)
    reason, comm = _model_eligible(model, hidden)
    if reason is not None:
        if _qk_eligible(model, "draft_qk_wide") and _qk_ready(hidden.device):
            bed.count(NAME, f"stock+qk_wide:rows={hidden.shape[0]}:" + reason)
            with _qk_wide_layers(model):
                return original(model, *args, **kwargs)
        bed.count(NAME, "stock:" + reason)
        return original(model, *args, **kwargs)
    m = hidden.shape[0]
    use_gemv = _gemv_eligible(model, m) and _gemv_ready(m, hidden.device)
    use_qk = _qk_eligible(model) and _qk_ready(hidden.device)
    use_push = (
        use_gemv
        and use_qk
        and _push_eligible(model, m, comm)
        and _push_ready(m, hidden.device, int(comm.world_size))
    )
    sites = ""
    if use_push and _push_sites() != (True, True):
        sites = ":sites=" + ("o" if _push_sites()[0] else "down")
    from .ops import draft_gemv as _dg

    if use_push and _dg.push_mc():
        sites += ":mc"
    if use_gemv and _dg.conv_prefetch():
        sites += ":conv_pf"
    if use_qk and _dg.qk_prefetch():
        sites += ":qk_pf"
    bed.count(
        NAME,
        f"fused:rows={m}:tp={comm.world_size}:gemv={int(use_gemv)}:qk={int(use_qk)}:push={int(use_push)}{sites}",
    )
    with torch.no_grad():
        return _fused_forward(
            model,
            bound["positions"],
            hidden,
            bound["forward_batch"],
            comm,
            use_gemv,
            use_qk,
            use_push,
            bound.get("input_ids"),
        )


def register() -> bool:
    if not enabled():
        return False
    ok, why = pins_match()
    if not ok:
        bed.report(NAME, False, reason=why)
        return False
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    def hook(original, *args, **kwargs):
        if not args:
            return original(*args, **kwargs)
        return draft_forward(original, *args, **kwargs)

    HookRegistry.register(TARGET, hook, HookType.AROUND)
    from .ops import draft_ar_norm as op
    from .ops import draft_gemv

    try:
        l2pf = {
            k: draft_gemv.l2pf(e)
            for k, e in (
                ("conv", draft_gemv.EPI_CONV),
                ("swiglu", draft_gemv.EPI_SWIGLU),
                ("push", draft_gemv.EPI_PUSH),
            )
        }
    except ValueError as e:
        bed.report(NAME, False, reason=str(e))
        return False
    bed.report(
        NAME,
        True,
        poll_all=op.poll_all_from_env(),
        early_trigger=op.early_trigger_from_env(),
        push_epi=bed.flag("draft_push_epi"),
        push_mc=draft_gemv.push_mc(),
        qk_store=bed.flag("draft_qk_store"),
        qkv_epi=bed.flag("draft_qkv_epi"),
        l2pf=l2pf,
        mc_push=op.mc_push_from_env(),
        split=op.split_from_env(),
        mbarx=op.mbarx_from_env(),
    )
    if bed.flag(TABLE):
        _register_table(HookRegistry, HookType)
    return True


# ---------------------------------------------------------------------------- the drafter layer-0 table
# MUSE_VR_DRAFT_L0_TABLE=1 (see the module docstring). Entry e of the table (0 <= e < vocab: id e; e = vocab: an id
# outside the vocabulary, built from id -1 as the masked lookup serves it) holds [QKV row 0 | QKV row 1 | output-side
# coefficients row 0] of the block [e, mask x 7]; ``cq`` holds QKV rows 2..7 and ``cc`` coefficient rows 1..7, equal for
# every block (checked over the whole build). Coefficients are kept in the layout the boundary reads (taps x groups per
# row), so the gathered block hands the boundary (coef, 0, 2 x groups) as ``_prepare``'s copied form does.
TABLE = "draft_l0_table"  # env MUSE_VR_DRAFT_L0_TABLE (with this component, MUSE_VR_DRAFT_GEMV / _QK / _QKV_EPI)
T_WORKER = "sglang.srt.speculative.dflash_worker_v2.DFlashWorkerV2.__init__"
WORKER_MODULE = "sglang.srt.speculative.dflash_worker_v2"
# sha256 of the SGLang sources the table path relies on: the worker's draft block ([bonus, mask x 7]), its noise embedding
# and the draft forward batch carrying the block ids as input_ids; the decode graph runner staging forward_batch.input_ids
# into the static buffer the captured forward reads, and the buffer registry's copy of it
TABLE_PINS = {
    "sglang/srt/speculative/dflash_worker_v2.py": "93214d7935961f620475e85e14b418ae9054b84c9a549c1ea62720cdd7088c5d",
    "sglang/srt/model_executor/runner/decode_cuda_graph_runner.py": "cee1cfcc527caa981f8ccf559572a499b6f176cbc086d3047e27c29ac8b48907",
    "sglang/srt/model_executor/cuda_graph_buffer_registry.py": "81431f2d2ca2f36b9a1d80d011774bc6b707674f846cc93f65d380131593b6e2",
}
CHECK_BLOCKS = 512  # random blocks of the install-time eager check (plus the edge ids)
CHECK_REPLAYS = 200  # CUDA-graph replays of the install-time replay check
GATHER_BLOCK = 256  # columns per gather program

_dt: dict = {
    "table": None,
    "cq": None,
    "cc": None,
    "model": None,
    "nq": 0,
    "nc": 0,
    "vocab": 0,
    "mask": None,
}

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU test environments; the table needs CUDA anyway
    triton = None

if triton is not None:

    @triton.jit
    def _draft_l0_gather_kernel(
        ids_ptr,
        table_ptr,
        cq_ptr,
        cc_ptr,
        qkv_ptr,
        coef_ptr,
        vocab,
        NQ: tl.constexpr,
        NC: tl.constexpr,
        ROWS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # program (row, chunk): BLOCK columns of [QKV row (NQ) | output-side coefficients (NC)] of draft row ``row``,
        # moved as their 16 bits: QKV rows 0 / 1 and coefficient row 0 from the entry of the block's first id (an id
        # outside [0, vocab): entry vocab), the other rows from the constant rows
        row = tl.program_id(0)
        chunk = tl.program_id(1)
        r = row % ROWS
        token = tl.load(ids_ptr + (row - r)).to(tl.int64)
        entry = tl.where((token >= 0) & (token < vocab), token, vocab)
        cols = chunk * BLOCK + tl.arange(0, BLOCK)
        is_q = cols < NQ
        in_c = (cols >= NQ) & (cols < NQ + NC)
        ccol = cols - NQ
        from_t = (is_q & (r < 2)) | (in_c & (r == 0))
        t_col = tl.where(is_q, r * NQ + cols, 2 * NQ + ccol)
        v_t = tl.load(table_ptr + entry * (2 * NQ + NC) + t_col, mask=from_t, other=0)
        v_q = tl.load(cq_ptr + (r - 2) * NQ + cols, mask=is_q & (r >= 2), other=0)
        v_c = tl.load(cc_ptr + (r - 1) * NC + ccol, mask=in_c & (r >= 1), other=0)
        val = tl.where(from_t, v_t, tl.where(is_q, v_q, v_c))
        tl.store(qkv_ptr + row.to(tl.int64) * NQ + cols, val, mask=is_q)
        tl.store(coef_ptr + row.to(tl.int64) * NC + ccol, val, mask=in_c)


def gather(ids):
    """``(qkv [rows, NQ], coef [rows, NC])`` bf16 for the draft rows of ``ids`` (1-D int64, whole blocks of
    ``BLOCK`` ids): the table's rows for each block's first id and the constant rows (Triton, 16-bit moves)."""
    t, nq, nc = _dt["table"], _dt["nq"], _dt["nc"]
    rows = ids.numel()
    qkv = torch.empty((rows, nq), dtype=torch.bfloat16, device=t.device)
    coef = torch.empty((rows, nc), dtype=torch.bfloat16, device=t.device)
    _draft_l0_gather_kernel[(rows, triton.cdiv(nq + nc, GATHER_BLOCK))](
        ids,
        t.view(torch.int16),
        _dt["cq"].view(torch.int16),
        _dt["cc"].view(torch.int16),
        qkv.view(torch.int16),
        coef.view(torch.int16),
        _dt["vocab"],
        NQ=nq,
        NC=nc,
        ROWS=BLOCK,
        BLOCK=GATHER_BLOCK,
        num_warps=4,
    )
    return qkv, coef


def gather_torch(ids):
    """The same rows by torch indexing (the CPU tests' gather; at install, the Triton gather's cross-check)."""
    t, nq, nc, vocab = _dt["table"], _dt["nq"], _dt["nc"], _dt["vocab"]
    first = ids.view(-1, BLOCK)[:, 0]
    entry = torch.where(
        (first >= 0) & (first < vocab), first, torch.full_like(first, vocab)
    )
    ent = t[entry]
    blocks = first.numel()
    qkv = torch.empty((blocks, BLOCK, nq), dtype=t.dtype, device=t.device)
    qkv[:, 0], qkv[:, 1], qkv[:, 2:] = ent[:, :nq], ent[:, nq : 2 * nq], _dt["cq"]
    coef = torch.empty((blocks, BLOCK, nc), dtype=t.dtype, device=t.device)
    coef[:, 0], coef[:, 1:] = ent[:, 2 * nq :], _dt["cc"]
    return qkv.view(-1, nq), coef.view(-1, nc)


def _gather_fn():
    return (
        gather
        if triton is not None and _dt["table"] is not None and _dt["table"].is_cuda
        else gather_torch
    )


def _worker_parts(worker):
    """(draft model, noise embedding module, mask id, noise scale, replicated embedding or None, vocabulary size) as the
    worker's decode step uses them (dflash_worker_v2: the draft block and its noise embedding)."""
    wm = sys.modules[WORKER_MODULE]
    tw = getattr(worker, "_target_worker", None) or worker.target_worker
    target = tw.model_runner.model
    embed = wm.unwrap_lora_layer(
        wm._resolve_dflash_embedding_module(worker.draft_model, target)
    )
    vocab = int(target.model.language_model.embed_tokens.num_embeddings)
    return (
        worker.draft_model,
        embed,
        int(worker._mask_token_id),
        float(worker._noise_embed_scale),
        worker._full_embed_gpu,
        vocab,
    )


def _noise_rows(embed, scale, full, block_ids):
    """The draft forward's input rows for ``block_ids`` [blocks, BLOCK], as the worker's decode step computes them."""
    if full is not None:
        x = torch.nn.functional.embedding(block_ids, full)
    else:
        x = embed(block_ids)
    if scale != 1.0:
        x = x * scale
    return x.view(-1, x.shape[-1])


def _qk_tactic() -> str:
    from .ops import draft_gemv

    return draft_gemv.default_tactic(draft_gemv.EPI_QK, draft_gemv.token_tile(BLOCK))


def _front(model, hidden):
    """Layer 0's served attention input side up to the QKV projection: (convolved input, coefficients as
    ``(base, offset, row stride)``), by the fused forward's own calls."""
    layer = model.layers[0]
    return _prepare(layer.attention_conv, layer.input_layernorm(hidden), True)


def _block_rows(model, h_conv, coef):
    """(QKV projection [BLOCK, NQ] by the plain GEMV with the served QK tactic, output-side coefficients [BLOCK, NC])."""
    from .ops import draft_gemv

    qkv = draft_gemv.plain(
        h_conv, model.layers[0].self_attn.qkv_proj.weight, tactic=_qk_tactic()
    )
    base, off, stride = coef
    return qkv, base[:, off:stride]


def _block_ids(mask: int, first, device):
    ids = torch.full((1, BLOCK), mask, dtype=torch.int64, device=device)
    ids[0, 0] = first
    return ids


def build(model, embed, scale, full, mask: int, vocab: int, device):
    """``(table [vocab + 1, 2 NQ + NC], cq [BLOCK - 2, NQ], cc [BLOCK - 1, NC], blocks whose constant rows differ)``:
    the served front and the plain QKV GEMV on every block [e, mask x 7] (e = 0 .. vocab - 1) and on [-1, mask x 7]
    (entry vocab), one 8-row block per call as served. Blocks 0 .. vocab - 1 replay one CUDA graph whose first id is a
    device counter."""
    with torch.inference_mode():
        ids = _block_ids(mask, -1, device)
        qkv, cout = _block_rows(
            model, *_front(model, _noise_rows(embed, scale, full, ids))
        )
        nq, nc = int(qkv.shape[1]), int(cout.shape[1])
        table = torch.empty(
            (vocab + 1, 2 * nq + nc), dtype=torch.bfloat16, device=device
        )
        table[vocab, :nq], table[vocab, nq : 2 * nq], table[vocab, 2 * nq :] = (
            qkv[0],
            qkv[1],
            cout[0],
        )
        cq, cc = qkv[2:].clone(), cout[1:].clone()
        ctr = torch.zeros((), dtype=torch.int64, device=device)
        bad = torch.zeros((), dtype=torch.int64, device=device)
        ids = _block_ids(mask, 0, device)

        def one():
            ids[0, 0].copy_(ctr)
            q, c = _block_rows(
                model, *_front(model, _noise_rows(embed, scale, full, ids))
            )
            table.index_copy_(0, ctr.view(1), torch.cat([q[0], q[1], c[0]]).view(1, -1))
            bad.add_(
                (q[2:].view(torch.int16) != cq.view(torch.int16)).any().long()
                + (c[1:].view(torch.int16) != cc.view(torch.int16)).any().long()
            )
            ctr.add_(1)

        _repeat(one, vocab, device)
        done = int(ctr.item())
        if done != vocab:
            raise RuntimeError(f"draft_l0_table: built {done} of {vocab} entries")
    return table, cq, cc, int(bad.item())


def _repeat(fn, n: int, device) -> None:
    """``fn`` ``n`` times on a side stream: once eagerly (every kernel loaded before the capture), then captured once and
    replayed ``n - 1`` times."""
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        fn()
        if n > 1:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
                graph, stream=stream, capture_error_mode="thread_local"
            ):  # other init threads may run
                fn()
            for _ in range(n - 1):
                graph.replay()
    torch.cuda.current_stream(device).wait_stream(stream)


def check_ids(vocab: int, mask: int):
    """The first ids the check covers: the edge ids, then ``CHECK_BLOCKS`` random ids."""
    generator = torch.Generator(device="cpu").manual_seed(20261105)
    return torch.cat(
        [
            torch.tensor([0, 1, vocab - 1, mask, vocab, -1], dtype=torch.int64),
            torch.randint(
                0, vocab, (CHECK_BLOCKS,), generator=generator, dtype=torch.int64
            ),
        ]
    )


def _served_l0(model, embed, scale, full, ids, pos, loc, kb, vb):
    """The served layer-0 attention input side for the block ``ids`` [1, BLOCK]: (query heads, coefficients
    [BLOCK, NC]); writes the K / V rows ``loc`` of ``kb`` / ``vb`` (the QKV-epilogue GEMV, as ``_attention_qk``)."""
    from .ops import draft_gemv

    attn = model.layers[0].self_attn
    h_conv, coef = _front(model, _noise_rows(embed, scale, full, ids))
    q = torch.empty(
        (BLOCK, attn.num_heads, attn.head_dim), dtype=torch.bfloat16, device=ids.device
    )
    draft_gemv.qkv_qk(
        h_conv,
        attn.qkv_proj.weight,
        q,
        kb,
        vb,
        loc,
        pos,
        attn.rotary_emb.cos_sin_cache,
        attn.q_norm.weight,
        attn.k_norm.weight,
        attn.q_norm.variance_epsilon,
    )
    base, off, stride = coef
    return q, base[:, off:stride]


def _table_l0(model, ids, pos, loc, kb, vb, gather_fn):
    """The table's layer-0 attention input side for the block ``ids`` [BLOCK]: (query heads, coefficients)."""
    from .ops.draft_qk import qk_norm_rope_qout_store

    attn = model.layers[0].self_attn
    qkv, coef = gather_fn(ids)
    q = qk_norm_rope_qout_store(
        qkv,
        pos,
        attn.q_norm.weight,
        attn.k_norm.weight,
        attn.rotary_emb.cos_sin_cache,
        attn.num_heads,
        attn.num_kv_heads,
        attn.head_dim,
        attn.q_norm.variance_epsilon,
        loc,
        kb,
        vb,
    )
    return q, coef


def _scratch(model, device, slots: int = 16):
    attn = model.layers[0].self_attn
    return [
        torch.zeros(
            (slots, attn.num_kv_heads, attn.head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        for _ in range(4)
    ]


def _diff(a, b) -> int:
    return int(
        (a.contiguous().view(torch.int16) != b.contiguous().view(torch.int16)).sum()
    )


def check(model, embed, scale, full, mask: int, vocab: int, device) -> tuple:
    """``(blocks checked, mismatching blocks (at most 16 first ids))``: for each first id of ``check_ids`` at random
    positions and cache slots, the served layer-0 front against the table's: query heads, K / V rows, coefficients bit
    for bit, and the Triton gather against the torch one."""
    gen = torch.Generator(device="cpu").manual_seed(20261106)
    cs = model.layers[0].self_attn.rotary_emb.cos_sin_cache
    ks, vs, kt, vt = _scratch(model, device)
    bad: list = []
    firsts = check_ids(vocab, mask)
    with torch.inference_mode():
        for first in firsts.tolist():
            ids = _block_ids(mask, first, device)
            p0 = int(torch.randint(0, int(cs.shape[0]) - BLOCK, (1,), generator=gen))
            pos = torch.arange(p0, p0 + BLOCK, dtype=torch.int64, device=device)
            loc = torch.randperm(ks.shape[0], generator=gen)[:BLOCK].to(
                device=device, dtype=torch.int64
            )
            q_s, c_s = _served_l0(model, embed, scale, full, ids, pos, loc, ks, vs)
            q_t, c_t = _table_l0(model, ids.view(-1), pos, loc, kt, vt, _gather_fn())
            n = (
                _diff(q_s, q_t)
                + _diff(c_s, c_t)
                + _diff(ks[loc], kt[loc])
                + _diff(vs[loc], vt[loc])
            )
            if _gather_fn() is not gather_torch:
                n += sum(
                    _diff(a, b)
                    for a, b in zip(gather(ids.view(-1)), gather_torch(ids.view(-1)))
                )
            if n and len(bad) < 16:
                bad.append(first)
    return int(firsts.numel()), bad


def check_replays(
    model, embed, scale, full, mask: int, vocab: int, device, replays: int
) -> int:
    """Differing values over ``replays`` CUDA-graph replays of both layer-0 sides (one graph on static ids, positions and
    slots refilled before each replay)."""
    gen = torch.Generator(device="cpu").manual_seed(20261107)
    cs = model.layers[0].self_attn.rotary_emb.cos_sin_cache
    ks, vs, kt, vt = _scratch(model, device)
    ids = _block_ids(mask, 0, device)
    pos = torch.arange(BLOCK, dtype=torch.int64, device=device)
    loc = torch.arange(BLOCK, dtype=torch.int64, device=device)
    total = torch.zeros((), dtype=torch.int64, device=device)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.inference_mode(), torch.cuda.stream(stream):

        def both():
            q_s, c_s = _served_l0(model, embed, scale, full, ids, pos, loc, ks, vs)
            q_t, c_t = _table_l0(model, ids.view(-1), pos, loc, kt, vt, gather)
            return (
                (q_s.view(torch.int16) != q_t.view(torch.int16)).sum()
                + (c_s.contiguous().view(torch.int16) != c_t.view(torch.int16)).sum()
                + (ks[loc].view(torch.int16) != kt[loc].view(torch.int16)).sum()
                + (vs[loc].view(torch.int16) != vt[loc].view(torch.int16)).sum()
            )

        both()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local"):
            diff = both()
        for _ in range(replays):
            ids[0, 0] = int(torch.randint(0, vocab, (1,), generator=gen))
            p0 = int(torch.randint(0, int(cs.shape[0]) - BLOCK, (1,), generator=gen))
            pos.copy_(torch.arange(p0, p0 + BLOCK, dtype=torch.int64))
            loc.copy_(torch.randperm(ks.shape[0], generator=gen)[:BLOCK])
            graph.replay()
            total += diff
    torch.cuda.current_stream(device).wait_stream(stream)
    return int(total.item())


def _table_static_reason(worker) -> str | None:
    """Why the table cannot stand in for this worker's draft layer 0 (construction-time state only; rank-uniform)."""
    from sglang.srt.environ import envs

    from .ops import draft_gemv

    if not (
        enabled()
        and bed.flag("draft_gemv")
        and bed.flag("draft_qk")
        and bed.flag("draft_qkv_epi")
    ):
        return "needs MUSE_VR_DRAFT_BOUNDARY, _GEMV, _QK and _QKV_EPI"
    model = getattr(worker, "draft_model", None)
    if (
        model is None
        or getattr(model, "embed_tokens", None) is not None
        or not hasattr(model, "layers")
    ):
        return "draft model form"
    if int(getattr(worker, "block_size", 0)) != BLOCK:
        return "block size"
    if not _gemv_eligible(model, BLOCK) or not _qk_eligible(model):
        return "draft layers not on the GEMV / QK path"
    layer = model.layers[0]
    conv, attn = layer.attention_conv, layer.self_attn
    if (
        conv is None
        or conv.taps != 2
        or conv.group_size != 16
        or conv.block_size != BLOCK
        or conv.base_kernel.dtype != torch.bfloat16
    ):
        return "convolution shape"
    if (
        attn.head_dim != 128
        or attn.qkv_proj.weight.shape[0]
        != (attn.num_heads + 2 * attn.num_kv_heads) * 128
    ):
        return "projection shape"
    if draft_gemv.Tactic.parse(_qk_tactic()).rows != 128:
        return "QK tactic"
    if envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
        return "async id asserts on (the table's outside entry is built from id -1)"
    if not bed.installs().get("embed_gather", {}).get("engaged"):
        return "embed_gather not engaged (the outside entry needs its masked lookup)"
    return None


def install_table(worker) -> str | None:
    """Build and check the table for ``worker``'s draft model and arm it; the reason it is not installed, or None."""
    reason = _table_static_reason(worker)
    if reason is not None:
        return reason
    model, embed, mask, scale, full, vocab = _worker_parts(worker)
    device = model.layers[0].self_attn.qkv_proj.weight.device
    torch.cuda.synchronize(device)
    free0 = torch.cuda.mem_get_info(device)[0]
    t0 = time.monotonic()
    table, cq, cc, bad_blocks = build(model, embed, scale, full, mask, vocab, device)
    torch.cuda.synchronize(device)
    built = time.monotonic() - t0
    if bad_blocks:
        return f"{bad_blocks} blocks' constant rows differ from block -1's"
    _dt.update(
        table=table,
        cq=cq,
        cc=cc,
        nq=int(cq.shape[1]),
        nc=int(cc.shape[1]),
        vocab=vocab,
        mask=mask,
    )
    try:
        checked, bad = check(model, embed, scale, full, mask, vocab, device)
        replay_diff = check_replays(
            model, embed, scale, full, mask, vocab, device, CHECK_REPLAYS
        )
    except Exception:
        _dt.update(table=None, model=None)
        raise
    if bad or replay_diff:
        _dt.update(table=None, model=None)
        return f"the table differs from the served layer 0 (first ids {bad}; {replay_diff} values over replays)"
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    free1 = torch.cuda.mem_get_info(device)[0]
    _dt["model"] = model
    gib = float(1 << 30)
    info = dict(
        rows=vocab + 1,
        cols=int(table.shape[1]),
        nq=_dt["nq"],
        nc=_dt["nc"],
        mask=mask,
        scale=scale,
        gib=round(table.numel() * 2 / gib, 3),
        built_s=round(built, 1),
        checked=f"{checked} blocks",
        replays=CHECK_REPLAYS,
        qk_tactic=_qk_tactic(),
        free_gib=f"{free0 / gib:.2f}->{free1 / gib:.2f}",
    )
    bed.report(TABLE, True, **info)
    logger.warning(
        "muse_vr_plugin: %s installed pid=%d %s",
        TABLE,
        os.getpid(),
        " ".join(f"{k}={v}" for k, v in info.items()),
    )
    return None


def _after_worker_init(result, worker, *args, **kwargs):
    """``DFlashWorkerV2.__init__`` (AFTER): mask id, noise scale and trained mask embedding are final; the KV pools and
    every CUDA graph come later."""
    try:
        reason = install_table(worker)
    except Exception as e:  # a failed build leaves layer 0 served
        _dt.update(table=None, model=None)
        reason = f"{type(e).__name__}: {e}"[:300]
    if reason is not None:
        _dt.update(table=None, model=None)
        bed.report(TABLE, False, reason=reason)
        logger.warning(
            "muse_vr_plugin: %s not installed pid=%d reason=%s",
            TABLE,
            os.getpid(),
            reason,
        )
    return result


def _register_table(HookRegistry, HookType) -> None:
    bad = [rel for rel in TABLE_PINS if not bed.pinned(rel, TABLE_PINS[rel])]
    if bad:
        bed.report(TABLE, False, reason="pins: " + ", ".join(bad))
        return
    HookRegistry.register(T_WORKER, _after_worker_init, HookType.AFTER)
    bed.report(
        TABLE,
        False,
        reason="registered: the table is built when the DFLASH worker is constructed",
    )


def _l0_arm(model, input_ids, positions, hidden, forward_batch, use_gemv, use_qk):
    """``(slots, K buffer, V buffer)`` of layer 0's cache write when this forward's layer 0 takes the table (one block
    whose layer 0 would take the QKV epilogue: the same decision, ``_qkv_epi_args``), else None."""
    if _dt["table"] is None or _dt["model"] is not model:
        return None
    capturing = torch.cuda.is_current_stream_capturing()

    def no(reason):
        if not capturing:
            bed.count(TABLE, "stock:" + reason)
        return None

    if not (use_gemv and use_qk):
        return no("layer 0 not on the GEMV / QK path")
    if hidden.shape[0] != BLOCK:
        return no("rows")
    if (
        not isinstance(input_ids, torch.Tensor)
        or input_ids.device != hidden.device
        or input_ids.dtype != torch.int64
        or input_ids.dim() != 1
        or input_ids.numel() != BLOCK
        or not input_ids.is_contiguous()
    ):
        return no("block ids")
    epi = _qkv_epi_args(model.layers[0].self_attn, positions, hidden, forward_batch)
    if epi is None:
        return no("layer 0 not on the QKV epilogue")
    return epi[5], epi[3], epi[4]


def _attention_table(attn, positions, ids, forward_batch, tgt, push_comm):
    """Layer 0's attention with its input side from the table: the gathered rows, the QK norm + RoPE + K / V cache write,
    the QKV-epilogue path's attention call and output projection. Returns (partial, coefficients (base, 0, row stride))."""
    q, coef = _table_l0_from(attn, positions, ids, tgt)
    out = attn.attn(
        q.view(q.shape[0], attn.q_size), None, None, forward_batch, save_kv_cache=False
    )
    if not torch.cuda.is_current_stream_capturing():
        bed.count(TABLE, "table")
    return _attention_out(attn, out, push_comm), (coef, 0, int(coef.stride(0)))


def _table_l0_from(attn, positions, ids, tgt):
    from .ops.draft_qk import qk_norm_rope_qout_store

    loc, kb, vb = tgt
    qkv, coef = _gather_fn()(ids)
    q = qk_norm_rope_qout_store(
        qkv,
        positions,
        attn.q_norm.weight,
        attn.k_norm.weight,
        attn.rotary_emb.cos_sin_cache,
        attn.num_heads,
        attn.num_kv_heads,
        attn.head_dim,
        attn.q_norm.variance_epsilon,
        loc,
        kb,
        vb,
    )
    return q, coef

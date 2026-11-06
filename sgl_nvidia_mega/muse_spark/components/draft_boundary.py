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
activation fused (``ops/draft_gemv.swiglu``). Rounding points are the stock ones (bf16 operands, fp32 accumulation,
bf16 coefficients / projections before the convolution / activation); the projections' fp32 summation order differs
(tests/test_draft_gemv.py: error against fp64 equal to stock's). The attention's qkv and output projections keep
SGLang's dispatch (tests/test_draft_gemv.py also times the plain GEMV against the kernels the server selects for them).

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
    key = (hdim, group.world_size, op.use_pdl(), cluster, hidden.device.index)
    if torch.cuda.is_current_stream_capturing():
        if key not in _state["loaded"]:
            return "kernel not loaded before graph capture", None
    else:
        op.module(hdim, group.world_size, op.use_pdl(), cluster)
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


def _boundary(comm, partial, coef, conv, residual, norm):
    """``coef``: (base tensor, element offset of the output-side coefficients, row stride)."""
    from .ops import draft_ar_norm as op

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
    )


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

    key = ("gemv", draft_gemv.token_tile(m), draft_gemv.use_pdl(), device.index)
    if key in _state["loaded"]:
        return True
    if torch.cuda.is_current_stream_capturing():
        return False
    t = draft_gemv.token_tile(m)
    for epi, boxes in ((draft_gemv.EPI_CONV, 4), (draft_gemv.EPI_SWIGLU, 2)):
        draft_gemv.module(
            t,
            draft_gemv.Tactic.parse(draft_gemv.DEFAULTS[epi]),
            boxes,
            epi,
            draft_gemv.use_pdl(),
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


def _attention_qk(attn, positions, hidden_states, forward_batch):
    """DFlashAttention.forward (pinned revision, table QK-norm + RoPE path, no value scale, no sinks) with the query
    heads handed to the attention layer as a contiguous tensor; o_proj without its all-reduce."""
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
    model, positions, hidden, forward_batch, comm, use_gemv, use_qk=False
):
    """DFlashDraftModel.forward (pinned revision) with the layer boundaries fused."""
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput

    layers = list(model.layers)
    residual = hidden
    h = layers[0].input_layernorm(hidden)
    for i, layer in enumerate(layers):
        h_conv, coef = _prepare(layer.attention_conv, h, use_gemv)
        if use_qk:
            partial = _attention_qk(layer.self_attn, positions, h_conv, forward_batch)
        else:
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
        )
        h_conv, coef = _prepare(layer.mlp_conv, h, use_gemv)
        if use_gemv:
            from .ops import draft_gemv

            act = draft_gemv.swiglu(h_conv, layer.mlp.gate_up_proj.weight)
            partial, _ = layer.mlp.down_proj(act, skip_all_reduce=True)
        else:
            with _deferred_reduce(layer.mlp.down_proj):
                partial = layer.mlp(h_conv)
        nxt = layers[i + 1].input_layernorm if i + 1 < len(layers) else model.norm
        h = _boundary(comm, partial, coef, layer.mlp_conv, residual, nxt)
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
    bed.count(
        NAME,
        f"fused:rows={m}:tp={comm.world_size}:gemv={int(use_gemv)}:qk={int(use_qk)}",
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
    bed.report(NAME, True)
    return True

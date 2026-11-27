# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/qwen3_5.py (Qwen3_5GatedDeltaNet._forward_input_proj,
# Qwen3_5AttentionDecoderLayer.forward_prepare_cuda_fused) and python/sglang/srt/models/qwen2_moe.py
# (Qwen2MoeMLP.forward, Qwen2MoeSparseMoeBlock._gate_shared_output_out_of_place).
"""Slots ``lowm_in``, ``lowm_out`` and ``lowm_se``: a covered verify step's static per-tensor FP8 (W8A8) projections on one
low-M warp-MMA launch each (``sm103/lowm``), the weights loaded before the programmatic dependency wait.

Numeric: the same E4M3 operands as SGLang's static quantize + FlashInfer/cuBLAS FP8 GEMM (BF16 activations are
quantized in registers with the module's static input scale, the quantize kernel's arithmetic: the same FP8
values), FP32 sums in another order; BF16 projections stay BF16 x BF16 with FP32 sums.

- ``lowm_in`` (``Slots.lowm_in``): GDN in_proj_qkvz with in_proj_ba's BF16 rows in the same launch (SGLang
  runs in_proj_ba on its alternate stream), attention qkv_proj.
- ``lowm_out`` (``Slots.lowm_outproj`` / ``lowm_oproj``): GDN out_proj on gdn_verify's FP8 norm, attention
  o_proj on the gated BF16 output; both write SGLang's symmetric pool, as the all-reduce after them reads it.
- ``hop_fp8`` (``Slots.hop_fp8``, with ``hop``): the hop's all-reduce + norm also writes the next projection's
  E4M3 input with that projection's static scale (SGLang's quantization of the norm, byte for byte): the layer
  input for in_proj_qkvz / qkv_proj, the MoE input for the shared expert's gate_up; the projection then reads
  it (the low-M launch, or FlashInfer's BMM without the quantize kernel).
- ``lowm_se`` (``Slots.lowm_se``): the shared expert of the dual-stream deferred-finalize order in two launches
  on the alternate stream: gate_up with the SwiGLU and down_proj's static FP8 in the epilogue (the shared gate's
  logits as a BF16 segment of the same launch), then down_proj with its output times the gate's sigmoid.
"""

from __future__ import annotations

import logging

import msgspec
import torch

logger = logging.getLogger(__name__)

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/qwen3_5.py": (
        "Qwen3_5LinearDecoderLayer",
        "Qwen3_5AttentionDecoderLayer",
        "linear_attn",
        "in_proj_qkvz",
        "in_proj_ba",
        "out_proj",
        "qkv_proj",
        "o_proj",
        "attn_output_gate",
        "q_size",
        "kv_size",
        "q_norm",
        "k_norm",
        "rotary_emb",
        "num_heads",
        "num_kv_heads",
        "head_dim",
        "alt_stream",
        "_gdn_use_alt_stream",
        "get_is_capture_mode",
    ),
    "kernels/ops/quantization/fp8_kernel.py": ("static_quant_fp8",),
    "srt/models/qwen2_moe.py": (
        "mlp",
        "shared_expert",
        "shared_expert_gate",
        "gate_up_proj",
        "down_proj",
        "act_fn",
    ),
    "srt/layers/quantization/modelopt_quant.py": (
        "ModelOptFp8LinearMethod",
        "use_marlin",
        "use_sm120_fp8",
        "use_flashinfer_bmm",
        "orig_dtype",
        '"input_scale"',
        '"weight_scale"',
    ),
    "srt/layers/activation.py": ("SiluAndMul",),
    "kernels/ops/attention/fused_qk_rmsnorm_rope_gate.py": (
        "fused_qk_gemma_rmsnorm_rope_gate",
    ),
    "srt/distributed/device_communicators/pynccl_allocator.py": (
        "use_symmetric_memory",
    ),
    "srt/layers/dp_attention.py": ("is_allocation_symmetric",),
    "srt/runtime_context.py": ("get_parallel",),
    "srt/layers/quantization/fp8_utils.py": ("flashinfer_bmm_fp8",),
}

# The SGLang functions whose body this module mirrors, by the digest of their code
# (test/unit/test_contract.py recomputes them on the installed SGLang)
SGLANG_MIRRORS = {
    "srt/models/qwen3_5.py:Qwen3_5AttentionDecoderLayer.forward_prepare_cuda_fused": "d10e23f5651990d0",
}


# Per (N8, N16, K, rows, E4M3 input): the tactic (k warps, FP8 tiles, BF16 tiles, CTAs a SM, cluster K split)
# measured fastest on GB300 at TP8 (results/trt-1022/l6/data/micro x1-x4: CUDA-graph chains with and without a
# non-PDL kernel between launches); other shapes take sm103/lowm.default_tactic
def _t(*v):
    from sgl_nvidia_mega.qwen3_5.sm103.lowm import Tactic

    return Tactic(*v)


TACTICS: dict = {
    (2560, 16, 4096, 14, False): _t(16, 4, 2, 1, 1),  # in_proj_qkvz + in_proj_ba
    (2560, 0, 4096, 14, False): _t(8, 4, 1, 1, 1),  # qkv_proj
    (2560, 16, 4096, 14, True): _t(
        16, 4, 2, 1, 1
    ),  # in_proj_qkvz (E4M3 from the hop) + in_proj_ba
    (2560, 0, 4096, 14, True): _t(16, 4, 1, 1, 1),  # qkv_proj (E4M3 from the hop)
    (4096, 0, 1024, 14, True): _t(16, 2, 1, 1, 1),  # out_proj
    (4096, 0, 1024, 28, True): _t(8, 2, 1, 1, 1),
    (4096, 0, 1024, 14, False): _t(8, 4, 1, 1, 1),  # o_proj
    (4096, 0, 1024, 28, False): _t(4, 2, 1, 1, 1),
    (256, 8, 4096, 7, False): _t(
        8, 2, 1, 1, 4
    ),  # shared expert gate_up + SwiGLU, shared gate
    (256, 8, 4096, 14, False): _t(16, 2, 1, 1, 4),
    (256, 8, 4096, 28, False): _t(16, 2, 1, 1, 4),
    (256, 8, 4096, 56, False): _t(8, 2, 1, 1, 4),
    (256, 8, 4096, 7, True): _t(
        8, 2, 1, 1, 4
    ),  # shared expert gate_up on the hop's E4M3
    (256, 8, 4096, 14, True): _t(8, 2, 1, 1, 4),
    (256, 8, 4096, 28, True): _t(8, 2, 1, 1, 4),
    (4096, 0, 128, 7, True): _t(
        2, 1, 1, 1, 1
    ),  # shared expert down_proj x sigmoid(gate)
    (4096, 0, 128, 14, True): _t(2, 1, 1, 1, 1),
    (4096, 0, 128, 28, True): _t(2, 1, 1, 1, 1),
    (4096, 0, 128, 56, True): _t(2, 1, 1, 1, 1),
}

_ENGAGED: set = set()
HIDDEN = 4096  # the hop's row width


class Bound(msgspec.Struct, frozen=True):
    """The low-M slots' build result (``st.bound["lowm"]``)."""

    lowm_in: bool = False  # every GDN / attention FP8 input projection takes the launch
    lowm_out: bool = (
        False  # every GDN / attention FP8 output projection takes the launch
    )
    se: bool = False  # every shared expert takes the two launches
    se_pre: bool = False  # the launches also cover the MoE body's tier-M rows
    push: bool = False  # the output projections' launches also push into the hop
    hop_fp8: bool = False  # the hop also writes the next projection's E4M3 input
    # lowm_se: id(mlp) -> (gate_up rows paired for the SwiGLU, the shared gate zero-padded to 8 rows)
    shared: dict = msgspec.field(default_factory=dict)
    se_tep8: tuple = ()
    reason: str = ""


TEP8_SE_ROWS = (33, 128)
TEP8_SE_TAIL_ROWS = 32
# variant slot lowm_se_wide's bound: two launch pairs
TEP8_SE_WIDE_HI = 128 + TEP8_SE_TAIL_ROWS
TEP8_SE_T1 = (8, 2, 4)
TEP8_SE_T2 = (2, 1, 1)
TEP8_SE_CORESIDENT_ROWS = (84, 128)
TEP8_SE_CORESIDENT_SMEM = 23 * 1024
_TEP8_SE: dict = {}


def _configure_tep8_se(rows: tuple, coresident: bool) -> None:
    from sgl_nvidia_mega.qwen3_5.sm103 import lowm as sm

    _TEP8_SE.clear()
    _TEP8_SE.update(
        rows=rows,
        t1=sm.Tactic(TEP8_SE_T1[0], TEP8_SE_T1[1], 1, 1, TEP8_SE_T1[2]),
        t2=sm.Tactic(TEP8_SE_T2[0], TEP8_SE_T2[1], 1, 1, TEP8_SE_T2[2]),
        coresident=TEP8_SE_CORESIDENT_ROWS if rows and coresident else (),
        said=set(),
    )


def tep8_se_config() -> dict:
    if "rows" not in _TEP8_SE:
        _configure_tep8_se((), False)
    return _TEP8_SE


def tep8_se_down_smem(t2, rows: int) -> int:
    nc = t2.nt8 * 8
    smem = t2.kw * ((rows + 15) // 16) * 16 * nc * 4
    return smem + (rows * nc * 4 if t2.ksplit > 1 else 0)


def engaged(slot: str, what: str, rows: int) -> None:
    """Log once per (slot, projection, rows) that a low-M launch ran, as engagement evidence."""
    if (slot, what, rows) not in _ENGAGED:
        _ENGAGED.add((slot, what, rows))
        logger.info(
            "NVIDIA megakernel: Qwen3.5 slot %s runs %s at %d rows", slot, what, rows
        )


def _w8(linear) -> torch.Tensor:
    """ModelOpt's FP8 weight [K, N] (column-major) as the row-major [N, K] it views."""
    return linear.weight.t()


def fp8_unsupported(linear) -> str:
    """Why ``linear`` is not a static per-tensor FP8 linear on FlashInfer's BMM ("" when it is)."""
    from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp8LinearMethod

    method = getattr(linear, "quant_method", None)
    if (
        not isinstance(method, ModelOptFp8LinearMethod)
        or method.use_marlin
        or method.use_sm120_fp8
        or not linear.use_flashinfer_bmm
    ):
        return "not a ModelOpt FP8 linear on FlashInfer's BMM"
    w = _w8(linear)
    if (
        w.dtype != torch.float8_e4m3fn
        or not w.is_contiguous()
        or w.shape[1] % 64
        or w.shape[0] % 8
    ):
        return f"weight {w.dtype} {tuple(w.shape)}"
    if linear.weight_scale.numel() != 1 or linear.input_scale.numel() != 1:
        return "not per-tensor scales"
    if (
        linear.weight_scale.dtype != torch.float32
        or linear.input_scale.dtype != torch.float32
    ):
        return "scales are not FP32"
    if linear.bias is not None:
        return "a bias"
    if getattr(linear, "orig_dtype", torch.bfloat16) != torch.bfloat16:
        return f"output dtype {linear.orig_dtype} (low-M writes BF16)"
    return ""


def _bf16_unsupported(linear) -> str:
    w = linear.weight
    if getattr(linear, "quant_method", None) is not None and type(
        linear.quant_method
    ).__name__ != ("UnquantizedLinearMethod"):
        return f"{type(linear.quant_method).__name__}"
    if (
        w.dtype != torch.bfloat16
        or not w.is_contiguous()
        or w.shape[1] % 64
        or w.shape[0] % 8
    ):
        return f"weight {w.dtype} {tuple(w.shape)}"
    if getattr(linear, "bias", None) is not None:
        return "a bias"
    return ""


def build(model, ctx) -> Bound:
    """Slots ``lowm`` / ``lowm_se`` for every layer, or why a layer cannot take them; compiles every launch
    of the covered widths within the slots' rows."""
    from sglang.srt.models.qwen3_5 import (
        Qwen3_5AttentionDecoderLayer,
        Qwen3_5LinearDecoderLayer,
    )

    from sgl_nvidia_mega.qwen3_5 import plan
    from sgl_nvidia_mega.qwen3_5.runtime import SLOTS
    from sgl_nvidia_mega.qwen3_5.sm103 import lowm as sm

    want_in, want_out, want_se = (
        "lowm_in" in SLOTS,
        "lowm_out" in SLOTS,
        "lowm_se" in SLOTS,
    )
    se_rows = plan.LOWM_SE_MAX_ROWS
    se_tep8: tuple = ()
    if want_se and {"route", "route_topk"} <= SLOTS:
        from sglang.srt.runtime_context import get_parallel

        par = get_parallel()
        tep8_m = (
            par.tp_size == 8
            and par.moe_ep_size > 1
            and {"route", "route_topk"} <= SLOTS
        )
        se_rows = plan.lowm_se_pre_rows(par.tp_size, tep8_m) or se_rows
        if tep8_m and "lowm_se_tep8" in SLOTS:
            wide = "lowm_se_wide" in SLOTS
            se_tep8 = (TEP8_SE_ROWS[0], TEP8_SE_WIDE_HI if wide else TEP8_SE_ROWS[1])
    _configure_tep8_se(se_tep8, "lowm_se_coresident" in SLOTS)
    want_push = want_out and "lowm_push" in SLOTS and "hop" in SLOTS
    want_fp8 = "hop_fp8" in SLOTS and "hop" in SLOTS
    fp8_rows = plan.HOP_MID_MAX_ROWS if "hop_mid" in SLOTS else plan.HOP_MAX_ROWS
    want = want_in or want_out
    if not (want or want_se or want_fp8):
        return Bound(reason="stock (SGLANG_NVIDIA_MEGAKERNEL_SLOTS)")
    m = model.model
    dev = torch.cuda.current_device()
    layers = [(i, m.layers[i]) for i in range(m.start_layer, m.end_layer)]
    launches = set()  # (m, k, n8, n16, act, epi8, out2_f32)
    chunked = set()
    if want:
        for i, layer in layers:
            if isinstance(layer, Qwen3_5LinearDecoderLayer):
                gdn = layer.linear_attn
                for lin in (gdn.in_proj_qkvz, gdn.out_proj):
                    reason = fp8_unsupported(lin)
                    if reason:
                        return Bound(reason=f"lowm: layer {i}: {reason}")
                reason = _bf16_unsupported(gdn.in_proj_ba)
                if reason:
                    return Bound(reason=f"lowm: layer {i} in_proj_ba: {reason}")
                n8, k = _w8(gdn.in_proj_qkvz).shape
                n16 = gdn.in_proj_ba.weight.shape[0]
                for w in ctx.widths:
                    if want_in and w <= plan.LOWM_IN_MAX_ROWS:
                        launches.add((w, k, n8, n16, 1, 0, False))
                        if want_fp8 and w <= plan.HOP_MAX_ROWS:
                            launches.add((w, k, n8, n16, 0, 0, False))
                    if want_out and w <= plan.LOWM_OUTPROJ_MAX_ROWS:
                        n, ko = _w8(gdn.out_proj).shape
                        launches.add((w, ko, n, 0, 0, 0, False))
            elif isinstance(layer, Qwen3_5AttentionDecoderLayer):
                attn = layer
                if not attn.attn_output_gate:
                    return Bound(reason=f"lowm: layer {i}: no output gate")
                for lin in (attn.qkv_proj, attn.o_proj):
                    reason = fp8_unsupported(lin)
                    if reason:
                        return Bound(reason=f"lowm: layer {i}: {reason}")
                for w in ctx.widths:
                    if want_in and w <= plan.LOWM_IN_MAX_ROWS:
                        n, k = _w8(attn.qkv_proj).shape
                        launches.add((w, k, n, 0, 1, 0, False))
                        if want_fp8 and w <= plan.HOP_MAX_ROWS:
                            launches.add((w, k, n, 0, 0, 0, False))
                    if want_out and w <= plan.LOWM_OPROJ_MAX_ROWS:
                        n, k = _w8(attn.o_proj).shape
                        launches.add((w, k, n, 0, 1, 0, False))
                        # MKA's front path hands o_proj its gate's static FP8 output
                        launches.add((w, k, n, 0, 0, 0, False))
    shared = {}
    if want_se:
        for i, layer in layers:
            mlp = getattr(layer, "mlp", None)
            se = getattr(mlp, "shared_expert", None)
            gate = getattr(mlp, "shared_expert_gate", None)
            if se is None or gate is None:
                return Bound(reason=f"lowm_se: layer {i}: no gated shared expert")
            for lin in (se.gate_up_proj, se.down_proj):
                reason = fp8_unsupported(lin)
                if reason:
                    return Bound(reason=f"lowm_se: layer {i}: {reason}")
            if type(se.act_fn).__name__ != "SiluAndMul":
                return Bound(reason=f"lowm_se: layer {i}: {type(se.act_fn).__name__}")
            gw = gate.weight
            if (
                gw.dtype != torch.bfloat16
                or gw.shape[0] != 1
                or getattr(gate, "bias", None) is not None
            ):
                return Bound(
                    reason=f"lowm_se: layer {i}: shared gate {gw.dtype} {tuple(gw.shape)}"
                )
            gu = _w8(se.gate_up_proj)
            if gu.shape[0] % 16:
                return Bound(reason=f"lowm_se: layer {i}: gate_up rows {gu.shape[0]}")
            gate8 = torch.zeros((8, gw.shape[1]), dtype=gw.dtype, device=gw.device)
            gate8[:1].copy_(gw)
            shared[id(mlp)] = (sm.swiglu_pairs(gu), gate8)
            n, k = gu.shape
            nd, kd = _w8(se.down_proj).shape
            for w in ctx.widths:
                if w > sm.MAX_ROWS and _in(se_tep8, w):
                    t1 = _tep8_se_tactics(se_tep8, w, 8, 1)
                    t2 = _tep8_se_tactics(se_tep8, w, 0, 2)
                    for a, b, trig_gu, trig_dn in se_launches(se_tep8, w):
                        chunked.add((b - a, k, n, 8, 1, 1, True, t1, trig_gu))
                        chunked.add((b - a, kd, nd, 0, 0, 2, False, t2, trig_dn))
                elif w <= se_rows or _in(se_tep8, w):
                    launches.add((w, k, n, 8, 1, 1, True))
                    if want_fp8 and w <= fp8_rows:
                        launches.add((w, k, n, 8, 0, 1, True))
                    launches.add((w, kd, nd, 0, 0, 2, False))
    for w, k, n8, n16, act, epi8, f32 in sorted(launches):
        tactic = _tep8_se_tactics(se_tep8, w, n16, epi8) or TACTICS.get(
            (n8, n16, k, w, act == 0)
        )
        sm.prepare(
            dev,
            w,
            k,
            n8,
            n16,
            act,
            epi8,
            f32,
            tactic=tactic,
            trigger=_tep8_se_trigger(se_tep8, w) if n16 and epi8 == 1 else True,
        )
        if (
            want_push
            and n8 == HIDDEN
            and not n16
            and epi8 == 0
            and w <= plan.HOP_MAX_ROWS
        ):
            sm.prepare(dev, w, k, n8, n16, act, epi8, f32, tactic=tactic, push=True)
    for w, k, n8, n16, act, epi8, f32, tactic, trig in sorted(chunked, key=str):
        sm.prepare(dev, w, k, n8, n16, act, epi8, f32, tactic=tactic, trigger=trig)
    logger.info(
        "NVIDIA megakernel: Qwen3.5 low-M launches built: %s",
        sorted((w, n8, n16, k) for w, k, n8, n16, *_ in launches),
    )
    if se_tep8:
        logger.info(
            "NVIDIA megakernel: Qwen3.5 slot lowm_se_tep8 serves %d-%d rows (co-resident rows %s; "
            "%d row-chunked launches above %d rows)",
            *se_tep8,
            tep8_se_config()["coresident"] or "none",
            len(chunked),
            sm.MAX_ROWS,
        )
    return Bound(
        lowm_in=want_in,
        lowm_out=want_out,
        se=want_se,
        se_pre=want_se and se_rows > plan.LOWM_SE_MAX_ROWS,
        se_tep8=se_tep8,
        push=want_push,
        hop_fp8=want_fp8,
        shared=shared,
    )


def _gemm(rows: int, tactic=None, **kwargs):
    from sgl_nvidia_mega.qwen3_5.sm103 import lowm as sm

    fp8_in = kwargs.get("x8") is not None and kwargs.get("w8") is not None
    xa = kwargs["x8"] if fp8_in else kwargs["x16"]
    n8 = 0 if kwargs.get("w8") is None else kwargs["w8"].shape[0]
    n16 = 0 if kwargs.get("w16") is None else kwargs["w16"].shape[0]
    if tactic is None:
        tactic = TACTICS.get((n8, n16, xa.shape[1], rows, fp8_in))
    return sm.gemm(tactic=tactic, **kwargs)


def _in(rows_range: tuple, rows: int) -> bool:
    return bool(rows_range) and rows_range[0] <= rows <= rows_range[1]


def _tep8_se_trigger(se_tep8: tuple, rows: int) -> bool:
    if not _in(se_tep8, rows):
        return True
    cfg = tep8_se_config()
    if not _in(cfg["coresident"], rows):
        return True
    return tep8_se_down_smem(cfg["t2"], rows) > TEP8_SE_CORESIDENT_SMEM


def se_launches(se_tep8: tuple, rows: int, tail: int = 0) -> tuple:
    from sgl_nvidia_mega.qwen3_5.sm103 import lowm as sm

    if rows <= sm.MAX_ROWS or not _in(se_tep8, rows):
        return ((0, rows, _tep8_se_trigger(se_tep8, rows), True),)
    tail = int(tail or TEP8_SE_TAIL_ROWS)
    cuts = [0] + list(range(sm.MAX_ROWS, rows, tail)) + [rows]
    cfg = tep8_se_config()
    out = []
    for a, b in zip(cuts, cuts[1:]):
        n = b - a
        # within a pair: the co-residency rule at the chunk's rows
        late = (
            _in(cfg["coresident"], n)
            and tep8_se_down_smem(cfg["t2"], n) <= TEP8_SE_CORESIDENT_SMEM
        )
        out.append((a, b, not late, b == rows))
    return tuple(out)


def _tep8_se_tactics(se_tep8: tuple, rows: int, n16: int, epi8: int):
    if not _in(se_tep8, rows):
        return None
    cfg = tep8_se_config()
    if n16 and epi8 == 1:
        return cfg["t1"]
    if not n16 and epi8 == 2:
        return cfg["t2"]
    return None


def in_proj(gdn, x: torch.Tensor, x8=None):
    """``Qwen3_5GatedDeltaNet._forward_input_proj`` in one launch: (qkvz, ba) of the normed rows ``x`` (with
    ``x8``, in_proj_qkvz reads the hop's E4M3 rows)."""
    rows = x.shape[0]
    qkvz_lin, ba_lin = gdn.in_proj_qkvz, gdn.in_proj_ba
    w8 = _w8(qkvz_lin)
    qkvz = torch.empty((rows, w8.shape[0]), dtype=x.dtype, device=x.device)
    ba = torch.empty((rows, ba_lin.weight.shape[0]), dtype=x.dtype, device=x.device)
    engaged(
        "lowm_in", "in_proj_qkvz+ba" + ("(e4m3 in)" if x8 is not None else ""), rows
    )
    _gemm(
        rows,
        x8=x8,
        x16=x,
        w8=w8,
        w_scale=qkvz_lin.weight_scale,
        in_scale=qkvz_lin.input_scale,
        w16=ba_lin.weight,
        out=qkvz,
        out2=ba,
    )
    return qkvz, ba


def in_proj_fp8(gdn, x: torch.Tensor, x8: torch.Tensor, ba_tactic=None):
    """``Qwen3_5GatedDeltaNet._forward_input_proj`` of a captured step with in_proj_qkvz on the hop's E4M3 rows
    (FlashInfer's BMM, no quantize kernel) and in_proj_ba on the alternate stream, as SGLang orders them."""
    qkvz, ba, join = in_proj_fp8_deferred(gdn, x, x8, "hop_fp8", ba_tactic)
    join()
    return qkvz, ba


def in_proj_fp8_deferred(
    gdn, x: torch.Tensor, x8: torch.Tensor, slot: str, ba_tactic=None
):
    from sgl_nvidia_mega.qwen3_5 import gemm_tactics

    lin = gdn.in_proj_qkvz
    engaged(slot, "in_proj_qkvz bmm", x.shape[0])
    current = torch.cuda.current_stream()
    alt = gdn.alt_stream
    if alt is not None:
        alt.wait_stream(current)
    qkvz = gemm_tactics.bmm(x8, lin.weight, lin.input_scale, lin.weight_scale, x.dtype)
    if alt is None:
        return qkvz, in_proj_ba(gdn, x, ba_tactic), lambda: None
    with torch.cuda.stream(alt):
        ba = in_proj_ba(gdn, x, ba_tactic)
    return qkvz, ba, lambda: current.wait_stream(alt)


def reads_fp8(st) -> bool:
    return st.slots.hop_fp8 or st.slots.bt_tail


def in_proj_pinned(gdn, x: torch.Tensor, ba_tactic=None):
    from sgl_nvidia_mega.qwen3_5 import gemm_tactics

    if ba_tactic is None and not gemm_tactics.enabled():
        return None
    lin = gdn.in_proj_qkvz
    if x.dtype != torch.bfloat16 or fp8_unsupported(lin):
        return None
    rows = x.shape[0]
    pinned = gemm_tactics.entry(rows, _w8(lin).shape[0], x.shape[1]) is not None
    if not pinned and ba_tactic is None:
        return None
    from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8
    from sglang.srt.models.qwen3_5 import _gdn_use_alt_stream, get_is_capture_mode

    current = torch.cuda.current_stream()
    alt = gdn.alt_stream if _gdn_use_alt_stream and get_is_capture_mode() else None
    if alt is not None:
        alt.wait_stream(current)
    if pinned:
        engaged(gemm_tactics.OPTION, "in_proj_qkvz", rows)
    x8, _ = static_quant_fp8(x, lin.input_scale, repeat_scale=False)
    qkvz = gemm_tactics.bmm(x8, lin.weight, lin.input_scale, lin.weight_scale, x.dtype)
    if alt is None:
        return qkvz, in_proj_ba(gdn, x, ba_tactic)
    with torch.cuda.stream(alt):
        ba = in_proj_ba(gdn, x, ba_tactic)
    current.wait_stream(alt)
    return qkvz, ba


BA_TACTIC = (4, 1, 8)
BA_N, BA_K = 16, 4096


def ba_unsupported(linear) -> str:
    reason = _bf16_unsupported(linear)
    if reason:
        return reason
    if tuple(linear.weight.shape) != (BA_N, BA_K):
        return f"weight {tuple(linear.weight.shape)}"
    return ""


def ba_tactic(st, rows: int):
    """Slot lowm_ba's tactic for in_proj_ba at ``rows`` (compiled at the build), else None."""
    return st.bound["gdn"].ba_lowm.get(rows) if st.slots.lowm_ba else None


def prepare_ba(device_index: int, rows: int):
    from sgl_nvidia_mega.qwen3_5.sm103 import lowm as sm

    kw, nt16, ksplit = BA_TACTIC
    tactic = sm.Tactic(kw, 1, nt16, 1, ksplit)
    return sm.prepare(device_index, rows, BA_K, 0, BA_N, 1, 0, False, tactic=tactic)


def in_proj_ba(gdn, x: torch.Tensor, tactic=None) -> torch.Tensor:
    lin = gdn.in_proj_ba
    if tactic is None or x.stride(1) != 1 or x.stride(0) % 8 or x.data_ptr() % 16:
        return lin(x)[0]
    from sgl_nvidia_mega.qwen3_5.sm103 import lowm as sm

    rows = x.shape[0]
    ba = torch.empty((rows, lin.weight.shape[0]), dtype=x.dtype, device=x.device)
    engaged("lowm_ba", "in_proj_ba", rows)
    sm.gemm(x16=x, w16=lin.weight, out2=ba, tactic=tactic)
    return ba


def stash_fp8(st, x: torch.Tensor, x8: torch.Tensor) -> None:
    """Slot hop_fp8: the hop's E4M3 copy ``x8`` of the norm rows ``x`` for their next projection."""
    st.scratch["lowm.fp8"] = (x.data_ptr(), x.shape[0], x8)


def clear_fp8(st) -> None:
    st.scratch.pop("lowm.fp8", None)


def take_fp8(st, x: torch.Tensor):
    held = st.scratch.pop("lowm.fp8", None)
    if held is None or held[0] != x.data_ptr() or held[1] != x.shape[0]:
        return None
    return held[2]


def layer_input_fp8_scale(layer, st, rows: int):
    from sglang.srt.models.qwen3_5 import Qwen3_5LinearDecoderLayer

    from sgl_nvidia_mega.qwen3_5.attention import mka

    if not reads_fp8(st):
        return None
    if isinstance(layer, Qwen3_5LinearDecoderLayer):
        if not st.slots.gdn_verify:
            return None
        if st.slots.gdn_front and not st.slots.gdn_trt_front:
            return None
        return layer.linear_attn.in_proj_qkvz.input_scale
    if st.slots.mka and rows <= mka.FUSED_MAX_ROWS:
        return None
    return layer.qkv_proj.input_scale


def moe_input_fp8_scale(layer, st):
    """The shared expert's gate_up input scale where its low-M launch reads the MoE input (slot hop_fp8)."""
    if not (st.slots.hop_fp8 and st.slots.lowm_se) or st.slots.route_shared:
        return None
    return layer.mlp.shared_expert.gate_up_proj.input_scale


def _symmetric_empty(
    rows: int, n: int, like: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """A TP-partial output in SGLang's symmetric pool (what RowParallelLinear allocates under it)."""
    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        use_symmetric_memory,
    )
    from sglang.srt.layers.dp_attention import is_allocation_symmetric
    from sglang.srt.runtime_context import get_parallel

    with use_symmetric_memory(
        get_parallel().tp_group, disabled=not is_allocation_symmetric()
    ):
        return torch.empty((rows, n), dtype=dtype, device=like.device)


def row_parallel(
    linear, x: torch.Tensor, dtype: torch.dtype, what: str, hop=None
) -> torch.Tensor:
    """``RowParallelLinear.forward`` of a static FP8 ``linear`` (no reduce, no bias) on BF16 ``x`` (quantized
    in registers) or its E4M3 input: the TP-partial output [rows, N] in the symmetric pool. With ``hop``
    (slot lowm_push, where the MoE input's all-reduce runs the hop) the launch also pushes the rows into the
    hop's mailbox and the all-reduce only consumes them."""
    rows = x.shape[0]
    w8 = _w8(linear)
    out = _symmetric_empty(rows, w8.shape[0], x, dtype)
    engaged("lowm_push" if hop is not None else "lowm_out", what, rows)
    act = {"x8": x} if x.dtype == torch.float8_e4m3fn else {"x16": x}
    _gemm(
        rows,
        w8=w8,
        w_scale=linear.weight_scale,
        in_scale=linear.input_scale,
        out=out,
        push_site=None if hop is None else hop.site,
        **act,
    )
    if hop is not None:
        hop.mark_pushed(out)
    return out


def prepare_qkv_gate(
    layer,
    positions: torch.Tensor,
    x: torch.Tensor,
    x8=None,
    low_m: bool = True,
    slot: str = "hop_fp8",
):
    """``Qwen3_5AttentionDecoderLayer.forward_prepare_cuda_fused`` with qkv_proj on the low-M launch (slot
    lowm_in; with ``x8`` on the hop's E4M3 rows), or FlashInfer's BMM on ``x8`` (slot hop_fp8)."""
    from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import (
        fused_qk_gemma_rmsnorm_rope_gate,
    )

    from sgl_nvidia_mega.qwen3_5 import gemm_tactics

    rows = x.shape[0]
    lin = layer.qkv_proj
    if low_m:
        w8 = _w8(lin)
        qkv = torch.empty((rows, w8.shape[0]), dtype=x.dtype, device=x.device)
        engaged("lowm_in", "qkv_proj" + ("(e4m3 in)" if x8 is not None else ""), rows)
        act = {"x8": x8} if x8 is not None else {"x16": x}
        _gemm(
            rows,
            w8=w8,
            w_scale=lin.weight_scale,
            in_scale=lin.input_scale,
            out=qkv,
            **act,
        )
    else:
        engaged(slot, "qkv_proj bmm", rows)
        qkv = gemm_tactics.bmm(
            x8, lin.weight, lin.input_scale, lin.weight_scale, x.dtype
        )
    if layer.attn_output_gate:
        q_gate, k, v = qkv.split(
            [layer.q_size * 2, layer.kv_size, layer.kv_size], dim=-1
        )
    else:
        q_gate, k, v = qkv.split([layer.q_size, layer.kv_size, layer.kv_size], dim=-1)
    q_out, k_out, gate_out = fused_qk_gemma_rmsnorm_rope_gate(
        q_gate,
        k,
        layer.q_norm.weight.data,
        layer.k_norm.weight.data,
        layer.rotary_emb.cos_sin_cache,
        positions,
        layer.q_norm.variance_epsilon,
        layer.num_heads,
        layer.num_kv_heads,
        layer.head_dim,
        layer.rotary_emb.rotary_dim,
        has_gate=layer.attn_output_gate,
        mrope_axis_map=(layer.rotary_emb.axis_map if positions.dim() == 2 else None),
    )
    q = q_out.view(rows, -1)
    k = k_out.view(rows, -1)
    gate = gate_out.view(rows, -1) if gate_out is not None else None
    return q, k, v, gate


def shared_expert(mlp, x: torch.Tensor, bound: Bound, x8=None) -> torch.Tensor:
    """The gated shared expert (``Qwen2MoeMLP.forward`` then ``_gate_shared_output_out_of_place``) in two
    launches: [rows, hidden] BF16 (with ``x8``, gate_up reads the hop's E4M3 rows)."""
    rows = x.shape[0]
    se = mlp.shared_expert
    gu, dn = se.gate_up_proj, se.down_proj
    pairs, gate8 = bound.shared[id(mlp)]
    inter = pairs.shape[0] // 2
    h8 = torch.empty((rows, inter), dtype=torch.float8_e4m3fn, device=x.device)
    logits = torch.empty((rows, 8), dtype=torch.float32, device=x.device)
    engaged("lowm_se", "shared_expert" + ("(e4m3 in)" if x8 is not None else ""), rows)
    t1 = _tep8_se_tactics(bound.se_tep8, rows, 8, 1)
    t2 = _tep8_se_tactics(bound.se_tep8, rows, 0, 2)
    pairs_at = se_launches(bound.se_tep8, rows)
    if t1 is not None:
        engaged(
            "lowm_se_tep8",
            f"shared_expert ({len(pairs_at)} launch pair(s), gate_up released "
            f"{'after its weight loads' if pairs_at[0][2] else 'at its end'})",
            rows,
        )
    w8 = _w8(dn)
    out = torch.empty((rows, w8.shape[0]), dtype=x.dtype, device=x.device)
    for a, b, trig_gu, trig_dn in pairs_at:
        _gemm(
            b - a,
            tactic=t1,
            x8=None if x8 is None else x8[a:b],
            x16=x[a:b],
            w8=pairs,
            w_scale=gu.weight_scale,
            in_scale=gu.input_scale,
            w16=gate8,
            out=h8[a:b],
            out2=logits[a:b],
            swiglu_scale=dn.input_scale,
            trigger=trig_gu,
        )
        _gemm(
            b - a,
            tactic=t2,
            x8=h8[a:b],
            w8=w8,
            w_scale=dn.weight_scale,
            in_scale=dn.input_scale,
            out=out[a:b],
            gate=logits[a:b, 0],
            trigger=trig_dn,
        )
    return out

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/layers/quantization/sm107_bf16_smallm.py (NONBITWISE_ROUTES,
# NONBITWISE_LOWM_ROUTES, _cluster_out, route_for, gemm, _check_nonbitwise),
# python/sglang/srt/layers/quantization/sm107_shared_expert_act.py (_ineligible,
# _prepared, fused_act, precompile) and python/sglang/srt/models/qwen2_moe.py
# (Qwen2MoeMLP.forward with its TGV down_proj).
# Changed: the environment switches are ``Options``. The routes serve the executor's
# own projection calls, not SGLang's BF16 dispatch; the source's bitwise route table
# under them is runtime/sm107_bf16_smallm.py, on its own switch, which this slot does
# not wait for. The split-K runs on SGLang's epilogue copy of the kernel at its own K
# tile, and the static eligibility checks run once, at build.
"""Small-M BF16 GEMMs of a covered step on SM107 at TP1 (slot ``smallm_gemm``).

``out_proj``: the GDN out_proj and the attention o_proj (N 2560, K 6144) at 4-16 and
20-32 rows run on the cluster split-K kernel (this package's copy of FlashInfer's,
with FlashInfer #7443) as four K slices of 1536. That is 160 CTAs (320 with mma_n 16
at 9-16 rows), where SGLang's dispatch runs TGV on 40 CTAs (4-16 rows) or 80 (20-32).
The source measured 5.5 against 7.4 us a call at 4-8 rows and 5.7 against 7.4 at
9-16 on SM107.

``shared_expert``: ``Qwen2MoeMLP.forward`` of a shared expert (gate_up N 1280 K 2560,
down N 2560 K 640). At 1-16 rows its gate_up GEMM and act_and_mul run as one kernel
(``kernels/ops/gemm/splitk_act_mul.py``). On the target, at 4, 8 and 16 rows, its
down_proj runs on SGLang's TGV kernel instead of cuBLAS.

Numerics, against SGLang's dispatch (its cutedsl BF16 backend):

* out_proj is not bitwise: the FP32 partials are added in another order. ``build``
  checks each row count on random operands by relative L2 and leaves a row count past
  ``Options.rel_l2_max`` to SGLang. The source measured 1.2-1.7e-4 at 20-32 rows; one
  BF16 ulp on every element is about 4e-3.
* the fused act: ``build`` checks it bit for bit against SGLang's gate_up and
  SiluAndMul at 1-16 rows on random operands, and at the edge pair (gate -87.5,
  up -80) where only the fast-math divide gives the stock result. On a mismatch
  SGLang's pair stays. Its act is SGLang's act_and_mul as built before #48942, which
  the current kernel differs from on about 2% of all finite (gate, up) pairs
  (``splitk_act_mul``); none of them is among the check's operands.
* the TGV down_proj: TGV does not split K. There is no bit claim against cuBLAS.

``build`` compiles every route before any capture. A step function returns None when
it does not serve the call, and the caller then runs SGLang's module.
"""

from __future__ import annotations

import logging
import os
import types
import weakref
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Optional

import torch

from sgl_nvidia_mega.qwen3_8.runtime import bf16_gemm_pins

logger = logging.getLogger(__name__)

OUT_PROJ = (2560, 6144)  # (N, K): the GDN out_proj and the attention o_proj
INTER, HIDDEN = 640, 2560  # the shared expert
GATE_UP, DOWN = (2 * INTER, HIDDEN), (HIDDEN, INTER)
ACT_MAX_ROWS = 16
SM107 = (10, 7)

SGLANG_READS = {
    "srt/layers/quantization/unquant.py": (
        "UnquantizedLinearMethod",
        "get_bf16_gemm_backend",
        "is_cutedsl",
        "_bf16_gemm_dispatch_impl",
    ),
    "srt/runtime_context.py": ("get_parallel",),
    "srt/models/qwen4_exp.py": ("Qwen4ExpLinearDecoderLayer",),
    "srt/models/qwen3_5.py": ("linear_attn", "out_proj", "o_proj"),
    "srt/models/qwen2_moe.py": (
        "Qwen2MoeSparseMoeBlock",
        "shared_expert",
        "gate_up_proj",
        "down_proj",
        "act_fn",
        "_enable_silu_fp4_quant_fusion",
    ),
    "srt/layers/linear.py": ("quant_method", "gather_output"),
    "srt/layers/activation.py": ("SiluAndMul", "forward_cuda", "silu_and_mul"),
    "kernels/fused_op.py": (
        "get_fused_op_backend",
        "_forward_method",
        "_resolve_forward_method",
    ),
    "kernels/ops/gemm/cutedsl_bf16_gemm.py": ("cutedsl_bf16_gemm",),
    # sgl-project/sglang#50194's split-K epilogue GEMM
    "kernels/ops/gemm/dense_bf16_gemm_sm100_splitk_epilogue.py": (
        "SplitKTactic",
        "validate_tactic",
        "_get_compiled_splitk_kernel",
        "_to_cute_swap",
        "_mbarrier_wait_acquire_cluster",
    ),
}

# shared_expert runs this body (its NVFP4 branch is refused at build)
SGLANG_MIRRORS = {
    "srt/models/qwen2_moe.py:Qwen2MoeMLP.forward": "60b6d1e897d94ed9",
}

# Reads and mirrors that need an SGLang change newer than the unit job's pin: the
# contract test skips each, naming the change, only where the installed SGLang lacks it.
SGLANG_REQUIRES = {
    "kernels/ops/gemm/dense_bf16_gemm_sm100_splitk_epilogue.py:_mbarrier_wait_acquire_cluster": "sglang#50194",
}


@dataclass(frozen=True)
class Options:
    """The source's switches as values, with its tuned routes."""

    # SGLANG_ENABLE_SM107_BF16_SMALLM_NONBITWISE_LOWM_ROUTES=1: out_proj / o_proj rows
    # (first, last) on the split-K tactic (mma_m, mma_n, split_k, ab_stages)
    out_proj_low_m: tuple = (((4, 8), (64, 8, 4, 8)), ((9, 16), (64, 16, 4, 10)))
    # SGLANG_ENABLE_SM107_BF16_SMALLM_NONBITWISE_ROUTES=1: the same at 20-32 rows
    out_proj_mid_m: tuple = (((20, 32), (64, 32, 4, 8)),)
    # build's bound on a route's relative L2 distance to SGLang's dispatch
    rel_l2_max: float = 1e-3
    shared_fused_act: bool = True  # SGLANG_ENABLE_SM107_SHARED_EXPERT_FUSED_ACT=1
    # SGLANG_ENABLE_SHARED_EXPERT_DOWN_TGV=1: the target's shared-expert down_proj
    # rows on TGV; () keeps SGLang's
    shared_down_tgv_rows: tuple = (4, 8, 16)

    def __post_init__(self):
        rows = [
            m
            for (a, b), _ in self.out_proj_low_m + self.out_proj_mid_m
            for m in range(a, b + 1)
        ]
        if len(rows) != len(set(rows)):
            raise ValueError("the out_proj routes' row ranges overlap")


DEFAULT = Options()

# The source's switches, read under the names the recipes pass (its own names; the
# routes' switch is sm107_bf16_smallm.ENV). Its defaults are off.
NONBITWISE_LOWM_ENV = "SGLANG_ENABLE_SM107_BF16_SMALLM_NONBITWISE_LOWM_ROUTES"
NONBITWISE_ENV = "SGLANG_ENABLE_SM107_BF16_SMALLM_NONBITWISE_ROUTES"
FUSED_ACT_ENV = "SGLANG_ENABLE_SM107_SHARED_EXPERT_FUSED_ACT"
DOWN_TGV_ENV = "SGLANG_ENABLE_SHARED_EXPERT_DOWN_TGV"


def _env_on(name: str) -> bool:
    return os.environ.get(name, "false").strip().lower() in ("1", "true", "yes", "y")


def recipe_options() -> Options:
    """``Options`` as the recipe's switches set them, with the source's gating: the
    non-bitwise out_proj routes and the fused act need the bitwise routes' switch on."""
    from sgl_nvidia_mega.qwen3_8.runtime import sm107_bf16_smallm

    routes = _env_on(sm107_bf16_smallm.ENV)
    return Options(
        out_proj_low_m=DEFAULT.out_proj_low_m
        if routes and _env_on(NONBITWISE_LOWM_ENV)
        else (),
        out_proj_mid_m=DEFAULT.out_proj_mid_m
        if routes and _env_on(NONBITWISE_ENV)
        else (),
        shared_fused_act=routes and _env_on(FUSED_ACT_ENV),
        shared_down_tgv_rows=DEFAULT.shared_down_tgv_rows
        if _env_on(DOWN_TGV_ENV)
        else (),
    )


# Process-wide, as the source's: what build resolved, checked and prepared.
_k = None  # the kernels (_kernels)
# (rows, tactic) -> the SplitKTactic that passed the out_proj check, or None
_passed: dict = {}
_projs = weakref.WeakSet()  # the out_proj / o_proj modules a route serves
_act_passed: Optional[bool] = None
# gate_up_proj -> (weight data_ptr, row-permuted weight): the shared experts the act serves
_prepared_weights = weakref.WeakKeyDictionary()
# device index -> (16, 1280) BF16 shape anchor of the act kernel's C operand (never written)
_anchors: dict = {}
_tgv_downs = weakref.WeakSet()  # the shared-expert down_proj modules TGV serves


@lru_cache(maxsize=None)
def _out_proj_tactics(options: Options) -> dict:
    """rows -> (mma_m, mma_n, split_k, ab_stages) of the options' out_proj routes."""
    return {
        m: tactic
        for (first, last), tactic in options.out_proj_low_m + options.out_proj_mid_m
        for m in range(first, last + 1)
    }


def _kernels():
    import cuda.bindings.driver as cuda_driver
    import sglang.kernels.ops.gemm.dense_bf16_gemm_sm100_splitk_epilogue as epi
    from sglang.kernels.ops.gemm.cutedsl_bf16_gemm import cutedsl_bf16_gemm

    from sgl_nvidia_mega.qwen3_8.kernels.ops.gemm import splitk_act_mul as act

    if not hasattr(epi, "_mbarrier_wait_acquire_cluster"):
        raise ImportError(
            "SGLang's split-K epilogue GEMM lacks FlashInfer #7443's cluster ordering "
            "(sgl-project/sglang#50194)"
        )

    return types.SimpleNamespace(
        epi=epi, act=act, tgv=cutedsl_bf16_gemm, stream=cuda_driver.CUstream
    )


# -- build -------------------------------------------------------------------------


def build(model, options: Options = DEFAULT) -> str:
    """'' when a route serves ``model`` (target or MTP draft), compiled and checked
    against SGLang's dispatch, or why the slot stays stock."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != SM107:
        return "the routes are tuned on SM107"
    from sglang.srt.layers.quantization.unquant import (
        _bf16_gemm_dispatch_impl,
        get_bf16_gemm_backend,
    )
    from sglang.srt.runtime_context import get_parallel

    if get_parallel().tp_size != 1:
        return "TP1 only: a route replaces the whole row-parallel projection"
    if not get_bf16_gemm_backend().is_cutedsl():
        return (
            "the routes are checked against SGLang's cutedsl BF16 dispatch, not in use"
        )
    projs, mlps = _layers(model)
    return install(
        projs,
        mlps,
        options,
        device=torch.device("cuda", torch.cuda.current_device()),
        stock_gemm=lambda x, w: _bf16_gemm_dispatch_impl(x, w, None),
    )


def _layers(model) -> tuple:
    """The model's GDN out_proj / attention o_proj and its shared experts."""
    from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock
    from sglang.srt.models.qwen4_exp import Qwen4ExpLinearDecoderLayer

    m = model.model
    projs, mlps = [], []
    for i in range(m.start_layer, m.end_layer):
        layer = m.layers[i]
        if isinstance(layer, Qwen4ExpLinearDecoderLayer):
            projs.append(layer.linear_attn.out_proj)
        else:
            projs.append(layer.o_proj)
        if (
            isinstance(layer.mlp, Qwen2MoeSparseMoeBlock)
            and layer.mlp.shared_expert is not None
        ):
            mlps.append(layer.mlp.shared_expert)
    return projs, mlps


def install(projs, mlps, options: Options, *, device, stock_gemm: Callable) -> str:
    """Check and compile the routes for these modules (each check once a process);
    '' when any of them serves, else why none does."""
    global _k, _act_passed
    try:
        _k = _k or _kernels()
    except ImportError as e:
        return f"the split-K kernels do not import: {e}"
    served, why = [], []
    projs_ok = [p for p in projs if not _proj_reason(p)]
    if projs_ok:
        tactics = _out_proj_tactics(options)
        try:  # a route that does not compile keeps SGLang's GEMM
            _check_out_proj(tactics, options.rel_l2_max, stock_gemm, device)
        except Exception as e:
            why.append(f"out_proj: {type(e).__name__}: {e}")
        rows = sorted(m for m, t in tactics.items() if _passed.get((m, t)) is not None)
        if rows:
            _projs.update(projs_ok)
            served.append(f"out_proj split-K at rows {rows}")
    elif projs:
        why.append(f"out_proj: {_proj_reason(projs[0])}")
    use_act, tgv_rows = options.shared_fused_act, list(options.shared_down_tgv_rows)
    acts = [mlp for mlp in mlps if use_act and not _act_reason(mlp)]
    if acts:
        if _act_passed is None:
            try:
                _act_passed = _check_act(stock_gemm, device)
            except Exception as e:
                why.append(f"fused act: {type(e).__name__}: {e}")
                _act_passed = False
        if _act_passed:
            for mlp in acts:
                _prepared(mlp.gate_up_proj, device)
            served.append(f"the shared-expert fused act on {len(acts)} layers")
    elif use_act and mlps:
        why.append(f"fused act: {_act_reason(mlps[0])}")
    downs = [mlp.down_proj for mlp in mlps if tgv_rows and not _down_reason(mlp)]
    if downs:
        for rows in tgv_rows:  # compile TGV's pick for these rows before any capture
            _k.tgv(
                torch.zeros(rows, INTER, dtype=torch.bfloat16, device=device),
                downs[0].weight,
            )
        _tgv_downs.update(downs)
        served.append(f"the shared-expert down_proj on TGV at rows {tgv_rows}")
    elif tgv_rows and mlps:
        why.append(f"TGV down_proj: {_down_reason(mlps[0])}")
    if not served:
        return "no route serves this model" + (f" ({'; '.join(why)})" if why else "")
    logger.info("NVIDIA megakernel: Qwen3.8 smallm_gemm: %s", "; ".join(served))
    return ""


def _proj_reason(proj) -> str:
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    w = proj.weight
    if type(proj.quant_method) is not UnquantizedLinearMethod or proj.bias is not None:
        return "not an unquantized linear without bias"
    if w.dtype != torch.bfloat16 or tuple(w.shape) != OUT_PROJ:
        return f"weight {tuple(w.shape)} {w.dtype}, the routes take {OUT_PROJ} BF16"
    if not (w.is_contiguous() and w.data_ptr() % 32 == 0):
        return "weight not contiguous and 32-byte aligned"
    return ""


def _act_reason(mlp) -> str:
    """Why the stock pair of this shared expert is not (gate_up on SGLang's dispatch
    + SiluAndMul.forward_cuda) on a contiguous BF16 weight (the source's _ineligible)."""
    from sglang.kernels.fused_op import get_fused_op_backend
    from sglang.srt.layers.activation import SiluAndMul
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    gate_up = mlp.gate_up_proj
    w = gate_up.weight
    if mlp._enable_silu_fp4_quant_fusion:
        return "the NVFP4 silu fusion"
    if (
        type(gate_up.quant_method) is not UnquantizedLinearMethod
        or gate_up.bias is not None
        or gate_up.gather_output
    ):
        return "gate_up is not an unquantized linear without bias or gather"
    if w.dtype != torch.bfloat16 or tuple(w.shape) != GATE_UP or w.requires_grad:
        return f"gate_up weight {tuple(w.shape)} {w.dtype}, the kernel takes {GATE_UP} BF16"
    if not (w.is_contiguous() and w.data_ptr() % 32 == 0):
        return "gate_up weight not contiguous and 32-byte aligned"
    act_fn = mlp.act_fn
    if type(act_fn) is not SiluAndMul or get_fused_op_backend() is not None:
        return "act_fn is not SGLang's SiluAndMul"
    method = act_fn._forward_method
    if method is None:
        method = act_fn._resolve_forward_method()
    if not (
        isinstance(method, types.MethodType)
        and method.__func__ is SiluAndMul.forward_cuda
    ):
        return "act_fn does not run SiluAndMul.forward_cuda"
    return ""


def _down_reason(mlp) -> str:
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    down = mlp.down_proj
    if mlp._enable_silu_fp4_quant_fusion:
        return "the NVFP4 silu fusion"
    if (
        type(down.quant_method) is not UnquantizedLinearMethod
        or down.weight.dtype != torch.bfloat16
        or tuple(down.weight.shape) != DOWN
    ):
        return f"down_proj is not an unquantized {DOWN} BF16 linear"
    return ""


def _rel_l2(got: torch.Tensor, want: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(got.float() - want.float())
        / torch.linalg.vector_norm(want.float())
    )


def _check_out_proj(tactics: dict, bound: float, stock_gemm, device) -> None:
    """Validate, compile and check each unchecked (rows, tactic) against ``stock_gemm``
    on random operands; one past ``bound`` (relative L2) keeps SGLang's dispatch."""
    todo = sorted((m, t) for m, t in tactics.items() if (m, t) not in _passed)
    if not todo:
        return
    n, k = OUT_PROJ
    gen = torch.Generator(device=device).manual_seed(1)
    w = (torch.randn(n, k, generator=gen, device=device) * 0.02).to(torch.bfloat16)
    errors = {}
    for m, t in todo:
        tactic = _k.epi.SplitKTactic(*t)
        try:
            _k.epi.validate_tactic(tactic, m, n, k)
        except ValueError as e:
            logger.warning(
                "smallm_gemm: out_proj tactic %s at %d rows is invalid (%s); "
                "SGLang's dispatch serves it",
                t,
                m,
                e,
            )
            _passed[(m, t)] = None
            continue
        x = torch.randn(m, k, generator=gen, device=device).to(torch.bfloat16)
        out = torch.empty(m, n, dtype=torch.bfloat16, device=device)
        _splitk(x, w, out, tactic)
        errors[m] = err = _rel_l2(out, stock_gemm(x, w))
        # ``not <=`` also fails a NaN
        if not err <= bound:
            logger.warning(
                "smallm_gemm: out_proj split-K at %d rows is %.2e from SGLang's dispatch "
                "(relative L2, bound %.0e); SGLang's dispatch serves it",
                m,
                err,
                bound,
            )
        _passed[(m, t)] = tactic if err <= bound else None
    if errors:
        logger.info(
            "smallm_gemm: out_proj split-K relative L2 to SGLang's dispatch: at most %.2e",
            max(errors.values()),
        )


def _check_act(stock_gemm, device) -> bool:
    """The fused act bit for bit against ``stock_gemm`` then SiluAndMul: at 1-16 rows on
    random operands, and on one-hot rows that reach the epilogue with the edge pair
    (gate -87.5, up -80), where 1 + exp(-gate) passes 2^126 and only the fast-math
    divide gives the stock result."""
    from sglang.srt.layers.activation import silu_and_mul

    gen = torch.Generator(device=device).manual_seed(0)
    w = (torch.randn(*GATE_UP, generator=gen, device=device) * 0.02).to(torch.bfloat16)
    cases = [
        (
            (torch.randn(m, HIDDEN, generator=gen, device=device) * 4).to(
                torch.bfloat16
            ),
            w,
        )
        for m in range(1, ACT_MAX_ROWS + 1)
    ]
    edge = w.clone()
    edge[600, 5], edge[INTER + 600, 5] = -87.5, -80.0
    one_hot = torch.zeros(ACT_MAX_ROWS, HIDDEN, dtype=torch.bfloat16, device=device)
    rows = torch.arange(ACT_MAX_ROWS, device=device)
    # Row 5 then reaches the epilogue with exactly (gate, up) = edge[(600, 1240), 5].
    one_hot[rows, rows] = 1.0
    cases.append((one_hot, edge))
    anchor = _anchor(device)
    for x, weight in cases:
        m = x.shape[0]
        out = torch.empty(m, INTER, dtype=torch.bfloat16, device=device)
        got = _k.act.run(x, _k.act.permute_gate_up(weight), out, anchor[:m])
        want = silu_and_mul(stock_gemm(x, weight))
        if not torch.equal(got.view(torch.int16), want.view(torch.int16)):
            logger.warning(
                "smallm_gemm: the fused act differs from SGLang's gate_up + SiluAndMul "
                "at %d rows on this build; SGLang's pair serves",
                m,
            )
            return False
    return True


def _anchor(device) -> torch.Tensor:
    anchor = _anchors.get(device.index)
    if anchor is None:
        anchor = _anchors[device.index] = torch.empty(
            (ACT_MAX_ROWS, 2 * INTER), dtype=torch.bfloat16, device=device
        )
    return anchor


def _prepared(gate_up, device):
    """The row-permuted weight and the shape anchor, built outside graph capture (None
    while capturing)."""
    weight = gate_up.weight
    state = _prepared_weights.get(gate_up)
    anchor = _anchors.get(device.index)
    if state is not None and state[0] == weight.data_ptr() and anchor is not None:
        return state[1], anchor
    if torch.cuda.is_current_stream_capturing():
        return None
    if state is None or state[0] != weight.data_ptr():
        state = _prepared_weights[gate_up] = (
            weight.data_ptr(),
            _k.act.permute_gate_up(weight.detach()),
        )
    return state[1], _anchor(device)


# -- step functions ----------------------------------------------------------------


def _splitk(x_2d, weight, out, tactic) -> None:
    """``out = x_2d @ weight.T`` on the cluster split-K kernel (the source's
    ``_cluster_out`` at the kernel's own K tile)."""
    epi = _k.epi
    tensors = epi._to_cute_swap(x_2d, weight.T, out, None)
    compiled = epi._get_compiled_splitk_kernel(
        dtype=x_2d.dtype,
        tactic=tactic,
        use_pdl=True,
        has_bias=False,
        leading_dims=tensors[4],
    )
    compiled(
        *tensors[:3], _k.stream(torch.cuda.current_stream(x_2d.device).cuda_stream)
    )


def out_proj(executor, proj, x: torch.Tensor, st, op: str) -> Optional[torch.Tensor]:
    """``proj(x)[0]`` for a GDN out_proj or attention o_proj (TP1, no bias) on its
    split-K route, or None. ``op`` names it in the engagement record."""
    options = executor.smallm_gemm
    if options is None or x.dtype != torch.bfloat16:
        return None
    k = x.shape[-1]
    m = x.numel() // k
    tactic = _out_proj_tactics(options).get(m)
    route = None if tactic is None else _passed.get((m, tactic))
    if route is None or proj not in _projs:
        return None
    if not (x.is_contiguous() and x.data_ptr() % 32 == 0):
        return None
    out = torch.empty((m, OUT_PROJ[0]), dtype=x.dtype, device=x.device)
    _splitk(x.view(-1, k), proj.weight, out, route)
    st.ran(op, "splitk")
    return out.view(*x.shape[:-1], OUT_PROJ[0])


def _fused_act(options, mlp, x: torch.Tensor) -> Optional[torch.Tensor]:
    """silu(gate) * up of ``mlp.gate_up_proj(x)`` from one kernel, or None."""
    rows = x.shape[0]
    if not (
        _act_passed
        and options.shared_fused_act
        and 1 <= rows <= ACT_MAX_ROWS
        and mlp.gate_up_proj in _prepared_weights
        and x.is_contiguous()
        and x.data_ptr() % 32 == 0
    ):
        return None
    prepared = _prepared(mlp.gate_up_proj, x.device)
    if prepared is None:
        return None
    w_perm, anchor = prepared
    out = torch.empty((rows, INTER), dtype=x.dtype, device=x.device)
    return _k.act.run(x, w_perm, out, anchor[:rows])


def shared_expert(executor, mlp, x: torch.Tensor, st) -> Optional[torch.Tensor]:
    """``mlp(x)`` (``Qwen2MoeMLP.forward``) for a shared expert with the fused act
    and / or the TGV down_proj, or None when neither serves the call."""
    options = executor.smallm_gemm
    if options is None or x.dim() != 2 or x.dtype != torch.bfloat16:
        return None
    act = _fused_act(options, mlp, x)
    tgv = (
        not executor.draft
        and x.shape[0] in options.shared_down_tgv_rows
        and mlp.down_proj in _tgv_downs
    )
    if act is None and not tgv:
        return None
    impl = []
    if act is None:
        gate_up = bf16_gemm_pins.linear(mlp.gate_up_proj, x)
        act = mlp.act_fn(gate_up)
    else:
        impl.append("act_splitk")
    if tgv:
        out = _k.tgv(act, mlp.down_proj.weight)
        impl.append("down_tgv")
    else:
        out = bf16_gemm_pins.linear(mlp.down_proj, act)
    st.ran("moe.shared_expert", "+".join(impl))
    return out

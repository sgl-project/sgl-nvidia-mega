# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``sp_fp4_dispatch``: quantize the sequence-parallel MoE input to NVFP4 before the all-to-all dispatch.

Under ``--enable-layernorm-sp`` the Muse Spark routed NVFP4 expert layers of a prefill pass send each token's
normalized BF16 activation row to every rank that owns one of its experts (FlashInfer all-to-all, BF16 payload), and
each receiving rank then quantizes every received row to NVFP4 with per-token scaling (``nvfp4_quantize``,
``per_token_activation=True``) before the expert GEMMs. A token is received by up to four ranks, so its row crosses
the links up to three times at two bytes per element and is quantized up to four times.

This component quantizes each row once on the rank that holds it, with the same call and arguments, and dispatches
the packed NVFP4 codes, the block scales and the per-token scale (about a quarter of the BF16 bytes). The experts then
receive exactly the codes, scales and per-token scales the receiving rank would have computed from the same row, and
run the same kernel: the expert outputs are bitwise identical. BF16 expert layers and every non-prefill path keep the
stock BF16 dispatch.

Mechanics: the NVFP4 expert layer's sequence-parallel expert call sets a context flag; the sequence-parallel
dispatcher's ``dispatch`` sees it and sends the quantized payloads (largest element size first, as the all-to-all
requires) and returns the received codes and scales as the dispatch output; the trtllm-gen routed MoE call made for
that output gets the received per-token scales (the stock runner passes none for pre-quantized input).

Opt in with ``MUSE_VR_SP_FP4_DISPATCH=1`` (only meaningful with ``--enable-layernorm-sp``). Engagement is guarded by
sha256 pins on the SGLang and FlashInfer files the mechanics rely on.

Mode ``pfront`` (``MUSE_VR_PFRONT=1``, on top of this component): the layer's pre-expert RMSNorm is deferred into this
quantization and both run as one kernel (``ops/pfront.py``), bitwise to the norm -> bf16 -> per-token quantization
sequence, without writing the normalized bf16 rows. The norm is deferred only for a bf16 ``[T, 4096]`` input with a bf16
weight while this component's expert call is active; a dispatch that does not quantize applies the deferred norm
first (counted under ``pfront`` as ``stock_norm``). Guarded by sha256 pins on the torch RMSNorm and FlashInfer per-token
quantizer sources whose arithmetic the kernel reproduces; the 4-over-6 quantizer mode steps aside.
"""

from __future__ import annotations

import contextvars
import functools
import hashlib
import importlib.util
import logging
from pathlib import Path

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/layers/moe/token_dispatcher/flashinfer.py": (
        "FlashinferDispatcher",
        "FlashinferDispatchOutput",
    ),
    "srt/models/muse_spark.py": ("MuseSpark1xMoE",),
    "srt/environ.py": ("envs",),
    "srt/layers/muse_spark/muse_spark_fp4_moe.py|srt/layers/muse_spark_fp4_moe.py": (
        "MuseSpark1xFp4Experts",
    ),
    "srt/layers/dp_attention.py": (
        "get_dp_global_num_tokens",
        "get_is_extend_in_batch",
    ),
    "srt/layers/moe/topk.py": ("StandardTopKOutput", "TopKOutputChecker"),
    "srt/runtime_context.py": ("get_forward",),
}


logger = logging.getLogger("muse_vr_plugin")

NAME = "sp_fp4_dispatch"

SGLANG_PINS = {
    # FlashinferDispatcher.dispatch (SP-prefill branch) and its output types
    "sglang/srt/layers/moe/token_dispatcher/flashinfer.py": "61d79191720f8231f965be7f971155f139bf683b8b7625733b49e0031fbd2ad3",
    # MuseSpark1xMoE._forward_sp_experts
    "sglang/srt/models/muse_spark.py": (
        "7e5018b3cb2ef41ee6cf09407e91e28dde069e5a33719ea58beb0d7137087355",
        "8bafbb8802fb0d9f81c190e87b9012fca546182be08bef6c8b3db814d27bfe19",
        "c0db42467a960c509b658c3c40570ee889f73d41b52cdbd80b9c291c1792a33f",
    ),
    # fused_experts_none_to_flashinfer_trtllm_fp4: per-token quantization call, pre-quantized pass-through branch
    "sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py": (
        "2f6f14ae1d95030db1ec249856b331f4e6880b0218727e8701c3944376acaaf9",
        "70db650773c0dcbeec8056d70ad6f6a0c06acd5d583e9bbbe8f4d5548cf25929",
    ),
    "sglang/srt/layers/muse_spark_fp4_moe.py": (
        "1d377424e4c1f486f72a0e8ea67ea41dca4bd3f6929aa3729f5f7480861ed352",
        "1d7676d6d086e6aa45139671e345ef32d28a81b7de1b50810d0e512ee95cc738",
    ),
}
FLASHINFER_PINS = {
    "flashinfer/comm/trtllm_moe_alltoall.py": (
        "8ae4a8e618e3956f864d0e012dced7ffbac55d6db18e61d05d4425c51375fda2",  # FlashInfer 0.7.1
        "21da7a12c1ef08e9c87fc5071e3fa1fafe9fbff04f9571ec683f339a01ea5b3c",  # FlashInfer 0.7.2
    ),
    "flashinfer/data/csrc/trtllm_moe_alltoall.cu": "d672681fb81aad57b780bde8ec8e07edfea58b65ab420f91c8df6aa79652f60f",
    "flashinfer/fused_moe/core.py": (
        "d6a8333a2749e45e17f4ad6443c3ca84fcc8b15bca4d32adecc6cd0373528b17",  # FlashInfer 0.7.1
        "b25655f5cdc031ed003a53f92927d912154a6d4f8b37c92a52b19ae2e2da0d79",  # FlashInfer 0.7.2
    ),
}

_state: dict = {
    "engaged": None,
    "orig_dispatch": None,
    "orig_moe": None,
    "orig_norm": None,
    "pfront": None,
}
PFRONT = "pfront"
# sha256 of the sources whose arithmetic csrc/pfront.cuh reproduces (relative to site-packages)
PFRONT_PINS = {
    "torch/_vendor/quack/rmsnorm.py": "5e93beca566da3e91bc599668e64b5b2cdaf74dc9a4d4b0df2ddf0aa4c983a34",
    "torch/_vendor/quack/rmsnorm_config.py": "832b53456b89fbb643f7c3925ff2be38616a90d79f68df0a8dfc23fd41bbae5e",
    "torch/_vendor/quack/reduce.py": "59391796372b769a94a54465fbdaad93e44208af74ed937b1e0f5923db72cddd",
    "torch/_vendor/quack/reduction_base.py": "10737ec0dc31ac3ede1cec83f503f0598f460fb373296f7e52bef38f60f73ce2",
    "torch/_native/ops/norm/norms.py": "2e4e9d3f673cbdeb36febe5e64786e20d3f5a99c963b5c3c0fc7e399db00105e",
    "torch/_native/ops/norm/rmsnorm_impl.py": "101b23b9d0a9a2ad403d0ed52d3d93642d494285f7236b06af2e985226eb804d",
    "flashinfer/quantization/kernels/nvfp4_quantize.py": "88550172a3923e6e3417d151b22b0a97d5581bc9abc378d705d737058b4a6304",
    "flashinfer/quantization/quantization_cute_dsl_utils.py": "c4e2a0d938ce01953610a62f02a783aff82e877854cfab101dc79000c39f63c6",
    "flashinfer/cute_dsl/fp4_common.py": "6cda2e2fa955ca608844209e5792d5a2136f6a2e6e0a8d3fb7cc285ebc01531d",
}
# {"on": True} while an NVFP4 expert layer's sequence-parallel expert call runs; the dispatch adds "x" and "pts"
_ctx: contextvars.ContextVar = contextvars.ContextVar(
    "muse_vr_sp_fp4_dispatch", default=None
)


def _flashinfer_pins_ok() -> tuple[bool, str]:
    spec = importlib.util.find_spec("flashinfer")
    if spec is None or not spec.submodule_search_locations:
        return False, "flashinfer"
    root = Path(list(spec.submodule_search_locations)[0]).resolve().parent
    for rel, sha in FLASHINFER_PINS.items():
        try:
            got = hashlib.sha256((root / rel).read_bytes()).hexdigest()
        except OSError:
            got = None
        if not bed.sha_ok(got, sha):
            logger.warning(
                "muse_vr_plugin: %s sha256 %s does not match the pin %s",
                rel,
                (got or "<missing>")[:16],
                bed.sha_label(sha),
            )
            return False, rel
    return True, ""


# MuseSpark1xRMSNorm.forward (the deferred norm: F.rms_norm with the effective weight, cast to the input dtype)
PFRONT_SGLANG_PINS = {
    "sglang/srt/layers/muse_spark_layers.py": "930e66da715bae576ae681f2dfd8353a36a9ca469a229d88fa3f77bc1e74cef0",
}


def _pfront_pins_ok() -> tuple[bool, str]:
    for rel, sha in PFRONT_SGLANG_PINS.items():
        if not bed.pinned(rel, sha):
            return False, rel
    spec = importlib.util.find_spec("torch")
    if spec is None or not spec.submodule_search_locations:
        return False, "torch"
    root = Path(list(spec.submodule_search_locations)[0]).resolve().parent
    for rel, sha in PFRONT_PINS.items():
        try:
            got = hashlib.sha256((root / rel).read_bytes()).hexdigest()
        except OSError:
            got = None
        if got != sha:
            return False, rel
    return True, ""


def _pfront_on() -> bool:
    """Mode pfront requested and its pins match (decided once per process, reported under ``pfront``)."""
    if _state["pfront"] is None:
        if not bed.enabled(PFRONT):
            _state["pfront"] = False
        else:
            ok, why = _pfront_pins_ok()
            _state["pfront"] = ok
            bed.report(PFRONT, ok, **({} if ok else {"reason": f"pin mismatch: {why}"}))
    return bool(_state["pfront"])


def _e4m3_max() -> float | None:
    """The quantizer's e4m3 maximum as quantize_rows uses it; None in the 4-over-6 mode (not reproduced)."""
    from sglang.srt.environ import envs

    if envs.FLASHINFER_NVFP4_4OVER6.get():
        return None
    return 448.0


def _pfront_norm_ok(norm, x) -> bool:
    import torch

    from . import sgl_compat

    layers = sgl_compat.module("sglang.srt.layers.muse_spark_layers")
    if type(norm) is not layers.MuseSpark1xRMSNorm or _e4m3_max() is None:
        return False
    w = norm._get_effective_weight()
    return bool(
        w is not None
        and w.dtype == torch.bfloat16
        and w.is_contiguous()
        and w.numel() == 4096
        and w.data_ptr() % 16 == 0
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.dim() == 2
        and x.shape[1] == 4096
        and x.shape[0] > 0
        and x.is_contiguous()
        and x.data_ptr() % 16 == 0
    )


def _wrap_norm_forward(fn):
    @functools.wraps(fn)
    def forward(self, x):
        ctx = _ctx.get()
        if (
            ctx is not None
            and ctx.get("pfront_norm") is self
            and not ctx.get("pfront_done")
            and _pfront_norm_ok(self, x)
        ):
            ctx["pfront_done"] = True  # at most one deferral per expert call
            ctx["pfront_raw"] = x
            bed.count(PFRONT, "deferred")
            return x
        return fn(self, x)

    forward.__wrapped_by_pfront__ = True
    return forward


def _take_deferred(ctx, hidden_states):
    """The deferred norm's module when ``hidden_states`` is the deferred raw input (consumed), else None."""
    if ctx is None or ctx.get("pfront_raw") is None:
        return None
    raw = ctx["pfront_raw"]
    if raw.data_ptr() != hidden_states.data_ptr() or raw.shape != hidden_states.shape:
        raise RuntimeError(
            "sp_fp4_dispatch pfront: the dispatched rows are not the rows whose norm was deferred"
        )
    ctx["pfront_raw"] = None
    return ctx["pfront_norm"]


def quantize_rows(x):
    """Per-token NVFP4 quantization exactly as the sequence-parallel expert runner does it on the receiving rank."""
    import torch
    from flashinfer import SfLayout, nvfp4_quantize
    from sglang.srt.environ import envs

    e4m3_max = 448.0
    if (
        envs.FLASHINFER_NVFP4_4OVER6.get()
        and envs.FLASHINFER_NVFP4_4OVER6_E4M3_USE_256.get()
    ):
        e4m3_max = 256.0
    global_scale_inv = torch.full(
        (1,), 1.0 / (e4m3_max * 6.0), dtype=torch.float32, device=x.device
    )
    fp4_bytes, sf_bytes, per_token_scale = nvfp4_quantize(
        x,
        global_scale_inv,
        sfLayout=SfLayout.layout_linear,
        per_token_activation=True,
        backend="cute-dsl",
    )
    t, h = x.shape
    return (
        fp4_bytes.reshape(t, h // 2),
        sf_bytes.reshape(t, h // 16),
        per_token_scale.reshape(t),
    )


def _fp4_layer_eligible(moe) -> bool:
    from sglang.srt.layers.muse_spark_fp4_moe import MuseSpark1xFp4Experts

    ex = getattr(moe, "experts", None)
    return bool(
        getattr(moe, "nvfp4_checkpoint", False)
        and not getattr(moe, "use_megamoe", False)
        and isinstance(ex, MuseSpark1xFp4Experts)
        and not getattr(ex, "use_marlin", False)
        and not getattr(ex, "expert_tp", False)
        and getattr(ex.quant_config, "use_per_token_activation", False)
    )


def _wrap_forward_sp_experts(fn):
    @functools.wraps(fn)
    def _forward_sp_experts(self, expert_input, topk_scores, topk_ids, router_logits):
        if not _fp4_layer_eligible(self):
            return fn(self, expert_input, topk_scores, topk_ids, router_logits)
        ctx = {"on": True}
        if _pfront_on() and getattr(self, "pre_expert_norm", None) is not None:
            ctx["pfront_norm"] = self.pre_expert_norm
        token = _ctx.set(ctx)
        try:
            return fn(self, expert_input, topk_scores, topk_ids, router_logits)
        finally:
            _ctx.reset(token)

    _forward_sp_experts.__wrapped_by_sp_fp4__ = True
    return _forward_sp_experts


def _wrap_dispatch(fn):
    @functools.wraps(fn)
    def dispatch(self, hidden_states, topk_output):
        import torch
        from sglang.srt.layers.dp_attention import (
            get_dp_global_num_tokens,
            get_is_extend_in_batch,
        )
        from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
            FlashinferDispatchOutput,
        )
        from sglang.srt.layers.moe.topk import StandardTopKOutput, TopKOutputChecker
        from sglang.srt.runtime_context import get_forward

        ctx = _ctx.get()
        deferred = _take_deferred(ctx, hidden_states)
        dp_global = get_dp_global_num_tokens()
        if (
            ctx is None
            or not get_forward().sp_active
            or hidden_states.dtype != torch.bfloat16
            or self.hidden_size % 16
            or (dp_global is not None and len(dp_global) > 1)
        ):
            bed.count(NAME, "stock")
            if deferred is not None:
                bed.count(PFRONT, "stock_norm")
                hidden_states = _state["orig_norm"](deferred, hidden_states)
            return fn(self, hidden_states, topk_output)
        # The same all-to-all instance and geometry the stock dispatch picks for this call: prefill CUDA-graph
        # capture runs with the extend flag cleared, and the captured graph keeps the instance chosen then.
        is_sp_prefill = get_is_extend_in_batch()
        if TopKOutputChecker.format_is_bypassed(topk_output):
            topk_output = topk_output.to_standard()
        topk_ids = topk_output.topk_ids.to(torch.int32)
        topk_weights = topk_output.topk_weights
        if deferred is not None:
            from .ops import pfront as pfront_op

            x_fp4, x_sf, pts = pfront_op.pfront(
                hidden_states,
                deferred._get_effective_weight(),
                float(deferred.eps),
                1.0 / (_e4m3_max() * 6.0),
            )
            bed.count(PFRONT, "fused")
        else:
            x_fp4, x_sf, pts = quantize_rows(hidden_states)
        runtime_max = hidden_states.shape[
            0
        ]  # the stock geometry without a per-rank DP token list
        if runtime_max > self.max_num_tokens:
            raise RuntimeError(
                "sp_fp4_dispatch: runtime token geometry exceeds the all-to-all workspace"
            )
        # largest element size first, as the all-to-all requires
        payloads = [pts.view(-1, 1).contiguous(), topk_ids, topk_weights, x_fp4, x_sf]
        self.runtime_max_tokens_per_rank = runtime_max
        self._start_a2a(sp_prefill=is_sp_prefill)
        try:
            pts_r, ids_r, w_r, x_r, sf_r = self.active_moe_a2a.dispatch(
                topk_ids,
                payloads,
                runtime_max,
                invalid_token_expert_id=self.invalid_token_expert_id,
                expert_id_payload_index=1,
            )
            x = x_r.view(-1, x_r.shape[-1])
            moe_output = None
            if self.payload_in_workspace:
                moe_output = (
                    self.active_moe_a2a.get_combine_payload_tensor_in_workspace(
                        runtime_max, self.hidden_size, torch.bfloat16
                    ).view(-1, self.hidden_size)
                )
            ctx["x"] = x
            ctx["pts"] = pts_r.view(-1)
            bed.count(NAME, "fp4_dispatch")
            return FlashinferDispatchOutput(
                x,
                sf_r.view(-1, sf_r.shape[-1]),
                StandardTopKOutput(
                    w_r.view(-1, w_r.shape[-1]),
                    ids_r.view(-1, ids_r.shape[-1]),
                    topk_output.router_logits,
                ),
                moe_output,
                torch.bfloat16,
            )
        except Exception:
            self._clear_active_a2a()
            raise

    dispatch.__wrapped_by_sp_fp4__ = True
    return dispatch


def _routed_moe(*args, **kwargs):
    """Stand-in for ``flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe``: adds the dispatched per-token scales to
    the call made for the dispatched codes (identified by tensor identity); every other call passes through."""
    ctx = _ctx.get()
    if (
        ctx is not None
        and "x" in ctx
        and not args
        and kwargs.get("per_token_scale") is None
    ):
        hs = kwargs.get("hidden_states")
        if (
            hs is not None
            and hs.data_ptr() == ctx["x"].data_ptr()
            and hs.shape == ctx["x"].shape
        ):
            kwargs["per_token_scale"] = ctx["pts"]
            bed.count(NAME, "per_token_scale")
    return _state["orig_moe"](*args, **kwargs)


def install() -> None:
    import flashinfer.fused_moe as ffm
    import sglang.srt.layers.moe.token_dispatcher.flashinfer as fdisp
    import sglang.srt.models.muse_spark as muse_spark

    if getattr(fdisp.FlashinferDispatcher.dispatch, "__wrapped_by_sp_fp4__", False):
        return
    _state["orig_moe"] = ffm.trtllm_fp4_block_scale_routed_moe
    ffm.trtllm_fp4_block_scale_routed_moe = _routed_moe
    fdisp.FlashinferDispatcher.dispatch = _wrap_dispatch(
        fdisp.FlashinferDispatcher.dispatch
    )
    moe_cls = muse_spark.MuseSpark1xMoE
    moe_cls._forward_sp_experts = _wrap_forward_sp_experts(moe_cls._forward_sp_experts)
    if _pfront_on():
        from . import sgl_compat

        norm_cls = sgl_compat.module(
            "sglang.srt.layers.muse_spark_layers"
        ).MuseSpark1xRMSNorm
        if not getattr(norm_cls.forward, "__wrapped_by_pfront__", False):
            _state["orig_norm"] = norm_cls.forward
            norm_cls.forward = _wrap_norm_forward(norm_cls.forward)


def engage() -> bool:
    if _state["engaged"] is not None:
        return _state["engaged"]
    if not bed.enabled(NAME):
        _state["engaged"] = False
        return False
    for rel, sha in SGLANG_PINS.items():
        if not bed.pinned(rel, sha):
            bed.report(NAME, False, reason=f"pin mismatch: {rel}")
            _state["engaged"] = False
            return False
    ok, why = _flashinfer_pins_ok()
    if not ok:
        bed.report(NAME, False, reason=f"pin mismatch: {why}")
        _state["engaged"] = False
        return False
    try:
        install()
    except Exception as e:  # noqa: BLE001
        bed.report(NAME, False, reason=f"install failed: {type(e).__name__}: {e}")
        _state["engaged"] = False
        return False
    bed.report(NAME, True)
    _state["engaged"] = True
    return True

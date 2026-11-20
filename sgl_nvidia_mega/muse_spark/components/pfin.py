# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``pfin``: the sequence-parallel prefill MoE finalize on the plugin's persistent kernel (``ops/pfin.py``).

Under ``--enable-layernorm-sp`` every routed expert layer of a prefill pass ends with SGLang's fused finalize
(``FusedPostExpertNormWeightedSumTRTLLM``: unpermute, post-expert RMSNorm, router-weighted top-k sum) over the tokens the
all-to-all delivered to the rank. For more than 128 tokens at hidden 4096 the op launches its CuTe-DSL kernel with one
128-thread CTA per token, which loads one expert row at a time and reduces it before loading the next. This component
runs the same arithmetic (same thread map, products, reduction trees, expert order and rounding; bitwise identical
output, checked by ``tests/test_pfin.py`` against the stock op) on persistent CTAs that keep the token's expert rows in
flight through a shared-memory ring of bulk copies.

Engagement (``MUSE_VR_PFIN=1``): only the op's CuTe-DSL calls without an all-reduce workspace, with the default cluster
selection, hidden 4096, top-8, bf16 GEMM2 rows / norm weight / output, fp32 or bf16 router weights, int32 indices, all
contiguous and 16-byte aligned, at ``MUSE_VR_PFIN_MIN_T`` (default 8192) tokens or more, and (unless
``MUSE_VR_PFIN_SP_ONLY=0``) only while a sequence-parallel pass is active. Every other call runs the stock kernel and
is counted under ``pfin`` as ``stock:<reason>``; engaged calls are counted as ``pfin``. Ring depth
``MUSE_VR_PFIN_SLOTS``, consumer groups ``MUSE_VR_PFIN_GROUPS`` (a built pair of ``ops.pfin.CONFIGS``; default 24 / 4)
and ``MUSE_VR_PFIN_CTAS`` CTAs per SM (default 1). Engagement is guarded
by sha256 pins on the SGLang op and kernel sources whose arithmetic the kernel reproduces.
"""

from __future__ import annotations

import functools
import logging

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/runtime_context.py": ("get_forward",),
}


logger = logging.getLogger("muse_vr_plugin")

NAME = "pfin"
OP_MODULE = (
    "sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_trtllm"
)
OP_CLASS = "FusedPostExpertNormWeightedSumTRTLLM"
# One pin set per validated SGLang layout (the op: cluster-size selection, cluster 1 above 128 tokens at hidden 4096, and
# launch arguments; the CuTe-DSL kernel whose single-CTA arithmetic csrc/pfin.cuh reproduces). Either set must match.
SGLANG_PIN_SETS = (
    {  # SGLang main layout
        "sglang/kernels/ops/communication/muse_spark_moe_finalize_all_reduce.py": "b22ba8b4d895fd38c99a0835b5a37b82adaa8e7587379f09807cb072eceb9948",
        "sglang/kernels/ops/communication/muse_spark_moe_finalize_all_reduce_cutedsl.py": "9e503382ec0d904476aa039d77510486cc71563c2e1a8bfcbb8d9234e7e2e689",
    },
    {  # original branch layout
        "sglang/kernels/ops/muse_spark_v12/fused_post_expert_norm_weighted_sum_trtllm.py": "d2ba335e91affae92c8563a249f56cfaa2e54b84ae3f8d6d692624c105e0d048",
        "sglang/kernels/ops/muse_spark_v12/fi_trtllm_moe_finalize_with_rmsnorm_cutedsl.py": "01fac72ef834b126c502252dcef1c83ab5cecef4a2c6241a0ec3e2292fb008a4",
    },
)
HIDDEN = 4096
TOPK = 8

_state: dict = {"engaged": None, "orig": None, "warned": set()}


_DEFAULT_CFG = (24, 4, 1)  # ring depth, consumer groups per CTA, CTAs per SM


def _cfg() -> dict:
    from .ops import pfin as pfin_op

    slots = bed.env_int("MUSE_VR_PFIN_SLOTS", _DEFAULT_CFG[0])
    groups = bed.env_int("MUSE_VR_PFIN_GROUPS", _DEFAULT_CFG[1])
    if (slots, groups) not in pfin_op.CONFIGS:
        slots, groups = _DEFAULT_CFG[:2]
    return {
        "min_t": bed.env_int("MUSE_VR_PFIN_MIN_T", 8192),
        "slots": slots,
        "groups": groups,
        "ctas": max(1, bed.env_int("MUSE_VR_PFIN_CTAS", _DEFAULT_CFG[2])),
        "sp_only": bed.env_int("MUSE_VR_PFIN_SP_ONLY", 1) != 0,
    }


def _sp_active() -> bool:
    try:
        from sglang.srt.runtime_context import get_forward

        return bool(getattr(get_forward(), "sp_active", False))
    except Exception:  # noqa: BLE001
        return False


def _aligned(t) -> bool:
    return t.data_ptr() % 16 == 0


def ineligible(
    op, gemm2_output, expert_weights, idx, rms_w, output, ar_workspace, cfg: dict
) -> str:
    """'' when the plugin kernel reproduces this call; otherwise the step-aside reason."""
    import torch

    if ar_workspace is not None:
        return "ar_workspace"
    if getattr(op, "cluster_size", None) is not None:
        return "explicit_cluster"
    if int(op.hidden_dim) != HIDDEN or int(op.topk) != TOPK:
        return "shape"
    if cfg["sp_only"] and not _sp_active():
        return "not_sp"
    if output.dim() != 2 or output.shape[1] != HIDDEN:
        return "out_shape"
    t = int(output.shape[0])
    if t < max(int(cfg["min_t"]), 129):
        return "small_t"
    if not (
        gemm2_output.is_cuda
        and gemm2_output.dim() == 2
        and gemm2_output.dtype == torch.bfloat16
        and gemm2_output.shape[1] >= HIDDEN
        and gemm2_output.is_contiguous()
    ):
        return "gemm2"
    if not (output.dtype == torch.bfloat16 and output.is_contiguous()):
        return "out"
    if not (
        rms_w.dtype == torch.bfloat16
        and rms_w.is_contiguous()
        and rms_w.numel() == HIDDEN
    ):
        return "rms_w"
    if not (
        idx.dtype == torch.int32 and idx.is_contiguous() and idx.numel() == t * TOPK
    ):
        return "idx"
    if not (
        expert_weights.dtype in (torch.float32, torch.bfloat16)
        and expert_weights.is_contiguous()
        and expert_weights.numel() == t * TOPK
    ):
        return "expert_weights"
    if not (
        _aligned(gemm2_output)
        and _aligned(output)
        and _aligned(rms_w)
        and (gemm2_output.shape[1] * 2) % 16 == 0
    ):
        return "align"
    if torch.cuda.get_device_capability(output.device)[0] < 9:
        return "arch"
    return ""


def _wrap(fn):
    @functools.wraps(fn)
    def forward_cute_dsl(
        self,
        gemm2_output,
        expert_weights,
        expanded_idx_to_permuted_idx,
        rms_norm_weight,
        output,
        *,
        ar_workspace=None,
    ):
        cfg = _cfg()
        why = ineligible(
            self,
            gemm2_output,
            expert_weights,
            expanded_idx_to_permuted_idx,
            rms_norm_weight,
            output,
            ar_workspace,
            cfg,
        )
        if why:
            bed.count(NAME, "stock:" + why)
            if (
                why not in ("not_sp", "small_t", "ar_workspace")
                and why not in _state["warned"]
            ):
                _state["warned"].add(why)
                logger.warning(
                    "muse_vr_plugin: pfin stepped aside to the stock finalize (%s)", why
                )
            return fn(
                self,
                gemm2_output,
                expert_weights,
                expanded_idx_to_permuted_idx,
                rms_norm_weight,
                output,
                ar_workspace=ar_workspace,
            )
        from .ops import pfin as pfin_op

        if int(output.shape[0]) == 0:
            return None
        pfin_op.pfin(
            gemm2_output,
            expert_weights,
            expanded_idx_to_permuted_idx,
            rms_norm_weight,
            output,
            float(self.eps),
            bool(self.weight_first),
            slots=cfg["slots"],
            groups=cfg["groups"],
            ctas_per_sm=cfg["ctas"],
        )
        bed.count(NAME, "pfin")
        return None

    forward_cute_dsl.__wrapped_by_pfin__ = True
    return forward_cute_dsl


def install() -> None:
    from . import sgl_compat

    mod = sgl_compat.module(OP_MODULE)
    cls = getattr(mod, OP_CLASS)
    if getattr(cls.forward_cute_dsl, "__wrapped_by_pfin__", False):
        return
    _state["orig"] = cls.forward_cute_dsl
    cls.forward_cute_dsl = _wrap(cls.forward_cute_dsl)


def uninstall() -> None:
    from . import sgl_compat

    if _state["orig"] is None:
        return
    cls = getattr(sgl_compat.module(OP_MODULE), OP_CLASS)
    cls.forward_cute_dsl = _state["orig"]
    _state["orig"] = None


def engage() -> bool:
    if _state["engaged"] is not None:
        return _state["engaged"]
    if not bed.enabled(NAME):
        _state["engaged"] = False
        return False
    if not any(
        all(bed.file_sha256(rel) == sha for rel, sha in pins.items())
        for pins in SGLANG_PIN_SETS
    ):
        bed.report(NAME, False, reason="pin mismatch: finalize op / kernel sources")
        _state["engaged"] = False
        return False
    try:
        install()
    except Exception as e:  # noqa: BLE001
        bed.report(NAME, False, reason=f"install failed: {type(e).__name__}: {e}")
        _state["engaged"] = False
        return False
    bed.report(NAME, True, **_cfg())
    _state["engaged"] = True
    return True

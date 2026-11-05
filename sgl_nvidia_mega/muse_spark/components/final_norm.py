# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``final_norm`` (env ``MUSE_VR_FINAL_NORM=1``, default off).

After its MoE, the last Muse Spark decoder layer (the one without a fused FFN boundary) applies its post-feedforward
norm (weight times input, then RMSNorm) and the norm-preserving gated residual with PyTorch operations, and the model
then casts the fp32 stream to bf16 and applies the final RMSNorm: nine kernels before the LM head. This component runs
the same operations in one kernel (``csrc/final_norm.cuh``), with the same roundings and the same summation order as
PyTorch's row-wise RMSNorm, so the hidden states that reach the LM head are bit-identical
(tests/test_final_norm.py).

Mechanism: an AROUND hook on ``MuseSpark1xModel._forward``. For decode and target-verify batches (no sequence
parallelism), it lets SGLang's forward run unchanged, except that during the call the last layer's
post-feedforward norm and residual gate, and the model's final norm, are replaced by stand-ins: the first two record
their inputs (the layer's FFN output and the fp32 stream) and pass a bf16 tensor along, so the model's cast is a no-op;
the final norm's stand-in runs the fused kernel on the recorded inputs. The originals are restored when the call
returns. Every other call runs stock.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path

import torch

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/layers/layernorm_sp.py": ("runs_sp",),
    "kernels/jit/utils/__init__.py": (
        "is_arch_support_pdl",
        "load_jit",
        "make_cpp_args",
    ),
    "srt/layers/muse_spark/muse_spark_layers.py|srt/layers/muse_spark_layers.py": (
        "MuseSpark1xNormPreservingResidualConnection",
        "MuseSpark1xRMSNorm",
        "MuseSpark1xRMSNormWithInputScale",
    ),
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
}


logger = logging.getLogger(__name__)

NAME = "final_norm"  # env MUSE_VR_FINAL_NORM
TARGET = "sglang.srt.models.muse_spark.MuseSpark1xModel._forward"
_CUH = Path(__file__).resolve().parent / "csrc" / "final_norm.cuh"

PINS = {
    "sglang/srt/models/muse_spark.py": (
        "7e5018b3cb2ef41ee6cf09407e91e28dde069e5a33719ea58beb0d7137087355",
        "8bafbb8802fb0d9f81c190e87b9012fca546182be08bef6c8b3db814d27bfe19",
        "c0db42467a960c509b658c3c40570ee889f73d41b52cdbd80b9c291c1792a33f",
    ),
    "sglang/srt/layers/muse_spark_layers.py": "930e66da715bae576ae681f2dfd8353a36a9ca469a229d88fa3f77bc1e74cef0",
    "sglang/srt/layers/layernorm_sp.py": (
        "ffdde97e6282103614d7cec20cc29fb3cf0157d051778e315b03a99bb8e81090",
        "bc814d983b945e5bdd41db436966eb4b3fee4c410bb1dcccfc98857d4b026bda",
    ),
}
# PyTorch builds whose row-wise RMSNorm summation order the bitwise test matched (torch.version.git_version), with the
# vector width of that order: consecutive elements a thread sums per step (measured with tests/test_final_norm.py).
TORCH_VEC = {
    "e0474491d782a73b73cf132f732a8513b3c9f49d": 4,
    "7bf52b3a9782d3440a17eb3f00983795f3b12af0": 8,
}
VALIDATED_TORCH = tuple(TORCH_VEC)


def vec_width() -> int:
    """The running PyTorch build's vector width (4 for a build without a rule; such builds keep the stock path)."""
    return TORCH_VEC.get(torch.version.git_version, 4)


def enabled() -> bool:
    return bed.enabled(NAME)


def pins_match() -> tuple[bool, str]:
    for rel, want in PINS.items():
        if not bed.pinned(rel, want):
            return False, f"{rel} differs from the pinned revision"
    if torch.version.git_version not in VALIDATED_TORCH:
        return (
            False,
            f"PyTorch build {torch.version.git_version[:12]} not in the validated set",
        )
    return True, "ok"


@functools.lru_cache(maxsize=None)
def _use_pdl() -> bool:
    from sglang.kernels.jit.utils import is_arch_support_pdl

    return bool(is_arch_support_pdl())


@functools.lru_cache(maxsize=None)
def module(hidden: int, pdl: bool, vec: int = 4):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    args = make_cpp_args(hidden, pdl) if vec == 4 else make_cpp_args(hidden, pdl, vec)
    return load_jit(
        "muse_vr_final_norm",
        *args,
        cuda_files=[str(_CUH)],
        cuda_wrappers=[("run", f"muse_vr::final_norm::FinalNorm<{args}>::run")],
    )


def gate_coefficients(gate):
    """(alpha, beta) fp32 [H] exactly as MuseSpark1xNormPreservingResidualConnection.forward uses them."""
    if gate._alpha is None:
        alpha, beta = gate._coefficients(
            gate.gate.view(1, -1).float() / gate.temperature
        )
    else:
        alpha, beta = gate._alpha, gate._beta
    h = gate.gate.numel()
    if not isinstance(alpha, torch.Tensor):
        alpha = torch.full(
            (1, h), float(alpha), dtype=torch.float32, device=gate.gate.device
        )
    return alpha.reshape(-1).contiguous(), beta.reshape(-1).contiguous()


def fused(ffn, stream, post_norm, gate, final_norm, out=None, pdl=None, vec=None):
    """The output boundary for ``ffn`` (bf16) and ``stream`` (fp32): returns the final-norm output (bf16)."""
    alpha, beta = gate_coefficients(gate)
    ffn, stream = ffn.contiguous(), stream.contiguous()
    if out is None:
        out = torch.empty(ffn.shape, dtype=torch.bfloat16, device=ffn.device)
    module(
        ffn.shape[-1],
        _use_pdl() if pdl is None else pdl,
        vec_width() if vec is None else vec,
    ).run(
        ffn,
        stream,
        post_norm._get_effective_weight(),
        alpha,
        beta,
        final_norm._get_effective_weight(),
        out,
        float(post_norm.eps),
        float(final_norm.eps),
    )
    return out


class _Record(torch.nn.Module):
    def __init__(self, ctx: dict, key: str):
        super().__init__()
        self._ctx, self._key = ctx, key

    def forward(self, x):
        self._ctx[self._key] = x
        return x


class _Gate(torch.nn.Module):
    def __init__(self, ctx: dict):
        super().__init__()
        self._ctx = ctx

    def forward(self, stream, branch):
        self._ctx["stream"] = stream
        return branch  # bf16: the model's cast of the stream to bf16 is then a no-op


class _Norm(torch.nn.Module):
    def __init__(self, ctx: dict, post_norm, gate, final_norm):
        super().__init__()
        self._ctx = ctx
        self._mods = (post_norm, gate, final_norm)

    def forward(self, x):
        ffn, stream = self._ctx.pop("ffn"), self._ctx.pop("stream")
        if x.data_ptr() != ffn.data_ptr():
            raise RuntimeError("final_norm: unexpected call order")
        self._ctx["fused"] = True
        return fused(ffn, stream, *self._mods)


def _eligible(model, forward_batch) -> str | None:
    import sglang.srt.layers.layernorm_sp as layernorm_sp
    from sglang.srt.layers.muse_spark_layers import (
        MuseSpark1xNormPreservingResidualConnection,
        MuseSpark1xRMSNorm,
        MuseSpark1xRMSNormWithInputScale,
    )

    mode = forward_batch.forward_mode
    if not (mode.is_decode() or mode.is_target_verify()):
        return "not decode/verify"
    if layernorm_sp.runs_sp(mode):
        return "sequence parallel"
    last = model.layers[-1]
    if last._fused_ffn_boundary is not None:
        return "last layer has a fused boundary"
    if len(model.layers) - 1 in getattr(model, "layers_to_capture", ()):
        return "last layer's stream is captured"  # the stand-ins would hand the capture a different tensor
    pn, gt, fn = (
        last.post_feedforward_layernorm,
        last.post_feedforward_residual_gate,
        model.norm,
    )
    if (
        type(pn) is not MuseSpark1xRMSNormWithInputScale
        or type(fn) is not MuseSpark1xRMSNorm
        or (type(gt) is not MuseSpark1xNormPreservingResidualConnection)
    ):
        return "module types"
    if (
        pn.weight is None
        or fn.weight is None
        or model.residual_dtype != torch.float32
        or (model.operator_dtype != torch.bfloat16)
    ):
        return "dtypes"
    wp, wf = pn._get_effective_weight(), fn._get_effective_weight()
    if (
        wp.dtype != torch.bfloat16
        or wf.dtype != torch.bfloat16
        or wp.numel() % (128 * vec_width())
        or not gt.use_stream_gate
    ):
        return "weights"
    if torch.cuda.is_current_stream_capturing():
        if (wp.numel(), _use_pdl()) not in _loaded:
            return "kernel not loaded before graph capture"
    else:
        module(wp.numel(), _use_pdl(), vec_width())
        _loaded.add((wp.numel(), _use_pdl()))
    return None


_loaded: set = set()


def model_forward(
    original, model, input_ids, positions, forward_batch, input_embeds=None
):
    reason = _eligible(model, forward_batch)
    if reason is not None:
        bed.count(NAME, "stock:" + reason)
        return original(model, input_ids, positions, forward_batch, input_embeds)
    last = model.layers[-1]
    ctx: dict = {}
    saved = (
        last.post_feedforward_layernorm,
        last.post_feedforward_residual_gate,
        model.norm,
    )
    pn, gt, fn = saved
    last.post_feedforward_layernorm = _Record(ctx, "ffn")
    last.post_feedforward_residual_gate = _Gate(ctx)
    model.norm = _Norm(ctx, pn, gt, fn)
    try:
        out = original(model, input_ids, positions, forward_batch, input_embeds)
    finally:
        (
            last.post_feedforward_layernorm,
            last.post_feedforward_residual_gate,
            model.norm,
        ) = saved
    if not ctx.get("fused"):
        raise RuntimeError("final_norm: the output boundary was not reached")
    bed.count(NAME, "fused")
    return out


def register() -> bool:
    if not enabled():
        return False
    ok, why = pins_match()
    if not ok:
        bed.report(NAME, False, reason=why)
        return False
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    def hook(original, *args, **kwargs):
        names = ("input_ids", "positions", "forward_batch", "input_embeds")
        if len(args) < 4 or len(args) > 5 or set(kwargs) - set(names[len(args) - 1 :]):
            return original(*args, **kwargs)
        model, rest = args[0], list(args[1:])
        kw = dict(zip(names, rest))
        kw.update(kwargs)
        return model_forward(
            original,
            model,
            kw["input_ids"],
            kw["positions"],
            kw["forward_batch"],
            kw.get("input_embeds"),
        )

    HookRegistry.register(TARGET, hook, HookType.AROUND)
    bed.report(NAME, True, vec=vec_width())
    return True

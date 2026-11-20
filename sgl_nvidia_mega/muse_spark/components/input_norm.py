# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``input_norm`` (env ``MUSE_VR_INPUT_NORM=1``, default off).

Before the first Muse Spark decoder layer, SGLang applies the embedding RMSNorm (no weight), casts the result to the
fp32 residual stream, and applies the first layer's input RMSNorm to the stream (PyTorch's composite form, because the
stream is fp32 and the weight bf16): ten kernels per forward. This component runs them as three
(``csrc/input_norm.cuh``): one kernel for the embedding norm, the stream and the squares; PyTorch's own row mean of the
squares; one kernel for the rest of the input norm. Same operations, roundings and summation orders, bit-identical
outputs (tests/test_input_norm.py).

When the embeddings arrive already normalized (``input_embeds`` given, as the multimodal model wrapper passes them),
SGLang casts them to the fp32 stream and applies the first layer's input RMSNorm (eight kernels); this component then
runs one kernel for the stream and the squares (``stream_sq``), PyTorch's row mean and the same finishing kernel, and
hands the model the fp32 stream as ``input_embeds`` (so its cast is a no-op). Bit-identical as well.

Mechanism (as ``final_norm``): an AROUND hook on ``MuseSpark1xModel._forward`` that, for decode and target-verify
batches without sequence parallelism, temporarily replaces the model's embedding norm and the first layer's input norm by
stand-ins: the first returns the fp32 stream (so the model's cast is a no-op) and keeps the squares, the second
finishes the input norm. Everything else is SGLang's forward.
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
        "MuseSpark1xRMSNorm",
    ),
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
}


logger = logging.getLogger(__name__)

NAME = "input_norm"  # env MUSE_VR_INPUT_NORM
TARGET = "sglang.srt.models.muse_spark.MuseSpark1xModel._forward"
_CUH = Path(__file__).resolve().parent / "csrc" / "input_norm.cuh"

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
VALIDATED_TORCH = (
    "e0474491d782a73b73cf132f732a8513b3c9f49d",
    "7bf52b3a9782d3440a17eb3f00983795f3b12af0",
)
_loaded: set = set()


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
def module(hidden: int, pdl: bool):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    args = make_cpp_args(hidden, pdl)
    return load_jit(
        "muse_vr_input_norm",
        *args,
        cuda_files=[str(_CUH)],
        cuda_wrappers=[
            ("embed_stream", f"muse_vr::input_norm::InputNorm<{args}>::embed_stream"),
            ("stream_sq", f"muse_vr::input_norm::InputNorm<{args}>::stream_sq"),
            ("finish", f"muse_vr::input_norm::InputNorm<{args}>::finish"),
        ],
    )


def embed_stream(x, eps, pdl=None):
    """(stream fp32, squares fp32) for the embedding output ``x`` (bf16)."""
    x = x.contiguous()
    stream = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    sq = torch.empty_like(stream)
    module(x.shape[-1], _use_pdl() if pdl is None else pdl).embed_stream(
        x, stream, sq, float(eps)
    )
    return stream, sq


def stream_sq(x, pdl=None):
    """(stream fp32, squares fp32) for already normalized embeddings ``x`` (bf16)."""
    x = x.contiguous()
    stream = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    sq = torch.empty_like(stream)
    module(x.shape[-1], _use_pdl() if pdl is None else pdl).stream_sq(x, stream, sq)
    return stream, sq


def finish(stream, sq, weight, eps, pdl=None):
    mean = sq.mean(-1, keepdim=True)
    out = torch.empty(stream.shape, dtype=torch.bfloat16, device=stream.device)
    module(stream.shape[-1], _use_pdl() if pdl is None else pdl).finish(
        stream, mean, weight, out, float(eps)
    )
    return out


class _Embed(torch.nn.Module):
    def __init__(self, ctx, norm):
        super().__init__()
        self._ctx, self._norm = ctx, norm

    def forward(self, x):
        stream, sq = embed_stream(x, self._norm.eps)
        self._ctx["stream"], self._ctx["sq"] = stream, sq
        return stream  # fp32: the model's cast to the residual dtype is then a no-op


class _Input(torch.nn.Module):
    def __init__(self, ctx, norm):
        super().__init__()
        self._ctx, self._norm = ctx, norm

    def forward(self, stream):
        if (
            stream.data_ptr() != self._ctx.get("stream", stream).data_ptr()
            or "sq" not in self._ctx
        ):
            raise RuntimeError("input_norm: unexpected call order")
        out = finish(
            stream,
            self._ctx.pop("sq"),
            self._norm._get_effective_weight(),
            self._norm.eps,
        )
        self._ctx["fused"] = True
        return out  # bf16: the layer's cast to the operator dtype is then a no-op


def _eligible(model, input_embeds, forward_batch) -> str | None:
    import sglang.srt.layers.layernorm_sp as layernorm_sp
    from sglang.srt.layers.muse_spark_layers import MuseSpark1xRMSNorm

    mode = forward_batch.forward_mode
    if not (mode.is_decode() or mode.is_target_verify()):
        return "not decode/verify"
    if input_embeds is not None and (
        input_embeds.dtype != torch.bfloat16 or input_embeds.dim() != 2
    ):
        return "input embeddings layout"
    if input_embeds is None and model.embed_norm is None:
        return "no embedding norm"
    if layernorm_sp.runs_sp(mode):
        return "sequence parallel"
    en, ln = model.embed_norm, model.layers[0].input_layernorm
    if type(ln) is not MuseSpark1xRMSNorm or (
        input_embeds is None
        and (type(en) is not MuseSpark1xRMSNorm or en.weight is not None)
    ):
        return "module types"
    w = ln._get_effective_weight()
    if (
        w is None
        or w.dtype != torch.bfloat16
        or w.numel() % 512
        or model.residual_dtype != torch.float32
        or (model.operator_dtype != torch.bfloat16)
    ):
        return "dtypes"
    if torch.cuda.is_current_stream_capturing():
        if (w.numel(), _use_pdl()) not in _loaded:
            return "kernel not loaded before graph capture"
    else:
        module(w.numel(), _use_pdl())
        _loaded.add((w.numel(), _use_pdl()))
    return None


def model_forward(
    original, model, input_ids, positions, forward_batch, input_embeds=None
):
    reason = _eligible(model, input_embeds, forward_batch)
    if reason is not None:
        bed.count(NAME, "stock:" + reason)
        return original(model, input_ids, positions, forward_batch, input_embeds)
    first = model.layers[0]
    ctx: dict = {}
    saved = (model.embed_norm, first.input_layernorm)
    if input_embeds is None:
        model.embed_norm = _Embed(ctx, saved[0])
        tag = "fused"
    else:
        # the model casts input_embeds to the fp32 stream: hand it the stream (the cast is then a no-op)
        input_embeds, ctx["sq"] = stream_sq(input_embeds)
        ctx["stream"] = input_embeds
        tag = "fused:input_embeds"
    first.input_layernorm = _Input(ctx, saved[1])
    try:
        out = original(model, input_ids, positions, forward_batch, input_embeds)
    finally:
        model.embed_norm, first.input_layernorm = saved
    if not ctx.get("fused"):
        raise RuntimeError("input_norm: the input boundary was not reached")
    bed.count(NAME, tag)
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
    bed.report(NAME, True)
    return True

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Portions of this file are copied from SGLang (Apache-2.0, see include/LICENSE.sglang):
# python/sglang/srt/models/muse_spark.py (MuseSpark1xAttention.forward, MuseSpark1xDecoderLayer.forward) on SGLang main.
# Modified by NVIDIA: each method is split at the boundary the branch layout of that file had (gated_attention between
# the QKV and output projections; the deferred-unpool FFN boundary as FusedPostNormResidualPreNorm.forward_with_ag),
# with every statement otherwise unchanged.
"""Method boundaries on SGLang main's Muse Spark model that plugin components hook (main layout only).

SGLang main merged or inlined a few methods that components wrap on the branch layout. Each shim below restores one
boundary as an exact split of main's own code, so stock behaviour is unchanged and a component's hook sees the same
call it sees on the branch layout:

- ``MuseSpark1xModel._forward`` / ``MuseSparkTextForCausalLM._forward``: main's ``forward`` itself (the function
  object, not a copy); ``forward`` becomes a call of ``self._forward`` with the same arguments.
- ``MuseSpark1xForCausalLM``: the module-level name of the renamed class ``MuseSparkTextForCausalLM``.
- ``MuseSpark1xAttention.gated_attention``: the statements of main's ``forward`` between the QKV and the output
  projection; ``forward`` calls it between the two projections.
- ``FusedPostNormResidualPreNorm.forward_with_ag(stream, local, comm)``: main's call
  ``all_gather_norm_residual(comm, local, stream, **self.jit_kernel_kwargs())``; ``MuseSpark1xDecoderLayer.forward``
  makes it through this method, otherwise unchanged; ``MuseSpark1xDecoderLayer._megakernel_capable`` = False.

Installed only when the installed ``sglang/srt/models/muse_spark.py`` has a digest listed in ``MUSE_SPARK_MAIN`` (the
revisions these splits were checked against); otherwise nothing is installed and components that depend on a boundary
step aside through their own pins.
"""

from __future__ import annotations

import importlib
import logging

logger = logging.getLogger("muse_vr_plugin")

MODEL_RELPATH = "sglang/srt/models/muse_spark.py"
# sha256 of the main-layout revisions of MODEL_RELPATH these splits were checked against, line by line. The second entry is
# the first one with only the two gemm_ag device-class conditions of MuseSpark1xMoE widened (__init__'s workspace reserve
# and forward's pre-expert projection branch): a stock-path copy that both arms of a comparison may load; no statement any
# split copies or any component hooks differs.
MUSE_SPARK_MAIN: tuple = (
    "8bafbb8802fb0d9f81c190e87b9012fca546182be08bef6c8b3db814d27bfe19",
    "c0db42467a960c509b658c3c40570ee889f73d41b52cdbd80b9c291c1792a33f",
)

_installed: list = []


def _gated_attention(self, positions, qkv, forward_batch):
    split_sizes = [self.q_size, self.kv_size, self.kv_size]
    if self.use_o_gate:
        q, k, v, o_gate = qkv.split(split_sizes + [self.q_size], dim=-1)
    else:
        q, k, v = qkv.split(split_sizes, dim=-1)
        o_gate = None
    fuse_rope = self._fused_qk_norm is not None and self.rotary_emb is not None
    kv_store = (
        self._fused_kv_store(forward_batch)
        if self._fused_qk_norm is not None and (self.rotary_emb is None or fuse_rope)
        else None
    )
    if fuse_rope:
        q, k = self._fused_qk_norm(
            qkv, positions, self.rotary_emb.cos_sin_cache, kv_store=kv_store
        )
    elif self._fused_qk_norm is not None:
        q, k = self._fused_qk_norm(qkv, kv_store=kv_store)
    if self.rotary_emb is not None and not fuse_rope:
        q, k = self.rotary_emb(positions, q, k)
    output = self.attn(q, k, v, forward_batch, save_kv_cache=kv_store is None)
    return self._fused_o_norm_gate(output, o_gate)


def _attention_forward(self, positions, hidden_states, forward_batch):
    if hidden_states.shape[0] == 0:
        return hidden_states
    qkv, _ = self.qkv_proj(hidden_states)
    output, _ = self.o_proj(self.gated_attention(positions, qkv, forward_batch))
    return output


def _forward_with_ag(self, stream, local, comm):
    ms = importlib.import_module("sglang.srt.models.muse_spark")
    return ms.all_gather_norm_residual(comm, local, stream, **self.jit_kernel_kwargs())


def _decoder_layer_forward(self, positions, stream, attn_input, forward_batch):
    """Returns the updated residual stream and the next layer's attention input."""
    ms = importlib.import_module("sglang.srt.models.muse_spark")
    attn_output = self.self_attn(positions, attn_input, forward_batch)
    router_dtype = self._router_dtype
    if router_dtype is not None and ms._use_split_bf16_router(attn_output.shape[0]):
        router_dtype = None
    ffn_input, stream, router_input = self._fused_attn_boundary(
        stream,
        attn_output,
        branch_dtype=self.operator_dtype,
        router_dtype=router_dtype,
    )
    if self.is_moe_layer:
        ffn_output = self.mlp(ffn_input, router_input, forward_batch)
    else:
        ffn_output = self.mlp(ffn_input)
    # A TP-sliced branch means the unpooler gather was deferred to the FFN boundary.
    if ffn_output.shape[-1] != stream.shape[-1]:
        next_attn_input, stream = self._fused_ffn_boundary.forward_with_ag(
            stream, ffn_output, ms.get_parallel().tp_group.ca_comm
        )
        return stream, next_attn_input
    if self._fused_ffn_boundary is None:
        return self.post_feedforward_residual_gate(
            stream, self.post_feedforward_layernorm(ffn_output)
        ), None
    next_attn_input, stream, _ = self._fused_ffn_boundary(
        stream, ffn_output, branch_dtype=self.operator_dtype
    )
    return stream, next_attn_input


def _split_forward(cls):
    """``cls._forward`` = main's ``forward`` (same function object); ``forward`` calls ``self._forward``."""
    main_forward = cls.forward

    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        return self._forward(input_ids, positions, forward_batch, input_embeds)

    forward.__qualname__ = f"{cls.__qualname__}.forward"
    cls._forward = main_forward
    cls.forward = forward


def install() -> list:
    """Install the shims once per process on main's layout at a listed revision; returns what was installed."""
    from . import bed, sgl_compat

    if _installed or sgl_compat.layout() != "main":
        return list(_installed)
    got = bed.file_sha256(MODEL_RELPATH)
    if not bed.sha_ok(got, MUSE_SPARK_MAIN):
        bed.report(
            "sgl_shims",
            False,
            reason=f"{MODEL_RELPATH} sha256 {(got or '<missing>')[:16]} not a checked revision",
        )
        return []
    ms = importlib.import_module("sglang.srt.models.muse_spark")
    nr = importlib.import_module(
        "sglang.kernels.ops.layernorm.muse_spark_norm_residual"
    )
    if "_forward" not in vars(ms.MuseSpark1xModel):
        _split_forward(ms.MuseSpark1xModel)
        _installed.append("MuseSpark1xModel._forward")
    if "_forward" not in vars(ms.MuseSparkTextForCausalLM):
        _split_forward(ms.MuseSparkTextForCausalLM)
        _installed.append("MuseSparkTextForCausalLM._forward")
    if not hasattr(ms, "MuseSpark1xForCausalLM"):
        ms.MuseSpark1xForCausalLM = ms.MuseSparkTextForCausalLM
        _installed.append("MuseSpark1xForCausalLM")
    if "gated_attention" not in vars(ms.MuseSpark1xAttention):
        ms.MuseSpark1xAttention.gated_attention = _gated_attention
        ms.MuseSpark1xAttention.forward = _attention_forward
        _installed.append("MuseSpark1xAttention.gated_attention")
    if not hasattr(nr.FusedPostNormResidualPreNorm, "forward_with_ag"):
        nr.FusedPostNormResidualPreNorm.forward_with_ag = _forward_with_ag
        ms.MuseSpark1xDecoderLayer.forward = _decoder_layer_forward
        _installed.append("FusedPostNormResidualPreNorm.forward_with_ag")
    if not hasattr(ms.MuseSpark1xDecoderLayer, "_megakernel_capable"):
        ms.MuseSpark1xDecoderLayer._megakernel_capable = False
        _installed.append("MuseSpark1xDecoderLayer._megakernel_capable")
    bed.report("sgl_shims", True, installed=list(_installed))
    return list(_installed)

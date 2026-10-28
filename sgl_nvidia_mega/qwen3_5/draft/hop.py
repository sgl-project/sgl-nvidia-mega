# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0): python/sglang/srt/layers/layer_boundary/fusions/q35_draft_hop.py
# (DraftHopFusion) and python/sglang/srt/models/qwen3_5.py
# (Qwen3_5AttentionDecoderLayer._draft_o_proj), on branch nvidia/qwen35-agentx-gb300.
"""The one-row MTP draft step's two all-reduce + residual + norm points as hops on the
target's decode hop, and the draft o_proj GEMV that pushes into the first one.

- ``ffn_input``: the attention output's sum, the residual add and the post-attention
  norm, with FlashInfer's legacy MNNVL one-shot rounding (the norm's folded Gemma
  multiplier ``gemma_weight``);
- ``final_norm``: the MoE output's sum, the residual add and the final norm, with the
  rounding of SGLang's one-shot push all-reduce and FlashInfer's CuTe DSL
  FusedAddRMSNorm (multiplier 1 + the raw weight, in fp32);
- ``o_proj``: the attention output's one-row o_proj (fp32 accumulation), pushed into
  ``ffn_input``'s hop, which then only consumes; with the output gate it stages
  ``attn * sigmoid(gate)`` itself, in place of SGLang's gate kernel.

The draft shares the target's hop (``boundary.hop.active()``): one mailbox and one
stage order for every hop of the process, as SGLang runs the draft after the target's
build on every rank alike.
"""

from __future__ import annotations

from typing import Optional

import torch

from sgl_nvidia_mega.qwen3_5.sm103.draft import hop as kernels

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/layers/layernorm.py": ('"gemma_weight"', "weight", "variance_epsilon"),
}

# Draft steps carry one row; the verify and extend widths keep their own kernels.
MAX_M = 1
# Whether the attention output's o_proj runs as the one-row GEMV that pushes into the FFN
# input's hop, per TP size, as SGLang branch nvidia/qwen35-agentx-gb300 runs them: TP8 does
# (SGLANG_Q35MF_MKD_GEMV); TP4 runs SGLang's o_proj and the hop publishes its output (that
# branch's TP4 setup also pushes the single-launch MoE into the final norm's hop,
# SGLANG_Q35MF_MKD_HOP_PUSH, which changes no bits and is not ported).
GEMV_O_PROJ = {8: True, 4: False}


class DraftHop:
    """The draft layer's hop points on the target's hop. ``early``: the FFN input's
    consume releases its dependents at entry when the next launch is the single-launch
    draft MoE; ``early_head``: the final norm's does when the next launch is the draft
    head (both read only static data before their PDL wait)."""

    def __init__(
        self,
        hop,
        *,
        early: bool = True,
        early_head: bool = False,
        delay_ns: int = 0,
        gemv: bool = True,
    ):
        self.hop = hop  # the target's sm103.hop.Hop
        self.gemv = gemv  # the o_proj GEMV built: o_proj() pushes into ffn_input's hop
        self.early = early
        self.early_head = early and early_head
        self.delay_ns = delay_ns  # diagnostic: hold the consume's output stores
        # (data_ptr, shape) of a partial a producer already pushed; its all-reduce
        # only consumes
        self._pushed: Optional[tuple[int, torch.Size]] = None

    @classmethod
    def build(
        cls,
        *,
        tp_size: int,
        deadline_s: float,
        o_proj_k: Optional[int] = None,
        early: bool = True,
        early_head: bool = False,
    ) -> tuple[Optional[DraftHop], str]:
        """(the draft hop, "") on the target's hop, compiling its kernels (and the o_proj
        GEMV for reduction length ``o_proj_k``), or (None, why not)."""
        from sgl_nvidia_mega.qwen3_5.boundary.hop import active

        hop = active()
        if hop is None:
            return None, "the target built no decode hop (slot hop off)"
        if hop.world != tp_size or not hop.supports(MAX_M):
            return None, (
                f"the target's hop serves TP {hop.world}, 1-{hop.max_m} rows, "
                f"not TP {tp_size}, {MAX_M} row"
            )
        kernels.build(
            hop.device.index,
            int(deadline_s * 1e9),
            gemv_ks=() if o_proj_k is None else (o_proj_k,),
        )
        return (
            cls(hop, early=early, early_head=early_head, gemv=o_proj_k is not None),
            "",
        )

    def supports(self, m: int) -> bool:
        return 1 <= m <= MAX_M and self.hop.supports(m)

    def attn_push_target(self, m: int) -> Optional[DraftHop]:
        """This hop when the attention output's producer may push its m rows into it,
        else None."""
        return self if self.supports(m) else None

    def mark_pushed(self, partial: torch.Tensor) -> None:
        """A producer kernel already pushed this bf16 partial; its all-reduce only
        consumes."""
        if self._pushed is not None:
            raise RuntimeError("draft hop: a pushed partial was never consumed")
        self._pushed = (partial.data_ptr(), partial.shape)

    def has_pushed(self, partial: torch.Tensor) -> bool:
        return self._pushed == (partial.data_ptr(), partial.shape)

    def _take_pushed(self, partial: torch.Tensor) -> bool:
        if self.has_pushed(partial):
            self._pushed = None
            return True
        if self._pushed is not None:
            raise RuntimeError("draft hop: a pushed partial was never consumed")
        return False

    def o_proj(
        self,
        attn_output: torch.Tensor,
        weight: torch.Tensor,
        gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The one-row o_proj partial [1, N] bf16 (``weight`` [N, K] bf16), pushed into
        the FFN input's hop when that hop takes the row; with ``gate`` [1, K] its input
        is ``attn_output`` * sigmoid(``gate``)."""
        out = torch.empty(
            1, weight.shape[0], dtype=torch.bfloat16, device=weight.device
        )
        if self.attn_push_target(int(attn_output.shape[0])) is None:
            return kernels.draft_gemv(attn_output, weight, out, gate=gate)
        kernels.draft_gemv(attn_output, weight, out, hop_site=self.hop.site, gate=gate)
        self.mark_pushed(out)
        return out

    def ffn_input(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        norm,
        *,
        moe_single_launch: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(the MoE input, residual) of the attention output's partial
        ``hidden_states``; ``norm`` is the layer's post-attention GemmaRMSNorm."""
        return kernels.draft_all_reduce_rms_norm(
            self.hop,
            local_contribution=hidden_states,
            residual=residual,
            gamma=norm.gemma_weight,
            eps=norm.variance_epsilon,
            weight_bias=0.0,
            add_norm=False,
            early_release=self.early and moe_single_launch,
            delay_ns=self.delay_ns,
            pushed=self._take_pushed(hidden_states),
        )

    def final_norm(
        self,
        partial: torch.Tensor,
        residual: torch.Tensor,
        norm,
        *,
        head_follows: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(the final hidden state, residual) of the MoE output's unreduced
        ``partial``; ``norm`` is the model's final GemmaRMSNorm."""
        return kernels.draft_all_reduce_rms_norm(
            self.hop,
            local_contribution=partial,
            residual=residual,
            gamma=norm.weight,
            eps=norm.variance_epsilon,
            weight_bias=1.0,
            add_norm=True,
            early_release=self.early_head and head_follows,
            delay_ns=self.delay_ns,
            pushed=self._take_pushed(partial),
        )

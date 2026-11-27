# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

MIN_ROWS = 70
MAX_ROWS = 128
STAGES = 3
BOX_ROWS = 128
HIDDEN = 4096

SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py": (
        "VerifyShardHeadView",
        "hidden_states",
        "weight",
        "out",
    ),
    "srt/layers/logits_processor.py": (
        "logit_scale",
        "use_fp32_lm_head",
        "use_fp32_lm_head_output",
    ),
    "srt/layers/vocab_parallel_embedding.py": (
        "weight",
        "shard_indices",
        "org_vocab_start_index",
        "num_org_elements",
        "num_added_elements_padded",
    ),
}

# logits buffer identity (address, shape, strides) -> (buffer, keys, ctas, epoch)
_STASH: dict = {}


def _ident(t: torch.Tensor) -> tuple:
    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype)


def lookup(values) -> Optional[tuple]:
    """(keys, ctas, epoch) the keys build leaves for this logits buffer, or None."""
    if not _STASH or not isinstance(values, torch.Tensor):
        return None
    entry = _STASH.get(_ident(values))
    return None if entry is None else entry[1:]


class VerifyHeadMma:
    def __init__(
        self, weight: torch.Tensor, *, vocab_start: int, valid: int, keys: bool = False
    ):
        from sgl_nvidia_mega.qwen3_5.sm103.head import mma

        if (
            weight.dtype != torch.bfloat16
            or weight.dim() != 2
            or weight.shape[1] != HIDDEN
        ):
            raise ValueError(f"verify head weight {weight.dtype} {tuple(weight.shape)}")
        if not weight.is_contiguous() or weight.data_ptr() % 16:
            raise ValueError(
                "verify head weight must be contiguous and 16-byte aligned"
            )
        self.weight = weight
        self.vocab_start = int(vocab_start)
        self.valid = int(valid)
        self.device = weight.device
        self.keys = keys
        self._logged: set = set()
        # the launch's grid (box rows fixed, cluster 1)
        self.ctas = -(-int(weight.shape[0]) // BOX_ROWS)
        self._bufs: dict = {}
        # the keys build's {epoch, head ticket, keys ticket, 0}
        self._epoch = torch.zeros(4, dtype=torch.int32, device=self.device)
        # every width compiled now: no compile inside a capture
        for m in range(MIN_ROWS, MAX_ROWS + 1):
            n_tile, chunks, stages, c = mma.variant(m, 1, STAGES)
            mma._kernel(self.device.index, n_tile, chunks, stages, c, False)
            if self.keys:
                mma._kernel(self.device.index, n_tile, chunks, stages, c, True, True)

    def _decline_reason(self, view) -> str:
        hidden, out = view.hidden_states, view.out
        m = int(hidden.shape[0]) if hidden.dim() == 2 else -1
        rows = int(self.weight.shape[0])
        if not MIN_ROWS <= m <= MAX_ROWS:
            return "rows"
        if view.weight is not self.weight:
            return "weight"
        if out is not None and (
            out.dtype != torch.bfloat16 or tuple(out.shape) != (m, rows)
        ):
            return "output buffer"
        if out is not None and (
            not out.is_contiguous() or out.device != self.device or out.data_ptr() % 16
        ):
            return "output layout"
        if (
            hidden.dtype != torch.bfloat16
            or hidden.shape[1] != HIDDEN
            or hidden.stride(1) != 1
            or (hidden.stride(0) * 2) % 16
            or hidden.data_ptr() % 16
            or hidden.device != self.device
        ):
            return "hidden rows"
        return ""

    def __call__(self, view) -> Optional[torch.Tensor]:
        from sgl_nvidia_mega.qwen3_5.sm103.head import mma

        hidden, out = view.hidden_states, view.out
        m = int(hidden.shape[0]) if hidden.dim() == 2 else -1
        why = self._decline_reason(view)
        if why:
            if isinstance(out, torch.Tensor):
                _STASH.pop(_ident(out), None)
                if self.keys and _ident(out) in self._bufs:
                    # unspent keys of this buffer go stale: SGLang's GEMM writes it now
                    self._epoch[:1].add_(1)
            if m >= MIN_ROWS and ("decline", why) not in self._logged:
                self._logged.add(("decline", why))
                logger.info(
                    "NVIDIA megakernel: Qwen3.5 slot verify_head declines at %d rows "
                    "(%s); the GEMM runs",
                    m,
                    why,
                )
            return None
        pre = None
        if out is None:
            out = torch.empty(
                (m, int(self.weight.shape[0])), dtype=torch.bfloat16, device=self.device
            )
        elif self.keys:
            pre = self._bufs.get(_ident(out))
            if pre is None:
                # keys [2, m, ctas], then the epoch tag (-1: never written)
                pre = self._bufs[_ident(out)] = torch.full(
                    (2 * m * self.ctas + 1,), -1, dtype=torch.int64, device=self.device
                )
        mma.head_mma_keys(
            hidden,
            self.weight,
            None,
            vocab_start=self.vocab_start,
            valid=self.valid,
            nan_hi=0,
            logits=out,
            box_rows=BOX_ROWS,
            cluster=1,
            stages=STAGES,
            pre=pre,
            epoch=None if pre is None else self._epoch,
        )
        if pre is not None:
            _STASH[_ident(out)] = (out, pre, self.ctas, self._epoch)
        if (m, pre is not None) not in self._logged:
            self._logged.add((m, pre is not None))
            logger.info(
                "NVIDIA megakernel: Qwen3.5 slot verify_head runs at %d rows%s",
                m,
                " with the sampled keys (verify_head_keys)" if pre is not None else "",
            )
        return out


PLAIN_HEAD_METHODS = (
    "UnquantizedEmbeddingMethod",
    "UnquantizedLinearMethod",
    "NoneType",
)
# what a quantized head carries beside (or in place of) a plain weight
QUANT_ATTRS = (
    "weight_scale",
    "weight_scale_2",
    "weight_scale_inv",
    "weight_global_scale",
    "weight_scale_interleaved",
    "input_scale",
    "input_scale_inv",
    "alpha",
    "workspace",
    "qweight",
    "qzeros",
    "scales",
    "g_idx",
)


def head_refusal(processor, head) -> str:
    if hasattr(head, "set_lora") or hasattr(head, "apply_lora"):
        return "a LoRA LM head"
    method = type(getattr(head, "quant_method", None)).__name__
    if method not in PLAIN_HEAD_METHODS:
        return f"a quantized LM head ({method})"
    carried = [name for name in QUANT_ATTRS if getattr(head, name, None) is not None]
    if carried:
        return f"a quantized LM head ({method} with {', '.join(carried)})"
    if getattr(head, "bias", None) is not None:
        return "an LM head bias"
    weight = getattr(head, "weight", None)
    if type(weight) not in (torch.Tensor, torch.nn.Parameter):
        return f"the LM head weight is a {type(weight).__name__}, not a plain tensor"
    if (
        weight.dtype != torch.bfloat16
        or weight.dim() != 2
        or int(weight.shape[1]) != HIDDEN
    ):
        return (
            f"the LM head weight is {weight.dtype} {tuple(weight.shape)}, "
            f"not bf16 [rows, {HIDDEN}]"
        )
    if not weight.is_cuda:
        return "the LM head weight is not on a CUDA device"
    if not weight.is_contiguous() or weight.data_ptr() % 16:
        return "the LM head weight is not contiguous and 16-byte aligned"
    shard = head.shard_indices
    if int(shard.num_added_elements_padded) or int(shard.num_org_elements) != int(
        weight.shape[0]
    ):
        return "the shard has padded columns"
    if getattr(processor, "logit_scale", None) is not None:
        return "a logit scale"
    if getattr(processor, "use_fp32_lm_head", False) or getattr(
        processor, "use_fp32_lm_head_output", False
    ):
        return "an fp32 LM head or head output"
    return ""


def build(model, *, keys: bool) -> VerifyHeadMma:
    head = model.lm_head
    shard = head.shard_indices
    built = VerifyHeadMma(
        head.weight,
        vocab_start=int(shard.org_vocab_start_index),
        valid=int(shard.num_org_elements),
        keys=keys,
    )
    logger.info(
        "NVIDIA megakernel: Qwen3.5 slot verify_head built (%d-%d rows%s)",
        MIN_ROWS,
        MAX_ROWS,
        ", with verify_head_keys" if keys else "",
    )
    return built

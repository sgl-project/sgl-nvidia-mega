# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/models/minimax_m3_vl.py (the tail of MiniMaxM3SparseForConditionalGeneration.forward).
"""The LM head and the logits of a covered step, with the DSpark captures.

Stock (``logits``, the step the stock executor runs): SGLang's logits processor on
SGLang's LM head.

Slot lm_head_mxfp8 (opt-in, one request's 8-row step, where slot mega runs): the logits
processor gets a stand-in for the LM head (``Mxfp8LmHead``) whose quant method streams
an MXFP8 copy of this rank's vocabulary shard (``sm103/logits/lm_head_mxfp8.py``) where
SGLang's ``_compute_lm_head`` runs its BF16 GEMM on the shard. The projection gives the
same BF16 [rows, shard] tensor, so the cross-rank gather, the FP32 cast, the logits'
processing and the DSpark captures that follow stay SGLang's. ``head_reasons`` refuses
the heads SGLang would project another way; ``build`` makes the copy and the kernel at
load and checks the stand-in against SGLang's own projection on every rank (its bound:
``spec.LM_HEAD_TOLERANCES``).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

from sgl_nvidia_mega.minimax_m31 import ROWS
from sgl_nvidia_mega.minimax_m31.runtime import parity

logger = logging.getLogger(__name__)

SLOT = "lm_head_mxfp8"

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/models/minimax_m3_vl.py": (
        "capture_aux_hidden_states",
        "logits_processor",
        "lm_head",
    ),
    # slot lm_head_mxfp8: the stand-in's route (SGLang applies a quant method outside
    # its unquantized names; test/unit/test_minimax_m31_lm_head_mxfp8.py drives the
    # real one), the head attributes that route reads by name, and the processor's
    # switches that pick another projection
    "srt/layers/logits_processor.py": (
        "_compute_lm_head",
        "_compute_lm_head(hidden_states)",
        "_compute_lm_head(lm_head)",
        "should_apply_lm_head_quant_method",
        "should_apply_lm_head_quant_method(lm_head)",
        "should_apply_lm_head_quant_method(quant_method)",
        "_UNQUANTIZED_LM_HEAD_METHODS",
        '"quant_method"',
        '"weight"',
        '"set_lora"',
        '"apply_lora"',
        "use_fp32_lm_head",
        "use_fp32_lm_head_output",
        "rl_on_policy_target",
    ),
    "srt/runtime_context.py": ("get_parallel", '"tp_group"'),
    "srt/distributed/parallel_state.py": ("world_size", "cpu_group"),
}

# The SGLang functions slot lm_head_mxfp8's stand-in is routed by, by the digest of
# their code (test/unit/test_contract.py recomputes them on the installed SGLang):
# the stand-in replaces the last branch of _compute_lm_head's "normal linear layer"
# (the BF16 torch.matmul on lm_head.weight.T), which head_reasons checks SGLang takes.
# The second digest is SGLang with a split-K Triton head for gfx1250 inside the
# FP32-output branch, which head_reasons refuses: the mirrored branch is unchanged
SGLANG_MIRRORS = {
    "srt/layers/logits_processor.py:LogitsProcessor._compute_lm_head": (
        "b4fe8c96270053f7",
        "d6485ed1c8a88abe",
    ),
    "srt/layers/logits_processor.py:should_apply_lm_head_quant_method": (
        "5699b58992f22e41"
    ),
}

# the seed of the rows build checks the stand-in on
CHECK_SEED = 0


def logits(st, hidden, aux) -> Any:
    """SGLang's ``LogitsProcessorOutput`` for the step."""
    lm = st.lm
    # MiniMaxM3Model.forward returns the captures only when it took some
    aux_hidden = aux if lm.capture_aux_hidden_states and len(aux) != 0 else None
    return lm.logits_processor(st.input_ids, hidden, lm.lm_head, st.fb, aux_hidden)


# ------------------------------------------------------------ slot lm_head_mxfp8


class Mxfp8LmHeadMethod:
    """The stand-in's quant method. SGLang's ``_compute_lm_head`` calls ``apply`` for
    any method outside its unquantized names, ahead of its own projections (so
    ``head_reasons`` refuses the heads that would take another one than the BF16
    GEMM). It takes the rows as that GEMM does (cast to the weight's BF16) and serves
    the kernel's row counts only: any other input raises."""

    def __init__(self, copy):
        # sm103.logits.lm_head_mxfp8.Mxfp8Weight: this rank's shard
        self.copy = copy
        self.captured = False

    def apply(self, layer, x: torch.Tensor, bias=None) -> torch.Tensor:
        from sgl_nvidia_mega.minimax_m31.sm103.logits import lm_head_mxfp8

        if bias is not None:
            raise ValueError(f"slot {SLOT} projects without an embedding bias")
        if _capturing() and not self.captured:
            # a replay runs no Python: the capture is where the slot serves
            self.captured = True
            logger.info(
                "NVIDIA megakernel: %s captured at %d rows", SLOT, int(x.shape[0])
            )
        # SGLang's GEMM casts its rows to the weight's dtype; the stream reads them
        # in place, so dense (both no launch for the final norm's BF16 rows)
        x = x.to(torch.bfloat16)
        if not x.is_contiguous():
            x = x.contiguous()
        return lm_head_mxfp8.matmul(x, self.copy)


class Mxfp8LmHead:
    """What the logits processor gets for the LM head at a step slot lm_head_mxfp8
    serves: SGLang's module, with the MXFP8 stream as its quant method."""

    def __init__(self, lm_head, copy):
        self.lm_head, self.weight = lm_head, lm_head.weight
        self.quant_method = Mxfp8LmHeadMethod(copy)
        # set by build: the stand-in's logits against SGLang's projection on the
        # check's rows, over SGLang's largest |logit|
        self.rel: Optional[float] = None

    def __getattr__(self, name):
        if name == "lm_head":  # not built yet: no module to ask
            raise AttributeError(name)
        return getattr(self.lm_head, name)

    def device_bytes(self) -> int:
        from sgl_nvidia_mega.minimax_m31.sm103.logits import lm_head_mxfp8

        return lm_head_mxfp8.nbytes(self.quant_method.copy)

    def summary(self) -> str:
        n, k = self.quant_method.copy.shape
        return (
            f"the {ROWS}-row step's LM head streams an MXFP8 copy of this rank's "
            f"[{n}, {k}] shard, {self.device_bytes() / 2**20:.1f} MiB of device memory"
        )

    def log_parity(self, model, hidden) -> None:
        """Under the parity switch (test only, eager steps): this rank's shard logits
        from the stream against SGLang's BF16 projection of the same hidden rows, one
        ``NVIDIA megakernel parity`` line (its ``rel`` against
        ``spec.LM_HEAD_TOLERANCES``). The step's own line compares the whole step,
        slot mega's layers included."""
        if parity.mode() == "0" or _capturing():
            return
        proc = model.logits_processor
        want = proc._compute_lm_head(hidden, self.lm_head)
        got = proc._compute_lm_head(hidden, self)
        parity.log_parity(f"{SLOT} shard logits", want, got)


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def head_reasons(model) -> list[str]:
    """Why the MXFP8 stream cannot stand in for this model's LM-head projection ([]
    when it can): SGLang must run the BF16 GEMM of ``_compute_lm_head`` on a shard the
    kernel takes, so that the stream's BF16 [rows, shard] output is what follows
    expects."""
    from sglang.srt.layers.logits_processor import should_apply_lm_head_quant_method

    from sgl_nvidia_mega.minimax_m31.sm103.logits import lm_head_mxfp8

    head, proc = model.lm_head, model.logits_processor
    weight = getattr(head, "weight", None)
    if not isinstance(weight, torch.Tensor):
        return [f"the LM head {type(head).__name__} has no weight"]
    reasons = []
    if hasattr(head, "set_lora") and hasattr(head, "apply_lora"):
        reasons.append("the LM head is LoRA-wrapped")
    if should_apply_lm_head_quant_method(head, getattr(head, "quant_method", None)):
        reasons.append("SGLang runs the LM head through its quant method")
    if proc.use_fp32_lm_head or proc.use_fp32_lm_head_output:
        reasons.append("SGLang computes the LM head's logits in FP32")
    if proc.rl_on_policy_target is not None:
        reasons.append("SGLang's on-policy target projection")
    shape = lm_head_mxfp8.unsupported_reason(weight)
    if shape:
        reasons.append(shape)
    elif not weight.is_contiguous():
        reasons.append("the LM head weight is not contiguous")
    return reasons


def _cpu_group():
    """The TP ranks' CPU process group (None on one rank)."""
    from sglang.srt.runtime_context import get_parallel

    group = get_parallel().tp_group
    return None if group.world_size == 1 else group.cpu_group


def build(model, widths, slots) -> Optional[Mxfp8LmHead]:
    """Slot lm_head_mxfp8's stand-in (None when the slot is not requested, or the
    widths miss one request's rows): this rank's MXFP8 copy of the shard and the
    stream's kernel, made here, at load, and checked against SGLang's projection. A
    collective: if any rank fails, every rank raises with each rank's error."""
    if SLOT not in slots or ROWS not in widths:
        return None
    from sgl_nvidia_mega.minimax_m31.sm103 import host

    group = _cpu_group()
    what = f"building slot {SLOT}"
    head = (
        _make(model) if group is None else host.agree(group, lambda: _make(model), what)
    )
    logger.info(
        "MiniMax-M3.1 megakernel: slot %s: %s; on %d seeded rows its logits are "
        "within %.2e of SGLang's BF16 projection's largest",
        SLOT,
        head.summary(),
        ROWS,
        head.rel,
    )
    return head


@torch.no_grad()
def _make(model) -> Mxfp8LmHead:
    """One rank's copy, kernel and stand-in; raises with the reason it cannot serve."""
    from sgl_nvidia_mega.minimax_m31.sm103.logits import lm_head_mxfp8

    reasons = head_reasons(model)
    if reasons:
        raise ValueError("; ".join(reasons))
    weight = model.lm_head.weight.detach()
    copy = lm_head_mxfp8.quantize(weight)
    reason = lm_head_mxfp8.scale_reason(copy)
    if reason:
        raise ValueError(f"the LM head weight holds {reason}")
    lm_head_mxfp8.precompile(*copy.shape, weight.device)
    head = Mxfp8LmHead(model.lm_head, copy)
    head.rel = check_against_sglang(model, head)
    return head


def check_against_sglang(model, head: Mxfp8LmHead) -> float:
    """The stand-in's logits against SGLang's head through SGLang's own
    ``_compute_lm_head`` on one request's rows of seeded Gaussian values: the same
    dtype and shape, and their largest difference over SGLang's largest |logit| (the
    ratio returned) within ``spec.LM_HEAD_TOLERANCES``; raises otherwise."""
    from sgl_nvidia_mega.minimax_m31.spec import LM_HEAD_TOLERANCES

    bound = LM_HEAD_TOLERANCES[f"{SLOT}.logits.rel"]
    proc, w = model.logits_processor, head.weight
    gen = torch.Generator(device=w.device).manual_seed(CHECK_SEED)
    x = torch.randn(ROWS, w.shape[1], generator=gen, device=w.device)
    x = x.to(torch.bfloat16)
    ours = proc._compute_lm_head(x, head)
    theirs = proc._compute_lm_head(x, model.lm_head)
    if ours.dtype != theirs.dtype or ours.shape != theirs.shape:
        raise ValueError(
            f"the MXFP8 stream gives {ours.dtype} {tuple(ours.shape)}, SGLang's head "
            f"{theirs.dtype} {tuple(theirs.shape)}"
        )
    peak = theirs.float().abs().max()
    rel = ((ours.float() - theirs.float()).abs().max() / peak).item()
    if not rel <= bound:
        raise ValueError(
            f"the MXFP8 stream's logits differ by {rel:.2e} of SGLang's largest on "
            f"{ROWS} seeded rows (bound {bound:.2e})"
        )
    return rel

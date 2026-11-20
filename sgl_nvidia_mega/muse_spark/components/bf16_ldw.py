# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``bf16_ldw`` (env ``MUSE_VR_BF16_LDW=1``, default off): the routed experts of the checkpoint's BF16-expert layers
on MoE-LD-W BF16 (``ops.moe_ldwb``, K-half relay build, 32-row passes, 5-stage ring) for verify batches of
``MUSE_VR_BF16_LDW_MIN`` (default 64) .. 512 token rows per rank.

Hooks (AROUND, by attribute replacement):

* ``MuseSpark1xBf16Experts.prepare_trtllm_weights``: after SGLang shuffles a layer's BF16 expert stacks into BlockMajorK, the
  ``gate_up_proj`` / ``down_proj`` stacks move into the relay placement (``ops.moe_ldwb.place_relay``: same shape, strides
  and bytes; per expert, the first K half is backed by one locality domain and the second half by the other, alternating by
  expert parity) when the device has two locality domains, and the layer gets its own MoE-LD-W BF16 scratch (so a layer's
  scratch is reused only by the same layer's call of a later step). This runs at load time, before any CUDA graph is captured. Layers on another expert path (Triton, MegaMoE) are left unchanged.
* ``MuseSpark1xBf16Experts._forward_trtllm``: marks the layer for the duration of the call (thread-local).
* ``flashinfer.fused_moe.trtllm_bf16_routed_moe``: a call made from a marked, prepared layer with the decode argument set
  (packed int32 top-k ids, top-k 8, SwiGLU, ``do_finalize=False``, routed scaling 1.0, no expert groups, MIN..512 rows, at
  most 128 local experts) runs ``moe_ldwb`` and returns the same ``(gemm2_out, expert_weights, expanded_idx_to_permuted_idx)``
  triple; every other call runs FlashInfer unchanged on the same tensors.

Numerics: every routed FC2 row equals FlashInfer's routed BF16 MoE (default tactic) bitwise for the same inputs
(``tests/test_moe_ldwb.py``); SGLang's finalize reads the rows through the returned map.
Shadow (``MUSE_VR_MOE_LD_SHADOW=1``): calls made outside CUDA graph capture also run FlashInfer and compare every routed row
and the top-k weights bitwise; a mismatch turns the component off for the rest of the process.
Steps aside: another argument set, FlashInfer autotuning, sequence-parallel MoE, two-batch or single-batch overlap, a layer
whose weights were not prepared, a disabled component. Memory: one relay scratch per prepared layer (fp32 accumulator tiles).
"""

from __future__ import annotations

import logging
import threading

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/models/muse_spark.py": ("get_forward",),
    "srt/layers/muse_spark/muse_spark_bf16_moe.py": ("MuseSpark1xBf16Experts",),
    "srt/layers/moe/utils.py": ("is_sbo_enabled", "is_tbo_enabled"),
}


logger = logging.getLogger("muse_vr_plugin")

NAME = "bf16_ldw"
PINS = {
    "sglang/srt/layers/muse_spark_bf16_moe.py": (
        "687680077a7542cb1e2026d321c2f18a650fb56a654b50434acbf7be9c1aa6c5",
        "57ae1593b4496edfc658343d9ca039dead111484d78ec57ab3eaa14891f6afed",
    ),
}
_RING = ("-DMUSE_VR_LDW_PASS_ROWS=32", "-DMUSE_VR_LDW_STAGES=5")
_tls = threading.local()
_state: dict = {
    "engaged": None,
    "orig": None,
    "scratch": {},
    "built": set(),
    "prepared": 0,
    "skipped": "",
    "off": "",
    "min_m": 64,
}


def _layer_reason(layer) -> str:
    """Empty when this BF16 expert layer can use the kernel, else why not."""
    import torch

    from .ops import moe_ldwb

    if not getattr(layer, "use_trtllm", False) or getattr(layer, "use_megamoe", False):
        return "not the TRT-LLM BF16 expert path"
    w13, w2 = getattr(layer, "gate_up_proj", None), getattr(layer, "down_proj", None)
    if w13 is None or w2 is None:
        return "no expert weights"
    for name, w, rows in (
        ("gate_up_proj", w13, 2 * moe_ldwb.INTER),
        ("down_proj", w2, moe_ldwb.HIDDEN),
    ):
        if (
            w.dtype != torch.bfloat16
            or w.dim() != 4
            or tuple(w.shape[1:]) != (64, rows, 64)
            or not w.is_contiguous()
        ):
            return f"{name} {tuple(w.shape)} {w.dtype}"
    if w13.shape[0] > moe_ldwb.MAX_POOL or w2.shape[0] != w13.shape[0]:
        return f"{w13.shape[0]} expert slots"
    return ""


def _prepare_layer(layer) -> None:
    """Relay placement of the shuffled BF16 expert stacks + the layer's scratch (load time)."""
    import torch

    from . import ld_layout
    from .ops import moe_ldwb

    if layer.gate_up_proj.data_ptr() in _state["scratch"]:
        return  # already placed (a repeated preparation call): never copy twice
    why = _layer_reason(layer)
    if why:
        _state["skipped"] = why
        return
    dev = layer.gate_up_proj.device
    if ld_layout.domain_count(dev) != 2:
        _state["skipped"] = "one locality domain"
        return
    with torch.no_grad():
        for name in ("gate_up_proj", "down_proj"):
            p = getattr(layer, name)
            placed = moe_ldwb.place_relay(p.data)
            p.data = placed
            del placed
            torch.cuda.empty_cache()  # the replaced storage goes back before the next layer's placement
    dix = dev.index if dev.index is not None else torch.cuda.current_device()
    if dix not in _state["built"]:
        moe_ldwb.preload(_RING + moe_ldwb.RELAY)
        _state["built"].add(dix)
    _state["scratch"][layer.gate_up_proj.data_ptr()] = moe_ldwb.Scratch(
        dev, 32, relay=True
    )
    _state["prepared"] += 1


def _tuning() -> bool:
    try:
        from flashinfer.autotuner import AutoTuner

        return bool(AutoTuner.get().is_tuning_mode)
    except Exception:  # noqa: BLE001
        return False


def _blocked() -> bool:
    """Sequence-parallel MoE or two-batch / single-batch overlap (unknown configuration: blocked)."""
    try:
        import sglang.srt.models.muse_spark as ms

        if bool(getattr(ms.get_forward(), "sp_active", False)):
            return True
    except Exception:  # noqa: BLE001
        return True
    try:
        from sglang.srt.layers.moe.utils import is_sbo_enabled, is_tbo_enabled

        return bool(is_tbo_enabled() or is_sbo_enabled())
    except Exception:  # noqa: BLE001
        return True


def _call(kw):
    """MoE-LD-W BF16 for an engaged-layer call, else None."""
    import torch

    from .ops import moe_ldwb

    if (
        kw.get("do_finalize", True) is not False
        or kw.get("activation_type") != 3
        or kw.get("routing_method_type") != 1
    ):
        return None
    if (
        kw.get("top_k") != moe_ldwb.TOP_K
        or kw.get("intermediate_size") != moe_ldwb.INTER
        or kw.get("n_group")
        or kw.get("topk_group")
    ):
        return None
    rsf = kw.get("routed_scaling_factor")
    if rsf is not None and float(rsf) != 1.0:
        return None
    packed, hs = kw.get("topk_ids"), kw.get("hidden_states")
    if (
        not isinstance(packed, torch.Tensor)
        or packed.dtype != torch.int32
        or packed.dim() != 2
        or not packed.is_contiguous()
    ):
        return None
    m = packed.shape[0]
    if (
        not (_state["min_m"] <= m <= moe_ldwb.MAX_TOKENS)
        or packed.shape[1] != moe_ldwb.TOP_K
    ):
        return None
    if (
        hs is None
        or hs.dtype != torch.bfloat16
        or hs.dim() != 2
        or hs.shape != (m, moe_ldwb.HIDDEN)
        or not hs.is_contiguous()
    ):
        return None
    w13, w2 = kw.get("gemm1_weights"), kw.get("gemm2_weights")
    p = int(kw.get("local_num_experts") or 0)
    if w13 is None or w2 is None or not (1 <= p <= w13.shape[0]):
        return None
    scratch = _state["scratch"].get(w13.data_ptr())
    if scratch is None or _tuning() or _blocked():
        return None
    out = moe_ldwb.moe_ldwb(
        packed,
        hs,
        w13[:p],
        w2[:p],
        int(kw["local_expert_offset"]),
        scratch,
        pdl=True,
        extra=_RING + moe_ldwb.RELAY,
    )
    bed.count(NAME, f"m{m}_p{p}")
    if bed.flag("moe_ld_shadow") and not torch.cuda.is_current_stream_capturing():
        from .moe_ld import _rows_equal

        ref = _state["orig"](**kw)[:3]
        ok = _rows_equal(out, ref)
        bed.shadow_result(NAME, ok)
        if not ok:
            _state["off"] = "shadow mismatch"
            logger.error(
                "muse_vr_plugin: bf16_ldw shadow mismatch; the component is off for this process"
            )
            return ref
    return out


def _routed(*args, **kw):
    if getattr(_tls, "layer", None) is not None and not args and not _state["off"]:
        out = _call(kw)
        if out is not None:
            return out
    return _state["orig"](*args, **kw)


def register() -> bool:
    if _state["engaged"] is not None:
        return _state["engaged"]
    if not bed.enabled(NAME):
        _state["engaged"] = False
        return False
    for rel, sha in PINS.items():
        if not bed.pinned(rel, sha):
            bed.report(NAME, False, reason=f"pin mismatch: {rel}")
            _state["engaged"] = False
            return False
    from . import fi_guard

    fi_guard.check(NAME)
    import flashinfer.fused_moe as ffm
    import sglang.srt.layers.muse_spark_bf16_moe as b16

    cls = b16.MuseSpark1xBf16Experts
    orig_prepare = cls.prepare_trtllm_weights
    orig_fwd = cls._forward_trtllm

    def prepare_trtllm_weights(self, *a, **k):
        out = orig_prepare(self, *a, **k)
        try:
            _prepare_layer(self)
        except Exception as e:  # noqa: BLE001 - an unprepared layer simply keeps FlashInfer
            _state["skipped"] = f"prepare failed: {type(e).__name__}: {e}"
        bed.report(
            NAME,
            _state["prepared"] > 0,
            layers=_state["prepared"],
            shadow=bed.flag("moe_ld_shadow"),
            skipped=_state["skipped"] or "-",
            min_m=_state["min_m"],
        )
        return out

    def _forward_trtllm(self, *a, **k):
        prev = getattr(_tls, "layer", None)
        _tls.layer = self
        try:
            return orig_fwd(self, *a, **k)
        finally:
            _tls.layer = prev

    _state["min_m"] = max(17, bed.env_int("MUSE_VR_BF16_LDW_MIN", 64))
    _state["orig"] = ffm.trtllm_bf16_routed_moe
    ffm.trtllm_bf16_routed_moe = _routed
    cls.prepare_trtllm_weights = prepare_trtllm_weights
    cls._forward_trtllm = _forward_trtllm
    _state["engaged"] = True
    logger.warning("muse_vr_plugin: bf16_ldw hooks installed")
    return True

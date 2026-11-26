# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``moe_ld`` (env ``MUSE_VR_MOE_LD=1``, default off): the routed NVFP4 experts of a decode / verify step in
one kernel launch per MoE layer (``moe_ld_ops.moe_ld``) for at most 16 token rows per rank.

Hooks (AROUND, by attribute replacement):

* ``MuseSparkNvFp4FusedMoEMethod.process_weights_after_loading``: after SGLang prepares a layer's expert weights, the
  shuffled ``gate_up_proj`` and ``down_proj`` tensors move into ``ld_layout.split_rows_copy`` placements (same shape,
  strides and bytes; each expert's row halves backed by the two locality domains when the device has two) and the
  layer gets its call-site workspace (``moe_ld_ops.Site``). This runs at load time, before any CUDA graph is captured.
* ``MuseSpark1xFp4Experts.forward``: marks the layer for the duration of the call (thread-local).
* ``flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe``: a call made from a marked, prepared layer with the decode
  argument set (packed top-k ids, 1..16 rows, top-k 8, SwiGLU, no biases or clamps, per-token activation scales,
  ``do_finalize=False``, at most 128 local experts) runs ``moe_ld`` and returns the same
  ``(gemm2_out, expert_weights, expanded_idx_to_permuted_idx)`` triple; every other call goes to FlashInfer unchanged.
  SGLang's finalize consumes the triple as before.

Numerics: every FC2 output row equals FlashInfer's bitwise for the same inputs (``tests/test_moe_ld.py``); the finalize
reads the rows through the returned map, so the layer output is bitwise identical.

Shadow (``MUSE_VR_MOE_LD_SHADOW=1``): calls made outside CUDA graph capture also run FlashInfer and compare every routed
row and the top-k weights bitwise (engagement record keys ``shadow:match`` / ``shadow:MISMATCH``). A mismatch turns the
component off for the rest of the process.

Steps aside: more than 16 rows or another argument set, FlashInfer autotuning, sequence-parallel MoE, two-batch or
single-batch overlap, a layer whose weights were not prepared, a disabled component.

Large verify batches (env ``MUSE_VR_MOE_LDW=1`` in addition, default off): an engaged-layer call with the same argument set
but ``MUSE_VR_MOE_LDW_MIN`` (default 64) .. 512 rows runs ``ops.moe_ldw.moe_ldw`` (one launch, one per-device scratch, the
same placed weights) instead of FlashInfer; every routed FC2 row bitwise equal (``tests/test_moe_ldw.py``). Below the
minimum and above 512 rows FlashInfer runs unchanged. Shadow comparison as for ``moe_ld``. Calls of at most
``MUSE_VR_MOE_LDW_HEAD_MAX_M`` rows (default 64) use the build that copies the first FC1 item's token rows by row index;
larger calls use the same file built without that path (``-DMUSE_VR_LDW_GATHER=0``). Both builds give the same rows.
``MUSE_VR_MOE_LDW_RING48=1`` (default off) runs MoE-LD-W calls of ``MUSE_VR_MOE_LDW_RING48_MIN_M`` (default 96) ..
``MUSE_VR_MOE_LDW_RING48_MAX_M`` (default 512) rows on the RING48 build instead (a 4-stage weight ring with passes of up to 48
token rows: more weight bytes in flight per CTA, experts with 33-48 rows in one pass; the same rows; its own per-device
scratch). Tally ``m<M>_p<P>_r48``.
``MUSE_VR_MOE_LDW_RING4=1`` (default off) runs MoE-LD-W calls of ``MUSE_VR_MOE_LDW_RING4_MIN_M`` (default 65) ..
``MUSE_VR_MOE_LDW_RING4_MAX_M`` (default 95) rows on the RING4 build (``ops.moe_ldw.RING4``: 32-row passes, the same 4-stage ring;
the same rows; its own per-device scratch). The RING48 range takes precedence where the two overlap. Tally ``m<M>_p<P>_r4``.
``MUSE_VR_MOE_LDW_MAX_M`` (default 512; 1024 allowed) raises MoE-LD-W's row ceiling: at 1024 every MoE-LD-W build is the wide build
(``ops.moe_ldw.WIDE``) with scratch for 1024 rows (about twice the memory), so calls of up to 1024 rows engage (the same rows,
bitwise); RING48 then also runs up to that ceiling unless ``MUSE_VR_MOE_LDW_RING48_MAX_M`` lowers it.
``MUSE_VR_MOE_LD_PLACE_AT_CREATE=1`` (default off) creates the expert weights (gate_up_proj, down_proj) directly in the placement
layout (``ld_layout.alloc_rows_split``: split_rows_copy's chunks and domain order) inside the stock create_weights. The checkpoint
loader and the stock process_weights_after_loading write them in place, so the placed bytes are the ones the load-time copy would
place, and the load-time step finds them placed (``ld_layout.is_placed``) and skips its copy, swap and empty_cache. With expandable
segments, the copy's freed regions leave partly-live allocator pages per layer that empty_cache cannot unmap; this avoids them
(tests/test_moe_ld_place_create.py). ``MUSE_VR_MOE_LD_WFP=1`` (default off; diagnostics) folds every prepared layer's expert
weights and block scales into a per-rank digest (engagement record ``moe_ld_wfp``) for whole-tensor compares between runs.
``MUSE_VR_MOE_LD_MAX_M`` (default 16) bounds the rows MoE-LD takes; 0 leaves calls of up to 16 rows on FlashInfer, so only
MoE-LD-W engages.

Early-trigger quant for MoE-LD-W (env ``MUSE_VR_MOE_LDW_EARLY=1`` in addition to ``MUSE_VR_MOE_LDW=1``, default off): the
experts' input quant of a call MoE-LD-W will take (``MuseSpark1xFp4Experts.prepare_input``, AROUND) runs a copy of SGLang's
fused norm + NVFP4 quant whose programmatic trigger is issued at kernel entry (``ops.norm_fp4_quant_early``: the same
arithmetic and launch shape), so MoE-LD-W launches while the quant runs; MoE-LD-W reads every quant output only after its
dependency wait (every W build). Same outputs, bitwise. A call fed by the early quant that MoE-LD-W then does not take
raises (fail-stop): another consumer could read the outputs before they are complete. Tally ``early_quant:m<M>``
(component ``moe_ldw``). Pinned to SGLang's quant sources (loud record ``moe_ldw_early``).

KM2 consumer (env ``MUSE_VR_KM2=1`` in addition, default off): when a board producer component has published this layer
call's routed ids (and optionally its token operand) on the layer's call-site boards (``csrc/lb_ct1.cuh``) and set the
board context for the call (``set_board_context``), the engaged call runs ``ops.lb_km2.km2`` instead of ``moe_ld``: the
same rows, bitwise, with the ids / tokens read from the boards as soon as they are published. Pairing: the producer asks
``board_consumer_ready(m)`` before it publishes; whenever a context was set and the call steps aside after all, the
layer's consumer site still records this forward's sequence value (so a later missed per-forward advance can never make
that publication look current). ``preload()`` runs at load time for every KM2 build.
"""

from __future__ import annotations

import logging
import threading

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/models/muse_spark.py": ("get_forward",),
    "srt/layers/muse_spark/muse_spark_fp4_moe.py": ("MuseSpark1xFp4Experts",),
    "srt/layers/muse_spark/muse_spark_nvfp4_moe.py": ("MuseSparkNvFp4FusedMoEMethod",),
    "srt/layers/moe/utils.py": ("is_sbo_enabled", "is_tbo_enabled"),
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
}


logger = logging.getLogger("muse_vr_plugin")

NAME = "moe_ld"
PINS = {
    "sglang/srt/layers/muse_spark_fp4_moe.py": (
        "1d377424e4c1f486f72a0e8ea67ea41dca4bd3f6929aa3729f5f7480861ed352",
        "1d7676d6d086e6aa45139671e345ef32d28a81b7de1b50810d0e512ee95cc738",
    ),
    "sglang/srt/layers/quantization/muse_spark_nvfp4_moe.py": (
        "c09e0b4e32b7d5ac7af30caf74121dc0897e14e44f0e4503ea33a7acaf8f35a0",
        "6a1769ef903f1771040fc97e78c3fa46662ab0311cb45a5455984562da2c8a5d",
    ),
}
_NONE_ARGS = (
    "routing_bias",
    "gemm1_bias",
    "gemm1_alpha",
    "gemm1_beta",
    "gemm1_clamp_limit",
    "gemm2_bias",
)

_tls = threading.local()
_state: dict = {
    "engaged": None,
    "orig_moe": None,
    "sites": {},
    "prepared": 0,
    "skipped": "",
    "off": "",
    "km2": False,
    "km2_sites": {},
    "ldw": False,
    "ldw_scratch": {},
    "ldw_min": 64,
    "ldw_head_max_m": 64,
    "ld_max_m": 16,
    "ldw_ring48": False,
    "ldw_ring48_min_m": 96,
    "ldw_ring48_max_m": 512,
    "ldw_scratch48": {},
    "ldw_ring4": False,
    "ldw_ring4_min_m": 65,
    "ldw_ring4_max_m": 95,
    "ldw_scratch4": {},
    "ldw_max_m": 512,
    "ldw_early": False,
    "early_armed": None,
}

EARLY_NAME = "moe_ldw_early"  # env MUSE_VR_MOE_LDW_EARLY (with moe_ldw)
EARLY_PINS = {  # the quant the copy in csrc/norm_fp4_quant_early.cuh was taken from + its dispatcher
    "sglang/kernels/jit/csrc/muse_spark_v12/norm_fp4_quant.cuh": "7bfbd40d531ab9809da3febcba719e04c178ea66e7fee665dffb81c699bd068b",
    # the dispatcher module at both validated releases (the second: the same public function and JIT kernel, relocated)
    "sglang/kernels/ops/muse_spark_v12/norm_fp4_quant.py": (
        "17b954aef697cfb7d357b30a25a19810af0bbb85ba55e0496368b262fa872742",
        "f1c48a59a758f58751739859283a93b23f23540c065bf25bb6833f06113e301a",
    ),
}
T_PREPARE = "sglang.srt.layers.muse_spark_fp4_moe.MuseSpark1xFp4Experts.prepare_input"


class BoardContext:
    """One layer call's board state, set by the producer component around the experts call it published for."""

    def __init__(
        self,
        seq,
        ids_board,
        prod_err,
        tok_hdr=None,
        tok_writers: int = 0,
        release_form: bool = False,
    ):
        self.seq, self.ids_board, self.prod_err = seq, ids_board, prod_err
        self.tok_hdr, self.tok_writers, self.release_form = (
            tok_hdr,
            int(tok_writers),
            bool(release_form),
        )


def set_board_context(ctx: BoardContext | None) -> None:
    """Producer side: the boards this layer call's producer published on (None clears it)."""
    _tls.board = ctx


def board_consumer_ready(m: int) -> bool:
    """Producer side, before publishing: would the current layer's experts call engage the KM2 consumer at m rows?"""
    from .moe_ld_ops import MAX_TOKENS

    layer = getattr(_tls, "layer", None)
    return bool(
        _state["km2"]
        and not _state["off"]
        and 1 <= m <= MAX_TOKENS
        and layer is not None
        and _state["km2_sites"].get(layer.gate_up_proj.data_ptr()) is not None
        and not _tuning()
        and not _sp_active()
        and not _overlap_enabled()
    )


def _tuning() -> bool:
    try:
        from flashinfer.autotuner import AutoTuner

        return bool(AutoTuner.get().is_tuning_mode)
    except Exception:  # noqa: BLE001
        return False


def _overlap_enabled() -> bool:
    try:
        from sglang.srt.layers.moe.utils import is_sbo_enabled, is_tbo_enabled

        return bool(is_tbo_enabled() or is_sbo_enabled())
    except Exception:  # noqa: BLE001 - unknown configuration: step aside
        return True


def _sp_active() -> bool:
    try:
        import sglang.srt.models.muse_spark as ms

        return bool(getattr(ms.get_forward(), "sp_active", False))
    except Exception:  # noqa: BLE001
        return True


def _layer_reason(layer, quant_config=None) -> str:
    """Empty when this expert layer can use the kernel, else why not (quant_config: the one to judge by when the layer has
    none attached yet, i.e. at create_weights time)."""
    from .moe_ld_ops import HIDDEN, INTER, MAX_POOL

    if getattr(layer, "use_marlin", False) or getattr(layer, "use_megamoe", False):
        return "marlin / megamoe expert path"
    w13, w2 = getattr(layer, "gate_up_proj", None), getattr(layer, "down_proj", None)
    if w13 is None or w2 is None:
        return "no expert weights"
    if (
        w13.dim() != 3
        or tuple(w13.shape[1:]) != (2 * INTER, HIDDEN // 2)
        or not w13.is_contiguous()
    ):
        return f"gate_up_proj shape {tuple(w13.shape)}"
    if (
        w2.dim() != 3
        or tuple(w2.shape[1:]) != (HIDDEN, INTER // 2)
        or not w2.is_contiguous()
    ):
        return f"down_proj shape {tuple(w2.shape)}"
    if w13.shape[0] > MAX_POOL or w2.shape[0] != w13.shape[0]:
        return f"{w13.shape[0]} expert slots"
    qc = getattr(layer, "quant_config", None)
    if qc is None:
        qc = quant_config
    if not bool(getattr(qc, "use_per_token_activation", False)):
        return "per-token activation scaling is off"
    return ""


_PC: dict = {
    "on": None,
    "tensors": 0,
    "skipped": None,
}  # MUSE_VR_MOE_LD_PLACE_AT_CREATE
_WFP: dict = {"on": None, "tensors": 0, "h": None}  # MUSE_VR_MOE_LD_WFP (diagnostics)


def _pc_on() -> bool:
    """MUSE_VR_MOE_LD_PLACE_AT_CREATE=1 (read once)."""
    if _PC["on"] is None:
        _PC["on"] = bed.env_int("MUSE_VR_MOE_LD_PLACE_AT_CREATE", 0) == 1
    return _PC["on"]


def _place_at_create(method, layer) -> None:
    """Back gate_up_proj / down_proj by the placement layout from the start (MUSE_VR_MOE_LD_PLACE_AT_CREATE). The checkpoint
    loader and the stock process_weights_after_loading write these parameters in place (copy_ into the same storage), so
    the placed storage ends up holding exactly the bytes split_rows_copy would have copied there; _prepare_layer then finds
    them placed and skips its copy, swap and empty_cache. The PyTorch storage created by the stock create_weights is never
    written and goes back to the cache at once. Any failure leaves the stock storage (the load-time copy then places it)."""
    import torch

    from . import ld_layout

    why = _layer_reason(layer, getattr(method, "quant_config", None))
    if why:
        _PC["skipped"] = why
        return
    dev = layer.gate_up_proj.device
    if dev.type != "cuda" or ld_layout.domain_count(dev) != 2:
        _PC["skipped"] = f"device {dev}: not two locality domains"
        return
    with torch.no_grad():
        for name in ("gate_up_proj", "down_proj"):
            p = getattr(layer, name)
            if ld_layout.is_placed(p.data):
                continue
            p.data = ld_layout.alloc_rows_split(p.device, p.shape, p.dtype, 2)
            _PC["tensors"] += 1
    bed.report(
        "moe_ld_place_at_create",
        True,
        tensors=_PC["tensors"],
        skipped=_PC["skipped"] or "-",
    )


def _wfp_on() -> bool:
    """MUSE_VR_MOE_LD_WFP=1 (read once)."""
    if _WFP["on"] is None:
        _WFP["on"] = bed.env_int("MUSE_VR_MOE_LD_WFP", 0) == 1
    return _WFP["on"]


def _wfp_tensor(t, slice_bytes: int = 32 * 2**20) -> list:
    """Order-sensitive fingerprint of a contiguous tensor's bytes, on its device, in slices (small temporaries): per slice the
    sums of the low / high 32-bit words of every 8-byte word and two position-weighted 16-bit sums (no int64 overflow), then the
    tail bytes; a whole-tensor compare between two runs at the cost of one read."""
    import torch

    b = t.detach().contiguous().view(torch.uint8).view(-1)
    n8 = b.numel() // 8 * 8
    out: list = [int(b.numel())]
    w = None
    for o in range(0, n8, slice_bytes):
        x = b[o : min(o + slice_bytes, n8)].view(torch.int64)
        if w is None or w.numel() < x.numel():
            w = torch.arange(x.numel(), device=x.device, dtype=torch.int64) % 65521 + 1
        ww = w[: x.numel()]
        out.append(
            torch.stack(
                (
                    (x & 0xFFFFFFFF).sum(),
                    ((x >> 32) & 0xFFFFFFFF).sum(),
                    ((x & 0xFFFF) * ww).sum(),
                    (((x >> 16) & 0xFFFF) * ww).sum(),
                )
            )
        )
    vals = (
        [out[0]] + [v for r in torch.stack(out[1:]).tolist() for v in r]
        if len(out) > 1
        else [out[0]]
    )
    vals += b[n8:].tolist()
    return vals


def _wfp_add(layer, dev) -> None:
    """MUSE_VR_MOE_LD_WFP: fold the layer's expert weights + their block scales into the rank's running digest (load order)."""
    import hashlib

    if _WFP["h"] is None:
        _WFP["h"] = hashlib.sha256()
    for name in (
        "gate_up_proj",
        "down_proj",
        "gate_up_proj_weight_scale",
        "down_proj_weight_scale",
    ):
        t = getattr(layer, name, None)
        if t is None:
            continue
        _WFP["h"].update(f"{name}:{tuple(t.shape)}:{t.dtype}:".encode())
        _WFP["h"].update(repr(_wfp_tensor(t)).encode())
        _WFP["tensors"] += 1
    bed.report(
        "moe_ld_wfp",
        True,
        device=str(dev),
        tensors=_WFP["tensors"],
        digest=_WFP["h"].hexdigest(),
    )


def _prepare_layer(layer) -> None:
    """Placement of the shuffled expert weights + the layer's call-site workspace (load time)."""
    import torch

    from . import ld_layout
    from .moe_ld_ops import Site

    why = _layer_reason(layer)
    if why:
        _state["skipped"] = why
        return
    dev = layer.gate_up_proj.device
    two = False
    try:
        two = ld_layout.domain_count(dev) == 2
    except Exception:  # noqa: BLE001 - placement only changes speed
        two = False
    if two:
        with torch.no_grad():
            for name in ("gate_up_proj", "down_proj"):
                p = getattr(layer, name)
                if ld_layout.is_placed(
                    p.data
                ):  # MUSE_VR_MOE_LD_PLACE_AT_CREATE: already in the placement layout
                    continue
                placed = ld_layout.split_rows_copy(p.data, 2)
                p.data = placed
                del placed
                torch.cuda.empty_cache()  # the replaced storage goes back before the next layer's placement
    if _wfp_on():
        with torch.no_grad():
            _wfp_add(layer, dev)
    if not _state["prepared"]:
        # once per process, at load time (outside any graph capture): the SM -> locality-domain table (probe kernels
        # and a synchronization) and the kernel builds for both row configurations
        from . import moe_ld_ops

        moe_ld_ops.domain_slots(
            dev.index if dev.index is not None else torch.cuda.current_device()
        )
        for n in sorted(moe_ld_ops.CONFIGS):
            moe_ld_ops._module(*moe_ld_ops.config_for(n))
    _state["sites"][layer.gate_up_proj.data_ptr()] = Site(dev)
    from . import hb0_rr

    hb0_rr.note_layer(
        layer.gate_up_proj
    )  # the layer's call slot for the rows-ready finalize (MUSE_VR_HB0_RR; inert when off)
    dix = dev.index if dev.index is not None else torch.cuda.current_device()
    if _state["ldw"] and dix not in _state["ldw_scratch"]:
        from .ops import moe_ldw

        wide = _ldw_wide()
        if _state["ldw_ring48"]:
            moe_ldw.preload(moe_ldw.RING48 + wide)
            _state["ldw_scratch48"][dix] = moe_ldw.Scratch(
                dev, moe_ldw.RING48_PASS_ROWS, _state["ldw_max_m"]
            )
        if _state["ldw_ring4"]:
            moe_ldw.preload(moe_ldw.RING4 + wide)
            _state["ldw_scratch4"][dix] = moe_ldw.Scratch(
                dev, moe_ldw.RING4_PASS_ROWS, _state["ldw_max_m"]
            )
        moe_ldw.preload(wide)
        if _state["ldw_head_max_m"] < _state["ldw_max_m"]:
            moe_ldw.preload(_LDW_NO_HEAD + wide)
        _state["ldw_scratch"][dix] = moe_ldw.Scratch(
            dev, moe_ldw.PASS_ROWS, _state["ldw_max_m"]
        )
        if _state["ldw_early"]:
            from .ops import norm_fp4_quant_early

            norm_fp4_quant_early.preload(moe_ldw.HIDDEN)
    if _state["km2"]:
        from .ops import lb_km2

        if not _state["km2_sites"]:
            lb_km2.preload()  # board consumers are loaded before any forward can run them next to a producer
        _state["km2_sites"][layer.gate_up_proj.data_ptr()] = lb_km2.Site(dev)
    _state["prepared"] += 1


def _call_site(kw, lo: int = 1, hi: int = 0):
    """The layer's workspace when this FlashInfer call can run on the kernel (lo..hi rows; hi 0 = MoE-LD's maximum), else
    None."""
    import torch

    from .moe_ld_ops import HIDDEN, INTER, MAX_TOKENS, TOP_K

    hi = hi or MAX_TOKENS

    if (
        kw.get("do_finalize", True) is not False
        or kw.get("activation_type") != 3
        or kw.get("routing_method_type") != 1
    ):
        return None
    if (
        kw.get("top_k") != TOP_K
        or kw.get("intermediate_size") != INTER
        or kw.get("n_group")
        or kw.get("topk_group")
    ):
        return None
    rsf = kw.get("routed_scaling_factor")
    if rsf is not None and float(rsf) != 1.0:
        return None
    if any(kw.get(k) is not None for k in _NONE_ARGS):
        return None
    packed, hs, pts = (
        kw.get("topk_ids"),
        kw.get("hidden_states"),
        kw.get("per_token_scale"),
    )
    if (
        not isinstance(packed, torch.Tensor)
        or packed.dtype != torch.int32
        or packed.dim() != 2
    ):
        return None
    m = packed.shape[0]
    if not (lo <= m <= hi) or packed.shape[1] != TOP_K or not packed.is_contiguous():
        return None
    if (
        pts is None
        or hs is None
        or hs.shape[0] != m
        or hs.shape[-1] * hs.element_size() != HIDDEN // 2
    ):
        return None
    w13 = kw.get("gemm1_weights")
    p = int(kw.get("local_num_experts") or 0)
    if w13 is None or not (1 <= p <= w13.shape[0]):
        return None
    site = _state["sites"].get(w13.data_ptr())
    if site is None or _tuning() or _sp_active() or _overlap_enabled():
        return None
    return site


def _rows_equal(a, b) -> bool:
    """Every routed (token, k) row and the top-k weights of two (gemm2_out, expert_weights, map) triples agree bitwise."""
    import torch

    g_a, w_a, m_a = a
    g_b, w_b, m_b = b
    ia, ib = m_a.reshape(-1).long(), m_b.reshape(-1).long()
    if not torch.equal(ia >= 0, ib >= 0):
        return False
    va = g_a.index_select(0, ia.clamp(min=0)).view(torch.int16)
    vb = g_b.index_select(0, ib.clamp(min=0)).view(torch.int16)
    keep = (ia >= 0)[:, None]
    return bool(
        torch.equal(torch.where(keep, va, 0), torch.where(keep, vb, 0))
        and torch.equal(w_a.view(torch.int16), w_b.view(torch.int16))
    )


def _km2_call(kw, ctx):
    """The KM2 consumer for an engaged call with a board context, or None."""
    from .ops import lb_km2

    w13 = kw["gemm1_weights"]
    ksite = _state["km2_sites"].get(w13.data_ptr())
    if ksite is None:
        return None
    p = int(kw["local_num_experts"])
    flags = lb_km2.DEFAULT_FLAGS | lb_km2.FLAG_IDS_BOARD
    if ctx.release_form:
        flags |= lb_km2.FLAG_IDS_RELEASE
    if ctx.tok_hdr is not None and ctx.tok_writers > 0:
        flags |= lb_km2.FLAG_TOK_BOARD
    out = lb_km2.km2(
        kw["topk_ids"],
        kw["hidden_states"],
        kw["hidden_states_scale"],
        kw["per_token_scale"],
        w13[:p],
        kw["gemm1_weights_scale"][:p],
        kw["gemm2_weights"][:p],
        kw["gemm2_weights_scale"][:p],
        kw["output1_scale_gate_scalar"][:p],
        kw["output1_scale_scalar"][:p],
        kw["output2_scale_scalar"][:p],
        int(kw["local_expert_offset"]),
        ksite,
        seq=ctx.seq,
        ids_board=ctx.ids_board,
        tok_hdr=ctx.tok_hdr if flags & lb_km2.FLAG_TOK_BOARD else None,
        tok_writers=ctx.tok_writers if flags & lb_km2.FLAG_TOK_BOARD else 0,
        prod_err=ctx.prod_err,
        flags=flags,
        pdl=True,
    )
    bed.count("km2", f"m{kw['topk_ids'].shape[0]}_p{p}")
    return out


def _km2_mark_seen(kw, ctx) -> None:
    """A published call that stepped aside: the consumer site still records this forward's sequence value."""
    from .ops import lb_km2

    w13 = kw.get("gemm1_weights")
    ksite = _state["km2_sites"].get(w13.data_ptr()) if w13 is not None else None
    if ksite is not None:
        ksite.block[lb_km2.SITE_LAST_SEQ : lb_km2.SITE_LAST_SEQ + 1].copy_(
            ctx.seq.view(-1)[:1]
        )
        bed.count("km2", "stepped_aside_marked")


def _routed_moe(*args, **kw):
    fed = getattr(_tls, "early_fed", None)
    if fed is not None:
        _tls.early_fed = None
        out = (
            _ldw_call(kw)
            if (
                not args
                and not _state["off"]
                and _state["ldw"]
                and getattr(_tls, "layer", None) is fed
            )
            else None
        )
        if out is None:
            raise RuntimeError(
                "muse_vr_plugin: moe_ldw_early: an early-trigger quant fed a routed-MoE call that MoE-LD-W "
                "did not take (fail-stop: another consumer could read its outputs before they complete)"
            )
        return out
    layer = getattr(_tls, "layer", None)
    ctx = getattr(_tls, "board", None) if _state["km2"] else None
    if layer is not None and not args and not _state["off"]:
        # MoE-LD for 1..MUSE_VR_MOE_LD_MAX_M rows (default 16; 0 = MoE-LD-W only)
        site = _call_site(kw, 1, _state["ld_max_m"]) if _state["ld_max_m"] > 0 else None
        if site is not None and ctx is not None:
            out = _km2_call(kw, ctx)
            if out is not None:
                return out
        if site is not None:
            import torch

            from .moe_ld_ops import moe_ld

            if ctx is not None:
                _km2_mark_seen(kw, ctx)
            p = int(kw["local_num_experts"])
            out = moe_ld(
                kw["topk_ids"],
                kw["hidden_states"],
                kw["hidden_states_scale"],
                kw["per_token_scale"],
                kw["gemm1_weights"][:p],
                kw["gemm1_weights_scale"][:p],
                kw["gemm2_weights"][:p],
                kw["gemm2_weights_scale"][:p],
                kw["output1_scale_gate_scalar"][:p],
                kw["output1_scale_scalar"][:p],
                kw["output2_scale_scalar"][:p],
                int(kw["local_expert_offset"]),
                site,
                pdl=True,
            )
            bed.count(NAME, f"m{kw['topk_ids'].shape[0]}_p{p}")
            if (
                bed.flag("moe_ld_shadow")
                and not torch.cuda.is_current_stream_capturing()
            ):
                ref = _state["orig_moe"](**kw)[:3]
                ok = _rows_equal(out, ref)
                bed.shadow_result(NAME, ok)
                if not ok:
                    _state["off"] = "shadow mismatch"
                    logger.error(
                        "muse_vr_plugin: moe_ld shadow mismatch; the component is off for this process"
                    )
                    return ref
            return out
    if ctx is not None:
        _km2_mark_seen(kw, ctx)
    if layer is not None and not args and not _state["off"] and _state["ldw"]:
        out = _ldw_call(kw)
        if out is not None:
            return out
    return _state["orig_moe"](*args, **kw)


_LDW_NO_HEAD = (
    "-DMUSE_VR_LDW_GATHER=0",
)  # MoE-LD-W without the first-item head copies (larger calls)


def _ldw_wide() -> tuple:
    """The wide build's flag when MUSE_VR_MOE_LDW_MAX_M is 1024 (every MoE-LD-W build and scratch then takes 1024 rows)."""
    from .ops import moe_ldw

    return moe_ldw.WIDE if _state["ldw_max_m"] > moe_ldw.MAX_TOKENS else ()


def _ldw_form(m: int) -> str:
    """The MoE-LD-W build of an m-row call: "r48" (RING48 range), "r4" (RING4 range) or "base" (head / no-head build)."""
    if (
        _state["ldw_ring48"]
        and _state["ldw_ring48_min_m"] <= m <= _state["ldw_ring48_max_m"]
    ):
        return "r48"
    if (
        _state["ldw_ring4"]
        and _state["ldw_ring4_min_m"] <= m <= _state["ldw_ring4_max_m"]
    ):
        return "r4"
    return "base"


def _ldw_scratch(form: str, dix: int):
    """The per-device scratch of that build, or None."""
    return {"r48": _state["ldw_scratch48"], "r4": _state["ldw_scratch4"]}.get(
        form, _state["ldw_scratch"]
    ).get(dix)


def _ldw_call(kw):
    """MoE-LD-W for an engaged-layer call of ``ldw_min``..``ldw_max_m`` rows (the same argument set as MoE-LD's), else None."""
    import torch

    from .ops import moe_ldw

    if _call_site(kw, _state["ldw_min"], _state["ldw_max_m"]) is None:
        return None
    w13 = kw["gemm1_weights"]
    dix = (
        w13.device.index
        if w13.device.index is not None
        else torch.cuda.current_device()
    )
    m = int(kw["topk_ids"].shape[0])
    form = _ldw_form(m)
    scratch = _ldw_scratch(form, dix)
    if scratch is None:
        return None
    p = int(kw["local_num_experts"])
    if form == "r48":
        extra = moe_ldw.RING48
    elif form == "r4":
        extra = moe_ldw.RING4
    else:
        extra = () if m <= _state["ldw_head_max_m"] else _LDW_NO_HEAD
    extra = extra + _ldw_wide()
    # MUSE_VR_HB0_RR: the rows-ready finalize consumes this call (its build defines; the sequence / ring advance of the
    # first layer runs here, before the expert kernel, so the finalize is the expert kernel's immediate dependent)
    from . import hb0_rr

    rr = () if bed.flag("moe_ld_shadow") else hb0_rr.before_expert(kw, m, form)
    extra = extra + rr
    out = moe_ldw.moe_ldw(
        kw["topk_ids"],
        kw["hidden_states"],
        kw["hidden_states_scale"],
        kw["per_token_scale"],
        w13[:p],
        kw["gemm1_weights_scale"][:p],
        kw["gemm2_weights"][:p],
        kw["gemm2_weights_scale"][:p],
        kw["output1_scale_gate_scalar"][:p],
        kw["output1_scale_scalar"][:p],
        kw["output2_scale_scalar"][:p],
        int(kw["local_expert_offset"]),
        scratch,
        pdl=True,
        extra=extra,
    )
    if rr:
        hb0_rr.after_expert(kw, out, scratch)
    bed.count("moe_ldw", f"m{m}_p{p}_{form}" if form != "base" else f"m{m}_p{p}")
    if bed.flag("moe_ld_shadow") and not torch.cuda.is_current_stream_capturing():
        ref = _state["orig_moe"](**kw)[:3]
        ok = _rows_equal(out, ref)
        bed.shadow_result("moe_ldw", ok)
        if not ok:
            _state["off"] = "shadow mismatch (moe_ldw)"
            logger.error(
                "muse_vr_plugin: moe_ldw shadow mismatch; the component is off for this process"
            )
            return ref
    return out


def _early_eligible(experts, x) -> bool:
    """MUSE_VR_MOE_LDW_EARLY: this prepare_input call feeds a call MoE-LD-W will take (rank-uniform: the layer's prepared
    site, the row count, the input form, the configuration)."""
    import torch

    if (
        _state["off"]
        or not _state["ldw"]
        or _tuning()
        or _sp_active()
        or _overlap_enabled()
    ):
        return False
    w13 = getattr(experts, "gate_up_proj", None)
    if w13 is None or _state["sites"].get(w13.data_ptr()) is None:
        return False
    m = int(x.shape[0]) if x.dim() == 2 else 0
    lo = max(_state["ldw_min"], _state["ld_max_m"] + 1)
    if not (
        lo <= m <= _state["ldw_max_m"]
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
    ):
        return False
    # the scratch of the build MoE-LD-W picks for this row count (as _ldw_call) must exist on this device
    dix = (
        w13.device.index
        if w13.device.index is not None
        else torch.cuda.current_device()
    )
    return _ldw_scratch(_ldw_form(m), dix) is not None


def _early_around(original, experts, hidden_states, *a, **kw):
    """MUSE_VR_MOE_LDW_EARLY's AROUND hook on prepare_input: arms the quant dispatcher for an eligible call only."""
    _state["early_armed"] = experts if _early_eligible(experts, hidden_states) else None
    try:
        return original(experts, hidden_states, *a, **kw)
    finally:
        _state["early_armed"] = None


def _early_register(HookRegistry=None, HookType=None) -> bool:
    """MUSE_VR_MOE_LDW_EARLY: the quant dispatcher of SGLang's FP4 experts module defers to the early-trigger copy while
    an eligible prepare_input call runs (AROUND hook arms it); every other call keeps SGLang's dispatcher. (The registry
    arguments are for tests; SGLang's by default.)"""
    why = next(
        (
            f"pin mismatch: {rel}"
            for rel, sha in EARLY_PINS.items()
            if not bed.pinned(rel, sha)
        ),
        "",
    )
    if why:
        bed.report(EARLY_NAME, False, reason=why)
        return False
    import sglang.srt.layers.muse_spark_fp4_moe as fm

    from .ops import norm_fp4_quant_early as nfq

    if HookRegistry is None:
        from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    orig_quant = fm.fused_norm_fp4_quant

    def quant(x, weight, eps, global_scale_inv, *a, **kw):
        experts = _state["early_armed"]
        if experts is not None and not a and not kw:
            _state["early_armed"] = None  # one quant per prepare_input call
            _tls.early_fed = experts  # its routed-MoE call must be MoE-LD-W's
            bed.count("moe_ldw", f"early_quant:m{int(x.shape[0])}")
            return nfq.fused_norm_fp4_quant_early(x, weight, eps, global_scale_inv)
        return orig_quant(x, weight, eps, global_scale_inv, *a, **kw)

    fm.fused_norm_fp4_quant = quant
    HookRegistry.register(T_PREPARE, _early_around, HookType.AROUND)
    bed.report(EARLY_NAME, True, ldw_min=_state["ldw_min"], ld_max_m=_state["ld_max_m"])
    return True


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

    fi_guard.check(
        NAME
    )  # FlashInfer version / batched-GEMM artifact class; raises under MUSE_VR_REQUIRE_ENGAGED=1
    fi_guard.check("moe_ld_ops")
    import flashinfer.fused_moe as ffm
    import sglang.srt.layers.muse_spark_fp4_moe as fp4
    import sglang.srt.layers.quantization.muse_spark_nvfp4_moe as q

    orig_pwal = q.MuseSparkNvFp4FusedMoEMethod.process_weights_after_loading
    orig_forward = fp4.MuseSpark1xFp4Experts.forward

    def process_weights_after_loading(self, layer):
        out = orig_pwal(self, layer)
        try:
            _prepare_layer(layer)
        except Exception as e:  # noqa: BLE001 - an unprepared layer simply keeps FlashInfer
            _state["skipped"] = f"prepare failed: {type(e).__name__}: {e}"
        bed.report(
            NAME,
            _state["prepared"] > 0,
            layers=_state["prepared"],
            shadow=bed.flag("moe_ld_shadow"),
            skipped=_state["skipped"] or "-",
        )
        return out

    def experts_forward(self, *a, **kw):
        prev = getattr(_tls, "layer", None)
        _tls.layer = self
        try:
            return orig_forward(self, *a, **kw)
        finally:
            _tls.layer = prev

    _state["km2"] = bed.enabled("km2")
    from .moe_ld_ops import MAX_TOKENS as _LD_MAX

    _state["ld_max_m"] = max(
        0, min(_LD_MAX, bed.env_int("MUSE_VR_MOE_LD_MAX_M", _LD_MAX))
    )
    _state["ldw"] = bed.enabled("moe_ldw")
    _state["ldw_min"] = max(17, bed.env_int("MUSE_VR_MOE_LDW_MIN", 64))
    _state["ldw_head_max_m"] = bed.env_int("MUSE_VR_MOE_LDW_HEAD_MAX_M", 64)
    _state["ldw_ring48"] = bed.env_int("MUSE_VR_MOE_LDW_RING48", 0) == 1
    _state["ldw_ring48_min_m"] = bed.env_int("MUSE_VR_MOE_LDW_RING48_MIN_M", 96)
    max_m = bed.env_int("MUSE_VR_MOE_LDW_MAX_M", 512)
    if max_m not in (512, 1024):
        logger.error(
            "muse_vr_plugin: MUSE_VR_MOE_LDW_MAX_M=%s is not 512 or 1024; using 512",
            max_m,
        )
        max_m = 512
    _state["ldw_max_m"] = max_m
    if max_m > 512:
        bed.report("moe_ldw_wide", _state["ldw"], max_m=max_m)
    _state["ldw_ring48_max_m"] = min(
        max_m, bed.env_int("MUSE_VR_MOE_LDW_RING48_MAX_M", max_m)
    )
    _state["ldw_ring4"] = bed.env_int("MUSE_VR_MOE_LDW_RING4", 0) == 1
    _state["ldw_ring4_min_m"] = bed.env_int("MUSE_VR_MOE_LDW_RING4_MIN_M", 65)
    _state["ldw_ring4_max_m"] = min(
        max_m, bed.env_int("MUSE_VR_MOE_LDW_RING4_MAX_M", 95)
    )
    if _state["ldw"] and bed.env_int("MUSE_VR_MOE_LDW_EARLY", 0) == 1:
        _state["ldw_early"] = _early_register()
    _state["orig_moe"] = ffm.trtllm_fp4_block_scale_routed_moe
    ffm.trtllm_fp4_block_scale_routed_moe = _routed_moe
    q.MuseSparkNvFp4FusedMoEMethod.process_weights_after_loading = (
        process_weights_after_loading
    )
    if _pc_on():  # MUSE_VR_MOE_LD_PLACE_AT_CREATE: the expert weights are created in the placement layout
        orig_create = q.MuseSparkNvFp4FusedMoEMethod.create_weights

        def create_weights(self, layer, *a, **kw):
            out = orig_create(self, layer, *a, **kw)
            try:
                _place_at_create(self, layer)
            except Exception as e:  # noqa: BLE001 - the stock storage stays; the load-time copy places it
                _PC["skipped"] = f"place at create failed: {type(e).__name__}: {e}"
                bed.report(
                    "moe_ld_place_at_create",
                    _PC["tensors"] > 0,
                    tensors=_PC["tensors"],
                    skipped=_PC["skipped"],
                )
            return out

        q.MuseSparkNvFp4FusedMoEMethod.create_weights = create_weights
        logger.warning(
            "muse_vr_plugin: moe_ld expert weights are created in the placement layout (MUSE_VR_MOE_LD_PLACE_AT_CREATE=1)"
        )
    fp4.MuseSpark1xFp4Experts.forward = experts_forward
    _state["engaged"] = True
    logger.warning("muse_vr_plugin: moe_ld hooks installed")
    return True

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``prefill_moe_x``: prefill-sized routed-expert calls on CuTe-DSL grouped GEMM kernels.

SGLang's Muse Spark NVFP4 expert layers call ``flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe`` with per-token
NVFP4 activations and ``do_finalize=False`` (the caller applies the post-expert norm and the weighted sum). For calls
with more than ``MUSE_VR_PREFILL_MOE_X_MIN_TOKENS`` tokens (default 2048) this component computes the same deferred
outputs with the ``moe_x_kernels`` pipeline: FlashInfer's ``moe_sort``, a sorted-row copy of the NVFP4 activations, a
grouped GEMM with the SwiGLU fused in its epilogue (which also reduces each row's maximum), a per-token NVFP4 quantizer
with the arithmetic of the one the fused-MoE runner uses between its GEMMs, and a grouped GEMM that writes one unweighted
row per (token, slot). Both GEMMs read the expert weights and block scales the server already
holds (the fused-MoE preparation's row order and scale layout) in place; no weight copy is made.

The returned triple has the stock ABI (``gemm2_out`` rows, ``expert_weights``, ``expanded_idx_to_permuted_idx`` with
``-1`` for slots of other ranks); the row order inside ``gemm2_out`` is this pipeline's own and is consistent with the
returned index map. Numerics: same inputs, scales and quantizer recipe as the stock path; the GEMM accumulation order
and the SwiGLU evaluation differ (reduction-order class).

Every call that does not match (other shapes or dtypes, finalize requested, no per-token scales, a non-SwiGLU
activation, gate parameters, bias, routing scale or bias, too few tokens, kernels not yet built while a CUDA graph is
being captured) goes to the stock op; the reason is counted once per kind. A failure inside the pipeline turns the
component off for the rest of the process (logged) and that call goes to the stock op.

Calls made outside the prefill CUDA-graph runner's capture session (no stream capturing and the runner's
``capturing_prefill_graph`` flag clear: real eager prefill passes with prefill CUDA graphs off or outside the captured range,
and pre-capture autotune warm-ups) also go to the stock op, decided before any launch and logged once per process. The
pipeline raised a fatal, sticky "misaligned address" fault in real eager prefill passes, which no fallback after the launch
can survive. The runner sets the flag around every per-shape warm-up and capture, and a replay runs no Python, so the
captured served path is unchanged by this check. (SGLang's per-forward extend flag cannot serve here: it is written only on
the DP-attention sync path.)

A call whose NVFP4 activation rows do not start on a 32-byte boundary also goes to the stock op before any launch
("stock:hs_align"): the row-permute kernel reads every row with 32-byte vector loads, and PTX requires a load address
aligned to its access size. The all-to-all receive packs the dispatch payloads back to back (per-token scale, expert ids
and weights, then the NVFP4 rows), so the rows start 16 bytes off a 32-byte boundary when the per-rank token count is odd,
which happens in eager prefill passes (captured prefill sizes above the token floor keep every region aligned).

Opt in with ``MUSE_VR_PREFILL_MOE_X=1``. The device-specific MMA builder module is named by
``MUSE_VR_MOE_X_MMA_HELPERS`` (see ``moe_x_kernels/arch.py``); without it the component stays off. Kernel
configurations: ``MUSE_VR_PREFILL_MOE_X_TACTIC`` = ``<tile_m>,<tile_n1>,<a_path>,<tile_n2>[,2cta]`` (tunable; default
from ``moe_x_kernels.launch``). Register this component before any component that wraps the same FlashInfer entry point and
adds per-token scales, so that it receives the completed call.

Work-unit schedule of both expert GEMMs: ``MUSE_VR_MOE_X_SCHED`` = ``static`` (default) | ``dyn`` (units claimed from a
counter) | ``local`` (units claimed per locality domain, for expert weights whose row halves are placed on the two domains,
as the ``moe_ld`` load hook places them); outputs are bitwise identical for every schedule (see
``moe_x_kernels.grouped_gemm_rows.GroupedGemmRowsKernel``). Calls with a claimed schedule are counted as ``sched:<mode>``.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import logging
import os
import threading
from pathlib import Path

import torch

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/runtime_context.py": ("get_flags",),
}


logger = logging.getLogger("muse_vr_plugin")

NAME = "prefill_moe_x"

# sha256 of files relative to the directory that holds the ``sglang`` package.
SGLANG_PINS = {
    # replicated-token expert call and the deferred-finalize consumer
    "sglang/srt/layers/muse_spark_fp4_moe.py": (
        "1d377424e4c1f486f72a0e8ea67ea41dca4bd3f6929aa3729f5f7480861ed352",
        "1d7676d6d086e6aa45139671e345ef32d28a81b7de1b50810d0e512ee95cc738",
    ),
    # eager-pass guard: the prefill CUDA-graph runner's capture-session flag (set / cleared around warm-ups and captures)
    "sglang/srt/runtime_context.py": (
        "b8510f48bb90fa22474a64f6769c1e71e3fb47a95dd313cffb11df7bf414c975",
        "17c498293ca672c39fea298fa104d5b5603323581cb18d695e5d8fc4f06d1bc6",
    ),
    "sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py": (
        "72a372b5243e8d21ef943fa4f1348419c8c4413a522c0e6be662d6d5b0fb505c",
        "1ce58cd316c4e07ab4e9044f031833ef82777ea0bf1263ffd3d962e3f32eb770",
    ),
    # sequence-parallel prefill expert call
    "sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py": (
        "2f6f14ae1d95030db1ec249856b331f4e6880b0218727e8701c3944376acaaf9",
        "70db650773c0dcbeec8056d70ad6f6a0c06acd5d583e9bbbe8f4d5548cf25929",
    ),
    # expert weight preparation (gate / up order, row shuffle, block-scale interleave, global scales)
    "sglang/srt/layers/quantization/muse_spark_nvfp4_moe.py": (
        "c09e0b4e32b7d5ac7af30caf74121dc0897e14e44f0e4503ea33a7acaf8f35a0",
        "6a1769ef903f1771040fc97e78c3fa46662ab0311cb45a5455984562da2c8a5d",
    ),
    "sglang/srt/layers/quantization/utils.py": "a5c45d9e5a35b996d254e59ecefedaae04cf3fee48b85082d4f23539741f93e2",
}
# sha256 of files relative to the directory that holds the ``flashinfer`` package.
FLASHINFER_PINS = {
    # the stock routed op (arguments, deferred-finalize outputs)
    "flashinfer/fused_moe/core.py": (
        "d6a8333a2749e45e17f4ad6443c3ca84fcc8b15bca4d32adecc6cd0373528b17",  # FlashInfer 0.7.1
        "b25655f5cdc031ed003a53f92927d912154a6d4f8b37c92a52b19ae2e2da0d79",  # FlashInfer 0.7.2
    ),
    "flashinfer/data/csrc/trtllm_fused_moe_runner.cu": "f155765b0062a5d8a2395d4c654d10b2a56075abdf0aefd26a6f4274c56867c4",
    # weight row shuffle used by the preparation
    "flashinfer/utils.py": (
        "7f7da22cf3b312590ac260e39d551772d61959746f998a5650c39804c876291a",  # FlashInfer 0.7.1
        "abc97f7494cecc485395aeecd87d15e05ea518cf3d6ab1cc8c7e96fcdec3ad4d",  # FlashInfer 0.7.2
    ),
    # per-token NVFP4 quantizer (binding and kernel)
    "flashinfer/quantization/fp4_quantization.py": (
        "c3fa702c5637267d941e5c0be10c3d89272bc13e1a59a6af0f5f0e593366665a",  # FlashInfer 0.7.1
        "50221b0ebc1b76ec896c78e6fe368177a717953f5cb6d2643db108dc5fa91003",  # FlashInfer 0.7.2
    ),
    "flashinfer/data/csrc/nv_internal/tensorrt_llm/kernels/quantization.cuh": (
        "65be729913825a828fde20462cea9d687fef02537bdf07add790f9cccc297b61",  # FlashInfer 0.7.1
        "2943cb8a066ff7dd329a6f7793ac1012eebc766e7942e4d46eb7a38b5ac9c0c6",
    ),  # FlashInfer 0.7.2
    # moe_sort and the scale-layout / pointer helpers
    "flashinfer/fused_moe/cute_dsl/moe_utils.py": "d2f603e086876d7fa7256e95df3b8ba802ba174bc68efaccbd2785068815eee1",
    "flashinfer/cute_dsl/utils.py": "971faf2b712c8d973cf7b81fdb97b12d04d0fb9ff350c314981c1049b3816a25",
}

HIDDEN = 4096
INTERMEDIATE = 4096
TOP_K = 8
SWIGLU = 3
DEFAULT_MIN_TOKENS = 2048
HS_ALIGN = 32  # bytes: the row-permute kernel's per-lane vector load width (see the module docstring)

_lock = threading.Lock()
_state: dict = {
    "engaged": None,
    "orig": None,
    "broken": False,
    "tactics": None,
    "steps": set(),
    "eager_warned": False,
}


def _pkg_root(pkg: str) -> Path | None:
    spec = importlib.util.find_spec(pkg)
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(list(spec.submodule_search_locations)[0]).resolve().parent


def _pins_ok() -> tuple[bool, str]:
    for rel, sha in SGLANG_PINS.items():
        if not bed.pinned(rel, sha):
            return False, rel
    root = _pkg_root("flashinfer")
    if root is None:
        return False, "flashinfer"
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


def tactics():
    """(first GEMM, second GEMM) kernel configurations."""
    t = _state["tactics"]
    if t is None:
        from .moe_x_kernels import launch as L

        t = (L.DEFAULT_GEMM1, L.DEFAULT_GEMM2)
        spec = os.environ.get("MUSE_VR_PREFILL_MOE_X_TACTIC", "").strip()
        if spec:
            f = spec.split(",")
            two = len(f) > 4 and f[4] == "2cta"
            t = (
                L.Tactic(int(f[0]), int(f[1]), f[2], two),
                L.Tactic(int(f[0]), int(f[3]), "cpasync", two),
            )
        _state["tactics"] = t
    return t


@functools.lru_cache(maxsize=None)
def min_tokens() -> int:
    """Token-count threshold of the path (environment read once per process)."""
    try:
        return int(
            os.environ.get("MUSE_VR_PREFILL_MOE_X_MIN_TOKENS", str(DEFAULT_MIN_TOKENS))
        )
    except ValueError:
        return DEFAULT_MIN_TOKENS


def _step_aside(reason: str) -> None:
    bed.count(NAME, "stock:" + reason)


def _real_eager_extend() -> bool:
    """True for a call outside the prefill CUDA-graph runner's capture session: no stream is capturing and the runner's
    ``capturing_prefill_graph`` flag (set around its per-shape warm-ups and captures) is clear."""
    if torch.cuda.is_current_stream_capturing():
        return False
    try:
        from sglang.srt.runtime_context import get_flags

        return not bool(get_flags().dp.capturing_prefill_graph)
    except Exception:  # noqa: BLE001 - no SGLang runtime context (unit harness): not a served pass
        return False


def _hs_aligned(hs) -> bool:
    """True when the activation rows start on an HS_ALIGN-byte boundary (rows are H / 2 bytes, a multiple of it)."""
    return hs.data_ptr() % HS_ALIGN == 0


def _is_f32_vec(t, n) -> bool:
    return (
        torch.is_tensor(t)
        and t.dtype == torch.float32
        and t.dim() == 1
        and t.shape[0] == n
        and t.is_contiguous()
    )


def _eligible(kw: dict) -> str:
    """'' when the call can run on this path, else the step-aside reason."""
    if kw.get("do_finalize", True) is not False:
        return "finalize"
    if kw.get("activation_type") != SWIGLU:
        return "activation"
    for k in (
        "gemm1_alpha",
        "gemm1_beta",
        "gemm1_clamp_limit",
        "gemm1_bias",
        "gemm2_bias",
        "routing_bias",
    ):
        if kw.get(k) is not None:
            return k
    if kw.get("routed_scaling_factor") not in (None, 1.0):
        return "routed_scaling_factor"
    if kw.get("n_group") not in (None, 0) or kw.get("topk_group") not in (None, 0):
        return "group_routing"
    if "output" in kw and kw["output"] is not None:
        return "output"
    hs = kw.get("hidden_states")
    sf = kw.get("hidden_states_scale")
    pts = kw.get("per_token_scale")
    if (
        not torch.is_tensor(hs)
        or hs.dtype != torch.uint8
        or hs.dim() != 2
        or hs.shape[1] != HIDDEN // 2
    ):
        return "hidden_states"
    tokens = hs.shape[0]
    if tokens <= min_tokens():
        return "small"
    if not hs.is_cuda or not hs.is_contiguous():
        return "hidden_states"
    if not _hs_aligned(hs):
        return "hs_align"
    if (
        not torch.is_tensor(sf)
        or sf.numel() != tokens * (HIDDEN // 16)
        or sf.element_size() != 1
        or not sf.is_contiguous()
    ):
        return "hidden_states_scale"
    if pts is None:
        return "no_per_token_scale"
    if not _is_f32_vec(pts.reshape(-1) if torch.is_tensor(pts) else None, tokens):
        return "per_token_scale"
    if kw.get("top_k") != TOP_K or kw.get("intermediate_size") != INTERMEDIATE:
        return "shape"
    w13, w13s, w2, w2s = (
        kw.get(k)
        for k in (
            "gemm1_weights",
            "gemm1_weights_scale",
            "gemm2_weights",
            "gemm2_weights_scale",
        )
    )
    if not all(
        torch.is_tensor(x) and x.is_contiguous() and x.is_cuda
        for x in (w13, w13s, w2, w2s)
    ):
        return "weights"
    le = w13.shape[0]
    if w13.dtype != torch.uint8 or tuple(w13.shape) != (
        le,
        2 * INTERMEDIATE,
        HIDDEN // 2,
    ):
        return "weights"
    if w2.dtype != torch.uint8 or tuple(w2.shape) != (le, HIDDEN, INTERMEDIATE // 2):
        return "weights"
    if w13s.element_size() != 1 or w13s.numel() != le * 2 * INTERMEDIATE * (
        HIDDEN // 16
    ):
        return "weight_scales"
    if w2s.element_size() != 1 or w2s.numel() != le * HIDDEN * (INTERMEDIATE // 16):
        return "weight_scales"
    if kw.get("local_num_experts") != le or not isinstance(
        kw.get("local_expert_offset"), int
    ):
        return "experts"
    for k in (
        "output1_scale_scalar",
        "output1_scale_gate_scalar",
        "output2_scale_scalar",
    ):
        if not _is_f32_vec(kw.get(k), le):
            return k
    r = kw.get("topk_ids")
    if isinstance(r, tuple):
        if (
            len(r) != 2
            or not torch.is_tensor(r[0])
            or r[0].dtype != torch.int32
            or tuple(r[0].shape) != (tokens, TOP_K)
            or not torch.is_tensor(r[1])
            or r[1].shape != r[0].shape
        ):
            return "routing"
    elif (
        not torch.is_tensor(r)
        or r.dtype != torch.int32
        or tuple(r.shape) != (tokens, TOP_K)
    ):
        return "routing"
    if torch.cuda.is_current_stream_capturing():
        from .moe_x_kernels import launch as L

        if not L.compiled(*tactics()):
            if not _state.get("not_built_warned"):
                _state["not_built_warned"] = True
                logger.warning(
                    "muse_vr_plugin: %s steps aside in CUDA-graph capture: kernels not built in this process: %s",
                    NAME,
                    L.missing_builds(*tactics()),
                )
            return "not_built_in_capture"
    return ""


def _run(kw: dict):
    from .moe_x_kernels import launch as L

    t1, t2 = tactics()
    r = kw["topk_ids"]
    if isinstance(r, tuple):
        ids, expert_weights = r[0].contiguous(), r[1]
    else:  # packed: (expert id << 16) | bf16 weight bits
        ids = (r >> 16).to(torch.int32)
        expert_weights = (r & 0xFFFF).to(torch.int16).view(torch.bfloat16)
    hs = kw["hidden_states"]
    tokens = hs.shape[0]
    out, e2p = L.routed_experts(
        hs,
        kw["hidden_states_scale"].reshape(tokens, HIDDEN // 16),
        kw["per_token_scale"].reshape(-1),
        ids,
        kw["gemm1_weights"],
        kw["gemm1_weights_scale"],
        kw["gemm2_weights"],
        kw["gemm2_weights_scale"],
        kw["output1_scale_gate_scalar"],
        kw["output1_scale_scalar"],
        kw["output2_scale_scalar"],
        num_experts=int(kw["num_experts"]),
        local_expert_offset=int(kw["local_expert_offset"]),
        t1=t1,
        t2=t2,
    )
    return [out, expert_weights, e2p]


def _routed_moe(*args, **kwargs):
    """Stand-in for ``flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe``."""
    orig = _state["orig"]
    if args or _state["broken"]:
        return orig(*args, **kwargs)
    reason = _eligible(kwargs)
    if (
        not reason and _real_eager_extend()
    ):  # decided before any launch (see the module docstring)
        reason = "eager_extend"
        if not _state.get("eager_warned"):
            _state["eager_warned"] = True
            logger.warning(
                "muse_vr_plugin: %s steps aside outside prefill CUDA-graph capture (eager prefill passes: prefill "
                "CUDA graphs off or outside the captured range); stock path for those calls",
                NAME,
            )
    if reason:
        if reason != "small":
            _step_aside(reason)
        return orig(*args, **kwargs)
    try:
        res = _run(kwargs)
    except Exception as e:  # noqa: BLE001 - leave the server on the stock path
        _state["broken"] = True
        logger.error(
            "muse_vr_plugin: %s failed (%s: %s); stock path for the rest of this process",
            NAME,
            type(e).__name__,
            e,
        )
        bed.report(NAME, False, reason=f"runtime failure: {type(e).__name__}")
        return orig(*args, **kwargs)
    tokens = kwargs["hidden_states"].shape[0]
    bucket = (
        1 << (tokens - 1).bit_length()
    )  # power-of-two bucket of the call's token count
    bed.count(
        NAME,
        f"x:{'capture' if torch.cuda.is_current_stream_capturing() else 'eager'}:tokens<={bucket}",
    )
    from .moe_x_kernels import launch as L

    # guard: a call that just ran (and so built its kernels) must find them in the capture-time check, or every captured
    # call would step aside ("not_built_in_capture"); named once, loudly
    if not torch.cuda.is_current_stream_capturing() and not _state.get(
        "built_check_done"
    ):
        _state["built_check_done"] = True
        miss = L.missing_builds(*tactics())
        if miss:
            bed.count(NAME, "guard:built_but_not_found")
            logger.error(
                "muse_vr_plugin: %s ran eagerly but its capture-time build check misses %s: captured calls would "
                "take the stock path",
                NAME,
                miss,
            )

    sched = L._cache.get("last_sched", "static")
    if sched != "static":
        bed.count(NAME, f"sched:{sched}")
    return res


def install() -> None:
    import flashinfer.fused_moe as ffm

    with _lock:
        if _state["orig"] is not None:
            return
        _state["orig"] = ffm.trtllm_fp4_block_scale_routed_moe
        ffm.trtllm_fp4_block_scale_routed_moe = _routed_moe


def uninstall() -> None:
    import flashinfer.fused_moe as ffm

    with _lock:
        if _state["orig"] is not None:
            ffm.trtllm_fp4_block_scale_routed_moe = _state["orig"]
            _state["orig"] = None


def engage() -> bool:
    if _state["engaged"] is not None:
        return _state["engaged"]
    if not bed.enabled(NAME):
        _state["engaged"] = False
        return False
    from .moe_x_kernels.arch import HELPERS_ENV, helpers_configured

    if not helpers_configured():
        bed.report(NAME, False, reason=f"{HELPERS_ENV} not set")
        _state["engaged"] = False
        return False
    ok, rel = _pins_ok()
    if not ok:
        bed.report(NAME, False, reason=f"pin mismatch: {rel}")
        _state["engaged"] = False
        return False
    install()
    t1, t2 = tactics()
    from .moe_x_kernels import launch as L

    sched = L.sched_mode()
    # the non-aligned handshake covers the rows kernels except the fused-quantizer form; the gather first GEMM is a
    # different kernel
    g1_rows = L.gemm1_rows_mode() and L.quant_form(t1) != "fused"
    bed.report(
        NAME,
        True,
        min_tokens=min_tokens(),
        gemm1=f"{t1.tile_m}x{t1.tile_n}/{t1.a_path}",
        gemm2=f"{t2.tile_m}x{t2.tile_n}",
        **({"sched": sched} if sched != "static" else {}),
        **(
            {"tmem_bar": "unaligned" if g1_rows else "unaligned-gemm2"}
            if L.tmem_bar_unaligned()
            else {}
        ),
    )
    _state["engaged"] = True
    return True

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Components ``swa_verify_attn`` and ``full_attn_split`` (both opt-in, default off), served by one AROUND hook on
``MuseSpark1xAttention.gated_attention``.

``swa_verify_attn`` (``MUSE_VR_SWA_VERIFY_ATTN=1``)
    In a target-verify step with at most ``MUSE_VR_SWA_VERIFY_ATTN_MAX_REQS`` requests (default by KV heads per rank,
    see ``_LIMITS``; tunable, bench/bench_attn_slice.py), the attention of the sliding-window layers runs
    ``ops.swa_verify_attn`` instead of FlashInfer's trtllm-gen paged decode kernel. Everything else in the method is
    SGLang's own code in the same order: the fused QK norm (+RoPE) that stores the new K/V rows and returns FP8 q, then
    the attention, then the fused output norm and gate. A call takes this path only when the stock call would reach the
    fixed query-length decode kernel with an FP8 KV cache: TRT-LLM MHA backend, 8 query tokens per request, no ragged
    verify layout, no NVFP4 KV, no context parallelism, a decoder (causal) sliding-window layer, no attention sinks,
    unit q scale, bf16 output, no skip-softmax threshold. Every other call runs SGLang's method unchanged. Numerics: the
    same rounding points as the stock kernel (e4m3 q/k/v, fp32 scores, e4m3 probabilities scaled by 448, fp32
    accumulation, bf16 output); key-range split, per-range maxima and fp32 summation order differ
    (``tests/test_swa_verify_attn.py`` compares both kernels against float64 references).

    With ``MUSE_VR_SWA_VERIFY_ATTN_NORM_GATE=1`` the kernel also applies the output norm and gate (the stock
    ``_fused_o_norm_gate`` call is then skipped) when that call would run SGLang's norm_gate JIT kernel without a
    weight and would pick its 8-byte-vector configuration (16 threads per row): that kernel picks it for at most 16 x
    SM-count (token, head) rows (``select_config`` in SGLang's ``norm_gate.cuh``); above that the stock call runs. The
    arithmetic of that configuration is reproduced in the kernel epilogue, bitwise (``tests/test_swa_verify_attn.py``).

``attn_qk_prologue`` (``MUSE_VR_ATTN_QK_PROLOGUE=1``, with ``swa_verify_attn``)
    On the ``swa_verify_attn`` path, the kernel also does the work of the fused QK norm call (per-head RMS norm of q and
    k, RoPE, FP8 queries, FP8 K/V-cache store of the new tokens) from the bf16 QKV projection output, and the QK-norm
    kernel launch is skipped. It engages for at most ``MUSE_VR_ATTN_QK_PROLOGUE_MAX_REQS`` requests (default by KV heads
    per rank, see ``_LIMITS``; tunable, bench/bench_attn_slice.py) when that call would run SGLang's Triton kernel
    (``FusedQKNorm._forward_triton``) for a row count the prologue covers; the arithmetic is the Triton kernel's for
    that row count (bitwise: ``tests/test_swa_qk_prologue.py``), so outputs and the KV cache are unchanged. The Triton
    version the arithmetic was derived from is pinned as well.

``full_verify_attn`` (``MUSE_VR_FULL_VERIFY_ATTN=1``)
    In a target-verify step with at most ``MUSE_VR_FULL_VERIFY_ATTN_MAX_REQS`` requests (default by KV heads per rank,
    see ``_LIMITS``; tunable, bench/bench_attn_slice.py --full), the attention of the full-attention layers runs
    ``ops.full_verify_attn`` (tensor-core MMA kernel with a length-proportional key split and a merge kernel) instead of
    FlashInfer's trtllm-gen decode kernel, under the same conditions as ``swa_verify_attn`` plus page size 128 and an
    even KV-head count per rank. The QK norm call stays SGLang's; with the norm-gate conditions of ``swa_verify_attn``
    (and ``MUSE_VR_SWA_VERIFY_ATTN_NORM_GATE=1``) the output norm and gate run in the merge kernel. With
    ``MUSE_VR_ATTN_QK_PROLOGUE=1`` (and ``MUSE_VR_FULL_VERIFY_ATTN_QK`` not 0) the kernel also does the work of the
    fused QK norm call (no RoPE in these layers) from the bf16 QKV output under the conditions of ``attn_qk_prologue``
    (Triton kernel, covered row count), bitwise (``tests/test_full_qk_prologue.py``). Numerics: the same
    rounding points as the stock kernel; key-range split, per-tile maxima and fp32 summation order differ
    (``tests/test_full_verify_attn.py``). Takes precedence over ``full_attn_split`` where both apply.

``full_attn_split`` (``MUSE_VR_FULL_ATTN_SPLIT=1``)
    In a target-verify step with one request, the full-attention layers run the stock kernel with the key split
    FlashInfer derives for ``MUSE_VR_FULL_ATTN_SPLIT_FACTOR`` (default 0.75; tunable,
    bench/bench_full_attn_split.py) times the SM count: the SM count FlashInfer reads while the call is made (or
    captured) is scaled. Only the split of the key range across thread blocks changes (and with it the fp32
    merge order).

``full_attn_wide`` (``MUSE_VR_FULL_ATTN_WIDE=1``)
    In a target-verify step with more requests than ``full_verify_attn`` takes (or any count when that component is
    off), from ``MUSE_VR_FULL_ATTN_WIDE_MIN_REQS`` (default: the microbench break-even, 10 requests at 4 KV heads per
    rank and 9 at 2) up to 256, the attention of the full-attention layers runs
    ``ops.full_attn_wide`` (persistent kernel: the summed key range of all requests split evenly over one thread block
    per SM, whole token rows of the K/V pools per block, split requests merged by a second kernel) instead of
    FlashInfer's trtllm-gen decode kernel, under the conditions of ``full_verify_attn`` with 2 or 4 KV heads per rank.
    The QK norm call stays SGLang's; the output norm and gate too, unless ``MUSE_VR_FULL_ATTN_WIDE_NORM_GATE=1``: then,
    where SGLang's ``_fused_o_norm_gate`` call would run its weightless norm_gate JIT kernel in its 8-byte-vector
    configuration (or, with ``MUSE_VR_NORM_GATE_VEC16=1``, its 16-byte one, which it picks above 16 x SM-count rows), the
    merge kernel applies them with that configuration's arithmetic, bitwise (``tests/test_full_attn_wide_norm_gate.py``),
    and the stock call is skipped. Numerics: the rounding points of the stock kernel;
    key split, per-tile maxima, fp32 summation order and the merge order differ (``tests/test_full_attn_wide.py``).
    Every call that steps aside is counted with its reason and logged once per reason (WARNING).

SWA-T (``MUSE_VR_SWA_T=1``, ``ops.swa_t``)
    Sliding-window layers with MUSE_VR_SWA_T_MIN_REQS (12) to MUSE_VR_SWA_T_MAX_REQS (32) requests at 2 KV heads per rank
    (at 4 KV heads: MUSE_VR_SWA_T_KV4_MIN_REQS (6) to MUSE_VR_SWA_T_KV4_MAX_REQS (48), one cluster per request and KV-head pair)
    take SWA-T before the paths above: the QK-norm call, then one cluster of kC CTAs per request (kC from a runtime
    occupancy query of the kernel, ``ops.swa_t.kc_for``) with per-tile fp32 arithmetic and an in-cluster merge, the
    output norm + gate fused when the norm_gate option is on (in norm_gate's 8-byte form; with
    ``MUSE_VR_NORM_GATE_VEC16=1`` also in its 16-byte form, which SGLang picks above 16 x SM-count rows, bitwise to it:
    ``tests/test_swa_t.py``); with ``MUSE_VR_SWA_T_QK=1`` the kernel also does the
    QK-norm call's work (prologue). Numerics: the rounding points of the stock kernel; key split, per-tile maxima and
    fp32 summation order differ (``tests/test_swa_t.py``).

``swa_t6`` (``MUSE_VR_SWA_T6=1``, with ``MUSE_VR_SWA_T5_IMPL=6`` and a non-empty ``MUSE_VR_SWA_T6_BUCKETS``)
    KNOWN DEFECT, keep off: pool rows outside a request's key range (stale slots, entries past its last page) can reach
    the kernel's probability x V products, so non-finite stale bytes give NaN output rows (a masked score gives a zero
    probability, and zero times a non-finite value is not zero). Needs predicated or zero-filled K / V loads and a merge
    that skips absent partials before any use.
    A sliding-window verify call whose query-token count falls in a listed bucket of ``ops.swa_verify_t5`` (64, 128,
    192, 256) runs that op's t6 kernel for the attention, ahead of the SWA-T and ``swa_verify_attn`` paths: SGLang's
    own fused QK norm call (which stores the new K/V rows and returns FP8 q), then the t6 kernel (blocks per request
    and KV head, cluster merge and L2 prefetch by the op's policy, PDL), then SGLang's own fused output norm and gate.
    Under the conditions of ``swa_verify_attn`` (FP8 KV cache whose store and read buffers agree, bf16 output, float
    scales, no skip-softmax threshold) plus page size 128 and the op's shape support. The op's workspace and kernel
    variants are prepared at the first eager call; a call first met inside a CUDA-graph capture steps aside. Every
    other call keeps the order below. Numerics: the rounding points of the stock kernel; key split, per-block maxima
    and fp32 summation order differ (``tests/test_swa_verify_t5.py``).

Engagement needs the sha256 of every SGLang source whose behaviour the hook mirrors to match the pinned revision
(and, for ``full_attn_split``, FlashInfer's ``decode.py``); install decisions and per-shape call counts are
recorded with ``bed.report`` / ``bed.count``.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import logging
import os
from pathlib import Path

import torch

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/fused_op.py": ("_oot_dispatch_key", "get_fused_op_backend"),
    "srt/layers/radix_attention.py": ("AttentionType",),
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
    "kernels/spec.py": ("KernelBackend",),
    "srt/layers/attention/trtllm_mha_backend.py": ("TRTLLMHAAttnBackend",),
    "srt/layers/cp/utils.py": ("is_cp_active",),
    "srt/model_executor/forward_context.py": ("get_attn_backend",),
    "kernels/ops/layernorm/muse_spark_norm_gate.py|kernels/ops/muse_spark_v12/fused_norm_gate.py": (
        "FusedRMSNormGate",
    ),
    "srt/environ.py": ("envs",),
}


logger = logging.getLogger(__name__)

SWA = "swa_verify_attn"
QK = "attn_qk_prologue"
FULL = "full_attn_split"
FULLV = "full_verify_attn"
FULLW = "full_attn_wide"
SWAT = "swa_t"
SWAT6 = "swa_t6"
TARGET = "sglang.srt.models.muse_spark.MuseSpark1xAttention.gated_attention"

PINS = {
    "sglang/srt/models/muse_spark.py": (
        "7e5018b3cb2ef41ee6cf09407e91e28dde069e5a33719ea58beb0d7137087355",
        "8bafbb8802fb0d9f81c190e87b9012fca546182be08bef6c8b3db814d27bfe19",
        "c0db42467a960c509b658c3c40570ee889f73d41b52cdbd80b9c291c1792a33f",
    ),
    "sglang/srt/layers/attention/trtllm_mha_backend.py": (
        "566d1031d644f71a8adb84d1e2a1581d3a6d1c3cac1e507451ac4cf034c39da4",
        "07223cbf82ab5ac14fcbaa9a47a8351132991016bae66c122bf09762ed8ed646",
    ),
    "sglang/srt/layers/radix_attention.py": (
        "df6e94b67b61feb4f8e3a7b1fdb0c0444cc56e4bf9d90a622e028bddff31619e",
        "56a6acb0aaeb78c7c9bb8d1d78e5a23ac51fa35436b993aeb5a630cc9f285ff0",
    ),
    "sglang/srt/layers/attention/base_attn_backend.py": (
        "66803e0026f90cdeced7471ca3e28ca1949229f06e6e627450f5a2598f4b1de8",
        "3ed449219c73b146bc3e6472390c933cdd1da18c88edcd6bfbb7d530cf4dafe0",
    ),
    "sglang/kernels/ops/muse_spark_v12/fused_qk_norm.py": (
        "99ed08f1f35bbd4174c4a40becfc7978b7a5ba0a055818e79470254e5c8a2b67",
        "af4a5bb8de205a99761e6959447748e2e00af0831e58e37fa0f862ce9e964a25",
    ),
}
# Additionally for the fused output norm + gate.
PINS_NORM_GATE = {
    "sglang/kernels/ops/muse_spark_v12/fused_norm_gate.py": (
        "f127386eafe01455722466c9abf58c8f265c680e90a5e8b8b30175a6b27b44a6",
        "9463859fa6df630123239833d6486c8b8069942ebf127246c49217b44683c662",
    ),
    "sglang/kernels/ops/muse_spark_v12/norm_gate.py": (
        "376f33a77014dd3f6ed160d3b6a46eb7ef50f3f4bde783bafdbcbaa4fe39aed9",
        "fb66534284d3980d9add7f2417d2b5f7c6e94c5f230c05eba93bf31db72d4815",
    ),
    "sglang/kernels/jit/csrc/muse_spark_v12/norm_gate.cuh": "c2a99a7b37fb810310539010cebea30352b25c1e841b41f8a0fb44aae657bf80",
    "sglang/kernels/fused_op.py": (
        "652ceeea3bbbdc2b864c3fa3cf914148d486392a90384506043d5c2ccf60eb3e",
        "749bc702db4102bb6e2620626a906ada67856daa1459bb5141bd1d652168cbd7",
    ),
}
# The prologue arithmetic reproduces the fused QK-norm Triton kernel as each validated Triton release compiles it
# (ops/swa_verify_attn.py, QK_ARITH / qk_arith); other releases keep the stock call.
# flashinfer/decode.py (FlashInfer 0.7.1 and 0.7.2) for full_attn_split: the SM count is read through the
# module-level name ``get_device_sm_count`` inside ``trtllm_batch_decode_with_kv_cache`` (once, in both releases).
FLASHINFER_DECODE_SHA256 = (
    "8917d7f1a80707d823455d69d105d90a57e8bb8a71b248348f02d62c547a6f91",  # FlashInfer 0.7.1
    "de1f242bf2f598bc973a9d2f9625f8db1e4ccfc24085f686b2d0304c9076b20d",  # FlashInfer 0.7.2
)

TOKENS = 8
# Default request limits per component by KV heads per rank (8 query heads each): the largest request count at which
# the component measured faster than stock in bench/bench_attn_slice.py (BENCH_KV_HEADS=2/4/8; 64K context for the
# full-attention layers). Other head counts: the 4-head limit scaled by 4 / KV heads (at least 1). The environment
# variables override the limit for every head count.
_LIMITS = {
    "max_reqs": ("MUSE_VR_SWA_VERIFY_ATTN_MAX_REQS", {2: 16, 4: 6, 8: 4}),
    "qk_max_reqs": ("MUSE_VR_ATTN_QK_PROLOGUE_MAX_REQS", {2: 4, 4: 3, 8: 1}),
    "fullv_max_reqs": ("MUSE_VR_FULL_VERIFY_ATTN_MAX_REQS", {2: 8, 4: 4, 8: 1}),
}
_state = {
    "swa": False,
    "norm_gate": False,
    "qk": False,
    "full": False,
    "factor": 0.75,
    "max_reqs": None,
    "qk_max_reqs": None,
    "fullv": False,
    "fullv_max_reqs": None,
    "fullv_pdl": True,
    "scale_sm": None,
    "fullv_qk": False,
    "fullw": False,
    "fullw_ng": False,
    "ng16": False,
    "fullw_min": None,
    "fullw_pdl": True,
    "swa_t": False,
    "swa_t_min": 12,
    "swa_t_max": 32,
    "swa_t_qk": False,
    "swa_t_min4": 6,
    "swa_t_max4": 48,
    "swa_t6": False,
    "swa_t6_buckets": frozenset(),
}
_warned: set = set()
_t6_ready: set = set()  # (device index, KV heads, window) whose t6 workspace and kernel variants are prepared
# device indices on which a t6 call has been captured into a CUDA graph: the op's workspace is per device and a graph
# keeps its pointers, so after a capture no new (KV heads, window) key may prepare (grow) it on that device
_t6_captured: set = set()


def _limit(name: str, num_kv_heads: int) -> int:
    """Request limit of a component for this KV-head count per rank (environment override, else ``_LIMITS``)."""
    if _state[name] is not None:
        return _state[name]
    table = _LIMITS[name][1]
    return table.get(num_kv_heads, max(1, table[4] * 4 // max(1, num_kv_heads)))


def _env_limit(name: str) -> int | None:
    raw = os.environ.get(_LIMITS[name][0], "").strip()
    try:
        return int(raw) if raw else None
    except ValueError:
        return None


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _flashinfer_decode_pinned() -> bool:
    spec = importlib.util.find_spec("flashinfer")
    if spec is None or not spec.submodule_search_locations:
        return False
    path = Path(list(spec.submodule_search_locations)[0]) / "decode.py"
    try:
        got = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return False
    if not bed.sha_ok(got, FLASHINFER_DECODE_SHA256):
        logger.warning(
            "muse_vr_plugin: flashinfer/decode.py sha256 %s does not match the pin %s",
            got[:16],
            bed.sha_label(FLASHINFER_DECODE_SHA256),
        )
        return False
    return True


def register() -> bool:
    """Register the hook when at least one component is enabled and its pins match."""
    want_swa, want_full, want_fullv = (
        bed.enabled(SWA),
        bed.enabled(FULL),
        bed.enabled(FULLV),
    )
    want_fullw = bed.enabled(FULLW)
    want_swat = bed.enabled(SWAT)
    want_swat6 = bed.enabled(SWAT6)
    if not (
        want_swa or want_full or want_fullv or want_fullw or want_swat or want_swat6
    ):
        return False
    pins_ok = all(bed.pinned(rel, sha) for rel, sha in PINS.items())
    _state["swa"] = want_swa and pins_ok
    ng_env = os.environ.get("MUSE_VR_SWA_VERIFY_ATTN_NORM_GATE", "0").strip().lower()
    want_ng = (want_swa or want_fullv or want_swat) and ng_env in (
        "1",
        "true",
        "yes",
        "on",
    )
    _state["norm_gate"] = (
        (_state["swa"] or ((want_fullv or want_swat) and pins_ok))
        and want_ng
        and all(bed.pinned(rel, sha) for rel, sha in PINS_NORM_GATE.items())
    )
    _state["swa_t"] = want_swat and pins_ok
    # The SWA-T QK prologue reproduces the stock fused QK-norm kernel's arithmetic as the running Triton release compiles
    # it (ops.swa_t.qk_arith: a validated (release, FP-fusion default) pair); otherwise SWA-T runs without it.
    from .ops.swa_t import qk_arith as _swat_qk_arith

    _state["swa_t_qk"] = (
        _state["swa_t"]
        and _swat_qk_arith() is not None
        and os.environ.get("MUSE_VR_SWA_T_QK", "0").strip().lower()
        in ("1", "true", "yes", "on")
    )
    _state["swa_t_min"] = _env_int("MUSE_VR_SWA_T_MIN_REQS", 12)
    _state["swa_t_max"] = _env_int("MUSE_VR_SWA_T_MAX_REQS", 32)
    # 4 KV heads per rank (one cluster per request and KV-head pair): its own request window
    _state["swa_t_min4"] = _env_int("MUSE_VR_SWA_T_KV4_MIN_REQS", 6)
    _state["swa_t_max4"] = _env_int("MUSE_VR_SWA_T_KV4_MAX_REQS", 48)
    _state["fullv"] = want_fullv and pins_ok
    _state["fullv_max_reqs"] = _env_limit("fullv_max_reqs")
    _state["fullv_pdl"] = os.environ.get(
        "MUSE_VR_FULL_VERIFY_ATTN_PDL", "1"
    ).strip() not in ("0", "false", "no")
    want_qk = bed.enabled(QK)
    from .ops.swa_verify_attn import qk_arith

    triton_ok = qk_arith() is not None
    _state["qk"] = _state["swa"] and want_qk and triton_ok
    _state["fullw"] = want_fullw and pins_ok
    _state["fullw_ng"] = (
        _state["fullw"]
        and os.environ.get("MUSE_VR_FULL_ATTN_WIDE_NORM_GATE", "0").strip().lower()
        in ("1", "true", "yes", "on")
        and all(bed.pinned(rel, sha) for rel, sha in PINS_NORM_GATE.items())
    )
    # the fused norm + gate epilogues of full_attn_wide (and SWA-T) also in SGLang norm_gate's 16-byte configuration
    _state["ng16"] = os.environ.get("MUSE_VR_NORM_GATE_VEC16", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    raw_min = os.environ.get("MUSE_VR_FULL_ATTN_WIDE_MIN_REQS", "").strip()
    _state["fullw_min"] = int(raw_min) if raw_min.isdigit() else None
    _state["fullw_pdl"] = os.environ.get(
        "MUSE_VR_FULL_ATTN_WIDE_PDL", "1"
    ).strip() not in ("0", "false", "no")
    _state["fullv_qk"] = (
        _state["fullv"]
        and want_qk
        and triton_ok
        and os.environ.get("MUSE_VR_FULL_VERIFY_ATTN_QK", "1").strip()
        not in ("0", "false", "no")
    )
    _state["qk_max_reqs"] = _env_limit("qk_max_reqs")
    _state["full"] = want_full and pins_ok and _flashinfer_decode_pinned()
    _state["max_reqs"] = _env_limit("max_reqs")
    factor = _env_float("MUSE_VR_FULL_ATTN_SPLIT_FACTOR", 0.75)
    if not 0.0 < factor <= 1.0:
        # FlashInfer sizes its workspace and counters for at most the device's SM count; above 1 is not allowed.
        logger.warning(
            "muse_vr_plugin: MUSE_VR_FULL_ATTN_SPLIT_FACTOR=%s outside (0, 1]; using %s",
            factor,
            1.0 if factor > 1.0 else 0.75,
        )
        factor = 1.0 if factor > 1.0 else 0.75
    _state["factor"] = factor
    if want_swa:
        bed.report(
            SWA,
            _state["swa"],
            pins=pins_ok,
            max_reqs=_limit_desc("max_reqs"),
            norm_gate=_state["norm_gate"],
        )
    if want_qk:
        bed.report(
            QK,
            _state["qk"] or _state["fullv_qk"],
            swa=_state["swa"],
            full=_state["fullv_qk"],
            triton=_triton_version(),
            arith=qk_arith(),
            max_reqs=_limit_desc("qk_max_reqs"),
        )
    if want_full:
        bed.report(FULL, _state["full"], pins=pins_ok, factor=_state["factor"])
    if want_fullv:
        bed.report(
            FULLV,
            _state["fullv"],
            pins=pins_ok,
            max_reqs=_limit_desc("fullv_max_reqs"),
            pdl=_state["fullv_pdl"],
            norm_gate=_state["norm_gate"],
        )
    if want_fullw:
        bed.report(
            FULLW,
            _state["fullw"],
            pins=pins_ok,
            min_reqs=_state["fullw_min"]
            or "break-even per KV heads (and above full_verify_attn)",
            pdl=_state["fullw_pdl"],
            norm_gate=_state["fullw_ng"],
            norm_gate_vec16=_state["ng16"],
            **_fullw_skel_report(),
        )
    if want_swat:
        bed.report(
            SWAT,
            _state["swa_t"],
            pins=pins_ok,
            min_reqs=_state["swa_t_min"],
            max_reqs=_state["swa_t_max"],
            min_reqs_kv4=_state["swa_t_min4"],
            max_reqs_kv4=_state["swa_t_max4"],
            norm_gate=_state["norm_gate"],
            qk_prologue=_state["swa_t_qk"],
            triton=_triton_version(),
            qk_arith=_swat_qk_arith(),
        )
    if want_swat6:
        impl6 = os.environ.get("MUSE_VR_SWA_T5_IMPL", "5").strip() == "6"
        from .ops.swa_verify_t5 import t6_enabled_buckets

        buckets = t6_enabled_buckets()
        _state["swa_t6_buckets"] = buckets
        _state["swa_t6"] = pins_ok and impl6 and bool(buckets)
        why = (
            None
            if _state["swa_t6"]
            else (
                "pins"
                if not pins_ok
                else "MUSE_VR_SWA_T5_IMPL is not 6"
                if not impl6
                else "MUSE_VR_SWA_T6_BUCKETS is empty"
            )
        )
        bed.report(
            SWAT6,
            _state["swa_t6"],
            pins=pins_ok,
            buckets=",".join(str(b) for b in sorted(buckets)),
            cluster=os.environ.get("MUSE_VR_SWA_T6_CLUSTER", "0"),
            reason=why,
        )
    if not (
        _state["swa"]
        or _state["full"]
        or _state["fullv"]
        or _state["fullw"]
        or _state["swa_t"]
        or _state["swa_t6"]
    ):
        return False
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    def hook(original, *args, **kwargs):
        return _around_gated_attention(original, *args, **kwargs)

    HookRegistry.register(TARGET, hook, HookType.AROUND)
    return True


def _limit_desc(name: str):
    """The request limit as recorded at install: the override, or the per-KV-head table."""
    if _state[name] is not None:
        return _state[name]
    return "kv_heads " + " ".join(
        f"{h}:{n}" for h, n in sorted(_LIMITS[name][1].items())
    )


def _triton_version() -> str | None:
    try:
        import triton

        return str(triton.__version__)
    except Exception:  # noqa: BLE001
        return None


def _qk_prologue_args(attn, positions, qkv, kv_store):
    """(positions, cos_sin, eps) when the stock fused QK-norm call would run SGLang's Triton kernel on a row count
    the kernel prologue covers, else None."""
    if (
        not _state["qk"]
        or kv_store is None
        or qkv.shape[0] // TOKENS > _limit("qk_max_reqs", attn.attn.tp_k_head_num)
    ):
        return None
    import sglang.kernels.fused_op as fused_op
    from sglang.kernels.spec import KernelBackend

    from .ops import swa_verify_attn as op

    qkn = attn._fused_qk_norm
    rope = attn.rotary_emb
    if rope is None or qkn.is_torch_compile or qkn._head_dim != 64:
        return None
    if (
        fused_op.get_fused_op_backend() is not None
        or fused_op._oot_dispatch_key() is not None
    ):
        return None
    cands = qkn._auto_backend_candidates()
    cos_sin = rope.cos_sin_cache
    if not cands or cands[0] is not KernelBackend.TRITON:
        return None
    if not qkn.backend_eligible(
        KernelBackend.TRITON, qkv, positions, cos_sin, kv_store
    ):
        return None
    layer = attn.attn
    if op.qk_class(qkv.shape[0], layer.tp_q_head_num, layer.tp_k_head_num) is None:
        return None
    loc = kv_store[2]
    if not (
        positions.dtype == torch.int64
        and positions.is_contiguous()
        and loc.dtype == torch.int64
        and loc.is_contiguous()
        and cos_sin.is_contiguous()
        and qkv.stride(0) % 8 == 0
        and qkv.data_ptr() % 16 == 0
        and cos_sin.data_ptr() % 16 == 0
    ):
        return None
    return positions, cos_sin, float(qkn._eps)


def _qk_rope_args(attn, positions, qkv, kv_store):
    """(positions, cos_sin, eps) when the stock fused QK-norm call (RoPE layer) would run SGLang's Triton kernel on a row
    count whose column class the kernel prologues reproduce, with the operand layouts they read, else None."""
    if kv_store is None:
        return None
    import sglang.kernels.fused_op as fused_op
    from sglang.kernels.spec import KernelBackend

    from .ops import swa_verify_attn as op

    qkn = attn._fused_qk_norm
    rope = attn.rotary_emb
    if rope is None or qkn.is_torch_compile or qkn._head_dim != 64:
        return None
    if (
        fused_op.get_fused_op_backend() is not None
        or fused_op._oot_dispatch_key() is not None
    ):
        return None
    cands = qkn._auto_backend_candidates()
    cos_sin = rope.cos_sin_cache
    if not cands or cands[0] is not KernelBackend.TRITON:
        return None
    if not qkn.backend_eligible(
        KernelBackend.TRITON, qkv, positions, cos_sin, kv_store
    ):
        return None
    layer = attn.attn
    if op.qk_class(qkv.shape[0], layer.tp_q_head_num, layer.tp_k_head_num) is None:
        return None
    loc = kv_store[2]
    if not (
        positions.dtype == torch.int64
        and positions.is_contiguous()
        and loc.dtype == torch.int64
        and loc.is_contiguous()
        and cos_sin.is_contiguous()
        and qkv.stride(0) % 8 == 0
        and qkv.data_ptr() % 16 == 0
        and cos_sin.data_ptr() % 16 == 0
        and cos_sin.dtype == torch.float32
        and qkv.stride(1) == 1
    ):
        return None
    return positions, cos_sin, float(qkn._eps)


def _qk_full_prologue_eps(attn, qkv, kv_store):
    """eps when the stock fused QK-norm call of a layer without RoPE would run SGLang's Triton kernel on a row count
    the full_verify_attn prologue covers, else None."""
    if not _state["fullv_qk"] or kv_store is None:
        return None
    import sglang.kernels.fused_op as fused_op
    from sglang.kernels.spec import KernelBackend

    from .ops import swa_verify_attn as op

    qkn = attn._fused_qk_norm
    if attn.rotary_emb is not None or qkn.is_torch_compile or qkn._head_dim != 64:
        return None
    if (
        fused_op.get_fused_op_backend() is not None
        or fused_op._oot_dispatch_key() is not None
    ):
        return None
    cands = qkn._auto_backend_candidates()
    if not cands or cands[0] is not KernelBackend.TRITON:
        return None
    if not qkn.backend_eligible(KernelBackend.TRITON, qkv, None, None, kv_store):
        return None
    layer = attn.attn
    if op.qk_class(qkv.shape[0], layer.tp_q_head_num, layer.tp_k_head_num) is None:
        return None
    loc = kv_store[2]
    if not (
        loc.dtype == torch.int64
        and loc.is_contiguous()
        and qkv.stride(1) == 1
        and qkv.stride(0) % 8 == 0
        and qkv.data_ptr() % 16 == 0
    ):
        return None
    return float(qkn._eps)


def _stock_diverts_attention(ra, forward_batch) -> bool:
    """Whether stock RadixAttention.forward would leave the attention backend for this call (prefill-wrapper layers):
    the branch layout checks its tc-piecewise forward context; SGLang main (no such context) diverts to eager
    attention inside a prefill graph scope under a breakable or full prefill CUDA graph, for extend modes."""
    if hasattr(ra, "get_tc_piecewise_forward_context"):
        return ra.get_tc_piecewise_forward_context() is not None
    return bool(
        ra.is_in_prefill_graph_scope()
        and forward_batch.forward_mode.is_extend()
        and (ra.is_in_breakable_cuda_graph() or ra.is_in_full_prefill_graph())
    )


def _common_reason(attn, qkv, forward_batch):
    """Reason the call must run stock, or None; checks shared by both components. Returns (reason, backend)."""
    import sglang.srt.layers.radix_attention as ra
    from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend
    from sglang.srt.layers.cp.utils import is_cp_active
    from sglang.srt.model_executor.forward_context import get_attn_backend

    if not forward_batch.forward_mode.is_target_verify():
        return "not target verify", None
    m = qkv.shape[0]
    if m == 0 or m % TOKENS:
        return "row count", None
    layer = attn.attn
    if layer.attn_type != ra.AttentionType.DECODER:
        return "attention type", None
    if layer.use_prefill_attention_wrapper and _stock_diverts_attention(
        ra, forward_batch
    ):
        return "piecewise prefill context", None
    backend = get_attn_backend()
    if type(backend) is not TRTLLMHAAttnBackend:
        return "attention backend", None
    md = backend.forward_metadata
    if md is None or md.is_ragged_verify or md.max_seq_len_q != TOKENS:
        return "verify layout", None
    if (
        backend.data_type != torch.float8_e4m3fn
        or backend.is_nvfp4_kvcache
        or backend.is_xqa_impl
    ):
        return "kv cache format", None
    if backend._forward_extend_uses_native_fp4(forward_batch) or is_cp_active(
        forward_batch
    ):
        return "nvfp4 / context parallel", None
    if attn._fused_qk_norm is None or not attn.use_o_gate:
        return "model configuration", None
    return None, backend


def _norm_gate_eps(attn, qkv, enabled=None):
    """eps when the stock ``_fused_o_norm_gate`` call would run the weightless norm_gate JIT kernel, else None.
    ``enabled``: the component's own norm-gate switch (default: the swa / full_verify / SWA-T one)."""
    if not (_state["norm_gate"] if enabled is None else enabled):
        return None
    import sglang.kernels.fused_op as fused_op
    from sglang.kernels.ops.muse_spark_v12.fused_norm_gate import FusedRMSNormGate
    from sglang.kernels.spec import KernelBackend

    og = attn._fused_o_norm_gate
    if (
        type(og) is not FusedRMSNormGate
        or not (og._use_o_norm and og._use_o_gate)
        or og._head_dim != 64
    ):
        return None
    if (
        fused_op.get_fused_op_backend() is not None
        or fused_op._oot_dispatch_key() is not None
        or og.is_torch_compile
    ):
        return None
    cands = og._auto_backend_candidates()
    if (
        not cands
        or cands[0] is not KernelBackend.JIT
        or not og.backend_eligible(KernelBackend.JIT, qkv[:1, :64])
    ):
        return None
    og._ensure_cached_weight()
    if og._effective_weight is not None:
        return None
    # The fused epilogues reproduce the 8-byte-vector configuration (16 threads per row), which SGLang's norm_gate
    # selects while rows * head_dim * 2 / 8 <= SMs * 256 (select_config), i.e. for at most 16 x SMs rows of 64.
    rows = qkv.shape[0] * attn.attn.tp_q_head_num
    if rows > 16 * _sm_count(qkv.device):
        return None
    return float(og._eps)


# SGLang norm_gate's configuration table (vector bytes, warps per block) and its selection rule (select_config in its
# norm_gate.cuh, pinned in PINS_NORM_GATE), for head_dim 64 and bf16; the fused epilogues reproduce the 8- and 16-byte
# vector arithmetic only.
_NG_CONFIGS = (
    (4, 1),
    (4, 2),
    (4, 4),
    (4, 8),
    (8, 1),
    (8, 2),
    (8, 4),
    (8, 8),
    (16, 1),
    (16, 2),
    (16, 4),
    (16, 8),
    (32, 1),
    (32, 2),
    (32, 4),
    (32, 8),
)


def _ng_rows_per_block(vec: int, warps: int, vec32_ok: bool, max_vec: int) -> int:
    if vec > max_vec or (vec > 16 and not vec32_ok):
        return 0
    per_thread = vec // 4 * 2  # bf16 pairs per vector, two elements each
    if 64 % per_thread:
        return 0
    tpr = 64 // per_thread
    if tpr > 32:
        return 0
    return (32 // tpr) * warps


def _ng_select_vec(
    num_tokens: int, num_heads: int, vec32_ok: bool, num_sm: int, max_vec: int
):
    rows = num_tokens * num_heads
    desired = 8
    if rows * 128 // 8 > num_sm * 256:
        desired = 16
    if desired == 16 and rows * 128 // 16 > num_sm * 2048:
        desired = 32
    target = 128 if desired == 32 else 256
    best, best_key = -1, 0
    for i, (vec, warps) in enumerate(_NG_CONFIGS):
        rpb = _ng_rows_per_block(vec, warps, vec32_ok, max_vec)
        if rpb == 0 or num_heads % rpb:
            continue
        threads = warps * 32
        dist = min(abs(threads - target), 1023)
        key = (
            (int(threads >= 128) << 40)
            | (int(vec == desired) << 32)
            | ((1024 - dist) << 8)
            | vec
        )
        if best < 0 or key > best_key:
            best, best_key = i, key
    if best < 0:
        for i, (vec, warps) in enumerate(_NG_CONFIGS):
            if _ng_rows_per_block(vec, warps, vec32_ok, max_vec) > 0:
                return vec
        return None
    return _NG_CONFIGS[best][0]


def _norm_gate_cfg(attn, qkv, enabled, allow16: bool):
    """(eps, vector bytes 8 or 16) when the stock ``_fused_o_norm_gate`` call would run the weightless norm_gate JIT kernel
    in a configuration a fused epilogue reproduces (8 bytes; 16 bytes only with ``allow16``), else None."""
    if not enabled:
        return None
    import sglang.kernels.fused_op as fused_op
    from sglang.kernels.ops.muse_spark_v12.fused_norm_gate import FusedRMSNormGate
    from sglang.kernels.spec import KernelBackend

    og = attn._fused_o_norm_gate
    if (
        type(og) is not FusedRMSNormGate
        or not (og._use_o_norm and og._use_o_gate)
        or og._head_dim != 64
    ):
        return None
    if (
        fused_op.get_fused_op_backend() is not None
        or fused_op._oot_dispatch_key() is not None
        or og.is_torch_compile
    ):
        return None
    cands = og._auto_backend_candidates()
    if (
        not cands
        or cands[0] is not KernelBackend.JIT
        or not og.backend_eligible(KernelBackend.JIT, qkv[:1, :64])
    ):
        return None
    og._ensure_cached_weight()
    if og._effective_weight is not None:
        return None
    # the vector width SGLang's select_config picks for this call (the 32-byte form needs 32-byte aligned rows: its
    # outputs are fresh allocations, the gate a view of qkv; either way the 8 / 16-byte answer only depends on the rows)
    tokens, heads = qkv.shape[0], attn.attn.tp_q_head_num
    gate_off = attn.q_size + 2 * attn.kv_size
    vec32_ok = (qkv.data_ptr() + 2 * gate_off) % 32 == 0 and (
        qkv.stride(0) * 2
    ) % 32 == 0
    vecs = {
        _ng_select_vec(tokens, heads, vec32_ok, _sm_count(qkv.device), mv)
        for mv in (16, 32)
    }
    if len(vecs) != 1:
        return None
    vec = vecs.pop()
    if vec == 8 or (vec == 16 and allow16):
        return float(og._eps), vec
    return None


_SM_COUNT: dict = {}


def _sm_count(device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _SM_COUNT:
        _SM_COUNT[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _SM_COUNT[idx]


def _swa_reason(attn, qkv, forward_batch, backend):
    from .ops import swa_verify_attn as op

    layer = attn.attn
    if layer.sliding_window_size is None or layer.sliding_window_size < 0:
        return "full attention layer"
    hkv = layer.tp_k_head_num
    if qkv.shape[0] // TOKENS > _limit("max_reqs", hkv):
        return "requests above max"
    if not op.supported(
        layer.tp_q_head_num,
        layer.tp_k_head_num,
        layer.head_dim,
        layer.sliding_window_size,
    ):
        return "shape"
    if backend.q_data_type != torch.bfloat16:
        return "output dtype"
    from sglang.srt.environ import envs

    if envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get() is not None:
        return "skip-softmax threshold set"
    bmm1, bmm2 = backend._get_bmm_scales(layer)
    if not isinstance(bmm1, float) or not isinstance(bmm2, float):
        return "tensor scales"
    ng = _norm_gate_eps(attn, qkv) is not None
    reqs = qkv.shape[0] // TOKENS
    fuse_qk = _state["qk"] and reqs <= _limit("qk_max_reqs", hkv)
    qk = (
        op.qk_class(qkv.shape[0], layer.tp_q_head_num, layer.tp_k_head_num)
        if fuse_qk
        else 0
    )
    if torch.cuda.is_current_stream_capturing():
        if not op.is_loaded(op.module_key(reqs, layer.sliding_window_size, ng)):
            return "kernel not loaded before graph capture"
        if qk and not op.is_loaded(
            op.module_key(reqs, layer.sliding_window_size, ng, qk=qk, rope=True)
        ):
            return "kernel not loaded before graph capture"
    else:
        op.preload(
            layer.sliding_window_size,
            _limit("max_reqs", hkv),
            layer.tp_q_head_num,
            hkv,
            qkv.device,
            ng,
        )
        if _state["qk"]:
            op.preload(
                layer.sliding_window_size,
                min(_limit("max_reqs", hkv), _limit("qk_max_reqs", hkv)),
                layer.tp_q_head_num,
                layer.tp_k_head_num,
                qkv.device,
                ng,
                fuse_qk=True,
                rope=True,
            )
    return None


def _swa_path(attn, positions, qkv, forward_batch, backend):
    """MuseSpark1xAttention.gated_attention with the attention call replaced (same order of operations)."""
    from .ops import swa_verify_attn as op

    layer = attn.attn
    q, k, v, o_gate = qkv.split(
        [attn.q_size, attn.kv_size, attn.kv_size, attn.q_size], dim=-1
    )
    fuse_rope = attn.rotary_emb is not None
    kv_store = attn._fused_kv_store(forward_batch)
    if kv_store is None or kv_store[0].dtype != torch.float8_e4m3fn:
        return None  # nothing done yet: caller runs stock
    k_read, v_read = backend.token_to_kv_pool.get_kv_buffer(layer.layer_id)
    if (
        k_read.data_ptr() != kv_store[0].data_ptr()
        or v_read.data_ptr() != kv_store[1].data_ptr()
    ):
        return None  # the stock kernel would read other buffers than the store writes
    m = qkv.shape[0]
    reqs = m // TOKENS
    hq, d = layer.tp_q_head_num, layer.head_dim
    bmm1, bmm2 = backend._get_bmm_scales(layer)
    md = backend.forward_metadata
    eps = _norm_gate_eps(attn, qkv)
    qk_args = _qk_prologue_args(attn, positions, qkv, kv_store) if fuse_rope else None
    if qk_args is not None:
        pos, cos_sin, qk_eps = qk_args
        out = torch.empty(m, hq * d, dtype=torch.bfloat16, device=qkv.device)
        op.swa_verify_attn(
            None,
            kv_store[0],
            kv_store[1],
            backend._get_layer_page_table(layer, forward_batch),
            md.cache_seqlens_int32[:reqs],
            layer.sliding_window_size,
            backend.page_size,
            bmm1,
            bmm2,
            out=out.view(m, hq, d),
            gate=o_gate if eps is not None else None,
            norm_eps=eps or 0.0,
            qkv=qkv,
            positions=pos,
            cos_sin=cos_sin,
            cache_loc=kv_store[2],
            qk_eps=qk_eps,
            num_q_heads=hq,
        )
        bed.count(QK, f"fused_reqs{reqs}")
        if eps is not None:
            bed.count(SWA, f"fused_norm_gate_reqs{reqs}")
            return out
        bed.count(SWA, f"fused_reqs{reqs}")
        return attn._fused_o_norm_gate(out, o_gate)
    if _state["qk"]:
        bed.count(QK, "unfused")
    if fuse_rope:
        q, k = attn._fused_qk_norm(
            qkv, positions, attn.rotary_emb.cos_sin_cache, kv_store=kv_store
        )
    else:
        q, k = attn._fused_qk_norm(qkv, kv_store=kv_store)
    if q.dtype != torch.float8_e4m3fn:
        q = q.to(torch.float8_e4m3fn)  # what forward_extend does for an FP8 cache
    out = torch.empty(m, hq * d, dtype=torch.bfloat16, device=q.device)
    op.swa_verify_attn(
        q.reshape(m, hq, d),
        kv_store[0],
        kv_store[1],
        backend._get_layer_page_table(layer, forward_batch),
        md.cache_seqlens_int32[:reqs],
        layer.sliding_window_size,
        backend.page_size,
        bmm1,
        bmm2,
        out=out.view(m, hq, d),
        gate=o_gate if eps is not None else None,
        norm_eps=eps or 0.0,
    )
    if eps is not None:
        bed.count(SWA, f"fused_norm_gate_reqs{reqs}")
        return out
    bed.count(SWA, f"fused_reqs{reqs}")
    return attn._fused_o_norm_gate(out, o_gate)


def _swat6_reason(attn, qkv, forward_batch, backend):
    """Reason the t6 path does not take this call (None: it does)."""
    from .ops import swa_verify_t5 as op

    layer = attn.attn
    if layer.sliding_window_size is None or layer.sliding_window_size < 0:
        return "full attention layer"
    m = qkv.shape[0]
    b = op.t6_bucket(m)
    if b is None or b not in _state["swa_t6_buckets"]:
        return "bucket not enabled"
    hkv, window = layer.tp_k_head_num, layer.sliding_window_size
    if not op.supported(
        layer.tp_q_head_num, hkv, layer.head_dim, window, backend.page_size
    ):
        return "shape"
    if backend.q_data_type != torch.bfloat16:
        return "output dtype"
    from sglang.srt.environ import envs

    if envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get() is not None:
        return "skip-softmax threshold set"
    bmm1, bmm2 = backend._get_bmm_scales(layer)
    if not isinstance(bmm1, float) or not isinstance(bmm2, float):
        return "tensor scales"
    dev = qkv.device
    key = (
        dev.index if dev.index is not None else torch.cuda.current_device(),
        hkv,
        window,
    )
    if key not in _t6_ready:
        if torch.cuda.is_current_stream_capturing():
            return "kernel not loaded before graph capture"
        if key[0] in _t6_captured:
            return "late key after capture"  # preparing it could reallocate a workspace a captured graph still uses
        max_reqs = max(_state["swa_t6_buckets"]) // TOKENS
        op.prepare(dev, max_reqs, hkv, window)
        for r in range(
            1, max_reqs + 1
        ):  # the per-request-count split choice and its kernel variant, before capture
            s6 = op.splits_for6(r, hkv, dev, use_pdl=True)
            cl = 1 < s6 <= 3 and os.environ.get("MUSE_VR_SWA_T6_CLUSTER", "0") == "1"
            op._module6(True, cl, cl and s6 == 3)
        _t6_ready.add(key)
    return None


def _swat6_path(attn, positions, qkv, forward_batch, backend):
    """MuseSpark1xAttention.gated_attention with the attention call replaced by the t6 kernel (same order of operations)."""
    from .ops import swa_verify_t5 as op

    layer = attn.attn
    q, k, v, o_gate = qkv.split(
        [attn.q_size, attn.kv_size, attn.kv_size, attn.q_size], dim=-1
    )
    kv_store = attn._fused_kv_store(forward_batch)
    if kv_store is None or kv_store[0].dtype != torch.float8_e4m3fn:
        return None  # nothing done yet: caller runs the next path
    k_read, v_read = backend.token_to_kv_pool.get_kv_buffer(layer.layer_id)
    if (
        k_read.data_ptr() != kv_store[0].data_ptr()
        or v_read.data_ptr() != kv_store[1].data_ptr()
    ):
        return None  # the stock kernel would read other buffers than the store writes
    m = qkv.shape[0]
    reqs = m // TOKENS
    hq, d = layer.tp_q_head_num, layer.head_dim
    bmm1, bmm2 = backend._get_bmm_scales(layer)
    md = backend.forward_metadata
    if attn.rotary_emb is not None:
        q, k = attn._fused_qk_norm(
            qkv, positions, attn.rotary_emb.cos_sin_cache, kv_store=kv_store
        )
    else:
        q, k = attn._fused_qk_norm(qkv, kv_store=kv_store)
    if q.dtype != torch.float8_e4m3fn:
        q = q.to(torch.float8_e4m3fn)  # what forward_extend does for an FP8 cache
    out = torch.empty(m, hq * d, dtype=torch.bfloat16, device=q.device)
    if torch.cuda.is_current_stream_capturing():
        _t6_captured.add(
            q.device.index
            if q.device.index is not None
            else torch.cuda.current_device()
        )
    op.swa_verify_attn_t5(
        q.reshape(m, hq, d),
        kv_store[0],
        kv_store[1],
        backend._get_layer_page_table(layer, forward_batch),
        md.cache_seqlens_int32[:reqs],
        layer.sliding_window_size,
        backend.page_size,
        bmm1,
        bmm2,
        out=out.view(m, hq, d),
        splits=None,
        use_pdl=True,
        prefetch=None,
    )
    bed.count(SWAT6, f"t6:M{m}:b{op.t6_bucket(m)}")
    return attn._fused_o_norm_gate(out, o_gate)


def _swat_reason(attn, qkv, forward_batch, backend):
    """Reason the SWA-T path does not take this call (None: it does)."""
    from .ops import swa_t as op

    layer = attn.attn
    if layer.sliding_window_size is None or layer.sliding_window_size < 0:
        return "full attention layer"
    reqs = qkv.shape[0] // TOKENS
    hkv = layer.tp_k_head_num
    lo_r, hi_r = (
        (_state["swa_t_min"], _state["swa_t_max"])
        if hkv == 2
        else (_state["swa_t_min4"], _state["swa_t_max4"])
    )
    if reqs < lo_r:
        return "requests below min"
    if reqs > hi_r:
        return "requests above max"
    if not op.supported(layer.tp_q_head_num, hkv, layer.head_dim, backend.page_size):
        return "shape"
    if backend.q_data_type != torch.bfloat16:
        return "output dtype"
    from sglang.srt.environ import envs

    if envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get() is not None:
        return "skip-softmax threshold set"
    bmm1, bmm2 = backend._get_bmm_scales(layer)
    if not isinstance(bmm1, float) or not isinstance(bmm2, float):
        return "tensor scales"
    gate = _swat_gate_key(attn, qkv)
    if torch.cuda.is_current_stream_capturing():
        if not op.is_loaded(reqs, qkv.device, gate, True, hkv):
            return "kernel not loaded before graph capture"
    else:
        # the 16-byte norm-gate form serves the larger row counts of the same request window: build both variants
        for g in (gate, 1, 2) if gate and _state["ng16"] else (gate,):
            op.preload(
                hi_r,
                qkv.device,
                g,
                True,
                min_reqs=lo_r,
                qk=_state["swa_t_qk"],
                kv_heads=hkv,
            )
        if not op.kc_for(reqs, qkv.device, gate, True, hkv):
            return "no cluster size fits"
    return None


def _swat_gate_key(attn, qkv) -> int:
    """SWA-T's norm + gate module key for this call: 0 none, 1 the 8-byte norm_gate form, 2 the 16-byte form."""
    ng = _norm_gate_cfg(attn, qkv, _state["norm_gate"], _state["ng16"])
    return 0 if ng is None else (2 if ng[1] == 16 else 1)


def _swat_qk_args(attn, positions, qkv, kv_store):
    """(positions, cos_sin, eps) when SWA-T takes this call with its QK prologue (MUSE_VR_SWA_T_QK=1, RoPE layer, a column
    class it covers, the module built before graph capture), else None."""
    if not _state["swa_t_qk"] or attn.rotary_emb is None or kv_store is None:
        return None
    from .ops import swa_t as op

    args = _qk_rope_args(attn, positions, qkv, kv_store)
    if (
        args is None
        or op.qk_class(qkv.shape[0], attn.attn.tp_q_head_num, attn.attn.tp_k_head_num)
        is None
    ):
        return None
    gate = _swat_gate_key(attn, qkv)
    if torch.cuda.is_current_stream_capturing() and not op.is_loaded_qk(
        qkv.shape[0] // TOKENS, qkv.device, gate, True, attn.attn.tp_k_head_num
    ):
        return None
    return args


def _swat_path(attn, positions, qkv, forward_batch, backend):
    """MuseSpark1xAttention.gated_attention with the attention call replaced by SWA-T (same order of operations as the
    unfused sliding-window path: QK-norm call with the K/V store, attention,
    output norm + gate)."""
    from .ops import swa_t as op

    layer = attn.attn
    q, k, v, o_gate = qkv.split(
        [attn.q_size, attn.kv_size, attn.kv_size, attn.q_size], dim=-1
    )
    fuse_rope = attn.rotary_emb is not None
    kv_store = _kv_store_ok(attn, forward_batch, backend, True)
    if kv_store is None:
        return None  # nothing done yet: the next path or stock runs
    m = qkv.shape[0]
    reqs = m // TOKENS
    hq, d = layer.tp_q_head_num, layer.head_dim
    bmm1, bmm2 = backend._get_bmm_scales(layer)
    md = backend.forward_metadata
    ng = _norm_gate_cfg(attn, qkv, _state["norm_gate"], _state["ng16"])
    eps, ng_vec = (ng[0], ng[1]) if ng is not None else (None, 8)
    qk_args = _swat_qk_args(attn, positions, qkv, kv_store)
    if qk_args is not None:
        # SWA-T with the QK prologue: the kernel does the fused QK-norm call's work (q, this step's K / V rows)
        pos, cos_sin, qk_eps = qk_args
        out = torch.empty(m, hq * d, dtype=torch.bfloat16, device=qkv.device)
        res = op.swa_t_verify_attn_qk(
            qkv,
            kv_store[0],
            kv_store[1],
            backend._get_layer_page_table(layer, forward_batch),
            md.cache_seqlens_int32[:reqs],
            layer.sliding_window_size,
            bmm1,
            bmm2,
            pos,
            cos_sin,
            kv_store[2],
            qk_eps,
            out=out.view(m, hq, d),
            use_pdl=True,
            gate=o_gate if eps is not None else None,
            norm_eps=eps or 0.0,
            ng_vec=ng_vec,
        )
        if (
            res is None
        ):  # cannot happen after _swat_qk_args (class and cluster size known); nothing was stored
            raise RuntimeError(
                "swa_t: no QK-prologue variant for a call its checks admitted"
            )
        if eps is not None:
            bed.count(
                SWAT, f"qk_fused_norm_gate{'' if ng_vec == 8 else ng_vec}_reqs{reqs}"
            )
            return out
        bed.count(SWAT, f"qk_reqs{reqs}")
        return attn._fused_o_norm_gate(out, o_gate)
    if fuse_rope:
        q, k = attn._fused_qk_norm(
            qkv, positions, attn.rotary_emb.cos_sin_cache, kv_store=kv_store
        )
    else:
        q, k = attn._fused_qk_norm(qkv, kv_store=kv_store)
    if q.dtype != torch.float8_e4m3fn:
        q = q.to(torch.float8_e4m3fn)  # what forward_extend does for an FP8 cache
    out = torch.empty(m, hq * d, dtype=torch.bfloat16, device=q.device)
    res = op.swa_t_verify_attn(
        q.reshape(m, hq, d),
        kv_store[0],
        kv_store[1],
        backend._get_layer_page_table(layer, forward_batch),
        md.cache_seqlens_int32[:reqs],
        layer.sliding_window_size,
        bmm1,
        bmm2,
        out=out.view(m, hq, d),
        use_pdl=True,
        gate=o_gate if eps is not None else None,
        norm_eps=eps or 0.0,
        ng_vec=ng_vec,
    )
    if (
        res is None
    ):  # cannot happen after _swat_reason (kC known); kept as a guard - the QK-norm call already ran
        raise RuntimeError(
            "swa_t: no cluster size for a call its reason check admitted"
        )
    if eps is not None:
        bed.count(SWAT, f"fused_norm_gate{'' if ng_vec == 8 else ng_vec}_reqs{reqs}")
        return out
    bed.count(SWAT, f"reqs{reqs}")
    return attn._fused_o_norm_gate(out, o_gate)


def _kv_store_ok(attn, forward_batch, backend, contiguous: bool):
    kv_store = attn._fused_kv_store(forward_batch)
    if kv_store is None or kv_store[0].dtype != torch.float8_e4m3fn:
        return None
    k_read, v_read = backend.token_to_kv_pool.get_kv_buffer(attn.attn.layer_id)
    if (
        k_read.data_ptr() != kv_store[0].data_ptr()
        or v_read.data_ptr() != kv_store[1].data_ptr()
    ):
        return None
    if contiguous and not (kv_store[0].is_contiguous() and kv_store[1].is_contiguous()):
        return None
    return kv_store


def _fullv_reason(attn, qkv, forward_batch, backend):
    from .ops import full_verify_attn as op

    layer = attn.attn
    if layer.sliding_window_size is not None and layer.sliding_window_size >= 0:
        return "sliding window layer"
    reqs = qkv.shape[0] // TOKENS
    max_reqs = _limit("fullv_max_reqs", layer.tp_k_head_num)
    if reqs > max_reqs:
        return "requests above max"
    if not op.supported(
        layer.tp_q_head_num, layer.tp_k_head_num, layer.head_dim, backend.page_size
    ):
        return "shape"
    if backend.q_data_type != torch.bfloat16:
        return "output dtype"
    from sglang.srt.environ import envs

    if envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get() is not None:
        return "skip-softmax threshold set"
    bmm1, bmm2 = backend._get_bmm_scales(layer)
    if not isinstance(bmm1, float) or not isinstance(bmm2, float):
        return "tensor scales"
    ng = _norm_gate_eps(attn, qkv) is not None
    qk_reqs = max_reqs if _state["fullv_qk"] else 0
    key = (
        qkv.device.type,
        qkv.device.index,
        max_reqs,
        layer.tp_q_head_num,
        layer.tp_k_head_num,
        ng,
        _state["fullv_pdl"],
        qk_reqs,
    )
    if torch.cuda.is_current_stream_capturing():
        if key not in op._preloaded:
            return "kernel not loaded before graph capture"
    else:
        op.preload(
            max_reqs,
            layer.tp_q_head_num,
            layer.tp_k_head_num,
            qkv.device,
            ng,
            _state["fullv_pdl"],
            qk_max_requests=qk_reqs,
        )
    return None


def _fullv_path(attn, positions, qkv, forward_batch, backend):
    """MuseSpark1xAttention.gated_attention with the attention call replaced (same order of operations)."""
    from .ops import full_verify_attn as op

    layer = attn.attn
    q, k, v, o_gate = qkv.split(
        [attn.q_size, attn.kv_size, attn.kv_size, attn.q_size], dim=-1
    )
    if attn.rotary_emb is not None:
        return None  # the full-attention layers of this model have no RoPE; anything else runs stock
    kv_store = attn._fused_kv_store(forward_batch)
    if kv_store is None or kv_store[0].dtype != torch.float8_e4m3fn:
        return None
    k_read, v_read = backend.token_to_kv_pool.get_kv_buffer(layer.layer_id)
    if (
        k_read.data_ptr() != kv_store[0].data_ptr()
        or v_read.data_ptr() != kv_store[1].data_ptr()
    ):
        return None
    if not (kv_store[0].is_contiguous() and kv_store[1].is_contiguous()):
        return None
    qk_eps = _qk_full_prologue_eps(attn, qkv, kv_store)
    if qk_eps is not None:
        m = qkv.shape[0]
        reqs = m // TOKENS
        hq, d = layer.tp_q_head_num, layer.head_dim
        bmm1, bmm2 = backend._get_bmm_scales(layer)
        md = backend.forward_metadata
        eps = _norm_gate_eps(attn, qkv)
        out = torch.empty(m, hq * d, dtype=torch.bfloat16, device=qkv.device)
        op.full_verify_attn(
            None,
            kv_store[0],
            kv_store[1],
            backend._get_layer_page_table(layer, forward_batch),
            md.cache_seqlens_int32[:reqs],
            -1,
            backend.page_size,
            bmm1,
            bmm2,
            out=out.view(m, hq, d),
            use_pdl=_state["fullv_pdl"],
            gate=o_gate if eps is not None else None,
            norm_eps=eps or 0.0,
            qkv=qkv,
            cache_loc=kv_store[2],
            qk_eps=qk_eps,
            num_q_heads=hq,
        )
        if eps is not None:
            bed.count(FULLV, f"fused_qk_norm_gate_reqs{reqs}")
            return out
        bed.count(FULLV, f"fused_qk_reqs{reqs}")
        return attn._fused_o_norm_gate(out, o_gate)
    q, k = attn._fused_qk_norm(qkv, kv_store=kv_store)
    if q.dtype != torch.float8_e4m3fn:
        q = q.to(torch.float8_e4m3fn)  # what forward_extend does for an FP8 cache
    m = q.shape[0]
    reqs = m // TOKENS
    hq, d = layer.tp_q_head_num, layer.head_dim
    bmm1, bmm2 = backend._get_bmm_scales(layer)
    md = backend.forward_metadata
    eps = _norm_gate_eps(attn, qkv)
    out = torch.empty(m, hq * d, dtype=torch.bfloat16, device=q.device)
    op.full_verify_attn(
        q.reshape(m, hq, d),
        kv_store[0],
        kv_store[1],
        backend._get_layer_page_table(layer, forward_batch),
        md.cache_seqlens_int32[:reqs],
        -1,
        backend.page_size,
        bmm1,
        bmm2,
        out=out.view(m, hq, d),
        use_pdl=_state["fullv_pdl"],
        gate=o_gate if eps is not None else None,
        norm_eps=eps or 0.0,
    )
    if eps is not None:
        bed.count(FULLV, f"fused_norm_gate_reqs{reqs}")
        return out
    bed.count(FULLV, f"reqs{reqs}")
    return attn._fused_o_norm_gate(out, o_gate)


def _fullw_skel_report() -> dict:
    """full_attn_wide's skeleton options for the install report (only when set: the default report is unchanged)."""
    from .ops import full_attn_wide as op

    return {"skel": op.SKEL} if op.SKEL else {}


def _fullw_reason(attn, qkv, forward_batch, backend):
    from .ops import full_attn_wide as op

    layer = attn.attn
    if layer.sliding_window_size is not None and layer.sliding_window_size >= 0:
        return "sliding window layer"
    reqs = qkv.shape[0] // TOKENS
    lo = _state["fullw_min"]
    if lo is None:
        lo = op.BREAK_EVEN_REQUESTS.get(layer.tp_k_head_num, 1)
        if _state["fullv"]:
            lo = max(lo, _limit("fullv_max_reqs", layer.tp_k_head_num) + 1)
    if reqs < lo:
        return "requests below min"
    if reqs > op.MAX_REQUESTS:
        return "requests above 256"
    if not op.supported(
        layer.tp_q_head_num, layer.tp_k_head_num, layer.head_dim, backend.page_size
    ):
        return "shape"
    if backend.q_data_type != torch.bfloat16:
        return "output dtype"
    from sglang.srt.environ import envs

    if envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get() is not None:
        return "skip-softmax threshold set"
    bmm1, bmm2 = backend._get_bmm_scales(layer)
    if not isinstance(bmm1, float) or not isinstance(bmm2, float):
        return "tensor scales"
    ng = _norm_gate_cfg(attn, qkv, _state["fullw_ng"], _state["ng16"])
    gate = ng[1] if ng is not None else 0
    key = (
        qkv.device.type,
        qkv.device.index,
        layer.tp_q_head_num,
        layer.tp_k_head_num,
        _state["fullw_pdl"],
    ) + ((gate,) if gate else ())
    if torch.cuda.is_current_stream_capturing():
        if key not in op._preloaded:
            return "kernel not loaded before graph capture"
    else:
        op.preload(
            layer.tp_q_head_num,
            layer.tp_k_head_num,
            qkv.device,
            _state["fullw_pdl"],
            gate=gate,
        )
        if op.SKEL:
            _fullw_skel_selfcheck_counts(op)
    return None


def _fullw_skel_selfcheck_counts(op) -> None:
    """One engagement count per load-time self-check of full_attn_wide's skeleton options (pass / FAIL per key)."""
    seen = _state.setdefault("fullw_skel_checks_counted", set())
    for k, ok in op.skel_selfchecks().items():
        if k not in seen:
            seen.add(k)
            bed.count(
                FULLW,
                f"skel_selfcheck_{'pass' if ok else 'FAIL'}_kv{k[1]}_pdl{int(k[2])}_gate{k[3]}",
            )


def _fullw_path(attn, positions, qkv, forward_batch, backend):
    """MuseSpark1xAttention.gated_attention with the attention call replaced (same order of operations)."""
    from .ops import full_attn_wide as op

    layer = attn.attn
    q, k, v, o_gate = qkv.split(
        [attn.q_size, attn.kv_size, attn.kv_size, attn.q_size], dim=-1
    )
    if attn.rotary_emb is not None:
        return None
    kv_store = attn._fused_kv_store(forward_batch)
    if kv_store is None or kv_store[0].dtype != torch.float8_e4m3fn:
        return None
    k_read, v_read = backend.token_to_kv_pool.get_kv_buffer(layer.layer_id)
    if (
        k_read.data_ptr() != kv_store[0].data_ptr()
        or v_read.data_ptr() != kv_store[1].data_ptr()
    ):
        return None
    if not (kv_store[0].is_contiguous() and kv_store[1].is_contiguous()):
        return None
    q, k = attn._fused_qk_norm(qkv, kv_store=kv_store)
    if q.dtype != torch.float8_e4m3fn:
        q = q.to(torch.float8_e4m3fn)  # what forward_extend does for an FP8 cache
    m = q.shape[0]
    reqs = m // TOKENS
    hq, d = layer.tp_q_head_num, layer.head_dim
    q3 = q.reshape(m, hq, d)
    if (
        q3.stride(2) != 1
        or q3.stride(1) != d
        or (q3.stride(0) * q3.element_size()) % 16
        or q3.data_ptr() % 16
    ):
        q3 = q3.contiguous()
    bmm1, bmm2 = backend._get_bmm_scales(layer)
    md = backend.forward_metadata
    out = torch.empty(m, hq * d, dtype=torch.bfloat16, device=q.device)
    ng = _norm_gate_cfg(attn, qkv, _state["fullw_ng"], _state["ng16"])
    op.full_attn_wide(
        q3,
        kv_store[0],
        kv_store[1],
        backend._get_layer_page_table(layer, forward_batch),
        md.cache_seqlens_int32[:reqs],
        bmm1,
        bmm2,
        out=out.view(m, hq, d),
        use_pdl=_state["fullw_pdl"],
        gate=o_gate if ng is not None else None,
        norm_eps=ng[0] if ng is not None else 0.0,
        ng_vec=ng[1] if ng is not None else 8,
    )
    bed.count(FULLW, f"reqs{reqs}")
    if op.SKEL:
        bed.count(FULLW, f"skel{op.effective_skel(layer.tp_k_head_num)}_reqs{reqs}")
    if ng is not None:
        bed.count(FULLW, f"fused_norm_gate{ng[1]}_reqs{reqs}")
        return out
    return attn._fused_o_norm_gate(out, o_gate)


def _step_aside(name: str, reason: str) -> None:
    """Count a call that runs stock, and log each distinct reason once (loud step-aside)."""
    bed.count(name, "stock:" + reason)
    if (name, reason) not in _warned:
        _warned.add((name, reason))
        logger.warning("muse_vr_plugin: %s steps aside to stock: %s", name, reason)


def _full_reason(attn, qkv):
    layer = attn.attn
    if layer.sliding_window_size is not None and layer.sliding_window_size >= 0:
        return "sliding window layer"
    if qkv.shape[0] != TOKENS:
        return "more than one request"
    return None


@contextlib.contextmanager
def _scaled_sm_count(factor: float):
    import flashinfer.decode as fdec

    if _state["scale_sm"] is None:
        orig = fdec.get_device_sm_count

        def scaled(device, _orig=orig):
            n = _orig(device)
            f = _state.get("active_factor")
            return max(1, int(n * f)) if f else n

        fdec.get_device_sm_count = scaled
        _state["scale_sm"] = scaled
    _state["active_factor"] = factor
    try:
        yield
    finally:
        _state["active_factor"] = None


def _around_gated_attention(original, attn, positions, qkv, forward_batch):
    reason, backend = _common_reason(attn, qkv, forward_batch)
    if reason is None and _state["swa_t6"]:
        r = _swat6_reason(attn, qkv, forward_batch, backend)
        if r is None:
            res = _swat6_path(attn, positions, qkv, forward_batch, backend)
            if res is not None:
                return res
            r = "no fused FP8 KV store / buffer mismatch"
        if r not in ("full attention layer", "bucket not enabled"):
            bed.count(SWAT6, "step aside:" + r)
    if reason is None and _state["swa_t"]:
        r = _swat_reason(attn, qkv, forward_batch, backend)
        if r is None:
            res = _swat_path(attn, positions, qkv, forward_batch, backend)
            if res is not None:
                return res
            r = "no fused FP8 KV store / buffer mismatch"
        if r not in (
            "full attention layer",
            "requests below min",
            "requests above max",
        ):
            bed.count(SWAT, "step aside:" + r)
    if reason is None and _state["swa"]:
        r = _swa_reason(attn, qkv, forward_batch, backend)
        if r is None:
            res = _swa_path(attn, positions, qkv, forward_batch, backend)
            if res is not None:
                return res
            r = "no fused FP8 KV store / buffer mismatch"
        if (
            attn.attn.sliding_window_size is not None
            and attn.attn.sliding_window_size >= 0
        ):
            bed.count(SWA, "stock:" + r)
    if reason is None and _state["fullv"]:
        r = _fullv_reason(attn, qkv, forward_batch, backend)
        if r is None:
            res = _fullv_path(attn, positions, qkv, forward_batch, backend)
            if res is not None:
                return res
            r = "no fused FP8 KV store / buffer mismatch / RoPE"
        if attn.attn.sliding_window_size is None or attn.attn.sliding_window_size < 0:
            bed.count(FULLV, "stock:" + r)
    is_full_layer = (
        attn.attn.sliding_window_size is None or attn.attn.sliding_window_size < 0
    )
    if _state["fullw"] and is_full_layer:
        if reason is None:
            r = _fullw_reason(attn, qkv, forward_batch, backend)
            if r is None:
                res = _fullw_path(attn, positions, qkv, forward_batch, backend)
                if res is not None:
                    return res
                r = "no fused FP8 KV store / buffer mismatch / RoPE"
            if r != "requests below min":
                _step_aside(FULLW, r)
            else:
                bed.count(FULLW, "below min (stock path by design)")
        elif reason != "not target verify":
            _step_aside(FULLW, reason)
    if reason is None and _state["full"]:
        r = _full_reason(attn, qkv)
        if r is None:
            bed.count(FULL, "scaled")
            with _scaled_sm_count(_state["factor"]):
                return original(attn, positions, qkv, forward_batch)
    return original(attn, positions, qkv, forward_batch)

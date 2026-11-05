# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Paged FP8 sliding-window verify attention for mid / large batches: SWA-T, JIT CUDA (``csrc/swa_t_verify_attn.cuh``).

One thread-block cluster of kC CTAs per request: CTA c takes chunk c of the request's window tiles and runs
full_attn_wide's tensor-core tile machinery with its fp32 softmax arithmetic; the chunks' partial row records go to the row
owners of the cluster over distributed shared memory, where they are merged with full_attn_wide's merge arithmetic and
written as bf16 rows (optionally with the fused output norm + sigmoid gate). With use_pdl the kernel reads seq_lens, its
page-table entries and the first K/V tiles of keys older than this step before its dependency wait (late-writer class:
valid in decode graph replays), and lets its dependents launch right after the wait.

kC per request count: the largest kC <= KC_MAX whose clusters all fit on the device at once (runtime occupancy query of
the instantiation), or MUSE_VR_SWA_T_KC ("R:kC,..."). Covers 8 query tokens per request, 8 query heads per KV head,
2 KV heads per rank, head_dim 64, e4m3 Q/K/V, page size 128, bf16 output, a sliding window, at most 256 requests.
Numerics: CT-5 class C3 vs full_attn_wide windowed (same per-tile arithmetic; different piece boundaries).
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import torch

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/jit/utils/__init__.py": ("load_jit", "make_cpp_args"),
}


_CSRC = Path(__file__).resolve().parent.parent / "csrc" / "swa_t_verify_attn.cuh"

TOKENS_PER_REQUEST = 8
HEAD_DIM = 64
Q_PER_KV = 8
PAGE = 128
REC_FLOATS = 68
ROWS = 64
MAX_REQUESTS = 256
STAGES = 3
# full_attn_wide's kDiet with only its bitwise bits (4: one wait for a tile's output loads, 16: batched page-list loads,
# 32: the e4m3 probabilities to tensor memory instead of shared memory); the softmax arithmetic is the DIET 0 one (fp32
# exponentials and row sums)
DIET = 4 | 16 | 32
# kPf bits: 32 = pre-dependency staging (late-writer class), 256 = bulk DSMEM records (opt-in, see _pf), bits 16-17 = the
# QK prologue's column class (swa_t_verify_attn_qk)
PF_STAGE = 32
PF_BULK = 256
PF_QK = {2: 1 << 16, 4: 2 << 16, 8: 3 << 16}
PF_QK_UNFUSED = 1 << 18  # the prologue's unfused arithmetic form
# Arithmetic form of SGLang's fused QK-norm Triton kernel per (Triton release, FP-fusion default) the prologue was validated
# against bit for bit (tests/test_swa_t.py): "fused" = with FP fusion (fma contractions, the mean as div.full), "unfused" =
# without (every product and sum rounded separately, the mean as a product with div.approx(1, 64)). The Triton kernel is
# launched without an explicit enable_fp_fusion, so the release's default decides; any other combination: no prologue.
QK_ARITH = {("3.8.0", True): "fused", ("3.9.0", False): "unfused"}
KC_MAX = 10


def _pf() -> int:
    """kPf of this call: staging always; bulk DSMEM records only with MUSE_VR_SWA_T_BULK=1 (default: st.async records -
    compute-sanitizer memcheck reports the bulk shared::cta -> shared::cluster copies, the st.async form is clean)."""
    pf = PF_STAGE
    if os.environ.get("MUSE_VR_SWA_T_BULK", "0").strip() in ("1", "true", "yes", "on"):
        pf |= PF_BULK
    return pf


# compute-sanitizer builds only: a small mbarrier poll cap (an instrumented wait that cannot complete traps within seconds),
# built as its own module; 0 = the normal cap
SAN_POLLS = int(os.environ.get("MUSE_VR_SWA_T_SAN_POLLS", "0"))


def _gk(gate) -> int:
    """Module key of the norm + gate variant: 0 off, 1 SGLang norm_gate's 8-byte form (True), 2 its 16-byte form."""
    return 0 if not gate else (2 if int(gate) == 2 else 1)


@functools.lru_cache(maxsize=None)
def _module(pairs: int, kc: int, gate: int, use_pdl: bool, pf: int):
    """gate: 0 / False = no output norm + gate; 1 / True = with it (8-byte norm_gate form); 2 = with it (16-byte form)."""
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    tpr = 2 if pairs == 1 else 1
    g = _gk(gate)
    args = make_cpp_args(
        pairs,
        STAGES,
        use_pdl,
        0,
        pf,
        0,
        tpr,
        False,
        True,
        False,
        False,
        False,
        0,
        DIET,
        g != 0,
        False,
        kc,
        16 if g == 2 else 8,
    )
    wrappers = [
        ("swa_t", f"muse_vr::SwaTKernel<{args}>::run"),
        ("swa_t_max_clusters", f"muse_vr::SwaTKernel<{args}>::max_clusters"),
    ]
    if (pf >> 16) & 3:
        wrappers.append(("swa_t_qk", f"muse_vr::SwaTKernel<{args}>::run_qk"))
    extra = [f"-DSWAT_MAX_POLLS={SAN_POLLS}u"] if SAN_POLLS > 0 else []
    return load_jit(
        "muse_vr_swa_t" + (f"_san{SAN_POLLS}" if SAN_POLLS > 0 else ""),
        *args,
        cuda_files=[str(_CSRC)],
        cuda_wrappers=wrappers,
        extra_cuda_cflags=extra or None,
    )


def qk_class(num_tokens: int, num_q_heads: int, num_kv_heads: int) -> int | None:
    """Columns per thread of the QK prologue for this many tokens: the column count per thread of SGLang's fused QK-norm
    Triton kernel, whose block size follows the row count (as ops.swa_verify_attn.qk_class). None: not covered."""
    n_rows = num_tokens * (num_q_heads + 2 * num_kv_heads)
    if n_rows < 1024:
        return 2
    if n_rows < 8192:
        return 4
    if n_rows < 65536:
        return 8
    return None


@functools.lru_cache(maxsize=None)
def _probe_module():
    from sglang.kernels.jit.utils import load_jit

    return load_jit(
        "muse_vr_swa_t_exp2_probe",
        cuda_files=[str(_CSRC)],
        cuda_wrappers=[("exp2_probe", "muse_vr::SwaTExp2Probe::run")],
    )


def exp2_probe(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(ex2.approx.ftz.f32(y), the FMA-pipe polynomial exp2(y)) for a float32 CUDA tensor of even length (tests only)."""
    hw = torch.empty_like(y)
    poly = torch.empty_like(y)
    _probe_module().exp2_probe(y, hw, poly)
    return hw, poly


def supported(
    num_q_heads: int, num_kv_heads: int, head_dim: int, page_size: int
) -> bool:
    """2 KV heads per rank (one cluster per request), or 4 (one cluster per request and KV-head pair: grid z = pair; the
    same 2-head kernel, each cluster reads its pair's 128-byte half of the pool rows)."""
    return (
        head_dim == HEAD_DIM
        and num_kv_heads in (2, 4)
        and num_q_heads == Q_PER_KV * num_kv_heads
        and page_size == PAGE
    )


@functools.lru_cache(maxsize=None)
def _max_clusters(pairs: int, kc: int, device_index: int) -> int:
    """Resident clusters of kC CTAs; asked of one representative instantiation (gated, PDL, staging): the variants share
    block size, shared memory and cluster shape, which is all the occupancy depends on."""
    out = torch.zeros(1, dtype=torch.int32)
    with torch.cuda.device(device_index):
        _module(pairs, kc, True, True, PF_STAGE).swa_t_max_clusters(out)
    return int(out[0])


def _kc_override(reqs: int, kv_heads: int = 2):
    raw = os.environ.get(
        "MUSE_VR_SWA_T_KC" if kv_heads == 2 else "MUSE_VR_SWA_T_KC_KV4", ""
    ).strip()
    if not raw:
        return None
    table = dict(tuple(int(x) for x in e.split(":")) for e in raw.split(",") if e)
    return table.get(reqs)


_kc_cache: dict = {}  # (clusters, gate, use_pdl, device index, pf) -> kC (0: none fits)
_loaded: set = set()  # (kc, gate, use_pdl, pf) modules built


def kc_for(
    reqs: int, device, gate: bool = True, use_pdl: bool = True, kv_heads: int = 2
) -> int:
    """kC for `reqs` requests: MUSE_VR_SWA_T_KC's entry (2 KV heads; MUSE_VR_SWA_T_KC_KV4 at 4), else the largest kC in
    [2, KC_MAX] whose reqs x (kv_heads / 2) clusters are all resident at once (one wave); 0 when even kC = 2 does not fit
    (the caller runs another path)."""
    kc = _kc_override(reqs, kv_heads)
    if kc:
        return kc
    idx = device.index if device.index is not None else torch.cuda.current_device()
    pf = _pf()
    clusters = reqs * (kv_heads // 2)
    key = (clusters, bool(gate), bool(use_pdl), idx, pf)
    if key not in _kc_cache:
        _kc_cache[key] = next(
            (k for k in range(KC_MAX, 1, -1) if _max_clusters(1, k, idx) >= clusters), 0
        )
    return _kc_cache[key]


def _kc_known(reqs: int, device, gate: bool, use_pdl: bool, kv_heads: int) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    return _kc_override(reqs, kv_heads) or _kc_cache.get(
        (reqs * (kv_heads // 2), bool(gate), bool(use_pdl), idx, _pf()), 0
    )


def is_loaded(reqs: int, device, gate: bool, use_pdl: bool, kv_heads: int = 2) -> bool:
    """Whether this call's cluster size is known and its module built (graph capture must not compile)."""
    kc = _kc_known(reqs, device, gate, use_pdl, kv_heads)
    return bool(kc) and (kc, _gk(gate), bool(use_pdl), _pf()) in _loaded


@functools.lru_cache(maxsize=None)
def qk_arith() -> str | None:
    """The prologue arithmetic form for the running Triton, or None when it is not validated."""
    try:
        import triton
        from triton import knobs

        lang = knobs.language
        fusion = bool(getattr(lang, "default_fp_fusion", True)) and not bool(
            getattr(lang, "force_disable_fp_fusion", False)
        )
        return QK_ARITH.get((str(triton.__version__), fusion))
    except Exception:  # noqa: BLE001
        return None


def _pf_qk(reqs: int, kv_heads: int = 2) -> int:
    e = qk_class(reqs * TOKENS_PER_REQUEST, Q_PER_KV * kv_heads, kv_heads)
    arith = qk_arith()
    if not e or arith is None:
        return 0
    return _pf() | PF_QK[e] | (PF_QK_UNFUSED if arith == "unfused" else 0)


def is_loaded_qk(
    reqs: int, device, gate: bool, use_pdl: bool, kv_heads: int = 2
) -> bool:
    kc = _kc_known(reqs, device, gate, use_pdl, kv_heads)
    pf = _pf_qk(reqs, kv_heads)
    return bool(kc) and bool(pf) and (kc, _gk(gate), bool(use_pdl), pf) in _loaded


def preload(
    max_reqs: int,
    device,
    gate: bool = True,
    use_pdl: bool = True,
    min_reqs: int = 1,
    qk: bool = False,
    kv_heads: int = 2,
) -> None:
    """Pick kC and build the module for every request count in [min_reqs, max_reqs] (before graph capture); qk: also the
    QK-prologue module of each count's column class."""
    for r in range(max(1, min_reqs), max_reqs + 1):
        kc = kc_for(r, device, gate, use_pdl, kv_heads)
        if kc:
            _module(1, kc, _gk(gate), bool(use_pdl), _pf())
            _loaded.add((kc, _gk(gate), bool(use_pdl), _pf()))
            pq = _pf_qk(r, kv_heads)
            if qk and pq:
                _module(1, kc, _gk(gate), bool(use_pdl), pq)
                _loaded.add((kc, _gk(gate), bool(use_pdl), pq))


_workspace: dict = {}


def _ws(device):
    key = (
        device.type,
        device.index if device.index is not None else torch.cuda.current_device(),
    )
    ws = _workspace.get(key)
    if ws is None:
        # the kernel keeps its records in shared memory; the workspace only satisfies the launcher's argument checks
        ws = (
            torch.zeros(
                2 * 4 * ROWS * REC_FLOATS, dtype=torch.float32, device=device
            ),  # up to 4 KV heads per rank
            torch.zeros(MAX_REQUESTS, dtype=torch.int32, device=device),
        )
        _workspace[key] = ws
    return ws


def swa_t_verify_attn(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    window_left: int,
    bmm1_scale: float,
    bmm2_scale: float,
    out: torch.Tensor | None = None,
    use_pdl: bool = False,
    gate: torch.Tensor | None = None,
    norm_eps: float = 0.0,
    kc: int | None = None,
    ng_vec: int = 8,
) -> torch.Tensor | None:
    """Sliding-window attention of 8 consecutive query tokens per request over a paged FP8 KV pool.

    :param q: ``[R * 8, 8 * Hkv, 64]`` e4m3 (Hkv 2 or 4), heads and dims contiguous, token stride a multiple of 16 bytes.
    :param k_pool: ``[slots, Hkv * 64]`` e4m3, contiguous (slot = page * 128 + offset); ``v_pool`` the same layout.
    :param page_table: ``[>= R, pages]`` int32, inner dimension contiguous.
    :param seq_lens: ``[R]`` int32, KV length per request including the 8 query tokens.
    :param window_left: token p attends keys [max(0, p - window_left), p].
    :param gate: ``[R * 8, >= 8 * Hkv * 64]`` bf16: the owners also apply the output norm (eps ``norm_eps``, no weight) and the
        sigmoid gate with SGLang's norm_gate arithmetic in its ``ng_vec``-byte form (8: 16 threads per row, 16: 8 threads per
        row) - the form SGLang would pick for this call.
    :param kc: cluster size (default: ``kc_for``).
    :returns: ``out``, or None when no cluster size fits (nothing launched).
    """
    m, hq = q.shape[0], q.shape[1]
    if out is None:
        out = torch.empty(m, hq, HEAD_DIM, dtype=torch.bfloat16, device=q.device)
    reqs = m // TOKENS_PER_REQUEST
    hkv = k_pool.shape[1] // HEAD_DIM
    kc = kc or kc_for(reqs, q.device, gate is not None, use_pdl, hkv)
    if not kc:
        return None
    pf = _pf()
    ws, counters = _ws(q.device)
    gk = 0 if gate is None else (2 if ng_vec == 16 else 1)
    mod = _module(1, kc, gk, bool(use_pdl), pf)
    _loaded.add((kc, gk, bool(use_pdl), pf))
    off = counters[:1]
    mod.swa_t(
        q,
        k_pool,
        v_pool,
        page_table,
        seq_lens,
        out,
        ws,
        counters,
        1,
        float(bmm1_scale),
        float(bmm2_scale),
        counters,
        counters,
        0,
        0,
        0,
        0,
        0,
        int(window_left),
        gate if gate is not None else off,
        float(norm_eps),
    )
    return out


def swa_t_verify_attn_qk(
    qkv: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    window_left: int,
    bmm1_scale: float,
    bmm2_scale: float,
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
    cache_loc: torch.Tensor,
    qk_eps: float,
    out: torch.Tensor | None = None,
    use_pdl: bool = False,
    gate: torch.Tensor | None = None,
    norm_eps: float = 0.0,
    kc: int | None = None,
    ng_vec: int = 8,
) -> torch.Tensor | None:
    """SWA-T with the QK prologue: the work of SGLang's fused QK-norm call (QK RMS norm, RoPE, e4m3 q, this step's K / V
    rows stored into the pools at cache_loc) done inside the kernel, bitwise to the Triton kernel for the call's row count.

    :param qkv: ``[R * 8, >= (8 * Hkv + 2 * Hkv) * 64]`` bf16 QKV projection output (q heads, then k heads, then v heads), inner
        contiguous, 16-byte aligned rows. :param positions: ``[R * 8]`` int64. :param cos_sin: ``[max pos, 64]`` fp32 (32 cos,
        32 sin). :param cache_loc: ``[R * 8]`` int64 pool rows of this step's tokens (0 = not stored). Other parameters as
        swa_t_verify_attn. Returns None (nothing launched, nothing stored) when no cluster size fits or the row count has no
        column class.
    """
    m = qkv.shape[0]
    hq = Q_PER_KV * (k_pool.shape[1] // HEAD_DIM)
    if out is None:
        out = torch.empty(m, hq, HEAD_DIM, dtype=torch.bfloat16, device=qkv.device)
    reqs = m // TOKENS_PER_REQUEST
    hkv = k_pool.shape[1] // HEAD_DIM
    pf = _pf_qk(reqs, hkv)
    kc = kc or kc_for(reqs, qkv.device, gate is not None, use_pdl, hkv)
    if not kc or not pf:
        return None
    ws, counters = _ws(qkv.device)
    gk = 0 if gate is None else (2 if ng_vec == 16 else 1)
    mod = _module(1, kc, gk, bool(use_pdl), pf)
    _loaded.add((kc, gk, bool(use_pdl), pf))
    off = counters[:1]
    mod.swa_t_qk(
        qkv,
        k_pool,
        v_pool,
        page_table,
        seq_lens,
        out,
        ws,
        counters,
        positions,
        cos_sin,
        cache_loc,
        float(bmm1_scale),
        float(bmm2_scale),
        int(window_left),
        gate if gate is not None else off,
        float(norm_eps),
        float(qk_eps),
    )
    return out

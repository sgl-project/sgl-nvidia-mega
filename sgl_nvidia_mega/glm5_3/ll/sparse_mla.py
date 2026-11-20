# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Candidate ``ll.sparse_mla_fd``: the ``sparse_mla`` and ``uv_proj`` steps of the GLM-5.3 /
GLM-5.2 attention on a flash-decoding CUDA kernel over the top-k gathered FP8 KV rows.

The stock step calls TRTLLM-Gen's sparse MLA decode cubin, then ``torch.bmm`` for v_b.
Here each query row is one cluster of CTAs that gathers its (up to) 2048 cache rows with
bulk copies, computes QK and P.V with ``mma.sync`` on exactly unpacked E4M3 and merges
the partials through distributed shared memory (``csrc/attention/glm_sparse_mla_fd_*``):

- **one-tile** (16-CTA clusters, CTA ``s`` owns positions ``[128 s, 128 s + 128)``): rows
  up to one wave of clusters, 7 on GB300 at 8 and 16 heads. At 8 heads a CTA gets padded
  shared memory (one CTA per SM) up to the spread wave.
- **multi-tile** (8 heads, 8-35 rows): clusters of 8 / 6 / 4 CTAs stream several
  128-position tiles each with an online softmax. Built, not preferred at the target
  points (FU-2 deferred). At 16 heads it loses to TRTLLM-Gen and is never built.
- **v_b fold** (8 heads, draft decode, one-tile rows): the kernel also writes the o_proj
  input ``bf16(O) @ W_UV`` (one k-ordered ``mma.sync`` chain, bitwise cuBLAS's BMM) and
  the ``uv_proj`` step returns it. It stages its W_UV slices before the PDL wait.

Semantics against TRTLLM-Gen: the same masks (position ``j < seq_lens[r]``, slots
``0 <= id < num_slots``), FP32 sums of exact E4M3 products, ``P`` rounded to
``e4m3(448 p)``; the KV split and the merge order differ, so the output is within
``TOLERANCES`` of TRTLLM-Gen's, not bitwise (test/kernels/test_glm5_3_ll_sparse_mla.py).

Every decision depends on the profile, the row count, the step mode and the launch caps
agreed over the TP group at build, never on tensor contents, so every rank captures the
same kernels. Built only at ``compile`` (never under capture); a failed build raises on
every rank.

Adapted from TensorRT-LLM (Apache-2.0): sparseMlaFlashDecode.cu (Dongfeng Yu) and
glmLowLatency/w4/sparseMlaDecode.cu (Xiao Wang).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from sgl_nvidia_mega.glm5_3.registry import Candidate

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/jit/utils/__init__.py": ("is_arch_support_pdl", "make_cpp_args"),
    "srt/distributed/parallel_state.py": ("get_tp_group", "cpu_group", "world_size"),
    # the running forward's view (Step.view): its metadata, KV pool and DSA backend
    "srt/kernel_plugins/nvidia/binding.py|srt/megakernels/binding.py": (
        "metadata",
        "kv_pool",
        "dsa",
        "get_key_buffer",
        "use_fused_topk",
        "topk_backend",
        "dcp_size",
    ),
    "srt/layers/attention/dsa_backend.py": (
        "cache_seqlens_int32",
        "dsa_cache_seqlens_int32",
    ),
    "srt/layers/attention/dsa/dsa_topk_backend.py": (
        "is_sgl_kernel",
        "is_flashinfer",
    ),
    "srt/environ.py": ("envs", "SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR"),
}

NAME = "ll.sparse_mla_fd"
STEPS = ("sparse_mla", "uv_proj")

HEAD_DIM_QK = 576  # 512 latent + 64 RoPE, one FP8 byte each
HEAD_DIM_V = 512
V_HEAD_DIM = 256
TOP_K = 2048
NUM_HEADS = (8, 16)
FOLD_HEADS = 8
ONE_TILE_SPLITS = 16
MT_SPLITS = (2, 3, 4, 6, 8)
LOG2E = math.log2(math.e)

# The launch plan, from the 78-layer chain bench of glm52-ll branch 20 (job 4488724, per
# layer of RoPE / KV append / attention / v_b against TRTLLM-Gen as the server's graphs
# call it, 1k and 8k contexts) and branch 47 (job 4522135, the fold), on GB300: the
# row caps are those of one wave of clusters on its 152 SMs, re-checked against the
# device's occupancy at build. 8 heads: one-tile (spread) up to 7 rows; the multi-tile
# mode beats the packed one-tile launch and TRTLLM-Gen from 8 to 35 rows with one
# cluster of each size kept free (MT_HEADROOM), not at 36 or 48. 16 heads: one-tile up
# to 7 rows; the multi-tile mode loses to TRTLLM-Gen at 8k from 8 to 48 rows. The fold
# fills an SM per CTA, so its wave is the spread one (7 rows).
MEASURED_SM = (10, 3)
MEASURED_SMS = 152
ONE_TILE_MAX_ROWS = 7
MT_MAX_ROWS = {8: 35, 16: 0}
MT_HEADROOM = 1
FOLD_MAX_ROWS = 7

# The step modes (registry.MODES). The fold runs in the draft's decode passes only, the
# v_b owner rule of the measured stack (stack-v3: verify / draft extend keep a separate
# v_b, draft decode folds).
VERIFY, DRAFT_DECODE, DRAFT_EXTEND = "target_verify", "decode", "draft_extend"
FOLD_MODES = (DRAFT_DECODE,)

# Against TRTLLM-Gen on the same inputs (test/kernels/test_glm5_3_ll_sparse_mla.py: 8 / 16
# heads, 1-48 rows, 1k / 2048 / 8k contexts, one-tile and multi-tile, FP8 KV of std 1.4):
# the largest |ours - TRTLLM-Gen| of a BF16 output element, measured 0.052 (8 heads) and
# 0.055 (16 heads), two BF16 ulps of the largest outputs. Each case also holds the FP64
# oracle bound nrms(ours) <= 1.25 nrms(TRTLLM-Gen) + 2e-4. The v_b fold is bitwise the
# flash decode followed by the stock BMM.
TOLERANCES = {"ll.sparse_mla_fd.max_abs": 0.0625}

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------- kernel modules


def _check_device(heads: int) -> None:
    if heads not in NUM_HEADS:
        raise ValueError(f"{NAME}: {heads} heads per rank (supports {NUM_HEADS})")
    # 16-CTA clusters, st.async and the e4m3x2 converts: the SM100 family only.
    if torch.cuda.get_device_capability()[0] != 10:
        raise RuntimeError(f"{NAME} requires an SM100-family GPU")


_modules: Dict[Tuple[str, int, bool], Any] = {}


def module(kind: str, heads: int) -> Any:
    """The JIT module of ``kind`` ("one_tile", "fold" or "multi_tile") for ``heads``,
    built on first use (``ll.jit.load`` refuses to build under CUDA-graph capture)."""
    from sglang.kernels.jit.utils import is_arch_support_pdl, make_cpp_args

    from sgl_nvidia_mega.glm5_3.ll import jit

    pdl = bool(is_arch_support_pdl())
    key = (kind, heads, pdl)
    if key in _modules:
        return _modules[key]
    _check_device(heads)
    if kind == "multi_tile":
        args = make_cpp_args(heads, pdl)
        cls = f"GlmLlSparseMlaFdMultiTile<{args}>"
        cuh = "attention/glm_sparse_mla_fd_multi_tile.cuh"
        wrappers = [
            ("run", f"{cls}::run"),
            ("smem_bytes", f"{cls}::smem_bytes"),
            ("max_active_clusters", f"{cls}::max_active_clusters"),
        ]
    elif kind in ("one_tile", "fold"):
        fold = kind == "fold"
        if fold and heads != FOLD_HEADS:
            raise ValueError(f"{NAME}: the v_b fold supports {FOLD_HEADS} heads")
        args = make_cpp_args(heads, pdl, fold)
        cls = f"GlmLlSparseMlaFdOneTile<{args}>"
        cuh = "attention/glm_sparse_mla_fd_one_tile.cuh"
        wrappers = [
            ("run_fold" if fold else "run", f"{cls}::{'run_fold' if fold else 'run'}"),
            ("smem_bytes", f"{cls}::smem_bytes"),
            ("spread_smem_bytes", f"{cls}::spread_smem_bytes"),
            ("max_active_clusters", f"{cls}::max_active_clusters"),
        ]
    else:
        raise ValueError(f"{NAME}: unknown module kind {kind!r}")
    _modules[key] = jit.load(f"sparse_mla_fd_{kind}", cuh, wrappers, marker=tuple(args))
    return _modules[key]


def flash_decode(
    q: torch.Tensor,
    kv_rows: torch.Tensor,
    indices: torch.Tensor,
    seq_lens: torch.Tensor,
    scale_log2: float,
    *,
    out: Optional[torch.Tensor] = None,
    smem_bytes: Optional[int] = None,
    splits: int = ONE_TILE_SPLITS,
    vb_proj: Optional[torch.Tensor] = None,
    out_scale: float = 1.0,
) -> torch.Tensor:
    """Sparse MLA decode over FP8 KV rows selected by a top-k index list.

    Position ``j`` of row ``r`` is used iff ``j < min(seq_lens[r], 2048)`` and
    ``0 <= indices[r, j] < kv_rows.shape[0]``; repeated slots count once per
    occurrence and a row without a usable position outputs zeros.

    Args:
        q: ``[rows, H, 576]`` float8_e4m3fn, contiguous, H in (8, 16).
        kv_rows: ``[num_slots, 576]`` float8_e4m3fn (or uint8) KV rows.
        indices: ``[rows, >= 2048]`` int32 global slots, unit inner stride.
        seq_lens: ``[>= rows]`` int32 causal KV length of each row.
        scale_log2: softmax scale times the dequantization scales times log2(e).
        out: optional output (``[rows, H, 512]`` bfloat16; with ``vb_proj``
            ``[rows, >= 8 * 256]`` bfloat16 with unit inner stride).
        smem_bytes: dynamic shared memory per CTA of the one-tile kernel (default
            its minimum; more spreads the CTAs one per SM).
        splits: CTAs per row: 16 (one-tile) or one of MT_SPLITS (multi-tile).
        vb_proj: ``[8, 256, 512]`` bfloat16 W_UV (``w_vc.transpose(1, 2)``, contiguous):
            fold v_b, ``out[r, h * 256 + v] = bf16(sum_d bf16(O[r, h, d]) * vb_proj[h, v, d])``.
        out_scale: scale of the normalized output (FlashInfer's bmm2_scale).

    Returns:
        ``out``: the attention output, or with ``vb_proj`` the o_proj input.
    """
    rows, heads = q.shape[0], q.shape[1]
    if vb_proj is not None:
        if splits != ONE_TILE_SPLITS:
            raise ValueError(f"{NAME}: the v_b fold needs the one-tile kernel")
        mod = module("fold", heads)
        if out is None:
            out = q.new_empty((rows, heads * V_HEAD_DIM), dtype=torch.bfloat16)
        if smem_bytes is None:
            smem_bytes = int(mod.smem_bytes())
        mod.run_fold(
            out,
            q,
            kv_rows,
            indices,
            seq_lens,
            vb_proj,
            scale_log2,
            out_scale,
            smem_bytes,
        )
        return out
    if out is None:
        out = q.new_empty((rows, heads, HEAD_DIM_V), dtype=torch.bfloat16)
    if splits != ONE_TILE_SPLITS:
        module("multi_tile", heads).run(
            out, q, kv_rows, indices, seq_lens, scale_log2, out_scale, splits
        )
        return out
    mod = module("one_tile", heads)
    if smem_bytes is None:
        smem_bytes = int(mod.smem_bytes())
    mod.run(out, q, kv_rows, indices, seq_lens, scale_log2, out_scale, smem_bytes)
    return out


# ------------------------------------------------------------------- launch caps


@dataclass(frozen=True)
class Caps:
    """Launch geometry of one head count on one device (rows = co-resident clusters)."""

    smem_bytes: int
    spread_smem_bytes: int
    spread_rows: int
    packed_rows: int
    # Per entry of MT_SPLITS; empty without the multi-tile mode.
    mt_rows: Tuple[int, ...] = ()
    # 8 heads with the v_b fold (one CTA per SM); 0 without it.
    fold_rows: int = 0
    fold_smem_bytes: int = 0


def device_caps(heads: int, *, with_mt: bool, with_fold: bool) -> Caps:
    """Build the requested modules and query their co-resident clusters on the
    current device (``cudaOccupancyMaxActiveClusters`` on the real kernels)."""
    one = module("one_tile", heads)
    smem = int(one.smem_bytes())
    spread = int(one.spread_smem_bytes())
    caps = Caps(
        smem_bytes=smem,
        spread_smem_bytes=spread,
        spread_rows=int(one.max_active_clusters(spread)),
        packed_rows=int(one.max_active_clusters(smem)),
    )
    if with_mt:
        mt = module("multi_tile", heads)
        caps = replace(
            caps, mt_rows=tuple(int(mt.max_active_clusters(s)) for s in MT_SPLITS)
        )
    if with_fold:
        fold = module("fold", heads)
        fold_smem = int(fold.smem_bytes())
        caps = replace(
            caps,
            fold_rows=int(fold.max_active_clusters(fold_smem)),
            fold_smem_bytes=fold_smem,
        )
    return caps


def agree_caps(
    caps: Optional[Caps],
    error: Optional[BaseException],
    group,
    *,
    with_mt: bool,
    with_fold: bool,
) -> Caps:
    """The minimum co-resident clusters over ``group`` (a GroupCoordinator, or None
    alone), so GPUs with different floorsweeping launch one plan; raises on every rank
    when the build or the query failed on any rank (``error`` is this rank's).

    One MIN all-reduce of a fixed-length vector on the group's CPU group: every rank
    of the group must reach it (the requests depend on the profile and the env only).
    """
    nmt = len(MT_SPLITS)
    if caps is None:
        vec = [0] * (3 + nmt)
    else:
        mt = list(caps.mt_rows) + [0] * nmt
        vec = [caps.spread_rows, caps.packed_rows, *mt[:nmt], caps.fold_rows]
    if group is not None and group.world_size > 1:
        agreed = torch.tensor(vec, dtype=torch.int64)
        torch.distributed.all_reduce(
            agreed, op=torch.distributed.ReduceOp.MIN, group=group.cpu_group
        )
        vec = [int(v) for v in agreed.tolist()]
    if error is not None:
        raise RuntimeError(f"{NAME}: build failed: {error}") from error
    needed = (
        vec[:2]
        + (vec[2 : 2 + nmt] if with_mt else [])
        + (vec[-1:] if with_fold else [])
    )
    if caps is None or min(needed) <= 0:
        raise RuntimeError(
            f"{NAME}: the build or the occupancy query failed on another TP rank "
            f"(agreed {vec}, local {caps})"
        )
    agreed_caps = replace(
        caps,
        spread_rows=vec[0],
        packed_rows=vec[1],
        mt_rows=tuple(vec[2 : 2 + nmt]) if with_mt else (),
        fold_rows=vec[-1] if with_fold else 0,
    )
    if agreed_caps != caps:
        logger.warning(
            "%s: TP ranks differ in co-resident clusters; using the minimum %s (local %s)",
            NAME,
            agreed_caps,
            caps,
        )
    return agreed_caps


def mt_splits_for_rows(
    rows: int, mt_rows: Sequence[int], headroom: int = MT_HEADROOM
) -> int:
    """The multi-tile cluster size for ``rows``: the most CTAs per row whose clusters
    of every row fit one wave with ``headroom`` clusters of that size still free; 0
    when none does."""
    best = 0
    for splits, resident in zip(MT_SPLITS, mt_rows):
        if rows <= resident - headroom:
            best = max(best, splits)
    return best


@dataclass(frozen=True)
class Plan:
    """The kernel each row count of one head count launches, fixed at build."""

    heads: int
    caps: Caps
    one_tile_rows: int
    spread_rows: int
    # Index rows: the multi-tile cluster size, 0 = not served (len = its cap + 1).
    mt_splits: Tuple[int, ...] = ()
    fold_rows: int = 0

    def splits(self, rows: int) -> int:
        """CTAs per row at ``rows`` (16: one-tile), 0 when not served."""
        if 1 <= rows <= self.one_tile_rows:
            return ONE_TILE_SPLITS
        if rows < len(self.mt_splits):
            return self.mt_splits[rows]
        return 0

    def smem_bytes(self, rows: int) -> int:
        """One-tile shared memory at ``rows``: spread (one CTA per SM) when it fits one wave."""
        return (
            self.caps.spread_smem_bytes
            if rows <= self.spread_rows
            else self.caps.smem_bytes
        )

    def folds(self, rows: int, mode: str) -> bool:
        return mode in FOLD_MODES and 1 <= rows <= self.fold_rows


def make_plan(heads: int, caps: Caps) -> Plan:
    """The launch plan of the agreed ``caps``: one-tile up to its spread wave (one CTA
    per SM; at 16 heads the packed wave is the same), the multi-tile mode above it up to
    MT_MAX_ROWS when built, the fold up to its wave; all within the measured caps."""
    one_tile = min(caps.spread_rows, ONE_TILE_MAX_ROWS)
    mt = ()
    if caps.mt_rows and MT_MAX_ROWS[heads] > 0:
        mt = tuple(
            0 if r <= one_tile else mt_splits_for_rows(r, caps.mt_rows)
            for r in range(MT_MAX_ROWS[heads] + 1)
        )
    return Plan(
        heads=heads,
        caps=caps,
        one_tile_rows=one_tile,
        spread_rows=one_tile,
        mt_splits=mt,
        fold_rows=min(caps.fold_rows, FOLD_MAX_ROWS, one_tile),
    )


# ------------------------------------------------------------------- the candidate


def static_reasons() -> list:
    """Configuration facts the kernel never serves (the same on every rank)."""
    from sglang.srt.environ import envs

    reasons = []
    if envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get() is not None:
        reasons.append("TRTLLM-Gen skip-softmax decode is on")
    return reasons


def covers(profile, rows: int, mode: str) -> Optional[str]:
    """None when ``ll.sparse_mla_fd`` serves ``rows`` rows of ``mode`` under ``profile``,
    else the decline reason (logged once per width)."""
    if tuple(profile.sm) != MEASURED_SM:
        return f"SM {profile.sm[0]}.{profile.sm[1]} (measured on sm_103 only)"
    if profile.sms != MEASURED_SMS:
        return f"{profile.sms} SMs (the row caps are GB300's {MEASURED_SMS})"
    if profile.heads not in NUM_HEADS:
        return f"{profile.heads} heads per rank"
    reasons = static_reasons()
    if reasons:
        return "; ".join(reasons)
    if rows < 1:
        return f"{rows} rows"
    if rows <= ONE_TILE_MAX_ROWS or rows <= MT_MAX_ROWS[profile.heads]:
        return None
    if profile.heads == 16:
        return f"{rows} rows at 16 heads (TRTLLM-Gen is faster above one wave)"
    return f"{rows} rows (TRTLLM-Gen is faster above {MT_MAX_ROWS[8]})"


def _model_layers(model) -> list:
    """The decoder layers whose attention this family serves: the target's, or the
    NextN draft's one layer."""
    inner = model.model
    layers = getattr(inner, "layers", None)
    if layers is None:
        layers = [inner.decoder]
    return list(layers)


def _w_uv(attn) -> torch.Tensor:
    """The layer's W_UV as the fold reads it: ``w_vc.transpose(1, 2)``, a contiguous
    [8, 256, 512] BF16 view (the loader's layout)."""
    w = attn.w_vc.transpose(1, 2)
    if (
        w.dtype != torch.bfloat16
        or tuple(w.shape) != (FOLD_HEADS, V_HEAD_DIM, HEAD_DIM_V)
        or not w.is_contiguous()
        or w.data_ptr() % 16
    ):
        raise RuntimeError(
            f"{NAME}: layer {attn.layer_id} W_UV {tuple(w.shape)} {w.dtype} is not the "
            "contiguous BF16 [8, 256, 512] view the v_b fold reads"
        )
    return w


class SparseMlaFdFamily:
    """Modules, agreed launch plan and step groups of one model (target or draft)."""

    def __init__(self, model, derived, profile) -> None:
        self.profile = profile
        self.heads = int(profile.heads)
        self.layers = _model_layers(model)
        # the NextN draft (one decoder layer) runs the draft-decode passes the fold serves
        self.draft = getattr(model.model, "layers", None) is None
        self.plan: Optional[Plan] = None

    def compile(self, widths: Sequence[int]) -> None:
        """Build the kernels ``widths`` need and agree on the launch plan over the TP
        group. Raises on every rank when any rank's build fails, or when the device cannot
        launch a width ``covers`` promised in one wave."""
        widths = [w for w in widths if covers(self.profile, w, VERIFY) is None]
        with_mt = (
            any(w > ONE_TILE_MAX_ROWS for w in widths) and MT_MAX_ROWS[self.heads] > 0
        )
        with_fold = self.draft and self.heads == FOLD_HEADS
        caps = error = None
        try:
            caps = device_caps(self.heads, with_mt=with_mt, with_fold=with_fold)
            if with_fold:
                for layer in self.layers:
                    _w_uv(layer.self_attn)
        except Exception as e:  # every rank raises after the agreement below
            error = e
        caps = agree_caps(
            caps, error, _tp_group(), with_mt=with_mt, with_fold=with_fold
        )
        plan = make_plan(self.heads, caps)
        missing = [w for w in widths if plan.splits(w) == 0]
        if missing:
            raise RuntimeError(
                f"{NAME}: the device runs {missing} rows in more than one wave "
                f"(plan {plan}); the row caps were measured on GB300"
            )
        self.plan = plan
        logger.info(
            "%s: %d heads, one-tile <= %d rows (spread <= %d), multi-tile %s, v_b fold <= %d rows in %s",
            NAME,
            self.heads,
            plan.one_tile_rows,
            plan.spread_rows,
            f"<= {len(plan.mt_splits) - 1} rows" if plan.mt_splits else "off",
            plan.fold_rows,
            "/".join(FOLD_MODES),
        )

    def step_reasons(self, model, forward_batch, view) -> list:
        """Host-only facts read at the first capture of a width (any reason on any
        rank stops every rank): the KV pool layout and the fused top-k page table;
        ``view`` is the running forward's ForwardView."""
        dsa = view.dsa
        if dsa is None:
            return ["the attention backend is not the DSA backend (no DSA view)"]
        reasons = []
        if dsa.dcp_size != 1:
            reasons.append("decode context parallelism")
        topk = dsa.topk_backend
        if (
            not dsa.use_fused_topk
            or topk is None
            or not (topk.is_sgl_kernel() or topk.is_flashinfer())
        ):
            reasons.append("no fused top-k page table")
        pool = view.kv_pool
        for layer in self.layers:
            kv = pool.get_key_buffer(layer.self_attn.layer_id)
            if (
                kv.shape[-1] != HEAD_DIM_QK
                or kv.element_size() != 1
                or not kv.is_contiguous()
                or kv.data_ptr() % 16
            ):
                reasons.append(
                    f"layer {layer.self_attn.layer_id}: MLA cache {tuple(kv.shape)} {kv.dtype}"
                )
                break
        return reasons

    def provider(self, candidate, rows: int, mode: str) -> SparseMlaFdSteps:
        if self.plan is None:
            raise RuntimeError(f"{NAME}: provider before compile")
        if self.plan.splits(rows) == 0:
            raise RuntimeError(
                f"{NAME}: {rows} rows are not in the launch plan {self.plan}"
            )
        return SparseMlaFdSteps(self, rows, mode)


def _tp_group():
    """The TP group whose ranks build this model together, or None alone."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return None
    from sglang.srt.distributed.parallel_state import get_tp_group

    return get_tp_group()


def _fail(what: str) -> RuntimeError:
    return RuntimeError(f"{NAME}: {what}")


class SparseMlaFdSteps:
    """The ``sparse_mla`` and ``uv_proj`` steps at one row count and mode. With the fold
    ``sparse_mla`` returns the o_proj input and ``uv_proj`` passes it on."""

    def __init__(self, family: SparseMlaFdFamily, rows: int, mode: str) -> None:
        self.family = family
        self.plan = family.plan
        self.rows = rows
        self.mode = mode
        self._folded: Optional[torch.Tensor] = None

    def sparse_mla(self, attn, q_fp8: torch.Tensor, topk_indices, step, decode=None):
        from sgl_nvidia_mega.glm5_3.attention.step import pad_topk_indices

        view = step.view
        layer = attn.attn_mqa
        fb = step.forward_batch
        if decode is None:
            decode = fb.forward_mode.is_decode()
        rows, heads = q_fp8.shape[0], self.plan.heads
        splits = self.plan.splits(rows)
        if splits == 0:
            raise _fail(f"{rows} rows are not in the launch plan")
        if topk_indices is None:
            raise _fail("no top-k cache rows")
        # As the stock step: the fused top-k page table (the top-k indices themselves
        # on the served top-k backends, step_reasons) holds physical cache rows, -1
        # padded; decode reads the full cache lengths, verify / extend the clipped ones.
        ids = pad_topk_indices(topk_indices, rows)
        metadata = view.metadata
        seq_lens = None if decode else metadata.dsa_cache_seqlens_int32
        if seq_lens is None:
            seq_lens = metadata.cache_seqlens_int32
        kv = view.kv_pool.get_key_buffer(layer.layer_id)
        if (
            q_fp8.dtype != torch.float8_e4m3fn
            or tuple(q_fp8.shape) != (rows, heads, HEAD_DIM_QK)
            or not q_fp8.is_contiguous()
            or q_fp8.data_ptr() % 16
            or ids.dtype != torch.int32
            or ids.dim() != 2
            or ids.shape[0] != rows
            or ids.shape[1] < TOP_K
            or ids.stride(1) != 1
            or seq_lens.dtype != torch.int32
            or seq_lens.dim() != 1
            or seq_lens.shape[0] < rows
            or kv.shape[-1] != HEAD_DIM_QK
            or not kv.is_contiguous()
            or kv.data_ptr() % 16
        ):
            raise _fail(
                f"unexpected inputs: q {tuple(q_fp8.shape)} {q_fp8.dtype}, ids "
                f"{tuple(ids.shape)} {ids.dtype}, seq_lens {tuple(seq_lens.shape)} "
                f"{seq_lens.dtype}, cache {tuple(kv.shape)}"
            )
        k_scale = getattr(layer, "k_scale_float", None)
        scale_log2 = (
            (1.0 if k_scale is None else float(k_scale)) * layer.scaling * LOG2E
        )
        kv_rows = kv.view(-1, HEAD_DIM_QK)
        mode = DRAFT_DECODE if decode else self.mode
        if splits == ONE_TILE_SPLITS and self.plan.folds(rows, mode):
            out = flash_decode(
                q_fp8,
                kv_rows,
                ids,
                seq_lens,
                scale_log2,
                vb_proj=_w_uv(attn),
                smem_bytes=self.plan.caps.fold_smem_bytes,
            )
            self._folded = out
            return out
        return flash_decode(
            q_fp8,
            kv_rows,
            ids,
            seq_lens,
            scale_log2,
            splits=splits,
            smem_bytes=self.plan.smem_bytes(rows)
            if splits == ONE_TILE_SPLITS
            else None,
        )

    def uv_proj(self, attn, attn_output: torch.Tensor) -> torch.Tensor:
        from sgl_nvidia_mega.glm5_3.attention.step import AttentionStep

        folded, self._folded = self._folded, None
        if folded is not None:
            if attn_output is not folded:
                raise _fail(
                    "uv_proj got another tensor than the folded sparse_mla output"
                )
            return folded
        # the stock v_b BMM (forward_absorb_core)
        return AttentionStep.uv_proj(self, attn, attn_output)


# the registry's view of this module (registry module docstring): the core steps of the
# target's verify attention and of the NextN draft layer's attention, one family per model
CANDIDATES = (
    Candidate(NAME, "attention", STEPS, NAME, covers),
    Candidate(NAME, "draft_attention", STEPS, NAME, covers),
)
FAMILIES = {NAME: SparseMlaFdFamily}

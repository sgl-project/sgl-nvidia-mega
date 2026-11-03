# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang's python/sglang/srt/layers/attention/trtllm_mha_backend.py (branch
# nvidia/qwen35-agentx-gb300): the SKV2-Q1 arm of TRTLLMHAAttnBackend._run_fixed_q_len_decode,
# _run_q1 and the _q1_enabled check, with _get_bmm_scales, _reshape_paged_kv_cache
# and SGLang's paged_kv_view (srt/mem_cache/layout/paged_view.py).
"""The MTP draft's q1 decode attention on SKV2-Q1 (``q1_kernel``): FP8 paged KV,
trtllm-gen's KV split (one CTA per SM, contiguous 256-token steps) and its rounding
points, the CTAs merged in fp32 by a small Triton kernel. Only the fp32 association
differs from trtllm-gen; SGLang branch nvidia/qwen35-agentx-gb300 serves it at KV >= 16384.

The draft step has no forward batch or attention backend. ``Q1Decode.step`` takes the
step's ``AttnStepMetadata`` and the draft's ``KvPoolBuffers``
(``sglang.srt.kernel_plugins.nvidia.eagle_binding``), forms the paged cache view and the BMM
scales the way the backend does, and runs the plan of that input layout. The query
is the step's FP8 query as the backend hands it to the kernel: the attention front's
FP8 output, with no query scale (q_scale 1.0).
"""

from __future__ import annotations

from functools import cache
from typing import Optional

import torch

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/ops/attention/utils.py": ("canonicalize_stride",),
    "srt/environ.py": ("envs", "SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR"),
    "srt/kernel_plugins/nvidia/eagle_binding.py|srt/megakernels/eagle_binding.py": (
        "AttnStepMetadata",
        "KvPoolBuffers",
        "page_table",
        "cache_seqlens",
        "layer_ids",
        "k_scale",
        "v_scale",
    ),
}

# The SGLang functions whose body this module mirrors, by the digest of their code
# (test/unit/test_contract.py recomputes them on the installed SGLang); SGLang branch
# nvidia/qwen35-agentx-gb300 has the same digests
SGLANG_MIRRORS = {
    "srt/layers/attention/trtllm_mha_backend.py:TRTLLMHAAttnBackend._get_bmm_scales": "90a1387e8f561611",
    "srt/layers/attention/trtllm_mha_backend.py:TRTLLMHAAttnBackend._reshape_paged_kv_cache": "53ae20937769b99a",
    "srt/mem_cache/layout/paged_view.py:paged_kv_view": "e73ce9ff261c4a9b",
    "srt/mem_cache/layout/paged_view.py:paged_view": "5fd2565ec50e2620",
}

PAGE_SIZE = 64
HEAD_DIM = 256
KV_HEADS = 1  # the draft's KV heads on a rank: 2 KV heads over TP8, replicated
QUERY_HEADS = (4, 8)
# the KV length from which the branch serves the draft's q1 decode with SKV2-Q1
# (SGLANG_Q35MF_MKA_Q1_MIN_LENGTH, unset in its TP8 serving configuration)
MIN_LENGTH = 16384
# the merge's head-dim block and warps (Q1SplitPlan.run)
MERGE_BLOCK_D, MERGE_WARPS = 32, 4


def enabled_reason(*, device_index: int, max_context_len: int) -> Optional[str]:
    """None where the branch's backend enables SKV2-Q1 (``_q1_enabled``), else why
    not: an SM10x GPU, and a context of at least one 256-token step per SM, the
    lengths at which trtllm-gen splits the KV one CTA per SM."""
    major, _ = torch.cuda.get_device_capability(device_index)
    if major != 10:
        return f"SKV2-Q1 runs on SM10x, not SM{major}x"
    sms = torch.cuda.get_device_properties(device_index).multi_processor_count
    if -(-max_context_len // 256) < sms:
        return (
            f"a {max_context_len}-token context spans fewer 256-token steps than the "
            f"{sms} SMs trtllm-gen splits over"
        )
    return None


def max_context_len_bound(page_table: torch.Tensor, page_size: int = PAGE_SIZE) -> int:
    """The backend's max_context_len rounded up to whole pages: its graph page tables
    are ``ceil(max_context_len / page_size)`` wide."""
    return int(page_table.shape[1]) * page_size


def paged_view(flat: torch.Tensor, page_size: int) -> torch.Tensor:
    """SGLang's ``paged_view``: ``[slots, ...]`` -> ``[slots // page_size, page_size,
    ...]`` from the buffer's own strides."""
    num_slots = int(flat.shape[0])
    assert num_slots % page_size == 0, (num_slots, page_size)
    slot_stride = flat.stride(0)
    return flat.as_strided(
        (num_slots // page_size, page_size, *flat.shape[1:]),
        (slot_stride * page_size, slot_stride, *flat.stride()[1:]),
    )


def paged_kv_view(
    buf: torch.Tensor, page_size: int, head_num: int, head_dim: int
) -> torch.Tensor:
    """SGLang's ``paged_kv_view``: an MHA pool buffer (slot-major or HND) as
    ``[num_pages, page_size, head_num, head_dim]``."""
    if buf.dim() <= 3:
        return paged_view(buf.view(buf.shape[0], head_num, head_dim), page_size)
    assert tuple(buf.shape[1:]) == (head_num, page_size, head_dim), (
        f"expected an HND buffer [pages, {head_num}, {page_size}, {head_dim}], "
        f"got {tuple(buf.shape)}"
    )
    return buf.permute(0, 2, 1, 3)


def paged_kv_cache(
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    *,
    page_size: int = PAGE_SIZE,
    head_num: int = KV_HEADS,
    head_dim: int = HEAD_DIM,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The backend's ``_reshape_paged_kv_cache`` on one layer's pool buffers:
    ``[pages, heads, page_size, head_dim]`` views, a single head's stride made
    canonical."""
    from sglang.kernels.ops.attention.utils import canonicalize_stride

    k_cache = paged_kv_view(k_buffer, page_size, head_num, head_dim).permute(0, 2, 1, 3)
    v_cache = paged_kv_view(v_buffer, page_size, head_num, head_dim).permute(0, 2, 1, 3)
    if head_num == 1:
        k_cache = canonicalize_stride(k_cache)
        v_cache = canonicalize_stride(v_cache)
    return k_cache, v_cache


def bmm_scales(
    *, k_scale: float, v_scale: float, scaling: float, q_scale: float = 1.0
) -> tuple[float, float]:
    """The backend's ``_get_bmm_scales`` for an FP8 KV cache: (bmm1, bmm2) =
    (q_scale * k_scale * scaling, v_scale), in Python floats as there."""
    return q_scale * k_scale * scaling, v_scale


def _skip_softmax_factor():
    from sglang.srt.environ import envs

    return envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get()


def unsupported_reason(
    query: torch.Tensor,
    kv_cache,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    bmm1_scale,
    bmm2_scale,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    page_size: int = PAGE_SIZE,
    q_len_per_req: int = 1,
    attn_dcp_size: int = 1,
    window_left: int = -1,
    sinks: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    kv_cache_sf=None,
) -> Optional[str]:
    """None when the SKV2-Q1 arm of the branch's ``_run_fixed_q_len_decode`` takes
    the call, else the first condition it fails (the branch then runs trtllm-gen).
    The branch also requires trtllm-gen rather than XQA, which SM10x always is."""
    checks = (
        (attn_dcp_size == 1, f"decode context parallel size {attn_dcp_size}"),
        (page_size == PAGE_SIZE, f"page size {page_size}"),
        (q_len_per_req == 1, f"{q_len_per_req} query rows a request"),
        (seq_lens.numel() == 1, f"{seq_lens.numel()} requests"),
        (
            query.shape == (1, query.shape[-2], HEAD_DIM)
            and query.shape[-2] in QUERY_HEADS,
            f"query shape {tuple(query.shape)}",
        ),
        (query.dtype == torch.float8_e4m3fn, f"query dtype {query.dtype}"),
        (query.is_contiguous(), "a non-contiguous query"),
        (
            isinstance(kv_cache, tuple) and len(kv_cache) == 2,
            "a KV cache that is not a (k, v) pair",
        ),
    )
    for ok, why in checks:
        if not ok:
            return why
    for x in kv_cache:
        if not (
            x.dtype == query.dtype
            and x.ndim == 4
            and x.shape[1:] == (1, PAGE_SIZE, HEAD_DIM)
            and x.is_contiguous()
        ):
            return f"KV cache view {x.dtype} {tuple(x.shape)} {tuple(x.stride())}"
    rest = (
        (out_dtype == torch.bfloat16, f"output dtype {out_dtype}"),
        (
            out is None
            or (
                out.is_contiguous()
                and out.shape == query.shape
                and out.dtype == out_dtype
            ),
            "an output buffer of another shape or layout",
        ),
        (isinstance(bmm1_scale, (int, float)), "a tensor bmm1 scale"),
        # where trtllm-gen applies bmm2_scale relative to its bf16 partials is not
        # pinned down, so the branch takes only 1.0
        (
            isinstance(bmm2_scale, (int, float)) and bmm2_scale == 1.0,
            f"bmm2 scale {bmm2_scale!r}",
        ),
        (window_left < 0, f"a sliding window ({window_left})"),
        (sinks is None and mask is None, "attention sinks or a mask"),
        (kv_cache_sf is None, "KV cache block scales"),
        (
            _skip_softmax_factor() in (None, 0.0),
            "SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR is set",
        ),
    )
    for ok, why in rest:
        if not ok:
            return why
    return None


class Q1Decode:
    """One rank's SKV2-Q1 plans by input layout (the backend's ``_q1_plans``): a
    plan owns the CTA partials and the producer compiled for that layout, so the
    first call of a layout must not be under graph capture (graph warmup runs it
    eagerly, as in the branch). ``pdl``: the producer and the merge launch as
    programmatic dependents, the producer loading the leading K / V tiles this step
    does not write before its wait; ``release``: with ``pdl``, the merge releases its
    own dependent at entry (same arithmetic)."""

    def __init__(self, pdl: bool = True, release: bool = True):
        self._plans = {}
        self.pdl, self.release = pdl, release

    @staticmethod
    def key(query, kv_cache, block_tables, seq_lens) -> tuple:
        return (
            query.shape,
            kv_cache[0].shape,
            block_tables.shape,
            block_tables.stride(),
            seq_lens.shape,
        )

    def plan(self, query, kv_cache, block_tables, seq_lens):
        """The layout's plan, compiled on its first use."""
        from sgl_nvidia_mega.qwen3_5.sm103.draft.q1_kernel import Q1SplitPlan

        key = self.key(query, kv_cache, block_tables, seq_lens)
        plan = self._plans.get(key)
        if plan is None:
            plan = Q1SplitPlan(
                query,
                kv_cache,
                block_tables,
                seq_lens,
                pdl=self.pdl,
                release=self.release,
            )
            self._plans[key] = plan
        return plan

    def __call__(
        self, query, kv_cache, block_tables, seq_lens, *, out, qk_scale, v_scale
    ) -> torch.Tensor:
        """``_run_q1``: the merged output [1, heads, 256] bf16 in ``out``."""
        from sgl_nvidia_mega.qwen3_5.sm103.draft.q1_kernel import skv2_q1_decode

        plan = self.plan(query, kv_cache, block_tables, seq_lens)
        return skv2_q1_decode(
            query,
            kv_cache,
            block_tables,
            seq_lens,
            plan,
            qk_scale=qk_scale,
            v_scale=v_scale,
            out=out,
        )

    def step(
        self,
        query: torch.Tensor,
        kv,
        layer_id: int,
        step,
        *,
        scaling: float,
        out: Optional[torch.Tensor] = None,
        q_scale: float = 1.0,
    ) -> torch.Tensor:
        """One draft step's attention for ``layer_id`` on the draft's pool ``kv``
        (``KvPoolBuffers``) and the step's ``AttnStepMetadata``: [1, heads, 256]
        bf16. Raises where the branch would run trtllm-gen instead."""
        i = kv.layer_ids.index(layer_id)
        kv_cache = paged_kv_cache(kv.k[i], kv.v[i])
        bmm1, bmm2 = bmm_scales(
            k_scale=kv.k_scale[i],
            v_scale=kv.v_scale[i],
            scaling=scaling,
            q_scale=q_scale,
        )
        why = unsupported_reason(
            query,
            kv_cache,
            step.page_table,
            step.cache_seqlens,
            bmm1_scale=bmm1,
            bmm2_scale=bmm2,
            out=out,
        )
        if why is not None:
            raise ValueError(f"SKV2-Q1 cannot serve this step: {why}")
        if out is None:
            out = torch.empty_like(query, dtype=torch.bfloat16)
        return self(
            query,
            kv_cache,
            step.page_table,
            step.cache_seqlens,
            out=out,
            qk_scale=float(bmm1),
            v_scale=float(bmm2),
        )

    def prepare(self, heads: int, kv, layer_id: int, step) -> None:
        """Compile the plan of a step's layout (an FP8 query of ``heads`` heads)
        without running it; the producer is specialized to the layout, so this needs
        the step's own pool and metadata tensors."""
        i = kv.layer_ids.index(layer_id)
        query = torch.empty(
            (1, heads, HEAD_DIM), dtype=torch.float8_e4m3fn, device=kv.k[i].device
        )
        self.plan(
            query,
            paged_kv_cache(kv.k[i], kv.v[i]),
            step.page_table,
            step.cache_seqlens,
        )


@cache
def build(
    device_index: int, heads: int = 4, pdl: bool = True, release: bool = True
) -> None:
    """Import CuTe DSL and Triton and compile the fp32 merge for ``heads`` rows and
    this GPU's SM count (as ``Q1Decode(pdl, release)`` runs it); each layout's
    producer compiles at its plan's first use."""
    import triton

    from sgl_nvidia_mega.qwen3_5.sm103.draft.q1_kernel import _q1_merge

    device = torch.device("cuda", device_index)
    nsplit = torch.cuda.get_device_properties(device_index).multi_processor_count
    partial = torch.zeros(
        (nsplit, heads, HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    maximum = torch.zeros((nsplit, heads), dtype=torch.float32, device=device)
    denom = torch.ones_like(maximum)
    lengths = torch.full((1,), 256, dtype=torch.int32, device=device)
    out = torch.empty((1, heads, HEAD_DIM), dtype=torch.bfloat16, device=device)
    with torch.cuda.device(device):
        _q1_merge[(heads, HEAD_DIM // MERGE_BLOCK_D, 1)](
            partial,
            maximum,
            denom,
            lengths,
            out,
            heads,
            nsplit,
            triton.next_power_of_2(nsplit),
            MERGE_BLOCK_D,
            USE_GDC=pdl,
            RELEASE=pdl and release,
            num_warps=MERGE_WARPS,
            **({"launch_pdl": True} if pdl else {}),
        )
    torch.cuda.synchronize(device)

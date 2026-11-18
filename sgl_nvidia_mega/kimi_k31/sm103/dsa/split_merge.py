# SPDX-License-Identifier: Apache-2.0
# Split-mode implementation preserved from SGLang k31 69950c2797.
# Day-0's merge only returns this rank's keys and its certificate counts
# token-interleaved ownership; context split needs page ownership and the
# complete global selection on every rank. Reuse day-0 allocation only.
"""Exact global top-k merge for DSA decode context parallelism; the algorithm
is described in ``jit/csrc/dsa/dcp_topk_merge.cuh``."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import sglang
import torch
from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

# Names read from SGLang. A contract test checks these before GPU use.
SGLANG_READS = {
    "kernels/jit/utils/__init__.py": (
        "cache_once",
        "load_jit",
        "make_cpp_args",
    ),
}

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_dcp_topk_merge_module(
    num_ranks: int, topk: int, cand_k: int, split: bool = False
) -> Module:
    args = make_cpp_args(num_ranks, topk, cand_k, int(split))
    return load_jit(
        "k31_legacy_split_merge",
        *args,
        cuda_files=[str(Path(__file__).with_suffix(".cuh"))],
        extra_include_paths=[str(Path(sglang.__file__).parent / "kernels/jit/csrc")],
        cuda_wrappers=[("dcp_topk_merge", f"DcpTopKMergeKernel<{args}>::run")],
    )


def dcp_topk_merge_buffer_numel(
    num_ranks: int, max_rows: int, topk: int, cand_k: int
) -> int:
    """int64 entries of the symmetric buffer: two parities of max_rows row slots,
    each holding every rank's candidates and this rank's full local top-k."""
    return 2 * max_rows * (num_ranks * cand_k + topk)


def dcp_topk_merge(
    candidates: torch.Tensor,
    global_lens: torch.Tensor,
    page_table: torch.Tensor,
    *,
    buffer: torch.Tensor,
    multicast_ptr: int,
    peer_ptrs: torch.Tensor,
    counters: torch.Tensor,
    union_scratch: torch.Tensor,
    out_slots: torch.Tensor,
    out_fallback: torch.Tensor,
    rank: int,
    span: int,
    cand_k: int,
    split: bool = False,
) -> None:
    """Merge one batch of rows; every rank of the group must call it with the
    same rows, in the same order.

    ``candidates``: [rows, topk, 2] int32 (fp32 score bits, global position or
    -1), this rank's local top-k. ``global_lens``: [rows] int32 visible keys of
    each row. ``page_table``: [>= rows, pages] int32 widened page ids, row i
    for row i. ``buffer``: the int64 symmetric buffer
    (``dcp_topk_merge_buffer_numel``), ``multicast_ptr`` its multicast address,
    ``peer_ptrs``: [num_ranks] int64 its address on every rank. ``counters``:
    [max_rows] int64 calls per row slot, zero-initialized together with the
    buffer on every rank. ``union_scratch``: [max_rows, num_ranks * topk] int64
    working space of the fallback. ``out_slots``: [rows, topk] int32, this
    rank's selected keys as widened slots ``page_table[p // span] * span + p %
    span``, -1 after them. ``out_fallback``: [rows] int32, 1 where a
    certificate failed.

    ``split`` (KV replicated): rank q scored the 64-token pages p with
    p % num_ranks == q, and every rank writes all selected keys, not only its own.
    """
    num_ranks = peer_ptrs.shape[0]
    module = _jit_dcp_topk_merge_module(num_ranks, candidates.shape[1], cand_k, split)
    module.dcp_topk_merge(
        candidates,
        buffer,
        multicast_ptr,
        peer_ptrs,
        counters,
        global_lens,
        page_table,
        out_slots,
        out_fallback,
        union_scratch,
        rank,
        span,
    )

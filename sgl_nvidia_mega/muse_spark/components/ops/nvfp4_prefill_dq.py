# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Prefill attention over a native NVFP4 KV cache through an FP8 (E4M3) workspace, JIT CUDA + the stock FP8-KV kernel.

``dequant_pages`` converts the batch's NVFP4 pages into the workspace (``csrc/nvfp4_prefill_dq.cuh``: one RNE rounding
of the exact dequantized value to E4M3, global scales left to the caller) and writes the workspace page table.
``context_over_workspace`` then calls a FlashInfer ``trtllm_batch_context_with_kv_cache`` function with the workspace as
an FP8 KV cache: the trtllm-gen FP8-KV context kernel instead of the E2M1-KV one, which re-converts every K/V tile for
each query tile that reads it.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/jit/utils/__init__.py": ("load_jit", "make_cpp_args"),
}


_CSRC = Path(__file__).resolve().parent.parent / "csrc" / "nvfp4_prefill_dq.cuh"

HEAD_DIM = 64


@functools.lru_cache(maxsize=None)
def _module(use_pdl: bool):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    args = make_cpp_args(use_pdl)
    return load_jit(
        "muse_vr_nvfp4_prefill_dq",
        *args,
        cuda_files=[str(_CSRC)],
        cuda_wrappers=[
            ("nvfp4_prefill_dq", f"muse_vr::Nvfp4PrefillDqKernel<{args}>::run")
        ],
    )


class Workspace:
    """FP8 K/V pages for one prefill batch (reused by every layer of a pass) + the workspace page table."""

    def __init__(
        self,
        pages: int,
        num_heads: int,
        page_size: int,
        max_requests: int,
        max_pages_per_request: int,
        device,
    ):
        self.pages, self.num_heads, self.page_size = (
            int(pages),
            int(num_heads),
            int(page_size),
        )
        # NHD token rows, as SGLang's FP8 pool stores them; the kernel reads the HND view
        shape = (self.pages * self.page_size, self.num_heads, HEAD_DIM)
        self.k = torch.zeros(shape, dtype=torch.float8_e4m3fn, device=device)
        self.v = torch.zeros(shape, dtype=torch.float8_e4m3fn, device=device)
        self.table = torch.zeros(
            (int(max_requests), int(max_pages_per_request)),
            dtype=torch.int32,
            device=device,
        )
        # views built once: the HND workspace views the FP8 kernel reads, and the table rows of each batch size
        self.k_hnd, self.v_hnd = self.hnd(self.k), self.hnd(self.v)
        self.k_hnd_u8, self.v_hnd_u8 = (
            self.k_hnd.view(torch.uint8),
            self.v_hnd.view(torch.uint8),
        )
        self._rows = [self.table[:i] for i in range(self.table.shape[0] + 1)]

    def hnd(self, t: torch.Tensor) -> torch.Tensor:
        return t.view(self.pages, self.page_size, self.num_heads, HEAD_DIM).permute(
            0, 2, 1, 3
        )

    def fits(self, total_pages: int, batch: int, max_pages: int) -> bool:
        return (
            total_pages <= self.pages
            and batch <= self.table.shape[0]
            and max_pages <= self.table.shape[1]
        )

    def nbytes(self) -> int:
        return self.k.numel() + self.v.numel() + self.table.numel() * 4


def _u8(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype == torch.uint8 else t.view(torch.uint8)


def pages_of(seq_lens_host, page_size: int) -> tuple[int, int]:
    """(total pages, max pages per request) of a batch from host-side KV lengths."""
    tot, mx = 0, 0
    for s in seq_lens_host:
        n = (int(s) + page_size - 1) // page_size
        tot += n
        mx = max(mx, n)
    return tot, mx


def dequant_pages(
    k4: torch.Tensor,
    v4: torch.Tensor,
    ksf: torch.Tensor,
    vsf: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    ws: Workspace,
    total_pages: int,
    max_pages: int,
    use_pdl: bool = False,
) -> torch.Tensor:
    """Convert the batch's NVFP4 pages (HND views [pages, H, page, 32] / scales [pages, H, page, 4]) into ``ws``.

    Returns the workspace page table rows [B, table width] (entries past a request's pages are not written)."""
    table = ws._rows[seq_lens.shape[0]]
    _module(bool(use_pdl)).nvfp4_prefill_dq(
        _u8(k4),
        _u8(v4),
        _u8(ksf),
        _u8(vsf),
        block_tables,
        seq_lens,
        ws.k_hnd_u8,
        ws.v_hnd_u8,
        table,
        int(max_pages),
        int(total_pages),
    )
    return table


def context_over_workspace(
    context_fn,
    ws: Workspace,
    total_pages: int,
    max_pages: int,
    kwargs: dict,
    use_pdl: bool = False,
):
    """Dequantize the NVFP4 cache named in ``kwargs`` (FlashInfer keyword call) into ``ws`` and run ``context_fn`` over
    the FP8 workspace with otherwise identical arguments (``kv_cache_sf`` dropped)."""
    k4, v4 = kwargs["kv_cache"]
    ksf, vsf = kwargs["kv_cache_sf"]
    table = dequant_pages(
        k4,
        v4,
        ksf,
        vsf,
        kwargs["block_tables"],
        kwargs["seq_lens"],
        ws,
        total_pages,
        max_pages,
        use_pdl=use_pdl,
    )
    kw = dict(kwargs)
    kw["kv_cache"] = (ws.k_hnd, ws.v_hnd)
    kw["kv_cache_sf"] = None
    kw["block_tables"] = table
    return context_fn(**kw)

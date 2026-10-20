# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""wq_b on poisoned memory on one SM100-family GPU (GB300): the q-LoRA RMSNorm + MXFP8
quant feeding wq_b on memory that holds garbage (``fill``: every ``torch.empty`` filled
with the poison byte; ``pool``: the allocator's blocks filled and freed first), the
output bit for bit the same as on zeroed memory, eager and replayed (``POISON`` lines).

The file name sorts first in test/kernels, so plain ``pytest test/kernels`` runs it
before any other kernel test allocates in the process: the ``pool`` cases need the
caching allocator to hand back the blocks they poisoned.
"""

from contextlib import contextmanager
from unittest import mock

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll
from sgl_nvidia_mega.deepseek_v41.sm103.attention.q_norm_quant import rmsnorm_mxfp8


def _ll_gemm_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import cutlass  # noqa: F401
        import flashinfer  # noqa: F401
    except ImportError:
        return False
    return mxfp8_ll.supported()


needs_ll_gemm = pytest.mark.skipif(
    not _ll_gemm_ready(), reason="needs an SM100-family GPU"
)


# -------------------------------------------------------------- wq_b on poisoned memory

N, K = 8192, 1280  # wq_b at TP4: target and draft
POISON_EPS = 1e-6
# 1-8: plain swizzle; 9-48: row tiles (the target's 9-48, the draft's 10-45)
# a 128x4-scale width, the first row-tile width and the widest
ROWS = (6, 9, 48)
# 0xFF: NaN as an E4M3 code and as a UE8M0 scale; 0x7F: NaN code, scale 1.0
POISON = (0xFF, 0x7F)

_empty = torch.empty


@contextmanager
def _filled_empty(byte: int):
    """Every ``torch.empty`` in the block comes back filled with ``byte``."""

    def empty(*args, **kwargs):
        t = _empty(*args, **kwargs)
        t.view(-1).view(torch.uint8).fill_(byte)
        return t

    with mock.patch.object(torch, "empty", empty):
        yield


def _poison_pool(byte: int) -> None:
    """Fill and free small-pool blocks (up to 1 MB each) so the next allocations
    reuse garbage."""
    blocks = [_empty(1 << 19, dtype=torch.uint8, device="cuda") for _ in range(64)]
    for b in blocks:
        b.fill_(byte)
    del blocks


def _poison_inputs(rows):
    g = torch.Generator(device="cuda").manual_seed(rows)
    x = torch.randn(rows, K, device="cuda", generator=g).bfloat16()
    norm = (1 + 0.1 * torch.randn(K, device="cuda", generator=g)).bfloat16()
    w = torch.randn(N, K, device="cuda", generator=g) / K**0.5
    w_q, w_sf = mxfp8_ll.quantize(w.bfloat16())
    return x, norm, w_q.contiguous(), w_sf.reshape(-1).contiguous()


def _step(x, norm, w_q, w_sf, mode, byte):
    """The chain's normalize_q_lora + compute_q_b on poisoned memory."""

    def run():
        _, q, s = rmsnorm_mxfp8(
            x, norm, POISON_EPS, pdl=True, row_tiles=x.shape[0] > mxfp8_ll.MAX_ROWS
        )
        return s, mxfp8_ll.linear(q, s, w_q, w_sf)

    if mode == "pool":
        _poison_pool(byte)
        return run()
    with _filled_empty(byte):
        return run()


@needs_ll_gemm
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("mode", ["fill", "pool"])
@pytest.mark.parametrize("rows", ROWS)
def test_wq_b_output_ignores_unwritten_bytes(rows, mode, graph):
    x, norm, w_q, w_sf = _poison_inputs(rows)
    mxfp8_ll.precompile(w_q, [rows])
    # compile the norm outside any capture
    _step(x, norm, w_q, w_sf, "fill", 0)
    torch.cuda.synchronize()
    want_s, want = (t.clone() for t in _step(x, norm, w_q, w_sf, "fill", 0))
    for byte in POISON:
        if graph:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                s, out = _step(x, norm, w_q, w_sf, mode, byte)
            g.replay()
        else:
            s, out = _step(x, norm, w_q, w_sf, mode, byte)
        torch.cuda.synchronize()
        unwritten = int((s != want_s).sum())
        nans = int(out.isnan().sum())
        same = torch.equal(out, want)
        print(
            f"POISON rows={rows} mode={mode} graph={graph} byte=0x{byte:02X} "
            f"scale_bytes={s.numel()} unwritten={unwritten} nan_out={nans} "
            f"bitwise={'yes' if same else 'NO'}"
        )
        assert same, f"wq_b output changed with poisoned memory 0x{byte:02X}"
        # a pass proves something only if the poison reached the bytes left unwritten
        if rows > mxfp8_ll.MAX_ROWS and byte == 0xFF:
            assert unwritten > 0, "poison did not land in the unwritten scale bytes"

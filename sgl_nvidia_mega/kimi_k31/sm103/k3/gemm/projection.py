# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Split-K projection on the FlashInfer CuTe core vendored in ``splitk.py`` (with an
evict_first weight TMA; policies map padded rows to (M tile, N tile, split K,
stages); attention owns live slot masks)."""

import logging
import os
from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

from ..common.host import architecture
from ..common.host import stream as current_stream
from ..common.host import tensor_maps, to_cute
from .splitk import SplitKGemm, SplitKTactic


def matrix_view(tensor, *, output=False):
    """Keep packed row pitches and FlashInfer's singleton-safe dynamic major axis."""
    rows, cols = tensor.shape
    shape = (cols, rows, 1) if output else (rows, cols, 1)
    stride = (1, tensor.stride(0), 0) if output else (tensor.stride(0), 1, 0)
    result = to_cute(tensor.as_strided(shape, stride))
    return result.mark_layout_dynamic(leading_dim=0 if output else 1)


# Packed attention input projections (TP8: MLA K3.1 [3776, 7168], K3 [3648, 7168]; KDA
# [6288, 7168]): (M tile, N tile, split K, stages) per batch of rows, from GB300 sweeps.
# The fallback is the tactic of up to 8 rows (one N8 tile); a wider unlisted batch would reread
# the weights once per 8 rows, so it is logged once per (kind, batch) until a sweep adds it.
GB300_TACTICS = {
    "mla": {
        1: (64, 8, 1, 12),
        64: (64, 64, 2, 5),
        32: (64, 32, 2, 8),
        16: (64, 16, 2, 10),
    },
    "kda": {64: (64, 64, 1, 6), 32: (128, 32, 2, 5), 16: (128, 16, 2, 6)},
}
GB300_FALLBACK = {"mla": (64, 8, 4, 8), "kda": (128, 8, 2, 6)}
FALLBACK_MAX_ROWS = 8
_FALLBACK_LOGGED: set = set()


def _gb300_tactic(kind, batch):
    tactic = GB300_TACTICS[kind].get(batch)
    if tactic is not None:
        return tactic
    if batch > FALLBACK_MAX_ROWS and (kind, batch) not in _FALLBACK_LOGGED:
        _FALLBACK_LOGGED.add((kind, batch))
        logger.warning(
            "NVIDIA megakernel: no %s projection tactic for B%d; using the %d-row tactic %s, "
            "which rereads the weights per 8 rows",
            kind,
            batch,
            FALLBACK_MAX_ROWS,
            GB300_FALLBACK[kind],
        )
    return GB300_FALLBACK[kind]


# GB300 (SM103) TP16 overrides keyed by (kind, per-rank rows, batch), from a projection sweep
# on GB300 (cold L2), both bitwise vs the default tactic (same split K: only the M tile and
# stages change) and faster: MLA B8 (64, 8, 4, 8) 188 CTAs -> (128, 8, 4, 6) 96 CTAs; KDA B8
# (128, 8, 2, 6) 52 CTAs -> (64, 8, 2, 12) 102 CTAs.
# MLA rows: 3008 where the in-proj carries
# the index-K rows (kimi_k31 spec / dsa_spec pass ``pitch``); 2880 on the K3.1 verify path (verify_spec leaves
# build_resources' default ``packed_mla_pitch``: [qkv-a | gate], index-K is separate).
# 2880 shares 3008's split K (bitwise by construction); its time is projected from 3008, not measured.
SM103_TP16_TACTICS = {
    ("mla", 3008, 8): (128, 8, 4, 6),
    ("mla", 2880, 8): (128, 8, 4, 6),
    ("kda", 3208, 8): (64, 8, 2, 12),
}
# GB300 (SM103) TP8 overrides keyed by (kind, per-rank rows, batch rows) for the Q8 Breq 3/5/6/7 widths, from a TP8
# projection sweep (cold L2, the projection kernel gate): the N8 fallback rereads the weights per 8 rows
# and was 2.2-4.2x slower. Not bitwise vs the fallback (split K differs).
SM103_TP8_TACTICS = {
    ("mla", 3648, 24): (64, 24, 2, 9),
    ("mla", 3648, 40): (64, 40, 2, 8),
    ("mla", 3648, 48): (64, 48, 2, 7),
    ("mla", 3648, 56): (64, 56, 2, 7),
    ("kda", 6288, 24): (64, 24, 1, 10),
    ("kda", 6288, 40): (64, 40, 1, 8),
    ("kda", 6288, 48): (64, 48, 1, 8),
    ("kda", 6288, 56): (64, 64, 1, 7),
}
_SM103_TP16_LOGGED: set = set()
logger = logging.getLogger(__name__)


def projection_tactic(kind, batch, device, tp=8, rows=None):
    """The split-K tactic of the packed ``kind`` ("mla" / "kda") projection at ``batch``
    (``rows``: the per-rank pitch, for the shape-keyed SM103 TP16 entries)."""
    capability = torch.cuda.get_device_capability(device)
    # Q5/R3 uses actual 15 rows in a single N16 tile. Reuse the existing B16
    # policies, without a tactic sweep or padding the target step back to Q8.
    if (
        os.environ.get("K31MK_VERIFY_Q5", "0") == "1"
        and capability == (10, 3)
        and tp == 8
        and batch == 15
        and (kind, rows) in (("mla", 3648), ("kda", 6288))
    ):
        return GB300_TACTICS[kind][16]
    # New Q5 production widths reuse existing instruction tiles. Tensor rows stay
    # at 5 * requests; the Split-K kernel predicates the partial tile's outputs.
    if (
        os.environ.get("K31MK_VERIFY_Q5", "0") == "1"
        and capability == (10, 3)
        and tp == 8
        and batch in (10, 20, 25, 30, 35)
        and (kind, rows) in (("mla", 3648), ("kda", 6288))
    ):
        tile_rows = {10: 16, 20: 24, 25: 32, 30: 32, 35: 40}[batch]
        if tile_rows in (16, 32):
            return GB300_TACTICS[kind][tile_rows]
        return SM103_TP8_TACTICS[(kind, rows, tile_rows)]
    if capability == (10, 3):
        tactic = SM103_TP8_TACTICS.get((kind, rows, batch))
        if tactic is not None:
            return tactic
        tactic = SM103_TP16_TACTICS.get((kind, rows, batch))
        if tactic is not None:
            if (
                kind,
                rows,
                batch,
            ) not in _SM103_TP16_LOGGED:  # once per process, not per layer
                _SM103_TP16_LOGGED.add((kind, rows, batch))
                logger.info(
                    "NVIDIA megakernel: SM103 TP16 %s projection rows=%d B%d tactic %s",
                    kind,
                    rows,
                    batch,
                    tactic,
                )
            return tactic
    return _gb300_tactic(kind, batch)


@cache
def compile_projection(tactic, output_dtype, arch):
    """Code-only cache; weights, descriptors and activation pointers stay private."""

    def fake(leading, dtype=cutlass.BFloat16):
        shape = tuple(cute.sym_int() for _ in range(3))
        stride = (1, cute.sym_int(64), 0) if leading == 0 else (cute.sym_int(64), 1, 0)
        return make_fake_tensor(dtype, shape, stride, assumed_align=16)

    y = fake(0, cutlass.Float32 if output_dtype == torch.float32 else cutlass.BFloat16)
    split = SplitKTactic(*tactic)
    kernel = SplitKGemm(tactic=split)
    args = (fake(1), fake(1), y, y, cutlass.Int64(0), make_fake_stream())
    return cute.experimental.compile(kernel, *args, options=f"--gpu-arch={arch}")


def weight_map(weight, rows):
    """Device TMA map of ``weight`` [N, K] (row pitch kept): box {64, rows}, SW128."""
    types = cuda.CUtensorMapDataType
    dtype = {
        torch.bfloat16: types.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        torch.float16: types.CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
    }[weight.dtype]
    swizzle = cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
    l2 = cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B
    n, k = weight.shape
    pitch = weight.stride(0) * weight.element_size()
    spec = (weight.data_ptr(), dtype, (k, n), (pitch,), (64, rows), swizzle, l2)
    return tensor_maps([spec], weight.device)


class PreparedProjection:
    """Split-K projection with a per-batch tactic table and caller-owned outputs."""

    def __init__(self, weight, policies, *, output_dtype=torch.bfloat16):
        if weight.dim() != 2 or weight.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(
                f"projection weight must be a BF16/FP16 [N, K] matrix, got "
                f"{tuple(weight.shape)} {weight.dtype}"
            )
        pitch = weight.stride(0) * weight.element_size()
        if weight.stride(1) != 1 or pitch % 16 or weight.data_ptr() % 16:
            raise ValueError(
                "projection weight rows need unit stride and a 16 B TMA pitch"
            )
        if output_dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(f"unsupported projection output dtype {output_dtype}")
        self.weight, self.device = weight, weight.device
        self.output_dtype, self.policies = output_dtype, dict(policies)
        self.weight_view = matrix_view(weight)
        self.kernels, self.extra, self.maps = {}, {}, {}

    def prepare(self):
        arch, dtype = architecture(self.device), self.output_dtype
        with torch.accelerator.device_index(self.device.index):
            for batch, tactic in self.policies.items():
                code = compile_projection(tactic, dtype, arch)
                self.kernels[batch] = code
                self.extra[batch] = self.weight_map(tactic[0])
        return self

    def weight_map(self, rows):
        """The kernel's extra argument: A's map for ``rows``-row tiles."""
        if rows not in self.maps:
            self.maps[rows] = weight_map(self.weight, rows)
        return (cutlass.Int64(self.maps[rows].data_ptr()),)

    def run(self, x, out):
        y = matrix_view(out, output=True)
        b, stream = x.shape[0], current_stream(self.device)
        if b not in self.kernels or x.shape[1] != self.weight.shape[1]:
            raise ValueError(
                f"projection input {tuple(x.shape)}: prepared batches "
                f"{sorted(self.kernels)}, K {self.weight.shape[1]}"
            )
        code, extra = self.kernels[b], self.extra[b]
        code(self.weight_view, matrix_view(x), y, y, *extra, stream)
        return out

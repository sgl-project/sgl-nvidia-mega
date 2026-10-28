# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0): python/sglang/kernels/ops/speculative/{mkd_draft_xchg,
# mkd_draft_fc, mkd_draft_head, draft_fc_partial, ordered_partial_sum,
# draft_vocab_argmax}.py (branch nvidia/qwen35-agentx-gb300).
"""The MTP draft step boundary's kernels (``src/mkd_xchg.cu``, ``src/mkd_fc.cu``,
``src/mkd_head.cu``, ``src/draft_fc.cu``) with the launch signatures of the 2.2 ops:

- ``mkd_draft_head``: the one-row draft lm_head on this rank's vocabulary shard, with the
  per-CTA packed maxima the exchange reads;
- ``mkd_xchg``: the shard argmax, the multicast exchange of each rank's candidate and
  hidden-half fc partial, the selection, and with memo rows the next step's fc output;
- ``mkd_fc_sum`` / ``mkd_fc_memo`` / ``mkd_hidden_partial``: the fc of an exchanged input,
  the memo (T-table) rows, and the hidden-slice partial behind a flag;
- ``draft_fc_partial`` + ``FcGather`` + ``ordered_partial_sum``: the row-split fc of up to
  16 tokens (fused norm + slice GEMV, all-gather of the fp32 partials, rank-order sum).

Every launch keeps the 2.2 grid, block, shared memory and PDL setting (the draft head may
add unused shared memory, ``smem_pad``, and run beside a narrower hidden partial,
``ctas``). ``FcGather`` moves bytes only; the arithmetic of the split fc is
``draft_fc_partial`` and ``ordered_partial_sum``, copied unchanged.
"""

from __future__ import annotations

import ctypes
import struct
from functools import cache
from pathlib import Path
from typing import Optional

import torch
import triton
import triton.language as tl

from sgl_nvidia_mega.common import jit

SRC = Path(__file__).parent / "src"
HIDDEN = 4096
WORLDS = (4, 8)
# CTAs per exchange; each owns HIDDEN / CHUNKS columns of the candidate and partial rows
CHUNKS = 16
XCHG_THREADS = 1024
# Rotating epoch slots of the exchange buffer (mkd_xchg::kSlots)
SLOTS = 3
# fc output rows per CTA held in shared memory by mkd_fc_sum (2.2 mkd_draft_fc.MAX_ROWS)
FC_MAX_ROWS = 27
MEMO_BLOCK_ROWS = 8  # embedding rows a mkd_fc_memo CTA normalizes at once (kB)
HEAD_WARPS = 16
# Rows of x per draft_fc_partial launch (2.2 draft_fc_partial.MAX_TOKENS)
MAX_TOKENS = 2
_MAX_SMEM_BYTES = 216 * 1024
# fc_gather: the split fc's widest batch (2.2 _FC_SPLIT_MAX_TOKENS) and its grid
GATHER_MAX_TOKENS = 16
GATHER_CTAS, GATHER_THREADS = 32, 512


def _order_key(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return (~bits & 0xFFFFFFFF) if bits & 0x80000000 else (bits | 0x80000000)


# NaN handling of the reference argmax: the split argmax of draft steps reads NaN as
# -1e30; torch.argmax ranks NaN above every value (2.2 draft_vocab_argmax.py)
NAN_KEY_SPLIT_ARGMAX = _order_key(-1e30)
NAN_KEY_TORCH_ARGMAX = 0xFFFFFFFF


class MkdXchgParams(ctypes.Structure):
    """sglang::MkdXchgParams."""

    _fields_ = [
        ("logits", ctypes.c_void_p),
        ("cand", ctypes.c_void_p),
        ("hot_ids", ctypes.c_void_p),
        ("own_partial", ctypes.c_void_p),
        ("partial_flag", ctypes.c_void_p),
        ("scan_part", ctypes.c_void_p),
        ("head_part", ctypes.c_void_p),
        ("head_parts", ctypes.c_int32),
        ("mc", ctypes.c_void_p),
        ("uc", ctypes.c_void_p),
        ("epoch", ctypes.c_void_p),
        ("done", ctypes.c_void_p),
        ("out_cand", ctypes.c_void_p),
        ("out_partials", ctypes.c_void_p),
        ("fc_out", ctypes.c_void_p),
        ("token_only", ctypes.c_int32),
        ("topk_index", ctypes.c_void_p),
        ("topk_p", ctypes.c_void_p),
        ("draft_tokens", ctypes.c_void_p),
        ("positions", ctypes.c_void_p),
        ("mrope", ctypes.c_void_p),
        ("vocab", ctypes.c_int32),
        ("nan_key", ctypes.c_uint32),
        ("rank", ctypes.c_int32),
        ("draft_column", ctypes.c_int32),
        ("mrope_stride", ctypes.c_int32),
        ("trace", ctypes.c_void_p),
        ("trace_slots", ctypes.c_int32),
        ("pf", ctypes.c_void_p),
        ("pf_bytes", ctypes.c_int64),
    ]


class MkdFcSumParams(ctypes.Structure):
    """sglang::MkdFcSumParams."""

    _fields_ = [
        (name, ctypes.c_void_p)
        for name in ("emb", "memo", "hidden_partials", "fc_w", "norm_w", "out")
    ] + [("eps", ctypes.c_float)]


class MkdFcMemoParams(ctypes.Structure):
    """sglang::MkdFcMemoParams."""

    _fields_ = [
        (name, ctypes.c_void_p) for name in ("emb", "fc_w", "norm_w", "memo")
    ] + [
        ("eps", ctypes.c_float),
        ("n", ctypes.c_int32),
    ]


class MkdHiddenPartialParams(ctypes.Structure):
    """sglang::MkdHiddenPartialParams."""

    _fields_ = [
        (name, ctypes.c_void_p)
        for name in ("h", "w", "norm_w", "out", "epoch", "done", "flag")
    ] + [("eps", ctypes.c_float), ("col", ctypes.c_int32)]


class MkdHeadParams(ctypes.Structure):
    """sglang::MkdHeadParams."""

    _fields_ = [(name, ctypes.c_void_p) for name in ("x", "w", "logits", "part")] + [
        ("n", ctypes.c_int32),
        ("reverse", ctypes.c_int32),
    ]


class DraftFcPartialParams(ctypes.Structure):
    """sglang::DraftFcPartialParams."""

    _fields_ = [(name, ctypes.c_void_p) for name in ("x", "norm_w", "w", "out")] + [
        ("eps", ctypes.c_float),
        ("col_start", ctypes.c_int32),
        ("rows", ctypes.c_int32),
    ]


class FcGatherParams(ctypes.Structure):
    """sglang::FcGatherParams."""

    _fields_ = [
        (name, ctypes.c_void_p)
        for name in ("partial", "out", "mc", "uc", "epoch", "done")
    ] + [("rank", ctypes.c_int32), ("m", ctypes.c_int32)]


def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _compile(device_index: int, name: str, defines: tuple[str, ...]) -> bytes:
    major, minor = torch.cuda.get_device_capability(device_index)
    return jit.compile_source(
        (SRC / f"{name}.cu").read_text(),
        "q35_" + "_".join([name, *(d.split("=")[-1] for d in defines)]),
        f"sm_{major}{minor}a",
        ("-std=c++17", "-default-device", *(f"-D{d}" for d in defines)),
        headers={"mkd_common.cuh": (SRC / "mkd_common.cuh").read_text()},
    )


@cache
def _xchg_kernels(device_index: int, world: int) -> dict[str, jit.Kernel]:
    cubin = _compile(device_index, "mkd_xchg", (f"MKD_WORLD={world}",))
    return {
        name: jit.load(cubin, name, (XCHG_THREADS, 1, 1), 0)
        for name in ("mkd_xchg_memo", "mkd_xchg_emb")
    }


@cache
def _fc_kernels(device_index: int, world: int) -> dict[str, jit.Kernel]:
    cubin = _compile(
        device_index, "mkd_fc", (f"MKD_WORLD={world}", f"MKD_FC_MAX_ROWS={FC_MAX_ROWS}")
    )
    return {
        "mkd_fc_sum_memo": jit.load(cubin, "mkd_fc_sum_memo", (256, 1, 1), 0),
        "mkd_fc_sum_emb": jit.load(
            cubin, "mkd_fc_sum_emb", (256, 1, 1), FC_MAX_ROWS * HIDDEN * 2
        ),
        "mkd_fc_memo": jit.load(
            cubin, "mkd_fc_memo", (256, 1, 1), MEMO_BLOCK_ROWS * HIDDEN * 2
        ),
        "mkd_hidden_partial": jit.load(cubin, "mkd_hidden_partial", (256, 1, 1), 0),
    }


@cache
def _head_kernel(device_index: int, smem_pad: int = 0) -> jit.Kernel:
    cubin = _compile(device_index, "mkd_head", ())
    return jit.load(cubin, "mkd_head", (HEAD_WARPS * 32, 1, 1), smem_pad)


@cache
def _draft_fc_kernels(device_index: int, world: int, max_rows: int) -> dict:
    cubin = _compile(
        device_index,
        "draft_fc",
        (f"MKD_WORLD={world}", f"DRAFT_FC_MAX_ROWS={max_rows}"),
    )
    width = 2 * HIDDEN // world
    smem = max_rows * width * 2
    kernels = {
        f"draft_fc_partial_m{m}": jit.load(
            cubin, f"draft_fc_partial_m{m}", (256, 1, 1), smem
        )
        for m in range(1, MAX_TOKENS + 1)
    }
    kernels["fc_gather"] = jit.load(cubin, "fc_gather", (GATHER_THREADS, 1, 1), 0)
    return kernels


def build(device_index: int, world: int) -> None:
    """Compile every kernel of the draft step boundary at TP ``world`` (no compile
    inside a capture)."""
    if world not in WORLDS:
        raise ValueError(f"the draft step boundary serves TP {WORLDS}, not {world}")
    _xchg_kernels(device_index, world)
    _fc_kernels(device_index, world)
    _head_kernel(device_index)
    rows = draft_fc_partial_rows(HIDDEN, 2 * HIDDEN // world, _sm_count(device_index))
    if rows is not None:
        _draft_fc_kernels(device_index, world, rows)


def _launch(kernel: jit.Kernel, grid: int, params, device, pdl: bool) -> None:
    """``jit.launch`` on the current stream; ``pdl`` False launches without programmatic
    stream serialization, as 2.2 launches its kernels that never wait on a primary."""
    stream = torch.cuda.current_stream(device).cuda_stream
    if pdl:
        jit.launch(kernel, (grid, 1, 1), [params], stream)
        return
    cu = jit.driver()
    pointer_array = (ctypes.c_void_p * 1)(ctypes.addressof(params))
    config = cu.CUlaunchConfig()
    config.gridDimX, config.gridDimY, config.gridDimZ = grid, 1, 1
    config.blockDimX, config.blockDimY, config.blockDimZ = kernel.block
    config.sharedMemBytes = kernel.shared_memory_bytes
    config.hStream = cu.CUstream(stream)
    config.attrs = []
    config.numAttrs = 0
    jit.check(
        cu.cuLaunchKernelEx(
            config, kernel.function, ctypes.addressof(pointer_array), 0
        ),
        "cuLaunchKernelEx",
    )


def _check(t: torch.Tensor, what: str, shape, dtype, aligned: bool = False) -> None:
    ok = (
        tuple(t.shape) == tuple(shape)
        and t.dtype == dtype
        and t.is_cuda
        and t.is_contiguous()
        and (not aligned or t.data_ptr() % 16 == 0)
    )
    if not ok:
        raise ValueError(
            f"{what}: {t.dtype} {tuple(t.shape)} strides {tuple(t.stride())} "
            f"(want {dtype} {tuple(shape)}{', 16-byte aligned' if aligned else ''})"
        )


def _embed_slices(world: int) -> int:
    """Slices of the fc input's embedding half (kEmbed): one per rank of the first half."""
    return HIDDEN // (2 * HIDDEN // world)


# ---------------------------------------------------------------- the exchange


def mkd_xchg_words(hidden: int, world: int, memo: bool) -> int:
    """64-bit words of the symmetric exchange buffer (mirrors MkdXchg::kWords)."""
    cols = hidden // CHUNKS
    rec = 2 + (cols if memo else cols // 2) + cols
    return SLOTS * world * CHUNKS * rec


def mkd_xchg(
    logits: torch.Tensor,
    cand: torch.Tensor,
    hot_ids: torch.Tensor,
    own_partial: torch.Tensor,
    epoch_done: torch.Tensor,
    out_cand: torch.Tensor,
    out_partials: torch.Tensor,
    topk_index: torch.Tensor,
    topk_p: torch.Tensor,
    draft_tokens: Optional[torch.Tensor],
    positions: Optional[torch.Tensor],
    mrope_positions: Optional[torch.Tensor],
    mc_ptr: int,
    uc_ptr: int,
    rank: int,
    world: int,
    draft_column: int,
    nan_key: int,
    trace: Optional[torch.Tensor] = None,
    prefetch: Optional[torch.Tensor] = None,
    prefetch_bytes: int = 0,
    fc_out: Optional[torch.Tensor] = None,
    token_only: bool = False,
    partial_flag: Optional[torch.Tensor] = None,
    scan_part: Optional[torch.Tensor] = None,
    head_part: Optional[torch.Tensor] = None,
) -> None:
    """Shard argmax + multicast exchange + selection for one draft token (2.2's
    ``mkd_xchg``, same arguments). ``cand`` holds bf16 embedding rows or fp32 memo rows of
    the local shard (memo selects the kernel); the winner's row lands in ``out_cand`` and
    the hidden-slice partials in ``out_partials``, or with memo rows and ``fc_out`` the
    next step's fc output. ``token_only`` exchanges the keys alone; ``partial_flag`` makes
    a hidden rank wait for the side-stream partial; ``scan_part`` splits large shard
    scans; ``head_part`` (the head's per-CTA maxima) replaces the scan."""
    memo = cand.dtype == torch.float32
    device = logits.device
    v = int(logits.shape[-1])
    if cand.shape[-1] != HIDDEN or world not in WORLDS:
        raise ValueError(f"mkd_xchg: hidden {cand.shape[-1]} at TP {world}")
    _check(logits, "logits", (1, v), torch.bfloat16, aligned=True)
    row_dtype = torch.float32 if memo else torch.bfloat16
    _check(cand, "cand", (v, HIDDEN), row_dtype, aligned=True)
    _check(out_cand, "out_cand", (1, HIDDEN), row_dtype)
    _check(hot_ids, "hot_ids", (world * v,), torch.int64)
    _check(own_partial, "own_partial", (1, HIDDEN), torch.float32)
    _check(epoch_done, "epoch_done", (2,), torch.int32)
    _check(
        out_partials,
        "out_partials",
        (world - _embed_slices(world), HIDDEN),
        torch.float32,
    )
    if fc_out is not None and fc_out.numel() > 0:
        if not memo:
            raise ValueError("mkd_xchg: fc_out needs memo candidates")
        _check(fc_out, "fc_out", (1, HIDDEN), torch.bfloat16)
    else:
        fc_out = None
    _check(topk_index, "topk_index", (1, 1), torch.int64)
    _check(topk_p, "topk_p", (1, 1), torch.float32)
    if v % 8 != 0 or v > 65536:
        raise ValueError("mkd_xchg: local vocab must be a multiple of 8 and <= 65536")
    head_parts = head_part.numel() if head_part is not None else 0
    if head_parts > XCHG_THREADS:
        raise ValueError("mkd_xchg: at most one head partial per exchange thread")
    trace_slots = trace.numel() // (CHUNKS * 9) if trace is not None else 1
    if trace_slots < 1:
        raise ValueError("mkd_xchg: trace_slots must be >= 1")
    tokens = None
    if draft_tokens is not None and draft_tokens.numel() > 0:
        _check(draft_tokens, "draft_tokens", (1, draft_tokens.shape[1]), torch.int64)
        if not 0 <= draft_column < draft_tokens.shape[1]:
            raise ValueError("mkd_xchg: draft column out of range")
        tokens = draft_tokens
    pos = None
    if positions is not None and positions.numel() > 0:
        _check(positions, "positions", (1,), torch.int64)
        pos = positions
    mrope, mrope_stride = None, 0
    if mrope_positions is not None and mrope_positions.numel() > 0:
        if (
            tuple(mrope_positions.shape) != (3, 1)
            or mrope_positions.stride(1) != 1
            or mrope_positions.dtype != torch.int64
        ):
            raise ValueError(
                f"mkd_xchg: mrope_positions {mrope_positions.dtype} "
                f"{tuple(mrope_positions.shape)} strides {tuple(mrope_positions.stride())}"
            )
        mrope, mrope_stride = mrope_positions, int(mrope_positions.stride(0))
    pf_bytes = (
        min(prefetch_bytes, prefetch.numel() * prefetch.element_size())
        if prefetch is not None
        else 0
    )
    p = MkdXchgParams(
        logits=logits.data_ptr(),
        cand=cand.data_ptr(),
        hot_ids=hot_ids.data_ptr(),
        own_partial=own_partial.data_ptr(),
        partial_flag=jit.null_or_pointer(partial_flag).value,
        scan_part=jit.null_or_pointer(scan_part).value,
        head_part=jit.null_or_pointer(head_part).value,
        head_parts=head_parts,
        mc=mc_ptr,
        uc=uc_ptr,
        epoch=epoch_done.data_ptr(),
        done=epoch_done.data_ptr() + 4,
        out_cand=out_cand.data_ptr(),
        out_partials=out_partials.data_ptr(),
        fc_out=jit.null_or_pointer(fc_out).value,
        token_only=int(bool(token_only)),
        topk_index=topk_index.data_ptr(),
        topk_p=topk_p.data_ptr(),
        draft_tokens=jit.null_or_pointer(tokens).value,
        positions=jit.null_or_pointer(pos).value,
        mrope=jit.null_or_pointer(mrope).value,
        vocab=v,
        nan_key=nan_key & 0xFFFFFFFF,
        rank=rank,
        draft_column=draft_column,
        mrope_stride=mrope_stride,
        trace=jit.null_or_pointer(trace).value,
        trace_slots=trace_slots,
        pf=jit.null_or_pointer(prefetch).value,
        pf_bytes=pf_bytes,
    )
    kernels = _xchg_kernels(device.index, world)
    _launch(
        kernels["mkd_xchg_memo" if memo else "mkd_xchg_emb"],
        CHUNKS,
        p,
        device,
        pdl=True,
    )


# ---------------------------------------------------------------- fc of an exchanged input


def mkd_fc_sum(
    emb: torch.Tensor,
    memo: Optional[torch.Tensor],
    hidden_partials: torch.Tensor,
    fc_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    out: torch.Tensor,
    world: int,
) -> torch.Tensor:
    """fc output [1, H] of one token: rank-order sum of the embedding-half partials
    (computed here from ``emb`` [1, H], or the memo row [1, H] fp32) and the received
    hidden-half partials [world / 2, H] fp32; bitwise equal to the row-split fc."""
    device = out.device
    _check(
        hidden_partials,
        "hidden_partials",
        (world - _embed_slices(world), HIDDEN),
        torch.float32,
    )
    _check(fc_weight, "fc_weight", (HIDDEN, 2 * HIDDEN), torch.bfloat16, aligned=True)
    _check(norm_weight, "norm_weight", (HIDDEN,), torch.bfloat16, aligned=True)
    _check(out, "out", (1, HIDDEN), torch.bfloat16)
    use_memo = memo is not None and memo.numel() > 0
    if use_memo:
        _check(memo, "memo", (1, HIDDEN), torch.float32)
    else:
        _check(emb, "emb", (1, HIDDEN), torch.bfloat16, aligned=True)
    grid = _sm_count(device.index)
    if (HIDDEN + grid - 1) // grid > FC_MAX_ROWS:
        raise ValueError("mkd_fc_sum: more fc rows per CTA than kMaxRows")
    p = MkdFcSumParams(
        emb=None if use_memo else emb.data_ptr(),
        memo=memo.data_ptr() if use_memo else None,
        hidden_partials=hidden_partials.data_ptr(),
        fc_w=fc_weight.data_ptr(),
        norm_w=norm_weight.data_ptr(),
        out=out.data_ptr(),
        eps=eps,
    )
    kernels = _fc_kernels(device.index, world)
    name = "mkd_fc_sum_memo" if use_memo else "mkd_fc_sum_emb"
    _launch(kernels[name], grid, p, device, pdl=True)
    return out


def mkd_fc_memo(
    emb: torch.Tensor,
    fc_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    world: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Memo rows [N, H] fp32 for embedding rows [N, H]: the rank-order sum of the
    embedding-half partials the row-split fc would compute for each row."""
    device = out.device
    n = int(emb.shape[0])
    _check(emb, "emb", (n, HIDDEN), torch.bfloat16, aligned=True)
    _check(fc_weight, "fc_weight", (HIDDEN, 2 * HIDDEN), torch.bfloat16, aligned=True)
    _check(norm_weight, "norm_weight", (HIDDEN,), torch.bfloat16, aligned=True)
    _check(out, "out", (n, HIDDEN), torch.float32)
    p = MkdFcMemoParams(
        emb=emb.data_ptr(),
        fc_w=fc_weight.data_ptr(),
        norm_w=norm_weight.data_ptr(),
        memo=out.data_ptr(),
        eps=eps,
        n=n,
    )
    kernels = _fc_kernels(device.index, world)
    _launch(kernels["mkd_fc_memo"], _sm_count(device.index) * 2, p, device, pdl=False)
    return out


def mkd_hidden_partial(
    h: torch.Tensor,
    shard: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    col: int,
    out: torch.Tensor,
    epoch_done: torch.Tensor,
    sync: torch.Tensor,
    world: int,
    ctas: int = 0,
) -> None:
    """Hidden-slice fc partial [1, H] fp32 of ``h`` [1, H] (draft_fc_partial arithmetic
    over ``shard`` [H, W] at hidden-half column ``col``); the last CTA then sets
    ``sync[1]`` to the exchange epoch ``epoch_done[0]`` (``sync[0]`` counts CTAs, zero
    between calls). ``ctas`` > 0 launches that many CTAs instead of one per SM (same
    values: each row is one warp's)."""
    device = out.device
    width = 2 * HIDDEN // world
    _check(h, "h", (1, HIDDEN), torch.bfloat16, aligned=True)
    _check(shard, "shard", (HIDDEN, width), torch.bfloat16, aligned=True)
    _check(norm_weight, "norm_weight", (HIDDEN,), torch.bfloat16, aligned=True)
    _check(out, "out", (1, HIDDEN), torch.float32)
    _check(epoch_done, "epoch_done", (2,), torch.int32)
    _check(sync, "sync", (2,), torch.int32)
    if not (col >= 0 and col % width == 0 and col + width <= HIDDEN):
        raise ValueError("mkd_hidden_partial: slice start must be a slice boundary")
    p = MkdHiddenPartialParams(
        h=h.data_ptr(),
        w=shard.data_ptr(),
        norm_w=norm_weight.data_ptr(),
        out=out.data_ptr(),
        epoch=epoch_done.data_ptr(),
        done=sync.data_ptr(),
        flag=sync.data_ptr() + 4,
        eps=eps,
        col=col,
    )
    kernels = _fc_kernels(device.index, world)
    grid = ctas if ctas > 0 else _sm_count(device.index)
    _launch(kernels["mkd_hidden_partial"], grid, p, device, pdl=False)


# ---------------------------------------------------------------- draft lm_head


def mkd_draft_head(
    x: torch.Tensor,
    weight: torch.Tensor,
    logits: torch.Tensor,
    part: torch.Tensor,
    reverse: bool = False,
    smem_pad: int = 0,
) -> None:
    """Draft lm_head of one token, ``logits`` [1, N] bf16 = ``x`` [1, 4096] @ ``weight``
    [N, 4096]^T with fp32 accumulation, and per CTA (``part.numel()`` CTAs) the packed
    maximum of its rows for mkd_xchg. ``reverse`` walks the rows from the end (same
    values). ``smem_pad`` bytes of unused dynamic shared memory per CTA (same values;
    above half an SM it allows one CTA per SM)."""
    device = logits.device
    n = int(weight.shape[0])
    _check(x, "x", (1, HIDDEN), torch.bfloat16, aligned=True)
    _check(weight, "weight", (n, HIDDEN), torch.bfloat16, aligned=True)
    _check(logits, "logits", (1, n), torch.bfloat16)
    _check(part, "part", (part.numel(),), torch.int32)
    if n > 65536:
        raise ValueError(
            "mkd_draft_head: rows must fit the 16-bit row of the order key"
        )
    p = MkdHeadParams(
        x=x.data_ptr(),
        w=weight.data_ptr(),
        logits=logits.data_ptr(),
        part=part.data_ptr(),
        n=n,
        reverse=int(bool(reverse)),
    )
    _launch(_head_kernel(device.index, smem_pad), part.numel(), p, device, pdl=True)


# ---------------------------------------------------------------- the row-split fc


def draft_fc_partial_rows(out_rows: int, width: int, num_sms: int) -> Optional[int]:
    """Weight rows per CTA for the fused kernel, or None if they do not fit in shared
    memory."""
    rows = -(-out_rows // num_sms)
    static = MAX_TOKENS * width * 2
    return rows if rows * width * 2 + static <= _MAX_SMEM_BYTES else None


def draft_fc_partial(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    weight_slice: torch.Tensor,
    col_start: int,
    out: torch.Tensor,
    max_rows: int,
) -> torch.Tensor:
    """Gemma RMSNorm of the input half x [M, H] (bf16 output; M <= MAX_TOKENS), then
    this rank's fc slice: out [M, N] fp32 = norm(x)[:, col_start : col_start + W] @
    weight_slice [N, W]^T."""
    device = out.device
    m, width = int(x.shape[0]), int(weight_slice.shape[-1])
    world = 2 * HIDDEN // width
    if (
        world not in WORLDS
        or weight_slice.shape[0] != HIDDEN
        or not 1 <= m <= MAX_TOKENS
    ):
        raise ValueError(
            f"draft_fc_partial: {m} rows of a {tuple(weight_slice.shape)} slice "
            f"(built for hidden {HIDDEN} at TP {WORLDS}, 1-{MAX_TOKENS} rows)"
        )
    _check(x, "x", (m, HIDDEN), torch.bfloat16, aligned=True)
    _check(norm_weight, "norm_weight", (HIDDEN,), torch.bfloat16, aligned=True)
    _check(weight_slice, "weight_slice", (HIDDEN, width), torch.bfloat16, aligned=True)
    _check(out, "out", (m, HIDDEN), torch.float32)
    if not (col_start >= 0 and col_start + width <= HIDDEN and col_start % 8 == 0):
        raise ValueError("draft_fc_partial: slice out of range")
    grid = _sm_count(device.index)
    if (HIDDEN + grid - 1) // grid > max_rows:
        raise ValueError("draft_fc_partial: more weight rows per CTA than kMaxRows")
    p = DraftFcPartialParams(
        x=x.data_ptr(),
        norm_w=norm_weight.data_ptr(),
        w=weight_slice.data_ptr(),
        out=out.data_ptr(),
        eps=eps,
        col_start=col_start,
        rows=m,
    )
    kernels = _draft_fc_kernels(device.index, world, max_rows)
    _launch(kernels[f"draft_fc_partial_m{m}"], grid, p, device, pdl=True)
    return out


@triton.jit
def _ordered_partial_sum_kernel(
    src_ptr,  # [M, world * N] fp32: rank r's partial at columns [r * N, (r + 1) * N)
    dst_ptr,  # [M, N] bf16
    N,
    WORLD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    base = src_ptr + row * WORLD * N + offs
    acc = tl.load(base, mask=mask, other=0.0)
    for r in tl.static_range(1, WORLD):
        acc += tl.load(base + r * N, mask=mask, other=0.0)
    tl.store(dst_ptr + row * N + offs, acc.to(tl.bfloat16), mask=mask)


def ordered_partial_sum(
    gathered: torch.Tensor, world: int, out: torch.Tensor
) -> torch.Tensor:
    """Sum ``world`` fp32 partials in rank order, rounding once to bf16 into ``out``."""
    m, n = out.shape
    block = 1024
    _ordered_partial_sum_kernel[(m, triton.cdiv(n, block))](
        gathered, out, n, WORLD=world, BLOCK=block, num_warps=4
    )
    return out


def symmetric_words(words: int, group, device: torch.device):
    """(tensor, handle, multicast pointer) of a zeroed symmetric int64 buffer on
    ``group``; a collective on every rank of it."""
    import torch.distributed._symmetric_memory as symm_mem

    if symm_mem.get_backend(device) is None:
        # the backend FlashInfer's fusion workspace would select later in this process
        symm_mem.set_backend("NCCL")
    buf = symm_mem.empty((words,), dtype=torch.int64, device=device)
    handle = symm_mem.rendezvous(buf, group)
    multicast = int(handle.multicast_ptr or 0)
    if not multicast or multicast % 16:
        raise RuntimeError("the draft step boundary needs an NVLink multicast mapping")
    buf.zero_()
    return buf, handle, multicast


class FcGather:
    """The split fc's all-gather of fp32 partials, [m, H] on every rank -> [m, world * H]
    with rank r's partial at columns [r * H, (r + 1) * H) (the layout
    ``ordered_partial_sum`` reads), for m <= GATHER_MAX_TOKENS; graph-capturable. Build on
    every rank of ``group`` (a collective)."""

    def __init__(
        self, *, world: int, rank: int, group, cpu_group, device: torch.device
    ):
        import torch.distributed as dist

        if world not in WORLDS:
            raise ValueError(f"the fc gather serves TP {WORLDS}, not {world}")
        self.world, self.rank, self.device = world, rank, device
        words = SLOTS * world * GATHER_MAX_TOKENS * HIDDEN
        self._buf, self._handle, self.mc = symmetric_words(words, group, device)
        self.uc = self._buf.data_ptr()
        # [epoch, CTA arrivals]: the epoch tags the call's words, from 1
        self.epoch_done = torch.tensor([1, 0], dtype=torch.int32, device=device)
        sms = _sm_count(device.index)
        self._kernels = _draft_fc_kernels(
            device.index, world, draft_fc_partial_rows(HIDDEN, 2 * HIDDEN // world, sms)
        )
        torch.cuda.synchronize(device)
        dist.barrier(group=cpu_group)

    def __call__(self, partial: torch.Tensor) -> torch.Tensor:
        m = int(partial.shape[0])
        if not 1 <= m <= GATHER_MAX_TOKENS:
            raise ValueError(f"fc gather: {m} rows (1-{GATHER_MAX_TOKENS})")
        _check(partial, "partial", (m, HIDDEN), torch.float32)
        out = torch.empty(
            (m, self.world * HIDDEN), dtype=torch.float32, device=partial.device
        )
        p = FcGatherParams(
            partial=partial.data_ptr(),
            out=out.data_ptr(),
            mc=self.mc,
            uc=self.uc,
            epoch=self.epoch_done.data_ptr(),
            done=self.epoch_done.data_ptr() + 4,
            rank=self.rank,
            m=m,
        )
        _launch(self._kernels["fc_gather"], GATHER_CTAS, p, self.device, pdl=False)
        return out

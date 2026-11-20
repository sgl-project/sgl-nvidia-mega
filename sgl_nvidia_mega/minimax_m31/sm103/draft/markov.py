# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MiniMax M3.1's DSpark draft Markov chain in one cooperative, vocab-sharded kernel per
rank: a port of ``deepseek_v41/sm103/markov.py`` to M3.1's vocabulary of 200,064 (an
18-bit index field in the exchanged keys, a 14-bit launch tag) whose anchors are read in
place from the draft's int64 input ids (no copy launch). Built with this package's nvcc
helper (``sm103/nvcc.py``), loaded and launched through ``sm103/host.py``.

Problem. The M3.1 DSpark draft samples ``num_steps`` (7) tokens per request row, one
after the other: step i adds the bias ``markov_w2 . markov_w1[previous token]`` to the
base logits, picks the argmax (temperature 0) or the Gumbel-max sample, and keeps the
step logits for the accept. SGLang runs every step as a GEMV, a sample and a cross-rank
exchange (VanillaMarkov's vocab shard and SGLang's MarkovSampleStep).

Responsibility. :class:`ShardedMarkov` runs all steps of all rows in one launch on each
rank (markov.cu). Rank k keeps rows ``[k V / world, (k + 1) V / world)`` of the Markov head
in registers, adds the bias to its shard of the base logits, picks its best (score,
index) key and exchanges one 64-bit key per rank, request row and step through a mailbox
in symmetric memory; the kept step logits reach every rank through peer staging. Up to
``narrow_rows`` rows run the narrow kernel (the direct exchange at one row, the
rank-local fold at 2-8), more rows (up to 64) the wide kernel. Every cross-rank wait is
bounded by ``deadline_s``; a rank whose peer stops faults instead of hanging.

Numerics. Tokens and kept logits equal SGLang's ``draft_markov_block`` step by step when
the bias products sum exactly: the bias is rounded to BF16, added in FP32 and rounded
once to the cache dtype; the noise is SGLang's Philox draft noise (``markov_sample.py``:
same key, counters and offset), drawn for this rank's shard before the chain. With
``advance_offset`` (SGLang's int64 noise counter) the chain kernel adds 1 to it at its
end, as ``draft_markov_block`` does once per sampled block: the caller must not advance
it again.
"""

from __future__ import annotations

import ctypes
import functools
from pathlib import Path
from typing import NamedTuple, Optional

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sgl_nvidia_mega.common.jit import check as _check
from sgl_nvidia_mega.common.jit import driver as _driver
from sgl_nvidia_mega.common.jit import pointer as _pointer
from sgl_nvidia_mega.minimax_m31.sm103 import host, nvcc

SOURCE = Path(__file__).resolve().parent / "markov.cu"
NARROW_SYMBOL = "m31_markov_narrow"
WIDE_SYMBOL = "m31_markov_wide"
# markov.cu annotates every function, so it needs no default-device switch; the
# floating-point options stay the compiler's defaults, as the kernels were tested
NVCC_OPTIONS = ("-std=c++17",)
_C_TYPES = {torch.bfloat16: "__nv_bfloat16", torch.float32: "float"}

# Must match the constants of markov.cu.
RANK_DIM = 256
THREADS = 512
WARPS = THREADS // 32
MAX_STEPS = 8
MAX_RANKS = 8
# MMA columns: request rows of the narrow kernel.
MAX_ROWS = 8
# Request rows of the wide kernel.
WIDE_ROWS = 64
# The candidate key's 18-bit index field.
MAX_VOCAB = (1 << 18) - 1
# Element types of the base logits (the BF16 LM head's) and of the step-logits cache.
BASE_DTYPES = (torch.bfloat16,)
LOGIT_DTYPES = (torch.bfloat16, torch.float32)
# Head tiles a CTA keeps in shared memory beyond its warps' (markov.cu SMEM_TILES): one
# per warp at most, 8 KB each plus two steps of their base logits and noise.
MAX_SMEM_TILES = WARPS
# Peer buffer bytes before the staging logits: the fold's mailbox records.
STAGING_OFFSET = 16384
# Per-CTA launch counters at words [256, 512) (fold) and [512, 768) (direct).
CONTROL_WORDS = 768
MAX_CTAS = 256
# The wide kernel's control words: combine, arrivals, per-CTA counters at 1280.
WIDE_CONTROL_WORDS = 1280 + MAX_CTAS
WIDE_RECORD_BYTES = 2 * MAX_STEPS * WIDE_ROWS * MAX_RANKS * 8
NOISE_BLOCK = 1024

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3))
SUPPORTED_WORLDS = (1, 2, 4, MAX_RANKS)
# Measured on GB300 TP4: the direct exchange wins at 1 row, the rank-local fold at 2-8.
DIRECT_MAX_ROWS = 1

# markov_sample.py's Philox draft noise.
_UNIFORM_SCALE = tl.constexpr(2.3283064365386963e-10)
_PHILOX_ROUNDS = tl.constexpr(10)


@triton.jit
def _shard_noise_kernel(
    NOISE,
    OFFSET,
    KEY,
    VOCAB_START,
    VOCAB_LOCAL,
    TEMPERATURE,
    STEPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Gumbel noise of (row, step) for this rank's vocab shard, as SGLang's draft step draws it;
    nothing for a row at temperature 0, whose argmax reads none."""
    sample = tl.program_id(0)
    row = sample // STEPS
    step = sample % STEPS
    part = tl.program_id(1)
    if tl.load(TEMPERATURE + row) != 0.0:
        quad = (part * BLOCK + VOCAB_START) // 4 + tl.arange(0, BLOCK // 4)
        lane = tl.zeros([BLOCK // 4], tl.int64)
        w0, w1, w2, w3 = tl.philox(
            KEY,
            quad.to(tl.uint32),
            (lane + row).to(tl.uint32),
            (lane + step).to(tl.uint32),
            ((lane + tl.load(OFFSET)) & 0xFFFFFFFF).to(tl.uint32),
            _PHILOX_ROUNDS,
        )
        bits = tl.interleave(tl.interleave(w0, w2), tl.interleave(w1, w3))
        uniform = (bits.to(tl.float32) + 0.5) * _UNIFORM_SCALE
        exp_noise = -libdevice.log1p(-uniform)
        gumbel = -tl.log(exp_noise)
        cols = part * BLOCK + tl.arange(0, BLOCK)
        tl.store(NOISE + sample * VOCAB_LOCAL + cols, gumbel, mask=cols < VOCAB_LOCAL)


def unsupported_reason(
    *,
    world: int,
    vocab_size: int,
    num_steps: int,
    embed_weight: torch.Tensor,
    head_weight: torch.Tensor,
    cache_dtype: torch.dtype,
    capability: tuple[int, int],
) -> Optional[str]:
    """Why the chain kernel cannot serve this configuration, or None; the same on every rank."""
    if world not in SUPPORTED_WORLDS:
        return f"a tensor-parallel size of {world}"
    if tuple(capability) not in SUPPORTED_CAPABILITIES:
        return f"compute capability {capability} (needs SM100 or SM103)"
    if not 1 <= num_steps <= MAX_STEPS:
        return f"{num_steps} draft positions"
    if vocab_size > MAX_VOCAB or vocab_size % (16 * world) != 0:
        return f"a vocabulary of {vocab_size} for {world} ranks"
    if cache_dtype not in LOGIT_DTYPES:
        return f"a {cache_dtype} step-logits buffer"
    for name, weight in (("embedding", embed_weight), ("head", head_weight)):
        if (
            weight.dim() != 2
            or weight.shape[1] != RANK_DIM
            or weight.dtype != torch.bfloat16
            or not weight.is_contiguous()
        ):
            return f"a Markov {name} that is not contiguous BF16 [*, {RANK_DIM}]"
    if head_weight.shape[0] < vocab_size or embed_weight.shape[0] < vocab_size:
        return "a Markov table smaller than the vocabulary"
    return None


def narrow_row_limit(*, world: int, max_bs: int) -> int:
    """Request rows of the narrow kernel: its exchange gives each (row, rank) pair one lane."""
    return max(1, min(MAX_ROWS, 32 // world, max_bs))


def direct_mailbox_bytes(*, steps: int, world: int, ctas: int, max_rows: int) -> int:
    """Direct-exchange records u64 [parity 2][step][source rank][cta][row]."""
    return 2 * steps * world * ctas * max_rows * 8


# Static shared memory of the kernels, a bound (markov.cu: the narrow kernel's 9.5-13.5
# KiB, the wide kernel's 41-45 KiB, with a BF16 or FP32 step-logits cache)
STATIC_SMEM = {"narrow": 16 * 1024, "wide": 48 * 1024}


def smem_tiles(*, tiles: int, ctas: int, cap: int = MAX_SMEM_TILES) -> int:
    """Shared-memory tiles per CTA: the tiles beyond one per warp, spread over the CTAs
    (M3.1's TP4 shard: 3,126 - 152 x 16 = 694 over 152 CTAs: 5); past ``cap`` (what
    shared memory holds) the rest stream from L2."""
    beyond = max(0, tiles - ctas * WARPS)
    return min(cap, MAX_SMEM_TILES, -(-beyond // ctas))


def input_bytes(*, in_rows: int, base_dtype: torch.dtype) -> int:
    """markov.cu InputBytes: two steps of a tile's base logits and noise."""
    return 2 * in_rows * 16 * base_dtype.itemsize + 2 * in_rows * 16 * 4


def smem_tile_bytes(*, in_rows: int, base_dtype: torch.dtype) -> int:
    """markov.cu SmemTileBytes: the tile, then its inputs."""
    return 8192 + input_bytes(in_rows=in_rows, base_dtype=base_dtype)


def dynamic_smem_bytes(
    *, tiles_in_smem: int, in_rows: int, base_dtype: torch.dtype
) -> int:
    """markov.cu DynamicBytes: the shared-memory tiles, then every warp's register-tile
    inputs."""
    return tiles_in_smem * smem_tile_bytes(
        in_rows=in_rows, base_dtype=base_dtype
    ) + WARPS * input_bytes(in_rows=in_rows, base_dtype=base_dtype)


def smem_tile_cap(*, opt_in_bytes: int, wide_rows: int, base_dtype: torch.dtype) -> int:
    """The shared-memory tiles both kernels can hold beside their static memory."""
    cap = MAX_SMEM_TILES
    for name, in_rows in (("narrow", MAX_ROWS), ("wide", wide_in_rows(wide_rows))):
        free = (
            opt_in_bytes
            - STATIC_SMEM[name]
            - dynamic_smem_bytes(
                tiles_in_smem=0, in_rows=in_rows, base_dtype=base_dtype
            )
        )
        cap = min(
            cap, max(0, free // smem_tile_bytes(in_rows=in_rows, base_dtype=base_dtype))
        )
    return cap


def wide_in_rows(wide_rows: int) -> int:
    """Request rows the wide kernel's shared-memory inputs hold (a multiple of 8)."""
    return max(MAX_ROWS, -(-wide_rows // MAX_ROWS) * MAX_ROWS)


def defines(
    *,
    steps: int,
    base_dtype: torch.dtype,
    cache_dtype: torch.dtype,
    direct: bool,
    smem_tiles: int = 0,
    wide_rows: int = WIDE_ROWS,
) -> tuple[str, ...]:
    return (
        f"-DBASE_T={_C_TYPES[base_dtype]}",
        f"-DCACHE_T={_C_TYPES[cache_dtype]}",
        f"-DMARKOV_DIRECT={int(direct)}",
        f"-DSTEPS={steps}",
        f"-DSMEM_TILES={smem_tiles}",
        f"-DWIDE_IN_ROWS={wide_in_rows(wide_rows)}",
    )


def compile_cubin(
    *,
    steps: int,
    base_dtype: torch.dtype,
    cache_dtype: torch.dtype,
    direct: bool,
    arch: str,
    smem_tiles: int = 0,
    wide_rows: int = WIDE_ROWS,
) -> bytes:
    """The module of one exchange: the narrow kernel, plus the wide kernel with the fold."""
    options = NVCC_OPTIONS + defines(
        steps=steps,
        base_dtype=base_dtype,
        cache_dtype=cache_dtype,
        direct=direct,
        smem_tiles=smem_tiles,
        wide_rows=wide_rows,
    )
    # the translation unit includes the toolkit's headers only: no include root
    return nvcc.compile_cubin(
        "minimax_m31_markov", SOURCE.read_text(), (), arch, options
    )


class _Kernel(NamedTuple):
    function: object  # cuda.bindings.driver.CUfunction
    max_ctas: int  # co-resident CTAs on the device (the cooperative grid's bound)
    dynamic_smem: int  # bytes: the shared-memory tiles and their inputs


@functools.cache
def load_kernels(
    device_index: int,
    steps: int,
    base_dtype: torch.dtype,
    cache_dtype: torch.dtype,
    direct: bool,
    smem_tiles: int = 0,
    wide_rows: int = WIDE_ROWS,
) -> dict[str, _Kernel]:
    """Compile (or read from the cache) and load one exchange's kernels."""
    device = torch.device("cuda", device_index)
    in_rows = {"narrow": MAX_ROWS, "wide": wide_in_rows(wide_rows)}
    symbols = {"narrow": NARROW_SYMBOL}
    if not direct:
        symbols["wide"] = WIDE_SYMBOL
    kernels = {}
    with torch.cuda.device(device):
        torch.zeros(1, device=device)  # make the primary context current
        cubin = compile_cubin(
            steps=steps,
            base_dtype=base_dtype,
            cache_dtype=cache_dtype,
            direct=direct,
            arch=nvcc.device_arch(device),
            smem_tiles=smem_tiles,
            wide_rows=wide_rows,
        )
        for name, symbol in symbols.items():
            module = host.Module(cubin, symbol)
            dynamic = dynamic_smem_bytes(
                tiles_in_smem=smem_tiles, in_rows=in_rows[name], base_dtype=base_dtype
            )
            module.set_shared_memory(dynamic)
            kernels[name] = _Kernel(
                module.function,
                int(module.resident_ctas(device, THREADS, dynamic)),
                dynamic,
            )
    return kernels


def _launch_cooperative(
    kernel: _Kernel, ctas: int, arguments: list, stream: int
) -> None:
    """Cooperative launch (co-resident grid) without programmatic dependent launch;
    graph-capturable."""
    host.launch(
        kernel.function,
        (ctas, 1, 1),
        (THREADS, 1, 1),
        kernel.dynamic_smem,
        arguments,
        stream,
        cooperative=True,
    )


class ShardedMarkov:
    """Persistent buffers, peer mappings and launchers of the sharded chain.

    Every rank of ``group`` (the shard's CPU process group, None for one rank) must
    construct it: the constructor runs CPU collectives. It compiles every kernel a
    launch may pick and checks their co-residency before the collective set-up, and
    raises on every rank, with each rank's error, if any rank failed (``host.agree``).
    ``symm_group`` is the process group the peer buffer's symmetric memory meets on
    (default: ``group``).
    """

    def __init__(
        self,
        *,
        embed_weight: torch.Tensor,
        head_weight: torch.Tensor,
        vocab_size: int,
        num_steps: int,
        cache_dtype: torch.dtype,
        group,
        rank: int,
        world: int,
        max_bs: int,
        deadline_s: float,
        keeps_logits: bool,
        symm_group=None,
    ):
        device = embed_weight.device
        reason = unsupported_reason(
            world=world,
            vocab_size=vocab_size,
            num_steps=num_steps,
            embed_weight=embed_weight,
            head_weight=head_weight,
            cache_dtype=cache_dtype,
            capability=torch.cuda.get_device_capability(device),
        )
        if reason is not None:
            raise ValueError(f"sharded Markov sampling does not support {reason}")
        if deadline_s <= 0:
            raise ValueError(
                f"sharded Markov sampling needs a deadline, got {deadline_s}"
            )
        self.group, self.rank, self.world = group, rank, world
        self.symm_group = group if symm_group is None else symm_group
        self.device = device
        self.vocab_size = vocab_size
        self.num_steps = num_steps
        self.cache_dtype = cache_dtype
        # without it the peer buffer has no staging for kept logits
        self.keeps_logits = keeps_logits
        self.timeout_ns = int(deadline_s * 1e9)
        self.vocab_local = vocab_size // world
        self.vocab_offset = rank * self.vocab_local
        self.embed_weight = embed_weight
        self.head_shard = head_weight[
            self.vocab_offset : self.vocab_offset + self.vocab_local
        ]
        tiles = self.vocab_local // 16
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.ctas = min(sms, -(-tiles // WARPS))
        if self.ctas > MAX_CTAS:
            raise ValueError(
                f"the Markov chain counts launches for up to {MAX_CTAS} CTAs, not {self.ctas}"
            )
        self.narrow_rows = narrow_row_limit(world=world, max_bs=max_bs)
        # The wide kernel takes the rows past the narrow kernel's.
        wide = max_bs > self.narrow_rows
        self.wide_rows = min(WIDE_ROWS, max_bs) if wide else 0
        self.max_rows = max(self.narrow_rows, self.wide_rows)
        # The tiles beyond one per warp: in shared memory, past that from L2.
        cu = _driver()
        (opt_in_bytes,) = _check(
            cu.cuDeviceGetAttribute(
                cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                device.index
                if device.index is not None
                else torch.cuda.current_device(),
            ),
            "cuDeviceGetAttribute",
        )
        self.smem_tiles = smem_tiles(
            tiles=tiles,
            ctas=self.ctas,
            cap=smem_tile_cap(
                opt_in_bytes=int(opt_in_bytes),
                wide_rows=self.wide_rows,
                base_dtype=torch.bfloat16,
            ),
        )
        self.noise = torch.empty(
            self.max_rows * num_steps,
            self.vocab_local,
            dtype=torch.float32,
            device=device,
        )
        self.control = torch.zeros(CONTROL_WORDS, dtype=torch.int32, device=device)
        # Request row r reads temperature[r] and keeps its step logits at cache[r].
        rows = torch.arange(self.max_rows, dtype=torch.int32, device=device)
        self.sampled_rows = rows.repeat_interleave(num_steps)
        self.inert_rows = torch.full_like(self.sampled_rows, -1)
        size = self._layout_peer_buffer(keeps_logits=keeps_logits)
        self.wide_control = (
            torch.zeros(WIDE_CONTROL_WORDS, dtype=torch.int32, device=device)
            if self.wide_rows
            else None
        )
        # Compile before the collective set-up so a build failure cannot leave peers
        # waiting: every rank raises, with each rank's error.
        self._kernels: dict[tuple[torch.dtype, str], _Kernel] = {}
        if group is None:
            self._load_kernels()
        else:
            host.agree(group, self._load_kernels, "building the Markov chain kernels")
        self._peer_buffer, peer_pointers = self._allocate_peer_buffer(size)
        self.peers = torch.tensor(peer_pointers, dtype=torch.int64, device=device)
        self._noise_stream = torch.cuda.Stream(device)

    def _layout_peer_buffer(self, *, keeps_logits: bool) -> int:
        """Byte size of the peer buffer; sets the direct mailbox and wide-kernel offsets.

        Fold records, then staging logits [exchange 2][parity 2][narrow rows] (none for a
        greedy draft, which never writes them), then the direct mailbox, then the wide
        kernel's records and staging logits [parity 2][wide rows].
        """
        row_bytes = self.num_steps * self.vocab_size * self.cache_dtype.itemsize
        if not keeps_logits:
            row_bytes = 0
        size = STAGING_OFFSET + 4 * self.narrow_rows * row_bytes
        self.mailbox_offset = size
        size += direct_mailbox_bytes(
            steps=self.num_steps,
            world=self.world,
            ctas=self.ctas,
            max_rows=self.narrow_rows,
        )
        self.wide_offset = -(-size // 16) * 16
        if self.wide_rows:
            size = self.wide_offset + WIDE_RECORD_BYTES
            size += 2 * self.wide_rows * row_bytes
        return size

    def device_bytes(self) -> dict[str, int]:
        """Device memory this chain holds, by buffer."""
        small = (
            self.control,
            self.wide_control,
            self.peers,
            self.sampled_rows,
            self.inert_rows,
        )
        return {
            "peer buffer (symmetric)": self._peer_buffer.numel(),
            "noise": self.noise.numel() * self.noise.element_size(),
            "control": sum(
                t.numel() * t.element_size() for t in small if t is not None
            ),
        }

    def _kernel_names(self) -> list[str]:
        """``narrow-direct``, ``narrow-fold`` and ``wide``: those a launch may pick."""
        names = []
        for rows in range(1, self.narrow_rows + 1):
            name = "narrow-direct" if rows <= DIRECT_MAX_ROWS else "narrow-fold"
            if name not in names:
                names.append(name)
        if self.wide_rows:
            names.append("wide")
        return names

    def _load_kernels(self) -> None:
        index = self.device.index
        if index is None:
            index = torch.cuda.current_device()
        for base_dtype in BASE_DTYPES:
            for name in self._kernel_names():
                module = load_kernels(
                    index,
                    self.num_steps,
                    base_dtype,
                    self.cache_dtype,
                    name == "narrow-direct",
                    self.smem_tiles,
                    max(self.wide_rows, MAX_ROWS),
                )
                kernel = module["wide" if name == "wide" else "narrow"]
                if self.ctas > kernel.max_ctas:
                    raise RuntimeError(
                        f"sharded Markov sampling needs {self.ctas} co-resident CTAs; "
                        f"the device holds {kernel.max_ctas}"
                    )
                self._kernels[base_dtype, name] = kernel

    def _allocate_peer_buffer(self, size: int) -> tuple[torch.Tensor, list[int]]:
        if self.group is None:
            buffer = torch.zeros(size, dtype=torch.uint8, device=self.device)
            return buffer, [buffer.data_ptr()]
        from torch.distributed import _symmetric_memory as symm_mem

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "the Markov chain's peer buffer must be created before CUDA-graph "
                "capture"
            )
        # as sm103/host.py's Arena: every rank allocates or every rank raises, the
        # ranks meet, and none leaves before all have mapped and zeroed theirs
        with torch.cuda.device(self.device):
            buffer = host.agree(
                self.group,
                lambda: symm_mem.empty(size, dtype=torch.uint8, device=self.device),
                "allocating the Markov chain's peer buffer",
            )
            handle = symm_mem.rendezvous(buffer, self.symm_group)

            def mapped() -> list[int]:
                pointers = [int(p) for p in handle.buffer_ptrs]
                if len(pointers) != self.world or not all(pointers):
                    raise RuntimeError(
                        f"{len(pointers)} peer mappings ({pointers.count(0)} null) "
                        f"for {self.world} ranks"
                    )
                return pointers

            pointers = host.agree(
                self.group, mapped, "mapping the Markov chain's peer buffer"
            )
            buffer.zero_()
            self._handle = handle
            torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        return buffer, pointers

    def supports(self, base_logits: torch.Tensor) -> bool:
        """``[rows, num_steps, vocab or vocab // world]`` base logits the chain takes; depends
        only on shapes, which tensor parallelism keeps identical across ranks."""
        size = base_logits.element_size()
        return (
            base_logits.dim() == 3
            and 1 <= base_logits.shape[0] <= self.max_rows
            and base_logits.shape[1] == self.num_steps
            and base_logits.shape[2] in (self.vocab_size, self.vocab_local)
            and base_logits.dtype in BASE_DTYPES
            and base_logits.stride(2) == 1
            # the shared-memory tiles' inputs arrive by 16-byte cp.async
            and (
                not self.smem_tiles
                or (
                    base_logits.data_ptr() % 16 == 0
                    and base_logits.stride(0) * size % 16 == 0
                    and base_logits.stride(1) * size % 16 == 0
                    and self.vocab_offset * size % 16 == 0
                )
            )
        )

    def launch_noise(
        self,
        *,
        rows: int,
        key: int,
        offset: torch.Tensor,
        temperatures: torch.Tensor,
    ) -> None:
        """The Gumbel noise of ``rows`` rows on the current stream, none for the rows at
        temperature 0 (``temperatures``: the float32 step temperatures sample() reads,
        on the device); ``offset`` is SGLang's int64 draft-noise counter (on the device)."""
        assert temperatures.dtype == torch.float32 and temperatures.numel() >= rows
        grid = (rows * self.num_steps, triton.cdiv(self.vocab_local, NOISE_BLOCK))
        _shard_noise_kernel[grid](
            self.noise,
            offset,
            key,
            self.vocab_offset,
            self.vocab_local,
            temperatures,
            STEPS=self.num_steps,
            BLOCK=NOISE_BLOCK,
        )

    def fork_noise(
        self,
        *,
        rows: int,
        key: int,
        offset: torch.Tensor,
        temperatures: torch.Tensor,
    ) -> None:
        """Draw the noise on a side stream; join_noise must precede sample()."""
        self._noise_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self._noise_stream):
            self.launch_noise(
                rows=rows, key=key, offset=offset, temperatures=temperatures
            )

    def join_noise(self) -> None:
        torch.cuda.current_stream(self.device).wait_stream(self._noise_stream)

    def sample(
        self,
        *,
        base_logits: torch.Tensor,
        anchor_tokens: torch.Tensor,
        temperatures: torch.Tensor,
        logits_cache: Optional[torch.Tensor],
        draft_tokens: torch.Tensor,
        advance_offset: Optional[torch.Tensor] = None,
    ) -> None:
        """Draft tokens of every row into ``draft_tokens`` [rows, num_steps] and, when
        ``logits_cache`` [>= rows, num_steps, vocab] is given, the full-vocab step logits
        of every row at a nonzero temperature (temperature-0 rows are left untouched);
        without it every row is argmaxed. The noise must be drawn already. With
        ``advance_offset`` (SGLang's int64 draft-noise counter) the kernel adds 1 to it at
        its end, as draft_markov_block does after a sampled block."""
        rows = base_logits.shape[0]
        assert self.supports(base_logits)
        assert temperatures.dtype == torch.float32
        assert draft_tokens.dtype == torch.int64 and draft_tokens.stride(1) == 1
        # read in place (the draft graph's input ids, one anchor per request row)
        assert anchor_tokens.dtype == torch.int64 and anchor_tokens.dim() == 1
        assert anchor_tokens.shape[0] == rows
        assert advance_offset is None or (
            advance_offset.dtype == torch.int64 and advance_offset.numel() == 1
        )
        wide = rows > self.narrow_rows
        if logits_cache is None:
            idx_mapping, cache, stride0, stride1 = self.inert_rows, self.noise, 0, 0
        else:
            assert self.keeps_logits, "built without logit staging (keeps_logits=False)"
            assert logits_cache.dtype == self.cache_dtype
            assert logits_cache.size(2) >= self.vocab_size
            assert logits_cache.stride(2) == 1 and logits_cache.data_ptr() % 16 == 0
            idx_mapping, cache = self.sampled_rows, logits_cache
            stride0, stride1 = logits_cache.stride(0), logits_cache.stride(1)
        base_column = (
            self.vocab_offset if base_logits.shape[2] == self.vocab_size else 0
        )
        # Host-static per row count, so a CUDA graph of one batch size captures one kernel.
        if wide:
            name = "wide"
        elif rows <= DIRECT_MAX_ROWS:
            name = "narrow-direct"
        else:
            name = "narrow-fold"
        arguments = [
            _pointer(base_logits),
            ctypes.c_int64(base_logits.stride(0)),
            ctypes.c_int64(base_logits.stride(1)),
            ctypes.c_int64(base_column),
            _pointer(self.embed_weight),
            _pointer(self.head_shard),
            _pointer(anchor_tokens),
            ctypes.c_int64(anchor_tokens.stride(0)),
            _pointer(idx_mapping),
            _pointer(temperatures),
            _pointer(self.noise),
            _pointer(cache),
            ctypes.c_int64(stride0),
            ctypes.c_int64(stride1),
            _pointer(draft_tokens),
            ctypes.c_int64(draft_tokens.stride(0)),
            _pointer(self.peers),
            _pointer(self.wide_control if wide else self.control),
            ctypes.c_int32(self.rank),
            ctypes.c_int32(self.world),
            ctypes.c_int32(self.vocab_local),
            ctypes.c_int32(rows),
            ctypes.c_int32(self.wide_rows if wide else self.narrow_rows),
            ctypes.c_int64(self.timeout_ns),
            ctypes.c_int64(self.wide_offset if wide else self.mailbox_offset),
            ctypes.c_void_p(0 if advance_offset is None else advance_offset.data_ptr()),
        ]
        stream = torch.cuda.current_stream(self.device).cuda_stream
        _launch_cooperative(
            self._kernels[base_logits.dtype, name], self.ctas, arguments, stream
        )

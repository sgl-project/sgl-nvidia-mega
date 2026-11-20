# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The plugin-owned MNNVL Lamport buffer of the GLM low-latency hops, and the hop
kernels' JIT modules and launches.

The hops (``ll.hops``) run ``csrc/distributed/glm_ll_mnnvl_oneshot.cuh``: a one-shot
(or two-shot) all-reduce + residual add + RMSNorm whose partial may be the MoE
finalize (P9) and whose norm output may also be written as NVFP4 (P10). The kernels
speak FlashInfer v0.7.1's MNNVL Lamport protocol: three buffers of
``bytes_per_buffer`` bytes, every 32-bit word -0.0 when clean, and the flags
``[cur idx, dirty idx, bytes per buffer, dirty stages, bytes to clear x4, arrival
counter]``. The buffer and the flags here are this module's own, not SGLang's
FlashInfer workspace: inside a covered step every hop is ours, and the stock steps
keep their FlashInfer workspace untouched (no shared flags).

The buffer is symmetric memory with an NVLink multicast mapping, from torch
(``_symmetric_memory.empty`` + ``rendezvous``, the package's house allocator) or from
FlashInfer's documented ``flashinfer.comm.mnnvl.McastGPUBuffer``. Creating one is
collective over the tensor-parallel group.

Fail-stop: every wait on a peer is bounded by ``deadline_s``. A wait that times out
writes the host-mapped ``ErrorRecord`` (readable after the CUDA context is lost) and
traps, so a rank that never joins a hop stops its peers instead of hanging them.
"""

from __future__ import annotations

import ctypes
import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch
import torch.distributed as dist

SGLANG_READS = {
    "kernels/jit/utils/__init__.py": ("make_cpp_args",),
}

logger = logging.getLogger(__name__)

HOP_CUH = "distributed/glm_ll_mnnvl_oneshot.cuh"  # under ll/csrc

SUPPORTED_WORLD_SIZES = (2, 4, 8)
NUM_LAMPORT_BUFFERS = 3
# FlashInfer's buffer alignment is 2 stages x 16 bytes; a page keeps every stage of
# every buffer 16-byte aligned whatever the row count.
BUFFER_ALIGN = 8192
NEG_ZERO_F32_AS_I32 = -(2**31)  # the clean Lamport word, fp32 -0.0
MAX_TOP_K = 16

# Error record: [magic, kind, site, rank | peer << 8 | row << 16, buffer index, ...]
ERR_RECORD_WORDS = 8
ERR_MAGIC = 0x474C4C48  # "GLLH", kErrMagic in the .cuh
ERR_KIND_NAMES = {
    1: "one-shot poll",
    2: "two-shot scatter poll",
    3: "two-shot broadcast poll",
    4: "arrival wait",
}
ALLOCATORS = ("symm_mem", "mcast")


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def oneshot_bytes(world: int, rows: int, hidden: int, itemsize: int = 2) -> int:
    """Lamport bytes per rank of a one-shot hop (every rank's row lands everywhere)."""
    return rows * hidden * world * itemsize


def twoshot_bytes(world: int, rows: int, hidden: int, itemsize: int = 2) -> int:
    """Lamport bytes per buffer of a two-shot hop (scatter + broadcast stages)."""
    return 2 * _ceil_div(rows, world) * world * hidden * itemsize


def buffer_bytes(world: int, max_rows: int, hidden: int) -> int:
    """Bytes per Lamport buffer serving every strategy up to ``max_rows`` rows."""
    need = max(
        oneshot_bytes(world, max_rows, hidden), twoshot_bytes(world, max_rows, hidden)
    )
    return _ceil_div(need, BUFFER_ALIGN) * BUFFER_ALIGN


# =================================================================== error record
def _cu_check(cu, err, what: str) -> None:
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{what} failed: {err}")


class ErrorRecord:
    """Host-mapped pinned words that a timed-out hop wait writes just before its
    ``trap``: the host can name the wait after the CUDA context is lost."""

    def __init__(self) -> None:
        from cuda.bindings import driver as cu

        flags = cu.CU_MEMHOSTALLOC_DEVICEMAP | cu.CU_MEMHOSTALLOC_PORTABLE
        err, host = cu.cuMemHostAlloc(4 * ERR_RECORD_WORDS, flags)
        _cu_check(cu, err, "cuMemHostAlloc")
        self.host_ptr = int(host)
        err, dptr = cu.cuMemHostGetDevicePointer(self.host_ptr, 0)
        _cu_check(cu, err, "cuMemHostGetDevicePointer")
        self.device_ptr = int(dptr)
        self.clear()

    def _view(self):
        return (ctypes.c_uint32 * ERR_RECORD_WORDS).from_address(self.host_ptr)

    def clear(self) -> None:
        v = self._view()
        for i in range(ERR_RECORD_WORDS):
            v[i] = 0

    def read(self) -> Optional[dict]:
        """None if no wait timed out, else the decoded record: the wait ``kind``, the
        caller's ``site``, the waiting ``rank``, the late ``peer``, the ``row`` it
        polled and the Lamport ``buffer`` index."""
        v = list(self._view())
        if v[0] != ERR_MAGIC:
            return None
        return {
            "kind": ERR_KIND_NAMES.get(v[1], v[1]),
            "site": v[2],
            "rank": v[3] & 0xFF,
            "peer": (v[3] >> 8) & 0xFF,
            "row": v[3] >> 16,
            "buffer": v[4],
        }


# =================================================================== the buffer
@dataclass
class LamportBuffer:
    """One rank's view of a symmetric Lamport buffer of the group (collective)."""

    world: int
    rank: int
    hidden: int
    max_rows: int
    bytes_per_buffer: int
    mcast_ptr: int
    buffer_ptrs: Tuple[int, ...]
    buffer_ptrs_dev: int
    flags: torch.Tensor  # uint32 [9]
    err: ErrorRecord
    deadline_ns: int
    allocator: str
    cpu_group: Any
    _keep: Tuple[Any, ...] = ()

    @classmethod
    def create(
        cls,
        *,
        device_group,
        cpu_group,
        max_rows: int,
        hidden: int,
        deadline_s: float,
        allocator: str = "symm_mem",
        device: Optional[torch.device] = None,
    ) -> LamportBuffer:
        """Allocate, rendezvous and initialize the buffer on every rank of
        ``device_group`` (``cpu_group``: the same ranks on gloo). Raises on every
        rank if the multicast mapping is missing on any rank."""
        if allocator not in ALLOCATORS:
            raise ValueError(f"allocator {allocator!r} is not one of {ALLOCATORS}")
        if deadline_s <= 0:
            raise ValueError("the hop deadline must be positive")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("the hop Lamport buffer is created inside graph capture")
        device = device or torch.device("cuda", torch.cuda.current_device())
        world = dist.get_world_size(device_group)
        rank = dist.get_rank(device_group)
        if world not in SUPPORTED_WORLD_SIZES:
            raise ValueError(f"hop world size {world} not in {SUPPORTED_WORLD_SIZES}")
        nbytes = buffer_bytes(world, max_rows, hidden)
        total = NUM_LAMPORT_BUFFERS * nbytes
        keep: tuple = ()
        reason = ""
        mc, ptrs, ptrs_dev = 0, (), 0
        try:
            if allocator == "symm_mem":
                import torch.distributed._symmetric_memory as symm_mem

                storage = symm_mem.empty(total, dtype=torch.uint8, device=device)
                handle = symm_mem.rendezvous(storage, device_group)
                if int(handle.world_size) != world or int(handle.rank) != rank:
                    raise RuntimeError("symmetric-memory rendezvous on another group")
                ptrs = tuple(int(p) for p in handle.buffer_ptrs)
                if ptrs[rank] != storage.data_ptr():
                    raise RuntimeError("symmetric buffer is not the allocation's base")
                mc = int(handle.multicast_ptr or 0)
                table = torch.tensor(ptrs, dtype=torch.int64, device=device)
                ptrs_dev = table.data_ptr()
                storage.view(torch.int32).fill_(NEG_ZERO_F32_AS_I32)
                keep = (storage, handle, table)
            else:
                from flashinfer.comm.mnnvl import McastGPUBuffer, TorchDistBackend

                # handles travel over the gloo group (no NCCL work at build)
                buf = McastGPUBuffer(
                    total, world, rank, device, TorchDistBackend(group=cpu_group)
                )
                if buf.buf_size < total:
                    raise RuntimeError(f"McastGPUBuffer holds {buf.buf_size} < {total}")
                ptrs = tuple(int(buf.get_unicast_ptr(r)) for r in range(world))
                mc = int(buf.get_multicast_ptr() or 0)
                ptrs_dev = int(buf.get_buffer_ptrs_dev())
                buf.lamport_initialize(rank, torch.float32)
                keep = (buf,)
            if not mc or mc % 16 or any(p % 16 for p in ptrs):
                raise RuntimeError("no 16-byte aligned NVLink multicast mapping")
        except Exception as e:  # noqa: BLE001 - gathered and raised on every rank
            reason = f"rank {rank}: {type(e).__name__}: {e}"
        gathered = [None] * world
        dist.all_gather_object(gathered, reason, group=cpu_group)
        failed = [r for r in gathered if r]
        if failed:
            raise RuntimeError(
                f"GLM LL hop Lamport buffer ({allocator}, {world} ranks, "
                f"{total} bytes) unavailable: " + "; ".join(failed)
            )
        flags = torch.tensor(
            [0, 2, nbytes, 0, 0, 0, 0, 0, 0], dtype=torch.uint32, device=device
        )
        torch.cuda.synchronize(device)
        dist.barrier(group=cpu_group)
        return cls(
            world=world,
            rank=rank,
            hidden=hidden,
            max_rows=max_rows,
            bytes_per_buffer=nbytes,
            mcast_ptr=mc,
            buffer_ptrs=ptrs,
            buffer_ptrs_dev=ptrs_dev,
            flags=flags,
            err=ErrorRecord(),
            deadline_ns=int(deadline_s * 1e9),
            allocator=allocator,
            cpu_group=cpu_group,
            _keep=keep,
        )

    def oneshot_fits(self, rows: int) -> bool:
        return oneshot_bytes(self.world, rows, self.hidden) <= self.bytes_per_buffer

    def twoshot_fits(self, rows: int) -> bool:
        return twoshot_bytes(self.world, rows, self.hidden) <= self.bytes_per_buffer

    def describe(self) -> str:
        return (
            f"{self.allocator}, {self.world} ranks, {NUM_LAMPORT_BUFFERS} x "
            f"{self.bytes_per_buffer / 2**20:.2f} MiB, deadline "
            f"{self.deadline_ns / 1e9:g} s"
        )


# =================================================================== JIT modules
def _load(name: str, cls: str, args: tuple) -> Any:
    from sgl_nvidia_mega.glm5_3.ll import jit

    return jit.load(
        name,
        HOP_CUH,
        [("run", f"{cls}::run"), ("configure", f"{cls}::configure")],
        # FlashInfer builds its MNNVL module with fast math: same flags, same bits.
        extra_cuda_cflags=["--use_fast_math"],
        marker=args,
    )


_modules: dict = {}


def _module(kind: str, world: int, trigger_early: bool, *extra) -> Any:
    from sglang.kernels.jit.utils import make_cpp_args

    key = (kind, world, trigger_early, *extra)
    mod = _modules.get(key)
    if mod is None:
        if world not in SUPPORTED_WORLD_SIZES:
            raise ValueError(f"hop world size {world}")
        args = make_cpp_args(world, trigger_early, *extra)
        cls = {
            "plain": "GlmLLMnnvlOneshot",
            "finalize": "GlmLLMnnvlFinalizeOneshot",
            "nvfp4": "GlmLLMnnvlOneshotNvfp4",
        }[kind]
        mod = _load(f"glm_ll_hop_{kind}", f"glm_ll_mnnvl::{cls}<{args}>", tuple(args))
        mod.configure()  # load every instantiation now (not in a hop, not in capture)
        _modules[key] = mod
    return mod


def plain_module(world: int, trigger_early: bool = False) -> Any:
    return _module("plain", world, trigger_early)


def finalize_module(
    world: int,
    top_k: int,
    weight_dtype: torch.dtype,
    identity_rows: bool = False,
    trigger_early: bool = False,
) -> Any:
    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"finalize top_k {top_k}")
    if weight_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"finalize weight dtype {weight_dtype}")
    return _module("finalize", world, trigger_early, top_k, weight_dtype, identity_rows)


def nvfp4_module(world: int, trigger_early: bool = False) -> Any:
    return _module("nvfp4", world, trigger_early)


# =================================================================== launches
@dataclass(frozen=True)
class Launch:
    """The launch plan of one hop (``ll.hops`` decides it from rank-invariant inputs)."""

    block: int
    cluster: int
    unicast: bool = False
    max_carveout: bool = False
    two_shot: bool = False


def _tail(buf: LamportBuffer, eps: float, launch: Launch, site: int) -> tuple:
    return (
        buf.flags,
        buf.mcast_ptr,
        buf.buffer_ptrs_dev,
        buf.rank,
        float(eps),
        0.0,  # weight_bias: the standard RMSNorm
        launch.block,
        launch.cluster,
        int(launch.unicast),
        int(launch.max_carveout),
        int(launch.two_shot),
        buf.err.device_ptr,
        buf.deadline_ns,
        int(site),
    )


def _check_rows(buf: LamportBuffer, rows: int, launch: Launch) -> None:
    fits = buf.twoshot_fits(rows) if launch.two_shot else buf.oneshot_fits(rows)
    if not fits:
        raise ValueError(
            f"{rows} rows exceed the hop buffer ({buf.max_rows} rows, "
            f"{'two' if launch.two_shot else 'one'}-shot)"
        )


def ar_rmsnorm(
    buf: LamportBuffer,
    x: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    eps: float,
    launch: Launch,
    *,
    site: int = 0,
    trigger_early: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(norm_out, residual_out)`` = RMSNorm and value of ``residual`` + the sum over
    ranks of ``x``. Every rank makes this call with the same shapes and launch."""
    _check_rows(buf, x.shape[0], launch)
    norm_out = torch.empty_like(x)
    residual_out = torch.empty_like(residual)
    plain_module(buf.world, trigger_early).run(
        x, residual, gamma, norm_out, residual_out, *_tail(buf, eps, launch, site)
    )
    return norm_out, residual_out


def finalize_ar_rmsnorm(
    buf: LamportBuffer,
    *,
    gemm2_out: torch.Tensor,
    expert_weights: torch.Tensor,
    permuted_idx: Optional[torch.Tensor],
    shared_output: Optional[torch.Tensor],
    residual: torch.Tensor,
    gamma: torch.Tensor,
    eps: float,
    launch: Launch,
    site: int = 0,
    identity_rows: bool = False,
    trigger_early: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The hop on this rank's MoE finalize: ``gemm2_out`` ``[P, >= H]`` FC2 rows
    (trtllm-gen ``do_finalize=False``), ``expert_weights`` ``[M, top_k]`` (routed
    scaling folded in), ``permuted_idx`` ``[M, top_k]`` int32 (-1: empty slot) and the
    optional shared-expert output, summed as ``moe_finalize_fuse_shared`` does."""
    _check_rows(buf, residual.shape[0], launch)
    top_k = int(expert_weights.shape[1])
    norm_out = torch.empty_like(residual)
    residual_out = torch.empty_like(residual)
    finalize_module(
        buf.world, top_k, expert_weights.dtype, identity_rows, trigger_early
    ).run(
        gemm2_out,
        None if identity_rows else permuted_idx,
        expert_weights,
        shared_output,
        residual,
        gamma,
        norm_out,
        residual_out,
        *_tail(buf, eps, launch, site),
    )
    return norm_out, residual_out


def ar_rmsnorm_nvfp4(
    buf: LamportBuffer,
    x: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    eps: float,
    launch: Launch,
    *,
    global_scale: torch.Tensor,
    site: int = 0,
    trigger_early: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """:func:`ar_rmsnorm` that also returns the NVFP4 quant of ``norm_out``:
    ``fp4`` ``[M, H / 2]`` uint8 and ``sf`` ``[M, H / 16]`` e4m3, the layout of
    ``fp4_quantize(norm_out, global_scale, is_sf_swizzled_layout=False)``."""
    rows, hidden = x.shape
    _check_rows(buf, rows, launch)
    residual_out = torch.empty_like(residual)
    norm_out = torch.empty_like(x)
    fp4 = torch.empty((rows, hidden // 2), dtype=torch.uint8, device=x.device)
    sf = torch.empty((rows, hidden // 16), dtype=torch.float8_e4m3fn, device=x.device)
    nvfp4_module(buf.world, trigger_early).run(
        x,
        residual,
        gamma,
        norm_out,
        residual_out,
        fp4,
        sf,
        global_scale.reshape(1),
        *_tail(buf, eps, launch, site),
    )
    return norm_out, residual_out, fp4, sf


def compile_all(
    world: int,
    *,
    finalize: Tuple[Tuple[int, torch.dtype], ...] = ((8, torch.float32),),
    nvfp4: bool = True,
) -> List[str]:
    """Build (and load) every module a hop family may launch; the names built."""
    built = [f"plain W{world}"]
    plain_module(world)
    for top_k, dtype in finalize:
        finalize_module(world, top_k, dtype)
        built.append(f"finalize W{world} top{top_k} {str(dtype).split('.')[-1]}")
    if nvfp4:
        nvfp4_module(world)
        built.append(f"nvfp4 W{world}")
    return built

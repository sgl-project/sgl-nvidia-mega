# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``fin2``: the Muse Spark MoE finalize fused with a TWO-SHOT all-reduce (``ops/fin2_cutedsl.py``).

Stock decode / verify runs the routed-MoE finalize (unpermute, post-expert RMSNorm, weighted top-k sum) fused with a
one-shot push all-reduce of the finalized rows (``FusedPostExpertNormWeightedSumTRTLLM`` with a
``FinalizePushWorkspace``): every rank sends its whole bf16 partial to every rank. At wide batches that exchange carries
``world - 1`` copies of the payload into every rank. This component keeps the finalize arithmetic and replaces the
exchange with a two-shot one (reduce-scatter by column slice in source-rank order, then an all-gather of the reduced
slices), through a plugin-owned arena on the tensor-parallel group.

Numerics: the finalize partial is computed by the same kernel code with the same cluster / thread configuration the
stock call would use (``_ar_launch_config``), the zero-word remap of the one-shot exchange is applied before the sum,
and the sum runs over source ranks 0..world-1 in fp32 with one bf16 rounding, as the one-shot exchange does. The
output is bitwise the stock fused path's output, except the NaN payload of a word whose two values are both the 0xFFFF
NaN (sent as two quiet NaNs; see the kernel module). ``tests/dist_fin2.py`` checks it against the stock op.

Engagement: ``MUSE_VR_FIN2=1``; only calls that the stock op makes WITH its all-reduce workspace (decode / verify,
non-SP, bf16, within the workspace capacity, where the stock model fused the all-reduce), on a 2 / 4 / 8-rank group,
with ``MUSE_VR_FIN2_MIN_M <= M <= MUSE_VR_FIN2_MAX_M`` (defaults 64 and 384) and M within the arena's row capacity.
Everything else (and any call while the arena could not be built) runs the stock op; each decision is counted in
``bed`` under ``fin2`` (``fin2:M=<m>`` engaged, ``stock:<reason>`` otherwise).
``MUSE_VR_FIN2_MAX_POLLS`` bounds every wait (default 8_000_000 polls); a wait that runs out sets the arena's sticky
error word and traps.
"""

from __future__ import annotations

import logging
import threading

import torch

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/distributed/device_communicators/custom_all_reduce_v2.py": (
        "_allocate_symmetric_memory",
    ),
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
}


logger = logging.getLogger("muse_vr_plugin")

NAME = "fin2"
OP_MODULE = "sglang.kernels.ops.communication.muse_spark_moe_finalize_all_reduce"
TARGET = OP_MODULE + ".FusedPostExpertNormWeightedSumTRTLLM.forward_cute_dsl"
# The finalize arithmetic of ops/fin2_cutedsl.py is SGLang's at this revision; the call contract is the op's.
PINS = {
    "sglang/kernels/ops/communication/muse_spark_moe_finalize_all_reduce_cutedsl.py": (
        "9e503382ec0d904476aa039d77510486cc71563c2e1a8bfcbb8d9234e7e2e689"
    ),
    "sglang/kernels/ops/communication/muse_spark_moe_finalize_all_reduce.py": (
        "b22ba8b4d895fd38c99a0835b5a37b82adaa8e7587379f09807cb072eceb9948"
    ),
    "sglang/srt/layers/communication/muse_spark_finalize_workspace.py": (
        "578c07d454011b0972ff22bb017a015d13ce03bf61fd3445497c313d0ab3de3a"
    ),
}
K_SLOTS = 3
_ALIGN = 4096
_CREDIT_BYTES = 256
_COUNTER_PAD = (
    16  # counters beyond the token blocks: one release cluster of up to 16 blocks
)

_lock = threading.Lock()
_state = {"arena": None, "arena_err": None}


def enabled() -> bool:
    return bed.enabled(NAME)


def min_m() -> int:
    return bed.env_int("MUSE_VR_FIN2_MIN_M", 64)


def max_m() -> int:
    return bed.env_int("MUSE_VR_FIN2_MAX_M", 384)


def pins_match() -> tuple[bool, str]:
    for rel, want in PINS.items():
        if not bed.pinned(rel, want):
            return False, f"{rel} differs from the pinned revision"
    return True, "ok"


def _align(n: int) -> int:
    return (n + _ALIGN - 1) // _ALIGN * _ALIGN


class Fin2Arena:
    """Two-shot exchange memory of one tensor-parallel group.

    Layout of the symmetric allocation (identical on every rank): phase-1 slots ``[kSlots][world][rows_max][H/world]``
    bf16 (empty word 0), phase-2 slots of the same shape (empty word 0xFFFFFFFF), credit words ``[world]``. Per-rank
    device state: one transfer counter per block index (``max_blocks + 16``) and a sticky error word.

    ``views``: when given (single-process emulation in tests), the list of every emulated rank's uint8 view of an
    arena-sized buffer; no collective is used. Otherwise the arena is allocated with SGLang's symmetric-memory helper on
    ``group`` (collective: every rank constructs it at the same point, outside CUDA-graph capture).
    """

    def __init__(
        self,
        group,
        device: torch.device,
        rank: int,
        world: int,
        hidden: int,
        rows_max: int,
        max_blocks: int,
        *,
        max_polls: int | None = None,
        views=None,
        counters=None,
        err=None,
    ):
        if world not in (2, 4, 8):
            raise ValueError("world size must be 2, 4 or 8")
        if hidden % (16 * world):
            raise ValueError("hidden must be a multiple of 16 * world")
        self.rank, self.world, self.hidden, self.rows_max, self.device = (
            rank,
            world,
            hidden,
            rows_max,
            device,
        )
        self.max_blocks = max_blocks
        self.num_counters = max_blocks + _COUNTER_PAD
        self.max_polls = int(
            max_polls
            if max_polls is not None
            else bed.env_int("MUSE_VR_FIN2_MAX_POLLS", 8_000_000)
        )
        self.region_bytes = _align(K_SLOTS * rows_max * hidden * 2)
        self.p1_off, self.p2_off = 0, self.region_bytes
        self.credit_off = 2 * self.region_bytes
        self.total_bytes = self.credit_off + _align(_CREDIT_BYTES)
        self.counters = (
            counters
            if counters is not None
            else torch.zeros(self.num_counters, dtype=torch.int32, device=device)
        )
        self.err = (
            err if err is not None else torch.zeros(1, dtype=torch.int32, device=device)
        )
        if views is None:
            import torch.distributed as dist
            from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
                _allocate_symmetric_memory,
            )

            self._tensor, symm = _allocate_symmetric_memory(
                self.total_bytes, device=device, group=group
            )
            self.views = [
                symm.get_buffer(r, [self.total_bytes], torch.uint8)
                for r in range(world)
            ]
            # Initial state (phase-1 slots 0, phase-2 slots 0xFFFFFFFF, credits 0, counters 0) written and complete on every
            # rank before any rank can launch a transfer: device sync, then a barrier of the group.
            self._init_own()
            torch.cuda.synchronize(device)
            dist.barrier(group=group)
        else:
            self.views = views

    def _init_own(self) -> None:
        v = self.views[self.rank]
        v[self.p1_off : self.p1_off + self.region_bytes].zero_()
        v[self.p2_off : self.p2_off + self.region_bytes].fill_(0xFF)
        v[self.credit_off : self.total_bytes].zero_()

    def arguments(self) -> tuple:
        import cutlass
        from cutlass.cute.runtime import make_ptr

        def ptr(addr: int):
            return make_ptr(
                cutlass.Uint32, addr, cutlass.cute.AddressSpace.gmem, assumed_align=16
            )

        bases = [v.data_ptr() for v in self.views]
        me = bases[self.rank]
        return (
            tuple(ptr(b + self.p1_off) for b in bases),
            tuple(ptr(b + self.p2_off) for b in bases),
            tuple(ptr(b + self.credit_off) for b in bases),
            ptr(me + self.p1_off),
            ptr(me + self.p2_off),
            ptr(me + self.credit_off),
            ptr(self.counters.data_ptr()),
            ptr(self.err.data_ptr()),
            cutlass.Int32(self.rank),
            cutlass.Int32(self.rows_max),
            cutlass.Int32(min(self.max_polls, 2**31 - 1)),
        )

    def error_word(self) -> int:
        return int(self.err.item()) & 0xFFFFFFFF


def arena(ar_workspace, hidden: int):
    """The arena on the stock workspace's tensor-parallel group (collective on first use; None when unavailable)."""
    a = _state["arena"]
    if a is not None or _state["arena_err"] is not None:
        return a
    with _lock:
        if _state["arena"] is None and _state["arena_err"] is None:
            try:
                comm = ar_workspace.communicator
                rows = min(max_m(), int(ar_workspace.max_tokens))
                blocks = int(ar_workspace.sm_count)
                _state["arena"] = Fin2Arena(
                    comm.group,
                    comm.device,
                    comm.rank,
                    comm.world_size,
                    hidden,
                    rows,
                    blocks,
                )
                bed.report(
                    NAME + ":arena",
                    True,
                    bytes=_state["arena"].total_bytes,
                    rows_max=rows,
                    blocks=blocks,
                )
            except Exception as e:  # noqa: BLE001 - stay on the stock path if the arena cannot be built
                _state["arena_err"] = f"{type(e).__name__}: {e}"
                bed.report(NAME + ":arena", False, reason=_state["arena_err"])
    return _state["arena"]


def launch(
    arena_: Fin2Arena,
    gemm2_output,
    expert_weights,
    expanded_idx_to_permuted_idx,
    rms_norm_weight,
    output,
    *,
    topk: int,
    hidden_dim: int,
    eps: float,
    weight_first: bool,
    cluster_size: int,
    threads_per_row: int,
    use_pdl: bool,
    stream=None,
) -> None:
    """One fused finalize + two-shot all-reduce call (``output`` receives the reduced rows on every rank)."""
    import cuda.bindings.driver as cuda
    from cutlass import Int32
    from cutlass.cute.runtime import make_ptr

    from .ops import fin2_cutedsl as K

    num_tokens = output.shape[0]
    if not 0 < num_tokens <= arena_.rows_max:
        raise ValueError("token count outside the arena's row capacity")
    dt = K._torch_to_cutlass_dtype[gemm2_output.dtype]
    ewt = K._torch_to_cutlass_dtype[expert_weights.dtype]
    s = stream if stream is not None else torch.cuda.current_stream()
    cs = cuda.CUstream(s.cuda_stream)
    compiled = K.get_compiled_fin2(
        dt,
        num_tokens,
        hidden_dim,
        topk,
        gemm2_output.shape[1],
        cs,
        threads_per_row=threads_per_row,
        use_pdl=use_pdl,
        eps=eps,
        weight_first=weight_first,
        expert_weights_dtype=ewt,
        cluster_size=cluster_size,
        world=arena_.world,
        max_blocks=arena_.max_blocks,
        num_counters=arena_.num_counters,
    )
    compiled(
        make_ptr(dt, gemm2_output.data_ptr()),
        make_ptr(ewt, expert_weights.view(-1).data_ptr()),
        make_ptr(Int32, expanded_idx_to_permuted_idx.view(-1).data_ptr()),
        make_ptr(dt, rms_norm_weight.data_ptr()),
        make_ptr(dt, output.data_ptr()),
        Int32(num_tokens),
        cs,
        arena_.arguments(),
    )


def forward_cute_dsl(
    original,
    self,
    gemm2_output,
    expert_weights,
    expanded_idx_to_permuted_idx,
    rms_norm_weight,
    output,
    *,
    ar_workspace=None,
):
    """AROUND ``FusedPostExpertNormWeightedSumTRTLLM.forward_cute_dsl``: the two-shot form for eligible calls."""

    def stock(reason: str):
        bed.count(NAME, "stock:" + reason)
        return original(
            self,
            gemm2_output,
            expert_weights,
            expanded_idx_to_permuted_idx,
            rms_norm_weight,
            output,
            ar_workspace=ar_workspace,
        )

    if ar_workspace is None:
        return original(
            self,
            gemm2_output,
            expert_weights,
            expanded_idx_to_permuted_idx,
            rms_norm_weight,
            output,
            ar_workspace=ar_workspace,
        )
    m = output.shape[0]
    if not min_m() <= m <= max_m():
        return stock(f"M={m}")
    if (
        ar_workspace.world_size not in (2, 4, 8)
        or gemm2_output.shape[1] % 8
        or gemm2_output.dtype != torch.bfloat16
    ):
        return stock("shape")
    if not (
        expert_weights.is_contiguous()
        and expanded_idx_to_permuted_idx.is_contiguous()
        and expanded_idx_to_permuted_idx.dtype == torch.int32
    ):
        return stock("layout")
    a = arena(ar_workspace, self.hidden_dim)
    if a is None or m > a.rows_max:
        return stock("arena")
    import importlib

    op = importlib.import_module(OP_MODULE)
    cluster, threads = op._ar_launch_config(
        m, ar_workspace.controller_count, self.cluster_size, self.hidden_dim
    )
    ar_workspace.validate(output, cluster)
    launch(
        a,
        gemm2_output,
        expert_weights,
        expanded_idx_to_permuted_idx,
        rms_norm_weight,
        output,
        topk=self.topk,
        hidden_dim=self.hidden_dim,
        eps=self.eps,
        weight_first=self.weight_first,
        cluster_size=cluster,
        threads_per_row=threads,
        use_pdl=op._should_use_pdl(m * cluster),
    )
    bed.count(NAME, f"fin2:M={m}")
    return None


def register() -> bool:
    if not enabled():
        return False
    ok, why = pins_match()
    if not ok:
        bed.report(NAME, False, reason=why)
        return False
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(TARGET, forward_cute_dsl, HookType.AROUND)
    bed.report(NAME, True, min_m=min_m(), max_m=max_m())
    return True

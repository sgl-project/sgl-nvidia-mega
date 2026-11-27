# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from TensorRT-LLM (Apache-2.0): the launch of its slot all-reduce,
# tensorrt_llm/_torch/kernels/blackwell/mnnvl_slot_allreduce_norm.py.
# This file has been modified from its original version (``bt_numerics``).

from __future__ import annotations

import functools

import torch

CTAS = 8


class _GraphSafeDlpack:
    def __init__(self, tensor):
        self._tensor = tensor.detach()

    def __dlpack__(self, stream=None):
        return self._tensor.__dlpack__(stream=-1)

    def __dlpack_device__(self):
        return self._tensor.__dlpack_device__()


def _view(t: torch.Tensor, dynamic_leading_dim=None):
    from cutlass.cute.runtime import from_dlpack

    v = from_dlpack(_GraphSafeDlpack(t), assumed_align=16)
    if dynamic_leading_dim is not None:
        v = v.mark_layout_dynamic(leading_dim=dynamic_leading_dim)
    return v


def _rows(t: torch.Tensor):
    """CuTe view of ``t`` [M, ...] / [M] with M a runtime value."""
    return _view(t, t.dim() - 1)


def _example_operands(hidden: int, device):
    rows = 2
    return (
        torch.empty((rows, hidden), dtype=torch.bfloat16, device=device),
        torch.empty((rows,), dtype=torch.bfloat16, device=device),
        torch.empty((hidden,), dtype=torch.bfloat16, device=device),
        torch.empty((1,), dtype=torch.float32, device=device),
        torch.empty((rows, hidden), dtype=torch.uint8, device=device),
    )


@functools.lru_cache(maxsize=None)
def _compiled(
    device_index: int,
    hidden: int,
    eps: float,
    nranks: int,
    use_pdl: bool,
    se_fold: bool,
    norm_out: bool,
    quant: bool,
    moe_topk: int,
    fc2_row: int,
    region_bytes: int,
    fc2_lamport: bool = False,
    ts_ctas: int = 0,
    poll_ns: int = 0,
    test_no_poll: bool = False,
    bt_numerics: bool = False,
):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from sgl_nvidia_mega.qwen3_5.sm103.hop.slot_ar_kernel import (
        MnnvlSlotAllReduceNormKernel,
    )

    kernel = MnnvlSlotAllReduceNormKernel(
        hidden=hidden,
        norm_eps=eps,
        nranks=nranks,
        ctas=CTAS,
        use_pdl=use_pdl,
        se_fold=se_fold,
        norm_out=norm_out,
        quant=quant,
        region_bytes=region_bytes,
        moe_topk=moe_topk,
        fc2_row=fc2_row,
        fc2_lamport=fc2_lamport,
        test_no_poll=test_no_poll,
        ts_ctas=ts_ctas,
        poll_ns=poll_ns,
        bt_numerics=bt_numerics,
    )
    x, gate, w, scale, q = _example_operands(hidden, torch.device("cuda", device_index))
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    z = cutlass.Int64(0)
    return cute.compile(
        kernel,
        _rows(x),
        _rows(x),
        _rows(x),
        _rows(gate),
        _rows(x),
        _view(w),
        _view(scale),
        _rows(x),
        _rows(x),
        _rows(q),
        z,
        z,
        z,
        z,
        z,
        z,
        cutlass.Int32(0),
        stream,
    )


class SlotAllReduce:
    def __init__(self, *, hidden: int, eps: float, group, device, max_rows: int):
        import torch.distributed as dist
        import torch.distributed._symmetric_memory as symm_mem

        from sgl_nvidia_mega.qwen3_5.runtime import require_multicast

        self.hidden, self.eps, self.max_rows = int(hidden), float(eps), int(max_rows)
        self.device = torch.device(device)
        self.world = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        threads = self.hidden // 8 // CTAS
        if (
            threads * 8 * CTAS != self.hidden
            or threads % 32
            or self.world not in (2, 4, 8, 16)
        ):
            raise ValueError(f"slot all-reduce: hidden {hidden} / world {self.world}")
        max_ctas = self.max_rows * CTAS
        self.region_bytes = max_ctas * self.world * threads * 16
        size = 3 * self.region_bytes
        if symm_mem.get_backend(self.device) is None:
            symm_mem.set_backend("NCCL")
        self._buf = symm_mem.empty(
            (size // 4,), dtype=torch.float32, device=self.device
        )
        self._handle = symm_mem.rendezvous(self._buf, group)
        mcast = int(self._handle.multicast_ptr or 0)
        require_multicast("fc2ar", mcast, group, self.device)
        self._buf.fill_(-0.0)
        self._state = torch.zeros(
            (max_ctas * 32,), dtype=torch.int32, device=self.device
        )
        self._one_gate = torch.full(
            (self.max_rows,), float("inf"), dtype=torch.bfloat16, device=self.device
        )
        self._unused_gate = torch.zeros((1,), dtype=torch.bfloat16, device=self.device)
        self._unused_quant = torch.zeros(
            (1, self.hidden), dtype=torch.uint8, device=self.device
        )
        self._unused_scale = torch.zeros((1,), dtype=torch.float32, device=self.device)
        self.own, self.mcast, self.ctl = (
            self._buf.data_ptr(),
            mcast,
            self._state.data_ptr(),
        )
        torch.cuda.synchronize(self.device)
        # every rank has cleared its slots before any rank can write into them
        dist.barrier(group=group)

    def supports(self, m: int) -> bool:
        return 1 <= m <= self.max_rows

    def __call__(
        self,
        residual: torch.Tensor,
        gamma: torch.Tensor,
        *,
        x: torch.Tensor | None = None,
        moe=None,
        fp8_scale: torch.Tensor | None = None,
        fc2_lamport: bool = False,
        ts_ctas: int = 0,
        poll_ns: int = 0,
        test_no_poll: bool = False,
        bt_numerics: bool = False,
    ):
        import cuda.bindings.driver as cuda
        import cutlass

        m = residual.shape[0]
        if not self.supports(m):
            raise ValueError(
                f"the slot all-reduce serves 1-{self.max_rows} rows, not {m}"
            )
        hidden = self.hidden
        topk = fc2_row = 0
        fc2_ptr = topw_ptr = perm_ptr = 0
        if fc2_lamport and moe is None:
            raise ValueError("slot all-reduce: fc2_lamport needs the MoE fold")
        if moe is not None:
            fc2, topw, perm, shared = moe
            perm = perm.reshape(m, -1)
            topw = topw.reshape(m, -1)
            if not (
                fc2.dtype == torch.bfloat16
                and fc2.dim() == 2
                and fc2.stride(1) == 1
                and fc2.shape[1] >= hidden
                and fc2.stride(0) % 8 == 0
                and fc2.data_ptr() % 16 == 0
                and topw.dtype == torch.bfloat16
                and perm.dtype == torch.int32
                and topw.shape == perm.shape
                and 0 < perm.shape[1] <= 16
                and shared is not None
                and shared.dtype == torch.bfloat16
            ):
                raise ValueError("slot all-reduce: unsupported MoE handoff")
            topk, fc2_row = perm.shape[1], fc2.stride(0)
            topw, perm = topw.contiguous(), perm.contiguous()
            fc2_ptr, topw_ptr, perm_ptr = (
                fc2.data_ptr(),
                topw.data_ptr(),
                perm.data_ptr(),
            )
            shared = shared.contiguous()
            xin, routed, gate = shared, shared, self._one_gate[:m]
        else:
            xin = routed = shared = x.contiguous()
            gate = self._unused_gate
        res_out = torch.empty_like(residual)
        norm = torch.empty_like(residual)
        quant = (
            torch.empty(
                residual.shape, dtype=torch.float8_e4m3fn, device=residual.device
            )
            if fp8_scale is not None
            else None
        )
        fn = _compiled(
            self.device.index,
            hidden,
            self.eps,
            self.world,
            True,
            moe is not None,
            True,
            fp8_scale is not None,
            topk,
            fc2_row,
            self.region_bytes,
            bool(fc2_lamport),
            int(ts_ctas),
            int(poll_ns) if fc2_lamport else 0,
            bool(fc2_lamport and test_no_poll),
            bool(bt_numerics),
        )
        fn(
            _rows(xin),
            _rows(routed),
            _rows(shared),
            _rows(gate),
            _rows(residual.contiguous()),
            _view(gamma.reshape(-1)),
            _view(
                fp8_scale.reshape(1) if fp8_scale is not None else self._unused_scale
            ),
            _rows(res_out),
            _rows(norm),
            _rows(quant.view(torch.uint8) if quant is not None else self._unused_quant),
            cutlass.Int64(self.own),
            cutlass.Int64(self.mcast),
            cutlass.Int64(self.ctl),
            cutlass.Int64(fc2_ptr),
            cutlass.Int64(topw_ptr),
            cutlass.Int64(perm_ptr),
            cutlass.Int32(self.rank),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )
        if quant is None:
            return norm, res_out
        return norm, res_out, quant

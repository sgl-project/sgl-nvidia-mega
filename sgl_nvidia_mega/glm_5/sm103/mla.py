# SPDX-License-Identifier: Apache-2.0
"""Compile MLA once; bind native weights, cache views and layer state once."""

from dataclasses import dataclass
from typing import Any

from sgl_nvidia_mega.glm_5 import KV_LORA, ROPE_DIM
from sgl_nvidia_mega.glm_5.plan import MLA_CONFIGS, ExecutionPlan
from sgl_nvidia_mega.glm_5.runtime.state import KernelState, check_binding
from sgl_nvidia_mega.glm_5.runtime.weights import MlaWeights
from sgl_nvidia_mega.glm_5.sm103 import jit

SUPPORTED = MLA_CONFIGS


@dataclass(frozen=True)
class Kernel:
    plan: ExecutionPlan
    module: Any
    metadata: jit.KernelMetadata

    @property
    def variant(self):
        return self.metadata.variant

    @property
    def native_cvt(self):
        return self.metadata.native_cvt

    def bind(
        self, weights: MlaWeights, *, kv_cache, pe_cache, layout="sglang", state=None
    ):
        import torch

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("bind MLA before CUDA-graph capture")
        if layout not in ("sglang", "legacy"):
            raise ValueError("MLA layout must be sglang or legacy")
        device = weights.x_gamma.device
        weights.validate(self.variant, device, norm_dtype=self.metadata.norm_dtype)
        _validate_cache(kv_cache, pe_cache, device, layout == "sglang")
        state = (
            state if state is not None else KernelState.allocate(self.metadata, device)
        )
        check_binding(state, self.metadata, device)
        run = self.module.bind(
            *weights.kernel_args(),
            kv_cache,
            pe_cache,
            int(layout == "sglang"),
            state.workspace,
            state.epochs,
            list(state.peer_ptrs),
            state.rank,
        )
        return BoundKernel(self, weights, state, kv_cache, pe_cache, layout, run)


@dataclass(frozen=True)
class BoundKernel:
    kernel: Kernel
    weights: MlaWeights
    state: KernelState
    kv_cache: Any
    pe_cache: Any
    layout: str
    _run: Any

    def __call__(
        self,
        x,
        *,
        selected_indices,
        selected_counts,
        cache_positions,
        rope,
        residual,
        local_output,
        final_output,
        rope_positions=None,
    ):
        if tuple(x.shape) != self.kernel.plan.shape:
            raise ValueError(f"MLA input must have shape {self.kernel.plan.shape}")
        if (rope_positions is None) != (self.layout == "legacy"):
            raise ValueError("rope_positions must match the layout fixed at binding")
        self._run(
            x,
            selected_indices,
            selected_counts,
            cache_positions,
            rope,
            cache_positions if rope_positions is None else rope_positions,
            residual,
            local_output,
            final_output,
        )


def _validate_cache(kv, pe, device, combined: bool):
    import torch

    if kv.ndim != 2 or kv.shape[1] != KV_LORA or pe.shape != (kv.shape[0], ROPE_DIM):
        raise ValueError("MLA cache views must be [rows,512] and [rows,64]")
    for tensor in (kv, pe):
        if tensor.dtype != torch.float8_e4m3fn or tensor.device != device:
            raise ValueError("MLA caches must be E4M3 on the weight device")
    if combined:
        if kv.stride() != (576, 1) or pe.stride() != (576, 1):
            raise ValueError("combined E4M3 cache views must retain row stride 576")
        if (
            kv.untyped_storage().data_ptr() != pe.untyped_storage().data_ptr()
            or pe.data_ptr() - kv.data_ptr() != 512
        ):
            raise ValueError("KV and PE must alias the same 576-byte cache row")
    elif not kv.is_contiguous() or not pe.is_contiguous():
        raise ValueError("separate MLA cache tensors must be contiguous")


def load(*, draft_tokens=4, tp_size=4, batch_size=1, norm_dtype="float32") -> Kernel:
    plan = ExecutionPlan(draft_tokens, tp_size, batch_size)
    module, metadata = jit.load(plan.variant("mla"), norm_dtype=norm_dtype)
    return Kernel(plan, module, metadata)

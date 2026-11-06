# SPDX-License-Identifier: Apache-2.0
"""Compile MoE once and bind zero-copy native weights and layer state."""

from dataclasses import dataclass
from typing import Any

from sgl_nvidia_mega.glm_5.plan import MOE_CONFIGS, ExecutionPlan
from sgl_nvidia_mega.glm_5.runtime.state import KernelState, check_binding
from sgl_nvidia_mega.glm_5.runtime.weights import MoeWeights
from sgl_nvidia_mega.glm_5.sm103 import jit

SUPPORTED = MOE_CONFIGS


@dataclass(frozen=True)
class Kernel:
    plan: ExecutionPlan
    module: Any
    metadata: jit.KernelMetadata

    @property
    def variant(self):
        return self.metadata.variant

    def bind(self, weights: MoeWeights, *, state=None):
        import torch

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("bind MoE before CUDA-graph capture")
        device = weights.rms_gamma.device
        weights.validate(self.variant, device, norm_dtype=self.metadata.norm_dtype)
        state = (
            state if state is not None else KernelState.allocate(self.metadata, device)
        )
        check_binding(state, self.metadata, device)
        run = self.module.bind(
            *weights.kernel_args(),
            state.workspace,
            state.epochs,
            list(state.peer_ptrs),
            state.rank,
        )
        return BoundKernel(self, weights, state, run)


@dataclass(frozen=True)
class BoundKernel:
    kernel: Kernel
    weights: MoeWeights
    state: KernelState
    _run: Any

    def __call__(
        self,
        x,
        *,
        residual,
        final_output,
        selected_indices,
        selected_probs,
        local_output=None,
    ):
        if tuple(x.shape) != self.kernel.plan.shape:
            raise ValueError(f"MoE input must have shape {self.kernel.plan.shape}")
        self._run(
            x, residual, local_output, final_output, selected_indices, selected_probs
        )


def load(*, draft_tokens=4, tp_size=4, batch_size=1, norm_dtype="float32") -> Kernel:
    plan = ExecutionPlan(draft_tokens, tp_size, batch_size)
    module, metadata = jit.load(plan.variant("moe"), norm_dtype=norm_dtype)
    return Kernel(plan, module, metadata)

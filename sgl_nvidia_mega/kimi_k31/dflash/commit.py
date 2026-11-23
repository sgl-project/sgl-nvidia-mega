# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""The post-verify KDA commit on the megakernel: SGLang's recurrent-state tensors, read
from its hybrid linear-attention backend, handed to ``BuiltK31Verify.kda_commit``;
``batch`` is the round's ScheduleBatch."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.layers.attention.linear.utils import pp_spec_stable_rows_enabled
from sglang.srt.runtime_context import mamba_track_grid

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch

# The SGLang names this module reads, by file under the sglang package; test/unit/test_contract.py pins them.
SGLANG_READS = {
    "srt/layers/attention/hybrid_attn_backend.py": (
        "HybridAttnBackend",
        "target_verify_backend",
    ),
    "srt/layers/attention/hybrid_linear_attn_backend.py": (
        "HybridLinearAttnBackend",
        "linear_attn_backend",
        "forward_state_slots",
    ),
    "srt/layers/attention/linear/utils.py": ("pp_spec_stable_rows_enabled",),
    "srt/mem_cache/memory_pool.py": (
        "replayssm_is_kda",
        "replayssm_request_ring",
        "get_speculative_mamba2_params_all_layers",
        "conv",
        "temporal",
        "intermediate_conv_window",
        "replayssm_rawk",
        "replayssm_rawv",
        "replayssm_g",
        "replayssm_beta",
    ),
    "srt/layers/attention/mamba/mamba2_metadata.py": ("working", "track"),
    "srt/runtime_context.py": ("mamba_track_grid",),
    "srt/managers/schedule_batch.py": (
        "ScheduleBatch",
        "mamba_track_indices",
        "tree_cache",
    ),
    "srt/mem_cache/base_prefix_cache.py": ("page_size",),
}
# _megakernel_kda_commit, _kda_commit_kwargs and _kda_commit_inputs moved here from
# SGLang (#51676), which no longer defines them; the commit reads its inputs in place.


class KdaCommit:
    """The KDA commit of one DFlash worker: ``megakernel_kda`` is the target build (its
    commit switch on), ``target_attn_backend`` the target runner's attention backend."""

    def __init__(self, *, megakernel_kda, target_attn_backend, device) -> None:
        self._megakernel_kda = megakernel_kda
        self._target_attn_backend = target_attn_backend
        self.device = device
        self._kda_confirmed = True
        self._kda_commit_buffers: Optional[torch.Tensor] = None

    def serves(self) -> bool:
        """Whether the commit kernel is laid out for the target's recurrent state;
        once, before the draft graph captures (the state pool is sized by then)."""
        state = self._kda_state()
        return state is not None and self._megakernel_kda.kda_commit_serves(state[1])

    def _megakernel_kda_commit(
        self,
        *,
        batch: ScheduleBatch,
        seq_lens_pre_verify: torch.Tensor,
        commit_lens: torch.Tensor,
    ) -> bool:
        if self._megakernel_kda is None:
            return False
        bs = int(commit_lens.shape[0])
        kwargs = self._kda_commit_kwargs(batch=batch, bs=bs)
        if kwargs is None:
            return False
        # The kernel reads the round's tensors in place (no-op casts at their dtypes):
        # nothing is copied into static inputs, as nothing captures this launch.
        used = self._megakernel_kda.kda_commit(
            **kwargs,
            commit_lens=commit_lens.to(torch.int32),
            prefix_lens=seq_lens_pre_verify.to(torch.int64),
        )
        if not used:
            # The switch is fixed at build: skip these copies from now on.
            self._megakernel_kda = None
        self._kda_confirmed = used
        return used

    def _kda_commit_kwargs(self, *, batch: ScheduleBatch, bs: int) -> Optional[dict]:
        state = self._kda_state()
        if state is None:
            return None
        linear, spec_state = state
        state_slots = linear.forward_state_slots(bs, batch.mamba_track_indices)
        # The slots are read before the kernel's griddepcontrol.wait: the forward's
        # metadata wrote them before the verify, not the launch right before this one.
        track = (
            self._kda_commit_inputs(bs)
            if state_slots.track is None
            else state_slots.track.to(torch.int32)
        )
        return dict(
            temporal=spec_state.temporal,
            rawk=spec_state.replayssm_rawk,
            rawv=spec_state.replayssm_rawv,
            gk=spec_state.replayssm_g,
            beta=spec_state.replayssm_beta,
            conv=spec_state.conv[0],
            window=spec_state.intermediate_conv_window[0],
            slots=state_slots.working.to(torch.int32),
            track_slots=track,
            track_interval=mamba_track_grid(batch.tree_cache.page_size),
        )

    def _kda_state(self):
        """(the linear-attention backend, its speculative state) when the commit's
        kernel is laid out for the target's recurrent state, else None."""
        backend = self._target_attn_backend
        if isinstance(backend, HybridAttnBackend):
            backend = backend.target_verify_backend()
        if (
            not isinstance(backend, HybridLinearAttnBackend)
            or pp_spec_stable_rows_enabled()
        ):
            return None
        linear = backend.linear_attn_backend
        state_pool = linear.state_pool
        # Host-only checkpoints index the KDA ring by req_pool_idx; this commit
        # writes it through the mamba slot.
        if not state_pool.replayssm_is_kda or state_pool.replayssm_request_ring:
            return None
        spec_state = state_pool.get_speculative_mamba2_params_all_layers()
        if len(spec_state.conv) != 1:
            return None
        return linear, spec_state

    def _kda_commit_inputs(self, bs: int) -> torch.Tensor:
        """The track slots of a batch without checkpoints: -1, filled once per size."""
        if self._kda_commit_buffers is None or self._kda_commit_buffers.numel() < bs:
            self._kda_commit_buffers = torch.full(
                (bs,), -1, dtype=torch.int32, device=self.device
            )
        return self._kda_commit_buffers[:bs]

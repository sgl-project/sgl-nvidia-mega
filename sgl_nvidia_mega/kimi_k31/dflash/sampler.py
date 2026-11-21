# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""The DFlash worker's draft sampler when the megakernel head proposes inside the draft
forward: the head's static outputs, and the staging of its sampling parameters before
each replay."""

from __future__ import annotations

from sglang.srt.speculative.dspark_components.dspark_draft import resolve_greedy_mask

# The SGLang names this module reads, by file under the sglang package; test/unit/test_contract.py pins them.
SGLANG_READS = {
    "srt/speculative/dspark_components/dspark_draft.py": ("resolve_greedy_mask",),
    "srt/sampling/sampling_batch_info.py": ("is_all_greedy", "temperatures"),
}
# SGLang's helper this module copies; it stays in SGLang.
SGLANG_MIRRORS = {
    "srt/speculative/dflash_worker_v2.py:_is_all_greedy": "4fc30cefaa3e17c8",
}
# SGLang code moved here (test/unit/test_contract.py): while SGLang still defines it,
# its digest must match, so both copies run the same code.
SGLANG_MOVED = {
    "srt/speculative/dflash_worker_v2.py:_MegakernelSelectorDraftSampler.__init__": "181121a25a37f792",
    "srt/speculative/dflash_worker_v2.py:_MegakernelSelectorDraftSampler.stage_sampling_params": "06b66b272012e6a2",
    "srt/speculative/dflash_worker_v2.py:_MegakernelSelectorDraftSampler.__call__": "d5fdedf146edcd90",
}


class MegakernelSelectorDraftSampler:
    """The NVIDIA megakernel head proposes inside the draft forward."""

    def __init__(self, *, head, out, candidate_out, q_out, sampling_enabled):
        self.out, self.candidate_out, self.q_out = out, candidate_out, q_out
        self.sampling_enabled = sampling_enabled
        # Staging goes to the megakernel head's own buffers, not ones sized here.
        self.max_bs = None
        self._head = head
        self._staged_greedy = False

    def stage_sampling_params(self, *, bs: int, sampling_info) -> None:
        if not self.sampling_enabled:
            return
        if _is_all_greedy(sampling_info):
            if not self._staged_greedy:
                self._head.stage_draft_head_sampling(
                    temperatures=None, greedy_mask=None
                )
                self._staged_greedy = True
            return
        self._head.stage_draft_head_sampling(
            temperatures=sampling_info.temperatures.view(-1)[:bs],
            greedy_mask=resolve_greedy_mask(
                bs=bs, sampling_info=sampling_info, device=self.q_out.device
            )[:bs],
        )
        self._staged_greedy = False

    def __call__(self, hidden_states, input_ids):
        pass


def _is_all_greedy(sampling_info) -> bool:
    return sampling_info is None or sampling_info.is_all_greedy

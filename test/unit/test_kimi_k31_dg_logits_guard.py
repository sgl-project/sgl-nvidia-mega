# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""dg_logits.build() refuses a vendored logits header that lost its PDL synchronization (CPU only).

Each mutant below is a header that still compiles and still gives bitwise logits, but whose launch is either no longer
fenced (a TMA warp may read the predecessors' plain stores unordered), no longer triggers its dependents, triggers them
without the fence.acq_rel.gpu that orders the dependents' pre-wait reads after the stores this grid waited for, or leaves
warp 0 unconverged after thread 0's trigger branch (its next instruction reconfigures registers warp-wide)."""

import pytest

from sgl_nvidia_mega.kimi_k31.dsa import dg_logits

WAIT = "cudaGridDependencySynchronize();"
TRIGGER = r'asm volatile("fence.acq_rel.gpu;\n\tgriddepcontrol.launch_dependents;" ::: "memory");'
SYNCWARP = "    __syncwarp();\n"
BLOCK = "    if (threadIdx.x == 0) {\n        " + TRIGGER + "\n    }\n" + SYNCWARP
UNFENCED_TRIGGER = "cudaTriggerProgrammaticLaunchCompletion();"
FENCE_LINE = '        asm volatile("fence.proxy.async.global;" ::: "memory");\n'
MMA_BRANCH = "    } else if (warp_idx == kSpecWarpStart + 2) {\n"


def shipped() -> str:
    return dg_logits.HEADER.read_text()


def drop_fence(text: str, which: int) -> str:
    """Remove the ``which``-th (0 = q / weights branch, 1 = index-K branch) fence statement."""
    parts = text.split(FENCE_LINE)
    assert len(parts) == 3, "the shipped header has two fence lines"
    kept = ["" if i == which else FENCE_LINE for i in range(2)]
    return parts[0] + kept[0] + parts[1] + kept[1] + parts[2]


def fence_before_wait(text: str) -> str:
    """Move the q / weights branch's fence in front of the wait: still two fences, but the TMA branch no longer starts with one."""
    return drop_fence(text, 0).replace(f"    {WAIT}\n", FENCE_LINE + f"    {WAIT}\n", 1)


MUTANTS = {
    "no_fence_q_branch": lambda t: drop_fence(t, 0),
    "no_fence_index_k_branch": lambda t: drop_fence(t, 1),
    "no_trigger": lambda t: t.replace(f"        {TRIGGER}\n", "", 1),
    "trigger_before_wait": lambda t: t.replace(f"    {WAIT}\n", "", 1).replace(BLOCK, BLOCK + f"    {WAIT}\n", 1),
    "fence_before_wait": fence_before_wait,
    "unfenced_trigger": lambda t: t.replace(f"        {TRIGGER}\n", f"        {UNFENCED_TRIGGER}\n", 1),
    "fence_after_trigger": lambda t: t.replace(
        TRIGGER, r'asm volatile("griddepcontrol.launch_dependents;\n\tfence.acq_rel.gpu;" ::: "memory");', 1),
    "cta_scope_fence": lambda t: t.replace(TRIGGER, TRIGGER.replace("fence.acq_rel.gpu", "fence.acq_rel.cta"), 1),
    "extra_unfenced_trigger": lambda t: t.replace(MMA_BRANCH, MMA_BRANCH + f"        {UNFENCED_TRIGGER}\n", 1),
    "no_syncwarp_after_trigger": lambda t: t.replace(BLOCK, BLOCK[: -len(SYNCWARP)], 1),
    "tma_copy_in_mma_branch": lambda t: t.replace(
        MMA_BRANCH, MMA_BRANCH + "        tma::copy<kHeadDim, kPageKV, 0>(&tensor_map_kv, nullptr, nullptr, 0, 0);\n", 1),
}


def test_shipped_header_passes():
    dg_logits.check_header(shipped())


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_mutant_header_is_refused(name):
    mutant = MUTANTS[name](shipped())
    assert mutant != shipped(), f"mutant {name} did not change the header"
    with pytest.raises(RuntimeError, match="lost its PDL synchronization"):
        dg_logits.check_header(mutant)


def test_build_checks_the_header_before_compiling(tmp_path, monkeypatch):
    header = tmp_path / dg_logits.HEADER.name
    header.write_text(drop_fence(shipped(), 1))
    monkeypatch.setattr(dg_logits, "HEADER", header)
    monkeypatch.setattr(dg_logits, "compile_unit", lambda *a, **k: pytest.fail("build() compiled an unchecked header"))
    with pytest.raises(RuntimeError, match="lost its PDL synchronization"):
        dg_logits.build.__wrapped__(0, 8)

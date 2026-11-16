# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CAP-128 #4: the MoE tail box takes the boot's rows, and the prepared MoE refuses a grown tail box while
the experts still publish with a 64-row slot stride.

- test_guard_refuses_grown_publish: a 128-row tail box with published batches -> ValueError before any
  experts.prepare call (the experts would write partials at the 64-row stride into 128-row slots).
- test_guard_keeps_64_row_boots: the same wiring at the 64-row box prepares every batch (no change for today's boots).
- test_tail_weights_refuse_batch_over_cap: MoeTailWeights refuses a batch larger than its box's rows.
- test_wave_tail_smem_covers_its_plan: every WaveMoeTail form up to 128 rows allocates the shared memory it addresses.
- test_wave_tail_square_rows_fit_the_butterfly: every form's per-row group count (``padded``) is a power of two that
  holds its groups, and scatter_sums refuses a count its butterfly would drop (12 values: 6 -> 3 -> 1 loses one).
  The 65-96-row two-shot tail had 9-12 groups padded to 12, and the RMS of every token past the first 16 was wrong.
- test_wave_tail_group_sums_reach_their_tokens: for every form past one group (rows 9-128, ragged sizes included), the
  module's scatter_sums + gather_sums, run on 32 emulated lanes (shfl.bfly = lane ^ offset, shfl.idx = source lane), hand
  every lane each group's sum over the warp (squares exchange, width 32) and over the 16 slices (normalize, width 16).
  Expected sums are numpy sums of integer-valued inputs (exact in any order), not the code's.
- test_wave_tail_every_line_has_a_writer: every owned output line of every form has a writer warp (TP8 above 64 rows has
  9-16 lines per CTA and 8 warps).
Mutations (must fail): cap4_noguard (the prepared MoE's guard removed); tail_smem_fixed (the mma.sync form's allocation
fixed at 116 KB); tail_padded_v4 (groups padded to a multiple of 4 again); scatter_noguard (scatter_sums' count check removed);
tail_padded_v4_noguard (both: the code before the fix, which gave tokens past 16 another group's RMS at 65-96 rows);
tail_line_warp_nomod (line s written by warp owner_warp + s, none for s >= 8).

The first three tests need one GPU (PreparedMoE reads the device properties; MoeTailWeights builds TMA maps); the
shared-memory test runs on CPU."""

import os
from types import SimpleNamespace

import pytest
import torch

import _harness as H

MUTATIONS = {
    # (W8 Breq-16 port onto 0ed81a7: the guard there is moe 5cefcae's slot-stride form, so the mutation names that line)
    "cap4_noguard": ("kimi_k31/k3/cutedsl/moe/prepared.py", "if tail.publish and slot_rows != tail.tail_box.cap:",
                     "if False:"),
    # the mma.sync WaveMoeTail's allocation fixed at 116 KB whatever its plan
    "tail_smem_fixed": ("kimi_k31/k3/cutedsl/moe/tail.py",
                        "        self.smem = max(116 * 1024, self.offsets[4] + 40)\n        # ``umma`` (past one group)",
                        "        self.smem = 116 * 1024\n        # ``umma`` (past one group)"),
    # the groups' square rows padded to a multiple of 4 (12 for 9-12 groups) instead of a power of two
    "tail_padded_v4": ("kimi_k31/k3/cutedsl/moe/tail.py", "        self.padded = max(4, 1 << (self.groups - 1).bit_length())",
                       "        self.padded = -(-self.groups // 4) * 4"),
    # scatter_sums accepts any count
    "scatter_noguard": ("kimi_k31/k3/cutedsl/moe/tail.py", "    if len(values) & (len(values) - 1) or len(values) > width:",
                        "    if False:"),
    # both of the above: the tail before the fix
    "tail_padded_v4_noguard": [
        ("kimi_k31/k3/cutedsl/moe/tail.py", "        self.padded = max(4, 1 << (self.groups - 1).bit_length())",
         "        self.padded = -(-self.groups // 4) * 4"),
        ("kimi_k31/k3/cutedsl/moe/tail.py", "    if len(values) & (len(values) - 1) or len(values) > width:", "    if False:"),
    ],
    # owned line s written by warp owner_warp + s only (no warp for s >= 8: TP8 above 64 rows)
    "tail_line_warp_nomod": ("kimi_k31/k3/cutedsl/moe/tail.py", "        return (self.owner_warp + s) % (self.threads // 32)",
                             "        return self.owner_warp + s"),
}


def _wiring(cap, publish, batches):
    calls = []
    tail = SimpleNamespace(tail_box=SimpleNamespace(cap=cap), publish=publish, twoshots=set())
    front = SimpleNamespace(split={}, planned=lambda b: False)
    experts = SimpleNamespace(grids={}, grid_for=lambda b: 1, prepare=lambda *a: calls.append(a[0]))
    device = torch.device("cuda", 0)
    state = SimpleNamespace(tp=16, batches=tuple(batches), device=device, views={b: (None, None, None) for b in batches})
    return front, experts, tail, state, calls


def test_guard_refuses_grown_publish():
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import prepared as PM

    front, experts, tail, state, calls = _wiring(128, {16: (0, 0, 0), 128: (0, 0, 0)}, (8, 16, 128))
    with pytest.raises(ValueError, match="slot stride"):
        PM.PreparedMoE(front, experts, tail, state)
    assert calls == [], f"experts prepared before the refusal: {calls}"


def test_guard_keeps_64_row_boots():
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import prepared as PM

    front, experts, tail, state, calls = _wiring(64, {16: (0, 0, 0), 64: (0, 0, 0)}, (8, 16, 64))
    PM.PreparedMoE(front, experts, tail, state)
    assert calls == [8, 16, 64]


def test_tail_weights_refuse_batch_over_cap():
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import tail_runtime as TR

    dev = torch.device("cuda", 0)
    up = torch.zeros((7168 // 16, 3584), dtype=torch.bfloat16, device=dev)
    gamma = torch.ones(3584, dtype=torch.bfloat16, device=dev)
    with pytest.raises(ValueError, match="exceeds the tail box"):
        TR.MoeTailWeights(16, up, gamma, 1e-6, 1, 6, (8, 128), (), 0, cap=TR.CAP)


def test_wave_tail_smem_covers_its_plan():
    # WaveMoeTail.kernel's last shared-memory views are the barriers (4 x u64 at offsets[4]) and the TMEM slot (u32 at
    # offsets[4] + 32), so its allocation must reach offsets[4] + 36 bytes, in every form a published batch can take up to
    # the 128-row box (one-shot / two-shot, mma.sync / tcgen05). Constructor only: no GPU, nothing compiled.
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import tail as T
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import tail_runtime as TR

    short = []
    checked = 0
    for tp in (8, 16):
        for rows in list(range(1, 65)) + [72, 80, 88, 96, 104, 112, 120, 128]:
            if not TR.published(tp, rows):
                continue
            for twoshot, umma in sorted({(False, False), (rows > 8, False), (rows > 8, rows > 8)}):
                t = T.WaveMoeTail(rows, tp=tp, twoshot=twoshot, umma=umma)
                checked += 1
                if t.smem < max(116 * 1024, t.offsets[4] + 36):
                    short.append((tp, rows, twoshot, umma, t.offsets[4] + 36, t.smem))
    assert checked > 300, checked
    assert not short, f"{len(short)} forms allocate less shared memory than they address, e.g. {short[:3]}"


def test_wave_tail_square_rows_fit_the_butterfly():
    # Past one group, every (warp, slice) row of squares holds ``padded`` sums that scatter_sums reduces with a butterfly
    # halving its list per level, so ``padded`` must be a power of two >= the group count (and >= 4: v4 rows). Expected
    # values from the rule, not from the code: groups = ceil(rows / 8), padded = the next power of two >= max(4, groups).
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import tail as T
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import tail_runtime as TR

    wrong = []
    checked = 0
    for tp in (8, 16):
        for rows in range(1, 129):
            if not TR.published(tp, rows):
                continue
            for twoshot, umma in sorted({(False, False), (rows > 8, False), (rows > 8, rows > 8)}):
                t = T.WaveMoeTail(rows, tp=tp, twoshot=twoshot, umma=umma)
                checked += 1
                groups = -(-rows // 8)
                want = 4 if groups <= 4 else 8 if groups <= 8 else 16
                if t.padded != want:
                    wrong.append((tp, rows, twoshot, umma, t.padded, want))
    assert checked > 300, checked
    assert not wrong, f"{len(wrong)} forms pad their square rows wrongly, e.g. {wrong[:3]}"
    with pytest.raises(ValueError, match="power of two"):
        T.scatter_sums([0.0] * 12, 0, 16)


def test_wave_tail_group_sums_reach_their_tokens():
    # exchange_squares: values = the groups' warp sums padded with zeros to ``padded``, scatter_sums(width 32), then
    # gather_sums(total, padded, 32, padded) must give every lane group g's total over the 32 lanes. normalize_groups: the
    # row's ``padded`` sums (lanes 16-31 load lanes 0-15's rows), scatter_sums(width 16), then gather_sums(.., groups, 16,
    # padded) must give every lane group g's total over the 16 slices (the RMS of token g's group is taken from it).
    import numpy as np

    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import tail as T
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import tail_runtime as TR

    forms = []  # built with the real modules; only the two butterfly helpers run on emulated lanes
    for tp in (8, 16):
        for rows in range(9, 129):
            if not TR.published(tp, rows):
                continue
            for twoshot, umma in sorted({(False, False), (True, False), (True, True)}):
                t = T.WaveMoeTail(rows, tp=tp, twoshot=twoshot, umma=umma)
                assert t.groups == -(-rows // 8), (tp, rows, t.groups)
                forms.append((tp, rows, twoshot, umma, t.groups, t.padded))
    ragged = {f[1] for f in forms if 64 < f[1] < 128}
    assert len(forms) > 300 and {65, 72, 96, 97, 127} <= ragged, (len(forms), sorted(ragged)[:5])

    lane = np.arange(32)
    emulated = {
        "Float32": lambda v: v,
        "cutlass": SimpleNamespace(select_=np.where),
        "cute": SimpleNamespace(arch=SimpleNamespace(shuffle_sync_bfly=lambda v, off: v[lane ^ off])),
        "shuffle": lambda v, src: np.full(32, v[src & 31]),  # shfl.idx: source lane mod 32
    }
    saved = {k: getattr(T, k) for k in emulated}
    try:
        for k, v in emulated.items():
            setattr(T, k, v)
        wrong = _check_group_sums(T, np, lane, forms)
    finally:
        for k, v in saved.items():
            setattr(T, k, v)
    assert not wrong, f"{len(wrong)} of {len(forms)} forms hand a token another group's sum, e.g. {wrong[:3]}"


def _check_group_sums(T, np, lane, forms):
    rng = np.random.default_rng(7)
    wrong = []
    for tp, rows, twoshot, umma, groups, padded in forms:
        form = (tp, rows, twoshot, umma)
        # exchange: integer-valued sums, exact in any order
        warp = [rng.integers(0, 1 << 12, 32).astype(np.float64) for _ in range(groups)]
        try:
            got = T.gather_sums(T.scatter_sums(warp + [np.zeros(32)] * (padded - groups), lane, 32), padded, 32, padded)
        except (ValueError, IndexError) as e:
            wrong.append((*form, "exchange", repr(e)[:80]))
            continue
        bad = [g for g in range(groups) if not np.array_equal(got[g], np.full(32, warp[g].sum()))]
        if bad:
            wrong.append((*form, "exchange groups", bad))
            continue
        # normalize: lanes 16-31 hold lanes 0-15's row
        half = [rng.integers(0, 1 << 12, 16).astype(np.float64) for _ in range(groups)]
        sums = [np.tile(h, 2) for h in half] + [np.zeros(32)] * (padded - groups)
        try:
            got = T.gather_sums(T.scatter_sums(sums, lane, 16), groups, 16, padded)
        except (ValueError, IndexError) as e:
            wrong.append((*form, "normalize", repr(e)[:80]))
            continue
        bad = [g for g in range(groups) if not np.array_equal(got[g], np.full(32, half[g].sum()))]
        if bad:
            wrong.append((*form, "normalize groups", bad))
    return wrong


def test_wave_tail_every_line_has_a_writer():
    # Each owned output line s (slots = ceil(rows x lines / 16) per CTA; TP8 has 2 lines per token) must be written by one of
    # the CTA's 8 warps, else its rows are never multicast and the consumer's deadline fail-stops (cap4r_a_r1 / ima_r1: TP8
    # 65-128 rows). Expected: every s maps into warps 0..7, and the 8-or-fewer-slot forms keep warp owner_warp + s.
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import tail as T
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import tail_runtime as TR

    bad, checked, over8 = [], 0, 0
    for tp in (8, 16):
        lines = 2 if tp == 8 else 1
        for rows in range(1, 129):
            if not TR.published(tp, rows):
                continue
            for twoshot, umma in sorted({(False, False), (rows > 8, False), (rows > 8, rows > 8)}):
                t = T.WaveMoeTail(rows, tp=tp, twoshot=twoshot, umma=umma)
                checked += 1
                slots = -(-rows * lines // 16)
                writers = [t.line_warp(s) for s in range(slots)]
                over8 += slots > 8
                if t.slots != slots or any(not 0 <= w < 8 for w in writers) or (slots <= 8 and writers != list(range(slots))):
                    bad.append((tp, rows, twoshot, umma, t.slots, writers))
    assert checked > 300 and over8 > 0, (checked, over8)
    assert not bad, f"{len(bad)} forms leave an owned line without a writer warp, e.g. {bad[:2]}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"] + ([] if not os.environ.get("K31_MUTATION") else ["-x"])))

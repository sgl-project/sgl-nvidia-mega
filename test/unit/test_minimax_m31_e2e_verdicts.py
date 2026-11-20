# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU checks of the rules the MiniMax-M3.1 end-to-end pair judges by
(test/e2e/minimax_m31_verdict.py), on hand-made inputs: the paired accuracy rule, the
two speed rules, the tables the tests log, and the reading of a server log for the path
an arm ran, on lines rendered from the installed SGLang's own format strings. The
module is loaded by path and must import nothing but the standard library, so these
checks need neither SGLang's GPU test utilities nor the end-to-end tests' servers."""

import ast
import importlib.util
import sys
from pathlib import Path

import pytest
import sglang

E2E = Path(__file__).resolve().parents[1] / "e2e"
SOURCE = E2E / "minimax_m31_verdict.py"
_spec = importlib.util.spec_from_file_location("minimax_m31_verdict", SOURCE)
verdict = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = verdict
_spec.loader.exec_module(verdict)

VERIFY, DRAFT = "MiniMax-M3.1", "MiniMax-M3.1 DSpark draft"
SPECS = {VERIFY: (1, 2, 16), DRAFT: (1,)}
RANKS = 4


def test_the_rules_import_only_the_standard_library():
    imported = {
        (node.module if isinstance(node, ast.ImportFrom) else alias.name).split(".")[0]
        for node in ast.walk(ast.parse(SOURCE.read_text()))
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imported <= set(sys.stdlib_module_names), imported


def _answers(both, b, c, neither):
    """(baseline, candidate): `both` questions both arms answer, `b` only the
    baseline, `c` only the candidate, `neither` no arm."""
    baseline = [True] * (both + b) + [False] * (c + neither)
    candidate = [True] * both + [False] * b + [True] * c + [False] * neither
    return baseline, candidate


@pytest.mark.parametrize(
    "b, c, bound, expected",
    [
        (0, 0, 2, "pass"),
        (5, 0, 5, "pass"),
        (6, 0, 5, "fail"),  # six questions lost, none gained
        (9, 3, 7, "pass"),
        (10, 2, 7, "fail"),
        (0, 30, 11, "pass"),  # one-sided: a better candidate passes
    ],
)
def test_the_paired_rule_bounds_the_net_loss_by_the_discordant_count(
    b, c, bound, expected
):
    v = verdict.accuracy_verdict(*_answers(150, b, c, 10))
    assert (v.questions, v.b, v.c, v.bound) == (160 + b + c, b, c, bound)
    assert (v.baseline, v.candidate) == (150 + b, 150 + c)
    assert v.verdict == expected


def test_the_accuracy_rule_has_no_absolute_score():
    # 5% on both arms passes; 90.5% fails against a baseline that lost 14 net
    assert verdict.accuracy_verdict(*_answers(10, 0, 0, 190)).verdict == "pass"
    high = verdict.accuracy_verdict(*_answers(180, 15, 1, 4))
    assert high.candidate / high.questions > 0.9
    assert (high.b - high.c, high.bound, high.verdict) == (14, 8, "fail")


def test_the_accuracy_rule_pairs_by_question():
    # the same two scores: the same questions wrong, or disjoint ones
    same = verdict.accuracy_verdict(
        [True, True, False, False], [True, True, False, False]
    )
    disjoint = verdict.accuracy_verdict(
        [True, True, False, False], [False, False, True, True]
    )
    assert (same.baseline, same.candidate) == (disjoint.baseline, disjoint.candidate)
    assert (same.b, same.c) == (0, 0)
    assert (disjoint.b, disjoint.c) == (2, 2)


def test_a_baseline_with_no_correct_answer_is_unavailable_coverage():
    assert verdict.accuracy_verdict([False] * 8, [False] * 8).verdict == "unavailable"
    assert verdict.accuracy_verdict([False] * 8, [True] * 8).verdict == "unavailable"


@pytest.mark.parametrize("baseline, candidate", [([], []), ([True], [True, False])])
def test_the_accuracy_rule_refuses_arms_on_different_questions(baseline, candidate):
    with pytest.raises(ValueError, match="not the same non-empty set"):
        verdict.accuracy_verdict(baseline, candidate)


def test_an_arms_spread_is_half_its_range_over_its_mean():
    v = verdict.speed_verdict((98, 100, 102), (490, 500, 510), fused=True)
    assert v.baseline_mean == 100 and v.candidate_mean == 500
    assert v.baseline_spread == pytest.approx(0.02)
    assert v.candidate_spread == pytest.approx(0.02)
    assert v.change == pytest.approx(4.0)


@pytest.mark.parametrize(
    "baseline, candidate, expected",
    [
        ((690, 700, 680), (915, 925, 905), "pass"),
        # a higher mean whose slowest pass is inside the baseline's range
        ((680, 690, 720), (715, 760, 770), "fail"),
        ((680, 690, 720), (720, 760, 770), "fail"),  # equal is not faster
        ((680, 690, 720), (721, 760, 770), "pass"),
    ],
)
def test_the_fused_path_must_beat_the_baselines_fastest_pass(
    baseline, candidate, expected
):
    assert verdict.speed_verdict(baseline, candidate, fused=True).verdict == expected


@pytest.mark.parametrize(
    "baseline, candidate, tolerance, expected",
    [
        ((571, 575, 567), (585, 580, 590), 0.03, "pass"),  # faster
        ((100, 100, 100), (97.1, 97.1, 97.1), 0.03, "pass"),  # inside the 3% floor
        ((100, 100, 100), (96.9, 96.9, 96.9), 0.03, "fail"),
        # 5% slower: outside tight passes, inside passes that spread +-3% each
        ((99, 100, 101), (94, 95, 96), 0.03, "fail"),
        ((97, 100, 103), (92.15, 95, 97.85), 0.06, "pass"),
        ((97, 100, 103), (90.21, 93, 95.79), 0.06, "fail"),
    ],
)
def test_a_fallback_width_may_drop_by_3_percent_or_the_arms_spreads(
    baseline, candidate, tolerance, expected
):
    v = verdict.speed_verdict(baseline, candidate, fused=False)
    assert v.tolerance == pytest.approx(tolerance)
    assert v.verdict == expected


def test_the_two_speed_rules_differ_on_the_same_passes():
    # 1% faster with overlapping passes: no regression, and no speed claim either
    passes = ((98, 100, 102), (99, 101, 103))
    assert verdict.speed_verdict(*passes, fused=False).verdict == "pass"
    assert verdict.speed_verdict(*passes, fused=True).verdict == "fail"


@pytest.mark.parametrize(
    "baseline, candidate, message",
    [
        ((100, 100, 100), (100,), "the candidate has 1 passes"),
        ((100, 0, 100), (100, 100, 100), "baseline's passes are not all measured"),
    ],
)
def test_the_speed_rules_refuse_missing_passes(baseline, candidate, message):
    with pytest.raises(ValueError, match=message):
        verdict.speed_verdict(baseline, candidate, fused=False)


def test_the_tables_carry_both_arms_the_difference_and_the_verdict():
    accuracy = verdict.accuracy_row(
        "1 at a time", verdict.accuracy_verdict(*_answers(190, 3, 5, 2))
    )
    assert accuracy == (
        "1 at a time", "193/200 = 0.965", "195/200 = 0.975", 3, 5,
        "b - c = -2 <= 6", "pass",
    )  # fmt: skip
    speed = verdict.speed_row(
        16, verdict.speed_verdict((98, 100, 102), (94, 95, 96), False)
    )
    assert speed == (
        16, "100.0", "95.0", "-5.0%", "+-2.0%", "+-1.1%", "change >= -3.1%", "fail",
    )  # fmt: skip
    fused = verdict.speed_row(
        1, verdict.speed_verdict((680, 690, 700), (905, 915, 925), True)
    )
    assert fused[-2:] == ("slowest pass 905.0 > native's fastest 700.0", "pass")
    lines = verdict.table(verdict.SPEED_HEADER, [speed, fused]).splitlines()
    assert len(lines) == 4 and len({len(line.rstrip()) for line in lines[:2]}) == 1
    assert len(verdict.ACCURACY_HEADER) == len(accuracy)
    assert len(verdict.SPEED_HEADER) == len(speed)


# The log lines the engagement check reads, as the installed SGLang formats them
ROOT = Path(sglang.__file__).resolve().parent
BINDING = "srt/kernel_plugins/nvidia/binding.py"


def _sglang_format(start: str) -> str:
    """The one string constant of SGLang's megakernel binding that starts this way."""
    path = ROOT / BINDING
    found = [
        node.value
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith(start)
    ]
    assert len(found) == 1, f"{path}: {len(found)} strings start {start!r}"
    return found[0]


def _line(rank, message, pid=4000):
    # SGLang's scheduler prefix: "[<time>{ DP<n>} TP<rank> PID<pid>] <message>"
    return f"[2026-11-18 10:00:0{rank} TP{rank} PID{pid + rank}] {message}"


def _active(rank, name, summary):
    text = _sglang_format("NVIDIA megakernel: active (") % (
        name, summary, [8, 16], "tolerance", 31.2, 1024.0, 60.0,
    )  # fmt: skip
    return _line(rank, text)


def _graphs(rank, name, sizes, captured):
    form = _sglang_format("NVIDIA megakernel: %s runs the decode graphs")
    return _line(rank, form % (name, sizes, len(sizes), captured))


def _candidate_log(ranks=range(RANKS), verify_sizes=None, draft_sizes=(1,)):
    verify_sizes = list(range(1, 17)) if verify_sizes is None else verify_sizes
    lines = [_line(0, "Load weight begin."), "command=sglang serve --tp 4"]
    for rank in ranks:
        lines += [
            _active(rank, VERIFY, "one request's 8-row DSpark verify: MegaAttention"),
            _active(
                rank, DRAFT, "DSpark draft block of 7 rows: proposal: Markov chain"
            ),
            _graphs(rank, VERIFY, verify_sizes, 16),
            _graphs(rank, DRAFT, list(draft_sizes), 16),
        ]
    return "\n".join(lines) + "\n"


def _baseline_log(ranks=range(RANKS)):
    lines = ["command=sglang serve --tp 4"]
    lines += [_line(rank, "Load weight begin.") for rank in ranks]
    lines += [_line(rank, "Capture cuda graph end.") for rank in ranks]
    return "\n".join(lines) + "\n"


def test_the_candidates_log_is_read_as_sglang_writes_it():
    log = _candidate_log()
    assert verdict.active_ranks(log) == {VERIFY: {0, 1, 2, 3}, DRAFT: {0, 1, 2, 3}}
    assert verdict.graph_batch_sizes(log) == {VERIFY: list(range(1, 17)), DRAFT: [1]}
    assert verdict.candidate_reasons(log, RANKS, SPECS) == []
    # a data-parallel tag before the rank's, milliseconds in the time
    tagged = log.replace(" TP", ".123 DP0 TP")
    assert verdict.candidate_reasons(tagged, RANKS, SPECS) == []


def test_a_candidate_whose_megakernel_did_not_serve_is_named():
    # another plugin took the flag: a server that is up, with no line of this package
    reasons = verdict.candidate_reasons(_baseline_log(), RANKS, SPECS)
    assert reasons == [
        f"{VERIFY}: no 'active' line on ranks [0, 1, 2, 3]",
        f"{VERIFY}: no decode-graph coverage line",
        f"{DRAFT}: no 'active' line on ranks [0, 1, 2, 3]",
        f"{DRAFT}: no decode-graph coverage line",
    ]
    one_rank_short = verdict.candidate_reasons(
        _candidate_log(ranks=(0, 1, 3)), RANKS, SPECS
    )
    assert one_rank_short == [
        f"{VERIFY}: no 'active' line on ranks [2]",
        f"{DRAFT}: no 'active' line on ranks [2]",
    ]


@pytest.mark.parametrize(
    "verify_sizes, draft_sizes, spec",
    [
        ([2, 4, 8, 16], (1,), VERIFY),  # the one-request graph runs SGLang's forward
        (list(range(1, 33)), (1,), VERIFY),  # past what one server covers
        (None, (1, 2), DRAFT),
    ],
)
def test_a_candidate_whose_graphs_miss_a_width_is_named(
    verify_sizes, draft_sizes, spec
):
    log = _candidate_log(verify_sizes=verify_sizes, draft_sizes=draft_sizes)
    (reason,) = verdict.candidate_reasons(log, RANKS, SPECS)
    assert reason.startswith(f"{spec}: runs the decode graphs of batch sizes")


def test_the_baselines_log_holds_every_rank_and_no_megakernel_line():
    assert verdict.baseline_reasons(_baseline_log(), RANKS) == []
    (reason,) = verdict.baseline_reasons(_candidate_log(), RANKS)
    assert reason.startswith("16 megakernel lines, the first: ")
    # a log that was not captured proves nothing about the baseline
    assert verdict.baseline_reasons("", RANKS) == [
        "the log holds no line of ranks [0, 1, 2, 3]: nothing to judge"
    ]
    assert verdict.baseline_reasons(_baseline_log(ranks=(0, 1)), RANKS) == [
        "the log holds no line of ranks [2, 3]: nothing to judge"
    ]

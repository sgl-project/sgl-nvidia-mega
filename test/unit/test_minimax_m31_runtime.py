# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU checks of the MiniMax M3.1 megakernel's runtime: the covered-step plan, the
stock executor's step against SGLang's own forward (MiniMaxM3SparseForConditionalGeneration,
MiniMaxM3Model and MiniMaxM3DecoderLayer forwards over recording fakes), the parity
switch, the per-width slot plan, the slot list, and the build: slot mega at one
request, moe_nofill and moe_swap at 2-16, the ranks' agreement on the plan and the
refusal of a requested slot that serves no width."""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

pytest.importorskip("sglang.srt.layers.quantization.minimax_m3_nvfp4")

from sglang.kernels.plugin_kernels.nvidia.spec import Step  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

from sgl_nvidia_mega.minimax_m31 import moe as m31_moe  # noqa: E402
from sgl_nvidia_mega.minimax_m31 import plan as m31_plan  # noqa: E402
from sgl_nvidia_mega.minimax_m31 import runtime as m31_runtime  # noqa: E402
from sgl_nvidia_mega.minimax_m31 import spec as m31_spec  # noqa: E402
from sgl_nvidia_mega.minimax_m31.moe import moe_nofill as mn  # noqa: E402
from sgl_nvidia_mega.minimax_m31.moe import moe_swap  # noqa: E402
from sgl_nvidia_mega.minimax_m31.runtime import agree as m31_agree  # noqa: E402
from sgl_nvidia_mega.minimax_m31.runtime import executor as m31_executor  # noqa: E402
from sgl_nvidia_mega.minimax_m31.runtime import parity as m31_parity  # noqa: E402

VERIFY_WIDTHS = tuple(range(8, 129, 8))
STOCK = m31_plan.Slots()
MEGA = m31_plan.Slots(mega=True)
NOFILL_SWAP = m31_plan.Slots(moe_nofill=True, moe_swap=True)


class _Mode:
    def __init__(self, name):
        self.name = name

    def is_target_verify(self):
        return self.name == "target_verify"

    def is_decode(self):
        return self.name == "decode"

    def is_extend(self):
        return self.name in ("extend", "target_verify")


def _batch(mode, batch_size=1, **kw):
    return SimpleNamespace(forward_mode=_Mode(mode), batch_size=batch_size, **kw)


def _built(widths=VERIFY_WIDTHS, plan=None, **kw):
    return m31_executor.MegakernelDecode(
        None, widths=widths, plan=plan or {}, bound={}, **kw
    )


class TestPlan(CustomTestCase):
    """MegakernelDecode.plan: a DSpark verify step of a built width, 8 rows a request,
    while capturing (or under the parity switch)."""

    def test_verify_steps_of_built_widths_are_covered_only_while_capturing(self):
        built = _built()
        ids = torch.zeros(16, dtype=torch.long)
        verify = _batch("target_verify", 2)
        with patch.object(m31_executor, "serves_this_step", return_value=True):
            for requests in (1, 2, 16):
                rows = 8 * requests
                self.assertEqual(
                    built.plan(torch.zeros(rows), _batch("target_verify", requests)),
                    Step(mode="target_verify", rows=rows),
                )
            for mode in ("decode", "extend"):
                self.assertIsNone(built.plan(ids, _batch(mode, 2)))
            # rows that are not 8 a request, past 16 requests, or a width not built
            self.assertIsNone(built.plan(torch.zeros(12), verify))
            self.assertIsNone(built.plan(ids, _batch("target_verify", 1)))
            self.assertIsNone(built.plan(torch.zeros(136), _batch("target_verify", 17)))
            self.assertIsNone(_built(widths=(8, 24)).plan(ids, verify))
            # SGLang's other entries, and two-batch overlap
            self.assertIsNone(built.plan(ids, verify, True))
            self.assertIsNone(built.plan(ids, verify, pp_proxy_tensors=object()))
            self.assertIsNone(
                built.plan(ids, _batch("target_verify", 2, can_run_tbo=True))
            )
        with patch.object(m31_executor, "serves_this_step", return_value=False):
            self.assertIsNone(built.plan(ids, verify))


class TestSlotPlan(CustomTestCase):
    """plan.slots per width: mega at one request only; moe_nofill / moe_swap at their builds' widths where mega does not take the
    step."""

    def test_the_plan_by_width(self):
        every = frozenset(VERIFY_WIDTHS)
        self.assertEqual(m31_plan.served_widths((6, 8, 12, 16, 136)), (8, 16))
        facts = m31_plan.Facts(mega=frozenset({8}), moe_nofill=every, moe_swap=every)
        plan = m31_plan.plan(VERIFY_WIDTHS, facts)
        self.assertEqual(plan[8], MEGA)
        self.assertTrue(all(plan[w] == NOFILL_SWAP for w in VERIFY_WIDTHS[1:]))
        self.assertEqual(
            m31_plan.describe(plan), "8: mega; 16-128: moe_nofill, moe_swap"
        )
        # without mega the MoE slots serve 8 rows too; no slot is SGLang's step code
        moe = m31_plan.plan(
            VERIFY_WIDTHS, m31_plan.Facts(moe_nofill=every, moe_swap=every)
        )
        self.assertEqual(m31_plan.describe(moe), "8-128: moe_nofill, moe_swap")
        stock = m31_plan.plan(VERIFY_WIDTHS, m31_plan.Facts())
        self.assertEqual(m31_plan.describe(stock), "8-128: stock")

    def test_bounds(self):
        # mega serves only 8 rows, whatever its build says
        facts = m31_plan.Facts(mega=frozenset(VERIFY_WIDTHS))
        self.assertEqual(
            [w for w in VERIFY_WIDTHS if m31_plan.slots(w, facts).mega], [8]
        )
        # moe_swap only where moe_nofill serves
        facts = m31_plan.Facts(
            moe_nofill=frozenset({16, 24}), moe_swap=frozenset(VERIFY_WIDTHS)
        )
        self.assertEqual(
            [w for w in VERIFY_WIDTHS if m31_plan.slots(w, facts).moe_swap], [16, 24]
        )
        for rows in (0, 4, 12, 136):
            with self.assertRaises(ValueError):
                m31_plan.slots(rows, facts)
        # a gap between runs of the same slots
        plan = m31_plan.plan((8, 16, 32), m31_plan.Facts(moe_nofill=frozenset({8, 32})))
        self.assertEqual(
            m31_plan.describe(plan), "8: moe_nofill; 16: stock; 32: moe_nofill"
        )


class TestSlotList(CustomTestCase):
    """SGLANG_NVIDIA_MEGAKERNEL_SLOTS: unset ships every slot, a list replaces it, ""
    is SGLang's step code everywhere (the refusals of a list go through SGLang's
    argument resolution: test_minimax_m31_spec.py)."""

    def test_the_list_and_its_two_specs(self):
        env = m31_runtime.SLOTS_ENV
        shipped = frozenset({"mega", "moe_nofill", "moe_swap", "markov"})
        self.assertEqual(m31_runtime.requested_slots({}), shipped)
        self.assertEqual(m31_runtime.requested_slots({env: ""}), frozenset())
        self.assertEqual(m31_runtime.requested_slots({env: " , "}), frozenset())
        listed = m31_runtime.requested_slots({env: "moe_nofill, moe_swap,markov,"})
        # the target's plan takes its slots, the draft spec the draft's
        self.assertEqual(m31_runtime.target_slots(listed), {"moe_nofill", "moe_swap"})
        self.assertEqual(m31_runtime.draft_slots(listed), {"markov"})
        # every slot ships: a new kernel runs by default
        self.assertFalse(hasattr(m31_runtime, "OPT_IN_SLOTS"))
        for slot in m31_runtime.DRAFT_SLOTS:
            self.assertNotIn(slot, m31_plan.Slots.__struct_fields__)
        self.assertEqual(m31_runtime.slot_reasons(shipped), [])
        self.assertEqual(m31_runtime.slot_reasons(frozenset()), [])

    def test_the_targets_numerics(self):
        from sglang.kernels.plugin_kernels.nvidia.spec import NumericsKind

        shipped = m31_spec._numerics(frozenset(m31_runtime.SHIPPED_SLOTS))
        self.assertEqual(shipped.kind, NumericsKind.TOLERANCE)
        self.assertEqual(dict(shipped.tolerances), m31_spec.MEGA_TOLERANCES)
        # the MoE slots are bitwise; a draft slot does not change the target's steps
        for slots in ({"moe_nofill", "moe_swap", "markov"}, {"moe_nofill"}, set()):
            self.assertEqual(
                m31_spec._numerics(frozenset(slots)).kind, NumericsKind.BITWISE
            )


# --------------------------------------------------------------- the stock step


class _Trace:
    """Fake SGLang modules that record each call and transform their input, so that
    two forwards that make the same calls on the same data give equal outputs."""

    def __init__(self):
        self.calls = []

    def add(self, *call):
        self.calls.append(call)


def _fake_lm(trace: _Trace, *, num_layers=5, capture=(), dense=2, hidden=8):
    from sglang.srt.models.minimax_m3 import MiniMaxM3DecoderLayer, MiniMaxM3Model
    from sglang.srt.models.minimax_m3_vl import (
        MiniMaxM3SparseForConditionalGeneration as RealVL,
    )

    class Exit:
        def __init__(self, i):
            self.i = i

        def __enter__(self):
            trace.add("ffn.exit.enter", self.i)
            return self

        def __exit__(self, *exc):
            trace.add("ffn.exit.leave", self.i)
            return False

        def finish(self, x):
            trace.add("ffn.exit.finish", self.i)
            return x - 0.5

    class Boundary:
        def __init__(self, name, i):
            self.name, self.i = name, i

        def prepare(self, x, fb, capture_gathered=None, **kw):
            trace.add(f"{self.name}.prepare", self.i, capture_gathered is not None, kw)
            y = x * 1.5 + self.i
            if capture_gathered is not None:
                capture_gathered.capture(y + 100)
            return y

        def finish(self, x, fb):
            trace.add(f"{self.name}.finish", self.i)
            return x + 0.25

        def exit(self, fb):
            trace.add(f"{self.name}.exit", self.i)
            return Exit(self.i)

    class Attn(torch.nn.Module):
        def __init__(self, i):
            super().__init__()
            self.i = i

        def forward(self, positions, hidden_states, forward_batch):
            trace.add("self_attn", self.i, positions.tolist())
            return hidden_states * 2 + self.i

    class Mlp(torch.nn.Module):
        def __init__(self, i):
            super().__init__()
            self.i = i

        def forward(self, x, forward_batch=None):
            trace.add("mlp", self.i, forward_batch is not None)
            return x + 3 + self.i

    class Layer(torch.nn.Module):
        forward = MiniMaxM3DecoderLayer.forward

        def __init__(self, i):
            super().__init__()
            self.attn_boundary = Boundary("attn", i)
            self.ffn_boundary = Boundary("ffn", i)
            self.self_attn, self.mlp = Attn(i), Mlp(i)
            self.is_layer_sparse = i >= dense
            self._norm_quant = None  # gfx950 only
            if i in capture:
                self._is_layer_to_capture = True

    class Embed(torch.nn.Module):
        def forward(self, ids):
            trace.add("embed", ids.tolist())
            return ids.float()[:, None].expand(-1, hidden) + 0.125

    class Model(torch.nn.Module):
        forward = MiniMaxM3Model.forward
        get_input_embeddings = MiniMaxM3Model.get_input_embeddings

        def __init__(self):
            super().__init__()
            self.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
            self.embed_tokens = Embed()
            self.layers = torch.nn.ModuleList(Layer(i) for i in range(num_layers))
            self.start_layer, self.end_layer = 0, num_layers
            self.norm = "norm"
            self.finalize_norm = "finalize"
            self.layers_to_capture = [i for i in capture] + (
                [num_layers] if capture else []
            )

    class Lm(torch.nn.Module):
        forward = RealVL.forward

        def __init__(self):
            super().__init__()
            self.model = Model()
            self.pp_group = SimpleNamespace(is_last_rank=True)
            self.is_mrope_enabled = False
            self.capture_aux_hidden_states = bool(capture)
            self.lm_head = "lm_head"

        def logits_processor(self, input_ids, hidden, lm_head, fb, aux):
            trace.add(
                "logits", input_ids.tolist(), lm_head, None if aux is None else len(aux)
            )
            return SimpleNamespace(
                next_token_logits=hidden * 7,
                hidden_states=None if aux is None else torch.cat(list(aux), dim=-1),
            )

    return Lm()


def _patched_runtime(trace: _Trace):
    """SGLang's runtime queries both forwards make, faked alike."""
    from sglang.srt.eplb import expert_distribution
    from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
    from sglang.srt.model_executor import cuda_graph_config
    from sglang.srt.model_executor.runner_backend_utils import breakable_cuda_graph
    from sglang.srt.models import minimax_m3

    @contextlib.contextmanager
    def with_current_layer(i):
        trace.add("recorder", i)
        yield

    recorder = SimpleNamespace(with_current_layer=with_current_layer)

    def start(fb):
        trace.add("residual.start")

    def final_norm(x, fb, norm, *, finalize_norm, capture, skip_empty):
        trace.add(
            "residual.final_norm", norm, finalize_norm, capture is not None, skip_empty
        )
        if capture is not None:
            capture(x + 1000)
        return x / 4

    stack = contextlib.ExitStack()
    for target, name, value in (
        (minimax_m3, "check_cuda_graph_backend", lambda *a: False),
        (cuda_graph_config, "check_cuda_graph_backend", lambda *a: False),
        (minimax_m3, "is_in_breakable_cuda_graph", lambda: False),
        (breakable_cuda_graph, "is_in_breakable_cuda_graph", lambda: False),
        (minimax_m3, "get_global_expert_distribution_recorder", lambda: recorder),
        (
            expert_distribution,
            "get_global_expert_distribution_recorder",
            lambda: recorder,
        ),
        (residual_batch, "start", start),
        (residual_batch, "final_norm", final_norm),
    ):
        stack.enter_context(patch.object(target, name, value))
    return stack


def _fb(rows):
    return SimpleNamespace(
        forward_mode=_Mode("target_verify"),
        batch_size=rows // 8,
        input_embeds=None,
        can_run_tbo=False,
    )


class TestStockStep(CustomTestCase):
    """With every slot off the executor makes SGLang's calls in SGLang's order."""

    def _compare(self, rows, **lm_kwargs):
        ids = torch.arange(rows, dtype=torch.long) + 3
        positions = torch.arange(rows, dtype=torch.long) + 40
        runs = []
        for use_executor in (False, True):
            trace = _Trace()
            lm = _fake_lm(trace, **lm_kwargs)
            fb = _fb(rows)
            with (
                _patched_runtime(trace),
                patch.dict("os.environ", {"SGLANG_NVIDIA_MEGAKERNEL_PARITY": "0"}),
            ):
                if use_executor:
                    built = m31_executor.MegakernelDecode(
                        lm, widths=(rows,), plan={rows: STOCK}, bound={}
                    )
                    step = Step(mode="target_verify", rows=rows)
                    out = built.forward(step, lm.forward, ids, positions, fb)
                else:
                    out = lm(ids, positions, fb)
            runs.append((trace.calls, out))
        (want_calls, want), (got_calls, got) = runs
        self.assertEqual(got_calls, want_calls)
        self.assertTrue(torch.equal(got.next_token_logits, want.next_token_logits))
        if want.hidden_states is None:
            self.assertIsNone(got.hidden_states)
        else:
            self.assertTrue(torch.equal(got.hidden_states, want.hidden_states))
        return want_calls

    def test_dspark_verify_step_with_captures(self):
        calls = self._compare(16, capture=(1, 3))
        self.assertIn(("logits", list(range(3, 19)), "lm_head", 3), calls)

    def test_verify_step_without_captures(self):
        calls = self._compare(32)
        self.assertEqual(calls[-1], ("logits", list(range(3, 35)), "lm_head", None))

    def test_the_moe_step_takes_the_widths_slots(self):
        """A width planned for moe_nofill / moe_swap reaches moe.step with them."""
        rows = 16
        trace = _Trace()
        lm = _fake_lm(trace)
        seen = []

        def step(layer, x, st):
            seen.append((layer.is_layer_sparse, st.slots))
            return layer.mlp(x, forward_batch=st.fb)

        built = m31_executor.MegakernelDecode(
            lm, widths=(rows,), plan={rows: NOFILL_SWAP}, bound={}
        )
        with _patched_runtime(trace), patch.object(m31_moe, "step", step):
            built._forward(
                Step(mode="target_verify", rows=rows),
                torch.arange(rows),
                torch.arange(rows),
                _fb(rows),
            )
        self.assertEqual(len(seen), 5)
        self.assertTrue(all(s is NOFILL_SWAP for _, s in seen))


class TestParity(CustomTestCase):
    """SGLANG_NVIDIA_MEGAKERNEL_PARITY: SGLang's forward, then the executor's (or
    SGLang's again under "control"), the executor's output returned, one greppable
    line; slot mega's steps log the tolerance line."""

    def _run(self, value, rows=16, plan=None, mega=None):
        ids = torch.arange(rows, dtype=torch.long) + 3
        positions = torch.arange(rows, dtype=torch.long) + 40
        trace = _Trace()
        lm = _fake_lm(trace, capture=(2,))
        built = m31_executor.MegakernelDecode(
            lm, widths=(rows,), plan=plan or {rows: STOCK}, bound={}, mega=mega
        )
        with (
            _patched_runtime(trace),
            patch.dict("os.environ", {"SGLANG_NVIDIA_MEGAKERNEL_PARITY": value}),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
            self.assertLogs(m31_parity.logger, level="INFO") as logs,
        ):
            out = built.forward(
                Step(mode="target_verify", rows=rows),
                lm.forward,
                ids,
                positions,
                _fb(rows),
            )
            m31_parity.logger.info("end")
        return trace.calls, out, [x for x in logs.output if "megakernel parity" in x]

    def test_stock_widths_log_bitwise_equality(self):
        calls0, out0, lines = self._run("0")
        self.assertEqual(lines, [])
        for value in ("1", "control"):
            calls, out, lines = self._run(value)
            # SGLang's forward, then the executor's (or SGLang's again): the same calls
            self.assertEqual(calls, calls0 + calls0)
            self.assertTrue(torch.equal(out.next_token_logits, out0.next_token_logits))
            self.assertTrue(torch.equal(out.hidden_states, out0.hidden_states))
            self.assertEqual(len(lines), 1, lines)
            self.assertIn(
                "target_verify rows=16 logits_max_abs=0 (rel 0.00e+00, 0.00% differ)",
                lines[0],
            )
            self.assertIn("hidden_max_abs=0 (rel", lines[0])
            self.assertEqual("parity control" in lines[0], value == "control")

    def test_slot_mega_logs_the_tolerance_line(self):
        def run(step, ids, positions, fb):
            return SimpleNamespace(
                next_token_logits=torch.ones(8, 8), hidden_states=None
            )

        mega = SimpleNamespace(run=run)
        _, out, lines = self._run("1", rows=8, plan={8: MEGA}, mega=mega)
        self.assertTrue(torch.equal(out.next_token_logits, torch.ones(8, 8)))
        (line,) = lines
        self.assertIn("NVIDIA megakernel parity target logits rows=8 max_abs=", line)
        self.assertIn("min_cosine=", line)
        self.assertIn("top1=", line)


# ------------------------------------------------------------------- the build


@contextlib.contextmanager
def _builds(slots, *, moe=None, swap=None, agree=None, mega=None):
    """``runtime.build`` with ``slots`` requested, moe_nofill's build returning ``moe``
    and moe_swap's ``swap``, slot mega's step ``mega`` and the ranks' agreement
    ``agree`` (one rank: nothing to disagree with)."""
    from sgl_nvidia_mega.minimax_m31.runtime import mega as m31_mega

    with (
        patch.object(m31_runtime, "SLOTS", frozenset(slots)),
        patch.object(m31_moe, "build", return_value=moe),
        patch.object(moe_swap, "build", return_value=swap),
        patch.object(m31_agree, "agree_tp", agree or (lambda value, what: None)),
        patch.object(m31_mega, "MegaStep", mega or Mock()),
    ):
        yield


VERIFY = SimpleNamespace(widths=VERIFY_WIDTHS)


class TestBuild(CustomTestCase):
    def test_the_default_plan(self):
        mega = Mock()
        mega.return_value.summary.return_value = "MegaAttention"
        nofill = mn.Bound(widths=VERIFY_WIDTHS)
        swap = moe_swap.Bound(widths=VERIFY_WIDTHS)
        # SGLang may ask for wider steps than the megakernel covers
        ctx = SimpleNamespace(widths=tuple(range(8, 257, 8)))
        with _builds(m31_runtime.SHIPPED_SLOTS, moe=nofill, swap=swap, mega=mega):
            built = m31_runtime.build("model", "hf", ctx)
        mega.assert_called_once_with("model", "hf", ctx)
        self.assertIs(built.mega, mega.return_value)
        self.assertEqual(built.widths, VERIFY_WIDTHS)
        self.assertEqual(built.slot_plan[8], MEGA)
        self.assertTrue(
            all(built.slot_plan[w] == NOFILL_SWAP for w in VERIFY_WIDTHS[1:])
        )
        self.assertIn(
            "slots by width 8: mega; 16-128: moe_nofill, moe_swap", built.summary()
        )
        self.assertIn("mega: MegaAttention", built.summary())
        # the built object SGLang reads (BuiltMegakernel)
        for name in (
            "draft_sampler",
            "draft_loop",
            "draft_extend",
            "begin_draft_round",
            "graph_variants",
            "speculative_runtime_hooks",
        ):
            self.assertIsNone(getattr(built, name))

    def test_no_mega_no_routed_expert_copy(self):
        mega = Mock()
        nofill = mn.Bound(widths=VERIFY_WIDTHS)
        swap = moe_swap.Bound(widths=VERIFY_WIDTHS)
        with _builds({"moe_nofill", "moe_swap"}, moe=nofill, swap=swap, mega=mega):
            built = m31_runtime.build("model", "hf", VERIFY)
        mega.assert_not_called()
        self.assertIsNone(built.mega)
        self.assertEqual(built.slot_plan[8], NOFILL_SWAP)

    def test_mega_needs_one_requests_width(self):
        ctx = SimpleNamespace(widths=(16, 24))
        mega = Mock()
        with (
            _builds({"mega"}, mega=mega),
            self.assertRaisesRegex(
                RuntimeError, "slot mega serves one request's 8 verify rows"
            ),
        ):
            m31_runtime.build("model", "hf", ctx)
        mega.assert_not_called()
        # without mega the same widths build
        with _builds(set()):
            built = m31_runtime.build("model", "hf", ctx)
        self.assertEqual(built.widths, (16, 24))
        with _builds(set()), self.assertRaisesRegex(RuntimeError, "no width"):
            m31_runtime.build("model", "hf", SimpleNamespace(widths=(6, 12)))

    def test_one_request_server_with_the_default_slots(self):
        """--cuda-graph-max-bs-decode 1: mega takes the only width; moe_nofill built
        there is not a failed build."""
        ctx = SimpleNamespace(widths=(8,))
        with _builds(
            m31_runtime.SHIPPED_SLOTS,
            moe=mn.Bound(widths=(8,)),
            swap=moe_swap.Bound(widths=(8,)),
        ):
            built = m31_runtime.build("model", "hf", ctx)
        self.assertEqual(built.slot_plan, {8: MEGA})

    def test_ranks_agree_on_the_facts(self):
        agreed = []
        with _builds(
            m31_runtime.SHIPPED_SLOTS,
            moe=mn.Bound(widths=VERIFY_WIDTHS),
            swap=moe_swap.Bound(widths=VERIFY_WIDTHS),
            agree=lambda value, what: agreed.append((value, what)),
        ):
            m31_runtime.build("model", "hf", VERIFY)
        ((facts, what),) = agreed
        self.assertEqual(what, "slot plans (widths per slot)")
        self.assertEqual(
            facts,
            m31_plan.Facts(
                mega=frozenset({8}),
                moe_nofill=frozenset(VERIFY_WIDTHS),
                moe_swap=frozenset(VERIFY_WIDTHS),
            ),
        )

        def disagree(value, what):
            raise RuntimeError(f"the ranks' {what} differ")

        with (
            _builds({"moe_nofill"}, moe=mn.Bound(widths=VERIFY_WIDTHS), agree=disagree),
            self.assertRaisesRegex(RuntimeError, "slot plans"),
        ):
            m31_runtime.build("model", "hf", VERIFY)

    def test_a_requested_slot_that_serves_no_width_refuses(self):
        nofill = mn.Bound(reason="rank 1: layer 3: the module has forward hooks")
        with (
            _builds(
                {"moe_nofill", "moe_swap"},
                moe=nofill,
                swap=moe_swap.Bound(reason="slot moe_nofill serves no width"),
            ),
            self.assertRaises(RuntimeError) as caught,
        ):
            m31_runtime.build("model", "hf", VERIFY)
        self.assertEqual(
            str(caught.exception),
            "NVIDIA megakernel: requested slots ['moe_nofill', 'moe_swap'] serve no "
            f"width of {list(VERIFY_WIDTHS)} (mega: not requested; moe_nofill: "
            f"{nofill.reason}; moe_swap: slot moe_nofill serves no width)",
        )


class TestModelChecks(CustomTestCase):
    def test_the_executor_follows_sglangs_classes(self):
        from sglang.srt.models.minimax_m3 import MiniMaxM3DecoderLayer, MiniMaxM3Model

        def model(norm_quant=None, layer_cls=MiniMaxM3DecoderLayer):
            m = MiniMaxM3Model.__new__(MiniMaxM3Model)
            torch.nn.Module.__init__(m)
            layer = layer_cls.__new__(layer_cls)
            torch.nn.Module.__init__(layer)
            layer._norm_quant = norm_quant
            m.layers = [layer]
            m.start_layer, m.end_layer = 0, 1
            m.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
            return SimpleNamespace(
                model=m, is_mrope_enabled=False, logits_processor=object()
            )

        stock = frozenset()
        self.assertEqual(m31_runtime.model_disable_reasons(model(), None, stock), [])
        (why,) = m31_runtime.model_disable_reasons(
            model(norm_quant=object()), None, stock
        )
        self.assertIn("quantize in their input norms", why)
        (why,) = m31_runtime.model_disable_reasons(
            model(layer_cls=torch.nn.Module), None, stock
        )
        self.assertIn("not MiniMaxM3DecoderLayer", why)
        (why,) = m31_runtime.model_disable_reasons(SimpleNamespace(model=None), None)
        self.assertIn("not MiniMaxM3Model", why)
        # slot mega adds its operand checks
        from sgl_nvidia_mega.minimax_m31.runtime import mega as m31_mega

        with patch.object(m31_mega, "model_disable_reasons", return_value=["mega why"]):
            self.assertEqual(
                m31_runtime.model_disable_reasons(model(), None, frozenset({"mega"})),
                ["mega why"],
            )


if __name__ == "__main__":
    unittest.main()

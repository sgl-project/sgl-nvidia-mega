# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU checks of slot moe_swap of the MiniMax M3.1 megakernel (moe/moe_swap.py).

The slot's MoE step against SGLang's MiniMaxM3MoE.forward on the fake pieces of
moe_nofill's CPU test (test_minimax_m31_moe_nofill_slot.py): the same launches with
the same arguments and the same handoff; while capturing, the fork first, then the
routed chain on the current stream, then the shared expert on the alternate stream,
then the join; otherwise one stream, the routed chain first; the MoE step taking it
where the plan says; its build. On GPU the slot is checked by the composition check
(SGLANG_NVIDIA_MEGAKERNEL_PARITY=1) at every width it serves.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

pytest.importorskip("sglang.srt.layers.quantization.minimax_m3_nvfp4")

# the MoE fakes of the sibling slot's CPU test
import test_minimax_m31_moe_nofill_slot as nofill_tests  # noqa: E402
from sglang.srt.models.minimax_m3 import MiniMaxM3MoE  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

from sgl_nvidia_mega.minimax_m31 import moe as m31_moe  # noqa: E402
from sgl_nvidia_mega.minimax_m31 import plan as m31_plan  # noqa: E402
from sgl_nvidia_mega.minimax_m31.moe import moe_nofill as mn  # noqa: E402
from sgl_nvidia_mega.minimax_m31.moe import moe_swap  # noqa: E402

Trace, Streams = nofill_tests.Trace, nofill_tests.Streams
rows = nofill_tests.rows

# the routed chain's launches and the shared expert's, as the fakes record them
ROUTED = ("router", "topk", "dispatch", "quant_kernel", "routed_moe", "combine")
SHARED = ("shared",)


def _calls(trace):
    """(what, stream) of each recorded call."""
    return [c[:2] for c in trace]


class TestSwappedMoe(CustomTestCase):
    """moe_swap.forward against SGLang's MiniMaxM3MoE.forward on the same pieces."""

    def _run(self, *, capturing, m=8, alt=True):
        out = {}
        for path in ("stock", "swap"):
            trace = Trace()
            streams = Streams(trace)
            fi = nofill_tests.FakeFlashInfer(trace, streams)
            mlp = nofill_tests.Moe(trace, streams, alt=alt)
            quant = nofill_tests._quant()
            x = rows(m)
            with fi.installed(), nofill_tests.runtime(streams, capturing=capturing):
                if path == "stock":
                    result = mlp(x, forward_batch=None)
                else:
                    result = moe_swap.forward(mlp, x, quant)
            out[path] = SimpleNamespace(trace=trace.calls, result=result, fi=fi)
        return out["stock"], out["swap"]

    def _check_same_work(self, stock, swap):
        # the same handoff, bit for bit
        nofill_tests._handoff_equal(self, swap.result, stock.result)
        # the same launches with the same arguments in the same order on each chain,
        # only on other streams (minus FlashInfer's nvfp4_quantize wrapper, whose
        # allocations hold moe_nofill's fill)
        for chain in (ROUTED, SHARED):
            self.assertEqual(
                [(c[0], *c[2:]) for c in swap.trace if c[0] in chain],
                [(c[0], *c[2:]) for c in stock.trace if c[0] in chain],
            )
        # the routed MoE call with the stock arguments
        (want,), (got,) = stock.fi.moe_calls, swap.fi.moe_calls
        nofill_tests._moe_call_equal(self, got, want)

    def test_capture_order_routed_chain_on_the_current_stream(self):
        stock, swap = self._run(capturing=True)
        self._check_same_work(stock, swap)
        # SGLang: the shared expert on the current stream, the routed chain forked
        self.assertEqual(
            [c for c in _calls(stock.trace) if c[0] != "nvfp4_quantize"],
            [
                ("wait_stream", "alt"),
                ("shared", "main"),
                ("enter", "alt"),
                *((what, "alt") for what in ROUTED),
                ("leave", "alt"),
                ("wait_stream", "main"),
            ],
        )
        # the slot: the same fork first, then the routed chain on the current stream
        # (the boundary kernel's successor there), then the shared expert forked
        self.assertEqual(
            _calls(swap.trace),
            [
                ("wait_stream", "alt"),
                *((what, "main") for what in ROUTED),
                ("enter", "alt"),
                ("shared", "alt"),
                ("leave", "alt"),
                ("wait_stream", "main"),
            ],
        )
        # the fork makes the alternate stream wait for the current one, the join the
        # current one for the alternate one
        waits = [c for c in swap.trace if c[0] == "wait_stream"]
        self.assertEqual(
            waits, [("wait_stream", "alt", "main"), ("wait_stream", "main", "alt")]
        )

    def test_one_stream_the_routed_chain_first(self):
        for capturing, alt in ((False, True), (True, False)):
            with self.subTest(capturing=capturing, alt=alt):
                stock, swap = self._run(capturing=capturing, alt=alt)
                self._check_same_work(stock, swap)
                self.assertEqual(
                    _calls(swap.trace),
                    [*((what, "main") for what in ROUTED), ("shared", "main")],
                )


class TestMoeStep(CustomTestCase):
    """moe.step: moe_swap's chain where planned, moe_nofill's where only it is,
    SGLang's module otherwise."""

    def _step(self, *, slots, x, layer_id=1):
        trace = Trace()
        streams = Streams(trace)
        fi = nofill_tests.FakeFlashInfer(trace, streams)
        model = nofill_tests.fake_model(trace, streams)
        quant = nofill_tests._quant()
        layer = model.model.layers[layer_id]
        st = SimpleNamespace(
            slots=slots, bound={"moe": nofill_tests._bound(model, quant)}, fb="fb"
        )
        forwards = {name: Mock(return_value=name) for name in ("swap", "nofill")}
        with (
            fi.installed(),
            nofill_tests.runtime(streams),
            patch.object(moe_swap, "forward", forwards["swap"]),
            patch.object(mn, "forward", forwards["nofill"]),
            patch.object(
                MiniMaxM3MoE, "forward", lambda self, h, forward_batch: "stock"
            ),
        ):
            return m31_moe.step(layer, x, st), forwards, layer, quant

    def test_moe_swap_runs_where_planned(self):
        x = rows(8)
        out, forwards, layer, quant = self._step(
            slots=m31_plan.Slots(moe_nofill=True, moe_swap=True), x=x
        )
        self.assertEqual(out, "swap")
        forwards["swap"].assert_called_once_with(layer.mlp, x, quant)
        forwards["nofill"].assert_not_called()
        out, forwards, _, _ = self._step(slots=m31_plan.Slots(moe_nofill=True), x=x)
        self.assertEqual(out, "nofill")
        forwards["swap"].assert_not_called()

    def test_slot_mega_or_no_slot_runs_sglangs_module(self):
        for slots in (m31_plan.Slots(), m31_plan.Slots(mega=True)):
            out, forwards, _, _ = self._step(slots=slots, x=rows(8))
            self.assertEqual(out, "stock")
            forwards["swap"].assert_not_called()
            forwards["nofill"].assert_not_called()


class TestBuild(CustomTestCase):
    def _model(self, **moe_kw):
        trace = Trace()
        streams = Streams(trace)
        return nofill_tests.fake_model(trace, streams, **moe_kw)

    def test_moe_swap_serves_moe_nofills_widths(self):
        model = self._model()
        quant = nofill_tests._quant()
        nofill = nofill_tests._bound(model, quant)
        built = moe_swap.build(model, (8, 16), frozenset({"moe_swap"}), nofill)
        self.assertEqual((built.widths, built.reason), (nofill.widths, ""))
        self.assertIsNone(moe_swap.build(model, (8,), frozenset(), nofill))
        # nothing to swap: no moe_nofill, or a MoE on one stream
        for bound in (None, mn.Bound()):
            built = moe_swap.build(model, (8,), frozenset({"moe_swap"}), bound)
            self.assertEqual(
                (built.widths, built.reason), ((), "slot moe_nofill serves no width")
            )
        model = self._model(alt=False)
        built = moe_swap.build(
            model, (8,), frozenset({"moe_swap"}), nofill_tests._bound(model, quant)
        )
        self.assertEqual(built.widths, ())
        self.assertIn("layer 1: the MoE has no alternate stream", built.reason)


if __name__ == "__main__":
    unittest.main()

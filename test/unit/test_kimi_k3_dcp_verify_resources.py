# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""q6 resource CPU contracts; mocked CUDA/compiler boundaries are not GPU proof."""

from types import SimpleNamespace as NS
from unittest import TestCase
from unittest.mock import Mock, patch

import torch
from test_kimi_k3_dcp_resources import (
    DEVICE,
    CpuBox,
    FakeCache,
    FakeWorkspace,
    NativeFixture,
    ResourceFixture,
)
from test_kimi_k3_requested_widths import BuildFixture

from sgl_nvidia_mega.kimi_k3 import dcp_resources as d


class Fixture:
    def __init__(self, rank=3, widths=(6, 12, 60)):
        self.native = NativeFixture()
        self.native.arguments.update(tokens_per_request=6, widths=widths)
        for group in (self.native.parallel.tp_group, self.native.parallel.dcp_group):
            group.rank_in_group, group.rank = rank, group.ranks[rank]
        self.native.parallel.tp_rank = self.native.parallel.dcp_rank = rank
        self.native.parallel.attn_dcp_rank = rank
        self.owner = self.native.capture()
        self.module = NS(
            DcpBox=CpuBox,
            local_max=lambda n, size: ((n + size * 64 - 1) // (size * 64)) * 64,
            stock_decode=Mock(),
            fill_splits=Mock(return_value=32),
            _WORKSPACE={},
        )
        zeros, ones, full = torch.zeros, torch.ones, torch.full
        with (
            patch.object(
                torch,
                "zeros",
                side_effect=lambda *a, **k: zeros(*a, **(k | {"device": "cpu"})),
            ),
            patch.object(
                torch,
                "ones",
                side_effect=lambda *a, **k: ones(*a, **(k | {"device": "cpu"})),
            ),
            patch.object(
                torch,
                "full",
                side_effect=lambda *a, **k: full(*a, **(k | {"device": "cpu"})),
            ),
        ):
            self.resources = d.DcpResources(
                self.owner,
                self.native.options,
                self.module,
                widths=widths,
                max_context=8192,
                deadline_ns=5_000_000_000,
                tokens_per_request=6,
            )

    def core(self, ordinal=0):
        r = self.resources
        core = NS(
            device=DEVICE,
            heads=12,
            max_seq=r.max_seq,
            dcp=r.box,
            parity=ordinal % 2,
            deadline_ns=r.deadline_ns,
            cores={(m, 6): "tokenspeed" for m in r.widths},
            pitches={m: 3648 for m in r.widths},
            kernels={},
            weights=(),
            stock_scales=(0.25, 1.0),
            run=Mock(return_value=object()),
        )

        def prepare(cache, prewarm):
            for m in r.widths:
                prewarm(core, m, 6)
                core.kernels[m, 6] = object()
            self.module._WORKSPACE[DEVICE, 96] = FakeWorkspace(64 * (ordinal + 1))

        core.prepare_cache = Mock(side_effect=prepare)
        return core


class TestVerifyResources(TestCase):
    def test_actual_builder_threads_q6_without_changing_default_q1_storage(self):
        ordinary = ResourceFixture()
        self.assertEqual(ordinary.resources.tokens_per_request, 1)
        self.assertFalse(hasattr(ordinary.resources, "warm_totals"))
        with self.assertRaisesRegex(ValueError, "all-rank agreed"):
            ordinary.owner.check_plan(
                ordinary.native.options, (1, 8), 8192, 5_000_000_000, True
            )
        f, box, owner = BuildFixture(), object(), object()
        r = NS(box=box)
        f.modules["mla.dcp"] = object()
        with (
            patch("importlib.metadata.version", return_value="2.0.2"),
            patch.object(d.NativeDcpOwner, "capture", return_value=owner) as capture,
            patch.object(d, "DcpResources", return_value=r) as build,
            patch.object(d, "PreparedDcpMla", side_effect=lambda core, resources: core),
        ):
            resources = f.build(
                6, (6, 24), mla_core="tokenspeed", dcp=d.DcpBuildOptions()
            )
        self.assertIs(resources.dcp, r)
        self.assertEqual(capture.call_args.kwargs["tokens_per_request"], 6)
        self.assertEqual(build.call_args.kwargs["tokens_per_request"], 6)
        self.assertEqual(build.call_args.kwargs["widths"], (6, 24))
        self.assertEqual(
            build.call_args.kwargs["max_context"],
            capture.call_args.kwargs["max_context"],
        )
        self.assertEqual(f.modules["mla.prepared"].PreparedMla.call_count, 24)
        for ordinal, call in enumerate(
            f.modules["mla.prepared"].PreparedMla.call_args_list
        ):
            self.assertEqual(call.kwargs["queries"], (6,))
            self.assertEqual(call.kwargs["parity"], ordinal % 2)
            self.assertIs(call.kwargs["dcp"], box)
        f.module._prepare_mla_workspace.assert_not_called()

    def test_q6_plan_is_explicit_and_all_rank_agreed(self):
        f = Fixture()
        self.assertEqual(f.owner._contract[-1], 6)
        self.assertEqual(f.native.agree.call_args.args[0][1][-1], 6)
        f.owner.check_plan(f.native.options, (6, 12, 60), 8192, 5_000_000_000, 6)
        with self.assertRaisesRegex(ValueError, "all-rank agreed"):
            f.owner.check_plan(f.native.options, (6, 12, 60), 8192, 5_000_000_000)
        for changes in (
            {"tokens_per_request": 5},
            {"widths": (1, 6)},
            {"widths": (6, 64)},
            {"widths": (12, 6)},
            {"widths": (6, 6)},
            {"tokens_per_request": True},
        ):
            native = NativeFixture()
            native.arguments.update(tokens_per_request=6, widths=(6, 12))
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                native.capture(**changes)
            native.agree.assert_called_once()
        native = NativeFixture()
        native.capture(tokens_per_request=6, widths=(6,), max_context=262150)
        with self.assertRaises(ValueError):
            native.capture(tokens_per_request=6, widths=(6,), max_context=262151)

    def test_real_callback_shapes_finite_causal_inputs_all_eight_ranks(self):
        for rank in range(8):
            f = Fixture(rank)
            r = f.resources
            mailboxes = [t.clone() for t in (r.box.q, r.box.partial, r.box.lse)]
            with (
                patch.object(
                    torch.cuda, "is_current_stream_capturing", return_value=False
                ),
                patch.object(torch.cuda, "synchronize") as sync,
            ):
                for m in r.widths:
                    with self.subTest(rank=rank, rows=m):
                        r.prewarm(f.core(), m, 6)
                        args, kwargs = f.module.stock_decode.call_args
                        (
                            query,
                            cache,
                            table,
                            lengths,
                            maximum,
                            scales,
                            out,
                            lse,
                            splits,
                        ) = args
                        b = m // 6
                        self.assertEqual(tuple(query.shape), (b, 6, 96, 576))
                        self.assertEqual(query.dtype, torch.float8_e4m3fn)
                        self.assertTrue(torch.isfinite(query.float()).all())
                        self.assertTrue(
                            torch.equal(
                                query.view(torch.uint8),
                                torch.zeros_like(query.view(torch.uint8)),
                            )
                        )
                        self.assertIs(cache, r.warm_cache)
                        self.assertEqual(tuple(table.shape), (b, 1))
                        self.assertTrue(
                            torch.equal(
                                lengths,
                                torch.full(
                                    (b,), 2 if rank < 6 else 1, dtype=torch.int32
                                ),
                            )
                        )
                        self.assertEqual(
                            (maximum, scales, splits), (1024, (0.25, 1.0), 32)
                        )
                        self.assertEqual(tuple(out.shape), (b, 6, 96, 512))
                        self.assertEqual(tuple(lse.shape), (b, 6, 96))
                        self.assertEqual(out.data_ptr(), r.box.rows.data_ptr())
                        self.assertEqual(lse.data_ptr(), r.box.rows_lse.data_ptr())
                        self.assertEqual(kwargs["cp"], (8, rank, 1))
                        self.assertIs(kwargs["pack"], True)
                        self.assertTrue(
                            torch.equal(
                                kwargs["totals"],
                                torch.full((b,), 14, dtype=torch.int32),
                            )
                        )
                        f.module.fill_splits.assert_called_with(m, DEVICE)
                        sync.assert_called_with(DEVICE)
            for current, before in zip((r.box.q, r.box.partial, r.box.lse), mailboxes):
                self.assertTrue(
                    torch.equal(current.view(torch.uint8), before.view(torch.uint8))
                )

    def test_unprepared_wrong_query_capture_and_owner_drift_are_rejected(self):
        f = Fixture()
        with patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        ):
            for m, q in ((1, 1), (6, 1), (18, 6), (64, 6)):
                with self.subTest(rows=m, q=q), self.assertRaises(RuntimeError):
                    f.resources.prewarm(f.core(), m, q)
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            with self.assertRaises(RuntimeError):
                f.resources.prewarm(f.core(), 6, 6)
        f.module.stock_decode.assert_not_called()
        f.resources.warm_totals = f.resources.warm_totals.clone()
        with self.assertRaisesRegex(RuntimeError, "sealed storage"):
            f.resources.assert_live()
        f = Fixture()
        f.resources.tokens_per_request = 1
        with self.assertRaisesRegex(RuntimeError, "geometry"):
            f.resources.assert_live()

    def test_all_24_wrappers_retain_only_final_shared_96_workspace(self):
        f = Fixture()
        providers = {
            i * 4: NS(
                kind="mla", core=d.PreparedDcpMla(f.core(i), f.resources), cache=None
            )
            for i in range(24)
        }
        caches = {layer: FakeCache() for layer in providers}
        with (
            patch.object(torch, "Tensor", FakeWorkspace),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
            patch.object(torch.cuda, "synchronize"),
        ):
            f.resources.prepare_mla_caches(providers, caches)
            final = f.module._WORKSPACE[DEVICE, 96]
            self.assertIs(f.resources.workspace, final)
            self.assertEqual(f.module.stock_decode.call_count, 24 * 3)
            for ordinal, provider in enumerate(providers.values()):
                self.assertIs(provider.core.workspace, final)
                self.assertEqual(provider.core.core.parity, ordinal % 2)
                self.assertEqual(provider.core.heads, 12)
                provider.core.run(q=6, totals=object())
                provider.core.core.run.assert_called_once()
                with self.assertRaises(ValueError):
                    provider.core.run(totals=object())
            with self.assertRaisesRegex(RuntimeError, "already finalized"):
                f.resources.prewarm(f.core(), 6, 6)
            f.module._WORKSPACE[DEVICE, 96] = FakeWorkspace(9999)
            with self.assertRaisesRegex(RuntimeError, "workspace changed"):
                f.resources.assert_live()
            self.assertTrue(all(p.core.workspace is final for p in providers.values()))

    def test_failure_poison_keeps_original_error(self):
        f, failure = Fixture(), RuntimeError("native prewarm failure")
        f.module.stock_decode.side_effect = failure
        with (
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
            patch.object(torch.cuda, "synchronize"),
        ):
            with self.assertRaises(RuntimeError) as raised:
                f.resources.prewarm(f.core(), 6, 6)
        self.assertIs(raised.exception, failure)
        self.assertTrue(f.resources.failed)
        with self.assertRaisesRegex(RuntimeError, "failed preparation"):
            f.resources.assert_live()

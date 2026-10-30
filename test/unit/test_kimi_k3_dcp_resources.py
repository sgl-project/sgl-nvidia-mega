# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Private CPU resource contracts, not native GPU/transport qualification.

Import the real resource module. Only native context/CUDA/compiled boundaries
are mocked; finite prewarm tensors use real CPU Torch storage. Existing source
definition helpers inspect the CuTe-dependent host constructors without claiming
that they import, compile or execute native GPU kernels.
"""

from __future__ import annotations

import ast
import builtins
import gc
import unittest
import weakref
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch
from test_kimi_k3_mla_readiness import DCP, PROVIDERS, Workspace, definitions
from test_kimi_k3_requested_widths import BuildFixture

from sgl_nvidia_mega.kimi_k3 import dcp_resources as d

PREPARED = DCP.with_name("prepared.py")
DEVICE = torch.device("cuda:0")


class Token:
    pass


def parallel(rank=3, first=16):
    def group():
        value = Token()
        value.ranks = list(range(first, first + 8))
        value.rank_in_group, value.world_size, value.rank = rank, 8, first + rank
        value.device_group, value.cpu_group = Token(), Token()
        return value

    return NS(
        dcp_enabled=True,
        tp_size=8,
        tp_rank=rank,
        dcp_size=8,
        dcp_rank=rank,
        attn_dcp_size=8,
        attn_dcp_rank=rank,
        tp_group=group(),
        dcp_group=group(),
    )


class NativeFixture:
    def __init__(self):
        self.parallel, self.agree = parallel(), Mock()
        self.options = d.DcpBuildOptions()
        self.arguments = dict(
            device=DEVICE,
            tokens_per_request=1,
            deadline_ns=5_000_000_000,
            widths=(1, 8),
            max_context=8192,
            mla_core="tokenspeed",
        )

    def capture(self, **changes):
        with (
            patch.object(
                d, "_native_context", return_value=(lambda: self.parallel, self.agree)
            ),
            patch.object(torch.cuda, "get_device_capability", return_value=(10, 3)),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
        ):
            return d.NativeDcpOwner.capture(self.options, **(self.arguments | changes))


class CpuBox:
    """CPU storage fixture only: no symmetric-memory or transport claim."""

    def __init__(self, group, device, size, rank, heads, cap, interleave):
        self.device_group, self.size, self.rank = group, size, rank
        self.heads, self.cap, self.interleave, self.q_mc = heads, cap, interleave, 4096
        self.handles = (Token(), Token(), Token())
        self.q = torch.full((2, cap, size * heads, 576), 255, dtype=torch.uint8)
        self.partial = torch.empty((2, size, cap, heads, 512), dtype=torch.bfloat16)
        self.partial.view(torch.int32).fill_(-0x80000000)
        self.lse = torch.empty((2, size, cap, heads), dtype=torch.float32)
        self.lse.view(torch.int32).fill_(-1)
        self.peers = torch.arange(size, dtype=torch.int64) + 4096
        self.lse_peers = self.peers.clone()
        self.rows = torch.zeros((cap, 1, size * heads, 512), dtype=torch.bfloat16)
        self.rows_lse = torch.zeros((cap, 1, size * heads), dtype=torch.float32)


class ResourceFixture:
    def __init__(self):
        self.native = NativeFixture()
        self.owner = self.native.capture()
        self.module = NS(
            DcpBox=CpuBox,
            local_max=lambda n, size: ((n + size * 64 - 1) // (size * 64)) * 64,
            stock_decode=Mock(),
            fill_splits=Mock(return_value=32),
            _WORKSPACE={},
        )
        zeros, ones = torch.zeros, torch.ones
        # Make preparation storage real CPU tensors, explicitly substituting
        # only their allocation device. This is not a native-device execution.
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
        ):
            self.resources = d.DcpResources(
                self.owner,
                self.native.options,
                self.module,
                widths=(1, 8),
                max_context=8192,
                deadline_ns=5_000_000_000,
            )

    def core(self, parity=0):
        r = self.resources
        core = NS(
            device=DEVICE,
            heads=12,
            max_seq=1024,
            dcp=r.box,
            parity=parity,
            deadline_ns=r.deadline_ns,
            cores={(1, 1): "tokenspeed", (8, 1): "tokenspeed"},
            pitches={1: 3648, 8: 3648},
            kernels={},
            weights=(),
            stock_scales=(0.25, 1.0),
        )
        core.run = Mock(return_value=object())
        core.prepare_cache = Mock(
            side_effect=lambda cache, prewarm: core.kernels.update(
                {(1, 1): object(), (8, 1): object()}
            )
        )
        return core


class FakeWorkspace(Workspace):
    def storage_offset(self):
        return 0


class FakeCache(FakeWorkspace):
    def __init__(self):
        super().__init__(2 * 64 * 576)
        self.dtype, self.shape = torch.uint8, (2, 64, 576)
        self.strides = (64 * 576, 576, 1)

    def is_contiguous(self):
        return True


class TestNativeOwner(unittest.TestCase):
    def test_exact_options_and_invalid_fields(self):
        self.assertEqual(d.DcpBuildOptions().cap, 64)
        for name, value in (
            ("size", 4),
            ("heads", 96),
            ("cap", 8),
            ("page_size", 128),
            ("interleave", 2),
            ("cap", True),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                d.DcpBuildOptions(**{name: value})

    def test_native_order_offsets_and_agreement_before_allocation(self):
        fixture = NativeFixture()
        owner = fixture.capture()
        owner.assert_live()
        self.assertEqual((owner.rank, owner.ranks), (3, tuple(range(16, 24))))
        self.assertIs(owner.device_group, fixture.parallel.dcp_group.device_group)
        self.assertFalse(hasattr(owner, "__dict__"))
        self.assertIs(
            fixture.agree.call_args.kwargs["group"], fixture.parallel.tp_group
        )

    def test_invalid_configuration_still_agrees_before_raising(self):
        for change in (
            {"tokens_per_request": 6},
            {"tokens_per_request": True},
            {"deadline_ns": 0},
            {"deadline_ns": True},
            {"deadline_ns": 1 << 63},
            {"mla_core": "trtllm"},
            {"device": torch.device("cpu")},
            {"widths": None},
            {"widths": [1, 8]},
            {"widths": (1, 1)},
            {"widths": (True,)},
            {"max_context": 0},
        ):
            fixture = NativeFixture()
            with self.subTest(change=change), self.assertRaises(ValueError):
                fixture.capture(**change)
            fixture.agree.assert_called_once()

    def test_sm103_and_capture_guards(self):
        f = NativeFixture()
        with (
            patch.object(
                d, "_native_context", return_value=(lambda: f.parallel, f.agree)
            ),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
            patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)),
        ):
            with self.assertRaisesRegex(ValueError, "SM103"):
                d.NativeDcpOwner.capture(f.options, **f.arguments)
        with (
            patch.object(
                d, "_native_context", return_value=(lambda: f.parallel, f.agree)
            ),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=True),
        ):
            with self.assertRaisesRegex(ValueError, "during capture"):
                d.NativeDcpOwner.capture(f.options, **f.arguments)

    def test_native_group_mutations_are_rejected(self):
        changes = (
            lambda p: setattr(p, "dcp_rank", 2),
            lambda p: setattr(p, "attn_dcp_size", 4),
            lambda p: p.dcp_group.ranks.reverse(),
            lambda p: p.dcp_group.ranks.__setitem__(0, 17),
            lambda p: setattr(p.dcp_group, "device_group", Token()),
            lambda p: setattr(p.tp_group, "cpu_group", Token()),
            lambda p: setattr(p, "dcp_group", parallel().dcp_group),
            lambda p: setattr(p.tp_group, "ranks", list(p.tp_group.ranks)),
        )
        for change in changes:
            f = NativeFixture()
            owner = f.capture()
            change(f.parallel)
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                owner.assert_live()

    def test_agreed_plan_and_retained_owner_replacement_fail_closed(self):
        f = NativeFixture()
        owner = f.capture()
        owner.check_plan(f.options, (1, 8), 8192, 5_000_000_000)
        for widths, context, deadline in (
            ((1,), 8192, 5_000_000_000),
            ((1, 8), 4096, 5_000_000_000),
            ((1, 8), 8192, 1),
        ):
            with self.assertRaisesRegex(ValueError, "all-rank agreed"):
                owner.check_plan(f.options, widths, context, deadline)
        owner._owners = (Token(), *owner._owners[1:])
        with self.assertRaisesRegex(RuntimeError, "native group changed"):
            owner.assert_live()

    def test_borrowed_owners_stay_alive_but_are_not_inventoried(self):
        f = NativeFixture()
        owner = f.capture()
        old = weakref.ref(f.parallel.dcp_group)
        f.parallel.dcp_group.borrowed_workspace = FakeWorkspace(128)
        f.parallel.dcp_group = parallel().dcp_group
        gc.collect()
        self.assertIsNotNone(old())
        fake_torch = NS(
            Tensor=FakeWorkspace,
            cuda=NS(Stream=type("Stream", (), {}), Event=type("Event", (), {})),
        )
        module = definitions(
            PROVIDERS,
            [
                "_owned_resource_inventory",
                "_tensor_row",
                "_resource_kind",
                "_handle_row",
            ],
            torch=fake_torch,
        )
        model = NS(modules=lambda: (), parameters=lambda: (), buffers=lambda: ())
        root = NS(
            boxes={},
            scratch={},
            workspaces={},
            runtime_handles={},
            attention={},
            moe_front_weights={},
            owner=owner,
        )
        self.assertEqual(module._owned_resource_inventory(root, model), [])
        root.owner = NS(borrowed=old())
        self.assertEqual(len(module._owned_resource_inventory(root, model)), 1)
        del root.owner, owner
        gc.collect()
        self.assertIsNone(old())


class TestFinitePrewarm(unittest.TestCase):
    def test_finite_inputs_exact_attention_geometry_and_untouched_mailboxes(self):
        f = ResourceFixture()
        r, core = f.resources, f.core()
        before = [tensor.clone() for tensor in (r.box.q, r.box.partial, r.box.lse)]
        with (
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
            patch.object(torch.cuda, "synchronize") as sync,
        ):
            for batch in (1, 8):
                r.prewarm(core, batch, 1)
                args = f.module.stock_decode.call_args.args
                self.assertEqual(tuple(args[0].shape), (batch, 1, 96, 576))
                self.assertTrue(bool(torch.isfinite(args[0].float()).all()))
                self.assertEqual(args[0].dtype, torch.float8_e4m3fn)
                self.assertIs(args[1], r.warm_cache)
                self.assertEqual(tuple(args[1].shape), (1, 64, 576))
                self.assertTrue(bool((args[1] == 0).all()))
                self.assertTrue(bool((args[2] == 0).all()))
                self.assertTrue(bool((args[3] == 1).all()))
                self.assertEqual(
                    (args[4], args[5], args[8]), (1024, core.stock_scales, 32)
                )
                self.assertEqual(args[6].data_ptr(), r.box.rows.data_ptr())
                self.assertEqual(args[7].data_ptr(), r.box.rows_lse.data_ptr())
            self.assertEqual(sync.call_count, 2)
        for tensor, snapshot in zip((r.box.q, r.box.partial, r.box.lse), before):
            self.assertTrue(
                torch.equal(tensor.view(torch.uint8), snapshot.view(torch.uint8))
            )

    def test_failed_native_prewarm_preserves_mailboxes_and_fails_closed(self):
        f = ResourceFixture()
        failure = RuntimeError("native prewarm failed")
        f.module.stock_decode.side_effect = failure
        before = [
            tensor.clone()
            for tensor in (
                f.resources.box.q,
                f.resources.box.partial,
                f.resources.box.lse,
            )
        ]
        with patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        ):
            with self.assertRaises(RuntimeError) as raised:
                f.resources.prewarm(f.core(), 1, 1)
        self.assertIs(raised.exception, failure)
        self.assertTrue(f.resources.failed)
        with self.assertRaisesRegex(RuntimeError, "failed preparation"):
            f.resources.assert_live()
        for tensor, snapshot in zip(
            (f.resources.box.q, f.resources.box.partial, f.resources.box.lse), before
        ):
            self.assertTrue(
                torch.equal(tensor.view(torch.uint8), snapshot.view(torch.uint8))
            )

    def test_prewarm_capture_unrequested_and_q6_rejected(self):
        f = ResourceFixture()
        with patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        ):
            for batch, q in ((2, 1), (6, 6)):
                with self.assertRaises(RuntimeError):
                    f.resources.prewarm(f.core(), batch, q)
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "during capture"):
                f.resources.prewarm(f.core(), 1, 1)
        f.module.stock_decode.assert_not_called()

    def test_box_storage_handles_and_placement_seals(self):
        for mutate in (
            lambda r: setattr(r.box, "q", r.box.q.clone()),
            lambda r: setattr(r.box, "handles", (Token(), *r.box.handles[1:])),
            lambda r: setattr(r.box, "device_group", Token()),
            lambda r: setattr(r.box, "cap", 8),
            lambda r: setattr(r, "deadline_ns", 1),
            lambda r: setattr(r.owner, "rank", 4),
        ):
            f = ResourceFixture()
            mutate(f.resources)
            with self.subTest(mutate=mutate), self.assertRaises(RuntimeError):
                f.resources.assert_live()

    def test_actual_prepare_cache_requires_dedicated_cuda_prewarm(self):
        module = definitions(PREPARED, ["PreparedMla"], to_cute=lambda value: value)
        core = object.__new__(module.PreparedMla)
        core.dcp = object()
        with self.assertRaisesRegex(ValueError, "finite prewarm owner"):
            core.prepare_cache(NS(is_cuda=True))
        core.dcp = None
        with self.assertRaisesRegex(ValueError, "only supported"):
            core.prepare_cache(NS(is_cuda=True), prewarm=Mock())


class TestWorkspace(unittest.TestCase):
    def prepared(self, f, size=64):
        core = f.core()
        adapter = d.PreparedDcpMla(core, f.resources)
        adapter.prepare_cache(object())
        f.module._WORKSPACE[DEVICE, 12] = FakeWorkspace(16)
        f.module._WORKSPACE[DEVICE, 96] = FakeWorkspace(size)
        return adapter

    def test_96_head_final_owner_and_local12_geometry(self):
        f = ResourceFixture()
        first = self.prepared(f, 64)
        second = self.prepared(f, 128)
        final = f.module._WORKSPACE[DEVICE, 96]
        with (
            patch.object(torch, "Tensor", FakeWorkspace),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
        ):
            self.assertIs(first.finalize_workspace(), final)
            self.assertIs(second.finalize_workspace(), final)
            self.assertEqual(first.heads, 12)
            first.check_workspace()
            self.assertIsNot(first.workspace, f.module._WORKSPACE[DEVICE, 12])
            f.module._WORKSPACE[DEVICE, 96] = FakeWorkspace(256)
            with self.assertRaisesRegex(RuntimeError, "workspace or geometry"):
                first.check_workspace()
            self.assertIs(first.workspace, final)

    def test_missing96_workspace_and_unprepared_rejected(self):
        f = ResourceFixture()
        adapter = d.PreparedDcpMla(f.core(), f.resources)
        with patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        ):
            with self.assertRaisesRegex(RuntimeError, "must prewarm"):
                adapter.finalize_workspace()
            adapter.prepare_cache(object())
            f.module._WORKSPACE[DEVICE, 12] = FakeWorkspace(64)
            with self.assertRaisesRegex(RuntimeError, "96-head"):
                adapter.finalize_workspace()

    def test_q6_different_box_device_or_missing_deadline_rejected(self):
        f = ResourceFixture()
        for name, value in (
            ("cores", {(6, 6): "tokenspeed"}),
            ("dcp", object()),
            ("heads", 96),
            ("deadline_ns", 0),
            ("device", torch.device("cpu")),
            ("parity", 2),
        ):
            core = f.core()
            setattr(core, name, value)
            with self.subTest(name=name), self.assertRaises(ValueError):
                d.PreparedDcpMla(core, f.resources)

    def test_workspace_capture_reprepare_and_geometry_mutation(self):
        f = ResourceFixture()
        adapter = self.prepared(f)
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "during capture"):
                adapter.finalize_workspace()
        with (
            patch.object(torch, "Tensor", FakeWorkspace),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
        ):
            adapter.finalize_workspace()
            with self.assertRaisesRegex(RuntimeError, "twice"):
                adapter.prepare_cache(object())
            adapter.core.parity = 1
            with self.assertRaisesRegex(RuntimeError, "geometry changed"):
                adapter.check_workspace()


class TestCacheHandoff(unittest.TestCase):
    def fixture(self):
        f = ResourceFixture()
        providers = {
            layer: NS(
                kind="mla",
                core=d.PreparedDcpMla(f.core(index % 2), f.resources),
                cache=None,
            )
            for index, layer in enumerate((0, 4))
        }
        providers[1] = NS(kind="kda", core=Mock())
        caches = {layer: FakeCache() for layer in (0, 4)}
        f.module._WORKSPACE[DEVICE, 96] = FakeWorkspace(256)
        return f, providers, caches

    def test_actual_handoff_retains_caches_and_never_enables_full_binding(self):
        f, providers, caches = self.fixture()
        module = definitions(
            PROVIDERS, ["ResourceSet", "_tensor_row", "_pointer_free_digest", "_digest"]
        )
        resources = object.__new__(module.ResourceSet)
        resources.dcp, resources.attention = f.resources, providers
        resources.workspaces, resources.resource_inventory = {}, []
        resources.binding_fingerprint = None
        with (
            patch.object(torch, "Tensor", FakeWorkspace),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
        ):
            resources.prepare_dcp_mla_caches(caches)
            self.assertTrue(f.resources._prepared)
            self.assertIsNone(resources.binding_fingerprint)
            self.assertEqual(len(resources.resource_inventory), 1)
            self.assertIs(
                resources.workspaces["mla.dcp.tokenspeed.workspace"],
                f.resources.workspace,
            )
            for layer in (0, 4):
                self.assertIs(providers[layer].cache, caches[layer])
                providers[layer].core.check_workspace()
                providers[layer].core.core.prepare_cache.assert_called_once_with(
                    caches[layer], prewarm=f.resources.prewarm
                )
            with self.assertRaisesRegex(RuntimeError, "already prepared"):
                resources.prepare_dcp_mla_caches(caches)
        providers[1].core.assert_not_called()

    def test_handoff_failure_keeps_owners_and_cannot_resume(self):
        f, providers, caches = self.fixture()
        original = RuntimeError("native second-layer preparation failed")
        providers[4].core.core.prepare_cache.side_effect = original
        with (
            patch.object(torch, "Tensor", FakeWorkspace),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
        ):
            with self.assertRaises(RuntimeError) as raised:
                f.resources.prepare_mla_caches(providers, caches)
            self.assertIs(raised.exception, original)
            self.assertTrue(f.resources.failed)
            self.assertIsNone(f.resources.workspace)
            for layer in (0, 4):
                self.assertIs(providers[layer].cache, caches[layer])
            with self.assertRaisesRegex(RuntimeError, "failed preparation"):
                f.resources.prepare_mla_caches(providers, caches)

    def test_bad_mapping_and_cache_fail_before_preparation(self):
        for change in (
            lambda caches: caches.pop(4),
            lambda caches: setattr(caches[4], "dtype", torch.float32),
            lambda caches: setattr(caches[4], "shape", (2, 32, 576)),
            lambda caches: setattr(caches[4], "device", torch.device("cpu")),
        ):
            f, providers, caches = self.fixture()
            change(caches)
            with (
                patch.object(torch, "Tensor", FakeWorkspace),
                patch.object(
                    torch.cuda, "is_current_stream_capturing", return_value=False
                ),
            ):
                with self.assertRaises(ValueError):
                    f.resources.prepare_mla_caches(providers, caches)
            providers[0].core.core.prepare_cache.assert_not_called()
            providers[4].core.core.prepare_cache.assert_not_called()


class TestBuilderAndBox(unittest.TestCase):
    def test_private_builder_passes_shared_box_ordinal_parity_and_deadline(self):
        fixture = BuildFixture()
        events = []
        owner, box = object(), object()
        resource = NS(box=box)
        fixture.modules["mla.dcp"] = object()
        options = d.DcpBuildOptions()
        with (
            patch("importlib.metadata.version", return_value="2.0.2"),
            patch.object(
                d.NativeDcpOwner,
                "capture",
                side_effect=lambda *a, **k: events.append("agree") or owner,
            ) as capture,
            patch.object(
                d,
                "DcpResources",
                side_effect=lambda *a, **k: events.append("allocate") or resource,
            ),
            patch.object(
                d, "PreparedDcpMla", side_effect=lambda core, resources: core
            ) as wrap,
        ):
            result = fixture.build(1, (1, 8), mla_core="tokenspeed", dcp=options)
        self.assertEqual(events, ["agree", "allocate"])
        self.assertIs(result.dcp, resource)
        self.assertIs(result.boxes["mla.dcp"], box)
        self.assertEqual(capture.call_args.kwargs["widths"], (1, 8))
        self.assertEqual(wrap.call_count, 24)
        for index, call in enumerate(
            fixture.modules["mla.prepared"].PreparedMla.call_args_list
        ):
            self.assertIs(call.kwargs["dcp"], box)
            self.assertEqual(call.kwargs["parity"], index % 2)
            self.assertEqual(call.kwargs["deadline_ns"], 123)
            self.assertEqual(call.kwargs["queries"], (1,))
        fixture.module._prepare_mla_workspace.assert_not_called()

    def test_default_builder_has_no_dcp_owner_or_box(self):
        fixture = BuildFixture()
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if "dcp_resources" in name:
                raise AssertionError("OFF imported DCP resources")
            return original_import(name, *args, **kwargs)

        with (
            patch.object(
                d.NativeDcpOwner,
                "capture",
                side_effect=AssertionError("OFF captured DCP"),
            ),
            patch.object(builtins, "__import__", side_effect=guarded_import),
        ):
            result = fixture.build(1, (1, 8), dcp=None)
        self.assertIsNone(result.dcp)
        self.assertNotIn("mla.dcp", result.boxes)
        for call in fixture.modules["mla.prepared"].PreparedMla.call_args_list:
            self.assertIsNone(call.kwargs["dcp"])
            self.assertNotIn("deadline_ns", call.kwargs)

    def test_actual_dcpbox_retains_original_handles_and_opaque_group(self):
        group, handles = Token(), []

        def rendezvous(tensor, actual_group):
            self.assertIs(actual_group, group)
            handle = Token()
            handle.multicast_ptr = 4096
            handle.buffer_ptrs = [8192 + rank * 4096 for rank in range(8)]
            handles.append(weakref.ref(handle))
            return handle

        fake_torch = NS(
            uint8=torch.uint8,
            bfloat16=torch.bfloat16,
            float32=torch.float32,
            int64=torch.int64,
            int32=torch.int32,
            empty=lambda *a, **k: torch.empty(*a, **(k | {"device": "cpu"})),
            tensor=lambda *a, **k: torch.tensor(*a, **(k | {"device": "cpu"})),
            accelerator=NS(synchronize=Mock()),
            distributed=NS(barrier=Mock()),
        )
        module = definitions(
            DCP,
            ["_DcpTransportOwner", "DcpBox"],
            torch=fake_torch,
            symm_mem=NS(empty=fake_torch.empty, rendezvous=rendezvous),
            Int64=int,
        )
        box = module.DcpBox(group, DEVICE, 8, 3, 12)
        gc.collect()
        self.assertTrue(all(ref() is not None for ref in handles))
        self.assertIs(box.device_group, group)
        self.assertFalse(hasattr(box._transport_owner, "__dict__"))
        self.assertEqual(box.mc(1) - box.mc(0), box.q[0].numel())
        del box
        gc.collect()
        self.assertTrue(all(ref() is None for ref in handles))

    def test_private_binding_handoff_does_not_set_full_executor_ready(self):
        source = ast.parse(PROVIDERS.read_text())
        resource = next(
            node for node in source.body if getattr(node, "name", None) == "ResourceSet"
        )
        handoff = next(
            node
            for node in resource.body
            if getattr(node, "name", None) == "prepare_dcp_mla_caches"
        )
        self.assertNotIn("binding_fingerprint =", ast.unparse(handoff))
        method = definitions(PROVIDERS, ["ResourceSet"])
        value = object.__new__(method.ResourceSet)
        value.dcp = object()
        with self.assertRaisesRegex(RuntimeError, "separate native hybrid binding"):
            value.bind(object())


if __name__ == "__main__":
    unittest.main()

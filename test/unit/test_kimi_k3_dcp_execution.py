# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Private host contracts; native binding/CUDA compilation are not simulated PASS."""

import ast
import sys
import unittest
from contextlib import ExitStack, contextmanager
from functools import partial
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch
from test_kimi_k3_mla_readiness import definitions

from sgl_nvidia_mega.kimi_k3 import dcp_execution as d

ROOT = Path(__file__).resolve().parents[2] / "sgl_nvidia_mega/kimi_k3"
DEVICE = torch.device("cuda:0")


class HostTensor(torch.Tensor):
    """Real CPU storage, explicit fake device ONLY for host ownership checks."""

    @property
    def device(self):
        return DEVICE


def tensor(shape, dtype=torch.float32):
    return torch.zeros(shape, dtype=dtype).as_subclass(HostTensor)


class Binding:
    def __init__(self, pool):
        self.pool = pool
        self.assert_live = Mock()

    def preparation_view(self):
        self.assert_live()
        return self.pool


class Step(NS):
    pass


class Bucket:
    def __init__(self, binding, batch=1):
        self.binding = binding
        self.view = Step(
            mla=NS(
                canonical_write_slots=tensor((batch,), torch.int64),
                local_sequence_lengths=tensor((batch,), torch.int32),
                global_sequence_lengths=tensor((batch,), torch.int32),
                local_block_table=tensor((batch, 2), torch.int32),
            ),
            kda=NS(
                state_indices=tensor((batch,), torch.int32),
                verify_scratch_indices=None,
                track_decode_state=Mock(),
            ),
        )
        self.step_view = Mock(return_value=self.view)
        self.update = Mock(side_effect=AssertionError("caller owns slot updates"))


class Dcp:
    def agreement_group(self):
        return self.owner.agreement_group()

    def assert_live(self):
        if self.failed:
            raise RuntimeError("poisoned DCP resources")


class Mla:
    def __init__(self, resources, parity):
        self.resources, self.core = resources, NS(parity=parity)
        self.check_workspace = Mock()


class Kda:
    def __init__(self):
        self.heads, self.device, self.conv_dim_first = 12, DEVICE, False
        self.policies, self.kernels = {1: (6288, "mode"), 8: (6288, "mode")}, {}
        self.prepare_caches = Mock(side_effect=self.prepare)

    def prepare(self, caches):
        self.kernels = {width: object() for width in self.policies}


class Fixture:
    def __init__(self):
        self.full = tuple(range(0, 93, 4))
        self.linear = tuple(n for n in range(93) if n not in self.full)
        self.tp, self.group = NS(), NS(device_group=object())
        self.dcp = Dcp()
        self.dcp.device, self.dcp.widths = DEVICE, (1, 8)
        self.dcp.deadline_ns, self.dcp.failed, self.dcp._prepared = (
            5_000_000_000,
            False,
            False,
        )
        self.dcp.owner = NS(
            rank=3,
            ranks=tuple(range(16, 24)),
            device_group=self.group.device_group,
            agreement_group=Mock(return_value=self.tp),
        )
        # Shared source pools with distinct native-shaped per-layer views.
        conv = tensor((69, 2, 3, 4608), torch.bfloat16)
        temporal = tensor((69, 2, 12, 128, 128))
        self.states = {
            layer: NS(
                conv=(conv[i],),
                temporal=temporal[i],
                intermediate_conv_window=(),
                intermediate_ssm=None,
            )
            for i, layer in enumerate(self.linear)
        }
        self.pool = NS(
            mla=NS(
                layout=NS(
                    size=8, rank=3, page_size=64, interleave=1, canonical_skip=-1
                ),
                write_space="static_widened",
                caches={
                    layer: tensor((2, 64, 576), torch.uint8) for layer in self.full
                },
                layer_mapping={layer: i for i, layer in enumerate(self.full)},
            ),
            kda=NS(
                capacity=1,
                layer_mapping={layer: i for i, layer in enumerate(self.linear)},
                state=self.states.__getitem__,
            ),
            group=NS(
                rank=3,
                global_ranks=self.dcp.owner.ranks,
                device_group=self.group.device_group,
                coordinator=self.group,
                tp_coordinator=self.tp,
            ),
        )
        self.binding = Binding(self.pool)
        self.attention = {
            layer: NS(kind="mla", core=Mla(self.dcp, i % 2), cache=None)
            for i, layer in enumerate(self.full)
        } | {layer: NS(kind="kda", core=Kda(), caches=None) for layer in self.linear}
        self.resources = NS(
            dcp=self.dcp,
            tokens_per_request=1,
            binding=None,
            binding_fingerprint=None,
            aux_capture_layers=None,
            prefetch=None,
            embedding=None,
            widths=(1, 8),
            deadline_ns=self.dcp.deadline_ns,
            model=NS(
                config=NS(
                    full_attention_layer_ids=self.full, linear_layer_ids=self.linear
                )
            ),
            layers=tuple(NS(layer_idx=i) for i in range(93)),
            attention=self.attention,
            compilation_inventory=[],
            resource_inventory=[{"untouched": True}],
            scratch={"hidden.0": tensor((8, 7168), torch.bfloat16)},
            prepare_dcp_mla_caches=Mock(side_effect=self.prepare_mla),
        )
        self.agree = Mock()

    def prepare_mla(self, caches):
        for layer in self.full:
            self.attention[layer].cache = caches[layer]
        self.dcp._prepared = True

    @contextmanager
    def boundaries(self):
        with ExitStack() as stack:
            for name, value in (
                ("_native_types", Mock(return_value=(Binding, Bucket, Step))),
                (
                    "_native_context",
                    Mock(return_value=(lambda: NS(tp_group=self.tp), self.agree)),
                ),
                ("DcpResources", Dcp),
                ("PreparedDcpMla", Mla),
                ("_kda_type", Mock(return_value=Kda)),
            ):
                stack.enter_context(patch.object(d, name, value))
            stack.enter_context(
                patch.object(
                    torch.cuda, "is_current_stream_capturing", return_value=False
                )
            )
            stack.enter_context(patch.object(torch.cuda, "synchronize"))
            yield

    def prepare(self):
        return d.PreparedDcpHybrid(self.resources, self.binding)


def executor_module():
    module = definitions(
        ROOT / "executor.py",
        [
            "_require_tensor",
            "_validate_step_inputs",
            "_verify_mla_metadata",
            "KimiK3MegakernelExecutor",
        ],
        partial=partial,
        VERIFY_QUERIES=(4, 5, 6, 7, 8),
        token_widths=lambda *a, **k: tuple(range(1, 65)),
    )
    module.__package__ = "sgl_nvidia_mega.kimi_k3"
    return module


class TestPrivateEntry(unittest.TestCase):
    def test_explicit_private_readiness(self):
        f = Fixture()
        with f.boundaries():
            ready = f.prepare()
            ready.assert_live()
            self.assertIsNone(f.resources.binding_fingerprint)
            self.assertIsNone(f.resources.binding)
            self.assertEqual(f.resources.resource_inventory, [{"untouched": True}])
            self.assertEqual(len(f.resources.compilation_inventory), 69)
            self.assertFalse(hasattr(ready, "__dict__"))
            for layer in f.linear:
                state = f.states[layer]
                f.attention[layer].core.prepare_caches.assert_called_once_with(
                    (state.conv[0], state.temporal)
                )
            self.assertEqual(f.agree.call_count, 2)
            agreed = f.agree.call_args_list[0].args[0]
            self.assertEqual(agreed, (None, ready.plan))
            self.assertIsNone(f.agree.call_args_list[1].args[0])
            self.assertNotIn("cuda", repr(ready.plan))
            bucket = Bucket(f.binding)
            self.assertIs(ready.step_view(f.resources, bucket), bucket.view)
            bucket.update.assert_not_called()

    def test_rejected_contracts_agree_before_compiling_and_poison(self):
        mutations = (
            lambda f: setattr(f.resources, "tokens_per_request", 6),
            lambda f: setattr(f.resources, "binding_fingerprint", "fake"),
            lambda f: setattr(f.resources, "prefetch", object()),
            lambda f: setattr(f.pool.mla.layout, "interleave", 2),
            lambda f: setattr(
                f.pool.group, "global_ranks", tuple(reversed(range(16, 24)))
            ),
            lambda f: setattr(f.pool.group, "device_group", object()),
            lambda f: f.pool.kda.layer_mapping.__setitem__(1, 9),
            lambda f: setattr(f.attention[0].core.core, "parity", 1),
            lambda f: setattr(f.attention[1].core, "conv_dim_first", True),
            lambda f: setattr(
                f.states[1], "temporal", tensor((2, 12, 128, 128), torch.bfloat16)
            ),
            lambda f: setattr(f.states[1], "intermediate_ssm", object()),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                f = Fixture()
                mutate(f)
                with f.boundaries(), self.assertRaises((ValueError, RuntimeError)):
                    f.prepare()
                f.agree.assert_called_once()
                self.assertIsNotNone(f.agree.call_args.args[0][0])
                f.resources.prepare_dcp_mla_caches.assert_not_called()
                self.assertTrue(f.dcp.failed)

    def test_partial_failure_retains_storage_and_cannot_retry(self):
        f = Fixture()
        failure = RuntimeError("compile failure")
        f.attention[2].core.prepare_caches.side_effect = failure
        with f.boundaries():
            with self.assertRaisesRegex(RuntimeError, "compile failure"):
                f.prepare()
            self.assertIs(f.attention[2].caches[0], f.states[2].conv[0])
            self.assertIs(f.attention[0].cache, f.pool.mla.caches[0])
            with self.assertRaisesRegex(ValueError, "RuntimeError: poisoned"):
                f.prepare()
        self.assertTrue(f.dcp.failed)
        self.assertIsNone(f.resources.binding_fingerprint)

    def test_owner_changes_and_wrong_buckets_are_rejected(self):
        f = Fixture()
        with f.boundaries():
            ready = f.prepare()
            for obj, name, replacement in (
                (f.resources, "widths", (1,)),
                (f.resources, "binding_fingerprint", "forged"),
                (
                    f.attention[1],
                    "caches",
                    (tensor((2, 3, 4608), torch.bfloat16), f.states[1].temporal),
                ),
                (f.attention[0], "cache", tensor((2, 64, 576), torch.uint8)),
                (f.attention[1], "core", Kda()),
                (f.binding, "pool", NS()),
            ):
                old = getattr(obj, name)
                setattr(obj, name, replacement)
                with (
                    self.subTest(name=name),
                    self.assertRaises((ValueError, RuntimeError)),
                ):
                    ready.assert_live()
                setattr(obj, name, old)
            ready.assert_live()
            with self.assertRaisesRegex(ValueError, "prepared native bucket"):
                ready.step_view(f.resources, Bucket(Binding(f.pool)))
            with self.assertRaisesRegex(ValueError, "prepared native bucket"):
                ready.step_view(NS(), Bucket(f.binding))

    def test_capture_and_second_preparation_are_rejected(self):
        f = Fixture()
        with (
            f.boundaries(),
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=True),
        ):
            with self.assertRaisesRegex(ValueError, "during capture"):
                f.prepare()
        self.assertTrue(f.dcp.failed)
        f = Fixture()
        with f.boundaries():
            ready = f.prepare()
            with self.assertRaisesRegex(ValueError, "one-shot"):
                f.prepare()
            with self.assertRaisesRegex(RuntimeError, "poisoned"):
                ready.assert_live()

    def test_collective_rejection_and_completion_failure_never_mark_ready(self):
        for stage in ("agreement", "mla", "completion", "peer completion"):
            with self.subTest(stage=stage):
                f = Fixture()
                failure = RuntimeError(stage)
                if stage == "agreement":
                    f.agree.side_effect = failure
                elif stage == "peer completion":
                    f.agree.side_effect = [None, failure]
                elif stage == "mla":
                    f.resources.prepare_dcp_mla_caches.side_effect = failure
                with (
                    f.boundaries(),
                    patch.object(
                        torch.cuda,
                        "synchronize",
                        side_effect=failure if stage == "completion" else None,
                    ),
                ):
                    with self.assertRaisesRegex(RuntimeError, stage):
                        f.prepare()
                self.assertTrue(f.dcp.failed)
                self.assertIsNone(f.resources.binding_fingerprint)
                if stage == "agreement":
                    f.resources.prepare_dcp_mla_caches.assert_not_called()

    def test_native_owner_and_compiled_kda_replacements_fail_before_execution(self):
        f = Fixture()
        with f.boundaries():
            ready = f.prepare()
            original = f.attention[1].core.kernels[8]
            f.attention[1].core.kernels[8] = object()
            with self.assertRaisesRegex(RuntimeError, "owners changed"):
                ready.assert_live()
            f.attention[1].core.kernels[8] = original
            f.binding.assert_live.side_effect = RuntimeError("native owner changed")
            with self.assertRaisesRegex(RuntimeError, "native owner changed"):
                ready.assert_live()

    def test_no_default_import_registration_or_legacy_fabrication(self):
        for filename in ("spec.py", "providers.py", "__init__.py"):
            self.assertNotIn("dcp_execution", (ROOT / filename).read_text())
        tree = ast.parse((ROOT / "dcp_execution.py").read_text())
        assigned = [
            ast.unparse(n)
            for n in ast.walk(tree)
            if isinstance(n, (ast.Assign, ast.AnnAssign))
        ]
        self.assertFalse(any("binding_fingerprint" in n for n in assigned))
        self.assertNotIn("._runner", (ROOT / "dcp_execution.py").read_text())
        with patch.object(d, "_native_types", return_value=(Binding, Bucket, Step)):
            with self.assertRaisesRegex(ValueError, "genuine native"):
                d.PreparedDcpHybrid(NS(dcp=None), object())

    def test_stale_owner_is_collectively_rejected_on_original_cpu_route(self):
        f = Fixture()
        f.dcp.assert_live = Mock(side_effect=RuntimeError("native owner stale"))
        with f.boundaries(), self.assertRaisesRegex(ValueError, "native owner stale"):
            f.prepare()
        f.agree.assert_called_once()
        self.assertIs(f.agree.call_args.kwargs["group"], f.tp)
        self.assertIn("native owner stale", f.agree.call_args.args[0][0])
        f.resources.prepare_dcp_mla_caches.assert_not_called()
        self.assertTrue(f.dcp.failed)

    def test_damaged_original_cpu_route_does_not_use_alternative_group(self):
        f = Fixture()
        f.dcp.owner.agreement_group.side_effect = RuntimeError("CPU route changed")
        with f.boundaries(), self.assertRaisesRegex(RuntimeError, "CPU route changed"):
            f.prepare()
        f.agree.assert_not_called()
        f.resources.prepare_dcp_mla_caches.assert_not_called()
        self.assertTrue(f.dcp.failed)


class TestPrivateExecutor(unittest.TestCase):
    def test_private_shape_validation_and_cpu_refusal(self):
        f = Fixture()
        view, module = Bucket(f.binding, 8).view, executor_module()
        ids = tensor((8,), torch.int64)
        with patch.dict(sys.modules, {"sgl_nvidia_mega.kimi_k3.executor": module}):
            d.validate_step(f.resources, ids, None, view)
            for obj, name, value in (
                (view.mla, "global_sequence_lengths", tensor((8,), torch.int64)),
                (view.mla, "canonical_write_slots", tensor((8,), torch.int32)),
                (view.mla, "local_block_table", tensor((8, 4), torch.int32)[:, ::2]),
                (view.kda, "verify_scratch_indices", tensor((8,), torch.int32)),
                (view.kda, "track_decode_state", None),
                (f.resources, "tokens_per_request", 6),
            ):
                old = getattr(obj, name)
                setattr(obj, name, value)
                with self.subTest(name=name), self.assertRaises(RuntimeError):
                    d.validate_step(f.resources, ids, None, view)
                setattr(obj, name, old)
            f.resources.scratch["hidden.0"] = torch.zeros(8, 7168)
            with self.assertRaisesRegex(RuntimeError, "CUDA q1"):
                d.validate_step(f.resources, ids, None, view)

    def test_private_entry_exact_mapping_tracking_and_no_implicit_slot_update(self):
        f = Fixture()
        module = executor_module()
        with (
            f.boundaries(),
            patch.dict(sys.modules, {"sgl_nvidia_mega.kimi_k3.executor": module}),
        ):
            ready = f.prepare()
            for batch in (1, 8):
                bucket = Bucket(f.binding, batch)
                executor = module.KimiK3MegakernelExecutor(f.resources)
                executor._run_layers = Mock(return_value="shared body")
                ids = tensor((batch,), torch.int64)
                self.assertEqual(
                    executor.run_dcp(ids, None, prepared=ready, bucket=bucket),
                    "shared body",
                )
                args, kwargs = executor._run_layers.call_args
                self.assertIs(args[2], bucket.view.kda)
                self.assertEqual(args[3:5], (batch, 1))
                self.assertIs(args[5], bucket.view.mla.canonical_write_slots)
                self.assertIs(args[6], bucket.view.mla.local_sequence_lengths)
                self.assertIs(args[7], bucket.view.mla)
                self.assertEqual(args[8], {})
                self.assertIs(kwargs["totals"], bucket.view.mla.global_sequence_lengths)
                bucket.update.assert_not_called()
                with self.assertRaisesRegex(RuntimeError, "not bound"):
                    executor.run(ids, None, bucket.view)

    def test_shared_layer_body_keeps_default_calls_and_native_tracking_order(self):
        module = executor_module()
        for private in (False, True):
            with self.subTest(private=private):
                events = []

                def core(layer):
                    return NS(
                        run=Mock(
                            side_effect=lambda *a, **k: events.append((layer, "run"))
                        )
                    )

                attention = {
                    i: NS(
                        kind="mla" if i % 4 == 0 else "kda",
                        projection=Mock(),
                        output=Mock(),
                        core=core(i),
                        cache=object(),
                        caches=(object(), object()),
                    )
                    for i in range(93)
                }
                scratch = {
                    key: torch.zeros(8, 7168)
                    for key in (
                        "prefix",
                        "blocks",
                        "hidden.0",
                        "hidden.1",
                        "hidden.out",
                        "mla.projected",
                        "kda.projected",
                        "attention.local",
                    )
                }
                resources = NS(
                    layers=tuple(NS(layer_idx=i) for i in range(93)),
                    attention=attention,
                    scratch=scratch,
                    before={i: Mock() for i in range(93)},
                    fused_before={},
                    feed_forward={i: Mock() for i in range(93)},
                    final=Mock(),
                )
                kda = NS(
                    state_indices=torch.zeros(8, dtype=torch.int32),
                    track_decode_state=lambda layer: events.append((layer, "track")),
                )
                table, lengths, slots, totals = (object() for _ in range(4))
                mla = NS(local_block_table=table) if private else NS(block_table=table)
                executor = module.KimiK3MegakernelExecutor(resources)
                kwargs = {"totals": totals} if private else {}
                result = executor._run_layers(
                    torch.zeros(8, dtype=torch.int64),
                    torch.zeros(8, 7168),
                    kda,
                    8,
                    1,
                    slots,
                    lengths,
                    mla,
                    {},
                    **kwargs,
                )
                self.assertEqual(result.data_ptr(), scratch["hidden.out"].data_ptr())
                expected = []
                for i in range(93):
                    expected.append((i, "run"))
                    if i % 4:
                        expected.append((i, "track"))
                    else:
                        call = attention[i].core.run.call_args.kwargs
                        self.assertIs(call["slots"], slots)
                        self.assertIs(call["block_table"], table)
                        self.assertIs(call["lengths"], lengths)
                        self.assertEqual(call["q"], 1)
                        self.assertEqual("totals" in call, private)
                        if private:
                            self.assertIs(call["totals"], totals)
                self.assertEqual(events, expected)

    def test_composed_private_owner_entry_runs_all93_host_providers(self):
        f, module = Fixture(), executor_module()
        with (
            f.boundaries(),
            patch.dict(sys.modules, {"sgl_nvidia_mega.kimi_k3.executor": module}),
        ):
            ready = f.prepare()
            r = f.resources
            r.scratch.update(
                {
                    key: tensor((8, 7168), torch.bfloat16)
                    for key in (
                        "prefix",
                        "blocks",
                        "hidden.1",
                        "hidden.out",
                        "mla.projected",
                        "kda.projected",
                        "attention.local",
                    )
                }
            )
            r.before, r.fused_before = {i: Mock() for i in range(93)}, {}
            r.feed_forward, r.final = {i: Mock() for i in range(93)}, Mock()
            for provider in r.attention.values():
                provider.projection, provider.output, provider.core.run = (
                    Mock(),
                    Mock(),
                    Mock(),
                )
            executor = module.KimiK3MegakernelExecutor(r)
            for batch in (1, 8, 1):
                bucket = Bucket(f.binding, batch)
                output = executor.run_dcp(
                    tensor((batch,), torch.int64),
                    tensor((batch, 7168), torch.bfloat16),
                    prepared=ready,
                    bucket=bucket,
                )
                self.assertEqual(tuple(output.shape), (batch, 7168))
                self.assertEqual(
                    [
                        call.args[0]
                        for call in bucket.view.kda.track_decode_state.call_args_list
                    ],
                    list(f.linear),
                )
                bucket.update.assert_not_called()
                for layer in f.full:
                    call = r.attention[layer].core.run.call_args.kwargs
                    self.assertIs(
                        call["totals"], bucket.view.mla.global_sequence_lengths
                    )
                    self.assertIs(call["slots"], bucket.view.mla.canonical_write_slots)
            self.assertIsNone(r.binding_fingerprint)


if __name__ == "__main__":
    unittest.main()

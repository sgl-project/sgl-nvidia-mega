# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Private q6 host contracts, not native imports, compilation or GPU proof.

The q1 fixture's real CPU storage explicitly reports a fake device. Only native
type/collective/compiler boundaries are replaced; the readiness validator and
executor definitions under test are the actual private implementation.
"""

import ast
import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch
from test_kimi_k3_dcp_execution import (
    DEVICE,
    ROOT,
    Binding,
)
from test_kimi_k3_dcp_execution import Fixture as DecodeFixture
from test_kimi_k3_dcp_execution import (
    Step,
    executor_module,
    tensor,
)

from sgl_nvidia_mega.kimi_k3 import dcp_verify_execution as d


class Kda:
    def __init__(self):
        self.heads, self.device, self.conv_dim_first = 12, DEVICE, False
        self.policies = {(1, 6): (6288, 1), (4, 6): (6288, 1)}
        self.kernels, self.verify_kernels, self.verify_layouts = {}, {}, {}
        self.prepare_caches = Mock(side_effect=AssertionError("q1 preparation"))
        self.prepare_verify_caches = Mock(side_effect=self.prepare)

    def prepare(self, caches, intermediates):
        self.verify_kernels = {key: object() for key in self.policies}
        self.verify_layouts = {
            key: tuple(
                (tuple(t.shape), tuple(t.stride())) for t in (*caches, *intermediates)
            )
            for key in self.policies
        }


class Bucket:
    def __init__(self, binding, requests=1):
        self.binding = binding
        rows = 6 * requests
        self.view = Step(
            mla=NS(
                tokens_per_request=6,
                canonical_write_slots=tensor((rows,), torch.int64),
                local_sequence_lengths=tensor((requests,), torch.int32),
                global_sequence_lengths=tensor((requests,), torch.int32),
                local_block_table=tensor((requests, 2), torch.int32),
            ),
            kda=NS(
                state_indices=tensor((requests,), torch.int32),
                verify_scratch_indices=tensor((requests,), torch.int32),
                track_decode_state=Mock(side_effect=AssertionError("decode tracking")),
            ),
            aux_hidden_states=tensor((rows, 2 * 7168), torch.bfloat16),
        )
        self.step_view = Mock(return_value=self.view)
        self.update = Mock(side_effect=AssertionError("caller owns slot updates"))


class Fixture(DecodeFixture):
    def __init__(self):
        super().__init__()
        self.events = []
        self.dcp.tokens_per_request = 6
        self.dcp.widths = (6, 24)
        self.pool.tokens_per_request = 6
        self.pool.aux_hidden_width = 2 * 7168
        # Host-only borrowed views share backing checkpoint storage to keep this
        # shape/ownership suite small. This makes no inter-layer isolation claim.
        conv = tensor((5, 6, 3, 4608), torch.bfloat16)
        ssm = tensor((5, 6, 12, 128, 128))
        for state in self.states.values():
            state.intermediate_conv_window = (conv,)
            state.intermediate_ssm = ssm
        for layer in self.linear:
            self.attention[layer].core = Kda()
            self.attention[layer].intermediates = None
        r = self.resources
        r.tokens_per_request, r.widths = 6, (6, 24)
        r.model.model = NS(dspark_layers_to_capture=[0, 92])
        r.before = {i: NS(bind_aux=Mock()) for i in range(93)}
        r.final = NS(bind_aux=Mock())
        r.scratch["hidden.0"] = tensor((24, 7168), torch.bfloat16)
        r.prepare_aux_capture = Mock(side_effect=self.prepare_aux)

    def prepare_aux(self):
        self.events.append("aux")
        self.resources.aux_capture_layers = tuple(
            self.resources.model.model.dspark_layers_to_capture
        )

    def prepare_mla(self, caches):
        self.events.append("mla")
        return super().prepare_mla(caches)

    def boundaries(self):
        return super().boundaries(d, Bucket, Kda)

    def prepare(self):
        return d.PreparedDcpVerify(self.resources, self.binding)


class TestPrivateVerifyReadiness(unittest.TestCase):
    def test_exact_borrowed_checkpoint_preparation_and_aux_first(self):
        f = Fixture()
        with f.boundaries():
            ready = f.prepare()
            ready.assert_live()
            self.assertEqual(f.events[:2], ["aux", "mla"])
            self.assertIsNone(f.resources.binding)
            self.assertIsNone(f.resources.binding_fingerprint)
            self.assertEqual(f.resources.resource_inventory, [{"untouched": True}])
            self.assertFalse(hasattr(ready, "__dict__"))
            for layer in f.linear:
                state, provider = f.states[layer], f.attention[layer]
                provider.core.prepare_caches.assert_not_called()
                provider.core.prepare_verify_caches.assert_called_once_with(
                    (state.conv[0], state.temporal),
                    (state.intermediate_conv_window[0], state.intermediate_ssm),
                )
                self.assertIs(provider.intermediates[1], state.intermediate_ssm)
            self.assertEqual(f.agree.call_count, 2)
            self.assertEqual(f.agree.call_args_list[0].args[0], (None, ready.plan))
            self.assertIsNone(f.agree.call_args_list[1].args[0])
            self.assertNotIn("cuda", repr(ready.plan))
            bucket = Bucket(f.binding, 4)
            self.assertIs(ready.step_view(f.resources, bucket), bucket.view)
            bucket.update.assert_not_called()

    def test_bad_geometry_checkpoint_capacity_and_aux_fail_before_compile(self):
        mutations = (
            lambda f: setattr(f.resources, "tokens_per_request", 1),
            lambda f: setattr(f.resources, "binding_fingerprint", "fake"),
            lambda f: setattr(f.resources, "prefetch", object()),
            lambda f: setattr(f.pool, "aux_hidden_width", 7168),
            lambda f: setattr(
                f.resources.model.model, "dspark_layers_to_capture", [0, 0]
            ),
            lambda f: setattr(f.states[1], "intermediate_conv_window", ()),
            lambda f: setattr(f.states[1], "intermediate_ssm", None),
            lambda f: setattr(
                f.states[1],
                "intermediate_ssm",
                tensor((5, 6, 12, 128, 128), torch.bfloat16),
            ),
            lambda f: setattr(
                f.states[1],
                "intermediate_conv_window",
                (tensor((3, 6, 3, 4608), torch.bfloat16),),
            ),
            lambda f: setattr(
                f.states[1],
                "intermediate_conv_window",
                (tensor((5, 5, 3, 4608), torch.bfloat16),),
            ),
            lambda f: f.attention[1].core.policies.__setitem__((4, 6), (6288, 2)),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                f = Fixture()
                mutate(f)
                with f.boundaries(), self.assertRaises((ValueError, RuntimeError)):
                    f.prepare()
                self.assertTrue(f.dcp.failed)
                f.resources.prepare_dcp_mla_caches.assert_not_called()
                f.agree.assert_called_once()
                self.assertIsNotNone(f.agree.call_args.args[0][0])

    def test_partial_compile_poison_and_retained_borrowed_owners(self):
        f = Fixture()
        f.attention[2].core.prepare_verify_caches.side_effect = RuntimeError(
            "verify compile failed"
        )
        with f.boundaries():
            with self.assertRaisesRegex(RuntimeError, "verify compile failed"):
                f.prepare()
            self.assertIs(f.attention[2].intermediates[1], f.states[2].intermediate_ssm)
            self.assertIs(f.attention[2].caches[0], f.states[2].conv[0])
            with self.assertRaises((ValueError, RuntimeError)):
                f.prepare()
        self.assertTrue(f.dcp.failed)
        self.assertIsNone(f.resources.binding_fingerprint)

    def test_checkpoint_compiled_owner_taps_and_bucket_replacement_rejected(self):
        f = Fixture()
        with f.boundaries():
            ready = f.prepare()
            for obj, name, replacement in (
                (f.states[1], "intermediate_ssm", f.states[1].intermediate_ssm.clone()),
                (
                    f.attention[1],
                    "intermediates",
                    (
                        f.states[1].intermediate_conv_window[0],
                        f.states[1].intermediate_ssm.clone(),
                    ),
                ),
                (f.resources, "aux_capture_layers", (1, 92)),
                (f.resources.model.model, "dspark_layers_to_capture", [1, 92]),
            ):
                old = getattr(obj, name)
                setattr(obj, name, replacement)
                with (
                    self.subTest(name=name),
                    self.assertRaises((ValueError, RuntimeError)),
                ):
                    ready.assert_live()
                setattr(obj, name, old)
            original = f.attention[1].core.verify_kernels[(4, 6)]
            f.attention[1].core.verify_kernels[(4, 6)] = object()
            with self.assertRaises(RuntimeError):
                ready.assert_live()
            f.attention[1].core.verify_kernels[(4, 6)] = original
            ready.assert_live()
            with self.assertRaises(ValueError):
                ready.step_view(f.resources, Bucket(Binding(f.pool)))
            with self.assertRaises(ValueError):
                ready.step_view(NS(), Bucket(f.binding))

    def test_capture_second_prepare_and_original_route_failure_poison(self):
        for phase in ("capture", "second", "route", "completion"):
            with self.subTest(phase=phase):
                f = Fixture()
                with f.boundaries():
                    if phase == "second":
                        ready = f.prepare()
                    elif phase == "route":
                        f.dcp.owner.agreement_group.side_effect = RuntimeError(
                            "route changed"
                        )
                    elif phase == "completion":
                        f.agree.side_effect = [None, RuntimeError("peer completion")]
                    with patch.object(
                        torch.cuda,
                        "is_current_stream_capturing",
                        return_value=phase == "capture",
                    ):
                        with self.assertRaises((ValueError, RuntimeError)):
                            f.prepare()
                    if phase == "second":
                        with self.assertRaises(RuntimeError):
                            ready.assert_live()
                self.assertTrue(f.dcp.failed)
                if phase == "route":
                    f.agree.assert_not_called()


class TestPrivateVerifyExecutor(unittest.TestCase):
    def test_exact_shapes_cpu_refusal_and_scratch_zero_is_not_padding(self):
        f, module = Fixture(), executor_module()
        view = Bucket(f.binding, 4).view
        ids = tensor((24,), torch.int64)
        view.kda.state_indices.copy_(torch.tensor([1, 1, 1, -1], dtype=torch.int32))
        view.kda.verify_scratch_indices.copy_(torch.arange(4, dtype=torch.int32))
        with patch.dict(sys.modules, {"sgl_nvidia_mega.kimi_k3.executor": module}):
            f.resources.aux_capture_layers = (0, 92)
            d.validate_step(f.resources, ids, None, view)
            for obj, name, value in (
                (view.mla, "global_sequence_lengths", tensor((24,), torch.int32)),
                (view.mla, "canonical_write_slots", tensor((4,), torch.int64)),
                (view.kda, "verify_scratch_indices", None),
                (view.kda, "verify_scratch_indices", tensor((4,), torch.int64)),
                (view, "aux_hidden_states", tensor((24, 7168), torch.bfloat16)),
                (view, "aux_hidden_states", tensor((24, 2 * 7168))),
                (f.resources, "tokens_per_request", 1),
                (f.resources, "widths", (6,)),
            ):
                old = getattr(obj, name)
                setattr(obj, name, value)
                with self.subTest(name=name), self.assertRaises(RuntimeError):
                    d.validate_step(f.resources, ids, None, view)
                setattr(obj, name, old)
            f.resources.scratch["hidden.0"] = torch.zeros(24, 7168)
            with self.assertRaises(RuntimeError):
                d.validate_step(f.resources, ids, None, view)

    def test_private_entry_forwards_owned_mla_and_borrowed_aux_without_update(self):
        f, module = Fixture(), executor_module()
        with (
            f.boundaries(),
            patch.dict(sys.modules, {"sgl_nvidia_mega.kimi_k3.executor": module}),
        ):
            ready = f.prepare()
            for requests in (1, 4):
                bucket = Bucket(f.binding, requests)
                executor = module.KimiK3MegakernelExecutor(f.resources)
                executor._run_layers = Mock(return_value="shared q6 body")
                ids = tensor((6 * requests,), torch.int64)
                with patch.object(
                    module,
                    "_verify_mla_metadata",
                    side_effect=AssertionError("second metadata normalization"),
                ):
                    self.assertEqual(
                        executor.run_dcp_verify(
                            ids, None, prepared=ready, bucket=bucket
                        ),
                        "shared q6 body",
                    )
                args, kwargs = executor._run_layers.call_args
                self.assertIs(args[2], bucket.view.kda)
                self.assertEqual(args[3:5], (6 * requests, 6))
                self.assertIs(args[5], bucket.view.mla.canonical_write_slots)
                self.assertIs(args[6], bucket.view.mla.local_sequence_lengths)
                self.assertIs(args[7], bucket.view.mla)
                self.assertIs(kwargs["totals"], bucket.view.mla.global_sequence_lengths)
                self.assertEqual(set(args[8]), {0, 92})
                for index, layer in enumerate((0, 92)):
                    expected = bucket.view.aux_hidden_states[
                        :, index * 7168 : (index + 1) * 7168
                    ]
                    self.assertEqual(args[8][layer].data_ptr(), expected.data_ptr())
                    self.assertEqual(args[8][layer].stride(), expected.stride())
                bucket.update.assert_not_called()
                bucket.view.kda.track_decode_state.assert_not_called()
                with self.assertRaises(RuntimeError):
                    executor.run(ids, None, bucket.view)

    def test_no_registration_fingerprint_fabrication_or_native_accept_replacement(self):
        for filename in ("spec.py", "providers.py", "__init__.py"):
            self.assertNotIn("dcp_verify_execution", (ROOT / filename).read_text())
        source = (ROOT / "dcp_verify_execution.py").read_text()
        tree = ast.parse(source)
        writes = [
            ast.unparse(n)
            for n in ast.walk(tree)
            if isinstance(n, (ast.Assign, ast.AnnAssign))
        ]
        self.assertFalse(any("binding_fingerprint" in n for n in writes))
        self.assertNotIn("._runner", source)
        self.assertNotIn("update_mamba_state_after_mtp_verify", source)
        self.assertNotIn("commit_hidden", source)


if __name__ == "__main__":
    unittest.main()

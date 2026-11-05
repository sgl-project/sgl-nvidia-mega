# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Source-isolated embedding ownership/policy tests and a CUDA gather check."""

from __future__ import annotations

import ast
import contextlib
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from _kimi_test_utils import execution_resources, inventory_fixture, resource_fields

from sgl_nvidia_mega.kimi_k3 import embedding

ROOT = Path(embedding.__file__).parent


def _class_from_source(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    exec(  # noqa: S102 -- execute only the named class from this checkout.
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace[name]


class _Unquantized:
    pass


class _Embedding:
    def __init__(self, group, start=8, end=11, rows=4):
        self.enable_tp = True
        self.tp_group = group
        self.tp_size = group.world_size
        self.use_attn_tp_group = False
        self.quant_method = _Unquantized()
        self.output_dtype = None
        self.num_added_embeddings = 0
        self.num_embeddings = self.org_vocab_size = 11
        self.num_embeddings_per_partition = rows
        self.weight = torch.arange(rows, dtype=torch.bfloat16)[:, None].repeat(1, 7168)
        self.shard_indices = SimpleNamespace(
            org_vocab_start_index=start,
            org_vocab_end_index=end,
            padded_org_vocab_start_index=start,
            padded_org_vocab_end_index=start + rows,
            num_org_elements_padded=rows,
        )
        self.native = Mock(return_value=object())

    def __call__(self, ids):
        return self.native(ids)


def _native_modules(scattered=False):
    modules = {}
    context = SimpleNamespace(input_scattered=scattered)
    definitions = {
        "sglang.srt.layers.quantization.unquant": {
            "UnquantizedEmbeddingMethod": _Unquantized
        },
        "sglang.srt.layers.vocab_parallel_embedding": {
            "VocabParallelEmbedding": _Embedding
        },
        "sglang.srt.layers.layer_boundary": {"get_attn_tp_context": lambda: context},
        "sglang.srt.utils.async_probe": {"maybe_detect_oob": Mock()},
    }
    for name, values in definitions.items():
        module = types.ModuleType(name)
        module.__dict__.update(values)
        modules[name] = module
    return modules


class TestEmbeddingPolicy(unittest.TestCase):
    def setUp(self):
        self.group = SimpleNamespace(world_size=8)
        self.embed = _Embedding(self.group)
        self.core = SimpleNamespace(run=Mock(side_effect=lambda ids, out: out))
        self.constructor = Mock(return_value=self.core)
        self.loader = Mock(return_value=self.constructor)
        self.modules = _native_modules()

    def prepare(self):
        with patch.dict("sys.modules", self.modules):
            return embedding.prepare_embedding(
                self.embed, tp_group=self.group, load_prepared=self.loader
            )

    def test_native_padded_bounds_and_borrowed_weight(self):
        prepared = self.prepare()
        self.assertIs(prepared.embedding, self.embed)
        self.constructor.assert_called_once_with(
            self.embed.weight, start=8, end=11, group=self.group
        )
        ids, out = torch.tensor([8, 10]), torch.empty(2, 7168)
        with patch.dict("sys.modules", self.modules):
            self.assertIs(prepared.run(ids, out), out)
        self.modules[
            "sglang.srt.utils.async_probe"
        ].maybe_detect_oob.assert_called_once()
        self.core.run.assert_called_once_with(ids, out)
        self.embed.native.assert_not_called()

    def test_empty_padded_shard_uses_native_empty_bounds(self):
        self.embed = _Embedding(self.group, start=11, end=11)
        self.embed.shard_indices.padded_org_vocab_start_index = 12
        self.embed.shard_indices.padded_org_vocab_end_index = 16
        self.assertIsNotNone(self.prepare())
        self.assertEqual(self.constructor.call_args.kwargs["start"], 11)
        self.assertEqual(self.constructor.call_args.kwargs["end"], 11)

    def test_unsupported_native_policies_do_not_load_kernel(self):
        changes = (
            ("enable_tp", False),
            ("tp_size", 1),
            ("tp_group", object()),
            ("use_attn_tp_group", True),
            ("quant_method", object()),
            ("output_dtype", torch.float32),
            ("num_added_embeddings", 1),
            ("num_embeddings", 12),
            ("num_embeddings_per_partition", 5),
            ("weight", torch.empty(4, 7168)),
            ("weight", torch.empty(4, 7168, dtype=torch.bfloat16).t()),
        )
        for name, value in changes:
            with self.subTest(name=name, value=type(value).__name__):
                self.embed = _Embedding(self.group)
                setattr(self.embed, name, value)
                self.assertIsNone(self.prepare())
        self.loader.assert_not_called()

    def test_malformed_shard_bounds_fall_back(self):
        for name, value in (
            ("org_vocab_start_index", 7),
            ("org_vocab_end_index", 12),
            ("padded_org_vocab_end_index", 10),
            ("num_org_elements_padded", 5),
        ):
            with self.subTest(name=name):
                self.embed = _Embedding(self.group)
                setattr(self.embed.shard_indices, name, value)
                self.assertIsNone(self.prepare())
        self.loader.assert_not_called()

    def test_custom_embedding_and_absent_group_stay_native(self):
        with patch.dict("sys.modules", self.modules):
            self.assertIsNone(
                embedding.prepare_embedding(
                    self.embed, tp_group=None, load_prepared=self.loader
                )
            )
            self.assertIsNone(
                embedding.prepare_embedding(
                    object(), tp_group=self.group, load_prepared=self.loader
                )
            )
        self.loader.assert_not_called()

    def test_scattered_rows_retain_native_no_reduce_policy(self):
        prepared = self.prepare()
        ids = torch.tensor([8])
        context = self.modules["sglang.srt.layers.layer_boundary"].get_attn_tp_context()
        context.input_scattered = True
        result = prepared.run(ids, torch.empty(1, 7168))
        self.assertIs(result, self.embed.native.return_value)
        self.embed.native.assert_called_once_with(ids)
        self.core.run.assert_not_called()
        context.input_scattered = False
        out = torch.empty(1, 7168)
        self.assertIs(prepared.run(ids, out), out)
        self.core.run.assert_called_once_with(ids, out)


class TestEmbeddingAdmission(unittest.TestCase):
    def test_inventory_excludes_borrowed_weight_and_counts_output_once(self):
        from test_kimi_k3_prefetch import _Tensor

        class Group:
            def __init__(self):
                self.communicator_buffer = _Tensor(pointer=16384)

            def all_reduce(self, out):
                return out

        module, _, _ = inventory_fixture(ROOT / "providers.py", _Tensor)
        weight, output = _Tensor(), _Tensor((64, 7168), pointer=8192)
        group = Group()
        core = SimpleNamespace(weight=weight, all_reduce=group.all_reduce)
        self.assertIs(core.all_reduce.__self__, group)
        model = SimpleNamespace(
            modules=lambda: (), parameters=lambda: (weight,), buffers=lambda: ()
        )
        resources = SimpleNamespace(
            **resource_fields(
                scratch={"embedding.out": output},
                embedding=SimpleNamespace(core=core, output=output),
            )
        )
        inventory = module._owned_resource_inventory(resources, model)
        self.assertEqual(
            inventory,
            [
                {
                    "kind": "scratch",
                    "name": "resources.scratch.embedding.out",
                    "data_ptr": output.data_ptr(),
                }
            ],
        )


class TestPreparedEmbeddingHost(unittest.TestCase):
    def setUp(self):
        self.prepared_type = _class_from_source(
            ROOT / "cutedsl/common/embedding.py", "PreparedEmbedding", {"torch": torch}
        )

    def core(self, *, new_reduction=False):
        result = object.__new__(self.prepared_type)
        result.weight = torch.arange(4, dtype=torch.bfloat16)[:, None].repeat(1, 7168)
        result.start, result.vocab = 8, 11

        def gather(ids, weight, out, start, end):
            out.zero_()
            valid = (ids >= start) & (ids < end)
            out[valid] = weight[ids[valid].long() - start]

        result.kernels = {
            torch.int32: Mock(side_effect=gather),
            torch.int64: Mock(side_effect=gather),
        }
        result.all_reduce = Mock(
            side_effect=lambda out: out.clone() if new_reduction else out
        )
        return result

    def test_boundaries_dtype_and_out_of_place_reduction_preserve_output(self):
        for dtype in (torch.int32, torch.int64):
            for new_reduction in (False, True):
                core = self.core(new_reduction=new_reduction)
                ids = torch.tensor([-1, 0, 7, 8, 9, 10, 11, 12], dtype=dtype)
                out = torch.full((8, 7168), -99, dtype=torch.bfloat16)
                ptr = out.data_ptr()
                self.assertIs(core.run(ids, out), out)
                self.assertEqual(out.data_ptr(), ptr)
                self.assertEqual(out[:, 0].tolist(), [0, 0, 0, 0, 1, 2, 0, 0])
                core.all_reduce.assert_called_once_with(out)

    def test_bad_runtime_buffers_fail_before_launch_or_collective(self):
        for ids, out in (
            (torch.ones(2), torch.empty(2, 7168, dtype=torch.bfloat16)),
            (
                torch.ones(2, 1, dtype=torch.int64),
                torch.empty(2, 7168, dtype=torch.bfloat16),
            ),
            (torch.arange(4)[::2], torch.empty(2, 7168, dtype=torch.bfloat16)),
            (torch.arange(2), torch.empty(2, 7168)),
            (torch.arange(2), torch.empty(2, 1, dtype=torch.bfloat16)),
            (torch.arange(2), torch.empty(7168, 2, dtype=torch.bfloat16).t()),
        ):
            core = self.core()
            with self.assertRaises(ValueError):
                core.run(ids, out)
            core.all_reduce.assert_not_called()
            for kernel in core.kernels.values():
                kernel.assert_not_called()

    def test_constructor_borrows_weight_and_compiles_both_id_types(self):
        weight = torch.empty(4, 7168, dtype=torch.bfloat16)
        compiler = Mock(return_value=object())
        namespace = {
            "torch": torch,
            "architecture": Mock(return_value="sm_103"),
            "compile_gather": compiler,
        }
        prepared_type = _class_from_source(
            ROOT / "cutedsl/common/embedding.py", "PreparedEmbedding", namespace
        )
        group = SimpleNamespace(all_reduce=Mock())
        with (
            patch.object(
                torch.Tensor,
                "is_cuda",
                new_callable=unittest.mock.PropertyMock,
                return_value=True,
            ),
            patch.object(
                torch.accelerator, "device_index", return_value=contextlib.nullcontext()
            ),
        ):
            core = prepared_type(weight, start=8, end=11, group=group)
        self.assertEqual(core.weight.data_ptr(), weight.data_ptr())
        self.assertIs(core.all_reduce, group.all_reduce)
        self.assertEqual(set(core.kernels), {torch.int32, torch.int64})
        self.assertEqual(compiler.call_count, 2)
        with self.assertRaises(ValueError):
            prepared_type(weight, start=8, end=13, group=group)


class TestExecutorEmbedding(unittest.TestCase):
    def test_inputs_embeds_native_and_prepared_paths(self):
        for q, rows in ((1, 1), (1, 64), (6, 6), (6, 60)):
            for route in ("provided", "native", "prepared"):
                with self.subTest(q=q, rows=rows, route=route):
                    namespace = {
                        "torch": torch,
                        "Any": object,
                        "HybridStepView": object,
                        "token_widths": lambda q, rows=rows: (rows,),
                        "_validate_step_inputs": lambda *args, **kwargs: None,
                        "_verify_mla_metadata": lambda slots, lengths, writes, scratch, q: (
                            writes,
                            lengths,
                        ),
                    }
                    executor_type = _class_from_source(
                        ROOT / "executor.py", "KimiK3MegakernelExecutor", namespace
                    )
                    x = torch.ones(rows, 7168, dtype=torch.bfloat16)
                    native = Mock(return_value=x)
                    prepared = Mock()
                    prepared.run.side_effect = lambda ids, out, x=x: out.copy_(x)
                    before = Mock()
                    # Stop at the first layer: test the real executor's input
                    # selection and buffer ownership, not a mock 93-layer model.
                    before.run.side_effect = StopIteration
                    resources = execution_resources(
                        rows,
                        layers=(0,),
                        tokens_per_request=q,
                        tensor=lambda shape: torch.empty(shape, dtype=torch.bfloat16),
                    )
                    resources.scratch["embedding.out"] = torch.empty_like(x)
                    resources.model = SimpleNamespace(
                        model=SimpleNamespace(embed_tokens=native)
                    )
                    resources.embedding = prepared if route != "native" else None
                    resources.before[0] = before
                    scratch = resources.scratch
                    view = SimpleNamespace(
                        mla=SimpleNamespace(write_slots=None, sequence_lengths=None),
                        kda=SimpleNamespace(state_indices=None),
                    )
                    with self.assertRaises(StopIteration):
                        executor_type(resources).run(
                            torch.zeros(rows, dtype=torch.int64),
                            x if route == "provided" else None,
                            view,
                        )
                    selected = before.run.call_args.args[0]
                    if route == "prepared":
                        native.assert_not_called()
                        prepared.run.assert_called_once()
                        self.assertEqual(
                            selected.data_ptr(), scratch["embedding.out"].data_ptr()
                        )
                    else:
                        prepared.run.assert_not_called()
                        self.assertIs(selected, x)
                        self.assertEqual(native.call_count, int(route == "native"))


@unittest.skipUnless(
    torch.cuda.is_available(), "CUDA gather qualification requires a GPU"
)
class TestEmbeddingCuda(unittest.TestCase):
    def test_real_gather_eager_and_graph_replay(self):
        from sgl_nvidia_mega.kimi_k3.cutedsl.common.embedding import PreparedEmbedding

        weight = torch.arange(4, device="cuda", dtype=torch.bfloat16)[:, None].repeat(
            1, 7168
        )
        group = SimpleNamespace(all_reduce=lambda out: out)
        core = PreparedEmbedding(weight, start=8, end=11, group=group)
        for dtype in (torch.int32, torch.int64):
            for rows in (1, 2, 6, 12, 60, 64):
                ids = torch.arange(rows, device="cuda", dtype=dtype) % 6 + 7
                out = torch.empty(rows, 7168, device="cuda", dtype=torch.bfloat16)
                expected = torch.zeros_like(out)
                valid = (ids >= 8) & (ids < 11)
                expected[valid] = weight[ids[valid].long() - 8]
                core.run(ids, out)
                torch.cuda.synchronize()
                torch.testing.assert_close(out, expected, rtol=0, atol=0)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    core.run(ids, out)
                pointer = out.data_ptr()
                for offset in (0, 1, 0):
                    ids.copy_(
                        (torch.arange(rows, device="cuda", dtype=dtype) + offset) % 6
                        + 7
                    )
                    graph.replay()
                    expected.zero_()
                    valid = (ids >= 8) & (ids < 11)
                    expected[valid] = weight[ids[valid].long() - 8]
                    torch.testing.assert_close(out, expected, rtol=0, atol=0)
                    self.assertEqual(out.data_ptr(), pointer)


if __name__ == "__main__":
    unittest.main()

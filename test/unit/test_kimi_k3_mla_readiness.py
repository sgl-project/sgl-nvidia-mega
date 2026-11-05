# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""CPU-only MLA preparation contracts; no native compilation or GPU qualification.

Execute actual source definitions without importing SGLang/CuTe dependencies.
Native GPU workspace, numerical, stream and graph validation remain separate.
"""

from __future__ import annotations

import sys
import unittest
from contextlib import nullcontext
from math import prod
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch
from _kimi_test_utils import definitions, resource_fields

ROOT = Path(__file__).resolve().parents[2]
KIMI = ROOT / "sgl_nvidia_mega/kimi_k3"
PROVIDERS = KIMI / "providers.py"
DCP = KIMI / "cutedsl/mla/dcp.py"


class TestStandaloneSourceContract(unittest.TestCase):
    def test_factory_delta_and_runtime_argument_order(self):
        calls = []
        md = NS(
            select_mla_decode_tilers=lambda *args, **kwargs: ((128, 128), None),
            get_mla_decode_fold_sq_factor=lambda *args: 1,
            get_num_sm=lambda device: 1,
            _get_split_kv_and_workspace_size=lambda *args: (2, 16),
            _get_reducer_d_tiles=lambda *args: 1,
            _get_reducer_max_splits=lambda *args: 8,
            _get_compiled_mla_kernel=lambda **kwargs: (
                calls.append(kwargs) or (lambda *args: calls.append(args))
            ),
        )
        fake_cuda = NS(
            get_device_capability=lambda device: (10, 3),
            is_current_stream_capturing=lambda: False,
        )
        fake_torch = NS(
            float8_e4m3fn=torch.float8_e4m3fn,
            int8=torch.int8,
            empty=torch.empty,
            cuda=fake_cuda,
        )
        namespace = definitions(
            DCP,
            ["stock_decode"],
            torch=fake_torch,
            _WORKSPACE={},
            Int32=int,
            Float32=float,
        )
        query = torch.zeros((1, 6, 12, 576), dtype=torch.uint8)
        cache = torch.zeros((1, 64, 576), dtype=torch.uint8)
        table, lengths = (
            torch.zeros((1, 1), dtype=torch.int32),
            torch.tensor([31], dtype=torch.int32),
        )
        out = torch.zeros((1, 6, 12, 512), dtype=torch.bfloat16)
        with patch.dict(
            sys.modules,
            {
                "tokenspeed_mla": NS(mla_decode=md),
                "tvm_ffi": NS(use_torch_stream=nullcontext),
            },
        ):
            namespace.stock_decode(
                query, cache, table, lengths, 262150, (0.25, 1.0), out
            )
        factory, runtime = calls
        self.assertIs(factory["has_local_visible_lens"], False)
        self.assertIs(factory["causal_mask"], True)
        self.assertIs(factory["partial_fp16"], False)
        self.assertIs(factory["use_pdl"], True)
        self.assertEqual((factory["num_heads"], factory["seq_len_q"]), (12, 6))
        self.assertEqual(len(runtime), 14)
        self.assertIs(runtime[4], table)
        self.assertIs(runtime[5], out)
        self.assertIs(runtime[9], lengths)
        self.assertIs(runtime[10], lengths)
        self.assertIsNone(runtime[11])
        self.assertEqual(runtime[12:], (0.25, 1.0))
        self.assertEqual(lengths.tolist(), [31])


class Workspace:
    """Host metadata only, deliberately not a CUDA allocation."""

    next_pointer = 4096

    def __init__(self, size=64):
        self.device, self.dtype = torch.device("cuda:0"), torch.int8
        self.shape, self.strides = (size,), (1,)
        self.pointer = self.storage_pointer = Workspace.next_pointer
        Workspace.next_pointer += 4096
        self.storage_bytes = size

    @property
    def ndim(self):
        return len(self.shape)

    def numel(self):
        return prod(self.shape)

    def stride(self):
        return self.strides

    def is_contiguous(self):
        return self.strides == (1,)

    def data_ptr(self):
        return self.pointer

    def untyped_storage(self):
        return NS(
            data_ptr=lambda: self.storage_pointer, nbytes=lambda: self.storage_bytes
        )


class OwnershipFixture:
    def __init__(self, q=1):
        self.capturing, self.events = False, []
        fake_torch = NS(
            Tensor=Workspace,
            int8=torch.int8,
            cuda=NS(
                is_current_stream_capturing=lambda: self.capturing,
                synchronize=lambda: self.events.append("synchronize"),
            ),
        )
        self.module = definitions(
            PROVIDERS,
            [
                "_PreparedTokenSpeedMla",
                "ResourceSet",
                "_digest",
                "_pointer_free_digest",
                "_tensor_row",
            ],
            torch=fake_torch,
            _cuda_memory=lambda: {"allocated": 0},
        )
        self.pool = NS(_WORKSPACE={})
        self.q = q
        self.key = torch.device("cuda:0"), 12

    def adapter(self, size=64):
        q = self.q
        core = NS(
            device=self.key[0],
            heads=12,
            max_seq=8192 + (q if q > 1 else 0),
            dcp=None,
            cores={(q, q): "tokenspeed"},
            pitches={q: 3648},
            kernels={},
        )

        def warmup(cache):
            self.events.append(("warmup", size))
            core.kernels[q, q] = object()
            self.pool._WORKSPACE[self.key] = Workspace(size)

        core.prepare_cache = Mock(side_effect=warmup)
        core.run = Mock(side_effect=lambda projected, out, **kwargs: out)
        return self.module._PreparedTokenSpeedMla(core, self.pool)

    def resource(self, adapters):
        self.borrowed = {i: Workspace(1024) for i in range(len(adapters))}
        state = NS(
            conv=[Workspace(128)],
            temporal=Workspace(256),
            intermediate_conv_window=[Workspace(512)],
            intermediate_ssm=Workspace(1024),
        )
        kda = NS(prepare_caches=Mock(), prepare_verify_caches=Mock())
        attention = {i: NS(kind="mla", core=core) for i, core in enumerate(adapters)}
        attention[len(adapters)] = NS(kind="kda", core=kda)
        resources = self.module.ResourceSet(
            **resource_fields(
                attention=attention,
                tokens_per_request=self.q,
                aux_capture_layers=(0,) if self.q > 1 else None,
            )
        )
        resources._binding_candidate = lambda hybrid: {"pool": "unchanged"}
        resources.refresh_inventory = Mock(
            side_effect=AssertionError("Must not rewalk borrowed native pools")
        )
        self.state, self.kda = state, kda
        self.hybrid = NS(
            mla=NS(cache=lambda layer: self.borrowed[layer]),
            kda=NS(state=lambda layer: state),
        )
        return resources


class TestWorkspaceOwnership(unittest.TestCase):
    def test_finalize_only_after_all_warmups_inventory_excludes_borrowed_pools(self):
        for q in (1, 6):
            with self.subTest(q=q):
                fixture = OwnershipFixture(q)
                first, last = fixture.adapter(16), fixture.adapter(256)
                resources = fixture.resource([first, last])
                resources.bind(fixture.hybrid)
                final = fixture.pool._WORKSPACE[fixture.key]
                self.assertIs(first.workspace, final)
                self.assertIs(last.workspace, final)
                self.assertEqual(final.numel(), 256)
                self.assertIs(resources.workspaces["mla.tokenspeed.workspace"], final)
                self.assertEqual(
                    [row["data_ptr"] for row in resources.resource_inventory],
                    [final.storage_pointer],
                )
                self.assertEqual(resources.resource_inventory[0]["kind"], "workspace")
                self.assertEqual(len(resources.resource_digest), 64)
                self.assertEqual(fixture.events[:2], [("warmup", 16), ("warmup", 256)])
                for i, provider in enumerate((first, last)):
                    provider.core.prepare_cache.assert_called_once_with(
                        fixture.borrowed[i]
                    )
                self.assertTrue(
                    all(
                        row["data_ptr"] != native.storage_pointer
                        for row in resources.resource_inventory
                        for native in fixture.borrowed.values()
                    )
                )
                resources.refresh_inventory.assert_not_called()
                if q == 1:
                    fixture.kda.prepare_caches.assert_called_once_with(
                        (fixture.state.conv[0], fixture.state.temporal)
                    )
                    fixture.kda.prepare_verify_caches.assert_not_called()
                else:
                    fixture.kda.prepare_verify_caches.assert_called_once_with(
                        (fixture.state.conv[0], fixture.state.temporal),
                        (
                            fixture.state.intermediate_conv_window[0],
                            fixture.state.intermediate_ssm,
                        ),
                    )
                    fixture.kda.prepare_caches.assert_not_called()
                inventory = list(resources.resource_inventory)
                resources.bind(fixture.hybrid)
                self.assertEqual(resources.resource_inventory, inventory)
                self.assertEqual(first.core.prepare_cache.call_count, 1)
                fixture.pool._WORKSPACE[fixture.key] = Workspace(512)
                with self.assertRaisesRegex(
                    RuntimeError, "workspace or geometry changed"
                ):
                    resources.bind(fixture.hybrid)

    def test_capture_unprepared_and_post_finalize_preparation_rejected(self):
        fixture = OwnershipFixture()
        adapter = fixture.adapter()
        with self.assertRaisesRegex(RuntimeError, "prewarmed"):
            adapter.finalize_workspace()
        with self.assertRaisesRegex(RuntimeError, "finalize"):
            adapter.run(object(), object())
        fixture.capturing = True
        with self.assertRaisesRegex(RuntimeError, "during capture"):
            adapter.prepare_cache(object())
        adapter.core.prepare_cache.assert_not_called()
        fixture.capturing = False
        adapter.prepare_cache(object())
        fixture.capturing = True
        with self.assertRaisesRegex(RuntimeError, "during capture"):
            adapter.finalize_workspace()
        fixture.capturing = False
        owner = adapter.finalize_workspace()
        self.assertIs(adapter.finalize_workspace(), owner)
        with self.assertRaisesRegex(RuntimeError, "cannot reprepare"):
            adapter.prepare_cache(object())
        self.assertEqual(adapter.core.prepare_cache.call_count, 1)

    def test_replacement_storage_layout_and_geometry_mutations_rejected_before_run(
        self,
    ):
        mutations = (
            lambda f, a: f.pool._WORKSPACE.__setitem__(f.key, Workspace(64)),
            lambda f, a: setattr(f.pool, "_WORKSPACE", {}),
            lambda f, a: setattr(a.workspace, "pointer", a.workspace.pointer + 16),
            lambda f, a: setattr(a.workspace, "storage_bytes", 128),
            lambda f, a: setattr(a.workspace, "shape", (32,)),
            lambda f, a: setattr(a.workspace, "strides", (2,)),
            lambda f, a: setattr(a.core, "max_seq", 9999),
            lambda f, a: a.core.cores.__setitem__((2, 1), "tokenspeed"),
            lambda f, a: a.core.kernels.__setitem__((2, 1), object()),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                fixture = OwnershipFixture()
                adapter = fixture.adapter()
                adapter.prepare_cache(object())
                owner = adapter.finalize_workspace()
                mutate(fixture, adapter)
                with self.assertRaisesRegex(
                    RuntimeError, "workspace or geometry changed"
                ):
                    adapter.run(object(), object())
                adapter.core.run.assert_not_called()
                self.assertIs(adapter.workspace, owner)  # The old owner stays retained.

    def test_run_preserves_outputs_metadata_and_checks_after_original(self):
        fixture = OwnershipFixture(6)
        adapter = fixture.adapter()
        adapter.prepare_cache(object())
        adapter.finalize_workspace()
        projected, out, lengths = object(), object(), object()
        kwargs = {"lengths": lengths, "q": 6, "cache": object()}
        self.assertIs(adapter.run(projected, out, **kwargs), out)
        adapter.core.run.assert_called_once_with(projected, out, **kwargs)

        def change_owner(*args, **kwargs):
            fixture.pool._WORKSPACE[fixture.key] = Workspace(128)
            return out

        adapter.core.run.side_effect = change_owner
        with self.assertRaisesRegex(RuntimeError, "workspace or geometry changed"):
            adapter.run(projected, out, **kwargs)

    def test_invalid_workspace_and_dcp_or_mixed_core_rejected(self):
        for change in (
            lambda x: setattr(x, "dtype", torch.float32),
            lambda x: setattr(x, "device", torch.device("cpu")),
            lambda x: setattr(x, "shape", (0,)),
            lambda x: setattr(x, "shape", (8, 8)),
            lambda x: setattr(x, "strides", (2,)),
        ):
            with self.subTest(change=change):
                fixture = OwnershipFixture()
                adapter = fixture.adapter()
                adapter.prepare_cache(object())
                change(fixture.pool._WORKSPACE[fixture.key])
                with self.assertRaisesRegex(RuntimeError, "valid workspace"):
                    adapter.finalize_workspace()
        fixture = OwnershipFixture()
        core = fixture.adapter().core
        for mode in ("dcp", "mixed"):
            core.dcp = object() if mode == "dcp" else None
            core.cores = (
                {(1, 1): "tokenspeed", (2, 1): "trtllm"}
                if mode == "mixed"
                else {(1, 1): "tokenspeed"}
            )
            with self.assertRaisesRegex(ValueError, "DCP-off forced core"):
                fixture.module._PreparedTokenSpeedMla(core, fixture.pool)

    def test_workspace_inventory_deduplicates_preexisting_owner(self):
        fixture = OwnershipFixture()
        adapter = fixture.adapter()
        owner = Workspace(64)

        def warmup(cache):
            adapter.core.kernels[1, 1] = object()
            fixture.pool._WORKSPACE[fixture.key] = owner

        adapter.core.prepare_cache.side_effect = warmup
        resources = fixture.resource([adapter])
        row = fixture.module._tensor_row(
            "workspace", "resources.workspaces.retained", owner
        )
        resources.resource_inventory = [row]
        resources.bind(fixture.hybrid)
        self.assertEqual(resources.resource_inventory, [row])


class TestAdmission(unittest.TestCase):
    def test_prefetch_borrows_native_weights_through_optional_wrapper(self):
        from sgl_nvidia_mega.kimi_k3.weight_prefetch import attention_weights

        module = definitions(PROVIDERS, ["_PreparedTokenSpeedMla"])
        weights = tuple(torch.empty(16) for _ in range(7))
        native = NS(dcp=None, cores={(6, 6): "tokenspeed"}, weights=weights)
        wrapped = module._PreparedTokenSpeedMla(native, NS(_WORKSPACE={}))
        projection, output = torch.empty(16), torch.empty(16)
        layer = NS(self_attn=NS(o_proj=NS(weight=output)))
        expected = (projection, *weights[2:5], output)
        for core in (native, wrapped):
            with self.subTest(wrapped=core is wrapped):
                prepared = NS(kind="mla", core=core, projection_weight=projection)
                actual = attention_weights(prepared, layer)
                self.assertEqual(len(actual), len(expected))
                self.assertTrue(all(a is b for a, b in zip(actual, expected)))
        self.assertIs(wrapped.weights, weights)

    def test_selected_dependency_version_and_invalid_selection_fail_before_build(self):
        from importlib.metadata import PackageNotFoundError

        module = definitions(PROVIDERS, ["_validate_mla_core", "build_resources"])
        module.prepared_widths = Mock(
            side_effect=AssertionError("Preparation began before validation")
        )
        for choice in (None, True, 1, "auto", "cute-dsl", [], {}):
            with self.subTest(choice=choice), self.assertRaises(ValueError):
                module.build_resources(
                    object(),
                    context_length=8192,
                    memory_before={},
                    deadline_ns=1,
                    mla_core=choice,
                )
        for version in ("0.1.8", "0.2.15", "2.0.2+local"):
            with (
                self.subTest(version=version),
                patch("importlib.metadata.version", return_value=version),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "requires tokenspeed-mla==2.0.2"
                ):
                    module._validate_mla_core("tokenspeed")
        with patch("importlib.metadata.version", side_effect=PackageNotFoundError):
            with self.assertRaisesRegex(RuntimeError, "requires tokenspeed-mla==2.0.2"):
                module._validate_mla_core("tokenspeed")
        with patch("importlib.metadata.version", return_value="2.0.2") as version:
            module._validate_mla_core("tokenspeed")
            version.assert_called_once_with("tokenspeed-mla")
        module.prepared_widths.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)

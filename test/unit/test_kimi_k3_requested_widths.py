# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""CPU/meta contracts for requested rows, not native CUDA qualification.

Execute the actual selection, provider construction and preparation definitions;
only unavailable framework/CUDA boundaries and native kernel constructors are
replaced. No model weights, optional attention backend or GPU is loaded.
"""

from __future__ import annotations

import builtins
import os
import sys
import unittest
from functools import partial
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch
from _kimi_test_utils import definitions, execution_resources

from sgl_nvidia_mega.kimi_k3 import embedding, weight_prefetch

ROOT = Path(__file__).resolve().parents[2] / "sgl_nvidia_mega" / "kimi_k3"


POLICY = definitions(
    ROOT / "shape_policy.py",
    ("token_widths", "prepared_widths", "mla_max_seq_len"),
    DECODE_WIDTHS=(1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64),
    VERIFY_QUERIES=(4, 5, 6, 7, 8),
    MAX_ROWS=64,
    MAX_CONTEXT=262144,
)


def providers(**values):
    return definitions(
        ROOT / "providers.py",
        (
            "_kda_policy",
            "_projection_policies",
            "_validate_mla_core",
            "build_resources",
        ),
        prepared_widths=POLICY.prepared_widths,
        mla_max_seq_len=POLICY.mla_max_seq_len,
        _MAX_BATCH=64,
        torch=torch,
        SimpleNamespace=NS,
        **values,
    )


class BuildFixture:
    """93 real-shape meta layers; record native constructor arguments only."""

    def __init__(self):
        self.modules = {}
        self.cores = {"mla": [], "kda": []}

        def factory(name, **constructors):
            module = NS(**constructors)
            self.modules[name] = module
            return module

        def projection(*args):
            return NS(prepare=Mock(return_value=NS()))

        def core(kind):
            def create(*args, **kwargs):
                value = NS(
                    heads=12,
                    prepare_cache=Mock(),
                    prepare_caches=Mock(),
                    prepare_verify_caches=Mock(),
                )
                self.cores[kind].append(value)
                return value

            return Mock(side_effect=create)

        def scratch(tp, device, batches):
            return NS(
                batches=batches,
                begin=Mock(),
                shared_done=Mock(),
                gate_done=Mock(),
                stream=Mock(),
                front=object(),
                plan=object(),
                shared=object(),
            )

        box = NS(buffer=torch.empty((1, 64, 7168), device="meta"))
        small = NS(front_views=(object(), object()), hint_views=(object(), object()))
        factory("gemm.projection", PreparedProjection=Mock(side_effect=projection))
        factory("mla.prepared", PreparedMla=core("mla"))
        factory("kda.prepared", PreparedKda=core("kda"))
        factory(
            "oproj.prepared",
            OprojMailbox=Mock(return_value=box),
            PreparedOproj=Mock(return_value=NS()),
        )
        factory("common.twoshot", TwoShot=Mock(return_value=NS()))
        factory(
            "moe.small_batch.chain",
            SmallBatchBox=Mock(return_value=small),
            SmallBatchChain=Mock(return_value=NS(batches=(1,), prefetch=None)),
        )
        factory(
            "moe.tail_runtime",
            MoeTailBox=Mock(return_value=NS(out=box)),
            PreparedTail=Mock(return_value=NS()),
        )
        factory(
            "moe.prepared",
            MoEScratch=Mock(side_effect=scratch),
            PreparedMoE=Mock(
                side_effect=lambda *a, **k: NS(small_batch=None, prefetch=k["prefetch"])
            ),
        )
        factory("moe.dense", PreparedDenseMLP=Mock(return_value=NS()))
        factory("moe.experts_runtime", PersistentExperts=Mock(return_value=NS()))
        factory("moe.front", SharedFront=Mock(return_value=NS()))
        factory("common.prefetch")
        factory(
            "attn_res.prepared",
            PreparedAttnRes=NS(
                for_target=Mock(side_effect=lambda *a, **k: NS(bind_ring=Mock()))
            ),
        )

        def weight():
            return NS(weight=torch.empty((896, 7168), device="meta"))

        expert = NS(
            **{
                name: object()
                for name in (
                    "w13_weight",
                    "w13_weight_scale",
                    "w2_weight",
                    "w2_weight_scale",
                )
            }
        )
        mlp = NS(
            experts=expert,
            routed_expert_norm=weight(),
            routed_expert_up_proj=weight(),
            shared_experts=NS(),
            gate=weight(),
            routed_expert_down_proj=weight(),
            routed_scaling_factor=1.0,
        )
        layers = tuple(
            NS(
                layer_idx=i,
                self_attn=NS(**({"q_b_proj": object()} if i % 4 == 0 else {})),
                _is_moe_layer=i > 0,
                mlp=mlp,
            )
            for i in range(93)
        )
        self.model = NS(
            parameters=lambda: iter((torch.empty(1, device="meta"),)),
            model=NS(
                layers=layers,
                start_layer=0,
                end_layer=93,
                config=NS(activation_situ_beta=1.0, activation_situ_linear_beta=1.0),
            ),
            config=NS(
                full_attention_layer_ids=list(range(0, 93, 4)),
                linear_layer_ids=[i for i in range(93) if i % 4],
                rms_norm_eps=1e-6,
                moe_renormalize=True,
            ),
        )
        self.module = providers(
            _module=Mock(side_effect=lambda name, origins: self.modules[name]),
            get_parallel=lambda: NS(
                tp_size=8, tp_rank=0, tp_group=NS(device_group=object())
            ),
            _mla_proxy=lambda module: (module, ()),
            _kda_proxy=lambda module: module,
            _layer_proxy=lambda layer: layer,
            _final_proxy=lambda backbone: backbone,
            coalesce_mla_projection_storage=lambda module: torch.empty(
                (3648, 7168), device="meta"
            ),
            coalesce_kda_projection_storage=lambda module: torch.empty(
                (6288, 7168), device="meta"
            ),
            _prepare_mla_workspace=Mock(return_value=(object(), object())),
            tensor_model_parallel_all_reduce=object(),
            PreparedAttention=lambda layer_id, kind, weight, projection, core, output: (
                NS(
                    layer_id=layer_id,
                    kind=kind,
                    projection_weight=weight,
                    projection=projection,
                    core=core,
                    output=output,
                )
            ),
            ResourceSet=lambda **kwargs: NS(**kwargs, refresh_inventory=Mock()),
            _cuda_memory=lambda: {},
        )
        self.module.__package__ = "sgl_nvidia_mega.kimi_k3"

    def build(self, q, widths=None, **options):
        with (
            patch.object(torch.cuda, "Stream", return_value=Mock()),
            patch.object(torch.cuda, "Event", return_value=Mock()),
            patch.object(torch.cuda, "synchronize"),
        ):
            return self.module.build_resources(
                self.model,
                context_length=262144,
                memory_before={},
                deadline_ns=123,
                tokens_per_request=q,
                widths=widths,
                **options,
            )


class TestWidthSelection(unittest.TestCase):
    def test_optional_preparation_matrix(self):
        # Full feature interactions on representative q1/q6 subsets; defaults,
        # B1-special and odd-row preparation reuse the same behavioral checks.
        defaults = [
            ({}, False),
            (dict(enable_prefetch=False, enable_embedding=False), False),
        ]
        interactions = [
            (dict(enable_prefetch=p, enable_embedding=True, mla_core=m), supported)
            for p in (False, True)
            for m in ("trtllm", "tokenspeed")
            for supported in (False, True)
        ]
        for q, widths in (
            (1, None),
            (1, (1, 4, 16)),
            (1, (2, 8, 64)),
            (1, (9, 15, 17, 31, 33, 63)),
            (6, None),
            (6, (6, 12, 60)),
        ):
            cases = defaults + [
                (dict(enable_prefetch=True, mla_core=m), False)
                for m in ("trtllm", "tokenspeed")
            ]
            if widths in ((2, 8, 64), (6, 12, 60)):
                cases += interactions
            for options, supported in cases:
                with self.subTest(
                    q=q, widths=widths, options=options, supported=supported
                ):
                    f = BuildFixture()
                    selected = POLICY.prepared_widths(q, widths)
                    prefetch_on = options.get("enable_prefetch", False)
                    embedding_on = options.get("enable_embedding", False)
                    mla = options.get("mla_core", "trtllm")
                    controller, gather, constructor, dependency = (
                        object() for _ in range(4)
                    )
                    f.model.model.embed_tokens = object()
                    f.modules["common.embedding"] = NS(PreparedEmbedding=constructor)
                    f.modules["mla.dcp"] = dependency
                    wrapper = f.module._PreparedTokenSpeedMla = Mock(
                        side_effect=lambda core, module: NS(heads=core.heads)
                    )
                    original_import = builtins.__import__

                    def guarded_import(name, *args, **kwargs):
                        for enabled, optional in (
                            (prefetch_on, "weight_prefetch"),
                            (embedding_on, "embedding"),
                        ):
                            if not enabled and name in (
                                optional,
                                "sgl_nvidia_mega.kimi_k3." + optional,
                            ):
                                self.fail("OFF imported " + optional)
                        if mla == "trtllm" and (
                            name == "importlib.metadata"
                            or name.startswith("tokenspeed")
                        ):
                            self.fail("default MLA touched optional dependency")
                        return original_import(name, *args, **kwargs)

                    def prepare(native, *, tp_group, load_prepared):
                        self.assertIs(native, f.model.model.embed_tokens)
                        if supported:
                            self.assertIs(load_prepared(), constructor)
                            return gather
                        return None

                    def resources(**kwargs):
                        value = NS(**kwargs)

                        def inventory():
                            self.assertIs(
                                value.prefetch, controller if prefetch_on else None
                            )
                            for layer in value.layers[1:]:
                                owner = value.feed_forward[layer.layer_idx]
                                self.assertIs(owner.prefetch, value.prefetch)
                                self.assertEqual(
                                    owner.small_batch is not None, 1 in selected
                                )
                                if owner.small_batch is not None:
                                    self.assertIs(
                                        owner.small_batch.prefetch, value.prefetch
                                    )

                        value.refresh_inventory = Mock(side_effect=inventory)
                        return value

                    f.module.ResourceSet = resources
                    with (
                        patch.object(
                            weight_prefetch,
                            "PrefetchController",
                            return_value=controller,
                        ) as construct,
                        patch.object(
                            embedding, "prepare_embedding", side_effect=prepare
                        ) as prepare_mock,
                        patch("importlib.metadata.version", return_value="2.0.2"),
                        patch.object(
                            builtins, "__import__", side_effect=guarded_import
                        ),
                    ):
                        result = f.build(q, widths, **options)
                    result.refresh_inventory.assert_called_once_with()
                    self.assertEqual(result.widths, selected)
                    self.assertEqual(construct.call_count, int(prefetch_on))
                    if prefetch_on:
                        construct.assert_called_once_with(
                            result.attention,
                            result.layers,
                            result.moe_front_weights,
                            device=torch.device("meta"),
                            batches=selected,
                        )
                    self.assertEqual(prepare_mock.call_count, int(embedding_on))
                    self.assertIs(result.embedding, gather if supported else None)
                    self.assertEqual("embedding.out" in result.scratch, supported)
                    if supported:
                        self.assertEqual(
                            result.scratch["embedding.out"].shape, (64, 7168)
                        )
                        self.assertEqual(
                            result.scratch["embedding.out"].dtype, torch.bfloat16
                        )
                    loads = [call.args[0] for call in f.module._module.call_args_list]
                    compiled = [row["kind"] for row in result.compilation_inventory]
                    for module, kind, enabled in (
                        ("common.prefetch", "weight_prefetch", prefetch_on),
                        ("common.embedding", "embedding", supported),
                    ):
                        self.assertEqual(loads.count(module), int(enabled))
                        self.assertEqual(compiled.count(kind), int(enabled))
                    for call in f.modules["mla.prepared"].PreparedMla.call_args_list:
                        self.assertEqual(call.args[1], dict.fromkeys(selected, 3648))
                        self.assertEqual(
                            call.kwargs,
                            dict(
                                max_context=262144 + (q if q > 1 else 0),
                                core=mla,
                                dcp=None,
                                queries=(q,),
                            ),
                        )
                        self.assertIs(call.args[2], result.scratch["mla.query"])
                    if mla == "tokenspeed":
                        self.assertEqual(wrapper.call_count, 24)
                        self.assertTrue(
                            all(
                                call.args[1] is dependency
                                for call in wrapper.call_args_list
                            )
                        )
                        f.module._prepare_mla_workspace.assert_not_called()
                        self.assertNotIn("mla.flashinfer.workspace", result.workspaces)
                    else:
                        wrapper.assert_not_called()
                        self.assertNotIn("mla.dcp", loads)
                        f.module._prepare_mla_workspace.assert_called_once_with(
                            f.modules["mla.prepared"], torch.device("meta"), 12
                        )
                        workspace, counter = (
                            f.module._prepare_mla_workspace.return_value
                        )
                        self.assertIs(
                            result.workspaces["mla.flashinfer.workspace"], workspace
                        )
                        self.assertIs(
                            result.workspaces["mla.flashinfer.counter"], counter
                        )

    def test_invalid_optional_switches_fail_before_preparation(self):
        for option in ("enable_prefetch", "enable_embedding"):
            for value in (None, 0, 1, "false", "true", [], object()):
                with self.subTest(option=option, value=value):
                    fixture = BuildFixture()
                    with self.assertRaisesRegex(ValueError, option + " must be a bool"):
                        fixture.build(1, **{option: value})
                    fixture.module._module.assert_not_called()

    def test_defaults_and_canonical_subsets(self):
        for q in (1, 4, 5, 6, 7, 8):
            supported = POLICY.token_widths(q)
            self.assertEqual(POLICY.prepared_widths(q), supported)
            self.assertEqual(
                POLICY.prepared_widths(q, tuple(reversed(supported))), supported
            )
        self.assertEqual(POLICY.prepared_widths(1, (64, 2, 16)), (2, 16, 64))
        self.assertEqual(POLICY.prepared_widths(6, (60, 6, 12)), (6, 12, 60))

    def test_explicit_ordinary_widths_cover_every_row_through_capacity(self):
        rows = tuple(range(1, 65))
        self.assertEqual(POLICY.prepared_widths(1, tuple(reversed(rows))), rows)
        self.assertEqual(POLICY.prepared_widths(1), POLICY.DECODE_WIDTHS)
        for row in rows:
            with self.subTest(row=row):
                self.assertEqual(POLICY.prepared_widths(1, (row,)), (row,))
        # Verification still accepts only complete requests, with no new q modes.
        for q in (4, 5, 6, 7, 8):
            self.assertEqual(POLICY.prepared_widths(q), tuple(range(q, 65, q)))

    def test_configured_ordinary_widths_keep_source_gate_policy(self):
        rows = tuple(range(1, 65))
        module = providers()
        policy = module._kda_policy(1, rows)
        self.assertEqual(
            policy,
            {row: (6288, 1 if row == 1 else 2 if row <= 8 else 0) for row in rows},
        )
        for pitch in (3648, 6288):
            all_policies = module._projection_policies(pitch, rows)
            old_policies = module._projection_policies(pitch, POLICY.DECODE_WIDTHS)
            self.assertEqual(
                {row: all_policies[row] for row in POLICY.DECODE_WIDTHS}, old_policies
            )

    def test_invalid_subsets_reject_before_native_work(self):
        for q, value in (
            (1, ()),
            (1, (1, 1)),
            (1, (True,)),
            (1, (1.0,)),
            (1, (0,)),
            (1, (65,)),
            (1, [1]),
            (1, "1"),
            (1, 1),
            (6, (1,)),
            (6, (6, 7)),
            (6, (64,)),
            (2, (2,)),
        ):
            fixture = BuildFixture()
            with self.subTest(q=q, value=value), self.assertRaises(ValueError):
                fixture.build(q, value)
            fixture.module._module.assert_not_called()

    def test_selected_policies_preserve_existing_tactics(self):
        module = providers()
        for q, rows in ((1, (2, 4, 16)), (6, (6, 12, 60))):
            expected = module._kda_policy(q)
            selected = module._kda_policy(q, rows)
            keys = rows if q == 1 else tuple((r // q, q) for r in rows)
            self.assertEqual(selected, {key: expected[key] for key in keys})
            for pitch in (3648, 6288):
                old = module._projection_policies(pitch, POLICY.token_widths(q))
                self.assertEqual(
                    module._projection_policies(pitch, rows), {r: old[r] for r in rows}
                )

    def test_actual_build_passes_only_selected_rows_to_every_width_provider(self):
        for q, selected in (
            (1, (1, 4, 16)),
            (1, (2, 8, 64)),
            (1, (9, 15, 17, 31, 33, 63)),
            (6, (6, 12, 60)),
        ):
            fixture = BuildFixture()
            result = fixture.build(q, selected)
            with self.subTest(q=q, selected=selected):
                self.assertEqual(result.widths, selected)
                self.assertEqual(result.tokens_per_request, q)
                self.assertEqual(result.scratch["mla.query"].shape, (64, 12, 576))
                projections = fixture.modules["gemm.projection"].PreparedProjection
                self.assertEqual(projections.call_count, 93)
                self.assertTrue(
                    all(
                        tuple(call.args[1]) == selected
                        for call in projections.call_args_list
                    )
                )
                mla = fixture.modules["mla.prepared"].PreparedMla
                self.assertEqual(mla.call_count, 24)
                for call in mla.call_args_list:
                    self.assertEqual(tuple(call.args[1]), selected)
                    self.assertEqual(
                        call.kwargs,
                        dict(
                            max_context=262144 + (q if q > 1 else 0),
                            core="trtllm",
                            dcp=None,
                            queries=(q,),
                        ),
                    )
                kda = fixture.modules["kda.prepared"].PreparedKda
                expected = fixture.module._kda_policy(q, selected)
                self.assertEqual(kda.call_count, 69)
                self.assertTrue(
                    all(call.args[1] == expected for call in kda.call_args_list)
                )
                for module, name, position, count in (
                    ("moe.prepared", "MoEScratch", 2, 1),
                    ("oproj.prepared", "PreparedOproj", 2, 93),
                    ("moe.dense", "PreparedDenseMLP", 1, 1),
                    ("moe.tail_runtime", "PreparedTail", 3, 92),
                    ("moe.front", "SharedFront", 4, 92),
                ):
                    constructor = getattr(fixture.modules[module], name)
                    self.assertEqual(constructor.call_count, count)
                    self.assertTrue(
                        all(
                            call.args[position] == selected
                            for call in constructor.call_args_list
                        )
                    )
                    if name == "SharedFront":
                        self.assertTrue(
                            all(
                                call.kwargs["q"] == q
                                for call in constructor.call_args_list
                            )
                        )
                chain = fixture.modules["moe.small_batch.chain"].SmallBatchChain
                self.assertEqual(chain.call_count, 92 if 1 in selected else 0)
                self.assertEqual(
                    sum(
                        row["kind"] == "moe_small_batch"
                        for row in result.compilation_inventory
                    ),
                    chain.call_count,
                )
                self.assertTrue(
                    all(
                        (moe.small_batch is not None) == (1 in selected)
                        for i, moe in result.feed_forward.items()
                        if i > 0
                    )
                )

    def test_omitted_widths_keep_full_legacy_provider_set(self):
        for q in (1, 6):
            fixture = BuildFixture()
            self.assertEqual(fixture.build(q).widths, POLICY.token_widths(q))

    def test_kda_preparation_compiles_only_policy_keys(self):
        module = definitions(
            ROOT / "cutedsl/kda/prepared.py",
            ("PreparedKda",),
            torch=torch,
            layout=lambda t: (tuple(t.shape), tuple(t.stride()), t.dtype),
            executable=Mock(return_value=object()),
            VERIFY_QUERIES=POLICY.VERIFY_QUERIES,
            token_widths=POLICY.token_widths,
        )
        for q, rows in ((1, (2, 16)), (1, (9, 15, 17, 31, 33, 63)), (6, (6, 18))):
            core = object.__new__(module.PreparedKda)
            core.policies = providers()._kda_policy(q, rows)
            core.dim, core.heads, core.has_bias = 1536, 12, False
            core.device = torch.device("cuda:0")
            core.kernels, core.verify_kernels, core.verify_layouts = {}, {}, {}
            core._operands = lambda caches, projected, slots, out, accepted: (
                projected,
                slots,
                out,
            )
            core._check_verify_caches = Mock()
            core._verify_operands = (
                lambda caches, projected, slots, out, intermediates, indices: (
                    projected,
                    slots,
                    out,
                )
            )
            module.executable.reset_mock()
            if q == 1:
                core.prepare_caches(object())
                self.assertEqual(tuple(core.kernels), rows)
            else:
                core.prepare_verify_caches(
                    object(), (torch.empty((1, q, 3, 4608), device="meta"), object())
                )
                self.assertEqual(tuple(core.verify_kernels), ((1, q), (3, q)))
            self.assertEqual(
                [call.args[0][0][0][0] for call in module.executable.call_args_list],
                list(rows),
            )

    def test_mla_preparation_and_native_warmup_are_subset_only(self):
        module = definitions(
            ROOT / "cutedsl/mla/prepared.py",
            ("PreparedMla",),
            torch=torch,
            layout=lambda t: tuple(t.shape),
            to_cute=lambda t: t,
            compile_core=Mock(return_value=(object(), object())),
        )
        for q, rows in (
            (1, (1, 4, 32)),
            (1, (9, 15, 17, 31, 33, 63)),
            (6, (6, 12, 30)),
        ):
            core = object.__new__(module.PreparedMla)
            core.weights = (torch.empty(1, device="meta"),)
            core.cores = dict.fromkeys(((r, q) for r in rows), "trtllm")
            core.pitches, core.heads = dict.fromkeys(rows, 3648), 12
            core.query_scratch = torch.empty((64, 12, 576), device="meta")
            core.latent = torch.empty((max(rows), 12, 512), device="meta")
            core.device, core.dcp, core.kernels = torch.device("meta"), None, {}
            core.stock_core = Mock()
            cache = NS(
                view=lambda dtype: torch.empty((2, 64, 576), device="meta"),
                is_cuda=True,
                device="meta",
            )
            with patch.object(
                torch.cuda,
                "get_device_properties",
                return_value=NS(multi_processor_count=148),
            ):
                core.prepare_cache(cache)
            self.assertEqual(tuple(core.kernels), tuple((r, q) for r in rows))
            self.assertEqual(
                [
                    (call.args[3], call.args[4])
                    for call in core.stock_core.call_args_list
                ],
                [(r, q) for r in rows],
            )

    def test_expert_preparation_uses_only_selected_scratch_views(self):
        module = definitions(
            ROOT / "cutedsl/moe/prepared.py",
            ("PreparedMoE",),
            torch=torch,
            batch_table=lambda value: {},
            JOIN={8: ""},
        )
        for rows in ((2, 4, 16), (9, 15, 17, 31, 33, 63), (6, 12, 60)):
            experts = NS(prepare=Mock())
            state = NS(
                tp=8,
                device="meta",
                batches=rows,
                views={r: (object(), object(), object()) for r in rows},
            )
            with patch.object(
                torch.cuda,
                "get_device_properties",
                return_value=NS(multi_processor_count=148),
            ):
                module.PreparedMoE(
                    NS(split={}, planned=lambda b: False),
                    experts,
                    NS(publish={}, twoshots=set()),
                    state,
                )
            self.assertEqual(
                [call.args[0] for call in experts.prepare.call_args_list], list(rows)
            )

    def test_actual_spec_handoff_summary_and_unselected_fallback(self):
        module = definitions(
            ROOT / "spec.py",
            ("_build", "_Built", "_call_option", "_startup_options"),
            os=os,
            token_widths=POLICY.token_widths,
            prepared_widths=POLICY.prepared_widths,
            _deadline_ns=lambda value: 123,
            get_model=lambda: NS(context_length=262144),
            Step=lambda **kwargs: NS(**kwargs),
        )
        module.__package__ = "width_fixture"
        compat = ModuleType("width_fixture.compat")
        compat.validate_runtime_compatibility = Mock()
        provider = ModuleType("width_fixture.providers")
        provider._cuda_memory = Mock(return_value={})
        executor = ModuleType("width_fixture.executor")
        executor.KimiK3MegakernelExecutor = Mock()
        with patch.dict(
            sys.modules,
            {
                "width_fixture.compat": compat,
                "width_fixture.providers": provider,
                "width_fixture.executor": executor,
            },
        ):
            for q, requested, configured, selected in (
                (1, (32, 3, 9, 1), False, (1, 3, 32)),
                (1, (32, 3, 9, 1), True, (1, 3, 9, 32)),
                (1, tuple(range(1, 65)), False, POLICY.DECODE_WIDTHS),
                (1, (9, 15, 17, 31, 33, 63), True, (9, 15, 17, 31, 33, 63)),
                (6, (1, 6, 12, 13), False, (6, 12)),
                (6, (1, 6, 12, 13), True, (6, 12)),
            ):
                provider.build_resources = Mock(return_value=NS(tokens_per_request=q))
                model = NS(language_model=NS())
                built = module._build(
                    model,
                    None,
                    NS(
                        widths=requested,
                        tokens_per_request=q,
                        deadline_s=1,
                        binding=object(),
                    ),
                    configured_widths=configured,
                )
                self.assertEqual(built.widths, selected)
                self.assertEqual(
                    provider.build_resources.call_args.kwargs["widths"], selected
                )
                self.assertEqual(
                    provider.build_resources.call_args.kwargs["tokens_per_request"], q
                )
                self.assertIsNone(built.draft_sampler)
                self.assertIn(
                    str(list(selected if q == 1 else (r // q for r in selected))),
                    built.summary(),
                )
                mode = NS(
                    is_decode=lambda q=q: q == 1,
                    is_target_verify=lambda q=q: q > 1,
                )
                for rows in POLICY.token_widths(q, configured=True):
                    batch = NS(forward_mode=mode, batch_size=rows // q)
                    planned = built.plan(NS(shape=(rows,)), batch)
                    self.assertEqual(planned is not None, rows in selected)
                self.assertIsNone(
                    built.plan(
                        NS(shape=(selected[0],)), NS(forward_mode=mode, batch_size=100)
                    )
                )

    def test_extended_spec_admission_requires_explicit_boolean_opt_in(self):
        module = definitions(
            ROOT / "spec.py",
            ("_build",),
            token_widths=POLICY.token_widths,
            prepared_widths=POLICY.prepared_widths,
        )
        context = NS(widths=(9, 15), tokens_per_request=1)
        # These checks precede bindings, imports, provider compilation or CUDA.
        with self.assertRaisesRegex(ValueError, "no supported width"):
            module._build(None, None, context)
        for value in (None, 1, "true"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "bool"):
                module._build(None, None, context, configured_widths=value)


class _CudaMetadataTensor(torch.Tensor):
    """Storage-free meta tensor with a synthetic CUDA device for guard tests.

    Shape, strides, dtype and slicing use actual Torch meta behavior. There is
    no CUDA allocation, kernel execution or numerical oracle in these tests.
    """

    @property
    def device(self):
        return torch.device("cuda:0")


class TestConfiguredWidthExecutionContract(unittest.TestCase):
    """Actual executor and validators over metadata-only tensors, not GPU math."""

    @staticmethod
    def executor_module():
        return definitions(
            ROOT / "executor.py",
            ("_require_tensor", "_validate_step_inputs", "KimiK3MegakernelExecutor"),
            torch=torch,
            partial=partial,
            VERIFY_QUERIES=POLICY.VERIFY_QUERIES,
            token_widths=POLICY.token_widths,
        )

    @staticmethod
    def fixture(rows, prepared):
        # Synthetic CUDA metadata lets the actual tensor validator run unchanged.
        # Provider calls record plumbing; no kernel or device code is executed.
        def tensor(shape, dtype=torch.bfloat16):
            return torch.empty(shape, dtype=dtype, device="meta").as_subclass(
                _CudaMetadataTensor
            )

        ids = tensor((rows,), torch.int64)
        embeds = tensor((rows, 7168))
        view = NS(
            mla=NS(
                write_slots=tensor((rows,), torch.int64),
                sequence_lengths=tensor((rows,), torch.int32),
                block_table=tensor((rows, 8), torch.int32),
                tokens_per_request=1,
            ),
            kda=NS(
                state_indices=tensor((rows,), torch.int32),
                verify_scratch_indices=None,
                track_decode_state=Mock(),
            ),
            aux_hidden_states=None,
        )
        resources = execution_resources(64, tensor=tensor, widths=prepared)
        projection, mla, kda = Mock(), Mock(), Mock()
        for index, provider in resources.attention.items():
            provider.projection.run = projection
            provider.core = mla if index == 0 else kda
        return resources, ids, embeds, view, projection, mla, kda

    def test_new_rows_execute_the_real_python_chain_and_tensor_validation(self):
        module = self.executor_module()
        for rows in (9, 12, 15, 17, 24, 31, 33, 48, 63):
            with self.subTest(rows=rows):
                resources, ids, embeds, view, projection, mla, kda = self.fixture(
                    rows, (rows,)
                )
                output = module.KimiK3MegakernelExecutor(resources).run(
                    ids, embeds, view
                )
                self.assertEqual(tuple(output.shape), (rows, 7168))
                self.assertEqual(projection.call_count, 93)
                self.assertEqual(mla.run.call_count, 1)
                self.assertEqual(kda.run.call_count, 92)
                self.assertEqual(
                    [
                        call.args[0]
                        for call in view.kda.track_decode_state.call_args_list
                    ],
                    list(range(1, 93)),
                )
                self.assertIs(mla.run.call_args.kwargs["slots"], view.mla.write_slots)
                self.assertIs(kda.run.call_args.kwargs["slots"], view.kda.state_indices)
                self.assertTrue(
                    all(
                        call.args[0].shape[0] == rows
                        for call in projection.call_args_list
                    )
                )

    def test_unprepared_rows_fail_before_validation_or_provider_execution(self):
        module = self.executor_module()
        for rows, selected in ((9, POLICY.DECODE_WIDTHS), (16, (9, 15)), (1, (9, 15))):
            with self.subTest(rows=rows, selected=selected):
                resources, ids, embeds, view, projection, mla, kda = self.fixture(
                    rows, selected
                )
                with patch.object(
                    module, "_validate_step_inputs", wraps=module._validate_step_inputs
                ) as validate:
                    with self.assertRaisesRegex(RuntimeError, "unsupported.*batch"):
                        module.KimiK3MegakernelExecutor(resources).run(
                            ids, embeds, view
                        )
                validate.assert_not_called()
                projection.assert_not_called()
                mla.run.assert_not_called()
                kda.run.assert_not_called()

    def test_new_rows_do_not_bypass_dtype_shape_or_capacity_guards(self):
        module = self.executor_module()
        for rows, corrupt in ((9, "dtype"), (9, "shape"), (65, None)):
            with self.subTest(rows=rows, corrupt=corrupt):
                resources, ids, embeds, view, projection, mla, kda = self.fixture(
                    rows, (rows,)
                )
                if corrupt == "dtype":
                    view.mla.write_slots = view.mla.write_slots.to(torch.int32)
                elif corrupt == "shape":
                    view.kda.state_indices = view.kda.state_indices[:-1]
                with self.assertRaises(RuntimeError):
                    module.KimiK3MegakernelExecutor(resources).run(ids, embeds, view)
                projection.assert_not_called()
                mla.run.assert_not_called()
                kda.run.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)

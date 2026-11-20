# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Optional providers: real host orchestration with mocked device constructors."""

import builtins
import importlib
import inspect
from types import SimpleNamespace as NS
from unittest.mock import Mock, call, patch

import pytest
import torch

from sgl_nvidia_mega.kimi_k3 import (
    dcp_execution,
    dcp_resources,
    dcp_verify_execution,
    embedding,
    providers,
    weight_prefetch,
)


class BuildFixture:
    """93 real-shape meta layers; record native constructor arguments only."""

    def __init__(self, monkeypatch):
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
        overrides = dict(
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
            _cuda_memory=lambda: {},
        )
        self.module = providers
        for name, value in overrides.items():
            monkeypatch.setattr(providers, name, value)

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


@pytest.mark.parametrize("q,widths", ((1, (1, 8)), (6, (6, 12))))
@pytest.mark.parametrize("enabled", (False, True))
def test_default_off_has_no_optional_resources_and_opt_in_binds_owners(
    monkeypatch, q, widths, enabled
):
    fixture = BuildFixture(monkeypatch)
    controller, gather = object(), object()
    construct = Mock(return_value=controller)
    monkeypatch.setattr(weight_prefetch, "PrefetchController", construct)

    def prepare(native, *, tp_group, load_prepared):
        assert load_prepared() is fixture.modules["common.embedding"].PreparedEmbedding
        return gather

    prepare = Mock(side_effect=prepare)
    monkeypatch.setattr(embedding, "prepare_embedding", prepare)
    fixture.model.model.embed_tokens = object()
    fixture.modules["common.embedding"] = NS(PreparedEmbedding=object())
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if not enabled and name.rsplit(".", 1)[-1] in ("embedding", "weight_prefetch"):
            raise AssertionError("default-OFF imported an optional provider")
        return original_import(name, *args, **kwargs)

    def inventory(resources):
        assert resources.prefetch is (controller if enabled else None)
        for layer in resources.layers[1:]:
            moe = resources.feed_forward[layer.layer_idx]
            assert moe.prefetch is resources.prefetch
            if moe.small_batch is not None:
                assert moe.small_batch.prefetch is resources.prefetch

    refresh = Mock(side_effect=inventory)
    monkeypatch.setattr(
        providers.ResourceSet, "refresh_inventory", lambda self: refresh(self)
    )
    options = dict(enable_prefetch=True, enable_embedding=True) if enabled else {}
    with patch.object(builtins, "__import__", side_effect=guarded):
        result = fixture.build(q, widths, **options)
    refresh.assert_called_once_with(result)
    assert construct.call_count == prepare.call_count == int(enabled)
    assert result.embedding is (gather if enabled else None)
    assert ("embedding.out" in result.scratch) is enabled
    assert result.dcp is None and result.widths == widths
    for kind in ("weight_prefetch", "embedding"):
        assert sum(row["kind"] == kind for row in result.compilation_inventory) == int(
            enabled
        )
    for call in fixture.modules["mla.prepared"].PreparedMla.call_args_list:
        assert call.kwargs == dict(
            max_context=262144 + (q if q > 1 else 0),
            core="trtllm",
            dcp=None,
            queries=(q,),
        )
    if enabled:
        construct.assert_called_once_with(
            result.attention,
            result.layers,
            result.moe_front_weights,
            device=torch.device("meta"),
            batches=widths,
        )


@pytest.mark.parametrize("option", ("enable_prefetch", "enable_embedding"))
def test_invalid_opt_in_rejected_before_loading_payload(option, monkeypatch):
    load = Mock(side_effect=AssertionError("payload load"))
    monkeypatch.setattr(providers, "_module", load)
    with pytest.raises(ValueError, match="must be a bool"):
        providers.build_resources(
            None, context_length=8192, memory_before={}, deadline_ns=1, **{option: 1}
        )
    load.assert_not_called()


@pytest.mark.parametrize(
    "module,name,constructor",
    (
        (
            dcp_execution,
            "sglang.srt.kernel_plugins.nvidia._dcp_binding",
            "PreparedDcpHybrid",
        ),
        (
            dcp_verify_execution,
            "sglang.srt.kernel_plugins.nvidia._dcp_verify_binding",
            "PreparedDcpVerify",
        ),
    ),
)
def test_optional_dcp_extension_fails_closed_without_hiding_broken_imports(
    module, name, constructor
):
    try:
        native = importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name != name:
            raise
        with pytest.raises(
            RuntimeError, match="requires the SGLang DCP binding extension"
        ):
            getattr(module, constructor)(None, None)
    else:
        assert module._native_types() == tuple(
            getattr(native, item)
            for item in next(iter(module.SGLANG_OPTIONAL_READS.values()))
        )
    for error in (
        ModuleNotFoundError(name=name),
        ModuleNotFoundError(name="torch"),
        ImportError("broken symbol"),
    ):
        with patch.object(builtins, "__import__", side_effect=error):
            with pytest.raises(
                RuntimeError if error.name == name else type(error)
            ) as caught:
                module._native_types()
            assert (
                caught.value.__cause__ if error.name == name else caught.value
            ) is error


@pytest.mark.parametrize("q,widths", ((1, (1, 8)), (6, (6, 12))))
def test_dcp_owner_rejects_changed_groups_and_unagreed_plan(monkeypatch, q, widths):
    def group():
        return NS(
            ranks=list(range(8)),
            rank_in_group=0,
            world_size=8,
            rank=0,
            device_group=object(),
            cpu_group=object(),
        )

    parallel = NS(
        dcp_enabled=True,
        tp_size=8,
        tp_rank=0,
        dcp_size=8,
        dcp_rank=0,
        attn_dcp_size=8,
        attn_dcp_rank=0,
        tp_group=group(),
        dcp_group=group(),
    )
    agree = Mock()
    monkeypatch.setattr(
        dcp_resources, "_native_context", lambda: (lambda: parallel, agree)
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (10, 3))
    options = dcp_resources.DcpBuildOptions()
    args = dict(
        device=torch.device("cuda:0"),
        tokens_per_request=q,
        deadline_ns=100,
        widths=widths,
        max_context=8192,
        mla_core="tokenspeed",
    )
    owner = dcp_resources.NativeDcpOwner.capture(options, **args)
    agree.assert_called_once()
    owner.assert_live()
    owner.check_plan(options, widths, 8192, 100, q)
    with pytest.raises(ValueError, match="all-rank agreed"):
        owner.check_plan(options, widths, 4096, 100, q)
    parallel.dcp_group.device_group = object()
    with pytest.raises(RuntimeError):
        owner.assert_live()
    with pytest.raises(ValueError):
        dcp_resources.NativeDcpOwner.capture(options, **(args | {"mla_core": "trtllm"}))
    assert agree.call_count == 2


def test_prefetch_orders_stream_dependencies_and_joins_only_pending_work(monkeypatch):
    controller = object.__new__(weight_prefetch.PrefetchController)
    events = Mock()
    controller.tables = {("attention_moe", 0, 1): events.table}
    controller.events = {("attention_moe", 0): events.event}
    controller.stream, controller.done = events.side, events.done
    controller.device, controller.launch_stream = torch.device("cuda:0"), object()
    controller.pending = False
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_: events.current)
    assert not controller.fork("missing", 0, 1)
    controller.fork("attention_moe", 0, 1)
    controller.join()
    controller.join()
    assert events.mock_calls == [
        # Event ordering rather than mocked timing is the regression contract.
        call.event.record(events.current),
        call.side.wait_event(events.event),
        call.table.launch(controller.launch_stream),
        call.done.record(events.side),
        call.current.wait_event(events.done),
    ]


def test_provider_defaults_do_not_activate_dormant_paths():
    defaults = inspect.signature(providers.build_resources).parameters
    assert defaults["enable_prefetch"].default is False
    assert defaults["enable_embedding"].default is False
    assert defaults["dcp"].default is None
    assert defaults["mla_core"].default == "trtllm"


class _Workspace(torch.Tensor):
    """CPU storage with CUDA device metadata; never used for native execution."""

    @property
    def device(self):
        return torch.device("cuda:0")


def test_mla_owns_final_workspace_and_rejects_replacement(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    pool = NS(_WORKSPACE={})
    key = torch.device("cuda:0"), 12
    adapters = []
    for size in (16, 64):
        workspace = torch.zeros(size, dtype=torch.int8).as_subclass(_Workspace)
        core = NS(
            dcp=None,
            device=key[0],
            heads=12,
            max_seq=8192,
            cores={(6, 6): "tokenspeed"},
            pitches={6: 3648},
            kernels={},
            prepare_cache=Mock(
                side_effect=lambda cache, w=workspace: pool._WORKSPACE.update({key: w})
            ),
            run=Mock(return_value=object()),
        )
        adapter = providers._PreparedTokenSpeedMla(core, pool)
        with pytest.raises(RuntimeError, match="finalize before execution"):
            adapter.run()
        adapter.prepare_cache(object())
        adapters.append(adapter)
    final = pool._WORKSPACE[key]
    for adapter in adapters:
        assert adapter.finalize_workspace() is final
        assert adapter.run() is adapter.core.run.return_value
    pool._WORKSPACE[key] = torch.zeros(128, dtype=torch.int8).as_subclass(_Workspace)
    for adapter in adapters:
        with pytest.raises(RuntimeError, match="workspace or geometry changed"):
            adapter.run()
        assert adapter.workspace is final


def test_inventory_deduplicates_owned_storage_and_excludes_borrowed_weights():
    from test_kimi_k3_dspark import _resources

    borrowed = torch.zeros(16).as_subclass(_Workspace)
    output = torch.zeros(16).as_subclass(_Workspace)
    model = NS(modules=lambda: (), parameters=lambda: (borrowed,), buffers=lambda: ())
    resources = _resources(
        model=model,
        scratch={"embedding.out": output},
        embedding=NS(weight=borrowed, output=output),
    )
    rows = providers._owned_resource_inventory(resources, model)
    assert len(rows) == 1
    assert rows[0]["kind"] == "scratch"
    assert rows[0]["data_ptr"] == output.untyped_storage().data_ptr()

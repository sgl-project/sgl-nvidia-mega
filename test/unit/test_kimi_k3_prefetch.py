# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""TP8 prefetch policy, borrowed ranges, hook ordering and native graph checks."""

from __future__ import annotations

import contextlib
import gc
import math
import sys
import unittest
import weakref
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch
from _kimi_test_utils import (
    definitions,
    execution_resources,
    inventory_fixture,
    resource_fields,
)

from sgl_nvidia_mega.kimi_k3 import weight_prefetch as prefetch
from sgl_nvidia_mega.kimi_k3.shape_policy import token_widths

ROOT = Path(prefetch.__file__).parent
MIB = 1024**2
POINTS = ("oproj_down", "attention_moe", "next_attention")


def _definitions(relative, names, namespace):
    return vars(definitions(ROOT / relative, names, **namespace))


class _Tensor(torch.Tensor):
    """Metadata-only CUDA tensor: reading or copying its values is not allowed."""

    @staticmethod
    def __new__(
        cls,
        shape=(64,),
        *,
        pointer=4096,
        strides=None,
        storage_pointer=None,
        storage_bytes=4096,
        dtype=torch.bfloat16,
        device="cuda:0",
    ):
        value = torch.Tensor._make_subclass(cls, torch.empty(0), False)
        value._shape = tuple(shape)
        value._pointer = pointer
        value._strides = strides or tuple(
            math.prod(shape[index + 1 :]) for index in range(len(shape))
        )
        value._storage_pointer = pointer if storage_pointer is None else storage_pointer
        value._storage_bytes = storage_bytes
        value._dtype = dtype
        value._device = torch.device(device)
        return value

    @property
    def shape(self):
        return torch.Size(self._shape)

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    @property
    def is_cuda(self):
        return self.device.type == "cuda"

    def numel(self):
        return math.prod(self._shape)

    def stride(self, dim=None):
        return self._strides if dim is None else self._strides[dim]

    def element_size(self):
        return torch.empty((), dtype=self.dtype).element_size()

    def data_ptr(self):
        return self._pointer

    def untyped_storage(self):
        return SimpleNamespace(
            data_ptr=lambda: self._storage_pointer,
            nbytes=lambda: self._storage_bytes,
        )

    def storage_offset(self):
        return (self._pointer - self._storage_pointer) // self.element_size()

    def is_contiguous(self):
        return self._strides == tuple(
            math.prod(self._shape[index + 1 :]) for index in range(len(self._shape))
        )

    def detach(self):
        return self

    def clone(self, *args, **kwargs):
        raise AssertionError("Prefetch copied a borrowed weight")

    contiguous = clone
    to = clone
    cpu = clone
    numpy = clone
    item = clone


def _resources(tensor_factory):
    """Small final-form storages, not model math or checkpoint substitutes."""
    layers, attention, packed = [], {}, {}
    for index, kind in enumerate(("mla", "kda", "mla")):

        def weight():
            return tensor_factory((16, 32))

        native = SimpleNamespace(
            f_b_proj=SimpleNamespace(weight=weight()),
            o_proj=SimpleNamespace(weight=weight()),
            # These obsolete aliases must not replace the prepared projection.
            in_proj_qkvgfab=SimpleNamespace(weight=weight()),
            q_b_proj=SimpleNamespace(weight=weight()),
            w_kc=weight(),
            w_vc=weight(),
        )
        shared = SimpleNamespace(
            gate_up_proj=SimpleNamespace(weight=weight()),
            down_proj=SimpleNamespace(weight=weight()),
        )
        layers.append(
            SimpleNamespace(
                layer_idx=index,
                _is_moe_layer=True,
                self_attn=native,
                mlp=SimpleNamespace(shared_experts=shared),
            )
        )
        attention[index] = SimpleNamespace(
            kind=kind,
            projection_weight=weight(),
            core=SimpleNamespace(weights=tuple(weight() for _ in range(5))),
        )
        packed[index] = weight()
    return attention, layers, packed


def _host_runtime(test):
    """Mock CUDA transport only; execute the production range/controller classes."""
    events = []

    class Stream:
        def __init__(self, name="side"):
            self.name, self.cuda_stream = name, 17 if name == "main" else 19

        def wait_event(self, event):
            events.append(("wait_event", self.name, id(event)))

        def wait_stream(self, stream):
            events.append(("wait_stream", self.name, stream.name))

        def synchronize(self):
            events.append(("synchronize", self.name))

    class Event:
        def record(self, stream):
            events.append(("record", stream.name, id(self)))

    main, side = Stream("main"), Stream()
    driver = ModuleType("cuda.bindings.driver")
    driver.CUstream = int
    driver.CUlaunchConfig = SimpleNamespace
    driver.CUlaunchAttribute = lambda: SimpleNamespace(value=SimpleNamespace())
    driver.CUlaunchAttributeID = SimpleNamespace(
        CU_LAUNCH_ATTRIBUTE_PRIORITY=1,
        CU_LAUNCH_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT=2,
    )
    driver.cuCtxGetStreamPriorityRange = Mock(return_value=(0, 0, -2))
    driver.cuLaunchKernelEx = Mock(
        side_effect=lambda config, *args: (
            events.append(("launch", config.hStream)),
            (0,),
        )[1]
    )
    cuda, bindings = ModuleType("cuda"), ModuleType("cuda.bindings")
    cuda.__path__, bindings.__path__ = [], []
    cuda.bindings, bindings.driver = bindings, driver
    host = ModuleType(prefetch.__package__ + ".cutedsl.common.host")
    host._checked = lambda result: result[1:]
    host.resolve_kernel = Mock(return_value=(object(), 29))
    host.to_cute = lambda value, alignment: value
    kernel = ModuleType(prefetch.__package__ + ".cutedsl.common.prefetch")
    kernel.compile_kernel = Mock(
        side_effect=lambda *args: Mock(
            side_effect=lambda views, stream: events.append(("launch", stream))
        )
    )
    test.enterContext(
        patch.dict(
            sys.modules,
            {
                "cuda": cuda,
                "cuda.bindings": bindings,
                "cuda.bindings.driver": driver,
                host.__name__: host,
                kernel.__name__: kernel,
            },
        )
    )
    tensor = torch.tensor
    allocate = test.enterContext(
        patch.object(
            torch,
            "tensor",
            side_effect=lambda value, **kwargs: tensor(value, dtype=kwargs["dtype"]),
        )
    )
    test.enterContext(
        patch.object(torch.cuda, "device", return_value=contextlib.nullcontext())
    )
    streams = test.enterContext(patch.object(torch.cuda, "Stream", return_value=side))
    event_factory = test.enterContext(
        patch.object(torch.cuda, "Event", side_effect=Event)
    )
    test.enterContext(patch.object(torch.cuda, "current_stream", return_value=main))
    return SimpleNamespace(
        events=events,
        driver=driver,
        host=host,
        kernel=kernel,
        allocate=allocate,
        streams=streams,
        event_factory=event_factory,
    )


class TestPrefetchPolicy(unittest.TestCase):
    def test_exact_tp8_token_row_policy(self):
        for kind in ("mla", "kda"):
            for rows in range(1, 65):
                for point in POINTS:
                    with self.subTest(kind=kind, rows=rows, point=point):
                        policy = prefetch.prefetch_policy(point, kind, rows)
                        enabled = (
                            3 <= rows <= 8
                            if point == "oproj_down"
                            else (5 <= rows <= 8 if kind == "kda" else rows == 8)
                            if point == "attention_moe"
                            else rows <= 8
                        )
                        if not enabled:
                            self.assertIsNone(policy)
                            continue
                        self.assertIsInstance(policy, prefetch.PrefetchPolicy)
                        if point == "next_attention":
                            cap = 96 if rows <= 3 else 64 if rows <= 7 else 32
                            expected = (
                                cap * MIB,
                                32,
                                100 if rows == 1 else None,
                                False,
                            )
                        else:
                            expected = (
                                0,
                                8 if point == "oproj_down" else 32,
                                100,
                                True,
                            )
                        self.assertEqual(
                            (
                                policy.byte_cap,
                                policy.grid,
                                policy.carveout,
                                policy.priority,
                            ),
                            expected,
                        )

    def test_q6_counts_token_rows_not_requests(self):
        self.assertEqual(token_widths(6)[:2], (6, 12))
        for point in POINTS:
            self.assertIsNotNone(prefetch.prefetch_policy(point, "kda", 6))
            self.assertIsNone(prefetch.prefetch_policy(point, "kda", 12))
        self.assertIsNone(prefetch.prefetch_policy("attention_moe", "mla", 6))

    def test_invalid_policy_contract_rejected(self):
        for rows in (0, -1, True, 1.0, "6"):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                prefetch.prefetch_policy("next_attention", "mla", rows)
        for point, kind in (("missing", "mla"), ("next_attention", "other")):
            with self.subTest(point=point, kind=kind), self.assertRaises(ValueError):
                prefetch.prefetch_policy(point, kind, 1)
        self.assertIsNone(prefetch.prefetch_policy("next_attention", "mla", 65))


class TestPrefetchRanges(unittest.TestCase):
    def test_ranges_are_aligned_deduplicated_and_capped_in_consumption_order(self):
        first = _Tensor((64,), pointer=4096, storage_bytes=512)
        overlap = _Tensor((64,), pointer=4160, storage_pointer=4096, storage_bytes=512)
        last = _Tensor((32,), pointer=8192)
        owners, ranges = prefetch.range_plan((first, first, overlap, last))
        self.assertTrue(
            all(
                any(owner is source for source in (first, overlap, last))
                for owner in owners
            )
        )
        blocks = [
            address
            for pointer, size in ranges
            for address in range(pointer, pointer + size, 16)
        ]
        self.assertEqual(len(blocks), len(set(blocks)))
        self.assertEqual(
            set(blocks), set(range(4096, 4288, 16)) | set(range(8192, 8256, 16))
        )
        self.assertTrue(
            all(pointer % 16 == size % 16 == 0 and size > 0 for pointer, size in ranges)
        )
        _, capped = prefetch.range_plan((first, overlap, last), byte_cap=160)
        self.assertEqual(sum(size for _, size in capped), 160)
        self.assertEqual(
            [
                address
                for pointer, size in capped
                for address in range(pointer, pointer + size, 16)
            ],
            list(range(4096, 4256, 16)),
        )

    def test_safe_strided_span_and_short_tail_stay_inside_storage(self):
        weight = _Tensor((2, 8), strides=(16, 1), storage_bytes=48)
        owners, ranges = prefetch.range_plan((weight,))
        self.assertIs(owners[0], weight)
        self.assertEqual(ranges, ((4096, 48),))
        _, ranges = prefetch.range_plan((_Tensor((17,), dtype=torch.uint8),))
        self.assertEqual(ranges, ((4096, 16),))

    def test_invalid_pointer_stride_or_storage_cannot_launch(self):
        invalid = (
            _Tensor(pointer=4097),
            _Tensor(pointer=0),
            _Tensor(strides=(-1,)),
            _Tensor(storage_bytes=127),
            _Tensor(pointer=4080, storage_pointer=4096),
        )
        for value in invalid:
            with self.subTest(metadata=vars(value)), self.assertRaises(ValueError):
                prefetch.range_plan((value,))
        for cap in (-1, True, 16.0, 15):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                prefetch.range_plan((_Tensor(),), byte_cap=cap)

    def test_prepared_rejects_empty_subline_or_mixed_device_before_compilation(self):
        runtime = _host_runtime(self)
        for weights in (
            (),
            (_Tensor((0,)),),
            (_Tensor((7,), dtype=torch.uint8),),
            (_Tensor(device="cpu"),),
            (_Tensor(), _Tensor(pointer=8192, device="cuda:1")),
        ):
            with self.subTest(count=len(weights)), self.assertRaises(ValueError):
                prefetch.PreparedWeightRanges(weights)
        runtime.kernel.compile_kernel.assert_not_called()
        runtime.allocate.assert_not_called()

    def test_prepared_retains_table_driver_abi_and_compiles_before_launch(self):
        runtime = _host_runtime(self)
        weight = _Tensor()
        for carveout, priority in ((None, False), (100, False), (100, True)):
            with self.subTest(carveout=carveout, priority=priority):
                prepared = prefetch.PreparedWeightRanges(
                    (weight, weight), grid=8, carveout=carveout, priority=priority
                )
                self.assertEqual(prepared.ranges, ((4096, 128),))
                self.assertIs(prepared.weights[0], weight)
                self.assertEqual(prepared.table.tolist(), [[4096, 128]])
                pointer, code = prepared.table.data_ptr(), prepared.compiled
                counts = (
                    runtime.kernel.compile_kernel.call_count,
                    runtime.allocate.call_count,
                )
                for stream in (19, 23, 19):
                    prepared.launch(stream)
                    self.assertEqual(prepared.table.data_ptr(), pointer)
                    self.assertIs(prepared.compiled, code)
                self.assertEqual(
                    counts,
                    (
                        runtime.kernel.compile_kernel.call_count,
                        runtime.allocate.call_count,
                    ),
                )
                if carveout is not None:
                    self.assertEqual(prepared.config.gridDimX, 8)
                    self.assertEqual(prepared.config.blockDimX, 32)
                    self.assertEqual(prepared.config.hStream, 19)
                    values = {
                        attribute.id: attribute.value
                        for attribute in prepared.attributes
                    }
                    self.assertEqual(values[2].sharedMemCarveout, 100)
                    if priority:
                        self.assertEqual(values[1].priority, -2)


class TestPrefetchController(unittest.TestCase):
    def test_requested_subsets_keep_only_eligible_tables_and_reuse_resources(self):
        runtime = _host_runtime(self)
        pointer = 4096

        def tensor(shape):
            nonlocal pointer
            value = _Tensor(shape, pointer=pointer)
            pointer += 4096
            return value

        attention, layers, packed = _resources(tensor)
        for widths in ((2, 8, 64), (6, 12, 60), (16, 32, 64)):
            with self.subTest(widths=widths):
                controller = prefetch.PrefetchController(
                    attention, layers, packed, device="cuda:0", batches=widths
                )
                expected = {
                    (point, layer.layer_idx, rows)
                    for layer in layers
                    for point in POINTS
                    for rows in widths
                    if prefetch.prefetch_policy(
                        point, attention[layer.layer_idx].kind, rows
                    )
                    is not None
                    and not (point == "next_attention" and layer is layers[-1])
                }
                self.assertEqual(set(controller.tables), expected)
                counts = (
                    runtime.kernel.compile_kernel.call_count,
                    runtime.allocate.call_count,
                    runtime.event_factory.call_count,
                )
                for rows in (1, 2, 4, 6, 8, 12, 16, 32, 60, 64):
                    for layer in layers:
                        for point in POINTS:
                            key = (point, layer.layer_idx, rows)
                            self.assertEqual(controller.fork(*key), key in expected)
                    controller.join()
                    self.assertFalse(controller.pending)
                self.assertEqual(
                    counts,
                    (
                        runtime.kernel.compile_kernel.call_count,
                        runtime.allocate.call_count,
                        runtime.event_factory.call_count,
                    ),
                )

    def test_resource_inventory_counts_shared_table_and_owned_handles_once(self):
        module, Stream, Event = inventory_fixture(
            ROOT / "providers.py", _Tensor, kind="prefetch_table"
        )
        weight, table = _Tensor(), _Tensor((2, 2), pointer=8192, dtype=torch.int64)
        ranges = SimpleNamespace(
            weights=(weight,),
            table=table,
            code=SimpleNamespace(hidden=_Tensor(pointer=12288)),
        )
        event = Event()
        controller = SimpleNamespace(
            tables={"first": ranges, "shared": ranges},
            events={"first": event, "shared": event},
            stream=Stream(),
            done=Event(),
        )
        model = SimpleNamespace(
            modules=lambda: (), parameters=lambda: (weight,), buffers=lambda: ()
        )
        resources = SimpleNamespace(**resource_fields(prefetch=controller))
        inventory = module._owned_resource_inventory(resources, model)
        self.assertEqual(
            sorted(row["kind"] for row in inventory),
            ["event", "event", "prefetch_table", "stream"],
        )
        self.assertEqual(
            [row["data_ptr"] for row in inventory if row["kind"] == "prefetch_table"],
            [table.data_ptr()],
        )

    def test_tables_use_final_packed_storage_and_groups_share_owners(self):
        runtime = _host_runtime(self)
        pointer = 4096

        def tensor(shape):
            nonlocal pointer
            value = _Tensor(shape, pointer=pointer)
            pointer += 4096
            return value

        attention, layers, packed = _resources(tensor)
        widths = (1, 2, 3, 4, 5, 6, 7, 8, 12)
        controller = prefetch.PrefetchController(
            attention, layers, packed, device="cuda:0", batches=widths
        )
        expected = {
            (point, layer.layer_idx, rows)
            for layer in layers
            for point in POINTS
            for rows in widths
            if prefetch.prefetch_policy(point, attention[layer.layer_idx].kind, rows)
            is not None
            and not (point == "next_attention" and layer is layers[-1])
        }
        self.assertEqual(set(controller.tables), expected)
        self.assertIs(
            controller.tables["oproj_down", 0, 3], controller.tables["oproj_down", 0, 8]
        )
        self.assertIs(
            controller.tables["next_attention", 0, 2],
            controller.tables["next_attention", 0, 3],
        )
        self.assertIsNot(
            controller.tables["next_attention", 0, 1],
            controller.tables["next_attention", 0, 2],
        )
        for layer in layers:
            index = layer.layer_idx
            self.assertIs(
                controller.tables["oproj_down", index, 6].weights[0], packed[index]
            )
            if index < len(layers) - 1:
                self.assertIs(
                    controller.tables["next_attention", index, 6].weights[0],
                    attention[index + 1].projection_weight,
                )
        self.assertEqual(runtime.streams.call_count, 1)

        runtime.events.clear()
        counts = (
            runtime.kernel.compile_kernel.call_count,
            runtime.allocate.call_count,
            runtime.event_factory.call_count,
        )
        self.assertFalse(controller.fork("attention_moe", 0, 6))
        self.assertFalse(controller.fork("next_attention", 0, 12))
        controller.join()
        self.assertEqual(runtime.events, [])
        for point in POINTS:
            event = controller.events[point, 1]
            self.assertTrue(controller.fork(point, 1, 6))
            self.assertTrue(controller.pending)
            self.assertEqual(
                runtime.events[-3:],
                [
                    ("record", "main", id(event)),
                    ("wait_event", "side", id(event)),
                    ("launch", 19),
                ],
            )
        controller.join()
        self.assertEqual(
            runtime.events[-2:],
            [
                ("record", "side", id(controller.done)),
                ("wait_event", "main", id(controller.done)),
            ],
        )
        self.assertFalse(controller.pending)
        events = list(runtime.events)
        controller.join()
        self.assertEqual(runtime.events, events)
        self.assertEqual(
            counts,
            (
                runtime.kernel.compile_kernel.call_count,
                runtime.allocate.call_count,
                runtime.event_factory.call_count,
            ),
        )

    def test_range_owners_keep_borrowed_storage_alive_without_copy(self):
        weight = _Tensor()
        reference = weakref.ref(weight)
        owners, ranges = prefetch.range_plan((weight, weight))
        del weight
        gc.collect()
        self.assertIsNotNone(reference())
        self.assertTrue(any(owner is reference() for owner in owners))
        self.assertEqual(ranges, ((4096, 128),))

    def test_final_prepared_projection_and_core_weights_are_mapped(self):
        pointer = 4096

        def tensor(shape):
            nonlocal pointer
            result = _Tensor(shape, pointer=pointer)
            pointer += 4096
            return result

        attention, layers, _ = _resources(tensor)
        for layer in layers:
            prepared = attention[layer.layer_idx]
            expected = (
                prepared.projection_weight,
                *(
                    prepared.core.weights[2:5]
                    if prepared.kind == "mla"
                    else (layer.self_attn.f_b_proj.weight,)
                ),
                layer.self_attn.o_proj.weight,
            )
            found = prefetch.attention_weights(prepared, layer)
            self.assertEqual(tuple(map(id, found)), tuple(map(id, expected)))


class TestPrefetchHooks(unittest.TestCase):
    def test_executor_orders_mla_kda_decode_verify_and_final_join(self):
        for q, rows in ((1, 1), (1, 6), (6, 6), (6, 12)):
            with self.subTest(q=q, rows=rows):
                events = []

                def record(*entry, output=events):
                    output.append(entry)

                namespace = _definitions(
                    "executor.py",
                    {"KimiK3MegakernelExecutor"},
                    {
                        "torch": torch,
                        "partial": partial,
                        "token_widths": token_widths,
                        "_validate_step_inputs": Mock(),
                        "_verify_mla_metadata": lambda slots, lengths, writes, scratch, q: (
                            writes,
                            lengths,
                        ),
                    },
                )
                resources = execution_resources(
                    rows, layers=(0, 1, 92), mla_layers=(0, 92), tokens_per_request=q
                )
                for index, prepared in resources.attention.items():
                    for operation, label in (
                        (prepared.projection.run, "projection"),
                        (prepared.core.run, "core"),
                        (prepared.core.run_verify, "verify"),
                        (prepared.output.run, "oproj"),
                    ):
                        operation.side_effect = (
                            lambda *a, index=index, label=label, **k: record(
                                label, index
                            )
                        )
                resources.final.run.side_effect = lambda *a, **k: record("final")
                resources.prefetch = SimpleNamespace(
                    fork=partial(record, "fork"), join=partial(record, "join")
                )

                def tensor():
                    return torch.empty((rows, 7168))

                view = SimpleNamespace(
                    mla=SimpleNamespace(
                        write_slots=None, sequence_lengths=None, block_table=None
                    ),
                    kda=SimpleNamespace(
                        state_indices=None,
                        verify_scratch_indices=None,
                        track_decode_state=partial(record, "track"),
                    ),
                )
                executor = namespace["KimiK3MegakernelExecutor"](resources)
                out = executor.run(torch.zeros(rows, dtype=torch.int64), tensor(), view)
                self.assertEqual(
                    out.data_ptr(), resources.scratch["hidden.out"].data_ptr()
                )
                expected = []
                for index, kind in ((0, "mla"), (1, "kda"), (92, "mla")):
                    expected.append(("projection", index))
                    if kind == "kda":
                        expected.append(("fork", "attention_moe", index, rows))
                    expected.append(
                        ("verify" if kind == "kda" and q > 1 else "core", index)
                    )
                    if kind == "mla":
                        expected.append(("fork", "attention_moe", index, rows))
                    elif q == 1:
                        expected.append(("track", index))
                    expected.extend(
                        (("oproj", index), ("fork", "oproj_down", index, rows))
                    )
                expected.extend((("final",), ("join",)))
                self.assertEqual(events, expected)
                # Default-off resources retain an explicit None, including q6.
                resources.prefetch = None
                events.clear()
                executor.run(torch.zeros(rows, dtype=torch.int64), tensor(), view)
                disabled = [
                    event for event in expected if event[0] not in ("fork", "join")
                ]
                self.assertEqual(events, disabled)
                # Existing source-isolated fixtures have no prefetch attribute.
                del resources.prefetch
                events.clear()
                executor.run(torch.zeros(rows, dtype=torch.int64), tensor(), view)
                self.assertEqual(events, disabled)

    def test_both_moe_paths_fork_after_experts_and_before_tail(self):
        events = []

        def record(*entry):
            events.append(entry)

        main = SimpleNamespace(wait_event=lambda event: record("wait_shared"))
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(current_stream=lambda device: main)
        )
        namespace = _definitions(
            "cutedsl/moe/prepared.py", {"PreparedMoE"}, {"torch": fake_torch}
        )
        generic = object.__new__(namespace["PreparedMoE"])
        generic.state = SimpleNamespace(device="cuda:0", shared_done=object())
        generic.experts = SimpleNamespace(run=lambda *args: record("experts"))
        generic.prefetch = SimpleNamespace(fork=lambda *args: record("fork", *args))
        generic.layer_idx = 1
        out = object()
        generic.tail = SimpleNamespace(
            run=lambda *args, **kwargs: (record("tail"), out)[1]
        )
        self.assertIs(generic.finish(6, out, None, None, None, None, None), out)
        self.assertEqual(
            events,
            [
                ("experts",),
                ("fork", "next_attention", 1, 6),
                ("wait_shared",),
                ("tail",),
            ],
        )

        events.clear()
        namespace = _definitions(
            "cutedsl/moe/small_batch/chain.py",
            {"SmallBatchChain"},
            {"stream": lambda device: None, "Int32": int},
        )
        small = object.__new__(namespace["SmallBatchChain"])
        small.box = SimpleNamespace(device="cuda:0", tp=8, g=None, w=None, s=None)
        small.parity = small.slot = 0
        small.scalars = ()
        small.front = (lambda *args: record("front"), ())
        small.routed = (lambda *args: record("experts"), ())
        small.shared = (lambda *args: record("shared"), ())
        small.tail = small.tail_box = None
        small.tail_weights = SimpleNamespace(run=lambda *args: record("tail"))
        small.prefetch, small.layer_idx = generic.prefetch, 1
        value = torch.empty((1, 8))
        self.assertIs(small.run(value, lambda out: out), value)
        self.assertEqual(
            events,
            [
                ("front",),
                ("experts",),
                ("shared",),
                ("fork", "next_attention", 1, 1),
                ("tail",),
            ],
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestPrefetchCuda(unittest.TestCase):
    def test_native_controller_eager_graph_all_points_and_alternating_widths(self):
        # This imports the real controller, CUDA driver launch and CuTe kernel.
        # Synthetic immutable weights only: no model download or distributed job.
        device = torch.device("cuda", torch.cuda.current_device())
        counter = 0

        def tensor(shape):
            nonlocal counter
            counter += 1
            return torch.full(shape, counter, dtype=torch.bfloat16, device=device)

        attention, layers, packed = _resources(tensor)
        widths = (1, 2, 3, 5, 6, 8, 12)
        controller = prefetch.PrefetchController(
            attention, layers, packed, device=device, batches=widths
        )
        self.assertTrue(controller.tables)
        tables = {
            key: (id(value), value.table.data_ptr(), id(value.compiled))
            for key, value in controller.tables.items()
        }
        weights = {
            id(weight): weight
            for value in controller.tables.values()
            for weight in value.weights
        }
        snapshots = {key: weight.clone() for key, weight in weights.items()}
        graph_stream = torch.cuda.Stream(device=device)
        graphs, inputs, outputs = {}, {}, {}
        for capture_round in range(2):
            for rows in widths:
                x = torch.zeros((rows, 32), dtype=torch.float32, device=device)
                out = torch.empty_like(x)

                def run(x=x, out=out, rows=rows):
                    out.copy_(x)
                    for layer in layers:
                        index = layer.layer_idx
                        kind = attention[index].kind
                        if kind == "kda":
                            controller.fork("attention_moe", index, rows)
                        out.add_(index + 1)
                        if kind == "mla":
                            controller.fork("attention_moe", index, rows)
                        controller.fork("oproj_down", index, rows)
                        controller.fork("next_attention", index, rows)
                    controller.join()

                graph_stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(graph_stream):
                    for _ in range(3):
                        run()
                torch.cuda.current_stream(device).wait_stream(graph_stream)
                torch.testing.assert_close(out, torch.full_like(out, 6), rtol=0, atol=0)
                graph = torch.cuda.CUDAGraph()
                graph_stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.graph(graph, stream=graph_stream):
                    run()
                torch.cuda.current_stream(device).wait_stream(graph_stream)
                graphs[rows], inputs[rows], outputs[rows] = graph, x, out
                self.assertFalse(controller.pending)
                for value in (3.0, -2.0, 3.0):
                    x.fill_(value)
                    graph.replay()
                    torch.testing.assert_close(
                        out, torch.full_like(out, value + 6), rtol=0, atol=0
                    )
            for rows in (6, 12, 1, 8, 2, 12, 6):
                inputs[rows].fill_(capture_round + rows)
                graphs[rows].replay()
                torch.testing.assert_close(
                    outputs[rows],
                    torch.full_like(outputs[rows], capture_round + rows + 6),
                    rtol=0,
                    atol=0,
                )
        torch.cuda.synchronize(device)
        self.assertEqual(
            tables,
            {
                key: (id(value), value.table.data_ptr(), id(value.compiled))
                for key, value in controller.tables.items()
            },
        )
        for key, weight in weights.items():
            self.assertTrue(torch.equal(weight, snapshots[key]))
        self.assertTrue(
            any(
                key[0] == "attention_moe" and key[1] == 1 and key[2] == 6
                for key in tables
            )
        )
        self.assertFalse(any(key[2] == 12 for key in tables))
        self.assertFalse(any(key[:2] == ("next_attention", 2) for key in tables))


if __name__ == "__main__":
    unittest.main()

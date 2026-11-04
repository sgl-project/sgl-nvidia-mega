# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Private q1 DCP hybrid readiness, without registration or runner hooks.

The caller owns native bucket preparation, in-graph slot update, and pre-replay
assertions. This module never discovers a runner, translates slots, or pretends
that the normal HybridPoolView/binding fingerprint describes DCP storage.
"""

from __future__ import annotations

import torch

from .dcp_resources import DcpResources, PreparedDcpMla, _identity, _native_context

SGLANG_OPTIONAL_READS = {
    "srt/kernel_plugins/nvidia/_dcp_binding.py": (
        "_DcpHybridBinding",
        "_DcpHybridBucket",
        "_DcpHybridStepView",
    ),
}


def _native_types():
    # Only this explicit private path requires the private framework successor.
    try:
        from sglang.srt.kernel_plugins.nvidia._dcp_binding import (
            _DcpHybridBinding,
            _DcpHybridBucket,
            _DcpHybridStepView,
        )
    except ModuleNotFoundError as error:
        if error.name != "sglang.srt.kernel_plugins.nvidia._dcp_binding":
            raise
        raise RuntimeError(
            "Kimi DCP decode requires the SGLang DCP binding extension; "
            "ordinary DCP-off execution remains available"
        ) from error

    return _DcpHybridBinding, _DcpHybridBucket, _DcpHybridStepView


def _pool_tensor(name, tensor, *, dtype, shape, device):
    if not isinstance(tensor, torch.Tensor) or (
        tensor.device != device
        or tensor.device.type != "cuda"
        or tensor.dtype != dtype
        or tuple(tensor.shape) != shape
        or not tensor.is_contiguous()
        or tensor.data_ptr() % 16
    ):
        raise ValueError(f"private DCP invalid {name} storage")


def _kda_type():
    from .cutedsl.kda.prepared import PreparedKda

    return PreparedKda


class PreparedDcpHybrid:
    """Strong, opaque borrowed owners plus a separate one-shot readiness seal.

    Failure poisons the DCP resources, including a failure before compilation. No
    retry may reuse partially prepared cores/mailboxes. Borrowed pools are not
    added to ResourceSet's owned-storage inventory, and its legacy fingerprint
    remains unset. Construction is collective over the existing native TP group.
    """

    __slots__ = (
        "_resources",
        "_binding",
        "_pool",
        "_snapshot",
        "_plan",
        "_ready",
        "_failed",
        "_prepared_providers",
    )

    def __init__(self, resources, binding):
        binding_type, _, _ = _native_types()
        if not isinstance(binding, binding_type) or not isinstance(
            resources.dcp, DcpResources
        ):
            raise ValueError(
                "private DCP requires genuine native hybrid binding/resources"
            )
        self._resources, self._binding = resources, binding
        self._ready, self._failed = False, False
        self._prepared_providers = ()
        dcp = resources.dcp
        try:
            # Retain the original native CPU route before checking current owners.
            # A replaced parallel view must not strand peers in their agreement.
            agreement_group = dcp.agreement_group()
            error = None
            try:
                dcp.assert_live()
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("private DCP hybrid preparation during capture")
                if dcp._prepared:
                    raise RuntimeError("private DCP hybrid preparation is one-shot")
                self._pool = binding.preparation_view()
                self._snapshot, self._plan = self._candidate()
            except (
                AttributeError,
                KeyError,
                TypeError,
                ValueError,
                RuntimeError,
            ) as exc:
                error = f"{type(exc).__name__}: {exc}"
            _, agree = _native_context()
            agree(
                (error, None if error else self._plan),
                what="private q1 DCP hybrid preparation",
                group=agreement_group,
            )
            if error:
                raise ValueError(error)
            completion_error = None
            try:
                resources.prepare_dcp_mla_caches(dict(self._pool.mla.caches))
                for layer_id in resources.model.config.linear_layer_ids:
                    provider = resources.attention[layer_id]
                    state = self._pool.kda.state(layer_id)
                    caches = state.conv[0], state.temporal
                    # Retain borrowed storage even if compilation fails midway.
                    provider.caches = caches
                    provider.core.prepare_caches(caches)
                    resources.compilation_inventory.append(
                        {"kind": "attention_core", "layer_id": layer_id, "name": "kda"}
                    )
                torch.cuda.synchronize(dcp.device)
                if self._candidate() != (self._snapshot, self._plan):
                    raise RuntimeError(
                        "private DCP hybrid owners changed during preparation"
                    )
                self._prepared_providers = self._provider_snapshot()
            except Exception as exc:
                completion_error = exc
            try:
                agree(
                    None
                    if completion_error is None
                    else f"{type(completion_error).__name__}: {completion_error}",
                    what="private q1 DCP hybrid preparation completion",
                    group=dcp.agreement_group(),
                )
            except Exception as exc:
                if completion_error is not None:
                    raise completion_error from exc
                raise
            if completion_error is not None:
                raise completion_error
            self._ready = True
            self.assert_live()
        except BaseException:
            self._failed, dcp.failed = True, True
            raise

    @property
    def plan(self):
        """Immutable pointer/rank-local-device-free preparation facts."""
        return self._plan

    def _candidate(self):
        r, pool, dcp = self._resources, self._pool, self._resources.dcp
        self._binding.assert_live()
        dcp.assert_live()
        if self._binding.preparation_view() is not pool:
            raise RuntimeError("private DCP native preparation view changed")
        layout, group = pool.mla.layout, pool.group
        if (
            r.tokens_per_request != 1
            or type(r.tokens_per_request) is not int
            or r.binding is not None
            or r.binding_fingerprint is not None
            or r.aux_capture_layers is not None
            or r.prefetch is not None
            or r.embedding is not None
            or (layout.size, layout.page_size, layout.interleave, layout.canonical_skip)
            != (8, 64, 1, -1)
            or layout.rank != dcp.owner.rank
            or group.rank != dcp.owner.rank
            or group.global_ranks != dcp.owner.ranks
            or group.device_group is not dcp.owner.device_group
            or group.coordinator.device_group is not group.device_group
            or group.tp_coordinator is not _native_context()[0]().tp_group
            or tuple(r.widths) != dcp.widths
            or r.deadline_ns != dcp.deadline_ns
        ):
            raise ValueError(
                "private DCP hybrid geometry, groups or preparation differs"
            )
        full = tuple(r.model.config.full_attention_layer_ids)
        linear = tuple(r.model.config.linear_layer_ids)
        mappings = pool.mla.layer_mapping, pool.kda.layer_mapping
        if (
            len(full) != 24
            or len(linear) != 69
            or set(full).intersection(linear)
            or set(full + linear) != set(range(93))
            or tuple(layer.layer_idx for layer in r.layers) != tuple(range(93))
            or set(r.attention) != set(range(93))
            or set(pool.mla.caches) != set(full)
        ):
            raise ValueError("private DCP requires the complete 24-MLA/69-KDA mapping")
        for ids, mapping in zip((full, linear), mappings):
            if set(mapping) != set(ids) or set(mapping.values()) != set(
                range(len(ids))
            ):
                raise ValueError("private DCP native layer mapping differs")
        capacity = pool.kda.capacity
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("private DCP invalid KDA state capacity")
        snapshot, layouts = [], []
        for layer_id in full + linear:
            provider = r.attention[layer_id]
            kind = "mla" if layer_id in full else "kda"
            if provider.kind != kind:
                raise ValueError("private DCP attention provider kind differs")
            snapshot.append((layer_id, id(provider), id(provider.core)))
            if kind == "mla":
                core = provider.core
                if not isinstance(core, PreparedDcpMla) or core.resources is not dcp:
                    raise ValueError("private DCP requires owned PreparedDcpMla cores")
                if core.core.parity != full.index(layer_id) % 2:
                    raise ValueError("private DCP MLA parity differs from layer order")
                cache = pool.mla.caches[layer_id]
                if (
                    not isinstance(cache, torch.Tensor)
                    or cache.ndim != 3
                    or cache.shape[0] <= 0
                ):
                    raise ValueError("private DCP invalid local MLA pages")
                tensors = (("mla", cache, torch.uint8, (cache.shape[0], 64, 576)),)
            else:
                core = provider.core
                if not isinstance(core, _kda_type()) or (
                    core.heads != 12
                    or core.device != dcp.device
                    or set(core.policies) != set(r.widths)
                    or core.conv_dim_first is not False
                ):
                    raise ValueError(
                        "private DCP requires prepared q1/12-head KDA cores"
                    )
                state = pool.kda.state(layer_id)
                if (
                    len(state.conv) != 1
                    or state.intermediate_conv_window
                    or state.intermediate_ssm is not None
                ):
                    raise ValueError("private DCP requires ordinary q1 KDA states")
                tensors = (
                    ("conv", state.conv[0], torch.bfloat16, (capacity + 1, 3, 4608)),
                    (
                        "temporal",
                        state.temporal,
                        torch.float32,
                        (capacity + 1, 12, 128, 128),
                    ),
                )
            for name, tensor, dtype, shape in tensors:
                _pool_tensor(name, tensor, dtype=dtype, shape=shape, device=dcp.device)
                snapshot.append((layer_id, name, _identity(tensor)))
                layouts.append(
                    (layer_id, name, str(dtype), shape, tuple(tensor.stride()))
                )
        snapshot = (
            id(r),
            id(r.dcp),
            id(r.attention),
            id(r.model),
            id(r.layers),
            id(self._binding),
            id(pool),
            tuple(snapshot),
        )
        # No process-local pointer, rank or device index enters all-rank agreement.
        plan = (
            "private-q1-dcp-hybrid-v1",
            r.widths,
            r.deadline_ns,
            group.global_ranks,
            str(pool.mla.write_space),
            capacity,
            tuple(tuple(sorted(mapping.items())) for mapping in mappings),
            tuple(layouts),
        )
        return snapshot, plan

    def _provider_snapshot(self):
        rows = []
        for layer_id, provider in self._resources.attention.items():
            if provider.kind == "mla":
                provider.core.check_workspace()
                tensors = (provider.cache,)
                expected = (self._pool.mla.caches[layer_id],)
            else:
                state = self._pool.kda.state(layer_id)
                tensors, expected = provider.caches, (state.conv[0], state.temporal)
                if set(provider.core.kernels) != set(self._resources.widths):
                    raise RuntimeError("private DCP KDA widths were not prepared")
            if (
                tensors is None
                or len(tensors) != len(expected)
                or any(a is not b for a, b in zip(tensors, expected))
            ):
                raise RuntimeError(
                    "private DCP prepared provider borrowed storage changed"
                )
            executable_ids = (
                ()
                if provider.kind == "mla"
                else tuple(
                    (width, id(provider.core.kernels[width]))
                    for width in self._resources.widths
                )
            )
            rows.append(
                (layer_id, tuple(_identity(t) for t in tensors), executable_ids)
            )
        return tuple(rows)

    def assert_live(self):
        if self._failed or not self._ready:
            raise RuntimeError("private DCP hybrid is not ready; recreate resources")
        if not self._resources.dcp._prepared:
            raise RuntimeError("private DCP MLA preparation readiness changed")
        if (
            self._candidate() != (self._snapshot, self._plan)
            or self._provider_snapshot() != self._prepared_providers
        ):
            raise RuntimeError("private DCP prepared hybrid owners changed")

    def step_view(self, resources, bucket):
        """Read only. The caller must update slots and assert before graph replay."""
        _, bucket_type, view_type = _native_types()
        self.assert_live()
        if (
            resources is not self._resources
            or not isinstance(bucket, bucket_type)
            or (bucket.binding is not self._binding)
        ):
            raise ValueError(
                "private DCP execution requires its prepared native bucket"
            )
        view = bucket.step_view()
        if not isinstance(view, view_type):
            raise ValueError(
                "private DCP execution requires its native private step view"
            )
        return view


def validate_step(resources, input_ids, inputs_embeds, view):
    """Shape/type validation only; no tensor reads, allocations or translations."""
    from .executor import _require_tensor

    device = resources.scratch["hidden.0"].device
    batch = int(input_ids.shape[0])
    if (
        device.type != "cuda"
        or resources.tokens_per_request != 1
        or batch not in resources.widths
        or not 1 <= batch <= 64
        or view.kda.verify_scratch_indices is not None
        or not callable(view.kda.track_decode_state)
    ):
        raise RuntimeError("private DCP execution requires prepared CUDA q1 rows")
    tensors = (
        ("input_ids", input_ids, torch.int64, (batch,), (1,)),
        (
            "canonical MLA slots",
            view.mla.canonical_write_slots,
            torch.int64,
            (batch,),
            (1,),
        ),
        (
            "local MLA lengths",
            view.mla.local_sequence_lengths,
            torch.int32,
            (batch,),
            (1,),
        ),
        (
            "global MLA lengths",
            view.mla.global_sequence_lengths,
            torch.int32,
            (batch,),
            (1,),
        ),
        ("KDA state indices", view.kda.state_indices, torch.int32, (batch,), (1,)),
    )
    if inputs_embeds is not None:
        tensors += (
            ("inputs_embeds", inputs_embeds, torch.bfloat16, (batch, 7168), (7168, 1)),
        )
    for name, tensor, dtype, shape, strides in tensors:
        _require_tensor(
            name, tensor, dtype=dtype, shape=shape, device=device, strides=strides
        )
    table = view.mla.local_block_table
    if not isinstance(table, torch.Tensor) or (
        table.dtype != torch.int32
        or table.device != device
        or table.ndim != 2
        or table.shape[0] != batch
        or table.shape[1] <= 0
        or table.stride(1) != 1
        or table.stride(0) < table.shape[1]
    ):
        raise RuntimeError("private DCP invalid local MLA block table")

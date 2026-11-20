# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Private fixed-q6 readiness; native acceptance and checkpoint commit stay native.

The caller owns metadata planning, the typed bucket update and replay assertions.
Committed KDA state and verification checkpoints are distinct borrowed owners.
Real requests require positive state IDs; scratch zero remains a valid row. No
tensor values are inspected here, and no public admission or hooks are installed.
"""

from __future__ import annotations

import torch

from .dcp_execution import _kda_type, _pool_tensor
from .dcp_resources import DcpResources, PreparedDcpMla, _identity, _native_context

SGLANG_OPTIONAL_READS = {
    "srt/kernel_plugins/nvidia/_dcp_verify_binding.py": (
        "_DcpVerifyBinding",
        "_DcpVerifyBucket",
        "_DcpVerifyStepView",
    ),
}


def _native_types():
    try:
        from sglang.srt.kernel_plugins.nvidia._dcp_verify_binding import (
            _DcpVerifyBinding,
            _DcpVerifyBucket,
            _DcpVerifyStepView,
        )
    except ModuleNotFoundError as error:
        if error.name != "sglang.srt.kernel_plugins.nvidia._dcp_verify_binding":
            raise
        raise RuntimeError(
            "Kimi DCP verification requires the SGLang DCP binding extension; "
            "ordinary DCP-off execution remains available"
        ) from error

    return _DcpVerifyBinding, _DcpVerifyBucket, _DcpVerifyStepView


def _checkpoint_tensor(name, tensor, *, dtype, shape, device):
    # Native checkpoints may pad the leading scratch-pool pitch. The query and
    # per-state dimensions are dense, as required by PreparedKda's split caches.
    stride, inner = 1, []
    for extent in reversed(shape[1:]):
        inner.append(stride)
        stride *= extent
    if not isinstance(tensor, torch.Tensor) or (
        tensor.device != device
        or tensor.device.type != "cuda"
        or tensor.dtype != dtype
        or tuple(tensor.shape) != shape
        or tuple(tensor.stride()[1:]) != tuple(reversed(inner))
        or tensor.stride(0) < stride
        or tensor.stride(0) * tensor.element_size() % 16
        or tensor.data_ptr() % 16
    ):
        raise ValueError(f"private DCP verify invalid {name} storage")


class PreparedDcpVerify:
    """One-shot preparation of all MLA/KDA providers and native hidden taps.

    A failure poisons the resources. Borrowed pools, checkpoints and hidden
    buffers never enter the ResourceSet's owned-storage inventory. The normal
    DCP-off binding fingerprint stays unset, as does ordinary q1 readiness.
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
        "_taps",
        "_configured_taps",
    )

    def __init__(self, resources, binding):
        binding_type, _, _ = _native_types()
        if not isinstance(binding, binding_type) or not isinstance(
            resources.dcp, DcpResources
        ):
            raise ValueError(
                "private DCP verify requires native typed binding/resources"
            )
        self._resources, self._binding = resources, binding
        self._ready, self._failed, self._prepared_providers = False, False, ()
        self._configured_taps = None
        dcp = resources.dcp
        try:
            group = dcp.agreement_group()
            error = None
            try:
                dcp.assert_live()
                if torch.cuda.is_current_stream_capturing() or dcp._prepared:
                    raise RuntimeError(
                        "private DCP verify preparation is one-shot outside capture"
                    )
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
                what="private q6 DCP verify preparation",
                group=group,
            )
            if error:
                raise ValueError(error)
            completion_error = None
            try:
                resources.prepare_aux_capture()
                resources.prepare_dcp_mla_caches(dict(self._pool.mla.caches))
                for layer_id in resources.model.config.linear_layer_ids:
                    provider, state = (
                        resources.attention[layer_id],
                        self._pool.kda.state(layer_id),
                    )
                    provider.caches = (state.conv[0], state.temporal)
                    provider.intermediates = (
                        state.intermediate_conv_window[0],
                        state.intermediate_ssm,
                    )
                    provider.core.prepare_verify_caches(
                        provider.caches, provider.intermediates
                    )
                    resources.compilation_inventory.append(
                        {
                            "kind": "attention_core",
                            "layer_id": layer_id,
                            "name": "kda_verify",
                        }
                    )
                torch.cuda.synchronize(dcp.device)
                if self._candidate() != (self._snapshot, self._plan):
                    raise RuntimeError(
                        "private DCP verify owners changed during preparation"
                    )
                self._prepared_providers = self._provider_snapshot()
            except Exception as exc:
                completion_error = exc
            try:
                agree(
                    None
                    if completion_error is None
                    else f"{type(completion_error).__name__}: {completion_error}",
                    what="private q6 DCP verify preparation completion",
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
        return self._plan

    def _candidate(self):
        r, pool, dcp = self._resources, self._pool, self._resources.dcp
        self._binding.assert_live()
        dcp.assert_live()
        if self._binding.preparation_view() is not pool:
            raise RuntimeError("private DCP verify preparation view changed")
        layout, group = pool.mla.layout, pool.group
        if (
            type(r.tokens_per_request) is not int
            or r.tokens_per_request != 6
            or dcp.tokens_per_request != 6
            or pool.tokens_per_request != 6
            or r.binding is not None
            or r.binding_fingerprint is not None
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
            or not r.widths
            or any(type(m) is not int or m % 6 or not 6 <= m <= 60 for m in r.widths)
        ):
            raise ValueError("private DCP verify geometry/groups differ")
        configured = r.model.model.dspark_layers_to_capture
        if (
            not isinstance(configured, (list, tuple))
            or not configured
            or any(
                type(layer) is not int or not 0 <= layer < 93 for layer in configured
            )
        ):
            raise ValueError(
                "private DCP verify requires configured native hidden taps"
            )
        if self._configured_taps is None:
            self._configured_taps = configured
        elif configured is not self._configured_taps:
            raise RuntimeError("private DCP verify configured hidden owner changed")
        taps = tuple(configured)
        if tuple(sorted(set(taps))) != taps or pool.aux_hidden_width != 7168 * len(
            taps
        ):
            raise ValueError("private DCP verify hidden tap geometry differs")
        if r.aux_capture_layers is not None and r.aux_capture_layers != taps:
            raise ValueError("private DCP verify prepared hidden taps differ")
        full, linear = (
            tuple(r.model.config.full_attention_layer_ids),
            tuple(r.model.config.linear_layer_ids),
        )
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
            raise ValueError("private DCP verify requires all 24 MLA and 69 KDA layers")
        for ids, mapping in zip((full, linear), mappings):
            if set(mapping) != set(ids) or set(mapping.values()) != set(
                range(len(ids))
            ):
                raise ValueError("private DCP verify native layer mapping differs")
        capacity = pool.kda.capacity
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("private DCP verify invalid persistent capacity")
        snapshot, layouts, scratch_layout = [], [], None
        policies = {(m // 6, 6) for m in r.widths}
        for layer in full + linear:
            provider, kind = r.attention[layer], "mla" if layer in full else "kda"
            if provider.kind != kind:
                raise ValueError("private DCP verify provider kind differs")
            snapshot.append((layer, id(provider), id(provider.core)))
            if kind == "mla":
                core, cache = provider.core, pool.mla.caches[layer]
                if (
                    not isinstance(core, PreparedDcpMla)
                    or core.resources is not dcp
                    or core.core.parity != full.index(layer) % 2
                ):
                    raise ValueError("private DCP verify MLA owner/parity differs")
                if (
                    not isinstance(cache, torch.Tensor)
                    or cache.ndim != 3
                    or cache.shape[0] <= 0
                ):
                    raise ValueError("private DCP verify invalid MLA pages")
                tensors = (("mla", cache, torch.uint8, (cache.shape[0], 64, 576)),)
            else:
                core, state = provider.core, pool.kda.state(layer)
                if not isinstance(core, _kda_type()) or (
                    core.heads != 12
                    or core.device != dcp.device
                    or core.conv_dim_first is not False
                    or set(core.policies) != policies
                    or any(value != (6288, 1) for value in core.policies.values())
                    or len(state.conv) != 1
                    or len(state.intermediate_conv_window) != 1
                ):
                    raise ValueError(
                        "private DCP verify requires native q6 KDA policies/checkpoints"
                    )
                windows, checkpoints = (
                    state.intermediate_conv_window[0],
                    state.intermediate_ssm,
                )
                if not isinstance(windows, torch.Tensor) or windows.ndim != 4:
                    raise ValueError(
                        "private DCP verify invalid convolution checkpoints"
                    )
                c, q = tuple(windows.shape[:2])
                if c < max(r.widths) // 6 or q < 6:
                    raise ValueError(
                        "private DCP verify insufficient scratch checkpoint capacity"
                    )
                if scratch_layout is not None and scratch_layout != (c, q):
                    raise ValueError(
                        "private DCP verify scratch layout differs between layers"
                    )
                scratch_layout = c, q
                tensors = (
                    ("conv", state.conv[0], torch.bfloat16, (capacity + 1, 3, 4608)),
                    (
                        "temporal",
                        state.temporal,
                        torch.float32,
                        (capacity + 1, 12, 128, 128),
                    ),
                    ("checkpoint_conv", windows, torch.bfloat16, (c, q, 3, 4608)),
                    (
                        "checkpoint_ssm",
                        checkpoints,
                        torch.float32,
                        (c, q, 12, 128, 128),
                    ),
                )
            for name, tensor, dtype, shape in tensors:
                check = (
                    _checkpoint_tensor
                    if name.startswith("checkpoint_")
                    else _pool_tensor
                )
                check(name, tensor, dtype=dtype, shape=shape, device=dcp.device)
                snapshot.append((layer, name, _identity(tensor)))
                layouts.append((layer, name, str(dtype), shape, tuple(tensor.stride())))
        snapshot = (
            id(r),
            id(r.dcp),
            id(r.attention),
            id(r.model),
            id(r.layers),
            id(self._binding),
            id(pool),
            id(configured),
            taps,
            tuple(snapshot),
        )
        plan = (
            "private-q6-dcp-verify-v1",
            r.widths,
            r.deadline_ns,
            group.global_ranks,
            str(pool.mla.write_space),
            capacity,
            scratch_layout,
            taps,
            pool.aux_hidden_width,
            tuple(tuple(sorted(mapping.items())) for mapping in mappings),
            tuple(layouts),
        )
        self._taps = taps
        return snapshot, plan

    def _provider_snapshot(self):
        r, rows = self._resources, []
        if r.aux_capture_layers != self._taps:
            raise RuntimeError("private DCP verify hidden taps were not prepared")
        expected_keys = {(m // 6, 6) for m in r.widths}
        for layer, provider in r.attention.items():
            if provider.kind == "mla":
                provider.core.check_workspace()
                tensors, expected, executables = (
                    (provider.cache,),
                    (self._pool.mla.caches[layer],),
                    (),
                )
            else:
                state = self._pool.kda.state(layer)
                if provider.caches is None or provider.intermediates is None:
                    raise RuntimeError(
                        "private DCP verify KDA caches were not prepared"
                    )
                tensors = (*provider.caches, *provider.intermediates)
                expected = (
                    state.conv[0],
                    state.temporal,
                    state.intermediate_conv_window[0],
                    state.intermediate_ssm,
                )
                core = provider.core
                if (
                    set(core.verify_kernels) != expected_keys
                    or set(core.verify_layouts) != expected_keys
                ):
                    raise RuntimeError(
                        "private DCP verify KDA widths were not prepared"
                    )
                executables = tuple(
                    (key, id(core.verify_kernels[key]), core.verify_layouts[key])
                    for key in sorted(expected_keys)
                )
            if len(tensors) != len(expected) or any(
                a is not b for a, b in zip(tensors, expected)
            ):
                raise RuntimeError(
                    "private DCP verify borrowed provider storage changed"
                )
            rows.append((layer, tuple(_identity(t) for t in tensors), executables))
        return tuple(rows)

    def assert_live(self):
        if self._failed or not self._ready or not self._resources.dcp._prepared:
            raise RuntimeError("private DCP verify is not ready; recreate resources")
        if (
            self._candidate() != (self._snapshot, self._plan)
            or self._provider_snapshot() != self._prepared_providers
        ):
            raise RuntimeError("private DCP verify prepared owners changed")

    def step_view(self, resources, bucket):
        _, bucket_type, view_type = _native_types()
        self.assert_live()
        if (
            resources is not self._resources
            or not isinstance(bucket, bucket_type)
            or bucket.binding is not self._binding
        ):
            raise ValueError("private DCP verify requires its prepared native bucket")
        view = bucket.step_view()
        if not isinstance(view, view_type):
            raise ValueError("private DCP verify requires its typed native step view")
        return view


def validate_step(resources, input_ids, inputs_embeds, view):
    """Metadata only: native ingress owns valid positive state and scratch IDs."""
    from .executor import _require_tensor

    device, m = resources.scratch["hidden.0"].device, int(input_ids.shape[0])
    if (
        device.type != "cuda"
        or resources.tokens_per_request != 6
        or m not in resources.widths
        or m % 6
        or not 6 <= m <= 60
        or view.mla.tokens_per_request != 6
        or not resources.aux_capture_layers
    ):
        raise RuntimeError("private DCP verify requires prepared CUDA q6 rows")
    b, aux = m // 6, 7168 * len(resources.aux_capture_layers)
    tensors = (
        ("input_ids", input_ids, torch.int64, (m,), (1,)),
        (
            "canonical MLA slots",
            view.mla.canonical_write_slots,
            torch.int64,
            (m,),
            (1,),
        ),
        ("local MLA lengths", view.mla.local_sequence_lengths, torch.int32, (b,), (1,)),
        (
            "global MLA lengths",
            view.mla.global_sequence_lengths,
            torch.int32,
            (b,),
            (1,),
        ),
        ("KDA state indices", view.kda.state_indices, torch.int32, (b,), (1,)),
        (
            "KDA scratch indices",
            view.kda.verify_scratch_indices,
            torch.int32,
            (b,),
            (1,),
        ),
        (
            "auxiliary hidden states",
            view.aux_hidden_states,
            torch.bfloat16,
            (m, aux),
            (aux, 1),
        ),
    )
    if inputs_embeds is not None:
        tensors += (
            ("inputs_embeds", inputs_embeds, torch.bfloat16, (m, 7168), (7168, 1)),
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
        or table.shape[0] != b
        or table.shape[1] <= 0
        or table.stride(1) != 1
        or table.stride(0) < table.shape[1]
    ):
        raise RuntimeError("private DCP verify invalid local MLA block table")

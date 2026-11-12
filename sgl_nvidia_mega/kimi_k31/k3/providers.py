# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Cache-independent Kimi K3 megakernel resources and runtime-pool binding.

SGLang's megakernel framework imports this module only after configuration and
all-rank validation.  It prepares the frozen B1-B8/B16/B32/B64 TP8 provider set
and later binds it to SGLang-owned typed cache views.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from sglang.srt.distributed.communication_op import (
    tensor_model_parallel_all_reduce,
)
from sgl_nvidia_mega.kimi_k31.k3._host import HybridPoolView
from sglang.srt.runtime_context import get_parallel

from .projection_storage import (
    coalesce_kda_projection_storage,
    coalesce_mla_projection_storage,
)
from .shapes import enabled_tp_sizes, kda_pitch
from .shapes import heads as tp_heads
from .shapes import mla_pitch as k3_mla_pitch

PAYLOAD_PACKAGE = "sgl_nvidia_mega.kimi_k31.k3.cutedsl"
_BATCHES = (1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64)
_MAX_BATCH = max(_BATCHES)


def _batch_scratch(device, heads, mla_pitch, kda_width, batches):
    """Row-wise scratch shared by every batch of one boot. It holds _MAX_BATCH rows, or the largest requested
    batch when a caller asks for more (verify at Breq 16 = 128 rows), so no kernel writes past its rows."""
    rows = max(_MAX_BATCH, *batches)

    def bf16(*shape):
        return torch.empty((rows, *shape), dtype=torch.bfloat16, device=device)

    return {
        # PreparedMla.prepare_cache performs one bounded warm call.  Zero is a
        # finite FP8 encoding, unlike arbitrary uninitialized byte patterns.
        "mla.query": torch.zeros((rows, heads, 576), dtype=torch.uint8, device=device),
        "mla.projected": bf16(mla_pitch),
        "kda.projected": bf16(kda_width),
        "attention.local": bf16(heads * 128),
        "hidden.0": bf16(7168),
        "hidden.1": bf16(7168),
        "hidden.out": bf16(7168),
        "prefix": bf16(7168),
        "blocks": bf16(8, 7168),
    }

# Names read from SGLang internals. A contract test checks these before GPU use.
SGLANG_READS = {
    "srt/distributed/communication_op.py": ("tensor_model_parallel_all_reduce",),
    "srt/distributed/parallel_state.py": ("device_group",),
    "srt/arg_groups/fields/parallel.py": ("tp_group", "tp_rank", "tp_size"),
    "srt/kernel_plugins/nvidia/binding.py|srt/megakernels/binding.py": (
        "HybridPoolView",
        "cache",
        "capacity",
        "conv",
        "kda",
        "layer_mapping",
        "logical_dtype",
        "mla",
        "page_size",
        "post_capture_active",
        "reserved_capacity",
        "serving_capacity",
        "state",
        "storage_dtype",
        "temporal",
    ),
    "srt/runtime_context.py": ("get_parallel",),
}


def _validate_pool_tensor(
    name: str,
    value: torch.Tensor,
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    device: torch.device,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} is not a tensor")
    if value.device.type != "cuda" or value.device != device:
        raise ValueError(
            f"{name} must be on CUDA device {device}, found {value.device}"
        )
    if value.dtype != dtype:
        raise ValueError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if tuple(value.stride()) != stride:
        raise ValueError(
            f"{name} has stride {tuple(value.stride())}, expected {stride}"
        )
    if not value.is_contiguous():
        raise ValueError(f"{name} is not contiguous")
    if int(value.data_ptr()) % 16:
        raise ValueError(f"{name} is not 16-byte aligned")


def _digest(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def _float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"expected a scalar tensor, got {tuple(value.shape)}")
        return float(value.detach().item())
    return float(value)


def _module(suffix: str, origins: dict[str, str]) -> Any:
    name = f"{PAYLOAD_PACKAGE}.{suffix}"
    module = importlib.import_module(name)
    origins[name] = str(Path(inspect.getsourcefile(module) or "").resolve())
    return module


def _cuda_memory() -> dict[str, int]:
    torch.cuda.synchronize()
    device = torch.cuda.current_device()
    return {
        "allocated": int(torch.cuda.memory_allocated(device)),
        "reserved": int(torch.cuda.memory_reserved(device)),
        "peak_allocated": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved": int(torch.cuda.max_memory_reserved(device)),
    }


def _tensor_row(kind: str, name: str, value: torch.Tensor) -> dict[str, Any]:
    return {
        "kind": kind,
        "name": name,
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "dtype": str(value.dtype),
        # Device ordinals are process-local and therefore are not part of the
        # all-rank resource identity.
        "device": value.device.type,
        "nbytes": int(value.untyped_storage().nbytes()),
        "data_ptr": int(value.untyped_storage().data_ptr()),
    }


def _resource_kind(name: str) -> str:
    if ".boxes." in name:
        return "mailbox"
    if ".scratch." in name:
        return "scratch"
    if ".workspaces." in name:
        return "workspace"
    if ".projection_weight" in name or ".moe_front_weights." in name:
        return "projection"
    return "provider"


def _handle_row(kind: str, name: str, value: Any) -> dict[str, Any]:
    if kind == "stream":
        handle = int(value.cuda_stream)
        device = value.device.type
    else:
        handle = int(value.cuda_event)
        device = "cuda"
    if handle <= 0:
        raise RuntimeError(f"uninitialized CUDA {kind}: {name}")
    return {
        "kind": kind,
        "name": name,
        "shape": [],
        "stride": [],
        "dtype": type(value).__name__,
        "device": device,
        "nbytes": 0,
        "data_ptr": handle,
    }


def _owned_resource_inventory(root: Any, model: Any) -> list[dict[str, Any]]:
    """Inventory unique non-model storages and explicit CUDA handles.

    The inventory is ownership-based rather than allocator-delta-based: compiler
    and CUDA runtime allocations remain visible in the separate memory snapshots.
    """

    model_storages: set[int] = set()
    model_modules = {id(module) for module in model.modules()}
    for value in (*model.parameters(), *model.buffers()):
        if value.device.type == "cuda":
            model_storages.add(int(value.untyped_storage().data_ptr()))
    # Kimi's post-load transforms also retain a few deliberately unregistered
    # tensor views/copies (for example absorbed MLA W_UK/W_UV and fused KDA
    # views).  They remain model-owned even though state_dict does not list
    # them; exclude direct tensor attributes on every model module as well.
    for module in model.modules():
        for value in vars(module).values():
            if isinstance(value, torch.Tensor) and value.device.type == "cuda":
                model_storages.add(int(value.untyped_storage().data_ptr()))

    rows: list[dict[str, Any]] = []
    seen_objects: set[int] = set()
    seen_storages: set[int] = set()

    def visit(
        value: Any, name: str, depth: int = 0, forced_kind: str | None = None
    ) -> None:
        if depth > 12:
            return
        if isinstance(value, torch.Tensor):
            if value.device.type != "cuda":
                return
            pointer = int(value.untyped_storage().data_ptr())
            if pointer in model_storages or pointer in seen_storages:
                return
            seen_storages.add(pointer)
            rows.append(_tensor_row(forced_kind or _resource_kind(name), name, value))
            return
        if isinstance(value, torch.cuda.Stream):
            if id(value) in seen_objects:
                return
            seen_objects.add(id(value))
            rows.append(_handle_row("stream", name, value))
            return
        if isinstance(value, torch.cuda.Event):
            if id(value) in seen_objects:
                return
            seen_objects.add(id(value))
            rows.append(_handle_row("event", name, value))
            return
        if value is None or isinstance(value, (str, bytes, int, float, bool, type)):
            return
        identity = id(value)
        if identity in model_modules:
            return
        if identity in seen_objects:
            return
        seen_objects.add(identity)
        if isinstance(value, dict):
            for key in sorted(value, key=lambda item: str(item)):
                visit(value[key], f"{name}.{key}", depth + 1, forced_kind)
            return
        if isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                visit(item, f"{name}[{index}]", depth + 1, forced_kind)
            return
        values = getattr(value, "__dict__", None)
        if values is not None:
            for key in sorted(values):
                # Compiled code objects may retain a large Python compiler graph;
                # their device storage is accounted by CUDA memory, not tensor
                # inventory.  Provider-owned tensor attributes remain traversed.
                if key in {
                    "activation",
                    "code",
                    "codes",
                    "kernels",
                    "layers",
                    "model",
                }:
                    continue
                visit(values[key], f"{name}.{key}", depth + 1, forced_kind)

    # Register shared roots before walking providers that hold views of them so
    # classification never depends on dataclass attribute order.
    visit(root.boxes, "resources.boxes", forced_kind="mailbox")
    visit(root.scratch, "resources.scratch", forced_kind="scratch")
    visit(root.workspaces, "resources.workspaces", forced_kind="workspace")
    visit(root.runtime_handles, "resources.runtime_handles")
    for layer_id in sorted(root.attention):
        visit(
            root.attention[layer_id].projection_weight,
            f"resources.attention.{layer_id}.projection_weight",
            forced_kind="projection",
        )
    for layer_id in sorted(root.moe_front_weights):
        visit(
            root.moe_front_weights[layer_id],
            f"resources.moe_front_weights.{layer_id}",
            forced_kind="projection",
        )
    visit(root, "resources")
    rows.sort(key=lambda row: (row["kind"], row["name"]))
    return rows


def _pointer_free_digest(rows: Iterable[dict[str, Any]]) -> str:
    return _digest([{k: v for k, v in row.items() if k != "data_ptr"} for row in rows])


def _layer_proxy(layer: Any) -> Any:
    """Supply the one source-only AttnRes index from SGLang's equivalent facts."""

    return SimpleNamespace(
        self_attention_res_norm=layer.self_attention_res_norm,
        self_attention_res_proj=layer.self_attention_res_proj,
        input_layernorm=layer.input_layernorm,
        prev_valid_blocks=int(layer.prev_valid_blocks),
        is_block_write_layer=bool(layer.is_block_write_layer),
        block_write_idx=int(layer.layer_idx // layer.attn_res_block_size),
    )


def _final_proxy(backbone: Any) -> Any:
    return SimpleNamespace(
        output_attn_res_norm=backbone.output_attn_res_norm,
        output_attn_res_proj=backbone.output_attn_res_proj,
        norm=backbone.norm,
        num_attn_res_blocks=math.ceil(
            int(backbone.config.num_hidden_layers)
            / int(backbone.config.attn_res_block_size)
        ),
    )


def _mla_proxy(attention: Any) -> tuple[Any, tuple[torch.Tensor, ...]]:
    source_w_uk = attention.w_kc
    heads = int(source_w_uk.shape[0])
    if heads not in (6, 12) or source_w_uk.shape != (heads, 128, 512):
        raise ValueError(f"unsupported MLA W_UK_T shape: {source_w_uk.shape}")
    if source_w_uk.stride() != (65536, 1, 128):
        raise ValueError(
            f"unsupported MLA source W_UK_T stride: {source_w_uk.stride()}"
        )
    kernel_w_uk = source_w_uk.contiguous()
    if kernel_w_uk.stride() != (65536, 512, 1):
        raise ValueError(f"unsupported MLA W_UK_T stride: {kernel_w_uk.stride()}")
    if kernel_w_uk.data_ptr() % 16:
        raise ValueError("MLA kernel W_UK_T is not 16-byte aligned")
    if not torch.equal(kernel_w_uk, source_w_uk):
        raise ValueError("MLA W_UK_T layout normalization changed values")
    if attention.w_vc.shape != (heads, 512, 128):
        raise ValueError(f"unsupported MLA W_UV shape: {attention.w_vc.shape}")
    if attention.w_vc.stride() != (65536, 1, 512):
        raise ValueError(f"unsupported MLA W_UV stride: {attention.w_vc.stride()}")
    if attention.w_vc.data_ptr() % 16:
        raise ValueError("MLA W_UV is not 16-byte aligned")
    q_scale = getattr(attention.attn_mqa, "q_scale_float", None)
    k_scale = getattr(attention.attn_mqa, "k_scale_float", None)
    q_scale = 1.0 if q_scale is None else _float(q_scale)
    k_scale = 1.0 if k_scale is None else _float(k_scale)
    if (q_scale, k_scale) != (1.0, 1.0):
        raise ValueError(f"unsupported MLA activation scales: {(q_scale, k_scale)}")
    device = attention.q_b_proj.weight.device
    q_inv = torch.tensor([1.0], dtype=torch.float32, device=device)
    k_inv = torch.tensor([1.0], dtype=torch.float32, device=device)
    proxy = SimpleNamespace(
        q_a_layernorm=attention.q_a_layernorm,
        kv_a_layernorm=attention.kv_a_layernorm,
        q_b_proj=attention.q_b_proj,
        W_UK_T=kernel_w_uk,
        W_UV=attention.w_vc,
        _q_scale_inv=q_inv,
        _k_scale_inv=k_inv,
        _q_scale_float=q_scale,
        _k_scale_float=k_scale,
        scale=_float(attention.scaling),
        rms_norm_eps=_float(attention.q_a_layernorm.variance_epsilon),
    )
    if _float(attention.w_scale) != 1.0:
        raise ValueError("unsupported absorbed MLA weight scale")
    if abs(proxy.scale - 192.0**-0.5) >= 1e-12:
        raise ValueError(f"unsupported MLA attention scale: {proxy.scale}")
    if proxy.rms_norm_eps != 1e-5:
        raise ValueError(f"unsupported MLA RMS epsilon: {proxy.rms_norm_eps}")
    return proxy, (kernel_w_uk, q_inv, k_inv)


def _kda_proxy(attention: Any) -> Any:
    lower = getattr(getattr(attention, "attn", None), "lower_bound", None)
    if lower is None:
        raise ValueError("KDA gate lower bound is absent")
    if _float(lower) != -5.0:
        raise ValueError(f"unsupported KDA gate lower bound: {_float(lower)}")
    if attention.qkv_conv1d.bias is not None:
        raise ValueError("KDA convolution bias is unsupported")
    return SimpleNamespace(
        conv1d=attention.qkv_conv1d,
        dt_bias=attention.dt_bias,
        A_log=attention.A_log.reshape(-1),
        f_b_proj=attention.f_b_proj,
        o_norm=attention.o_norm,
        gate_lower_bound=_float(lower),
    )


def _prepare_mla_workspace(
    module: Any, device: torch.device, heads: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate the payload's cache-independent FlashInfer workspace early."""

    key = (device, heads)
    if key in module._FLASHINFER:
        return module._FLASHINFER[key]
    from flashinfer.utils import get_trtllm_gen_multi_ctas_kv_counter_bytes

    from sgl_nvidia_mega.kimi_k31.k3.compat import TRTLLM_KV_COUNTER_ROWS

    sms = torch.cuda.get_device_properties(device).multi_processor_count
    count = get_trtllm_gen_multi_ctas_kv_counter_bytes(TRTLLM_KV_COUNTER_ROWS, heads, sms)
    workspace = torch.zeros(128 << 20, dtype=torch.int8, device=device)
    counter = torch.zeros(count, dtype=torch.uint8, device=device)
    module._FLASHINFER[key] = workspace, counter
    return workspace, counter


@dataclass
class PreparedAttention:
    layer_id: int
    kind: str
    projection_weight: torch.Tensor
    projection: Any
    core: Any
    output: Any
    cache: torch.Tensor | None = None
    caches: tuple[torch.Tensor, torch.Tensor] | None = None


@dataclass
class ResourceSet:
    model: Any
    layers: tuple[Any, ...]
    attention: dict[int, PreparedAttention]
    feed_forward: dict[int, Any]
    moe_front_weights: dict[int, torch.Tensor]
    before: dict[int, Any]
    final: Any
    fused_before: dict[int, Any]
    boxes: dict[str, Any]
    scratch: dict[str, Any]
    workspaces: dict[str, torch.Tensor]
    runtime_handles: dict[str, Any]
    payload_modules: dict[str, str]
    compilation_inventory: list[dict[str, Any]]
    memory_before: dict[str, int]
    memory_after: dict[str, int]
    deadline_ns: int
    resource_inventory: list[dict[str, Any]] = field(default_factory=list)
    heads: int = 12  # local MLA/KDA heads (TP8 12, TP16 6)
    resource_digest: str = ""
    binding: dict[str, Any] | None = None
    binding_fingerprint: str | None = None
    memory_before_bind: dict[str, int] | None = None
    memory_after_bind: dict[str, int] | None = None

    @property
    def persistent_resource_bytes(self) -> int:
        return sum(int(row["nbytes"]) for row in self.resource_inventory)

    @property
    def owned_tensor_bytes(self) -> int:
        return self.persistent_resource_bytes

    @property
    def cuda_allocated_delta(self) -> int:
        return max(
            0,
            int(self.memory_after["allocated"]) - int(self.memory_before["allocated"]),
        )

    def refresh_inventory(self) -> None:
        self.resource_inventory = _owned_resource_inventory(self, self.model)
        self.resource_digest = _pointer_free_digest(self.resource_inventory)

    def _binding_candidate(self, hybrid: HybridPoolView) -> dict[str, Any]:
        full_ids = [int(v) for v in self.model.config.full_attention_layer_ids]
        linear_ids = [int(v) for v in self.model.config.linear_layer_ids]
        full_mapping = {
            int(layer_id): int(index)
            for layer_id, index in hybrid.mla.layer_mapping.items()
        }
        mamba_mapping = {
            int(layer_id): int(index)
            for layer_id, index in hybrid.kda.layer_mapping.items()
        }
        if set(full_mapping) != set(full_ids):
            raise ValueError("full-attention pool mapping differs from Kimi config")
        if set(mamba_mapping) != set(linear_ids):
            raise ValueError("linear-attention pool mapping differs from Kimi config")
        if set(full_mapping.values()) != set(range(len(full_ids))):
            raise ValueError("full-attention pool mapping is not dense")
        if set(mamba_mapping.values()) != set(range(len(linear_ids))):
            raise ValueError("linear-attention pool mapping is not dense")

        page_size = int(hybrid.mla.page_size)
        if page_size != 64:
            raise ValueError(f"unsupported MLA page size: {page_size}")
        if hybrid.mla.logical_dtype != torch.float8_e4m3fn:
            raise ValueError(
                "unsupported MLA logical dtype: "
                f"{hybrid.mla.logical_dtype}; expected {torch.float8_e4m3fn}"
            )
        if hybrid.mla.storage_dtype != torch.uint8:
            raise ValueError(
                "unsupported MLA storage dtype: "
                f"{hybrid.mla.storage_dtype}; expected {torch.uint8}"
            )
        reserved_capacity = int(hybrid.mla.reserved_capacity)
        serving_capacity = int(hybrid.mla.serving_capacity)
        if reserved_capacity <= 0:
            raise ValueError("Kimi K3 MLA reserved capacity is empty")
        if reserved_capacity % page_size:
            raise ValueError(
                "Kimi K3 MLA reserved capacity is not page aligned: "
                f"{reserved_capacity}"
            )
        if serving_capacity <= 0 or serving_capacity > reserved_capacity:
            raise ValueError(
                "Kimi K3 MLA serving capacity is outside its reservation: "
                f"{serving_capacity} versus {reserved_capacity}"
            )
        kda_capacity = int(hybrid.kda.capacity)
        if kda_capacity <= 0:
            raise ValueError("Kimi K3 KDA state capacity is empty")

        first_cache = hybrid.mla.cache(full_ids[0])
        first_parameter = next(iter(self.model.parameters()), None)
        device = (
            first_cache.device if first_parameter is None else first_parameter.device
        )
        expected_mla_shape = (
            reserved_capacity // page_size + 1,
            page_size,
            576,
        )
        heads, dim = self.heads, self.heads * 128
        expected_conv_shape = (kda_capacity + 1, 3, 3 * dim)
        expected_temporal_shape = (kda_capacity + 1, heads, 128, 128)

        local_identities: dict[str, int] = {}
        local_layouts: list[dict[str, Any]] = []
        for layer_id in full_ids:
            cache_view = hybrid.mla.cache(layer_id)
            _validate_pool_tensor(
                f"layer {layer_id} MLA cache",
                cache_view,
                dtype=torch.uint8,
                shape=expected_mla_shape,
                stride=(page_size * 576, 576, 1),
                device=device,
            )
            local_identities[f"mla.{layer_id}"] = int(cache_view.data_ptr())
            local_layouts.append(
                {
                    "kind": "mla",
                    "layer_id": layer_id,
                    "shape": list(cache_view.shape),
                    "stride": list(cache_view.stride()),
                    "dtype": str(cache_view.dtype),
                }
            )
        for layer_id in linear_ids:
            state = hybrid.kda.state(layer_id)
            if len(state.conv) != 1:
                raise ValueError(
                    f"layer {layer_id} has {len(state.conv)} KDA convolution states"
                )
            conv_view = state.conv[0]
            temporal_view = state.temporal
            _validate_pool_tensor(
                f"layer {layer_id} KDA convolution state",
                conv_view,
                dtype=torch.bfloat16,
                shape=expected_conv_shape,
                stride=(3 * 3 * dim, 3 * dim, 1),
                device=device,
            )
            # FP32 = default; BF16 = an optional lossy setting (--mamba-ssm-dtype bfloat16)
            temporal_dtype = torch.bfloat16 if temporal_view.dtype == torch.bfloat16 else torch.float32
            _validate_pool_tensor(
                f"layer {layer_id} KDA temporal state",
                temporal_view,
                dtype=temporal_dtype,
                shape=expected_temporal_shape,
                stride=(heads * 128 * 128, 128 * 128, 128, 1),
                device=device,
            )
            local_identities[f"kda_conv.{layer_id}"] = int(conv_view.data_ptr())
            local_identities[f"kda_temporal.{layer_id}"] = int(temporal_view.data_ptr())
            local_layouts.extend(
                (
                    {
                        "kind": "kda_conv",
                        "layer_id": layer_id,
                        "shape": list(conv_view.shape),
                        "stride": list(conv_view.stride()),
                        "dtype": str(conv_view.dtype),
                    },
                    {
                        "kind": "kda_temporal",
                        "layer_id": layer_id,
                        "shape": list(temporal_view.shape),
                        "stride": list(temporal_view.stride()),
                        "dtype": str(temporal_view.dtype),
                    },
                )
            )

        binding = {
            # The live serving capacity may shrink after graph capture. The
            # reservation and the tensor identities/layouts are immutable.
            "kv_reserved_capacity": reserved_capacity,
            "kv_post_capture_active": bool(hybrid.mla.post_capture_active),
            "kda_capacity": kda_capacity,
            "page_size": page_size,
            "logical_dtype": str(hybrid.mla.logical_dtype),
            "storage_dtype": str(hybrid.mla.storage_dtype),
            "full_attention_layer_ids": full_ids,
            "linear_layer_ids": linear_ids,
            "full_layer_dense_indices": {
                str(layer_id): int(full_mapping[layer_id]) for layer_id in full_ids
            },
            "linear_layer_dense_indices": {
                str(layer_id): int(mamba_mapping[layer_id]) for layer_id in linear_ids
            },
            "local_identities": local_identities,
            "local_layouts": local_layouts,
        }
        return binding

    def bind(self, hybrid: HybridPoolView) -> None:
        candidate = self._binding_candidate(hybrid)
        fingerprint = _digest(candidate)
        if self.binding_fingerprint is not None:
            if fingerprint != self.binding_fingerprint or candidate != self.binding:
                raise RuntimeError("Kimi K3 megakernel pool binding changed")
            return

        self.memory_before_bind = _cuda_memory()
        for layer_id, provider in self.attention.items():
            if provider.kind == "mla":
                cache = hybrid.mla.cache(layer_id)
                provider.core.prepare_cache(cache)
                provider.cache = cache
            else:
                state = hybrid.kda.state(layer_id)
                caches = (state.conv[0], state.temporal)
                provider.core.prepare_caches(caches)
                provider.caches = caches
            self.compilation_inventory.append(
                {"kind": "attention_core", "layer_id": layer_id, "name": provider.kind}
            )
        torch.cuda.synchronize()
        after = self._binding_candidate(hybrid)
        if after != candidate:
            raise RuntimeError("Kimi K3 pool identity changed during binding")
        self.binding = candidate
        self.binding_fingerprint = fingerprint
        self.memory_after_bind = _cuda_memory()


def build_resources(
    model: Any,
    *,
    context_length: int,
    memory_before: dict[str, int],
    deadline_ns: int,
    query_widths: tuple[int, ...] = (1,),
    packed_projection_weights: tuple[torch.Tensor, ...] | None = None,
    batches: tuple[int, ...] | None = None,
    mla_pitch: int | None = None,
    mla_storage: Any = coalesce_mla_projection_storage,
    mla_factory: Any = None,
    qf_g18: bool = False,
) -> ResourceSet:
    """Build every frozen TP8/B1-B8/B16/B32/B64 provider after all-rank consensus. ``qf_g18``: the K3.1 QF verify build
    under K31MK_DSA_QF_G18 (mla/head_gate.py); every other build refuses the knob."""

    batches = _BATCHES if batches is None else tuple(batches)
    memory_before = dict(memory_before)
    origins: dict[str, str] = {}
    projection_module = _module("gemm.projection", origins)
    mla_module = _module("mla.prepared", origins)
    head_gate_module = _module("mla.head_gate", origins)
    if bool(qf_g18) != head_gate_module.G18:
        raise ValueError(
            "K31MK_DSA_QF_G18=1 serves only the K3.1 TARGET_VERIFY build with K31MK_DSA_QF=1 (its gate readers would read "
            "column 2272 of a weight this build does not reorder)"
            if head_gate_module.G18 else "qf_g18 needs K31MK_DSA_QF_G18=1"
        )
    kda_module = _module("kda.prepared", origins)
    kda_verify_module = _module("kda.verify", origins)  # fix A coupling
    attnres_module = _module("attn_res.prepared", origins)
    oproj_module = _module("oproj.prepared", origins)
    twoshot_module = _module("common.twoshot", origins)
    dense_module = _module("moe.dense", origins)
    experts_module = _module("moe.experts_runtime", origins)
    planned_module = _module("moe.planned", origins)  # opt-in K31MK_EXPERTS_PLANNED
    w4a4_module = _module("mkv2.experts_w4a4", origins)  # opt-in K31MK_EXPERTS_W4A4 (lossy, TP16 B<=8)
    front_module = _module("moe.front", origins)
    prepared_moe_module = _module("moe.prepared", origins)
    chain_module = _module("moe.small_batch.chain", origins)
    tail_module = _module("moe.tail_runtime", origins)
    route_tap_module = _module("moe.route_tap", origins)  # opt-in K31MK_MOE_ROUTE_TAP (census, never timed)

    parallel = get_parallel()
    tp, rank = int(parallel.tp_size), int(parallel.tp_rank)
    if tp not in enabled_tp_sizes():
        raise ValueError(
            f"Kimi K3 megakernel resources serve TP{sorted(enabled_tp_sizes())}, got TP{tp}"
            " (TP16 needs K31MK_TP16=1)"
        )
    heads, kda_width = tp_heads(tp), kda_pitch(tp)
    mla_pitch = k3_mla_pitch(tp) if mla_pitch is None else mla_pitch
    group = parallel.tp_group.device_group
    device = next(model.parameters()).device
    owner = (group, device, tp, rank)
    backbone = model.model
    layers = tuple(backbone.layers[backbone.start_layer : backbone.end_layer])
    if len(layers) != 93:
        raise ValueError(f"expected the full 93-layer Kimi K3 model, got {len(layers)}")
    full_ids = [int(value) for value in model.config.full_attention_layer_ids]
    linear_ids = [int(value) for value in model.config.linear_layer_ids]
    observed_full = [
        int(layer.layer_idx) for layer in layers if hasattr(layer.self_attn, "q_b_proj")
    ]
    observed_linear = [
        int(layer.layer_idx)
        for layer in layers
        if not hasattr(layer.self_attn, "q_b_proj")
    ]
    if len(full_ids) != 24 or len(linear_ids) != 69:
        raise ValueError(
            "unexpected Kimi attention split: "
            f"MLA={len(full_ids)} KDA={len(linear_ids)}"
        )
    if observed_full != full_ids or observed_linear != linear_ids:
        raise ValueError(
            "Kimi layer modules disagree with the configured attention split"
        )

    # O-projection mailbox and two-shot rows follow the largest batch (64-row boots unchanged).
    oproj_cap = oproj_module.capacity(batches)
    mailbox = oproj_module.OprojMailbox(*owner, (2, tp, oproj_cap, 7168))
    # H-HOME (K31MK_OPROJ_HHOME_G / K31MK_OPROJ_HHOME_R, default unset): the two-shot gather rows and / or
    # owner receive slots in two localized segments (one per locality domain); PreparedOproj reads the rule from the object.
    # Both knobs unset = the served TwoShot and oproj.mfx_hhome is not imported (trees without it build as before).
    # Engagement (BOOTCHECKS): one stderr line 'K31MK_OPROJ_HHOME engaged ...' per process with each segment's locality domain attribute and
    # the SM -> locality domain table counts; knobs set without this line = not engaged (e.g. this hunk missing).
    import os  # noqa: PLC0415 - H-HOME knob check only
    hhome_module = None
    if os.environ.get("K31MK_OPROJ_HHOME_G") or os.environ.get("K31MK_OPROJ_HHOME_R"):
        hhome_module = _module("oproj.mfx_hhome", origins)
    if hhome_module is not None and (hhome_module.gather_rule() is not None or hhome_module.recv_rule()):
        oproj_twoshot = hhome_module.HomedTwoShot(*owner, cap=oproj_cap)
        ts, facts = oproj_twoshot, ""
        try:
            loc = importlib.import_module(hhome_module.__package__ + ".mfx_localized")
            segs = []
            for name, box in (("gather", getattr(ts, "box", None)), ("recv", getattr(ts, "rbox", None))):
                if box is not None:
                    lds = (loc.pointer_ld(box.local_ptr + i * hhome_module.SEG_ELEMS * 2) for i in range(2))
                    segs.append(f"{name}:{box.kind}:" + "/".join(map(str, lds)))
            facts = f"segments_ld={','.join(segs) or '-'} sm_table={int(getattr(ts, 'smtab', None) is not None)}"
        except Exception as e:  # noqa: BLE001 - the line must never stop a boot
            facts = f"facts_error={e!r}"
        print(f"K31MK_OPROJ_HHOME engaged gather={getattr(ts, 'rule', None)} recv={int(bool(getattr(ts, 'recv_dynamic', False)))} "
              f"cap={oproj_cap} {facts} pid={os.getpid()}", file=__import__("sys").stderr, flush=True)
    else:
        oproj_twoshot = twoshot_module.TwoShot(*owner, cap=oproj_cap)
    # [FP32 latent 3584 | FP32 logits 896] per token at every TP (stock's FP32 front).
    front_width = 2 * 3584 + 2 * 896
    # DRR (K31MK_MOE_ROUTE_DEFER_REARM; default unset = one parity, today's box): the routing kernel of
    # MoE ordinal k re-arms ordinal k-1's cells at its entry, so consecutive MoE ordinals use the two parities of one box.
    front_drr = bool(
        getattr(front_module, "route_defer_batches", lambda _: set())(tp) & {b for b in batches if b > 8}
    )
    # one row per token (64-row boots unchanged); DRR: two parities
    front_box = oproj_module.OprojMailbox(*owner, (2 if front_drr else 1, oproj_cap, front_width))
    # CDR (K31MK_OPROJ_DEFER_REARM; default unset = no cdr_last argument,
    # today's build): the split consumer of o-proj layer L re-arms layer L-1's slots (other parity) at its entry; the last o-proj
    # layer of this rank also re-arms its own, so every slot is armed at the forward boundary.
    oproj_cdr = bool(getattr(oproj_module, "oproj_defer_batches", lambda _: set())(tp) & set(batches))
    last_oproj = int(layers[-1].layer_idx)
    if oproj_cdr:  # g2 CDR-C1: CDR's parity offsets assume this rank's o-proj layers run in order with consecutive layer_idx
        ids = [int(layer.layer_idx) for layer in layers]
        if ids != list(range(ids[0], ids[0] + len(ids))):
            raise ValueError(f"K31MK_OPROJ_DEFER_REARM: o-proj layer ids must be consecutive (parity alternation), got {ids}")
    small_box = chain_module.SmallBatchBox(*owner) if max(query_widths) == 1 else None
    tail_box = tail_module.MoeTailBox(*owner, cap=oproj_cap)  # same rows as the o_proj mailbox
    boxes = {
        "oproj": mailbox,
        "oproj_twoshot": oproj_twoshot,
        "moe_front": front_box,
        "small_batch": small_box,
        "moe_tail": tail_box,
    }
    moe_state = prepared_moe_module.MoEScratch(tp, device, batches)
    # Materialize the state-owned event handles before resource inventory.  The
    # prepared MoE records them again at each execution boundary, so completing
    # an initialization record on its own stream does not alter runtime order.
    for event in (moe_state.begin, moe_state.shared_done, moe_state.gate_done, moe_state.front_done):
        event.record(moe_state.stream)
    moe_state.stream.synchronize()
    scratch = {
        **_batch_scratch(device, heads, mla_pitch, kda_width, batches),
        "moe": moe_state,
    }
    # Explicitly own the side-stream/event lifecycle used by the future executor.
    # Recording the event materializes its CUDA handle without entering a model
    # forward path.
    provider_stream = torch.cuda.Stream(device=device)
    provider_ready = torch.cuda.Event()
    provider_ready.record(provider_stream)
    provider_stream.synchronize()
    runtime_handles = {"provider": provider_stream, "ready": provider_ready}
    workspaces: dict[str, torch.Tensor] = {}

    beta = float(backbone.config.activation_situ_beta)
    linear_beta = float(backbone.config.activation_situ_linear_beta)
    moe_layers = [layer for layer in layers if bool(layer._is_moe_layer)]
    # F-L (K31MK_MOE_FRONT_LAMX; default unset = no box, served code): the MoE front's Lamport x box [2, 32, 7168]
    # (16-bit words, all 0xFFFF = the sentinel at build). MoE ordinal k's o-proj consumer also stores x into parity k % 2; its front
    # polls that parity and re-arms parity (k + 1) % 2 (ordinal k - 1's) at its entry. An even MoE count per forward keeps
    # the forward boundary armed (the last ordinal re-arms the first ordinal's parity).
    lamx_rows: frozenset = frozenset()
    if os.environ.get("K31MK_MOE_FRONT_LAMX", ""):
        lamx_table = front_module.batch_table(os.environ["K31MK_MOE_FRONT_LAMX"])
        lamx_rows = frozenset(b for b, v in lamx_table.items() if v in (1, 2, 3)) & frozenset(batches)
    lamx_box = None
    if lamx_rows:
        if len(moe_layers) % 2:
            raise ValueError(f"K31MK_MOE_FRONT_LAMX: needs an even MoE layer count (parity alternation), got {len(moe_layers)}")
        lamx_box = torch.full((2, 32, 7168), -1, dtype=torch.int16, device=device)
    route_tap = (
        route_tap_module.RouteTap(device, [int(layer.layer_idx) for layer in moe_layers], dump=rank == 0)
        if route_tap_module.enabled()
        else None
    )
    attention: dict[int, PreparedAttention] = {}
    feed_forward: dict[int, Any] = {}
    moe_front_weights: dict[int, torch.Tensor] = {}
    compilation: list[dict[str, Any]] = []

    for layer in layers:
        layer_id = int(layer.layer_idx)
        module = layer.self_attn
        is_mla = hasattr(module, "q_b_proj")
        if is_mla:
            proxy, _owned = _mla_proxy(module)
            projection_weight = (
                packed_projection_weights[layer_id]
                if packed_projection_weights is not None
                else mla_storage(module)
            )
            if tuple(projection_weight.shape) != (mla_pitch, 7168):
                raise ValueError(
                    f"layer {layer_id} MLA projection shape changed: "
                    f"{tuple(projection_weight.shape)}"
                )
            mla_tactics = {
                batch: projection_module.projection_tactic(
                    "mla", batch, device, width=mla_pitch
                )
                for batch in batches
            }
            if qf_g18:  # a reordered copy; the host's packed storage stays as its stock Parameters view it
                projection_weight = head_gate_module.relayout(projection_weight, heads)
                projection = head_gate_module.G18Projection(
                    projection_weight, mla_tactics, heads, projection_module
                ).prepare()
            else:
                projection = projection_module.PreparedProjection(
                    projection_weight, mla_tactics
                ).prepare()
            core = (mla_factory or mla_module.PreparedMla)(
                proxy if mla_factory is None else (proxy, module.indexer),
                {batch: mla_pitch for batch in batches},
                scratch["mla.query"],
                max_context=int(context_length),
                core="trtllm",
                dcp=None,
                queries=query_widths,
            )
            if not workspaces:
                workspace, counter = _prepare_mla_workspace(
                    mla_module, device, int(core.heads)
                )
                workspaces = {
                    "mla.flashinfer.workspace": workspace,
                    "mla.flashinfer.counter": counter,
                }
            kind = "mla"
        else:
            proxy = _kda_proxy(module)
            projection_weight = (
                packed_projection_weights[layer_id]
                if packed_projection_weights is not None
                else coalesce_kda_projection_storage(module)
            )
            if tuple(projection_weight.shape) != (kda_width, 7168):
                raise ValueError(
                    f"layer {layer_id} KDA projection shape changed: "
                    f"{tuple(projection_weight.shape)}"
                )
            projection = projection_module.PreparedProjection(
                projection_weight,
                {
                    batch: projection_module.projection_tactic(
                        "kda", batch, device, width=kda_width
                    )
                    for batch in batches
                },
                fenced_trigger=kda_verify_module.projection_fence(),
            ).prepare()
            core = kda_module.PreparedKda(
                proxy,
                {
                    batch: (
                        kda_width,
                        kda_module.decode_mode(batch, device),
                    )
                    for batch in batches
                },
                False,
            )
            kind = "kda"

        front = None
        if layer._is_moe_layer and small_box is not None:
            parity = layer_id % 2
            front = (
                small_box.front_views[parity],
                small_box.hint_views[parity],
                (1,),
            )
        output = oproj_module.PreparedOproj(
            layer,
            mailbox,
            batches,
            is_mla,
            front,
            oproj_twoshot,
            deadline_ns=deadline_ns,
            **({"cdr_last": layer_id == last_oproj} if oproj_cdr else {}),  # CDR (K31MK_OPROJ_DEFER_REARM)
            **(
                {"lamx": (lamx_box[moe_layers.index(layer) % 2].view(torch.bfloat16), lamx_rows)}
                if lamx_box is not None and layer._is_moe_layer
                else {}
            ),  # F-L (K31MK_MOE_FRONT_LAMX): this MoE ordinal's parity of the front's x box
        )
        attention[layer_id] = PreparedAttention(
            layer_id, kind, projection_weight, projection, core, output
        )
        compilation.extend(
            (
                {"kind": "projection", "layer_id": layer_id, "name": kind},
                {"kind": "oproj", "layer_id": layer_id, "name": kind},
            )
        )

        if not layer._is_moe_layer:
            feed_forward[layer_id] = dense_module.PreparedDenseMLP(
                layer.mlp,
                batches,
                reduce=tensor_model_parallel_all_reduce,
            )
            compilation.append(
                {"kind": "dense_mlp", "layer_id": layer_id, "name": "dense"}
            )
            continue

        routed = layer.mlp.experts
        # K31MK_MOE_DIES: re-home this layer's routed weights before any engine builds descriptors over them
        dies = experts_module.place_routed(routed) if tp == 16 and experts_module.dies_enabled() else ()
        experts_cls, planned_options = experts_module.PersistentExperts, {}
        if planned_module.enabled():
            experts_cls, planned_options = planned_module.PlannedExperts, planned_module.options()
        if tp == 16 and w4a4_module.enabled():
            experts_cls, planned_options = w4a4_module.W4A4Experts, w4a4_module.options()
        experts = experts_cls(
            routed.w13_weight,
            routed.w13_weight_scale,
            routed.w2_weight,
            routed.w2_weight_scale,
            beta=beta,
            linear_beta=linear_beta,
            activation_floor=False if max(query_widths) > 1 else None,
            **planned_options,
            # the tail box's rows (the experts' latent slot stride), passed only when it is not 64
            **({"box_rows": oproj_cap} if oproj_cap != 64 else {}),
            **({"dies": dies} if dies else {}),
        )
        if experts_module.w2_one96_enabled():  # : one resident FC2 copy; the stock FusedMoE path refuses
            experts_module.release_padded_w2(routed, experts)
        op = SimpleNamespace(
            rank=rank,
            contract=SimpleNamespace(
                tp_size=tp, rms_eps=float(model.config.rms_norm_eps)
            ),
        )
        ordinal = moe_layers.index(layer)
        tail = tail_module.PreparedTail(
            op,
            layer.mlp.routed_expert_norm.weight,
            layer.mlp.routed_expert_up_proj.weight,
            batches,
            (tail_box, ordinal, len(moe_layers)),
            deadline_ns=deadline_ns,
        )
        moe = SimpleNamespace(
            shared_experts=layer.mlp.shared_experts,
            gate=layer.mlp.gate,
            routed_expert_down_proj=layer.mlp.routed_expert_down_proj,
            routed_scaling_factor=layer.mlp.routed_scaling_factor,
            moe_renormalize=bool(model.config.moe_renormalize),
        )
        router, down = moe.gate.weight, moe.routed_expert_down_proj.weight
        router_rows, down_rows = 896 // tp, 3584 // tp
        router = router[rank * router_rows : (rank + 1) * router_rows]
        down = down[rank * down_rows : (rank + 1) * down_rows]
        packed_front = torch.cat(
            (router, router.new_zeros((-router_rows % 16, 7168)), down)
        ).detach()
        moe_front_weights[layer_id] = packed_front
        front_view, front_drr_kw = front_box, {}
        if front_drr:  # DRR: this MoE ordinal's parity; the last ordinal also re-arms its own cells (forward boundary)
            front_view = front_module.FrontParityView(front_box, ordinal % 2)
            front_drr_kw = {"drr": (ordinal % 2, ordinal == len(moe_layers) - 1)}
        generic_front = front_module.SharedFront(
            moe,
            packed_front,
            front_view,
            moe_state.front,
            batches,
            beta,
            linear_beta,
            plan=moe_state.plan,
            route_box=moe_state.experts.get("rubin8_workspace"),  # ROUTE_POLL (used only under K31MK_MOE_ROUTE_POLL)
            counters=moe_state.shared,
            #  (K31MK_MOE_WIDE_FOLD): the wide engine's plan, which the routing kernel then writes
            **({"wide_plan": moe_state.experts["rubin_wide"]["plan"]} if "rubin_wide" in moe_state.experts else {}),
            q=max(query_widths),
            deadline_ns=deadline_ns,
            **front_drr_kw,  # DRR (K31MK_MOE_ROUTE_DEFER_REARM)
            **({"lamx": (lamx_box, ordinal % 2)} if lamx_box is not None else {}),  # F-L (K31MK_MOE_FRONT_LAMX)
        )
        chain = (
            None
            if small_box is None
            else chain_module.SmallBatchChain(
                small_box,
                moe,
                experts,
                tail,
                layer_id,
                beta,
                linear_beta,
                None,
                deadline_ns=deadline_ns,
            )
        )
        prepared_moe = prepared_moe_module.PreparedMoE(
            generic_front,
            experts,
            tail,
            moe_state,
            prefetch=None,
            layer_idx=layer_id,
        )
        prepared_moe.small_batch = chain
        prepared_moe.tap = route_tap
        feed_forward[layer_id] = prepared_moe
        compilation.extend(
            (
                {"kind": "moe_experts", "layer_id": layer_id, "name": "mxfp4"},
                {
                    "kind": "moe_front",
                    "layer_id": layer_id,
                    "name": "b1_b8_b16_b32_b64",
                },
                {"kind": "moe_tail", "layer_id": layer_id, "name": "b1_b8_b16_b32_b64"},
                {"kind": "moe_small_batch", "layer_id": layer_id, "name": "b1"},
                {
                    "kind": "moe_prepared",
                    "layer_id": layer_id,
                    "name": "b2_b8_b16_b32_b64",
                },
            )
        )

    before = {
        int(layer.layer_idx): attnres_module.PreparedAttnRes.for_target(
            _layer_proxy(layer),
            has_delta=layer is not layers[0],
            deadline_ns=deadline_ns,
        )
        for layer in layers
    }
    final = attnres_module.PreparedAttnRes.for_target(
        _final_proxy(backbone), final=True, deadline_ns=deadline_ns
    )
    compilation.extend(
        {"kind": "attn_res", "layer_id": int(layer.layer_idx), "name": "input"}
        for layer in layers
    )
    compilation.append({"kind": "attn_res", "layer_id": 93, "name": "final"})
    fused_before: dict[int, Any] = {}
    followers = [before[int(layer.layer_idx)] for layer in layers[1:]] + [final]
    for layer, consumer in zip(layers, followers):
        if not layer._is_moe_layer:
            continue
        if tail_box.dra:
            # K31MK_ATTNRES_DEFER_REARM (DRA): ordinal k polls parity k % 2 and re-arms parity (k - 1) % 2
            # (k > 0) before its poll; the last ordinal also re-arms its own rows (both parities armed at the end).
            k = moe_layers.index(layer)
            prev_view = tail_box.out_parity((k - 1) % 2)[0] if k > 0 else None
            consumer.bind_ring(tail_box.out_parity(k % 2)[0], previous=prev_view, own=k == len(moe_layers) - 1)
        else:
            consumer.bind_ring(tail_box.out.buffer)
        fused_before[int(layer.layer_idx)] = consumer

    torch.cuda.synchronize()
    resources = ResourceSet(
        model=model,
        layers=layers,
        attention=attention,
        feed_forward=feed_forward,
        moe_front_weights=moe_front_weights,
        before=before,
        final=final,
        fused_before=fused_before,
        boxes=boxes,
        scratch=scratch,
        workspaces=workspaces,
        runtime_handles=runtime_handles,
        payload_modules=origins,
        compilation_inventory=compilation,
        memory_before=memory_before,
        memory_after=_cuda_memory(),
        deadline_ns=deadline_ns,
        heads=heads,
    )
    resources.refresh_inventory()
    return resources

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Cache-independent Kimi K3 megakernel resources and runtime-pool binding.

SGLang's megakernel framework imports this module only after configuration and
all-rank validation. It prepares the frozen TP8 token-row provider set and later
binds it to SGLang-owned typed cache views. Fixed-q4-8 verification borrows separate
checkpoint pools; only stock DSpark commits them to persistent recurrent state.
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
from typing import TYPE_CHECKING, Any

import torch
from sglang.srt.distributed.communication_op import (
    tensor_model_parallel_all_reduce,
)
from sglang.srt.runtime_context import get_parallel

try:
    from sglang.srt.kernel_plugins.nvidia.binding import HybridPoolView
except ModuleNotFoundError as error:  # SGLang before sgl-project/sglang#45389
    if error.name not in (
        "sglang.srt.kernel_plugins",
        "sglang.srt.kernel_plugins.nvidia",
        "sglang.srt.kernel_plugins.nvidia.binding",
    ):
        raise
    from sglang.srt.megakernels.binding import HybridPoolView

from .projection_storage import (
    coalesce_kda_projection_storage,
    coalesce_mla_projection_storage,
)
from .shape_policy import MAX_ROWS, mla_max_seq_len, prepared_widths

if TYPE_CHECKING:
    from .dcp_resources import DcpBuildOptions

PAYLOAD_PACKAGE = "sgl_nvidia_mega.kimi_k3.cutedsl"
_MAX_BATCH = MAX_ROWS


def _kda_policy(tokens_per_request: int, widths: tuple[int, ...] | None = None) -> dict:
    # Verify keys are (requests, q), unlike all row-wise projection/MoE policies.
    widths = prepared_widths(tokens_per_request, widths)
    if tokens_per_request > 1:
        return {
            (rows // tokens_per_request, tokens_per_request): (6288, 1)
            for rows in widths
        }
    return {rows: (6288, 1 if rows == 1 else 2 if rows <= 8 else 0) for rows in widths}


def _projection_policies(pitch: int, widths: tuple[int, ...]) -> dict:
    """Use the existing TP8 tactic at the next row bin, including odd q rows."""
    if pitch not in (3648, 6288):
        raise ValueError(f"unsupported Kimi K3 projection pitch: {pitch}")
    policies = {}
    for rows in widths:
        if not 1 <= rows <= _MAX_BATCH:
            raise ValueError(f"unsupported Kimi K3 projection rows: {rows}")
        tile = next(size for size in (8, 16, 32, 64) if rows <= size)
        if pitch == 6288:
            tactic = (
                (64, 64, 1, 6) if tile == 64 else (128, tile, 2, 5 if tile == 32 else 6)
            )
        else:
            tactic = (
                64,
                tile,
                4 if tile == 8 else 2,
                {8: 8, 16: 10, 32: 8, 64: 5}[tile],
            )
        policies[rows] = tactic
    return policies


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
        "intermediate_conv_window",
        "intermediate_ssm",
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
    "srt/models/kimi_k3.py": ("dspark_layers_to_capture",),
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
    if source_w_uk.shape != (12, 128, 512):
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
    if attention.w_vc.shape != (12, 512, 128):
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

    sms = torch.cuda.get_device_properties(device).multi_processor_count
    count = get_trtllm_gen_multi_ctas_kv_counter_bytes(64, heads, sms)
    workspace = torch.zeros(128 << 20, dtype=torch.int8, device=device)
    counter = torch.zeros(count, dtype=torch.uint8, device=device)
    module._FLASHINFER[key] = workspace, counter
    return workspace, counter


def _validate_mla_core(core: str) -> None:
    if core not in ("trtllm", "tokenspeed"):
        raise ValueError(f"unsupported private MLA core: {core!r}")
    if core == "tokenspeed":
        from importlib.metadata import PackageNotFoundError, version

        try:
            installed = version("tokenspeed-mla")
        except PackageNotFoundError as error:
            raise RuntimeError(
                "TokenSpeed MLA requires tokenspeed-mla==2.0.2"
            ) from error
        if installed != "2.0.2":
            raise RuntimeError(
                f"TokenSpeed MLA requires tokenspeed-mla==2.0.2, found {installed}"
            )


class _PreparedTokenSpeedMla:
    """Retain the final shared workspace after all native MLA prewarms.

    Calls use the caller's current stream, like the native prepared core. The
    resource set owns one stream-ordered workspace, not concurrent MLA calls.
    No module-global registry is changed; replacement by another owner is an
    error at the next owned entry, while this strong reference keeps old graphs'
    storage alive.
    """

    def __init__(self, core: Any, module: Any):
        if core.dcp is not None or set(core.cores.values()) != {"tokenspeed"}:
            raise ValueError("TokenSpeed preparation requires a DCP-off forced core")
        self.core = core
        # A lookup closure avoids treating arbitrary module globals as owned
        # resources during the resource inventory's object traversal.
        self._workspace_pool = lambda: module._WORKSPACE
        self.workspace: torch.Tensor | None = None
        self._prepared_geometry: tuple | None = None
        self._workspace_identity: tuple | None = None

    @property
    def heads(self) -> int:
        return self.core.heads

    @property
    def weights(self) -> tuple[torch.Tensor, ...]:
        return self.core.weights

    def _geometry(self) -> tuple:
        core = self.core
        return (
            core.device,
            core.heads,
            core.max_seq,
            core.dcp,
            tuple(sorted(core.cores.items())),
            tuple(sorted(core.pitches.items())),
            tuple(sorted(core.kernels)),
        )

    @staticmethod
    def _identity(workspace: torch.Tensor) -> tuple:
        return (
            workspace.device,
            workspace.dtype,
            tuple(workspace.shape),
            tuple(workspace.stride()),
            int(workspace.data_ptr()),
            int(workspace.untyped_storage().data_ptr()),
            int(workspace.untyped_storage().nbytes()),
        )

    def prepare_cache(self, cache: torch.Tensor) -> _PreparedTokenSpeedMla:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("TokenSpeed prewarm cannot run during capture")
        if self.workspace is not None:
            raise RuntimeError("TokenSpeed workspace is finalized; cannot reprepare")
        self.core.prepare_cache(cache)
        self._prepared_geometry = self._geometry()
        return self

    def finalize_workspace(self) -> torch.Tensor:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("TokenSpeed workspace cannot finalize during capture")
        if self.workspace is not None:
            self.check_workspace()
            return self.workspace
        if (
            self._prepared_geometry is None
            or self._geometry() != self._prepared_geometry
        ):
            raise RuntimeError(
                "TokenSpeed shapes must be prewarmed before finalization"
            )
        workspace = self._workspace_pool().get((self.core.device, self.heads))
        if (
            not isinstance(workspace, torch.Tensor)
            or workspace.device != self.core.device
            or workspace.device.type != "cuda"
            or workspace.dtype != torch.int8
            or workspace.ndim != 1
            or not workspace.is_contiguous()
            or workspace.numel() == 0
        ):
            raise RuntimeError("TokenSpeed prewarm did not produce a valid workspace")
        self.workspace = workspace
        self._workspace_identity = self._identity(workspace)
        return workspace

    def check_workspace(self) -> None:
        if self.workspace is None:
            raise RuntimeError("TokenSpeed workspace must finalize before execution")
        if (
            self._workspace_pool().get((self.core.device, self.heads))
            is not self.workspace
            or self._identity(self.workspace) != self._workspace_identity
            or self._geometry() != self._prepared_geometry
        ):
            raise RuntimeError("TokenSpeed prepared workspace or geometry changed")

    def run(self, *args: Any, **kwargs: Any) -> Any:
        self.check_workspace()
        result = self.core.run(*args, **kwargs)
        self.check_workspace()
        return result


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
    intermediates: tuple[torch.Tensor, torch.Tensor] | None = None


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
    dcp: Any | None = None
    prefetch: Any | None = None
    embedding: Any | None = None
    tokens_per_request: int = 1
    widths: tuple[int, ...] = ()
    aux_capture_layers: tuple[int, ...] | None = None
    resource_inventory: list[dict[str, Any]] = field(default_factory=list)
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

    def prepare_aux_capture(self) -> None:
        """Compile the configured pre-norm mixture taps before graph capture.

        SGLang configures target taps after build but before preparation. The
        next consumer owns each tap's mixture weights; the final consumer owns
        a last-layer tap. No per-forward auxiliary storage is allocated here.
        """
        if self.tokens_per_request == 1:
            return
        configured = self.model.model.dspark_layers_to_capture
        if configured is None:
            raise ValueError("Kimi K3 DSPARK auxiliary capture was not configured")
        taps = tuple(int(layer_id) for layer_id in configured)
        if (
            not taps
            or tuple(sorted(set(taps))) != taps
            or any(layer_id < 0 or layer_id >= len(self.layers) for layer_id in taps)
        ):
            raise ValueError(f"invalid Kimi K3 DSPARK auxiliary capture layers: {taps}")
        if self.aux_capture_layers is not None:
            if taps != self.aux_capture_layers:
                raise RuntimeError("Kimi K3 DSPARK auxiliary capture layers changed")
            return
        consumers = [
            self.final
            if layer_id == len(self.layers) - 1
            else self.before[layer_id + 1]
            for layer_id in taps
        ]
        if any(consumer is None for consumer in consumers):
            raise ValueError("Kimi K3 DSPARK tap has no next consumer")
        for layer_id, consumer in zip(taps, consumers):
            consumer.bind_aux(mixture=True)
            self.compilation_inventory.append(
                {"kind": "attn_res_aux", "layer_id": layer_id, "name": "mixture"}
            )
        self.aux_capture_layers = taps

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
        expected_conv_shape = (kda_capacity + 1, 3, 4608)
        expected_temporal_shape = (kda_capacity + 1, 12, 128, 128)

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
                stride=(3 * 4608, 4608, 1),
                device=device,
            )
            _validate_pool_tensor(
                f"layer {layer_id} KDA temporal state",
                temporal_view,
                dtype=torch.float32,
                shape=expected_temporal_shape,
                stride=(12 * 128 * 128, 128 * 128, 128, 1),
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
            if self.tokens_per_request > 1:
                if len(state.intermediate_conv_window) != 1:
                    raise ValueError(
                        f"layer {layer_id} needs one KDA intermediate convolution window"
                    )
                windows = state.intermediate_conv_window[0]
                checkpoints = state.intermediate_ssm
                if not isinstance(windows, torch.Tensor) or windows.ndim != 4:
                    raise ValueError(f"layer {layer_id} KDA windows must have rank 4")
                capacity, queries = map(int, windows.shape[:2])
                if capacity <= 0 or queries < self.tokens_per_request:
                    raise ValueError(
                        f"layer {layer_id} KDA scratch cannot hold q{self.tokens_per_request}"
                    )
                for name, tensor, dtype, shape, stride in (
                    (
                        "kda_intermediate_conv",
                        windows,
                        torch.bfloat16,
                        (capacity, queries, 3, 4608),
                        (queries * 3 * 4608, 3 * 4608, 4608, 1),
                    ),
                    (
                        "kda_intermediate_ssm",
                        checkpoints,
                        torch.float32,
                        (capacity, queries, 12, 128, 128),
                        (queries * 12 * 128 * 128, 12 * 128 * 128, 128 * 128, 128, 1),
                    ),
                ):
                    _validate_pool_tensor(
                        f"layer {layer_id} {name}",
                        tensor,
                        dtype=dtype,
                        shape=shape,
                        stride=stride,
                        device=device,
                    )
                    local_identities[f"{name}.{layer_id}"] = int(tensor.data_ptr())
                    local_layouts.append(
                        {
                            "kind": name,
                            "layer_id": layer_id,
                            "shape": list(tensor.shape),
                            "stride": list(tensor.stride()),
                            "dtype": str(tensor.dtype),
                        }
                    )

        binding = {
            # The live serving capacity may shrink after graph capture. The
            # reservation and the tensor identities/layouts are immutable.
            "kv_reserved_capacity": reserved_capacity,
            "kv_post_capture_active": bool(hybrid.mla.post_capture_active),
            "kda_capacity": kda_capacity,
            "tokens_per_request": self.tokens_per_request,
            "aux_capture_layers": list(self.aux_capture_layers or ()),
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

    def prepare_dcp_mla_caches(self, caches: dict[int, torch.Tensor]) -> None:
        """Private MLA-only handoff, not a legacy or full-model pool binding."""
        if self.dcp is None:
            raise RuntimeError("DCP resources were not explicitly requested")
        self.dcp.prepare_mla_caches(self.attention, caches)
        workspace = self.dcp.workspace
        self.workspaces["mla.dcp.tokenspeed.workspace"] = workspace
        row = _tensor_row(
            "workspace", "resources.workspaces.mla.dcp.tokenspeed.workspace", workspace
        )
        if not any(
            item["data_ptr"] == row["data_ptr"] for item in self.resource_inventory
        ):
            self.resource_inventory.append(row)
            self.resource_inventory.sort(key=lambda item: (item["kind"], item["name"]))
        self.resource_digest = _pointer_free_digest(self.resource_inventory)
        # Deliberately leave binding_fingerprint unset: no KDA/step binding exists
        # in this resource-only checkpoint, and the full executor must refuse it.

    def bind(self, hybrid: HybridPoolView) -> None:
        if self.dcp is not None:
            raise RuntimeError("private DCP requires a separate native hybrid binding")
        if self.tokens_per_request > 1 and self.aux_capture_layers is None:
            raise RuntimeError(
                "Kimi K3 DSPARK auxiliary taps must be prepared before binding"
            )
        candidate = self._binding_candidate(hybrid)
        fingerprint = _digest(candidate)
        if self.binding_fingerprint is not None:
            for provider in self.attention.values():
                if isinstance(provider.core, _PreparedTokenSpeedMla):
                    provider.core.check_workspace()
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
                if self.tokens_per_request > 1:
                    intermediates = (
                        state.intermediate_conv_window[0],
                        state.intermediate_ssm,
                    )
                    provider.core.prepare_verify_caches(caches, intermediates)
                    provider.intermediates = intermediates
                else:
                    provider.core.prepare_caches(caches)
                provider.caches = caches
            self.compilation_inventory.append(
                {"kind": "attention_core", "layer_id": layer_id, "name": provider.kind}
            )
        tokenspeed = [
            provider.core
            for provider in self.attention.values()
            if isinstance(provider.core, _PreparedTokenSpeedMla)
        ]
        if tokenspeed:
            # Every layer and prepared shape has now warmed the module-global
            # workspace. Retain its final maximum-size owner, never an earlier
            # layer's buffer that a later warmup may have replaced.
            workspace = tokenspeed[0].finalize_workspace()
            for core in tokenspeed[1:]:
                if core.finalize_workspace() is not workspace:
                    raise RuntimeError(
                        "TP8 MLA layers must share one TokenSpeed workspace"
                    )
            self.workspaces["mla.tokenspeed.workspace"] = workspace
            # Native KV/KDA pools are borrowed by now. Do not re-walk providers
            # after binding: register only this explicitly owned new storage.
            row = _tensor_row(
                "workspace", "resources.workspaces.mla.tokenspeed.workspace", workspace
            )
            if not any(
                item["kind"] not in ("stream", "event")
                and item["data_ptr"] == row["data_ptr"]
                for item in self.resource_inventory
            ):
                self.resource_inventory.append(row)
                self.resource_inventory.sort(
                    key=lambda item: (item["kind"], item["name"])
                )
            self.resource_digest = _pointer_free_digest(self.resource_inventory)
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
    tokens_per_request: int = 1,
    enable_prefetch: bool = False,
    enable_embedding: bool = False,
    widths: tuple[int, ...] | None = None,
    mla_core: str = "trtllm",
    dcp: DcpBuildOptions | None = None,
) -> ResourceSet:
    """Build TP8 resources before graph capture, without enabling a server mode.

    Prefetch and specialized embedding default OFF; do not prepare their optional
    resources unless requested. Omitted widths use the normal query-mode policy.
    The default MLA core is TRTLLM. Explicit mla_core="tokenspeed" requires
    tokenspeed-mla==2.0.2 and prewarming/finalizing its owned workspace; its
    bounded q1/q6 evidence is at 8K, not arbitrary long contexts.

    DCP requires both mla_core="tokenspeed" and typed DcpBuildOptions, with native
    TP8/DCP8 groups already established. Supplying dcp alone is rejected. It does
    not enable a registered spec or full-model binding: separate native framework
    extensions and PreparedDcpHybrid/PreparedDcpVerify handoffs are required.
    """

    if type(enable_prefetch) is not bool:
        raise ValueError("enable_prefetch must be a bool")
    if type(enable_embedding) is not bool:
        raise ValueError("enable_embedding must be a bool")
    _validate_mla_core(mla_core)
    batches = prepared_widths(tokens_per_request, widths)
    max_seq_len = mla_max_seq_len(context_length, tokens_per_request)
    memory_before = dict(memory_before)
    dcp_owner = None
    if dcp is not None:
        from .dcp_resources import DcpResources, NativeDcpOwner, PreparedDcpMla

        dcp_owner = NativeDcpOwner.capture(
            dcp,
            device=next(model.parameters()).device,
            tokens_per_request=tokens_per_request,
            deadline_ns=deadline_ns,
            widths=batches,
            max_context=max_seq_len,
            mla_core=mla_core,
        )
    origins: dict[str, str] = {}
    projection_module = _module("gemm.projection", origins)
    mla_module = _module("mla.prepared", origins)
    kda_module = _module("kda.prepared", origins)
    attnres_module = _module("attn_res.prepared", origins)
    oproj_module = _module("oproj.prepared", origins)
    twoshot_module = _module("common.twoshot", origins)
    dense_module = _module("moe.dense", origins)
    experts_module = _module("moe.experts_runtime", origins)
    front_module = _module("moe.front", origins)
    prepared_moe_module = _module("moe.prepared", origins)
    chain_module = _module("moe.small_batch.chain", origins)
    tail_module = _module("moe.tail_runtime", origins)

    parallel = get_parallel()
    tp, rank = int(parallel.tp_size), int(parallel.tp_rank)
    if tp != 8:
        raise ValueError(f"Kimi K3 megakernel initial resources require TP8, got {tp}")
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

    embedding = None
    if enable_embedding:
        from .embedding import prepare_embedding

        embedding = prepare_embedding(
            backbone.embed_tokens,
            tp_group=parallel.tp_group,
            load_prepared=lambda: (
                _module("common.embedding", origins).PreparedEmbedding
            ),
        )

    mailbox = oproj_module.OprojMailbox(*owner)
    oproj_twoshot = twoshot_module.TwoShot(*owner)
    front_box = oproj_module.OprojMailbox(*owner, (1, 64, 3584 + 2 * 896))
    small_box = chain_module.SmallBatchBox(*owner)
    tail_box = tail_module.MoeTailBox(*owner)
    boxes = {
        "oproj": mailbox,
        "oproj_twoshot": oproj_twoshot,
        "moe_front": front_box,
        "small_batch": small_box,
        "moe_tail": tail_box,
    }
    dcp_resources = None
    if dcp_owner is not None:
        dcp_resources = DcpResources(
            dcp_owner,
            dcp,
            _module("mla.dcp", origins),
            widths=batches,
            max_context=max_seq_len,
            deadline_ns=deadline_ns,
            tokens_per_request=tokens_per_request,
        )
        boxes["mla.dcp"] = dcp_resources.box
    moe_state = prepared_moe_module.MoEScratch(tp, device, batches)
    # Materialize the state-owned event handles before resource inventory.  The
    # prepared MoE records them again at each execution boundary, so completing
    # an initialization record on its own stream does not alter runtime order.
    for event in (moe_state.begin, moe_state.shared_done, moe_state.gate_done):
        event.record(moe_state.stream)
    moe_state.stream.synchronize()
    scratch = {
        # PreparedMla.prepare_cache performs one bounded warm call.  Zero is a
        # finite FP8 encoding, unlike arbitrary uninitialized byte patterns.
        "mla.query": torch.zeros((64, 12, 576), dtype=torch.uint8, device=device),
        "mla.projected": torch.empty(
            (_MAX_BATCH, 3648), dtype=torch.bfloat16, device=device
        ),
        "kda.projected": torch.empty(
            (_MAX_BATCH, 6288), dtype=torch.bfloat16, device=device
        ),
        "attention.local": torch.empty(
            (_MAX_BATCH, 1536), dtype=torch.bfloat16, device=device
        ),
        "hidden.0": torch.empty(
            (_MAX_BATCH, 7168), dtype=torch.bfloat16, device=device
        ),
        "hidden.1": torch.empty(
            (_MAX_BATCH, 7168), dtype=torch.bfloat16, device=device
        ),
        "hidden.out": torch.empty(
            (_MAX_BATCH, 7168), dtype=torch.bfloat16, device=device
        ),
        "prefix": torch.empty((_MAX_BATCH, 7168), dtype=torch.bfloat16, device=device),
        "blocks": torch.empty(
            (_MAX_BATCH, 8, 7168), dtype=torch.bfloat16, device=device
        ),
        "moe": moe_state,
    }
    if embedding is not None:
        scratch["embedding.out"] = torch.empty(
            (_MAX_BATCH, 7168), dtype=torch.bfloat16, device=device
        )
    if tokens_per_request > 1:
        # Native graph padding uses physical KDA slots <= 0, but MLA requires
        # zero lengths and negative write slots. Own the translated metadata
        # before inventory/capture; never modify the framework's sealed views.
        scratch.update(
            {
                "mla.verify_inactive": torch.empty(
                    (_MAX_BATCH // tokens_per_request,), dtype=torch.bool, device=device
                ),
                "mla.verify_lengths": torch.empty(
                    (_MAX_BATCH // tokens_per_request,),
                    dtype=torch.int32,
                    device=device,
                ),
                "mla.verify_slots": torch.empty(
                    (_MAX_BATCH,), dtype=torch.int64, device=device
                ),
            }
        )
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
    attention: dict[int, PreparedAttention] = {}
    feed_forward: dict[int, Any] = {}
    moe_front_weights: dict[int, torch.Tensor] = {}
    compilation: list[dict[str, Any]] = []
    if embedding is not None:
        compilation.append({"kind": "embedding", "name": "prepared"})

    for layer in layers:
        layer_id = int(layer.layer_idx)
        module = layer.self_attn
        is_mla = hasattr(module, "q_b_proj")
        if is_mla:
            proxy, _owned = _mla_proxy(module)
            projection_weight = coalesce_mla_projection_storage(module)
            if tuple(projection_weight.shape) != (3648, 7168):
                raise ValueError(
                    f"layer {layer_id} MLA projection shape changed: "
                    f"{tuple(projection_weight.shape)}"
                )
            projection = projection_module.PreparedProjection(
                projection_weight,
                _projection_policies(3648, batches),
            ).prepare()
            core = mla_module.PreparedMla(
                proxy,
                {batch: 3648 for batch in batches},
                scratch["mla.query"],
                max_context=max_seq_len,
                core=mla_core,
                dcp=None if dcp_resources is None else dcp_resources.box,
                queries=(tokens_per_request,),
                **(
                    {}
                    if dcp_resources is None
                    else {
                        "parity": full_ids.index(layer_id) % 2,
                        "deadline_ns": deadline_ns,
                    }
                ),
            )
            if dcp_resources is not None:
                core = PreparedDcpMla(core, dcp_resources)
            elif mla_core == "tokenspeed":
                core = _PreparedTokenSpeedMla(core, _module("mla.dcp", origins))
            elif not workspaces:
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
            projection_weight = coalesce_kda_projection_storage(module)
            if tuple(projection_weight.shape) != (6288, 7168):
                raise ValueError(
                    f"layer {layer_id} KDA projection shape changed: "
                    f"{tuple(projection_weight.shape)}"
                )
            projection = projection_module.PreparedProjection(
                projection_weight,
                _projection_policies(6288, batches),
            ).prepare()
            core = kda_module.PreparedKda(
                proxy,
                _kda_policy(tokens_per_request, batches),
                False,
            )
            kind = "kda"

        front = None
        if layer._is_moe_layer:
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
        experts = experts_module.PersistentExperts(
            routed.w13_weight,
            routed.w13_weight_scale,
            routed.w2_weight,
            routed.w2_weight_scale,
            beta=beta,
            linear_beta=linear_beta,
        )
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
        generic_front = front_module.SharedFront(
            moe,
            packed_front,
            front_box,
            moe_state.front,
            batches,
            beta,
            linear_beta,
            plan=moe_state.plan,
            counters=moe_state.shared,
            q=tokens_per_request,
            deadline_ns=deadline_ns,
        )
        chain = None
        if 1 in batches:
            chain = chain_module.SmallBatchChain(
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
        prepared_moe = prepared_moe_module.PreparedMoE(
            generic_front,
            experts,
            tail,
            moe_state,
            prefetch=None,
            layer_idx=layer_id,
        )
        prepared_moe.small_batch = chain
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
                *(
                    ({"kind": "moe_small_batch", "layer_id": layer_id, "name": "b1"},)
                    if chain is not None
                    else ()
                ),
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
        consumer.bind_ring(tail_box.out.buffer)
        fused_before[int(layer.layer_idx)] = consumer

    prefetch = None
    if enable_prefetch:
        from .weight_prefetch import PrefetchController

        _module("common.prefetch", origins)
        # Final coalesced attention and packed-front storage must precede range
        # preparation. One retained controller owns all graph stream/event handles.
        prefetch = PrefetchController(
            attention, layers, moe_front_weights, device=device, batches=batches
        )
        for layer in moe_layers:
            prepared = feed_forward[int(layer.layer_idx)]
            prepared.prefetch = prefetch
            if prepared.small_batch is not None:
                prepared.small_batch.prefetch = prefetch
        compilation.append({"kind": "weight_prefetch", "name": "tp8"})

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
        dcp=dcp_resources,
        prefetch=prefetch,
        embedding=embedding,
        tokens_per_request=tokens_per_request,
        widths=batches,
    )
    resources.refresh_inventory()
    return resources

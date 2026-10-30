# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""TP8 q1 decode and fixed DSPARK q4-8 target-verify executor for Kimi K3."""

from __future__ import annotations

from functools import partial
from typing import Any

import torch

try:
    from sglang.srt.kernel_plugins.nvidia.binding import HybridStepView
except ModuleNotFoundError as error:  # SGLang before sgl-project/sglang#45389
    if error.name not in (
        "sglang.srt.kernel_plugins",
        "sglang.srt.kernel_plugins.nvidia",
        "sglang.srt.kernel_plugins.nvidia.binding",
    ):
        raise
    from sglang.srt.megakernels.binding import HybridStepView

from .shape_policy import VERIFY_QUERIES, token_widths

# Names read from SGLang internals. A contract test checks these before GPU use.
SGLANG_READS = {
    "srt/kernel_plugins/nvidia/binding.py|srt/megakernels/binding.py": (
        "HybridStepView",
        "aux_hidden_states",
        "block_table",
        "kda",
        "mla",
        "sequence_lengths",
        "state_indices",
        "tokens_per_request",
        "track_decode_state",
        "verify_scratch_indices",
        "write_slots",
    ),
}


def _require_tensor(
    name: str,
    value: torch.Tensor,
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    device: torch.device,
    strides: tuple[int, ...] | None = None,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"Kimi K3 megakernel {name} is not a tensor")
    if value.dtype != dtype:
        raise RuntimeError(
            f"Kimi K3 megakernel {name} has dtype {value.dtype}, expected {dtype}"
        )
    if tuple(value.shape) != shape:
        raise RuntimeError(
            f"Kimi K3 megakernel {name} has shape {tuple(value.shape)}, "
            f"expected {shape}"
        )
    if value.device != device:
        raise RuntimeError(
            f"Kimi K3 megakernel {name} is on {value.device}, expected {device}"
        )
    if strides is not None and tuple(value.stride()) != strides:
        raise RuntimeError(
            f"Kimi K3 megakernel {name} has stride {tuple(value.stride())}, "
            f"expected {strides}"
        )


def _validate_step_inputs(
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor | None,
    view: HybridStepView,
    *,
    batch: int,
    device: torch.device,
    tokens_per_request: int = 1,
    aux_width: int = 0,
) -> None:
    if device.type != "cuda":
        raise RuntimeError(
            f"Kimi K3 megakernel execution requires CUDA resources, found {device}"
        )
    if tokens_per_request not in (1, *VERIFY_QUERIES) or batch not in token_widths(
        tokens_per_request
    ):
        raise RuntimeError("Kimi K3 megakernel received an unsupported query width")
    if view.mla.tokens_per_request != tokens_per_request:
        raise RuntimeError(
            "Kimi K3 megakernel MLA query width differs from preparation"
        )
    requests = batch // tokens_per_request
    _require_tensor(
        "input_ids",
        input_ids,
        dtype=torch.int64,
        shape=(batch,),
        device=device,
        strides=(1,),
    )
    if inputs_embeds is not None:
        _require_tensor(
            "inputs_embeds",
            inputs_embeds,
            dtype=torch.bfloat16,
            shape=(batch, 7168),
            device=device,
            strides=(7168, 1),
        )
    _require_tensor(
        "MLA write slots",
        view.mla.write_slots,
        dtype=torch.int64,
        shape=(batch,),
        device=device,
        strides=(1,),
    )
    _require_tensor(
        "MLA sequence lengths",
        view.mla.sequence_lengths,
        dtype=torch.int32,
        shape=(requests,),
        device=device,
        strides=(1,),
    )
    _require_tensor(
        "KDA state indices",
        view.kda.state_indices,
        dtype=torch.int32,
        shape=(requests,),
        device=device,
        strides=(1,),
    )
    table = view.mla.block_table
    if not isinstance(table, torch.Tensor):
        raise RuntimeError("Kimi K3 megakernel MLA block table is not a tensor")
    if table.dtype != torch.int32 or table.device != device:
        raise RuntimeError(
            "Kimi K3 megakernel MLA block table must be int32 on "
            f"{device}, found {table.dtype} on {table.device}"
        )
    if table.dim() != 2 or int(table.shape[0]) != requests or int(table.shape[1]) <= 0:
        raise RuntimeError(
            "Kimi K3 megakernel MLA block table must be a nonempty "
            f"[{requests}, pages] tensor, found {tuple(table.shape)}"
        )
    rows, columns = int(table.stride(0)), int(table.stride(1))
    if columns != 1 or rows < int(table.shape[1]):
        raise RuntimeError(
            "Kimi K3 megakernel MLA block table has unsupported stride "
            f"{tuple(table.stride())}"
        )
    if tokens_per_request > 1:
        _require_tensor(
            "KDA verify scratch indices",
            view.kda.verify_scratch_indices,
            dtype=torch.int32,
            shape=(requests,),
            device=device,
            strides=(1,),
        )
        if aux_width <= 0:
            raise RuntimeError("Kimi K3 DSPARK auxiliary taps were not prepared")
        _require_tensor(
            "DSPARK auxiliary hidden states",
            view.aux_hidden_states,
            dtype=torch.bfloat16,
            shape=(batch, aux_width),
            device=device,
            strides=(aux_width, 1),
        )
    elif (
        view.kda.verify_scratch_indices is not None
        or view.aux_hidden_states is not None
    ):
        raise RuntimeError("Kimi K3 decode received verify-only runtime buffers")


def _verify_mla_metadata(
    state_indices: torch.Tensor,
    sequence_lengths: torch.Tensor,
    write_slots: torch.Tensor,
    scratch: dict[str, Any],
    tokens_per_request: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Translate native verify padding into MLA's skip-write/zero-output contract.

    Physical KDA slots <= 0 are inactive, including startup's all-zero slots.
    Verify scratch index zero is live and is deliberately not consulted. Every
    operation writes preallocated storage and is replayed inside the CUDA graph.
    """
    requests = int(state_indices.shape[0])
    inactive = scratch["mla.verify_inactive"][:requests]
    lengths = scratch["mla.verify_lengths"][:requests]
    slots = scratch["mla.verify_slots"][: requests * tokens_per_request]
    torch.le(state_indices, 0, out=inactive)
    lengths.copy_(sequence_lengths)
    lengths.masked_fill_(inactive, 0)
    slots.copy_(write_slots)
    slots.view(requests, tokens_per_request).masked_fill_(
        inactive.view(requests, 1), -1
    )
    return slots, lengths


class KimiK3MegakernelExecutor:
    """Run the prepared 93-layer chain against SGLang-owned runtime state."""

    def __init__(self, resources: Any):
        self.resources = resources

    def run(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
        view: HybridStepView,
    ) -> torch.Tensor:
        resources = self.resources
        if resources.binding_fingerprint is None:
            raise RuntimeError(
                "Kimi K3 megakernel pools were not bound before graph warmup"
            )
        mla_slots = view.mla.write_slots
        mla_lengths = view.mla.sequence_lengths
        kda_slots = view.kda.state_indices
        batch = int(input_ids.shape[0])
        q = resources.tokens_per_request
        widths = token_widths(q)
        if batch not in widths:
            raise RuntimeError(f"unsupported Kimi K3 megakernel batch: {batch}")
        device = resources.scratch["hidden.0"].device
        _validate_step_inputs(
            input_ids,
            inputs_embeds,
            view,
            batch=batch,
            device=device,
            tokens_per_request=q,
            aux_width=7168 * len(resources.aux_capture_layers or ()),
        )
        if q > 1:
            mla_slots, mla_lengths = _verify_mla_metadata(
                kda_slots, mla_lengths, mla_slots, resources.scratch, q
            )
        for layer_id, attention in resources.attention.items():
            if attention.kind == "mla" and attention.cache is None:
                raise RuntimeError(f"MLA cache for layer {layer_id} is not bound")
            if attention.kind == "kda" and attention.caches is None:
                raise RuntimeError(f"KDA caches for layer {layer_id} are not bound")
            if attention.kind == "kda" and q > 1 and attention.intermediates is None:
                raise RuntimeError(
                    f"KDA verify scratch for layer {layer_id} is not bound"
                )
        auxiliary = (
            {
                layer_id: view.aux_hidden_states[:, index * 7168 : (index + 1) * 7168]
                for index, layer_id in enumerate(resources.aux_capture_layers)
            }
            if q > 1
            else {}
        )
        x = (
            inputs_embeds
            if inputs_embeds is not None
            else resources.model.model.embed_tokens(input_ids)
        )
        if tuple(x.shape) != (batch, 7168):
            raise RuntimeError(f"unexpected Kimi K3 megakernel input shape: {x.shape}")

        prefix = resources.scratch["prefix"][:batch]
        blocks = resources.scratch["blocks"][:batch]
        first = resources.scratch["hidden.0"][:batch]
        second = resources.scratch["hidden.1"][:batch]
        output = resources.scratch["hidden.out"][:batch]
        mla_projected = resources.scratch["mla.projected"][:batch]
        kda_projected = resources.scratch["kda.projected"][:batch]
        local = resources.scratch["attention.local"][:batch]

        hidden = x
        next_input_ready = False
        last_layer_id = -1
        for layer in resources.layers:
            layer_id = int(layer.layer_idx)
            last_layer_id = layer_id
            if not next_input_ready:
                kwargs = (
                    {"aux": auxiliary[layer_id - 1]}
                    if layer_id - 1 in auxiliary
                    else {}
                )
                resources.before[layer_id].run(hidden, prefix, blocks, first, **kwargs)

            attention = resources.attention[layer_id]
            projected = mla_projected if attention.kind == "mla" else kda_projected
            attention.projection.run(first, projected)
            if attention.kind == "mla":
                attention.core.run(
                    projected,
                    local,
                    cache=attention.cache,
                    slots=mla_slots,
                    block_table=view.mla.block_table,
                    lengths=mla_lengths,
                    q=q,
                )
            elif q > 1:
                # Verification only writes checkpoints. Stock DSpark exclusively
                # owns acceptance selection and the eventual persistent commit.
                attention.core.run_verify(
                    projected,
                    local,
                    caches=attention.caches,
                    slots=kda_slots,
                    intermediates=attention.intermediates,
                    scratch_indices=view.kda.verify_scratch_indices,
                )
            else:
                attention.core.run(
                    projected, local, caches=attention.caches, slots=kda_slots
                )
                view.kda.track_decode_state(layer_id)
            attention.output.run(local, prefix, blocks, second)

            consumer = resources.fused_before.get(layer_id)
            next_input_ready = consumer is not None
            feed_forward = resources.feed_forward[layer_id]
            if consumer is not None:
                target = output if consumer is resources.final else first
                kwargs = {"aux": auxiliary[layer_id]} if layer_id in auxiliary else {}
                consume = partial(consumer.run_ring, prefix, blocks, **kwargs)
                hidden = feed_forward.run(second, target, consume=consume)
            else:
                hidden = feed_forward.run(second, first)
                first, second = second, first

        if not next_input_ready:
            kwargs = (
                {"aux": auxiliary[last_layer_id]} if last_layer_id in auxiliary else {}
            )
            resources.final.run(hidden, prefix, blocks, output, **kwargs)
        if last_layer_id != 92:
            raise RuntimeError(f"Kimi K3 megakernel stopped at layer {last_layer_id}")
        return output

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/layers/attention/linear/gdn_replayssm_ucache.py (GdnUcacheVerifier),
# python/sglang/srt/layers/attention/linear/gdn_backend.py (GDNAttnBackend.forward_extend's
# u-cache target-verify branch, _ucache_fused_conv, _ucache_fused_norm,
# _gdn_ucache_target_verify and the u-cache warm-up of init_cuda_graph_state) and
# python/sglang/srt/models/qwen3_5.py (Qwen3_5GatedDeltaNet.forward with the strided
# q/k/v views and the fused gated norm).
"""The GDN layer's target verify on the u-cache: the deferred-fold ReplaySSM verify,
whose rings the executor allocates, resets and commits (``gdn_ucache_state.py``).

Per layer, ``verify`` runs: the input projection; its strided q/k/v/z/b/a views (no
split kernel); the causal conv update of the verify rows, either separately (the
source's retiled conv kernel, ``Options.conv_retile``; else the plugin's hoisted copy of
SGLang's conv kernel, or SGLang's ``causal_conv1d_update``) or inside the flush launch
(``Options.fused_conv``); the flush (``UcacheVerify.core``: one
launch that outputs every draft position from the checkpoint ``S0`` plus the ring
window, appends the T new (k, u, g) records, and folds the window into ``S0`` on rows
whose window holds ``flush_min`` records); the gated RMSNorm, inside the flush's
epilogue (``Options.fused_norm``), else SGLang's kernel on 16 rows and 8 warps a program
(``Options.verify_norm_retile``, where SGLang's own launch holds >= 2 rows a program)
or SGLang's norm; and ``out_proj``. The flush kernel is the
vendored SM107 build (``kernels/ops/attention/linear/gdn_ucache_rubin``) on SM107 and
FlashInfer's ``gated_delta_rule_mtp_ucache_flush_umma`` on SM100 / SM103.

What it reads and writes: SGLang's pools through the step view ``view.mamba`` (the
fp16 checkpoint ``temporal`` by state slot, into which flush rows fold in place), the
executor's rings ``k_ring / u_ring / g_ring`` (runtime/gdn_ucache_state.py) by state slot
or, under ``by_request_row``, by the request row (appends after the window), their
cursors ``hist_len / cache_base`` read-only (the commit after the verify advances them), and the conv state by state slot and
the verify's conv windows by row (``view.gdn.intermediate_indices``), both written by the
conv as SGLang's conv does. Linear draft chains only (SGLang refuses topk > 1).

Numerics: the source's. Its flush kernel files are byte-identical; the
flush options do not change a result; the fused conv is SGLang's conv bit for bit
(source claim); the fused norm reduces a 128-wide row as 16 threads x 8 elements, the
stock gated norm's order where it applies (B >= 3), not shown bitwise; the verify norm
launch keeps each 128-wide row on one warp's 16 lanes x 8 elements, SGLang's launch's
order, bit for bit. The retiled conv
and the hoisted conv are SGLang's conv kernel's values and arithmetic per element (every
x row loaded first; the retiled one also loads the weights, slot indices and prior conv
state before its PDL wait), bit for bit (source claim)."""

from __future__ import annotations

import dataclasses
import logging
from typing import Optional

import torch

from sgl_nvidia_mega.qwen3_8.runtime import bf16_gemm_pins

logger = logging.getLogger(__name__)

SGLANG_READS = {
    "srt/models/qwen4_exp.py": ("Qwen4ExpLinearDecoderLayer",),
    "srt/models/qwen3_5.py": (
        "linear_attn",
        "_forward_input_proj",
        "num_k_heads",
        "num_v_heads",
        "head_k_dim",
        "head_v_dim",
        "attn_tp_size",
        "norm",
        "out_proj",
        "attn",
    ),
    "srt/layers/radix_linear_attention.py": (
        "layer_id",
        "conv_weights",
        "bias",
        "activation",
        "A_log",
        "dt_bias",
        "q_dim",
        "k_dim",
        "v_dim",
        "num_q_heads",
        "num_k_heads",
        "num_v_heads",
        "head_q_dim",
        "head_k_dim",
        "head_v_dim",
    ),
    "kernels/ops/attention/fla/layernorm_gated.py": (
        "weight",
        "eps",
        "group_size",
        "norm_before_gate",
        "activation",
    ),
    "kernels/ops/attention/triton_gdn_fused_proj.py": (
        "qwen3_5_gdn_prefill_projection_views",
        "fused_qkv_split_gdn_prefill",
    ),
    "kernels/ops/mamba/causal_conv1d_triton.py": ("causal_conv1d_update",),
    "srt/layers/attention/linear/gdn_backend.py": ("MAX_FUSED_QKV_SPLIT_DIM",),
    "srt/kernel_plugins/nvidia/eagle_binding.py": (
        "gdn",
        "mamba",
        "state_indices",
        "intermediate_indices",
        "layer_ids",
        "conv",
        "temporal",
        "intermediate_conv_window",
    ),
}

K_DIM = V_DIM = 128
MAX_DRAFT_TOKENS = 8
# Value heads per key head the projection views' contiguous [q | k | v | z] / [b | a]
# layout covers (Qwen3_5GatedDeltaNet's fused contiguous unpack on CUDA)
VIEW_RATIOS = (1, 2, 3, 4)
SM107 = (10, 7)
SUPPORTED_CC = ((10, 0), (10, 3), SM107)


@dataclasses.dataclass(frozen=True)
class Options:
    """The source's u-cache environment knobs, as fields."""

    # SGLANG_OPT_GDN_UCACHE_STRIDED_QKV: the flush reads the packed post-conv q / k / v
    # rows and the projection's b / a columns in place (SM107)
    strided_qkv: bool = True
    # SGLANG_OPT_GDN_UCACHE_PDL: the flush is a programmatic dependent of the kernel
    # before it (SM107)
    pdl: bool = True
    # SGLANG_OPT_GDN_UCACHE_FLUSH_FIRST: the folding rows first in the launch (SM107)
    flush_first: bool = True
    # SGLANG_OPT_GDN_UCACHE_FUSED_CONV: the verify conv of a linear chain in the flush's
    # prologue, at batches with one CTA per (key head, row); needs strided_qkv and pdl
    fused_conv: bool = False
    # SGLANG_OPT_GDN_UCACHE_FUSED_NORM: the gated RMSNorm in the flush's epilogue, for
    # T <= 4 and 3 <= B <= fused_norm_max_batch; needs strided_qkv and pdl
    fused_norm: bool = True
    # SGLANG_GDN_UCACHE_FUSED_NORM_MAX_BATCH
    fused_norm_max_batch: int = 48
    # With conv_retile off: the separate conv on the hoisted copy of SGLang's kernel
    # (gdn_front/conv.py), else SGLang's causal_conv1d_update
    hoisted_conv: bool = True
    # SGLANG_ENABLE_GDN_VERIFY_CONV_RETILE: the separate conv on the source's retiled
    # kernel (gdn_front/conv.py, RETILE_DEFAULT) where it applies, else SGLang's conv
    # kernel on 128-channel tiles with 2 warps (the source's call); over hoisted_conv
    conv_retile: bool = True
    # SGLANG_ENABLE_GDN_VERIFY_NORM_RETILE: the separate gated norm on 16 rows x 8 warps
    # a program where SGLang's launch holds >= 2 rows a program (gdn_prefill_norm.py)
    verify_norm_retile: bool = True

    def flush_options(self):
        from sgl_nvidia_mega.qwen3_8.kernels.ops.attention.linear.gdn_ucache_verify import (
            GdnUcacheFlushOptions,
        )

        return GdnUcacheFlushOptions(
            strided_qkv=self.strided_qkv, pdl=self.pdl, flush_first=self.flush_first
        )


# The high-concurrency tuning: every u-cache option
# but the fused conv
HIGH_CONCURRENCY = Options()
# The same with the fused conv: all of them
HIGH_CONCURRENCY_FUSED_CONV = Options(fused_conv=True)
# SM100 / SM103: FlashInfer's flush takes none of the options
PLAIN = Options(
    strided_qkv=False, pdl=False, flush_first=False, fused_conv=False, fused_norm=False
)
DEFAULT = HIGH_CONCURRENCY


def _kernels():
    from sgl_nvidia_mega.qwen3_8.kernels.ops.attention.linear import gdn_ucache_verify

    return gdn_ucache_verify


def _rubin():
    from sgl_nvidia_mega.qwen3_8.kernels.ops.attention.linear.gdn_ucache_rubin import (
        gdn_replay_mtp_umma_rubin,
    )

    return gdn_replay_mtp_umma_rubin


def default_options(device) -> Options:
    return DEFAULT if torch.cuda.get_device_capability(device) == SM107 else PLAIN


class UcacheVerify:
    """The per-layer flush launches of one target model (``GdnUcacheVerifier``).

    Per forward, the first GDN layer gathers the cursors, refreshes every layer's fp32
    gate parameters, builds the flush-first row order and zeroes the padded rows of the
    shared output buffer; each layer then costs the flush launch (plus a q/k/v split
    without ``strided_qkv``)."""

    def __init__(
        self,
        *,
        device: torch.device,
        first_layer_id: int,
        by_request_row: bool = False,
        options: Optional[Options] = None,
    ):
        K = _kernels()
        device = torch.device(device)
        capability = torch.cuda.get_device_capability(device)
        options = default_options(device) if options is None else options
        flush_options = options.flush_options()
        if flush_options != K.GdnUcacheFlushOptions() and capability != SM107:
            # FlashInfer's SM100 kernel takes none of them.
            raise ValueError(
                f"GDN u-cache flush options {flush_options} need the SM107 u-cache "
                f"kernel, got SM{capability[0]}{capability[1]}"
            )
        if by_request_row and capability != SM107:
            # Only the vendored SM107 kernel takes ring_indices.
            raise ValueError(
                "u-cache rings by request row need the SM107 u-cache kernel, "
                f"got SM{capability[0]}{capability[1]}"
            )
        self.options = options
        self.flush_options = flush_options
        self._flush_kernel = K._load_flush_kernel(device)
        self._kernel_kwargs = flush_options.kernel_kwargs()
        # The vendored SM107 kernel's conv prologue and norm epilogue, which its
        # strided, dependent launch carries
        self.fused_conv_capable = (
            options.strided_qkv and options.pdl and capability == SM107
        )
        self.fused_conv_enabled = options.fused_conv and self.fused_conv_capable
        self.fused_norm_enabled = options.fused_norm and self.fused_conv_capable
        for name, wanted, enabled in (
            ("fused_conv", options.fused_conv, self.fused_conv_enabled),
            ("fused_norm", options.fused_norm, self.fused_norm_enabled),
        ):
            if wanted and not enabled:
                logger.warning(
                    "GDN u-cache %s not taken: needs the SM107 flush with strided_qkv "
                    "and pdl",
                    name,
                )
        self._by_request_row = by_request_row
        self._device = device
        self.first_layer_id = first_layer_id
        # layer_id -> fp32 (A_log, dt_bias) passed to the flush, allocated eagerly.
        self._gate_params: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        # (bf16 source, fp32 destination) pairs refreshed by one launch per forward.
        self._gate_refresh: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._gate_refresh_tables: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        self._rows = None
        # This verify's flush-first row order, shared by all layers.
        self._row_order: Optional[torch.Tensor] = None
        self._strided_logged = False
        self._fused_conv_logged: set = set()
        self._fused_norm_logged = False
        # Shared by all layers (a layer's output is consumed by its gated norm and
        # out_proj before the next GDN layer's flush overwrites it); one per row shape,
        # since adaptive draft widths share this executor.
        self._outputs: dict[tuple, torch.Tensor] = {}
        self._qkv: dict[tuple, tuple[torch.Tensor, ...]] = {}
        # Outgrown shared buffers stay alive: captured graphs hold their pointers.
        self._retired_buffers: list = []
        logger.info(
            "GDN u-cache verify: %s.%s, first GDN layer %d, rings by %s, %s",
            self._flush_kernel.__module__,
            self._flush_kernel.__name__,
            first_layer_id,
            "request row" if by_request_row else "state slot",
            options,
        )

    def prepare(
        self,
        *,
        cache_indices: torch.Tensor,
        buffers,
        request_rows: Optional[torch.Tensor],
        output: torch.Tensor,
    ) -> None:
        """The forward's prep, at its first GDN layer: the cursors of its rows (slot <= 0
        is padding, -1; the ring rows under ``by_request_row``), the fp32 gate
        parameters, the flush-first row order and the padded output rows zeroed."""
        K = _kernels()
        assert buffers.by_request_row == self._by_request_row, (
            "the rings' indexing changed since the UcacheVerify was built"
        )
        batch = output.shape[0]
        self._rows = K.gather_gdn_ucache_rows(
            slots=cache_indices[:batch],
            buffers=buffers,
            request_rows=None if request_rows is None else request_rows[:batch],
        )
        self._refresh_gate_params()
        if self.flush_options.flush_first:
            self._row_order = K._flush_first_order(
                hist_len=self._rows.hist_len, flush_min=buffers.flush_min
            )
        # Last before the flush: under pdl its warps 0/1 read the row order and cursors
        # before griddepcontrol.wait, so the kernel just before it may write only what
        # they never read (padded output rows).
        self._zero_padded_rows(output)

    def core(
        self,
        layer_id: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        state: torch.Tensor,
        k_ring: torch.Tensor,
        u_ring: torch.Tensor,
        g_ring: torch.Tensor,
        buffers,
        cache_indices: torch.Tensor,
        num_draft_tokens: int,
        request_rows: Optional[torch.Tensor] = None,
        fused_conv=None,
        fused_norm=None,
    ) -> torch.Tensor:
        """``GdnUcacheVerifier.target_verify``: the verify output in ``value``'s shape;
        padded rows are zero. ``request_rows`` (the batch's request-table rows) index
        the rings under ``by_request_row``. ``fused_conv`` (a GdnFusedConv): the launch
        computes query / key / value, views of one [B, T, D] conv output, itself.
        ``fused_norm`` (a GdnFusedNorm): the output is the gated RMSNorm of it."""
        num_v_heads, head_v_dim = value.shape[-2], value.shape[-1]
        num_k_heads, head_k_dim = key.shape[-2], key.shape[-1]
        batch = value.numel() // (num_draft_tokens * num_v_heads * head_v_dim)
        token_shape = (batch, num_draft_tokens)
        output = self._output_buffer(
            batch=batch, row_shape=(num_draft_tokens, num_v_heads, head_v_dim)
        )
        if layer_id == self.first_layer_id:
            self.prepare(
                cache_indices=cache_indices,
                buffers=buffers,
                request_rows=request_rows,
                output=output,
            )
        rows = self._rows
        assert rows is not None and rows.slots.shape[0] == batch
        A_log_fp32, dt_bias_fp32 = self._fp32_gate_params(layer_id, A_log, dt_bias)
        if self.flush_options.strided_qkv:
            # The kernel reads the views in place (or copies them itself if it cannot).
            q, k, v = query, key, value
        else:
            q, k, v = self._contiguous_qkv(query, key, value)
        q = q.reshape(*token_shape, num_k_heads, head_k_dim)
        k = k.reshape(*token_shape, num_k_heads, head_k_dim)
        v = v.reshape(*token_shape, num_v_heads, head_v_dim)
        a = a.reshape(*token_shape, num_v_heads)
        b = b.reshape(*token_shape, num_v_heads)
        if self.flush_options.strided_qkv and not self._strided_logged:
            self._strided_logged = True
            _kernels()._log_strided_inputs(q=q, k=k, v=v, a=a, b=b)
        self._flush_kernel(
            A_log=A_log_fp32,
            a=a,
            dt_bias=dt_bias_fp32,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=state,
            initial_state_indices=rows.slots,
            scale=head_k_dim**-0.5,
            output=output,
            k_cache=k_ring,
            u_cache=u_ring,
            g_cache=g_ring,
            hist_len=rows.hist_len,
            cache_base=rows.cache_base,
            flush_min=buffers.flush_min,
            restart_hist_on_flush=False,
            **self._kernel_kwargs,
            **(
                {"row_order": self._row_order} if self.flush_options.flush_first else {}
            ),
            **({"ring_indices": rows.ring_rows} if self._by_request_row else {}),
            **({"fused_conv": fused_conv} if fused_conv is not None else {}),
            **({"fused_norm": fused_norm} if fused_norm is not None else {}),
        )
        return output.reshape(value.shape)

    def fused_conv(
        self,
        *,
        attn,
        mixed_qkv: torch.Tensor,
        conv_state: torch.Tensor,
        window: torch.Tensor,
        window_indices: torch.Tensor,
        batch_size: int,
        draft_token_num: int,
    ):
        """The GdnFusedConv for this verify's flush launches, or None (the separate conv
        runs): launches whose heads per CTA split a key head (small batches), a conv
        bias or a non-SiLU activation keep it. ``mixed_qkv``: the rows before the conv."""
        if not self.fused_conv_enabled:
            return None
        R = _rubin()
        fused = R.GdnFusedConv(
            x=mixed_qkv.view(batch_size, draft_token_num, -1),
            weight=attn.conv_weights,
            conv_state=conv_state,
            window=window,
            window_indices=window_indices,
        )
        reason = (
            f"conv bias / activation {attn.bias is not None} / {attn.activation}"
            if attn.bias is not None or attn.activation not in ("silu", "swish")
            else R.fused_conv_unsupported_reason(
                fused, num_k_heads=attn.num_k_heads, num_v_heads=attn.num_v_heads
            )
        )
        key = "kept" if reason is not None else "engaged"
        if key not in self._fused_conv_logged:
            self._fused_conv_logged.add(key)
            if reason is None:
                logger.info(
                    "GDN u-cache fused conv engaged: the flush runs the verify conv "
                    "(%d channels, %d tokens; first batch %d)",
                    mixed_qkv.shape[1],
                    draft_token_num,
                    batch_size,
                )
            else:
                logger.info(
                    "GDN u-cache fused conv: separate verify conv kept (%s)", reason
                )
        return fused if reason is None else None

    def fused_norm(
        self,
        *,
        norm,
        z: torch.Tensor,
        num_tokens: int,
        num_draft_tokens: int,
        num_v_heads: int,
    ):
        """The GdnFusedNorm for this verify's flush, or None (the model's gated norm
        runs): ``norm`` the layer's RMSNormGated (no groups, no bias, norm before a
        sigmoid gate); T <= 4 and verifies whose separate norm reduces a row as 16
        threads x 8 elements (B >= 3), up to ``fused_norm_max_batch``."""
        if (
            not self.fused_norm_enabled
            or num_draft_tokens > 4
            or num_tokens > num_draft_tokens * self.options.fused_norm_max_batch
            or norm.group_size is not None
            or norm.bias is not None
            or not norm.norm_before_gate
            or norm.activation != "sigmoid"
        ):
            return None
        if not _kernels().gdn_ucache_fused_norm_applies(
            num_tokens * num_v_heads, z.device
        ):
            return None
        R = _rubin()
        fused = R.GdnFusedNorm(z=z, weight=norm.weight, eps=float(norm.eps))
        reason = R.fused_norm_unsupported_reason(
            fused,
            batch=num_tokens // num_draft_tokens,
            T=num_draft_tokens,
            HV=num_v_heads,
        )
        if not self._fused_norm_logged:
            self._fused_norm_logged = True
            logger.info(
                "GDN u-cache fused norm %s (first verify %d tokens)",
                "engaged: the flush stores the gated RMSNorm of its output"
                if reason is None
                else f"not taken: {reason}",
                num_tokens,
            )
        return fused if reason is None else None

    def warm_up(
        self,
        *,
        max_batch_size: int,
        num_draft_tokens: int,
        state: torch.Tensor,
        k_ring: torch.Tensor,
        u_ring: torch.Tensor,
        g_ring: torch.Tensor,
        flush_min: int,
        max_eager_batch_size: int = 0,
        fused_conv: Optional[bool] = None,
        fused_norm_max_batch: Optional[int] = None,
    ) -> None:
        """Compiles the flush builds, allocates the shared buffers and builds the
        kernel's per-batch-size helper tensors eagerly, so CUDA graph capture creates
        none of them; ``fused_conv`` / ``fused_norm_max_batch`` (default: the enabled
        options) add the fused builds. The shared buffers also cover
        ``max_eager_batch_size`` so a later eager batch does not outgrow them. ``state``
        and the rings: one layer's (``temporal[0]``, ``k_ring[0]``, ...)."""
        if fused_conv is None:
            fused_conv = self.fused_conv_enabled
        if fused_norm_max_batch is None:
            fused_norm_max_batch = (
                self.options.fused_norm_max_batch if self.fused_norm_enabled else 0
            )
        num_k_heads, head_k_dim = k_ring.shape[1], k_ring.shape[3]
        num_v_heads, head_v_dim = u_ring.shape[1], u_ring.shape[3]
        device = state.device
        tokens = (max_batch_size, num_draft_tokens)
        buffer_batch = max(max_batch_size, max_eager_batch_size)
        self._output_buffer(
            batch=buffer_batch, row_shape=(num_draft_tokens, num_v_heads, head_v_dim)
        )
        self._qkv_buffers(
            num_tokens=buffer_batch * num_draft_tokens,
            cols=(
                num_k_heads * head_k_dim,
                num_k_heads * head_k_dim,
                num_v_heads * head_v_dim,
            ),
        )

        def zeros(*shape, dtype=torch.bfloat16):
            return torch.zeros(shape, dtype=dtype, device=device)

        if self.flush_options.strided_qkv:
            # The served layout (the strides are compiled in): q / k / v of packed
            # [q | k | v] token rows, b / a of [b | a] rows.
            qk_cols, v_cols = num_k_heads * head_k_dim, num_v_heads * head_v_dim
            packed = zeros(*tokens, 2 * qk_cols + v_cols)
            q = packed[..., :qk_cols].unflatten(-1, (num_k_heads, head_k_dim))
            k = packed[..., qk_cols : 2 * qk_cols].unflatten(
                -1, (num_k_heads, head_k_dim)
            )
            v = packed[..., 2 * qk_cols :].unflatten(-1, (num_v_heads, head_v_dim))
            gates = zeros(*tokens, 2 * num_v_heads)
            b, a = gates[..., :num_v_heads], gates[..., num_v_heads:]
        else:
            q = k = zeros(*tokens, num_k_heads, head_k_dim)
            v = zeros(*tokens, num_v_heads, head_v_dim)
            a = b = zeros(*tokens, num_v_heads)
        gate_param = zeros(num_v_heads, dtype=torch.float32)
        cursor = zeros(max_batch_size, dtype=torch.int32)
        padded = torch.full((max_batch_size,), -1, dtype=torch.int32, device=device)
        order = torch.arange(max_batch_size, dtype=torch.int32, device=device)
        ring_rows = torch.zeros(max_batch_size, dtype=torch.int32, device=device)
        for batch in range(1, max_batch_size + 1):
            self._flush_kernel(
                A_log=gate_param,
                a=a[:batch],
                dt_bias=gate_param,
                q=q[:batch],
                k=k[:batch],
                v=v[:batch],
                b=b[:batch],
                initial_state_source=state,
                initial_state_indices=padded[:batch],
                k_cache=k_ring,
                u_cache=u_ring,
                g_cache=g_ring,
                hist_len=cursor[:batch],
                cache_base=cursor[:batch],
                flush_min=flush_min,
                restart_hist_on_flush=False,
                **self._kernel_kwargs,
                **(
                    {"row_order": order[:batch]}
                    if self.flush_options.flush_first
                    else {}
                ),
                **({"ring_indices": ring_rows[:batch]} if self._by_request_row else {}),
            )
        if (fused_conv or fused_norm_max_batch > 0) and self.fused_conv_capable:
            self._warm_up_fused(
                q=q,
                k=k,
                v=v,
                a=a,
                b=b,
                gate_param=gate_param,
                state=state,
                k_ring=k_ring,
                u_ring=u_ring,
                g_ring=g_ring,
                cursor=cursor,
                padded=padded,
                order=order,
                ring_rows=ring_rows,
                flush_min=flush_min,
                fused_conv=fused_conv,
                fused_norm_max_batch=fused_norm_max_batch,
            )

    def _warm_up_fused(
        self,
        *,
        q,
        k,
        v,
        a,
        b,
        gate_param,
        state,
        k_ring,
        u_ring,
        g_ring,
        cursor,
        padded,
        order,
        ring_rows,
        flush_min,
        fused_conv,
        fused_norm_max_batch,
    ) -> None:
        """The conv-prologue / gated-norm builds at every batch size that takes them.
        Every row is padded, so the launches touch no buffer."""
        R = _rubin()
        max_batch, T = q.shape[0], q.shape[1]
        H, HV = q.shape[2], v.shape[2]
        D = (2 * H + HV) * q.shape[3]
        device = q.device
        bf16 = torch.bfloat16
        x_all = torch.zeros(max_batch * T * D, dtype=bf16, device=device)
        win = torch.zeros(1, D, T + 2, dtype=bf16, device=device)
        win_view = win.as_strided((1, T, D, 3), ((T + 2) * D, 1, T + 2, 1))
        weight = torch.zeros(D, 4, dtype=bf16, device=device)
        conv_state = torch.zeros(1, D, 3, dtype=bf16, device=device)
        win_rows = torch.zeros(max_batch, dtype=torch.int32, device=device)
        z_all = torch.zeros(max_batch * T, HV, q.shape[3], dtype=bf16, device=device)
        norm_w = torch.zeros(q.shape[3], dtype=bf16, device=device)
        built = 0
        for batch in range(1, max_batch + 1):
            fc = R.GdnFusedConv(
                x=x_all[: batch * T * D].view(batch, T, D),
                weight=weight,
                conv_state=conv_state,
                window=win_view,
                window_indices=win_rows[:batch],
            )
            if (
                not fused_conv
                or R.fused_conv_unsupported_reason(fc, num_k_heads=H, num_v_heads=HV)
                is not None
            ):
                fc = None
            fn = (
                R.GdnFusedNorm(z=z_all[: batch * T], weight=norm_w, eps=1e-6)
                if batch <= fused_norm_max_batch
                and _kernels().gdn_ucache_fused_norm_applies(batch * T * HV, device)
                else None
            )
            if fc is None and fn is None:
                continue
            built += 1
            self._flush_kernel(
                A_log=gate_param,
                a=a[:batch],
                dt_bias=gate_param,
                q=q[:batch],
                k=k[:batch],
                v=v[:batch],
                b=b[:batch],
                initial_state_source=state,
                initial_state_indices=padded[:batch],
                k_cache=k_ring,
                u_cache=u_ring,
                g_cache=g_ring,
                hist_len=cursor[:batch],
                cache_base=cursor[:batch],
                flush_min=flush_min,
                restart_hist_on_flush=False,
                **({"fused_conv": fc} if fc is not None else {}),
                **({"fused_norm": fn} if fn is not None else {}),
                **self._kernel_kwargs,
                **(
                    {"row_order": order[:batch]}
                    if self.flush_options.flush_first
                    else {}
                ),
                **({"ring_indices": ring_rows[:batch]} if self._by_request_row else {}),
            )
        logger.info(
            "GDN u-cache fused conv / norm (%s / up to batch %d): flush builds warmed "
            "at %d of %d batch sizes",
            fused_conv,
            fused_norm_max_batch,
            built,
            max_batch,
        )

    def _output_buffer(self, *, batch: int, row_shape: tuple) -> torch.Tensor:
        out = self._outputs.get(row_shape)
        if out is None or out.shape[0] < batch:
            _kernels()._assert_not_capturing("the verify output buffer")
            if out is not None:
                self._retired_buffers.append(out)
            out = torch.zeros(
                (batch, *row_shape), dtype=torch.bfloat16, device=self._device
            )
            self._outputs[row_shape] = out
        return out[:batch]

    def _zero_padded_rows(self, output: torch.Tensor) -> None:
        # Valid rows are fully written by every layer; padded rows are skipped by
        # the kernel and must read as zero.
        import triton

        rows = self._rows
        batch = output.shape[0]
        if batch == 0:
            return
        row_numel = output[0].numel()
        block = 4096
        _kernels()._zero_padded_rows_kernel[(batch, triton.cdiv(row_numel, block))](
            rows.slots, output, row_numel, BLOCK=block
        )

    def _qkv_buffers(
        self, *, num_tokens: int, cols: tuple[int, int, int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bufs = self._qkv.get(cols)
        if bufs is None or bufs[0].shape[0] < num_tokens:
            _kernels()._assert_not_capturing("the q/k/v split buffers")
            if bufs is not None:
                self._retired_buffers.append(bufs)
            bufs = tuple(
                torch.empty((num_tokens, c), dtype=torch.bfloat16, device=self._device)
                for c in cols
            )
            self._qkv[cols] = bufs
        return tuple(buf[:num_tokens] for buf in bufs)

    def _contiguous_qkv(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if query.is_contiguous() and key.is_contiguous() and value.is_contiguous():
            return query, key, value
        q_cols, k_cols, v_cols = (
            x.shape[-2] * x.shape[-1] for x in (query, key, value)
        )
        num_tokens = query.numel() // q_cols
        row_stride = query.stride(-3)
        elem = query.element_size()
        packed = (
            key.data_ptr() == query.data_ptr() + q_cols * elem
            and value.data_ptr() == key.data_ptr() + k_cols * elem
            and key.stride(-3) == row_stride
            and value.stride(-3) == row_stride
            and all(
                x.stride(-1) == 1 and x.stride(-2) == x.shape[-1]
                for x in (query, key, value)
            )
        )
        if not packed:
            return query.contiguous(), key.contiguous(), value.contiguous()
        import triton

        q, k, v = self._qkv_buffers(
            num_tokens=num_tokens, cols=(q_cols, k_cols, v_cols)
        )
        block = 1024
        _kernels()._split_qkv_kernel[
            (num_tokens, triton.cdiv(q_cols + k_cols + v_cols, block))
        ](
            query,
            q,
            k,
            v,
            row_stride,
            Q_COLS=q_cols,
            K_COLS=k_cols,
            V_COLS=v_cols,
            BLOCK=block,
        )
        return q, k, v

    def _fp32_gate_params(
        self, layer_id: int, A_log: torch.Tensor, dt_bias: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        K = _kernels()
        params = self._gate_params.get(layer_id)
        if params is None:
            K._assert_not_capturing(
                "layer gate parameters (run an eager forward first)"
            )
            params = (K._fp32_view_or_buffer(A_log), K._fp32_view_or_buffer(dt_bias))
            self._gate_params[layer_id] = params
            for buf, src in zip(params, (A_log, dt_bias)):
                if buf.data_ptr() == src.data_ptr():
                    continue
                buf.copy_(src)
                if src.dtype == torch.bfloat16 and src.is_contiguous():
                    self._gate_refresh.append((src, buf))
                    self._gate_refresh_tables = None
        # Other non-aliased sources (not bf16 contiguous) are refreshed per call.
        for buf, src in zip(params, (A_log, dt_bias)):
            if buf.data_ptr() != src.data_ptr() and not (
                src.dtype == torch.bfloat16 and src.is_contiguous()
            ):
                buf.copy_(src)
        return params

    def _refresh_gate_params(self) -> None:
        # In-place weight updates reach graph replays: the refresh is captured.
        if not self._gate_refresh:
            return
        import triton

        K = _kernels()
        if self._gate_refresh_tables is None:
            K._assert_not_capturing("the gate refresh tables")
            self._gate_refresh_tables = tuple(
                torch.tensor(
                    [t.data_ptr() for t in group],
                    dtype=torch.int64,
                    device=self._device,
                )
                for group in zip(*self._gate_refresh)
            )
        numel = self._gate_refresh[0][0].numel()
        assert all(src.numel() == numel for src, _ in self._gate_refresh)
        src_ptrs, dst_ptrs = self._gate_refresh_tables
        K._refresh_gate_params_kernel[(len(self._gate_refresh),)](
            src_ptrs, dst_ptrs, numel, BLOCK=triton.next_power_of_2(numel)
        )


def layer_reasons(gdn) -> str:
    """'' when a GDN layer (``Qwen3_5GatedDeltaNet``) fits the u-cache verify."""
    attn = gdn.attn
    h, hv = attn.num_k_heads, attn.num_v_heads
    if attn.head_k_dim != K_DIM or attn.head_v_dim != V_DIM:
        return (
            f"GDN head dims {attn.head_k_dim} / {attn.head_v_dim}, the flush takes 128"
        )
    if hv % h or not 1 <= hv // h <= 4:
        return f"GDN heads per rank {(h, hv)}, the flush takes 1-4 value heads per key head"
    if gdn.num_v_heads // gdn.num_k_heads not in VIEW_RATIOS:
        return "the projection's contiguous q/k/v/z layout needs 1-4 value heads per key head"
    return ""


def build(model, device) -> str:
    """'' when the target ``model``'s GDN layers can run the u-cache verify on
    ``device`` and its flush kernel loads (the vendored SM107 build, or FlashInfer's on
    SM100 / SM103), else why not. The rings are checked at the first verify
    (``pool_reasons``)."""
    from sglang.srt.models.qwen4_exp import Qwen4ExpLinearDecoderLayer

    device = torch.device(device)
    capability = torch.cuda.get_device_capability(device)
    if capability not in SUPPORTED_CC:
        return f"the u-cache flush needs SM100, SM103 or SM107, got SM{capability[0]}{capability[1]}"
    m = model.model
    for i in range(m.start_layer, m.end_layer):
        layer = m.layers[i]
        if isinstance(layer, Qwen4ExpLinearDecoderLayer):
            reason = layer_reasons(layer.linear_attn)
            if reason:
                return reason
    try:
        _kernels()._load_flush_kernel(device)
    except Exception as e:
        return f"the u-cache flush kernel does not load: {type(e).__name__}: {e}"
    return ""


def pool_reasons(ucache_buffers, tokens_per_request: int, *, state=None) -> str:
    """'' when the u-cache rings (``UcacheState.buffers``, a GdnUcacheBuffers) fit the
    flush for ``tokens_per_request`` verify rows a request, else why not; ``state`` (one
    layer's checkpoint, optional): fp16."""
    buffers = ucache_buffers
    if buffers is None:
        return "no u-cache rings"
    T = int(tokens_per_request)
    if not 1 <= T <= MAX_DRAFT_TOKENS:
        return f"{T} verify rows a request, the flush takes 1-{MAX_DRAFT_TOKENS}"
    K = _kernels()
    if not 1 <= buffers.flush_min <= K._MAX_WINDOW - T + 1:
        return (
            f"fold threshold {buffers.flush_min} cannot append {T} records to a "
            f"{K._MAX_WINDOW}-record window"
        )
    k_ring, u_ring, g_ring = buffers.k_ring, buffers.u_ring, buffers.g_ring
    if (
        k_ring.dim() != 5
        or k_ring.shape[-2] != K.RING_SLOTS
        or k_ring.shape[-1] != K_DIM
        or u_ring.shape[-2:] != (K.RING_SLOTS, V_DIM)
        or g_ring.shape[-1] != K.RING_SLOTS
    ):
        return f"u-cache ring shapes {tuple(k_ring.shape)} / {tuple(u_ring.shape)}"
    if k_ring.dtype not in (torch.bfloat16, torch.float16):
        return f"k ring dtype {k_ring.dtype}"
    if (
        u_ring.dtype not in (torch.bfloat16, torch.float16)
        or g_ring.dtype != torch.float32
    ):
        return f"u / g ring dtypes {u_ring.dtype} / {g_ring.dtype}"
    if buffers.hist_len.dtype != torch.int32 or buffers.cache_base.dtype != torch.int32:
        return "u-cache cursors are not int32"
    if state is not None and state.dtype != torch.float16:
        return f"GDN checkpoint dtype {state.dtype}, the flush takes fp16"
    capability = torch.cuda.get_device_capability(buffers.hist_len.device)
    if capability != SM107:
        if k_ring.dtype == torch.float16:
            return "an fp16 k ring needs the SM107 u-cache kernel"
        if buffers.by_request_row:
            return "u-cache rings by request row need the SM107 u-cache kernel"
    return ""


def verify(executor, gdn, x, st, view, buffers=None):
    """``Qwen3_5GatedDeltaNet.forward`` for a target verify on the u-cache rings
    (``buffers``, a GdnUcacheBuffers: by default the executor's own,
    ``executor.ucache_state.buffers``);
    ``executor.gdn_ucache`` is the executor's UcacheVerify. Returns the layer's mixer
    output [rows, hidden]."""
    import triton
    from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
        qwen3_5_gdn_prefill_projection_views,
    )

    from sgl_nvidia_mega.qwen3_8.runtime import smallm_gemm

    uv = executor.gdn_ucache
    if uv is None:
        raise RuntimeError(
            "GDN u-cache verify: the executor has no UcacheVerify (gdn_ucache); SGLang "
            "runs no u-cache verify of its own"
        )
    fb = st.fb
    attn = gdn.attn
    meta, pools = view.gdn, view.mamba
    if buffers is None:
        buffers = executor.ucache_state.buffers
    li = pools.layer_ids.index(attn.layer_id)
    tokens = int(fb.spec_info.draft_token_num)
    qkvz, ba = bf16_gemm_pins.in_proj(gdn, x)
    nk = triton.cdiv(gdn.num_k_heads, gdn.attn_tp_size)
    nv = triton.cdiv(gdn.num_v_heads, gdn.attn_tp_size)
    # The strided views: no split kernel; the conv reads q|k|v in place.
    mixed_qkv, z, b, a = qwen3_5_gdn_prefill_projection_views(
        qkvz, ba, nk, nv, gdn.head_k_dim, gdn.head_v_dim
    )
    rows = mixed_qkv.shape[0]
    batch = rows // tokens
    conv_state = pools.conv[0][li]
    window = pools.intermediate_conv_window[0][li]
    slots = meta.state_indices
    window_rows = meta.intermediate_indices[:batch]
    fused_conv = uv.fused_conv(
        attn=attn,
        mixed_qkv=mixed_qkv,
        conv_state=conv_state,
        window=window,
        window_indices=window_rows,
        batch_size=batch,
        draft_token_num=tokens,
    )
    if fused_conv is not None:
        # the flush computes the conv into these rows (GdnFusedConv)
        convolved = torch.empty(
            (rows, mixed_qkv.shape[1]), dtype=mixed_qkv.dtype, device=mixed_qkv.device
        )
    else:
        convolved = _conv(
            attn,
            mixed_qkv,
            conv_state,
            window,
            slots[:batch],
            window_rows,
            batch,
            tokens,
            uv.options.hoisted_conv,
            retile=uv.options.conv_retile,
        )
    query, key, value = _split_qkv(attn, convolved, uv.flush_options.strided_qkv)
    fused_norm = uv.fused_norm(
        norm=gdn.norm,
        z=z,
        num_tokens=rows,
        num_draft_tokens=tokens,
        num_v_heads=attn.num_v_heads,
    )
    core = uv.core(
        layer_id=attn.layer_id,
        query=query,
        key=key,
        value=value,
        a=a,
        b=b,
        A_log=attn.A_log,
        dt_bias=attn.dt_bias,
        # GDN has K == V; the pool's [slots, HV, V, K] layout is the kernel's.
        state=pools.temporal[li],
        k_ring=buffers.k_ring[li],
        u_ring=buffers.u_ring[li],
        g_ring=buffers.g_ring[li],
        buffers=buffers,
        cache_indices=slots,
        num_draft_tokens=tokens,
        request_rows=fb.req_pool_indices if buffers.by_request_row else None,
        fused_conv=fused_conv,
        fused_norm=fused_norm,
    )
    st.ran(
        "gdn",
        "ucache"
        + ("+conv" if fused_conv is not None else "")
        + ("+norm" if fused_norm is not None else ""),
    )
    if fused_norm is not None:
        # [rows, HV, V], gated and normalized
        core = core.reshape(x.shape[0], -1)
        out = smallm_gemm.out_proj(executor, gdn.out_proj, core, st, "gdn.out_proj")
        return bf16_gemm_pins.linear(gdn.out_proj, core) if out is None else out
    z_shape = z.shape
    core = core.reshape(-1, core.shape[-1])
    z_flat_shape = (z.numel() // z.shape[-1], z.shape[-1])
    if core.shape != z_flat_shape:  # DP-attention padding
        padded = z.new_zeros(z_flat_shape)
        padded[: core.shape[0], :] = core
        core = padded
    core = _gated_norm(uv.options, gdn.norm, core, z)
    core = core.reshape(z_shape)
    core = core.reshape(*core.shape[:-2], core.shape[-2] * core.shape[-1])
    out = smallm_gemm.out_proj(executor, gdn.out_proj, core, st, "gdn.out_proj")
    if out is None:
        out = bf16_gemm_pins.linear(gdn.out_proj, core)
    return out


def _gated_norm(options, norm, core, z):
    """``norm(core, z)``: the layer's gated RMSNorm of the flush output [rows * HV, V]
    with the strided gate view z [rows, HV, V]; on the verify launch where it applies."""
    from sgl_nvidia_mega.qwen3_8.kernels.ops.attention.linear import (
        gdn_prefill_norm,
    )

    if (
        options.verify_norm_retile
        and norm.group_size is None
        and norm.bias is None
        and core.stride(-1) == 1
        and gdn_prefill_norm.gdn_verify_norm_retile_applies(core)
    ):
        return gdn_prefill_norm.gdn_verify_norm_retiled(
            x=core,
            weight=norm.weight,
            z=z,
            eps=norm.eps,
            norm_before_gate=norm.norm_before_gate,
            activation=norm.activation,
        )
    return norm(core, z)


def _conv(
    attn,
    mixed_qkv,
    conv_state,
    window,
    slots,
    window_rows,
    batch,
    tokens,
    hoisted,
    retile=False,
):
    """The causal conv update of the verify rows (linear chain): [rows, channels],
    token-major. With ``retile``, the source's call: its retiled kernel where it applies
    (width 4, 4-8 rows a request, one dtype, a 4-D window), else SGLang's conv kernel on
    128-channel tiles. Else SGLang's conv kernel, or its hoisted copy where it applies
    (width 4, 4-8 rows a request, the deduplicated window layout). The same bits."""
    x = mixed_qkv.view(batch, tokens, -1).transpose(1, 2)
    if retile:
        from sgl_nvidia_mega.qwen3_8.kernels.ops.attention.gdn_front.conv import (
            RETILE_DEFAULT,
            conv_update,
        )

        y = conv_update(
            x,
            conv_state,
            attn.conv_weights,
            attn.bias,
            attn.activation,
            slots,
            window,
            window_rows,
            block_n=128,
            token_major_out=True,
            num_warps=2,
            retile=RETILE_DEFAULT,
        )
        return y.transpose(1, 2).view(batch * tokens, -1)
    st = window.stride()
    if (
        hoisted
        and attn.conv_weights.shape[-1] == 4
        and conv_state.shape[-1] == 3
        and 3 < tokens <= MAX_DRAFT_TOKENS
        and window.dim() == 4
        and st[1] == 1
        and st[3] == 1
        and st[2] >= tokens + 2
    ):
        from sgl_nvidia_mega.qwen3_8.kernels.ops.attention.gdn_front.conv import (
            conv_update,
        )

        y = conv_update(
            x,
            conv_state,
            attn.conv_weights,
            attn.bias,
            attn.activation,
            slots,
            window,
            window_rows,
            block_n=256,
            token_major_out=True,
            hoist=True,
            window_mode=1,
        )
    else:
        from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update

        y = causal_conv1d_update(
            x,
            conv_state,
            attn.conv_weights,
            attn.bias,
            attn.activation,
            conv_state_indices=slots,
            intermediate_conv_window=window,
            intermediate_state_indices=window_rows,
        )
    return y.transpose(1, 2).view(batch * tokens, -1)


def _split_qkv(attn, mixed_qkv, strided):
    """q / k / v [1, rows, heads, dim]: column views of the conv rows (``strided``, or
    a width the fused split does not take), else the fused split's contiguous copies."""
    rows = mixed_qkv.shape[0]
    if not strided:
        from sglang.srt.layers.attention.linear.gdn_backend import (
            MAX_FUSED_QKV_SPLIT_DIM,
        )

        if attn.q_dim + attn.k_dim + attn.v_dim <= MAX_FUSED_QKV_SPLIT_DIM:
            from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
                fused_qkv_split_gdn_prefill,
            )

            return fused_qkv_split_gdn_prefill(
                mixed_qkv,
                attn.num_q_heads,
                attn.num_k_heads,
                attn.num_v_heads,
                attn.head_q_dim,
                attn.head_k_dim,
                attn.head_v_dim,
            )
    query, key, value = torch.split(
        mixed_qkv, [attn.q_dim, attn.k_dim, attn.v_dim], dim=-1
    )
    return (
        query.view(1, rows, attn.num_q_heads, attn.head_q_dim),
        key.view(1, rows, attn.num_k_heads, attn.head_k_dim),
        value.view(1, rows, attn.num_v_heads, attn.head_v_dim),
    )

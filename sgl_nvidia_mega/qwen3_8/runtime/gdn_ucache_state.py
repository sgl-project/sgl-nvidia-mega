# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team): python/sglang/srt/mem_cache/memory_pool.py (MambaPool._allocate_gdn_ucache),
# python/sglang/srt/mem_cache/kv_cache_configurator.py (_gdn_ucache_ring_charge),
# python/sglang/srt/model_executor/model_runner.py and mem_cache/kv/state_storage.py (the
# u-cache cursor resets), python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
# (_commit_gdn_ucache_after_verify), python/sglang/srt/layers/attention/linear/gdn_backend.py
# (the track materializer and its warm-up), linear/utils.py (gdn_ucache_track_snapshots) and
# python/sglang/srt/arg_groups/attention_hook.py (the u-cache server-args checks).
"""The GDN u-cache state the Qwen3.8 executor owns on SGLang's binding v2: the rings
and cursors the u-cache target verify (``gdn_ucache.py``) reads and appends, their
allocation, the cursor resets and the commit after each verify. The executor holds one
``UcacheState`` and delegates to it the members of ``spec.SupportsStateSlotReset``
(``state_bytes_per_slot``, ``state_bytes_per_request_row``, ``keeps_draft_ssm``,
``reset_state_slots``, ``tracks_decode_state``, ``backs_up_state``) and of ``spec.SupportsRecurrentCommit``
(``commit_recurrent_state`` -> ``commit``), and calls ``prepare`` from its own.

When SGLang calls in:
- ``state_bytes_per_slot`` / ``state_bytes_per_request_row``: once, after the build and
  before any pool exists; charged per state slot / request-table row when the state
  pool is sized (one of them is 0).
- ``prepare(view)``: once the pools exist, and again on a recapture; it allocates the
  rings (the first time, or when the pool's size changed) and, with track snapshots,
  warms up the track materializer. Never during a capture.
- ``reset_state_slots``: before every target extend (forward stream, eager or graphed)
  and when a state slot is bound without an extend (a PD decode slot, on the
  scheduler's stream). One launch, no collective, no host sync.
- ``commit``: after every target verify, eagerly on the forward stream or captured
  into the post-verify commit graph on a side stream. It launches the cursor commit,
  then the track snapshots, then the conv-window scatter, and returns True; SGLang
  then commits the PLE side states. No host sync, no collective; every buffer it
  reuses across rounds exists before the capture (the rings, the cursors, the track
  plan); the conv scatter's index casts are per-capture temporaries."""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sgl_nvidia_mega.qwen3_8.kernels.ops.attention.linear import gdn_ucache_state as K

logger = logging.getLogger(__name__)

SGLANG_READS = {
    "srt/speculative/ragged_verify.py": ("RaggedVerifyMode", "read_ragged_verify_mode"),
    "srt/layers/attention/linear/utils.py": ("pp_spec_stable_rows_enabled",),
    "srt/speculative/adaptive_spec_params.py": ("resolve_candidate_steps_from_config",),
    "srt/models/qwen4_exp.py": ("Qwen4ExpLinearDecoderLayer",),
    # Qwen4ExpLinearDecoderLayer's base sets linear_attn
    "srt/models/qwen3_5.py": ("linear_attn", "attn"),
    "srt/layers/radix_linear_attention.py": (
        "num_k_heads",
        "num_v_heads",
        "head_k_dim",
        "head_v_dim",
    ),
    "srt/kernel_plugins/nvidia/eagle_binding.py": (
        "MambaPoolBuffers",
        "temporal",
        "conv",
        "intermediate_conv_window",
    ),
    "kernels/ops/mamba/mamba_state_scatter_triton.py": (
        "fused_conv_window_scatter_with_mask",
    ),
    "srt/utils/device.py|srt/utils/common.py": ("async_h2d",),
}
# Binding v2 (fn-binding-v2-20261112), not on SGLang main yet: what SGLang hands this
# module. Move them into SGLANG_READS when it lands.
SGLANG_V2_READS = {
    "kernels/plugin_kernels/nvidia/spec.py": (
        "SupportsStateSlotReset",
        "SupportsRecurrentCommit",
        "RecurrentCommitIO",
        "state_slots",
        "track_slots",
        "request_rows",
        "accept_lens",
        "last_step",
        "track_steps",
        "src_indices_raw",
    ),
    "EaglePreparationView (binding v2 B5)": ("mamba", "num_request_rows"),
}

# The flush takes 1-8 verify rows a request (a linear draft chain).
MAX_DRAFT_TOKENS = 8
# The radix cache keeps only states with an empty window: prefill writes them, the
# track snapshots fold decode-time ones. A no_buffer finish would donate the live slot,
# whose checkpoint lags its unfolded window.
EXTRA_BUFFER_STRATEGIES = ("extra_buffer", "extra_buffer_lazy")


def extra_buffer_radix(cfg) -> bool:
    """Whether the radix cache keeps decode-time states in an extra buffer: the
    condition ``track_snapshots`` needs from the server args."""
    return not getattr(cfg, "disable_radix_cache", False) and (
        getattr(cfg, "mamba_radix_cache_strategy", None) in EXTRA_BUFFER_STRATEGIES
    )


class UcacheState:
    """The GDN u-cache rings, cursors and track snapshots of one target model's rank.

    ``tokens_per_request``: the widest verify's rows a request (the draft tokens), which
    sets the fold threshold ``flush_min = 16 - T + 1``. The options are the
    tuned environment knobs, with its
    values as defaults:
    - ``key_ring_fp16`` (SGLANG_OPT_GDN_UCACHE_FP16_KEY_RING=1): an fp16 key ring, SM107 only;
    - ``by_request_row`` (SGLANG_OPT_GDN_UCACHE_REQUEST_RINGS=1): rings by request-table
      row instead of by state slot;
    - ``track_snapshots``: the radix cache keeps decode-time states
      (``extra_buffer_radix``). They are written only with slot rings and a bf16 key
      ring (the source's gdn_ucache_track_snapshots), so the recipe tracks none.
    SGLang charges the rings per request-table row (request rings) or per state slot."""

    def __init__(
        self,
        model,
        *,
        tokens_per_request: int,
        key_ring_fp16: bool = True,
        by_request_row: bool = True,
        track_snapshots: bool = False,
    ):
        (
            self.num_layers,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
        ) = _gdn_dims(model)
        self.tokens_per_request = int(tokens_per_request)
        self.key_ring_fp16 = bool(key_ring_fp16)
        self.by_request_row = bool(by_request_row)
        # gdn_prefix_materialize reads the state and the rings at one slot id and
        # takes bf16 k / u rings.
        self._tracks = bool(track_snapshots) and not (
            self.by_request_row or self.key_ring_fp16
        )
        self._buffers: Optional[K.GdnUcacheBuffers] = None
        self._mamba = None
        self._conv_scatter = None
        self._track: Optional[K.GdnUcacheTrackMaterializer] = None
        self._track_state: Optional[torch.Tensor] = None

    # --- spec.SupportsStateSlotReset ---------------------------------------------

    @property
    def ring_bytes_per_slot(self) -> int:
        """The rings' bytes per ring row: ``(H*32*K*2 + HV*32*V*2 + HV*32*4) * L``."""
        return K.gdn_ucache_ring_bytes_per_slot(
            num_layers=self.num_layers,
            num_k_heads=self.num_k_heads,
            num_v_heads=self.num_v_heads,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
        )

    @property
    def state_bytes_per_slot(self) -> int:
        """What the state pool reserves per state slot: the slot rings' bytes."""
        return 0 if self.by_request_row else self.ring_bytes_per_slot

    @property
    def state_bytes_per_request_row(self) -> int:
        """What the state pool reserves per request-table row: the request rings'."""
        return self.ring_bytes_per_slot if self.by_request_row else 0

    @property
    def keeps_draft_ssm(self) -> bool:
        # the rings roll the SSM state back: no per-draft SSM scratch
        return False

    @property
    def tracks_decode_state(self) -> bool:
        return self._tracks

    @property
    def backs_up_state(self) -> bool:
        # the checkpoint lags the unfolded window, which no host backup copies
        return False

    @property
    def buffers(self) -> Optional[K.GdnUcacheBuffers]:
        """The rings and cursors (layer order: the pool's ``layer_ids``), from
        ``prepare`` on; the verify reads and appends them."""
        return self._buffers

    def prepare(self, view) -> None:
        """Allocate the rings from the pool's sizes (``view``: an
        EaglePreparationView); a repeat keeps them when the sizes match."""
        mamba = view.mamba
        state = mamba.temporal
        if state.shape[0] != self.num_layers:
            raise RuntimeError(
                f"GDN u-cache: the state pool has {state.shape[0]} GDN layers, the "
                f"model {self.num_layers} on this rank"
            )
        num_rows = int(view.num_request_rows)
        rows = num_rows if self.by_request_row else int(state.shape[1])
        buffers = self._buffers
        if (
            buffers is None
            or buffers.hist_len.shape[0] != rows
            or buffers.hist_len.device != state.device
        ):
            K._assert_not_capturing("the GDN u-cache rings")
            # A pool of another size: graphs on the old rings are recaptured with it.
            self._buffers = buffers = K.allocate_gdn_ucache(
                num_layers=self.num_layers,
                num_slots=rows,
                num_k_heads=self.num_k_heads,
                num_v_heads=self.num_v_heads,
                head_k_dim=self.head_k_dim,
                head_v_dim=self.head_v_dim,
                max_num_draft_tokens=self.tokens_per_request,
                device=state.device,
                key_ring_fp16=self.key_ring_fp16,
                by_request_row=self.by_request_row,
                track_snapshots=self._tracks,
            )
            self._track = None
            logger.info(
                "GDN u-cache rings: %d %s x %d layers, %.3f GB; flush_min=%d, "
                "k_ring=%s, u_ring=%s, track snapshots %s",
                rows,
                "request rows" if self.by_request_row else "state slots",
                self.num_layers,
                rows * self.ring_bytes_per_slot / (1 << 30),
                buffers.flush_min,
                buffers.k_ring.dtype,
                buffers.u_ring.dtype,
                "on" if self._tracks else "off",
            )
        self._mamba = mamba
        if self._conv_scatter is None:
            from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
                fused_conv_window_scatter_with_mask,
            )

            self._conv_scatter = fused_conv_window_scatter_with_mask
        if self._tracks and (
            self._track is None or not _same_tensor(self._track_state, state)
        ):
            K._assert_not_capturing("the GDN u-cache track materializer")
            self._track = K.GdnUcacheTrackMaterializer(state=state, buffers=buffers)
            self._track_state = state
            # Graph batches and eager verifies run up to one row a request row.
            self._track.warm_up(max_batch_size=num_rows)

    def reset_state_slots(self, slots, request_rows) -> None:
        """Empty the windows of ``slots`` (state-pool ids), or of their
        ``request_rows`` with request rings; ids <= 0 are padding. Before ``prepare``
        there are no rings, and the ones it allocates are zeroed."""
        buffers = self._buffers
        if buffers is None:
            return
        device = buffers.hist_len.device
        if buffers.by_request_row:
            K.reset_gdn_ucache_request_rows(
                buffers=buffers,
                request_rows=_on_device(request_rows, device, "request rows"),
            )
        else:
            K.reset_gdn_ucache_cursors(
                buffers=buffers, slots=_on_device(slots, device, "state slots")
            )

    # --- spec.SupportsRecurrentCommit ----------------------------------------------

    def commit(self, io) -> bool:
        """Commit a chain verify (``io``: RecurrentCommitIO) in the source's order: the
        cursors, the track snapshots (which read the committed cursors), then the conv
        windows of the live and track slots. Rows with slot <= 0 are padding."""
        buffers = self._buffers
        if buffers is None:
            raise RuntimeError("GDN u-cache: a verify commit before prepare")
        slots, accepted = io.state_slots, io.accept_lens
        K.commit_gdn_ucache_cursors(
            buffers=buffers,
            slots=slots,
            num_accept_tokens=accepted,
            request_rows=io.request_rows if buffers.by_request_row else None,
        )
        targets = [(slots, io.last_step)]
        if io.track_slots is not None:
            if self._track is None:
                raise RuntimeError(
                    "GDN u-cache: the verify commit got track slots, but this build "
                    "tracks no decode-time state (tracks_decode_state=False)"
                )
            self._track.materialize(
                slots=slots,
                num_accept_tokens=accepted,
                track_slots=io.track_slots,
                steps_to_track=io.track_steps,
            )
            targets.append((io.track_slots, io.track_steps))
        mamba = self._mamba
        for conv, window in zip(mamba.conv, mamba.intermediate_conv_window):
            if conv.numel() == 0:
                continue
            for dst, steps in targets:
                self._conv_scatter(conv, window, dst, steps, io.src_indices_raw)
        return True

    # --- startup -------------------------------------------------------------------

    @staticmethod
    def config_reasons(cfg) -> list[str]:
        """The server args, environment and device the u-cache cannot serve (the source's
        u-cache server-args checks)."""
        what = "the GDN u-cache"
        reasons = _env_device_reasons(cfg, what)
        role = str(getattr(cfg, "disaggregation_mode", None) or "null")
        dtype = str(getattr(cfg, "mamba_ssm_dtype", None)).replace("torch.", "")
        if dtype != "float16":
            reasons.append(
                f"{what} keeps an fp16 GDN state pool: needs --mamba-ssm-dtype "
                f"float16, got {dtype!r}"
                + (
                    " (the PD prefill must run float16 too: the state transfer "
                    "copies bytes)"
                    if role == "decode"
                    else ""
                )
            )
        if role not in ("null", "decode"):
            reasons.append(
                f"{what} runs on a PD decode role or in aggregated serving, got "
                f"--disaggregation-mode {role!r} (a PD prefill runs no verify: serve "
                "it without the u-cache, with --mamba-ssm-dtype float16)"
            )
        if not getattr(cfg, "disable_radix_cache", False):
            strategy = getattr(cfg, "mamba_radix_cache_strategy", None)
            if strategy not in EXTRA_BUFFER_STRATEGIES:
                reasons.append(
                    f"{what} with the radix cache needs --mamba-radix-cache-strategy "
                    f"extra_buffer or extra_buffer_lazy, got {strategy!r}: a "
                    "no_buffer finish would donate a checkpoint that lags its window"
                )
            if role != "null":
                reasons.append(
                    f"{what} supports the radix cache in aggregated serving only, got "
                    f"--disaggregation-mode {role!r}"
                )
            if getattr(cfg, "enable_session_radix_cache", False):
                reasons.append(f"{what} does not support --enable-session-radix-cache")
        algorithm = getattr(cfg, "speculative_algorithm", None)
        algorithm = getattr(algorithm, "name", algorithm)
        if algorithm not in ("EAGLE", "EAGLE3", "NEXTN"):
            reasons.append(
                f"{what} is a verify path: needs a linear EAGLE / EAGLE3 / NEXTN draft "
                f"chain, got --speculative-algorithm {algorithm!r}"
            )
        topk = getattr(cfg, "speculative_eagle_topk", None)
        if topk != 1:
            reasons.append(
                f"{what} needs a linear draft chain (--speculative-eagle-topk 1), "
                f"got {topk!r}"
            )
        tokens = getattr(cfg, "speculative_num_draft_tokens", None)
        if tokens is None or not 1 <= int(tokens) <= MAX_DRAFT_TOKENS:
            reasons.append(
                f"{what} takes 1 to {MAX_DRAFT_TOKENS} draft tokens per verify, got "
                f"--speculative-num-draft-tokens {tokens!r}"
            )
        for field, flag, why in (
            (
                "enable_mixed_chunk",
                "--enable-mixed-chunk",
                "decode rows of a mixed batch would read the checkpoint without "
                "its window",
            ),
            (
                "enable_streaming_session",
                "--enable-streaming-session",
                "a session turn reuses its slot, and its next prefill would read the "
                "checkpoint without its window",
            ),
            (
                "enable_unified_memory",
                "--enable-unified-memory",
                "the arena state pool has no mamba pool view",
            ),
            (
                "enable_page_major_kv_layout",
                "--enable-page-major-kv-layout",
                "it needs unified memory",
            ),
            (
                "enable_int8_mamba_checkpoint",
                "--enable-int8-mamba-checkpoint",
                "the u-cache keeps an fp16 checkpoint pool",
            ),
            (
                "enable_linear_replayssm",
                "--enable-linear-replayssm",
                "the two verifies own the same state",
            ),
            (
                "enable_linear_replayssm_spec",
                "--enable-linear-replayssm-spec",
                "the two verifies own the same state",
            ),
        ):
            if getattr(cfg, field, False):
                reasons.append(f"{what} does not support {flag}: {why}")
        return reasons


def max_draft_tokens(cfg) -> Optional[int]:
    """The widest verify's rows a request: the draft tokens, or under adaptive
    speculation its largest candidate step + 1 (topk 1)."""
    tokens = getattr(cfg, "speculative_num_draft_tokens", None)
    if tokens is None or not getattr(cfg, "speculative_adaptive", False):
        return tokens
    from sglang.srt.speculative.adaptive_spec_params import (
        resolve_candidate_steps_from_config,
    )

    steps = resolve_candidate_steps_from_config(
        cfg_path=getattr(cfg, "speculative_adaptive_config", None)
    )
    return max(int(tokens), max(steps) + 1)


def _env_device_reasons(cfg, what: str) -> list[str]:
    """The source's checks that read the environment, the adaptive width and the device."""
    from sglang.srt.layers.attention.linear.utils import pp_spec_stable_rows_enabled
    from sglang.srt.speculative.ragged_verify import (
        RaggedVerifyMode,
        read_ragged_verify_mode,
    )

    reasons = []
    if read_ragged_verify_mode() is not RaggedVerifyMode.STATIC:
        reasons.append(f"{what} needs SGLANG_RAGGED_VERIFY_MODE=static")
    if pp_spec_stable_rows_enabled():
        reasons.append(f"{what} does not support SGLANG_ENABLE_PP_SPEC")
    widest = max_draft_tokens(cfg)
    if getattr(cfg, "speculative_adaptive", False) and (widest or 0) > MAX_DRAFT_TOKENS:
        reasons.append(
            f"{what} takes 1 to {MAX_DRAFT_TOKENS} draft tokens per verify; the "
            f"--speculative-adaptive candidate steps reach {widest}"
        )
    # The request rings and the fp16 key ring run on the vendored SM107 flush only.
    if torch.cuda.is_available() and torch.cuda.get_device_capability() != (10, 7):
        major, minor = torch.cuda.get_device_capability()
        reasons.append(
            f"{what} keeps request rings and an fp16 key ring, which only the SM107 "
            f"flush takes; got SM{major}{minor}"
        )
    return reasons


def _gdn_dims(model) -> tuple[int, int, int, int, int]:
    """(L, H, HV, K, V) of this rank's GDN layers, heads per rank."""
    from sglang.srt.models.qwen4_exp import Qwen4ExpLinearDecoderLayer

    m = model.model
    dims = set()
    num_layers = 0
    for i in range(m.start_layer, m.end_layer):
        layer = m.layers[i]
        if isinstance(layer, Qwen4ExpLinearDecoderLayer):
            attn = layer.linear_attn.attn
            dims.add(
                (attn.num_k_heads, attn.num_v_heads, attn.head_k_dim, attn.head_v_dim)
            )
            num_layers += 1
    if num_layers == 0 or len(dims) != 1:
        raise ValueError(
            f"GDN u-cache: {num_layers} GDN layers with head shapes {sorted(dims)}, "
            "needs at least one layer and one shape"
        )
    return (num_layers, *dims.pop())


def _on_device(ids, device, what: str) -> torch.Tensor:
    if ids is None:
        raise RuntimeError(f"GDN u-cache: the reset needs its {what}")
    if not isinstance(ids, torch.Tensor):
        from sglang.srt.utils import async_h2d

        return async_h2d(list(ids), dtype=torch.int64, device=device)
    return ids if ids.device == device else ids.to(device, non_blocking=True)


def _same_tensor(a: Optional[torch.Tensor], b: torch.Tensor) -> bool:
    return (
        a is not None
        and a.data_ptr() == b.data_ptr()
        and a.shape == b.shape
        and a.stride() == b.stride()
    )

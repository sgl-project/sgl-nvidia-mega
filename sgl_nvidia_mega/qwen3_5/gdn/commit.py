# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/kernels/ops/attention/linear/gdn_replayssm_spec_decode.py
# (commit_gdn_replayssm_after_verify).

from __future__ import annotations

import logging

import torch

from sgl_nvidia_mega.qwen3_5.sm103.gdn import commit as kernel

logger = logging.getLogger(__name__)

SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py": (
        "SupportsRecurrentCommit",
        "commit_recurrent_state",
        "RecurrentCommitIO",
        "state_slots",
        "track_slots",
        "request_rows",
        "accept_lens",
        "last_step",
        "track_steps",
        "draft_token_num",
        "src_indices_raw",
    ),
    "kernels/ops/attention/linear/gdn_replayssm_spec_decode.py": (
        "commit_gdn_replayssm_spec",
        "commit_gdn_replayssm_spec(write_pos)",
        "commit_gdn_replayssm_spec(cache_base)",
        "commit_gdn_replayssm_spec(is_flush)",
        "commit_gdn_replayssm_spec(num_accepted)",
        "commit_gdn_replayssm_spec(replay_indices)",
        "commit_gdn_replayssm_spec(state_batch_indices)",
        "commit_gdn_replayssm_spec(max_cache_len)",
        "commit_gdn_replayssm_spec(max_spec_len)",
        "commit_gdn_replayssm_spec(fold_every_commit)",
        "commit_gdn_replayssm_spec(null_block_id)",
    ),
    "kernels/ops/mamba/mamba_state_scatter_triton.py": (
        "fused_conv_window_scatter_with_mask",
    ),
    "srt/kernel_plugins/nvidia/eagle_binding.py|srt/megakernels/eagle_binding.py": (
        "conv",
        "temporal",
        "intermediate_conv_window",
        "ring_d",
        "ring_k",
        "ring_g",
        "ring_rawv",
        "ring_rawk",
        "ring_beta",
        "ring_write_pos",
        "ring_cache_base",
        "ring_is_flush",
    ),
}

SGLANG_MIRRORS = {
    "kernels/ops/attention/linear/gdn_replayssm_spec_decode.py:"
    "commit_gdn_replayssm_after_verify": "72dd9b7179e458c7",
}

K = V = kernel.K


def pool_reasons(pools) -> list[str]:
    if pools is None:
        return ["no GDN pools"]
    ring_d, ring_k, ring_g = pools.ring_d, pools.ring_k, pools.ring_g
    if any(t is None for t in (ring_d, ring_k, ring_g, pools.ring_cache_base)):
        return ["no circular ReplaySSM verify ring"]
    reasons = []
    ring = ring_d.shape[-2]
    if ring != kernel.RING:
        reasons.append(f"ring length {ring} (serves {kernel.RING})")
    if (
        pools.ring_beta is not None
        or pools.ring_rawk is None
        or pools.ring_rawv is None
    ):
        reasons.append("not the compact (low-part) ReplaySSM ring")
    temporal = pools.temporal
    if temporal.dtype != torch.bfloat16:
        reasons.append(f"{temporal.dtype} checkpoints")
    if (
        temporal.dim() != 5
        or tuple(temporal.shape[3:]) != (V, K)
        or temporal.stride()[2:] != (V * K, K, 1)
    ):
        reasons.append(f"checkpoints {tuple(temporal.shape)} {temporal.stride()}")
        return reasons
    layers, _, hv = temporal.shape[:3]
    hk = ring_k.shape[2] if ring_k.dim() == 5 else 0
    if not hk or hv % hk:
        reasons.append(f"{hv} value heads over {hk} key heads")
        return reasons
    rings = [(ring_d, hv, V), (ring_k, hk, K)]
    if pools.ring_rawv is not None and pools.ring_rawk is not None:
        rings += [(pools.ring_rawv, hv, V), (pools.ring_rawk, hk, K)]
    for t, heads, dim in rings:
        if (
            t.dtype != torch.bfloat16
            or t.dim() != 5
            or t.shape[0] != layers
            or tuple(t.shape[2:]) != (heads, ring, dim)
            or t.stride()[2:] != (ring * dim, dim, 1)
        ):
            reasons.append(f"ring {t.dtype} {tuple(t.shape)} {t.stride()}")
    if (
        ring_g.dtype != torch.float32
        or ring_g.dim() != 4
        or ring_g.shape[0] != layers
        or tuple(ring_g.shape[2:]) != (hv, ring)
        or ring_g.stride()[2:] != (ring, 1)
    ):
        reasons.append(f"ring gates {ring_g.dtype} {tuple(ring_g.shape)}")
    for t, dtype in (
        (pools.ring_write_pos, torch.int32),
        (pools.ring_cache_base, torch.int32),
        (pools.ring_is_flush, torch.int8),
    ):
        if t is None or t.dtype != dtype or t.dim() != 1 or t.stride(0) != 1:
            reasons.append(f"ring cursors are not unit-stride {dtype}")
            break
    tensors = [ring_d, ring_k, ring_g, pools.ring_rawv, pools.ring_rawk]
    tensors += [pools.ring_write_pos, pools.ring_cache_base, pools.ring_is_flush]
    if any(t is not None and t.device != temporal.device for t in tensors):
        reasons.append("the ring and the checkpoints sit on different devices")
    return reasons


class RecurrentCommit:
    def __init__(self, device_index: int):
        kernel.build(device_index)
        self.pools = None
        self.bound = False
        self._ops = None

    def bind(self, pools) -> None:
        if self.bound:
            return
        self.bound = True
        reasons = pool_reasons(pools)
        if reasons:
            logger.info(
                "NVIDIA megakernel: Qwen3.5 option gdn_commit: SGLang's recurrent "
                "commit runs (%s)",
                "; ".join(reasons),
            )
            return
        from sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode import (
            commit_gdn_replayssm_spec,
        )
        from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
            fused_conv_window_scatter_with_mask,
        )

        self._ops = (commit_gdn_replayssm_spec, fused_conv_window_scatter_with_mask)
        self.pools = pools
        logger.info(
            "NVIDIA megakernel: Qwen3.5 option gdn_commit: the CUDA ReplaySSM commit "
            "runs (ring %d, %d GDN layers)",
            pools.ring_d.shape[-2],
            pools.temporal.shape[0],
        )

    def __call__(self, io) -> bool:
        """Commit one chain verify on the current stream, or decline (False)."""
        pools = self.pools
        if pools is None:
            return False
        advance, conv_rollback = self._ops
        track = io.track_slots
        advance(
            write_pos=pools.ring_write_pos,
            cache_base=pools.ring_cache_base,
            is_flush=pools.ring_is_flush,
            num_accepted=io.accept_lens,
            replay_indices=io.request_rows,
            state_batch_indices=io.state_slots,
            max_cache_len=pools.ring_d.shape[-2],
            max_spec_len=io.draft_token_num,
            fold_every_commit=True,
            null_block_id=-1,
        )
        kernel.commit(
            checkpoint=pools.temporal,
            ring_d=pools.ring_d,
            ring_dlo=pools.ring_rawv,
            ring_k=pools.ring_k,
            ring_klo=pools.ring_rawk,
            ring_g=pools.ring_g,
            state_indices=io.state_slots,
            ring_rows=io.request_rows,
            write_pos=pools.ring_write_pos,
            cache_base=pools.ring_cache_base,
            is_flush=pools.ring_is_flush,
            accept_lens=io.accept_lens,
            track_indices=track,
            track_steps=io.track_steps,
            null_block=-1,
        )
        for conv, window in zip(pools.conv, pools.intermediate_conv_window):
            conv_rollback(
                conv, window, io.state_slots, io.last_step, io.src_indices_raw
            )
            if track is not None:
                conv_rollback(conv, window, track, io.track_steps, io.src_indices_raw)
        return True

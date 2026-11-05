# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The MiniMax-M3.1 routed experts on one SM100-family GPU (GB300): the copy the token-
parallel MoE reads is recovered bit for bit from the flashinfer_trtllm_routed layout.

One TP4 rank's 128 experts in the kernel's layout (FC1 gate rows then up rows, FC2,
row-major NVFP4 bytes and E4M3 scales, random bytes) are loaded as SGLang loads them
([w1 | w3] rows), then shuffled by SGLang's own processing (reorder_w1w3_to_w3w1 and
prepare_static_weights_for_trtllm_fp4_moe); weights.routed_experts must return the
original bytes."""

from types import SimpleNamespace

import pytest
import torch


def _gpu_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import flashinfer  # noqa: F401
        import sglang  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _gpu_ready(), reason="needs an SM100-family GPU")


def test_the_kernel_layout_is_recovered_bit_for_bit():
    from sglang.srt.layers.quantization.utils import (
        prepare_static_weights_for_trtllm_fp4_moe,
        reorder_w1w3_to_w3w1,
    )

    from sgl_nvidia_mega.minimax_m31 import weights

    g = torch.Generator(device="cuda").manual_seed(31)
    want = {
        operand: torch.randint(
            0, 256, (weights.E, rows, k), generator=g, dtype=torch.uint8, device="cuda"
        )
        for _, operand, rows, k in weights.ROUTED
    }
    e4m3 = torch.float8_e4m3fn
    w13, w13_sf = reorder_w1w3_to_w3w1(
        want["routed_fc1_weight"], want["routed_fc1_weight_sf"].view(e4m3), dim=-2
    )
    shuffled = prepare_static_weights_for_trtllm_fp4_moe(
        w13,
        want["routed_fc2_weight"],
        w13_sf,
        want["routed_fc2_weight_sf"].view(e4m3),
        weights.H,
        weights.I,
        weights.E,
    )
    experts = SimpleNamespace(
        **{name: t for (name, _, _, _), t in zip(weights.ROUTED, shuffled)}
    )
    got = weights.routed_experts(experts)
    for name, operand, _, _ in weights.ROUTED:
        shuffle = getattr(experts, name).view(torch.uint8).reshape(weights.E, -1)
        assert not torch.equal(shuffle, want[operand].reshape(weights.E, -1)), name
        assert torch.equal(got[operand].view(torch.uint8), want[operand]), operand

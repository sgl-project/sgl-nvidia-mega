"""The QSA PDL-fix QK norm launch (kernels/ops/attention/
fused_qk_rmsnorm_rope_gate.py: PDL, the kernel waiting for the QKV before its loads)
against SGLang's fused_qk_gemma_rmsnorm_rope_gate: the same bits, at the 4-row, 24 / 2
head shape it serves, 1-D and M-RoPE positions, after a QKV GEMM on the same stream."""

import pytest
import torch

from sgl_nvidia_mega.qwen3_8.kernels.ops.attention import (
    fused_qk_rmsnorm_rope_gate as plugin,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="needs PDL (SM90+)",
)


@pytest.mark.parametrize("mrope", [False, True])
@pytest.mark.parametrize("fix", [False, True])
def test_same_bits_as_sglang(mrope, fix):
    from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import (
        fused_qk_gemma_rmsnorm_rope_gate as sglang_fused,
    )

    g = torch.Generator(device="cuda").manual_seed(7)
    rows, heads, kv_heads, dim, rotary = 4, 24, 2, 256, 64
    x = torch.randn(rows, 2560, generator=g, device="cuda").bfloat16()
    w = torch.randn(13312, 2560, generator=g, device="cuda").bfloat16() * 0.02
    qkv = x @ w.T  # the producer the PDL launch waits for
    q_gate, k, _ = qkv.split([2 * heads * dim, kv_heads * dim, kv_heads * dim], dim=-1)
    q_w = torch.randn(dim, generator=g, device="cuda").bfloat16()
    k_w = torch.randn(dim, generator=g, device="cuda").bfloat16()
    cache = torch.randn(8192, rotary, generator=g, device="cuda")
    if mrope:
        positions = torch.randint(0, 8192, (3, rows), generator=g, device="cuda")
        axes = torch.randint(0, 3, (rotary // 2,), generator=g, device="cuda").int()
    else:
        positions = torch.randint(0, 8192, (rows,), generator=g, device="cuda")
        axes = None
    args = (q_gate, k, q_w, k_w, cache, positions, 1e-6, heads, kv_heads, dim, rotary)
    want = sglang_fused(*args, has_gate=True, mrope_axis_map=axes)
    got = plugin.fused_qk_gemma_rmsnorm_rope_gate(
        *args, has_gate=True, mrope_axis_map=axes, qsa_pdl_fix=fix
    )
    torch.cuda.synchronize()
    for a, b in zip(got, want):
        assert torch.equal(a, b)

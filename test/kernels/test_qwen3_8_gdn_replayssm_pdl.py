# SPDX-License-Identifier: Apache-2.0
"""Slot gdn_native_pdl's ReplaySSM verify (``gdn_front/replayssm_pdl.py``) on one
SM100-family GPU, at Qwen3.8's TP1 GatedDeltaNet shape: with the ring cursors as columns
of one entry per replay row (SGLang's request sub-pool, sgl-project/sglang#49719) it
writes the bits it writes with the cursors as their own arrays, the output and this
step's ring entries, with and without the dependent launch.
"""

import pytest
import torch

H, K, HV, V = 16, 128, 48, 128  # Qwen3.8 at TP1
D = 2 * H * K + HV * V
DRAFT, SLOTS = 4, 96


def _gpu_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _gpu_ready(), reason="needs an SM100-family GPU")


def _columns(write_pos, cache_base, is_flush):
    """The cursors as columns of one entry per replay row (int32 write_pos / cache_base,
    int8 flush flags); the other columns hold values a unit-stride read would trip on."""
    entry = torch.full((len(write_pos), 3), 7, dtype=torch.int32, device="cuda")
    entry[:, 0], entry[:, 2] = write_pos, cache_base
    flags = torch.ones(len(is_flush), 4, dtype=torch.int8, device="cuda")
    flags[:, 1] = is_flush
    return entry[:, 0], entry[:, 2], flags[:, 1]


@pytest.mark.parametrize("bs", [4, 8, 16, 33, 64])
@pytest.mark.parametrize("pdl, gate", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("ring_len", [16, 32])
def test_strided_cursors_are_the_unit_stride_bits(pdl, gate, ring_len, bs):
    from sgl_nvidia_mega.qwen3_8.kernels.ops.attention.gdn_front import replayssm_pdl

    g = torch.Generator(device="cuda").manual_seed(
        5 + int(pdl) + 2 * int(gate) + 10 * bs + ring_len
    )
    tokens = bs * DRAFT

    def rand(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, device="cuda", generator=g).to(dtype)

    qkv = rand(tokens, D)  # strided q / k / v, as the conv's split leaves them
    q = qkv[:, : H * K].view(tokens, H, K)
    k = qkv[:, H * K : 2 * H * K].view(tokens, H, K)
    v = qkv[:, 2 * H * K :].view(tokens, HV, V)
    ba = rand(tokens, 2 * HV)
    b, a = ba[:, :HV], ba[:, HV:]
    A_log, dt_bias = rand(HV, dtype=torch.float32), rand(HV, dtype=torch.float32)
    state = rand(SLOTS, HV, V, K)
    ring = [
        rand(SLOTS, HV, ring_len, V),
        rand(SLOTS, H, ring_len, K),
        -torch.rand(SLOTS, HV, ring_len, device="cuda", generator=g),
        rand(SLOTS, HV, ring_len, V),
        rand(SLOTS, H, ring_len, K),
    ]
    write_pos = torch.randint(0, ring_len - DRAFT, (SLOTS,), device="cuda", generator=g)
    write_pos = write_pos.to(torch.int32)
    cache_base = torch.randint(0, ring_len, (SLOTS,), device="cuda", generator=g)
    cache_base = cache_base.to(torch.int32)
    # flush rows return early in the verify launch; a misread flag would skip a row
    is_flush = (torch.rand(SLOTS, device="cuda", generator=g) < 0.1).to(torch.int8)
    qsl = torch.arange(0, tokens + 1, DRAFT, dtype=torch.int32, device="cuda")
    states = (torch.randperm(SLOTS - 1, device="cuda", generator=g)[:bs] + 1).int()
    states[0] = -1  # a padded request: zero output, ring untouched
    replay = torch.randperm(SLOTS, device="cuda", generator=g)[:bs].int()
    outs = []
    for cursors in (
        (write_pos, cache_base, is_flush),
        _columns(write_pos, cache_base, is_flush),
    ):
        got_ring = [t.clone() for t in ring]
        got = q.new_zeros(tokens, HV, V)  # flush rows keep these
        replayssm_pdl.verify(
            q, k, v, a, b, A_log, dt_bias, got, state, *got_ring, None, qsl, states,
            replay, *cursors, max_cache_len=ring_len, max_spec_len=DRAFT,
            scale=K**-0.5, null_block_id=-1, block_v=32, num_warps=2, pdl=pdl,
            wait_for_gate_producer=gate,
        )  # fmt: skip
        outs.append((got, got_ring))
    (want, want_ring), (got, got_ring) = outs
    assert want[DRAFT:].abs().sum() > 0  # the rows past the padded one ran
    assert torch.equal(got, want)
    assert all(torch.equal(x, y) for x, y in zip(got_ring, want_ring))

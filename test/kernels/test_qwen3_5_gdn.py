# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The kernels slot gdn_verify launches its own way, on one SM100-family GPU (GB300),
against SGLang's, bit for bit, at Qwen3.5's TP8 GatedDeltaNet shapes, for steps of 1, 2
and 4 requests:

- the causal conv update with the 128-channel tile and the token-major output, against
  ``causal_conv1d_update`` (output, conv states and intermediate windows), on the
  contiguous split and on the token-major projection views;
- the gated RMSNorm writing static FP8, against SGLang's gated norm followed by the FP8
  linear's static input quantization (``static_quant_fp8``), with z as rows and as
  token-major views;
- the ReplaySSM verify launch, with and without the dependent launch (and its wait
  before the gates, slot gdn_front's), against ``gdn_replayssm_spec_decode``'s (output and
  ring entries).

And slot gdn_front's kernels (numeric: those of SGLang branch nvidia/qwen35-agentx-gb300,
checked bit for bit against that branch on the same inputs, not here):

- the verify front: its in_proj_qkvz and in_proj_ba rows against fp64 on the same
  operands, at most 1.05 x the mean relative L2 and 2 x the max abs error of SGLang's ops
  (FlashInfer's FP8 BMM on SGLang's static quantization, cuBLAS's BF16 GEMM); its conv
  output, rolling state and deduplicated windows bit for bit ``causal_conv1d_update``'s
  on its own projected rows; a padded request leaves the state and windows alone; graph
  replays equal eager launches;
- its gated norm writing static FP8, bit for bit slot gdn_verify's.

And slot gdn_trt_front (TensorRT-LLM r3's own in_proj front, numeric) at the
TP8 shapes, on the hop's E4M3 rows and on BF16 rows: its q / k / v rows (read back from the windows it
writes) and z against fp64 on the same operands, at most 1.05 x the mean relative L2 and 2 x the max abs error of
SGLang's ops, b / a likewise; its conv output, rolling state and windows bit for bit ``causal_conv1d_update``'s on
those rows; a padded request leaves the pools alone; it reads its rows and state index only after its PDL wait (a
producer that releases it at entry and writes them late; the wait-less build as the positive control); graph
replays equal eager launches, and a launch that was not prepared is refused inside a capture.
"""

import pytest
import torch

H, K, HV, V = 2, 128, 8, 128  # per rank at TP8: 16 / 64 heads over 8
D = 2 * H * K + HV * V  # the conv's q / k / v channels
DRAFT, WIDTH, SLOTS = 7, 4, 136


def _gpu_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _gpu_ready(), reason="needs an SM100-family GPU")


def _projection(bs, views, g):
    """A verify step's mixed q / k / v rows [bs * 7, D]: a contiguous copy, or the
    token-major view of the q / k / v / z projection."""
    qkvz = torch.randn(bs * DRAFT, D + HV * V, device="cuda", generator=g).to(
        torch.bfloat16
    )
    return qkvz[:, :D] if views else qkvz[:, :D].contiguous()


# a covered step's requests (slot gdn_verify covers B1 to B128) and their slots
REQUESTS = (1, 2, 4, 8, 16, 32, 64, 128)
STATE_SLOTS = (4, 7, 0, 8, *range(39, 11, -1), *range(135, 39, -1))
RING_SLOTS = (3, 5, 0, 8, *range(12, 136))
REPLAY_SLOTS = (1, 2, 4, 6, *range(10, 134))


@pytest.mark.parametrize("bs", REQUESTS)
@pytest.mark.parametrize("views", [False, True])
@pytest.mark.parametrize("block_n", [256, 128])
def test_conv_update_is_sglangs(views, block_n, bs):
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update

    from sgl_nvidia_mega.qwen3_5.gdn.conv import conv_update

    g = torch.Generator(device="cuda").manual_seed(block_n + views + 10 * bs)
    rows = _projection(bs, views, g)
    weight = torch.randn(D, WIDTH, device="cuda", generator=g).to(torch.bfloat16)
    bias = torch.randn(D, device="cuda", generator=g).to(torch.bfloat16)
    state = torch.randn(SLOTS, D, WIDTH - 1, device="cuda", generator=g).to(
        torch.bfloat16
    )
    window = torch.zeros(
        SLOTS, DRAFT, D, WIDTH - 1, dtype=torch.bfloat16, device="cuda"
    )
    slots = torch.tensor(STATE_SLOTS[:bs], dtype=torch.int32, device="cuda")
    inter = torch.arange(bs, dtype=torch.int32, device="cuda")
    x = rows.view(bs, DRAFT, D).transpose(1, 2)
    want_state, want_window = state.clone(), window.clone()
    want = causal_conv1d_update(
        x,
        want_state,
        weight,
        bias,
        "silu",
        conv_state_indices=slots,
        intermediate_conv_window=want_window,
        intermediate_state_indices=inter,
    )
    got = conv_update(
        x,
        state,
        weight,
        bias,
        "silu",
        slots,
        window,
        inter,
        block_n=block_n,
        token_major_out=views,
    )
    assert torch.equal(got.contiguous(), want.contiguous())
    assert torch.equal(state, want_state) and torch.equal(window, want_window)
    # the slot reads the output back as token rows without a copy
    assert got.transpose(1, 2).is_contiguous()


@pytest.mark.parametrize("bs", REQUESTS)
@pytest.mark.parametrize("tile", [(128, 4), (64, 2)])
@pytest.mark.parametrize("window_mode", [0, 1])
def test_hoisted_conv_is_sglangs(window_mode, tile, bs):
    """Slot gdn_conv: x rows loaded first (and with window_mode 1 one window row a channel) writes the
    output, the conv states and the deduplicated windows bit for bit as SGLang's kernel."""
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update

    from sgl_nvidia_mega.qwen3_5.gdn.conv import conv_update

    g = torch.Generator(device="cuda").manual_seed(window_mode + tile[0] + 10 * bs)
    rows = _projection(bs, True, g)
    weight = torch.randn(D, WIDTH, device="cuda", generator=g).to(torch.bfloat16)
    state = torch.randn(SLOTS, D, WIDTH - 1, device="cuda", generator=g).to(
        torch.bfloat16
    )
    shared = torch.randn(SLOTS, D, DRAFT + WIDTH - 2, device="cuda", generator=g).to(
        torch.bfloat16
    )
    slots = torch.tensor(STATE_SLOTS[:bs], dtype=torch.int32, device="cuda")
    if bs > 2:
        slots[1] = -1  # a padded request
    inter = torch.arange(bs, dtype=torch.int32, device="cuda")
    x = rows.view(bs, DRAFT, D).transpose(1, 2)

    def windows(t):
        st = t.stride()
        return t.as_strided((SLOTS, DRAFT, D, WIDTH - 1), (st[0], 1, st[1], 1))

    want_state, want_shared = state.clone(), shared.clone()
    want = causal_conv1d_update(
        x,
        want_state,
        weight,
        None,
        "silu",
        conv_state_indices=slots,
        intermediate_conv_window=windows(want_shared),
        intermediate_state_indices=inter,
    )
    got = conv_update(
        x,
        state,
        weight,
        None,
        "silu",
        slots,
        windows(shared),
        inter,
        block_n=tile[0],
        token_major_out=True,
        hoist=True,
        num_warps=tile[1],
        window_mode=window_mode,
    )
    live = (slots >= 0)[:, None, None]
    assert torch.equal(
        torch.where(live, got, 0), torch.where(live, want.contiguous(), 0)
    )
    assert torch.equal(state, want_state) and torch.equal(shared, want_shared)


@pytest.mark.parametrize("bs", REQUESTS)
@pytest.mark.parametrize("views", [False, True])
def test_norm_fp8_is_sglangs_norm_then_static_quant(views, bs):
    from sglang.kernels.ops.attention.fla.layernorm_gated import RMSNorm
    from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8

    from sgl_nvidia_mega.qwen3_5.gdn.norm_fp8 import rms_norm_gated_fp8

    g = torch.Generator(device="cuda").manual_seed(int(views) + 10 * bs)
    tokens = bs * DRAFT
    core = torch.randn(tokens * HV, V, device="cuda", generator=g).to(torch.bfloat16)
    qkvz = torch.randn(tokens, D + HV * V, device="cuda", generator=g).to(
        torch.bfloat16
    )
    z = qkvz[:, D:].view(tokens, HV, V)  # token-major view, rows with gaps
    if not views:
        z = z.reshape(-1, V)
    norm = RMSNorm(
        V, eps=1e-6, norm_before_gate=True, device="cuda", dtype=torch.bfloat16
    )
    norm.weight.data = torch.randn(V, device="cuda", generator=g).to(torch.bfloat16)
    scale = torch.tensor([0.02], device="cuda")
    want, _ = static_quant_fp8(
        norm(core, z).reshape(tokens, HV * V), scale, repeat_scale=False
    )
    got = rms_norm_gated_fp8(
        core, norm.weight, norm.eps, z, scale, activation=norm.activation
    )
    assert torch.equal(
        got.view(torch.uint8).reshape(tokens, -1), want.view(torch.uint8)
    )


@pytest.mark.parametrize("bs", REQUESTS)
@pytest.mark.parametrize("pdl, gate", [(False, False), (True, False), (True, True)])
def test_replayssm_verify_is_sglangs(pdl, gate, bs):
    """The slot's verify launch (32-wide value tiles on 2 warps, and with or without the
    dependent launch, waiting before the gates with ``gate``) against
    gdn_replayssm_spec_decode's verify launch: the output and this step's ring entries."""
    try:
        from sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode import (
            gdn_replayssm_spec_decode,
        )
    except ImportError:  # in fla/ before sgl-project/sglang#45112
        from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode import (
            gdn_replayssm_spec_decode,
        )

    from sgl_nvidia_mega.qwen3_5.gdn import replayssm

    g = torch.Generator(device="cuda").manual_seed(
        3 + int(pdl) + 2 * int(gate) + 10 * bs
    )
    ring_len, slots = 16, SLOTS
    tokens = bs * DRAFT

    def rand(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, device="cuda", generator=g).to(dtype)

    qkv = rand(tokens, D)  # strided q / k / v, as the slot's split leaves them
    q = qkv[:, : H * K].view(tokens, H, K)
    k = qkv[:, H * K : 2 * H * K].view(tokens, H, K)
    v = qkv[:, 2 * H * K :].view(tokens, HV, V)
    ba = rand(tokens, 2 * HV)
    b, a = ba[:, :HV], ba[:, HV:]
    A_log, dt_bias = rand(HV, dtype=torch.float32), rand(HV, dtype=torch.float32)
    state = rand(slots, HV, V, K)
    ring = [
        rand(slots, HV, ring_len, V),
        rand(slots, H, ring_len, K),
        -torch.rand(slots, HV, ring_len, device="cuda", generator=g),
        rand(slots, HV, ring_len, V),
        rand(slots, H, ring_len, K),
    ]
    write_pos = torch.randint(0, ring_len - DRAFT, (slots,), device="cuda", generator=g)
    write_pos = write_pos.to(torch.int32)
    cache_base = torch.randint(0, ring_len, (slots,), device="cuda", generator=g)
    cache_base = cache_base.to(torch.int32)
    is_flush = torch.zeros(slots, dtype=torch.int8, device="cuda")
    qsl = torch.arange(0, tokens + 1, DRAFT, dtype=torch.int32, device="cuda")
    states = torch.tensor(RING_SLOTS[:bs], dtype=torch.int32, device="cuda")
    replay = torch.tensor(REPLAY_SLOTS[:bs], dtype=torch.int32, device="cuda")
    want_ring, got_ring = [t.clone() for t in ring], [t.clone() for t in ring]
    want, got = (q.new_empty(tokens, HV, V) for _ in range(2))
    gdn_replayssm_spec_decode(
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        checkpoint_state=state,
        d_cache=want_ring[0],
        k_cache=want_ring[1],
        g_cache=want_ring[2],
        rawv_cache=want_ring[3],
        rawk_cache=want_ring[4],
        beta_cache=None,
        out=want,
        query_start_loc=qsl,
        ssm_state_indices=states,
        replay_indices=replay,
        write_pos=write_pos,
        cache_base=cache_base,
        is_flush=is_flush,
        max_cache_len=ring_len,
        max_spec_len=DRAFT,
        scale=K**-0.5,
        use_qk_l2norm_in_kernel=True,
        null_block_id=-1,
        launch_mode="verify",
        block_v=32,
        num_warps=2,
    )
    replayssm.verify(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        got,
        state,
        *got_ring,
        None,
        qsl,
        states,
        replay,
        write_pos,
        cache_base,
        is_flush,
        max_cache_len=ring_len,
        max_spec_len=DRAFT,
        scale=K**-0.5,
        null_block_id=-1,
        block_v=32,
        num_warps=2,
        pdl=pdl,
        wait_for_gate_producer=gate,
    )
    assert torch.equal(got, want)
    assert all(torch.equal(x, y) for x, y in zip(got_ring, want_ring))


# -------------------------------------------------------------------- slot gdn_core


@pytest.mark.parametrize("bs", REQUESTS)
@pytest.mark.parametrize("history", [False, True])
@pytest.mark.parametrize("pad", [False, True])
def test_core_is_gdn_verify_numerically(pad, history, bs):
    """Slot gdn_core's one launch against slot gdn_verify's verify (32-wide value tiles,
    2 warps) and FP8 gated norm on the same inputs: the replay output and the FP8 bytes
    up to rounding, this step's ring entries (update and key high + low parts, log-decay)
    up to FP32 summation order; a padded request writes zeros and leaves its ring alone.
    History: an empty ring (BF16 checkpoints fold every commit) or 1-9 committed steps."""
    from sgl_nvidia_mega.qwen3_5.gdn import replayssm
    from sgl_nvidia_mega.qwen3_5.gdn.norm_fp8 import rms_norm_gated_fp8
    from sgl_nvidia_mega.qwen3_5.sm103.gdn.core import core

    if pad and bs == 1:
        pytest.skip("a padded single request leaves nothing to compare")
    g = torch.Generator(device="cuda").manual_seed(100 * bs + 10 * history + pad)
    ring_len, slots, tokens = 16, bs + 3, bs * DRAFT

    def rand(*shape, scale=1.0):
        return (torch.randn(*shape, device="cuda", generator=g) * scale).to(
            torch.bfloat16
        )

    qkvz = rand(tokens, D + HV * V)  # the convolved q / k / v rows and the gate z
    qkv, z = qkvz[:, :D], qkvz[:, D:].view(tokens, HV, V)
    ba = rand(tokens, 2 * HV)
    b, a = ba[:, :HV], ba[:, HV:]
    A_log = torch.rand(HV, device="cuda", generator=g)
    dt_bias = torch.rand(HV, device="cuda", generator=g)
    state = rand(slots, HV, V, K, scale=0.05)
    ring = [
        rand(slots, HV, ring_len, V, scale=0.05),
        rand(slots, H, ring_len, K, scale=0.09),
        -torch.rand(slots, HV, ring_len, device="cuda", generator=g) * 0.5,
        rand(slots, HV, ring_len, V, scale=1e-4),
        rand(slots, H, ring_len, K, scale=1e-4),
    ]
    states = torch.randperm(slots, device="cuda", generator=g)[:bs].to(torch.int32)
    replay = torch.randperm(slots, device="cuda", generator=g)[:bs]
    if pad:
        states[-1] = -1
    low = 1 if history else 0
    write_pos = torch.randint(
        low,
        ring_len - DRAFT + 1 if history else 1,
        (slots,),
        device="cuda",
        generator=g,
    ).to(torch.int32)
    cache_base = torch.randint(0, ring_len, (slots,), device="cuda", generator=g)
    cache_base = cache_base.to(torch.int32)
    is_flush = torch.zeros(slots, dtype=torch.int8, device="cuda")
    qsl = torch.arange(0, tokens + 1, DRAFT, dtype=torch.int32, device="cuda")
    weight = rand(V)
    scale = torch.tensor([0.02], device="cuda")
    want_ring, got_ring = [t.clone() for t in ring], [t.clone() for t in ring]
    o_want = torch.zeros(tokens, HV, V, dtype=torch.bfloat16, device="cuda")
    replayssm.verify(
        qkv[:, : H * K].view(tokens, H, K),
        qkv[:, H * K : 2 * H * K].view(tokens, H, K),
        qkv[:, 2 * H * K :].view(tokens, HV, V),
        a,
        b,
        A_log,
        dt_bias,
        o_want,
        state,
        *want_ring,
        None,
        qsl,
        states,
        replay,
        write_pos,
        cache_base,
        is_flush,
        max_cache_len=ring_len,
        max_spec_len=DRAFT,
        scale=K**-0.5,
        null_block_id=-1,
        block_v=32,
        num_warps=2,
    )
    y_want = rms_norm_gated_fp8(o_want.view(-1, V), weight, 1e-6, z, scale)
    o_got = torch.zeros_like(o_want)
    y_got = torch.empty(tokens, HV * V, dtype=torch.float8_e4m3fn, device="cuda")
    core(
        qkv,
        z,
        a,
        b,
        A_log,
        dt_bias,
        state,
        *got_ring,
        states,
        replay,
        write_pos,
        cache_base,
        is_flush,
        weight,
        scale,
        1e-6,
        y_got,
        rows_per_request=DRAFT,
        num_k_heads=H,
        o_out=o_got,
    )
    # the register cap of the widest grids (three CTAs an SM) leaves every byte as it was
    capped = [t.clone() for t in ring]
    y_cap = torch.empty_like(y_got)
    core(
        qkv,
        z,
        a,
        b,
        A_log,
        dt_bias,
        state,
        *capped,
        states,
        replay,
        write_pos,
        cache_base,
        is_flush,
        weight,
        scale,
        1e-6,
        y_cap,
        rows_per_request=DRAFT,
        num_k_heads=H,
        min_blocks=3,
    )
    assert torch.equal(y_cap.view(torch.uint8), y_got.view(torch.uint8))
    assert all(torch.equal(c, g) for c, g in zip(capped, got_ring))
    # the committed history loaded before the PDL wait (early_hist) leaves every byte as it was
    early = [t.clone() for t in ring]
    y_early = torch.empty_like(y_got)
    core(
        qkv,
        z,
        a,
        b,
        A_log,
        dt_bias,
        state,
        *early,
        states,
        replay,
        write_pos,
        cache_base,
        is_flush,
        weight,
        scale,
        1e-6,
        y_early,
        rows_per_request=DRAFT,
        num_k_heads=H,
        early_hist=True,
    )
    assert torch.equal(y_early.view(torch.uint8), y_got.view(torch.uint8))
    assert all(torch.equal(e, g) for e, g in zip(early, got_ring))
    live = torch.ones(tokens, dtype=torch.bool, device="cuda")
    if pad:
        live[-DRAFT:] = False
        assert not y_got[~live].float().any()
    ow, og = o_want[live].float(), o_got[live].float()
    assert (og - ow).norm() <= 2e-3 * ow.norm()
    same = (
        y_got.view(torch.uint8)[live] == y_want.view(torch.uint8).view(tokens, -1)[live]
    )
    assert same.float().mean() >= 0.99
    yw, yg = y_want.view(tokens, -1)[live].float(), y_got[live].float()
    assert (yg - yw).norm() <= 2e-2 * yw.norm()
    # the ring: updates and keys as high + low, the log-decays as written
    for hi, lo, tol in ((0, 3, 5e-3), (1, 4, 1e-5)):
        w = want_ring[hi].float() + want_ring[lo].float()
        o = got_ring[hi].float() + got_ring[lo].float()
        changed = w - (ring[hi].float() + ring[lo].float())
        assert (o - w).norm() <= tol * max(changed.norm().item(), 1e-30)
    assert torch.allclose(got_ring[2], want_ring[2], rtol=0, atol=1e-5)


@pytest.mark.parametrize(
    "variant", [{}, {"early_dependents": True}, {"min_blocks": 3}, {"early_hist": True}]
)
@pytest.mark.parametrize("bs", [1, 2, 4, 8, 64])
@pytest.mark.parametrize("low", [False, True])
def test_core_flat_stores_leave_every_byte_as_it_was(low, bs, variant):
    """Flat stores (predicated, with update vectors stored after the output) write
    the FP8 output and this step's ring entries, with and without the low parts, byte for byte as the
    branchy stores do at TP8, through the ring's end and with a padded request."""
    from sgl_nvidia_mega.qwen3_5.sm103.gdn.core import core

    h, hv = H, HV
    d = 2 * h * K + hv * V
    g = torch.Generator(device="cuda").manual_seed(8000 + 10 * bs + low)
    ring_len, slots, tokens = 16, bs + 3, bs * DRAFT

    def rand(*shape, scale=1.0):
        return (torch.randn(*shape, device="cuda", generator=g) * scale).to(
            torch.bfloat16
        )

    qkvz = rand(tokens, d + hv * V)
    qkv, z = qkvz[:, :d], qkvz[:, d:].view(tokens, hv, V)
    ba = rand(tokens, 2 * hv)
    b, a = ba[:, :hv], ba[:, hv:]
    A_log = torch.rand(hv, device="cuda", generator=g)
    dt_bias = torch.rand(hv, device="cuda", generator=g)
    state = rand(slots, hv, V, K, scale=0.05)
    ring = [
        rand(slots, hv, ring_len, V, scale=0.05),
        rand(slots, h, ring_len, K, scale=0.09),
        -torch.rand(slots, hv, ring_len, device="cuda", generator=g) * 0.5,
        rand(slots, hv, ring_len, V, scale=1e-4) if low else None,
        rand(slots, h, ring_len, K, scale=1e-4) if low else None,
    ]
    states = torch.randperm(slots, device="cuda", generator=g)[:bs].to(torch.int32)
    replay = torch.randperm(slots, device="cuda", generator=g)[:bs]
    if bs > 1:
        states[-1] = -1
    write_pos = torch.randint(0, ring_len, (slots,), device="cuda", generator=g).to(
        torch.int32
    )
    cache_base = torch.randint(0, ring_len, (slots,), device="cuda", generator=g).to(
        torch.int32
    )
    is_flush = torch.zeros(slots, dtype=torch.int8, device="cuda")
    weight = rand(V)
    scale = torch.tensor([0.02], device="cuda")
    got = []
    for flat in (False, True):
        r = [None if t is None else t.clone() for t in ring]
        y = torch.zeros(tokens, hv * V, dtype=torch.float8_e4m3fn, device="cuda")
        core(
            qkv,
            z,
            a,
            b,
            A_log,
            dt_bias,
            state,
            *r,
            states,
            replay,
            write_pos,
            cache_base,
            is_flush,
            weight,
            scale,
            1e-6,
            y,
            rows_per_request=DRAFT,
            num_k_heads=h,
            flat=flat,
            **variant,
        )
        got.append((y.view(torch.uint8), r))
    (y0, r0), (y1, r1) = got
    assert torch.equal(y0, y1)
    assert all(u is None and v is None or torch.equal(u, v) for u, v in zip(r0, r1))
    assert not torch.equal(r1[0], ring[0])  # the step wrote ring entries


# ------------------------------------------------------------------- slot gdn_front

FRONT_SLOTS = 12
# The independent rolling-state, intermediate-window and ring slots used by up
# to eight requests. The TRT front below uses the rolling-state tuple.
FRONT_STATES, FRONT_WINDOWS, FRONT_RINGS = (
    (3, 0, 4, 9, 6, 11, 1, 8),
    (1, 5, 0, 2, 10, 3, 7, 4),
    (2, 6, 9, 0, 11, 5, 3, 10),
)


def _front_operands(seed, *, pad=False, index_dtype=torch.int32):
    """One request of 7 rows at TP8: the normed rows, the static FP8 in_proj_qkvz
    [2560, 4096] with its scales, the BF16 in_proj_ba [16, 4096], the conv weights, the
    pools (rolling states, deduplicated windows, the verify's checkpoint and ring) and
    the step's slots."""
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rand(*shape, scale=1.0):
        return torch.randn(*shape, device="cuda", generator=g) * scale

    phys = rand(FRONT_SLOTS, D, 9).to(torch.bfloat16)
    index = lambda i: torch.tensor([i], dtype=index_dtype, device="cuda")  # noqa: E731
    return dict(
        x=rand(DRAFT, 4096).to(torch.bfloat16),
        w=rand(2560, 4096, scale=0.05).clamp(-440, 440).to(torch.float8_e4m3fn),
        sx=torch.tensor([0.01], device="cuda"),
        sw=torch.tensor([0.002], device="cuda"),
        ba_w=rand(16, 4096, scale=0.02).to(torch.bfloat16),
        conv_w=rand(D, WIDTH, scale=0.5).to(torch.bfloat16),
        state=rand(FRONT_SLOTS, D, WIDTH - 1).to(torch.bfloat16),
        phys=phys,
        windows=phys.as_strided((FRONT_SLOTS, DRAFT, D, WIDTH - 1), (D * 9, 1, 9, 1)),
        checkpoint=rand(FRONT_SLOTS, HV, V, K).to(torch.bfloat16),
        ring_d=rand(FRONT_SLOTS, HV, 16, V).to(torch.bfloat16),
        ring_k=rand(FRONT_SLOTS, H, 16, K).to(torch.bfloat16),
        ring_g=rand(FRONT_SLOTS, HV, 16),
        states=index(-1 if pad else 3),
        inter=index(1),
        req=index(2),
    )


def _run_front(o, packed, outs, **kw):
    from sgl_nvidia_mega.qwen3_5.sm103.gdn import front

    front.front(
        o["x"],
        packed,
        o["sx"],
        o["sw"],
        o["conv_w"],
        o["state"],
        o["windows"],
        o["states"],
        o["inter"],
        outs["projected"],
        outs["conv"],
        outs["error"],
        ba_weight=o["ba_w"],
        ba_out=outs["ba"],
        checkpoint=o["checkpoint"],
        history_d=o["ring_d"],
        history_k=o["ring_k"],
        history_g=o["ring_g"],
        ring_indices=o["req"],
        **kw,
    )


def _front_outputs():
    return dict(
        projected=torch.empty(DRAFT, 2560, dtype=torch.bfloat16, device="cuda"),
        conv=torch.empty(DRAFT, D, dtype=torch.bfloat16, device="cuda"),
        ba=torch.empty(DRAFT, 16, dtype=torch.bfloat16, device="cuda"),
        error=torch.zeros(1, dtype=torch.int32, device="cuda"),
    )


def _errors(got, ref):
    diff = got.double() - ref
    return (diff.norm() / ref.norm()).item(), diff.abs().max().item()


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_front_holds_its_projections_to_fp64_and_its_conv_is_sglangs(index_dtype):
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
    from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8
    from sglang.srt.layers.quantization.fp8_utils import flashinfer_bmm_fp8

    from sgl_nvidia_mega.qwen3_5.sm103.gdn import front

    errors = {n: [] for n in ("qkvz", "qkvz_stock", "ba", "ba_stock")}
    for seed in range(3):
        o = _front_operands(seed, index_dtype=index_dtype)
        state0, phys0 = o["state"].clone(), o["phys"].clone()
        outs = _front_outputs()
        _run_front(o, front.pack(o["w"]), outs)
        torch.cuda.synchronize()
        assert outs["error"].item() == 0
        # the projections against fp64 on the same FP8 / BF16 operands
        xq, _ = static_quant_fp8(o["x"], o["sx"], repeat_scale=False)
        ref = (xq.double() @ o["w"].double().t()) * (
            o["sx"].double() * o["sw"].double()
        )
        stock = flashinfer_bmm_fp8(xq, o["w"].t(), o["sx"], o["sw"], torch.bfloat16)
        errors["qkvz"].append(_errors(outs["projected"], ref))
        errors["qkvz_stock"].append(_errors(stock, ref))
        ref = o["x"].double() @ o["ba_w"].double().t()
        errors["ba"].append(_errors(outs["ba"], ref))
        errors["ba_stock"].append(_errors(o["x"] @ o["ba_w"].t(), ref))
        # the conv on the front's own projected q / k / v rows, as SGLang's update
        state, phys = state0.clone(), phys0.clone()
        windows = phys.as_strided(o["windows"].shape, o["windows"].stride())
        want = causal_conv1d_update(
            outs["projected"][:, :D].view(1, DRAFT, D).transpose(1, 2),
            state,
            o["conv_w"],
            None,
            "silu",
            conv_state_indices=o["states"],
            intermediate_conv_window=windows,
            intermediate_state_indices=o["inter"],
        )
        assert torch.equal(outs["conv"], want.transpose(1, 2).reshape(DRAFT, D))
        assert torch.equal(o["state"], state) and torch.equal(o["phys"], phys)
    e = {n: torch.tensor(v, dtype=torch.float64) for n, v in errors.items()}
    print(
        f"T0 gdn_front qkvz mean rel-L2 {e['qkvz'][:, 0].mean():.4g} vs stock "
        f"{e['qkvz_stock'][:, 0].mean():.4g}, max abs {e['qkvz'][:, 1].max():.4g} vs "
        f"{e['qkvz_stock'][:, 1].max():.4g}; ba {e['ba'][:, 0].mean():.4g} vs "
        f"{e['ba_stock'][:, 0].mean():.4g}, {e['ba'][:, 1].max():.4g} vs "
        f"{e['ba_stock'][:, 1].max():.4g}",
        flush=True,
    )
    for name in ("qkvz", "ba"):
        got, stock = e[name], e[f"{name}_stock"]
        assert got[:, 0].mean() <= 1.05 * stock[:, 0].mean(), name
        assert got[:, 1].max() <= 2 * stock[:, 1].max(), name


def test_front_leaves_a_padded_requests_state_alone():
    from sgl_nvidia_mega.qwen3_5.sm103.gdn import front

    o = _front_operands(7, pad=True)
    state0, phys0 = o["state"].clone(), o["phys"].clone()
    outs = _front_outputs()
    _run_front(o, front.pack(o["w"]), outs)
    torch.cuda.synchronize()
    assert outs["error"].item() == 0
    assert torch.equal(o["state"], state0) and torch.equal(o["phys"], phys0)


@pytest.mark.parametrize("pad", [False, True])
def test_front_takes_its_weights_by_bulk_copy_with_the_same_bytes(pad):
    """The default launch bulk-copies each projection CTA's weights into shared memory;
    every output and pool equals the launch that loads them into registers."""
    from sgl_nvidia_mega.qwen3_5.sm103.gdn import front

    for seed in range(3):
        o = _front_operands(20 + seed, pad=pad)
        packed = front.pack(o["w"])
        start = {n: o[n].clone() for n in ("state", "phys")}
        got = []
        for tma in (False, True):
            for n, t in start.items():
                o[n].copy_(t)
            outs = _front_outputs()
            for t in outs.values():  # a padded request leaves the conv rows unwritten
                t.zero_()
            _run_front(o, packed, outs, tma=tma)
            torch.cuda.synchronize()
            got.append({**outs, "state": o["state"].clone(), "phys": o["phys"].clone()})
        for n, t in got[0].items():
            assert torch.equal(t, got[1][n]), n


@pytest.mark.parametrize("pad", [False, True])
def test_front_holds_its_register_weights_with_the_same_bytes(pad):
    """The pinned launch (register weight loads held ahead of the PDL wait) writes every
    output and pool as the default bulk-copy launch, eager and from a CUDA graph."""
    from sgl_nvidia_mega.qwen3_5.sm103.gdn import front

    for seed in range(3):
        o = _front_operands(30 + seed, pad=pad)
        packed = front.pack(o["w"])
        start = {n: o[n].clone() for n in ("state", "phys")}
        got = []
        for kw in ({"pin": False}, {"pin": True}):
            for n, t in start.items():
                o[n].copy_(t)
            outs = _front_outputs()
            for t in outs.values():  # a padded request leaves the conv rows unwritten
                t.zero_()
            _run_front(o, packed, outs, **kw)
            torch.cuda.synchronize()
            assert outs["error"].item() == 0
            got.append({**outs, "state": o["state"].clone(), "phys": o["phys"].clone()})
        for n, t in got[0].items():
            assert torch.equal(t, got[1][n]), n
    # graph replay of the pinned launch
    for n, t in start.items():
        o[n].copy_(t)
    outs = _front_outputs()
    for t in outs.values():
        t.zero_()
    _run_front(o, packed, outs, pin=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run_front(o, packed, outs, pin=True)
    for n, t in start.items():
        o[n].copy_(t)
    graph.replay()
    torch.cuda.synchronize()
    for n, t in got[1].items():
        assert torch.equal(outs[n] if n in outs else o[n], t), n


def test_front_replays_from_a_cuda_graph():
    from sgl_nvidia_mega.qwen3_5.sm103.gdn import front

    o = _front_operands(11)
    packed = front.pack(o["w"])
    pools = ("state", "phys")
    start = {n: o[n].clone() for n in pools}
    outs = _front_outputs()
    _run_front(o, packed, outs)  # warm up (and compile) outside capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run_front(o, packed, outs)
    for seed in (12, 13):
        fresh = _front_operands(seed)
        o["x"].copy_(fresh["x"])
        for n in pools:
            o[n].copy_(start[n])
        graph.replay()
        torch.cuda.synchronize()
        got = {n: t.clone() for n, t in outs.items()}
        got.update({n: o[n].clone() for n in pools})
        for n in pools:
            o[n].copy_(start[n])
        _run_front(o, packed, outs)
        torch.cuda.synchronize()
        for n, t in outs.items():
            assert torch.equal(got[n], t), n
        for n in pools:
            assert torch.equal(got[n], o[n]), n


def test_front_norm_is_slot_gdn_verifys():
    from sgl_nvidia_mega.qwen3_5.gdn.front_norm import norm
    from sgl_nvidia_mega.qwen3_5.gdn.norm_fp8 import rms_norm_gated_fp8

    g = torch.Generator(device="cuda").manual_seed(5)
    core = torch.randn(DRAFT * HV, V, device="cuda", generator=g).to(torch.bfloat16)
    projected = torch.randn(DRAFT, 2560, device="cuda", generator=g).to(torch.bfloat16)
    z = projected[:, D:].view(DRAFT, HV, V)
    weight = torch.randn(V, device="cuda", generator=g).to(torch.bfloat16)
    scale = torch.tensor([0.02], device="cuda")
    quant = torch.empty(DRAFT * HV, V, dtype=torch.float8_e4m3fn, device="cuda")
    rstd = torch.empty(DRAFT * HV, dtype=torch.float32, device="cuda")
    norm(core, z, weight, scale, quant, rstd, 1e-6)
    want = rms_norm_gated_fp8(core, weight, 1e-6, z, scale, activation="silu")
    assert torch.equal(quant.view(torch.uint8), want.view(torch.uint8).view(-1, V))


# --------------------------------------------- TensorRT-LLM's in_proj front (TRTFRONT)

TRT_HEADS = {8: (2, 8)}  # (H, HV) per rank at TP8


def _trt_operands(seed, tp=8, pad=False, requests=1):
    """``requests`` requests of 7 rows at TP ``tp``: the operands of ``trt_front.front`` (the
    pools as in ``_front_operands``; ``pad`` gives the last request the pad state index -1)."""
    h, hv = TRT_HEADS[tp]
    d = 2 * h * K + hv * V
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rand(*shape, scale=1.0):
        return torch.randn(*shape, device="cuda", generator=g) * scale

    phys = rand(FRONT_SLOTS, d, 9).to(torch.bfloat16)
    return dict(
        d=d,
        x=rand(requests * DRAFT, 4096).to(torch.bfloat16),
        w=rand(d + hv * V, 4096, scale=0.05).clamp(-440, 440).to(torch.float8_e4m3fn),
        sx=torch.tensor([0.01], device="cuda"),
        sw=torch.tensor([0.002], device="cuda"),
        ba_w=rand(2 * hv, 4096, scale=0.02).to(torch.bfloat16),
        conv_w=rand(d, WIDTH, scale=0.5).to(torch.bfloat16),
        state=rand(FRONT_SLOTS, d, WIDTH - 1).to(torch.bfloat16),
        phys=phys,
        windows=phys.as_strided((FRONT_SLOTS, DRAFT, d, WIDTH - 1), (d * 9, 1, 9, 1)),
        states=torch.tensor(
            list(FRONT_STATES[: requests - 1])
            + [-1 if pad else FRONT_STATES[requests - 1]],
            dtype=torch.int32,
            device="cuda",
        ),
    )


def _trt_outputs(o):
    rows = o["x"].shape[0]
    return dict(
        out=torch.empty(rows, o["w"].shape[0], dtype=torch.bfloat16, device="cuda"),
        ba=torch.empty(rows, o["ba_w"].shape[0], dtype=torch.bfloat16, device="cuda"),
    )


def _run_trt_front(o, x8, outs, x=None):
    from sgl_nvidia_mega.qwen3_5.sm103.gdn import trt_front

    trt_front.prepare(
        o["w"].shape[0], o["ba_w"].shape[0], o["conv_w"], o["state"], o["windows"]
    )
    trt_front.front(
        o["x"] if x is None else x,
        x8,
        o["w"],
        o["sx"],
        o["sw"],
        o["ba_w"],
        o["conv_w"],
        o["state"],
        o["windows"],
        o["states"],
        outs["out"],
        outs["ba"],
    )


@pytest.mark.parametrize("requests", [1, 2, 3, 4])
@pytest.mark.parametrize("act", ["fp8", "quant"])
@pytest.mark.parametrize("tp", [8])
def test_trt_front_holds_its_projections_to_fp64_and_its_conv_is_sglangs(
    tp, act, requests
):
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
    from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8
    from sglang.srt.layers.quantization.fp8_utils import flashinfer_bmm_fp8

    errors = {n: [] for n in ("qkvz", "qkvz_stock", "ba", "ba_stock")}
    for seed in range(3):
        o = _trt_operands(60 + seed, tp, requests=requests)
        d, rows = o["d"], o["x"].shape[0]
        state0, phys0 = o["state"].clone(), o["phys"].clone()
        xq, _ = static_quant_fp8(o["x"], o["sx"], repeat_scale=False)
        outs = _trt_outputs(o)
        _run_trt_front(o, xq if act == "fp8" else None, outs)
        torch.cuda.synchronize()
        # the BF16 q / k / v rows the conv took: window row r holds 2 state columns, then request r's 7 rows
        qkv = torch.cat([o["phys"][r, :, 2:].t() for r in range(requests)])
        projected = torch.cat([qkv, outs["out"][:, d:]], dim=1)
        ref = (xq.double() @ o["w"].double().t()) * (
            o["sx"].double() * o["sw"].double()
        )
        stock = flashinfer_bmm_fp8(xq, o["w"].t(), o["sx"], o["sw"], torch.bfloat16)
        errors["qkvz"].append(_errors(projected, ref))
        errors["qkvz_stock"].append(_errors(stock, ref))
        ref = o["x"].double() @ o["ba_w"].double().t()
        errors["ba"].append(_errors(outs["ba"], ref))
        errors["ba_stock"].append(_errors(o["x"] @ o["ba_w"].t(), ref))
        # the conv on those rows, as SGLang's update (request r's windows at intermediate row r)
        state, phys = state0.clone(), phys0.clone()
        windows = phys.as_strided(o["windows"].shape, o["windows"].stride())
        want = causal_conv1d_update(
            qkv.contiguous().view(requests, DRAFT, d).transpose(1, 2),
            state,
            o["conv_w"],
            None,
            "silu",
            conv_state_indices=o["states"],
            intermediate_conv_window=windows,
            intermediate_state_indices=torch.arange(
                requests, dtype=torch.int32, device="cuda"
            ),
        )
        assert torch.equal(outs["out"][:, :d], want.transpose(1, 2).reshape(rows, d))
        assert torch.equal(o["state"], state) and torch.equal(o["phys"], phys)
    e = {n: torch.tensor(v, dtype=torch.float64) for n, v in errors.items()}
    print(
        f"T0 trt_front tp{tp} {act} requests={requests} qkvz mean rel-L2 {e['qkvz'][:, 0].mean():.4g} vs stock "
        f"{e['qkvz_stock'][:, 0].mean():.4g}, max abs {e['qkvz'][:, 1].max():.4g} vs "
        f"{e['qkvz_stock'][:, 1].max():.4g}; ba {e['ba'][:, 0].mean():.4g} vs "
        f"{e['ba_stock'][:, 0].mean():.4g}, {e['ba'][:, 1].max():.4g} vs "
        f"{e['ba_stock'][:, 1].max():.4g}",
        flush=True,
    )
    for name in ("qkvz", "ba"):
        got, stock = e[name], e[f"{name}_stock"]
        assert got[:, 0].mean() <= 1.05 * stock[:, 0].mean(), name
        assert got[:, 1].max() <= 2 * stock[:, 1].max(), name


@pytest.mark.parametrize("tp", [8])
def test_trt_front_leaves_a_padded_requests_pools_alone(tp):
    from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8

    o = _trt_operands(65, tp, pad=True)
    state0, phys0 = o["state"].clone(), o["phys"].clone()
    _run_trt_front(
        o, static_quant_fp8(o["x"], o["sx"], repeat_scale=False)[0], _trt_outputs(o)
    )
    torch.cuda.synchronize()
    assert torch.equal(o["state"], state0) and torch.equal(o["phys"], phys0)


_LATE_PRODUCER = r"""
extern "C" __global__ void late_producer(const uint4* src, uint4* dst, int n16, const int* slot_src,
                                         int* slot_dst, unsigned long long delay_ns) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  unsigned long long t0, t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  do {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  } while (t - t0 < delay_ns);
  for (int i = threadIdx.x; i < n16; i += blockDim.x) dst[i] = src[i];
  if (threadIdx.x == 0) *slot_dst = *slot_src;
}
"""


def test_trt_front_reads_its_rows_and_state_index_after_its_pdl_wait():
    """A producer that releases its dependents at entry and writes the front's BF16 / E4M3
    rows and its state index 20 ms later (well past the host's launch of the front): the
    front's outputs and pools are those of the written inputs. The build without the wait
    (TensorRT-LLM's positive control) reads the stale ones."""
    import ctypes

    from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8

    from sgl_nvidia_mega.common import jit
    from sgl_nvidia_mega.qwen3_5.sm103 import trt_lowm
    from sgl_nvidia_mega.qwen3_5.sm103.gdn import trt_front

    o = _trt_operands(66)  # first: the CUDA context the driver-API load below needs
    major, minor = torch.cuda.get_device_capability()
    cubin = jit.compile_source(
        _LATE_PRODUCER, "late_producer", f"sm_{major}{minor}a", ("-std=c++17",)
    )
    producer = jit.load(cubin, "late_producer", (256, 1, 1), 0)
    xq = static_quant_fp8(o["x"], o["sx"], repeat_scale=False)[0]
    # the inputs as one buffer: BF16 rows, E4M3 rows, and the state index
    nx = o["x"].numel() * 2
    src = torch.cat(
        [o["x"].view(torch.uint8).flatten(), xq.view(torch.uint8).flatten()]
    )
    dst = torch.zeros_like(src)
    x, x8 = (
        dst[:nx].view(torch.bfloat16).view(DRAFT, 4096),
        dst[nx:].view(torch.float8_e4m3fn).view(DRAFT, 4096),
    )
    slot_src, slot = (
        o["states"].clone(),
        torch.tensor([FRONT_STATES[1]], dtype=torch.int32, device="cuda"),
    )
    o["states"] = slot
    start = {n: o[n].clone() for n in ("state", "phys")}
    # both builds launched once first (a first launch loads the module: slower than the producer's delay)
    _run_trt_front(o, x8, _trt_outputs(o), x=x)
    skip = trt_front._SPECS["fp8", o["w"].shape[0], 1]._replace(unsafe_skip_wait=True)
    trt_lowm.prepare(skip)

    def run_skip(outs):
        trt_lowm.run(
            skip,
            out=outs["out"],
            out2=outs["ba"],
            x16=x,
            x8=x8,
            w_fp8=o["w"],
            weight_scale=o["sw"],
            input_scale=o["sx"],
            w_bf16=o["ba_w"],
            conv=(o["conv_w"], o["state"], slot, o["windows"]),
        )

    run_skip(_trt_outputs(o))
    torch.cuda.synchronize()
    want = None
    for late, skip_wait in ((False, False), (True, False), (True, True)):
        for n, t in start.items():
            o[n].copy_(t)
        dst.zero_()
        slot.fill_(FRONT_STATES[1])
        outs = _trt_outputs(o)
        torch.cuda.synchronize()
        args = [
            ctypes.c_void_p(src.data_ptr()),
            ctypes.c_void_p(dst.data_ptr()),
            ctypes.c_int(src.numel() // 16),
            ctypes.c_void_p(slot_src.data_ptr()),
            ctypes.c_void_p(slot.data_ptr()),
            ctypes.c_ulonglong(20_000_000 if late else 0),
        ]
        jit.launch(producer, (1, 1, 1), args, torch.cuda.current_stream().cuda_stream)
        if not late:
            torch.cuda.synchronize()
        if skip_wait:
            run_skip(outs)
        else:
            _run_trt_front(o, x8, outs, x=x)
        torch.cuda.synchronize()
        got = {**outs, "state": o["state"].clone(), "phys": o["phys"].clone()}
        if want is None:
            want = got
        elif skip_wait:
            assert not all(torch.equal(t, want[n]) for n, t in got.items())
        else:
            for n, t in got.items():
                assert torch.equal(t, want[n]), n


def test_trt_front_replays_from_a_cuda_graph_and_never_compiles_in_one():
    from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8

    from sgl_nvidia_mega.qwen3_5.sm103 import trt_lowm

    o = _trt_operands(67)
    x8 = static_quant_fp8(o["x"], o["sx"], repeat_scale=False)[0]
    pools = ("state", "phys")
    start = {n: o[n].clone() for n in pools}
    outs = _trt_outputs(o)
    _run_trt_front(o, x8, outs)  # prepare (compile) and run outside capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run_trt_front(o, x8, outs)
    for seed in (68, 69):
        fresh = _trt_operands(seed)
        o["x"].copy_(fresh["x"])
        x8.copy_(static_quant_fp8(o["x"], o["sx"], repeat_scale=False)[0])
        for n in pools:
            o[n].copy_(start[n])
        graph.replay()
        torch.cuda.synchronize()
        got = {n: t.clone() for n, t in outs.items()}
        got.update({n: o[n].clone() for n in pools})
        for n in pools:
            o[n].copy_(start[n])
        _run_trt_front(o, x8, outs)
        torch.cuda.synchronize()
        for n, t in outs.items():
            assert torch.equal(got[n], t), n
        for n in pools:
            assert torch.equal(got[n], o[n]), n
    # a launch that was not prepared is refused inside a capture, not compiled there
    unprepared = trt_lowm.spec(m=DRAFT, k=4096, n_bf16=16, tactic_bf16=(8, 1))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with pytest.raises(RuntimeError, match="before graph capture"):
            trt_lowm.prepare(unprepared)
        with pytest.raises(RuntimeError, match="not prepared"):
            trt_lowm.run(unprepared, out=outs["ba"], x16=o["x"], w_bf16=o["ba_w"])

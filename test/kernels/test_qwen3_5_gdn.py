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
"""

import pytest
import torch

H, K, HV, V = 2, 128, 8, 128  # per rank at TP8: 16 / 64 heads over 8
D = 2 * H * K + HV * V  # the conv's q / k / v channels
DRAFT, WIDTH, SLOTS = 7, 4, 9


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


# a covered step's requests (slot gdn_verify covers B1, B2 and B4) and their slots
REQUESTS = (1, 2, 4)
STATE_SLOTS = (4, 7, 0, 8)
RING_SLOTS = (3, 5, 0, 8)
REPLAY_SLOTS = (1, 2, 4, 6)


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
    ring_len, slots = 16, 9
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


# ------------------------------------------------------------------- slot gdn_front

FRONT_SLOTS = 5


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

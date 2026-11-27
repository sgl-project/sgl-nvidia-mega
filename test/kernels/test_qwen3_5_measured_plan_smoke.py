# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU",
)
# TP8's per-rank GatedDeltaNet heads and a 7-row verify
H, HV, K, V, T, RING = 2, 8, 128, 128, 7, 16
D = 2 * H * K + HV * V


def _bits(t):
    t = t.contiguous()
    return t.view(torch.int16) if t.element_size() == 2 else t.view(torch.int32)


def _rel(got, want):
    return float((got.double() - want.double()).norm() / want.double().norm())


def test_tep8_moe_engine_runs_once():
    from test_qwen3_5_moe import _ep_router_inputs, _experts_of, _local_layer, _same_moe

    from sgl_nvidia_mega.qwen3_5.moe import body
    from sgl_nvidia_mega.qwen3_5.sm103.moe import router

    m, local = 7, 128
    experts = _experts_of(_local_layer(local, 512, 17), 0)
    body._WIDTHS.clear()
    body.configure(False)
    probe = torch.empty((m, 4096), dtype=torch.bfloat16, device="cuda")
    width = body.width(experts, probe)
    x, w, scale = _ep_router_inputs(m, 7191, "random", (0, local))
    router.build_topk(x.device.index)
    router.build_big(x.device.index, local)
    logits = torch.nn.functional.linear(x, w)
    routed = router.route_topk_quant(logits, x, scale, tables=width.tables)
    x_fp4, x_sf = routed[2], routed[3].view(torch.float8_e4m3fn)
    want = [t.clone() for t in body.run(width, experts, x_fp4, x_sf)[:3]]
    got = body.run_engine(width, experts, x_fp4, x_sf)
    torch.cuda.synchronize()
    _same_moe(got, want, m)
    body._WIDTHS.clear()


def test_bt_tail_runs_once():
    pytest.importorskip("cutlass")
    from flashinfer.comm.mnnvl_cutedsl.kernel_bt.device_kernels import (
        STAGE_STATE_WORDS,
    )

    from sgl_nvidia_mega.qwen3_5.sm103.hop import bt_tail

    hidden, cap, m, stage, empty = 4096, 256, 7, 1, -32768  # empty: bf16 -0.0
    g = torch.Generator(device="cuda").manual_seed(7)
    bits = torch.randn(m, hidden, device="cuda", generator=g).bfloat16()
    bits = bits.view(torch.int16)
    rows = bits.masked_fill(bits == empty, 0).view(torch.bfloat16)
    gamma = (1 + 0.1 * torch.randn(hidden, device="cuda", generator=g)).bfloat16()
    box = torch.full((3, cap, hidden), empty, dtype=torch.int16, device="cuda")
    box[stage, :m] = rows.view(torch.int16)
    state = torch.zeros(STAGE_STATE_WORDS, dtype=torch.int32, device="cuda")
    state[:2] = torch.tensor([(stage + 1) % 3, stage], dtype=torch.int32)
    res = torch.empty(m, hidden, dtype=torch.bfloat16, device="cuda")
    norm = torch.empty_like(res)
    bt_tail.tail(
        bt_tail.kernels(torch.cuda.current_device()),
        prenorm=box.view(torch.bfloat16),
        stage_state=state,
        gamma=gamma,
        eps=1e-6,
        m=m,
        residual_out=res,
        norm_out=norm,
        fp8_scale=None,
        fp8_out=None,
        early=False,
    )
    torch.cuda.synchronize()
    assert torch.equal(res.view(torch.int16), rows.view(torch.int16))
    f = rows.float()
    ref = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + 1e-6) * gamma.float()
    assert ((norm.float() - ref).abs() <= ref.abs() * 2**-6).all()
    assert bool((box == empty).all())


def test_cp_lite_partial_runs_once():
    pytest.importorskip("cutlass")
    from sgl_nvidia_mega.qwen3_5.sm103.attention import q35_cp_fused as F

    hq, d, page, cp, ql, lens = 4, 256, 64, 4, 7, [9000, 7]
    dev = torch.device("cuda")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    F.q35_cp_fused_prepare(dev, sms)
    g = torch.Generator(device=dev).manual_seed(27)
    b, pages = len(lens), (max(lens) + page - 1) // page
    pool, rows = b * pages + 1, b * ql

    def fp8(*shape, s=2.0):
        x = torch.randn(*shape, device=dev, generator=g) * s
        return x.to(torch.float8_e4m3fn)

    k, v, q = fp8(pool, 1, page, d), fp8(pool, 1, page, d), fp8(rows, cp * hq, d, s=1.0)
    table = torch.randperm(pool - 1, device=dev, generator=g)[: b * pages]
    po = torch.zeros(cp, b, ql * hq, d, dtype=torch.bfloat16, device=dev)
    pl = torch.zeros(cp, b, ql * hq, device=dev)
    F.q35_cp_fused_partial(
        q.view(b, cp, ql * hq, d).permute(1, 0, 2, 3),
        k,
        table.int().view(b, pages).contiguous(),
        torch.tensor(lens, dtype=torch.int32, device=dev),
        1,
        0,
        cp,
        0.75 * d**-0.5,
        0.5,
        po,
        pl,
        enable_pdl=True,
        v_pool=v,
        len_sub=ql,
    )
    torch.cuda.synchronize()
    assert torch.isfinite(po[:, 0]).all() and bool(po[:, 0].ne(0).any())


def _verify_step(bs, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    slots, tokens = bs + 3, bs * T

    def rand(*shape, scale=1.0):
        x = torch.randn(*shape, device="cuda", generator=g) * scale
        return x.to(torch.bfloat16)

    def rows():
        return torch.randperm(slots, device="cuda", generator=g)[:bs].int()

    qkvz, ba = rand(tokens, D + HV * V), rand(tokens, 2 * HV)
    head = [qkvz[:, :D], qkvz[:, D:].view(tokens, HV, V), ba[:, HV:], ba[:, :HV]]
    head += [torch.rand(HV, device="cuda", generator=g) for _ in range(2)]
    head.append(rand(slots, HV, V, K, scale=0.05))
    ring = [
        rand(slots, HV, RING, V, scale=0.05),
        rand(slots, H, RING, K, scale=0.09),
        -torch.rand(slots, HV, RING, device="cuda", generator=g) * 0.5,
        rand(slots, HV, RING, V, scale=1e-4),
        rand(slots, H, RING, K, scale=1e-4),
    ]
    tail = [rows(), rows()]
    tail.append(torch.randint(0, RING - T + 1, (slots,), device="cuda", generator=g))
    tail.append(torch.randint(0, RING, (slots,), device="cuda", generator=g))
    tail[2:] = [t.int() for t in tail[2:]]
    tail.append(torch.zeros(slots, dtype=torch.int8, device="cuda"))
    tail += [rand(V), torch.tensor([0.02], device="cuda"), 1e-6]
    return g, rand, head, ring, tail


def test_gdn_conv_core_runs_once():
    from sgl_nvidia_mega.qwen3_5.gdn.conv import conv_update
    from sgl_nvidia_mega.qwen3_5.sm103.gdn.core import Conv, core

    bs = 2
    g, rand, head, ring, tail = _verify_step(bs, 1016)
    slots, tokens = bs + 3, bs * T
    weight, state = rand(D, 4, scale=0.5), rand(slots, D, 3)
    shared = rand(slots, D, T + 2)
    windex = torch.randperm(slots, device="cuda", generator=g)[:bs].int()

    def windows(t):
        return t.as_strided((slots, T, D, 3), (t.stride(0), 1, t.stride(1), 1))

    def launch(qkv, conv):
        out = [t.clone() for t in ring]
        o = torch.zeros(tokens, HV, V, dtype=torch.bfloat16, device="cuda")
        y = torch.zeros(tokens, HV * V, dtype=torch.float8_e4m3fn, device="cuda")
        core(
            *[qkv, *head[1:]],
            *out,
            *tail,
            y,
            rows_per_request=T,
            num_k_heads=H,
            o_out=o,
            conv=conv,
        )
        return [y.view(torch.uint8), o, *out]

    st_s, sh_s = state.clone(), shared.clone()
    y = conv_update(
        head[0].view(bs, T, -1).transpose(1, 2),
        st_s,
        weight,
        None,
        "silu",
        tail[0],
        windows(sh_s),
        windex,
        block_n=128,
        token_major_out=True,
        hoist=True,
        window_mode=1,
    )
    want = launch(y.transpose(1, 2).view(tokens, -1), None) + [st_s, sh_s]
    st_f, sh_f = state.clone(), shared.clone()
    arrive = torch.zeros(bs * H, dtype=torch.int32, device="cuda")
    conv = Conv(weight, None, st_f, windows(sh_f), windex, arrive)
    got = launch(head[0], conv) + [st_f, sh_f]
    torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(got, want, strict=True))


def test_draft_moe_engine_runs_once():
    from flashinfer.fused_moe import trtllm_bf16_moe
    from flashinfer.fused_moe.core import (
        ActivationType,
        _maybe_get_cached_w3_w1_permute_indices,
        convert_to_block_layout,
        get_w2_permute_indices_with_cache,
    )

    from sgl_nvidia_mega.qwen3_5.draft.draftmoe import DraftMoeEngine

    e_all, e, h, inter, top_k, m = 512, 128, 4096, 512, 10, 16
    g = torch.Generator(device="cuda").manual_seed(11)

    def block(shape, gated):
        w = (torch.randn(e, *shape, device="cuda", generator=g) * 0.02).bfloat16()
        cache, out = {}, []
        for i in range(e):
            t = w[i].contiguous().view(torch.uint8)
            idx = (
                _maybe_get_cached_w3_w1_permute_indices(
                    cache, t, 128, is_gated_act_gemm=True
                )
                if gated
                else get_w2_permute_indices_with_cache(cache, t, 128)
            )
            t = convert_to_block_layout(t[idx.cuda()].contiguous(), 128)
            out.append(t.view(torch.bfloat16))
        return torch.stack(out).contiguous()

    w13, w2 = block((2 * inter, h), True), block((h, inter), False)
    x = torch.randn(m, h, device="cuda", generator=g).bfloat16()
    logits = (torch.randn(m, e_all, device="cuda", generator=g) * 2).bfloat16()
    want = trtllm_bf16_moe(
        routing_logits=logits,
        routing_bias=None,
        hidden_states=x,
        gemm1_weights=w13,
        gemm2_weights=w2,
        num_experts=e_all,
        top_k=top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=inter,
        local_expert_offset=0,
        local_num_experts=e,
        routing_method_type=1,  # renormalize
        routed_scaling_factor=None,
        tune_max_num_tokens=m,
        activation_type=ActivationType.Swiglu.value,
        do_finalize=False,
    )
    experts = SimpleNamespace(
        w13_weight=w13, w2_weight=w2, moe_ep_rank=0, num_local_experts=e
    )
    engine = DraftMoeEngine(experts, wide=True)
    assert engine.takes(x, logits)
    got = engine.run(x, logits)
    weights = [t[1].reshape(m, top_k).view(torch.int16) for t in (want, got)]
    assert torch.equal(*weights)
    ia, ib = (t[2].reshape(m, top_k).long() for t in (want, got))
    assert torch.equal(ia < 0, ib < 0)
    both = (ia >= 0) & (ib >= 0)
    assert torch.equal(_bits(want[0][ia[both]]), _bits(got[0][ib[both]]))


def test_gumbel_fast_runs_once():
    from sglang.srt.speculative.sharded_verify import VocabShard, shard_keys

    from sgl_nvidia_mega.qwen3_5.sm103.sharded_gumbel import gumbel_shard_keys_fast

    vocab, local, splits = 1024, 128, 4
    g = torch.Generator().manual_seed(1)
    x = (torch.randn(6, local, generator=g) * 3).bfloat16().cuda()
    temps = torch.tensor([1.0, 0.7, 0.0, 1.0, -1.0, 2.0], device="cuda")
    seeds = torch.arange(6, dtype=torch.int64, device="cuda") * 977 + 5
    greedy = torch.empty(6, splits, dtype=torch.int64, device="cuda")
    keys = gumbel_shard_keys_fast(
        x,
        temps,
        seeds,
        vocab_start=0,
        valid=local,
        vocab_size=vocab,
        splits=splits,
        greedy_out=greedy,
    )
    want = shard_keys(x, VocabShard(start=0, valid=local, vocab_size=vocab), splits)
    assert torch.equal(greedy, want)
    # rows at T <= 0 keep the plain key
    assert torch.equal(keys[2], want[2]) and torch.equal(keys[4], want[4])


def test_gdn_commit_runs_once():
    from sglang.kernels.ops.attention.linear import gdn_replayssm_spec_decode as stock

    from sgl_nvidia_mega.qwen3_5.sm103.gdn import commit

    layers, d = 2, 128
    g = torch.Generator(device="cuda").manual_seed(5)
    state = torch.randn(layers, 6, HV, d, d, device="cuda", generator=g) * 0.05
    state = state.bfloat16()
    rings = [
        torch.randn(layers, 4, n, RING, d, device="cuda", generator=g).bfloat16() * s
        for n, s in ((HV, 0.2), (HV, 1e-3), (H, 0.1), (H, 1e-3))
    ]
    rings.append(-torch.rand(layers, 4, HV, RING, device="cuda", generator=g) * 0.2)
    sidx = torch.tensor([2, -1], dtype=torch.int32, device="cuda")  # one padded row
    rows = torch.tensor([1, 0], dtype=torch.int64, device="cuda")
    acc = torch.tensor([4, 1], dtype=torch.int32, device="cuda")
    cursors = []
    for _ in range(2):
        c = [torch.zeros(4, dtype=torch.int32, device="cuda") for _ in range(2)]
        c.append(torch.zeros(4, dtype=torch.int8, device="cuda"))
        stock.commit_gdn_replayssm_spec(
            write_pos=c[0],
            cache_base=c[1],
            is_flush=c[2],
            num_accepted=acc,
            replay_indices=rows,
            state_batch_indices=sidx,
            max_cache_len=RING,
            max_spec_len=T,
            fold_every_commit=True,
            null_block_id=-1,
        )
        cursors.append(c)
    want, got = state.clone(), state.clone()
    stock.commit_gdn_replayssm_circular(
        checkpoint_state=want,
        d_cache=rings[0],
        k_cache=rings[2],
        g_cache=rings[4],
        d_residual_cache=rings[1],
        k_residual_cache=rings[3],
        state_batch_indices=sidx,
        replay_indices=rows,
        write_pos=cursors[0][0],
        cache_base=cursors[0][1],
        is_flush=cursors[0][2],
        accept_lens=acc,
        mamba_track_indices=None,
        mamba_steps_to_track=None,
        null_block_id=-1,
    )
    commit.commit(
        checkpoint=got,
        ring_d=rings[0],
        ring_dlo=rings[1],
        ring_k=rings[2],
        ring_klo=rings[3],
        ring_g=rings[4],
        state_indices=sidx,
        ring_rows=rows,
        write_pos=cursors[1][0],
        cache_base=cursors[1][1],
        is_flush=cursors[1][2],
        accept_lens=acc,
        track_indices=None,
        track_steps=None,
    )
    torch.cuda.synchronize()
    assert _rel(got[:, 2], want[:, 2]) < 2e-2
    for s in (0, 1, 3, 4, 5):
        assert torch.equal(_bits(got[:, s]), _bits(state[:, s]))


def test_gdn_core_tc_runs_once():
    pytest.importorskip("cutlass")
    from sgl_nvidia_mega.qwen3_5.gdn import replayssm
    from sgl_nvidia_mega.qwen3_5.sm103.gdn.core_tc import core_tc

    bs = 16
    _, _, head, ring, tail = _verify_step(bs, 112)
    qkv, _, a, b, A_log, dt_bias, state = head
    tokens = bs * T
    want = torch.zeros(tokens, HV, V, dtype=torch.bfloat16, device="cuda")
    replayssm.verify(
        qkv[:, : H * K].view(tokens, H, K),
        qkv[:, H * K : 2 * H * K].view(tokens, H, K),
        qkv[:, 2 * H * K :].view(tokens, HV, V),
        a,
        b,
        A_log,
        dt_bias,
        want,
        state,
        *[t.clone() for t in ring],
        None,
        torch.arange(0, tokens + 1, T, dtype=torch.int32, device="cuda"),
        tail[0],
        tail[1].long(),
        *tail[2:5],
        max_cache_len=RING,
        max_spec_len=T,
        scale=K**-0.5,
        null_block_id=-1,
        block_v=32,
        num_warps=2,
    )
    got = torch.zeros_like(want)
    y = torch.empty(tokens, HV * V, dtype=torch.float8_e4m3fn, device="cuda")
    core_tc(
        *head,
        *[t.clone() for t in ring],
        *tail,
        y,
        rows_per_request=T,
        num_k_heads=H,
        early_dependents=False,
        o_out=got,
    )
    torch.cuda.synchronize()
    assert _rel(got, want) < 2e-2


def test_gdn_prefill_front_runs_once():
    from sglang.kernels.ops.attention.fla.l2norm import gdn_prefill_qkv_prepare_fwd
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn

    from sgl_nvidia_mega.qwen3_5.sm103.gdn import prefill_front

    nk, nv, d, lens = 16, 64, 128, [1, 63, 129, 700]
    cols, tokens = (2 * nk + nv) * d, sum(lens)
    g = torch.Generator(device="cuda").manual_seed(3)
    mixed = (torch.randn(tokens, cols, device="cuda", generator=g) * 1.5).bfloat16()
    w = (torch.randn(cols, 4, device="cuda", generator=g) * 0.35).bfloat16()
    pool = (torch.randn(48, cols, 3, device="cuda", generator=g) * 1.5).bfloat16()
    cidx = torch.tensor([5, 17, 3, 40], dtype=torch.int32, device="cuda")
    qsl = torch.tensor([0, 1, 64, 193, 893], dtype=torch.int32, device="cuda")
    hinit = torch.tensor([True, False, True, True], device="cuda")
    p_s, p_f = pool.clone(), pool.clone()
    out = causal_conv1d_fn(
        mixed.transpose(0, 1),
        w,
        None,
        activation="silu",
        conv_states=p_s,
        has_initial_state=hinit,
        cache_indices=cidx,
        query_start_loc=qsl,
        seq_lens_cpu=lens,
    ).transpose(0, 1)[:tokens]
    qkv = torch.split(out, [nk * d, nk * d, nv * d], dim=-1)
    want = gdn_prefill_qkv_prepare_fwd(*(t.view(tokens, -1, d) for t in qkv))
    assert prefill_front.eligible(mixed, w, None, p_f, qsl, cidx) == ""
    got = prefill_front.gdn_prefill_front(
        mixed,
        w,
        None,
        p_f,
        qsl,
        cidx,
        hinit,
        num_k_heads=nk,
        num_v_heads=nv,
        activation="silu",
    )
    torch.cuda.synchronize()
    for a, b in (*zip(want, got, strict=True), (p_s, p_f)):
        assert torch.equal(_bits(a), _bits(b))

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DeepSeek-V4.1's wo_b publish (hop A) and two-shot hops on 4 SM100-family GPUs (GB300)
with NVLink multicast. Hop B runs in test_deepseek_v41_hop_b.py: it registers its own
SGLang one-shot plane, so it keeps its own process.

- wo_b publishing into the Lamport all-reduce + mHC from its GEMM epilogue, per split-K
  tactic and row count: the published GEMM's output equals FlashInfer's low-latency GEMM
  with the same tactic (past 8 rows, on row-tile scales, the same kernel launched
  directly by mxfp8_ll), and the consumer's outputs equal the separate publish kernel's,
  bit for bit, over boundary sequences that cycle the three mailbox generations, mix
  both paths, delay one rank in turn and replay from a CUDA graph; the consumer also
  matches a torch all-reduce + mHC post + collapse + RMSNorm.
- The two-shot all-reduce + mHC hops (TwoShotMHC) at 49-384 rows: the attention hop, the
  MoE hop with and without the MXFP8 quant, and the closed-seam MoE hop. Bit for bit
  against SGLang's two-shot kernel when the SGLang tree on PYTHONPATH has it, and
  against SGLang's one-shot fused kernel (streams and quant bytes; the normalized row is
  reported, and its mismatches must stay within one BF16 ulp); within tolerance of a
  torch reference; unchanged over 50 back-to-back calls of mixed row counts and over 20
  replays of one CUDA graph with a late rank.

The two tests share one torchrun process, in order: two_shot must run after
wo_b_publish, because two_shot registers SGLang's comm plane process-wide.

Run with ``python test/kernels/test_deepseek_v41_hops.py --num-gpu 4 -s``.
"""

import math
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from sglang.srt.distributed import parallel_state as ps
from sglang.test.kernels.utils import multigpu_pytest_main
from sglang.test.test_utils import publish_build_topology

from sgl_nvidia_mega.deepseek_v41.spec import NUMERICS

pytestmark = pytest.mark.skipif(
    "RANK" not in os.environ,
    reason="run via torchrun: python test/kernels/test_deepseek_v41_hops.py --num-gpu 4 -s",
)

# ----------------------------------------------------------------- wo_b publish (hop A)

HIDDEN, HC, K = 5120, 4, 2048


@pytest.fixture(scope="module")
def rank():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    ps.set_custom_all_reduce(False)
    ps.init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
    )
    publish_build_topology(tp_size=world_size, world_rank=rank)
    ps.initialize_model_parallel()
    yield rank
    ps.destroy_model_parallel()
    ps.destroy_distributed_environment()


def _linear(device, gen):
    from flashinfer import mxfp8_quantize

    w = torch.randn(HIDDEN, K, device=device, generator=gen) / math.sqrt(K)
    w_q, w_sf = mxfp8_quantize(w.bfloat16(), is_sf_swizzled_layout=True)
    return SimpleNamespace(
        weight=w_q.contiguous(),
        weight_scale_inv_swizzled=w_sf.reshape(-1).contiguous(),
        bias=None,
    )


def _activation(rows, device, gen):
    from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize

    x = torch.randn(rows, K, device=device, generator=gen).bfloat16()
    if rows > 1:
        # A zero row: zero outputs, whose negative zeros the publish clears.
        x[-1] = 0
    q, sf = flashinfer_mxfp8_quantize(x, is_sf_swizzled_layout=True, alignment=32)
    return q, sf.reshape(-1)


def _row_tile_activation(rows, device, gen):
    """Rows past 8 as the fused wo_a writes them for hop A: each 8-row tile quantized on
    its own, its scales the first 128-row atom of FlashInfer's 128x4 swizzle."""
    from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize

    from sgl_nvidia_mega.deepseek_v41.sm103.mxfp8_ll import RowTileMxfp8

    x = torch.randn(rows, K, device=device, generator=gen).bfloat16()
    x[-1] = 0
    codes, scales = [], []
    for start in range(0, rows, 8):
        q, sf = flashinfer_mxfp8_quantize(
            x[start : start + 8].contiguous(), is_sf_swizzled_layout=True, alignment=32
        )
        codes.append(q)
        scales.append(sf.reshape(-1)[: K // 128 * 512])
    return RowTileMxfp8(torch.cat(codes), torch.cat(scales))


def _flashinfer_gemm(linear, qa, tactic):
    """FlashInfer's own low-latency GEMM with ``tactic`` (output, valid tactics); past 8
    rows, which its runner refuses, the same kernel launched by mxfp8_ll."""
    from flashinfer.gemm import gemm_base

    from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll

    a, a_scale = qa
    if a.shape[0] > mxfp8_ll.MAX_ROWS:
        out = mxfp8_ll.linear(
            a, a_scale, linear.weight, linear.weight_scale_inv_swizzled, tactic=tactic
        )
        return out, mxfp8_ll.offered_tactics(HIDDEN, K, mxfp8_ll.MAX_ROWS)
    out = torch.empty(a.shape[0], HIDDEN, dtype=torch.bfloat16, device=a.device)
    major, minor = torch.cuda.get_device_capability(a.device)
    runner = gemm_base._cutedsl_low_latency_blockscaled_gemm_runner(
        major * 10 + minor, True
    )
    workspace = gemm_base._get_cache_buf(
        "mm_mxfp8_workspace", gemm_base.DEFAULT_WORKSPACE_SIZE, a.device
    )
    inputs = [
        a,
        linear.weight.t(),
        a_scale,
        linear.weight_scale_inv_swizzled,
        torch.bfloat16,
        out,
        workspace,
    ]
    runner.forward(inputs, tactic=tactic)
    return out, runner.get_valid_tactics(inputs, None)


def _mixes(rows, device, rank):
    gen = torch.Generator(device=device).manual_seed(7 + rank)
    return (
        torch.randn(rows, HC, HIDDEN, device=device, generator=gen).bfloat16(),
        torch.rand(rows, HC, device=device, generator=gen),
        torch.randn(rows, HC, HC, device=device, generator=gen) * 0.1,
        torch.rand(rows, HC, device=device, generator=gen),
        torch.randn(HIDDEN, device=device, generator=gen).bfloat16(),
    )


def _torch_boundary(x, mix, group):
    residual, post, comb, pre, norm_weight = mix
    y = x.float()
    dist.all_reduce(y, group=group)
    y = y.bfloat16().float()
    streams = post.unsqueeze(-1) * y.unsqueeze(1) + (
        comb.unsqueeze(-1) * residual.float().unsqueeze(2)
    ).sum(dim=1)
    streams = streams.bfloat16()
    collapsed = (pre.unsqueeze(-1) * streams.float()).sum(dim=1)
    normed = collapsed * torch.rsqrt(collapsed.square().mean(-1, keepdim=True) + 1e-6)
    return streams, (normed * norm_weight.float()).bfloat16()


@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10),
    reason="Requires SM100/SM103",
)
def test_wo_b_publish_matches_separate_publish(rank):
    from sgl_nvidia_mega.deepseek_v41.sm103.all_reduce_mhc import AllReduceMHC
    from sgl_nvidia_mega.deepseek_v41.sm103.wo_b_publish import PublishingMxfp8Linear

    tp_size = ps.get_tp_group().world_size
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    with torch.inference_mode():
        arm = AllReduceMHC(
            hidden_size=HIDDEN,
            hc_mult=HC,
            max_num_tokens=16,
            top_k=6,
            device=device,
            group=ps.get_tp_group().device_group,
        )
        publisher = PublishingMxfp8Linear(arm)
        gen = torch.Generator(device=device).manual_seed(1000 + rank)
        layers = [_linear(device, gen) for _ in range(2)]

        def boundary(x, mix):
            residual, post, comb, pre, norm_weight = mix
            return arm(x, residual, post, comb, pre, norm_weight, 1e-6)

        # Which path each boundary of a sequence takes (True: published). The
        # sequence crosses the three generations several times on both paths.
        pattern = (True, True, True, False, True, False, False, True, True)
        for rows in (1, 6, 8, 12, 16):
            make = _activation if rows <= 8 else _row_tile_activation
            qas = [make(rows, device, gen) for _ in layers]
            mix = _mixes(rows, device, rank)
            assert publisher.supports(layers[0], rows)
            assert publisher.prepare(layers[0], qas[0]) is not None
            if rows > 8:
                # a 128x4 pair past 8 rows stays on the linear's own path
                assert publisher.prepare(layers[0], tuple(qas[0])) is None
            _, valid = _flashinfer_gemm(layers[0], qas[0], None)
            by_split = {}
            for tactic in valid:
                by_split.setdefault(tactic[3], tuple(tactic))
            # The default tactic (what an untuned mm_mxfp8 runs) and the first
            # without split-K and with split-K 4 (both epilogue branches).
            tactics = list(
                dict.fromkeys(
                    [tuple(valid[0])] + [by_split[s] for s in (1, 4) if s in by_split]
                )
            )
            for tactic in tactics:
                expected_c = [
                    _flashinfer_gemm(layer, qa, tactic)[0]
                    for layer, qa in zip(layers, qas)
                ]
                for late in range(tp_size):
                    for index, published in enumerate(pattern):
                        li = index % len(layers)
                        if rank == late:
                            torch.cuda._sleep(20_000)
                        if published:
                            prepared = publisher.prepare(layers[li], qas[li])
                            x = publisher(layers[li], prepared, tactic=tactic)
                            assert torch.equal(
                                x.view(torch.int16), expected_c[li].view(torch.int16)
                            ), f"C differs: rows={rows} tactic={tactic}"
                            got = boundary(x, mix)
                        else:
                            got = boundary(expected_c[li].clone(), mix)
                        # The separate publish kernel on the same C, one boundary
                        # later, so the generations must line up across paths.
                        reference = boundary(expected_c[li].clone(), mix)
                        for a, b in zip(got, reference):
                            assert torch.equal(
                                a.view(torch.int16), b.view(torch.int16)
                            ), (
                                f"boundary differs: rows={rows} tactic={tactic} "
                                f"late={late} step={index}"
                            )
                torch.cuda.synchronize()

            # Against torch: the math SGLang's own fused attention boundary runs.
            c = _flashinfer_gemm(layers[0], qas[0], tactics[0])[0]
            streams, normed = boundary(
                publisher(
                    layers[0],
                    publisher.prepare(layers[0], qas[0]),
                    tactic=tactics[0],
                ),
                mix,
            )
            ref_streams, ref_normed = _torch_boundary(
                c, mix, ps.get_tp_group().device_group
            )
            torch.testing.assert_close(
                streams.float(), ref_streams.float(), atol=3e-2, rtol=2e-2
            )
            torch.testing.assert_close(
                normed.float(), ref_normed.float(), atol=6e-2, rtol=3e-2
            )

            # One CUDA graph of published and separate boundaries, replayed.
            tactic = tactics[0]
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
                captured = []
                for index, published in enumerate(pattern):
                    li = index % len(layers)
                    if published:
                        prepared = publisher.prepare(layers[li], qas[li])
                        x = publisher(layers[li], prepared, tactic=tactic)
                    else:
                        x = _flashinfer_gemm(layers[li], qas[li], tactic)[0]
                    captured.append(boundary(x, mix))
            torch.cuda.current_stream().wait_stream(stream)
            for _ in range(3):
                graph.replay()
                torch.cuda.synchronize()
                for index, outputs in enumerate(captured):
                    li = index % len(layers)
                    reference = boundary(
                        _flashinfer_gemm(layers[li], qas[li], tactic)[0], mix
                    )
                    for a, b in zip(outputs, reference):
                        assert torch.equal(a.view(torch.int16), b.view(torch.int16))
            torch.cuda.synchronize()


# ------------------------------------------------------------------------ two-shot hops

HIDDEN, HC, TOP_K, EPS = 5120, 4, 6, 1e-6
ROWS = (49, 64, 96, 128, 152, 192, 256, 384)
ONE_SHOT_QUANT_ROWS = (
    128  # the one-shot quant epilogue addresses one 128-row scale tile
)


def _inputs(rows, device, gen):
    p = rows * TOP_K
    idx = torch.randperm(p, device=device, generator=gen).int()
    idx[::7] = -1  # dropped slots
    weights = torch.rand(rows, TOP_K, device=device, generator=gen)
    shared = torch.randn(rows, HIDDEN, device=device, generator=gen).bfloat16() / 8
    # an all-zero row: its +0 words must be sent as -0, not taken for empty slots
    weights[0] = 0
    shared[0] = 0
    return SimpleNamespace(
        gemm2=torch.randn(p, HIDDEN, device=device, generator=gen).bfloat16() / 8,
        idx=idx,
        weights=weights,
        shared=shared,
        x=torch.randn(rows, HIDDEN, device=device, generator=gen).bfloat16() / 4,
        residual=torch.randn(rows, HC, HIDDEN, device=device, generator=gen).bfloat16(),
        post=torch.rand(rows, HC, device=device, generator=gen),
        comb=torch.randn(rows, HC, HC, device=device, generator=gen) * 0.1,
        pre=torch.rand(rows, HC, device=device, generator=gen),
        norm_weight=torch.randn(HIDDEN, device=device, generator=gen).bfloat16(),
    )


def _run(hop, a, variant):
    """Our hop: the outputs as a tuple of tensors."""
    if variant == "attn":
        return hop.attn_post(a.x, a.residual, a.post, a.comb, a.pre, a.norm_weight, EPS)
    moe = (a.gemm2, a.idx, a.weights, TOP_K, a.shared, a.residual, a.post, a.comb)
    if variant == "post":
        return (hop.moe_post_only(*moe),)
    if variant == "moe":
        return hop.moe_post(*moe, a.pre, a.norm_weight, EPS)
    rows = a.weights.shape[0]
    q = torch.empty(rows, HIDDEN, dtype=torch.float8_e4m3fn, device=a.x.device)
    sf = torch.empty(hop.scale_bytes(rows), dtype=torch.uint8, device=a.x.device)
    updated, normalized = hop.moe_post(
        *moe, a.pre, a.norm_weight, EPS, mxfp8_out=(q, sf)
    )
    return updated, normalized, q, sf


def _sglang(a, variant, world, two_shot):
    """SGLang's kernel, outputs in _run's order; None where it does not apply."""
    if two_shot:
        from sglang.kernels.ops.communication import all_reduce_mhc_highbatch as m

        post_only = m.moe_finalize_all_reduce_mhc_post_2shot
        norm = m.moe_finalize_all_reduce_mhc_post_combine_norm_2shot
        attn = m.all_reduce_mhc_post_combine_norm_2shot
        quant_rows = m.MAX_ROWS
    else:
        from sglang.kernels.ops.communication import all_reduce_mhc as m

        post_only = m.moe_finalize_all_reduce_mhc_post
        norm = m.moe_finalize_all_reduce_mhc_post_combine_norm
        attn = m.all_reduce_mhc_post_combine_norm
        quant_rows = ONE_SHOT_QUANT_ROWS
    mix = (a.residual, a.post, a.comb)
    moe = (a.gemm2, a.idx, a.weights, TOP_K, a.shared, *mix)
    if variant == "attn":
        return attn(a.x, *mix, a.pre, a.norm_weight, EPS, world_size=world)[1:]
    if variant == "post":
        return post_only(*moe, world_size=world)[1:]
    if variant == "moe":
        return norm(*moe, a.pre, a.norm_weight, EPS, world_size=world)[1:]
    if a.weights.shape[0] > quant_rows:
        return None
    if two_shot:
        return norm(*moe, a.pre, a.norm_weight, EPS, world_size=world, quant=True)[1:]
    return m.moe_finalize_all_reduce_mhc_post_combine_norm_quant(
        *moe, a.pre, a.norm_weight, EPS, world_size=world
    )[1:]


def _torch(a, variant, group, world):
    """FP32 finalize, rank-order sum of the BF16 partials, then post / collapse / norm."""
    rows = a.weights.shape[0]
    if variant == "attn":
        part = a.x
    else:
        g = a.gemm2.float()
        acc = torch.zeros(rows, HIDDEN, device=g.device)
        idx = a.idx.view(rows, TOP_K).long()
        for k in range(TOP_K):
            valid = (idx[:, k] >= 0).unsqueeze(1)
            contrib = g[idx[:, k].clamp(min=0)] * a.weights[:, k : k + 1]
            acc += torch.where(valid, contrib, 0.0)
        part = (acc.bfloat16().float() + a.shared.float()).bfloat16()
    parts = [torch.empty_like(part) for _ in range(world)]
    dist.all_gather(parts, part.contiguous(), group=group)
    y = parts[0].float()
    for p in parts[1:]:
        y = y + p.float()
    y = y.bfloat16().float()
    streams = a.post.unsqueeze(-1) * y.unsqueeze(1) + (
        a.comb.unsqueeze(-1) * a.residual.float().unsqueeze(2)
    ).sum(dim=1)
    streams = streams.bfloat16()
    if variant == "post":
        return (streams,)
    collapsed = (a.pre.unsqueeze(-1) * streams.float()).sum(dim=1).bfloat16().float()
    normed = collapsed * torch.rsqrt(collapsed.square().mean(-1, keepdim=True) + EPS)
    return streams, (normed * a.norm_weight.float()).bfloat16()


def _scale_offsets(rows, device):
    r = torch.arange(rows, device=device)[:, None]
    g = torch.arange(HIDDEN // 32, device=device)[None, :]
    return (
        (r // 128) * (HIDDEN // 128) * 512
        + (g // 4) * 512
        + ((r % 32) * 4 + (r // 32) % 4) * 4
        + g % 4
    )


def _check_quant(normalized, q, sf):
    """MXFP8 of the normalized row: dequantizes within FP8 E4M3 rounding; pad rows zero."""
    rows = normalized.shape[0]
    offsets = _scale_offsets(rows, normalized.device)
    exponent = sf[offsets].int()
    scale = torch.exp2((exponent - 127).float()).repeat_interleave(32, dim=1)
    dequant = q.float() * scale
    ref = normalized.float()
    assert ((dequant - ref).abs() <= ref.abs() / 16 + scale / 512).all()
    pad = torch.ones(sf.numel(), dtype=torch.bool, device=sf.device)
    pad[offsets.flatten()] = False
    assert not sf[pad].any()


def _bits(t):
    return t.contiguous().view(torch.uint8)


def _equal(a, b):
    return a.shape == b.shape and torch.equal(_bits(a), _bits(b))


def _bf16_ulps(a, b):
    """Max distance in BF16 ulps (same-sign values; ordered bit patterns)."""
    ia = a.contiguous().view(torch.int16).int()
    ib = b.contiguous().view(torch.int16).int()
    ia = torch.where(ia < 0, -(ia & 0x7FFF), ia)
    ib = torch.where(ib < 0, -(ib & 0x7FFF), ib)
    return int((ia - ib).abs().max())


def _have_two_shot_ref() -> bool:
    try:
        from sglang.kernels.ops.communication import (  # noqa: F401
            all_reduce_mhc_highbatch,
        )
        from sglang.srt.layers.communication import hop_highbatch  # noqa: F401
    except ImportError:
        return False
    return True


def _register_references(device):
    """SGLang's one-shot plane (sized for 384 rows) and, if present, its two-shot plane."""
    from sglang.kernels.ops.communication import all_reduce_fusion
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )

    cpu_group = ps.get_tp_group().cpu_group
    plane = CustomAllReduceV2(
        cpu_group,
        device,
        max_pull_size=0,
        max_pull_blocks=0,
        max_push_size=4 << 20,
        max_push_blocks=512,
    )
    if plane.disabled:
        return None, False
    all_reduce_fusion.register_comm(plane.obj)
    two_shot = _have_two_shot_ref()
    if two_shot:
        from sglang.srt.layers.communication import hop_highbatch

        assert hop_highbatch.register_highbatch_push_plane(
            SimpleNamespace(group=cpu_group, device=device)
        )
    return plane, two_shot


def _max_over_ranks(value: float) -> float:
    t = torch.tensor([value], dtype=torch.float64, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=ps.get_tp_group().device_group)
    return float(t.item())


VARIANTS = ("attn", "moe", "quant", "post")


@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10),
    reason="Requires SM100/SM103",
)
def test_two_shot_mhc(rank):
    from sgl_nvidia_mega.deepseek_v41.sm103.two_shot_mhc import MAX_ROWS, TwoShotMHC

    tp = ps.get_tp_group()
    world = tp.world_size
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    # a live deadline: the polls run the timer path without faulting
    log = print if rank == 0 else (lambda *_, **__: None)
    with torch.inference_mode():
        one_shot_plane, two_shot_ref = _register_references(device)
        hop = TwoShotMHC(
            hidden_size=HIDDEN,
            hc_mult=HC,
            max_num_tokens=MAX_ROWS,
            top_k=TOP_K,
            device=device,
            group=tp.device_group,
            deadline_s=30.0,
        )
        assert hop.fits(49, HIDDEN) and hop.fits(384, HIDDEN)
        assert not hop.fits(48, HIDDEN) and not hop.fits(385, HIDDEN)
        assert not hop.fits(64, 7168)
        log(
            f"\nreferences: SGLang's one-shot {'yes' if one_shot_plane else 'absent'}; "
            "the in-tree two-shot prototype "
            f"{'yes' if two_shot_ref else 'SKIPPED (not in this SGLang)'}"
        )

        gen = torch.Generator(device=device).manual_seed(1000 + rank)
        expected = {}
        for rows in ROWS:
            a = _inputs(rows, device, gen)
            for variant in VARIANTS:
                got = _run(hop, a, variant)
                expected[(rows, variant)] = (a, got)
                line = f"rows={rows:3d} {variant:5s}"
                if variant == "quant":
                    _check_quant(got[1], got[2], got[3])
                # torch reference (streams, normalized)
                ref = _torch(a, variant, tp.device_group, world)
                errs = [
                    (g.float() - r.float()).abs().max().item() for g, r in zip(got, ref)
                ]
                torch.testing.assert_close(
                    got[0].float(), ref[0].float(), atol=3e-2, rtol=2e-2
                )
                if len(ref) > 1:
                    torch.testing.assert_close(
                        got[1].float(), ref[1].float(), atol=6e-2, rtol=3e-2
                    )
                line += " torch max_abs " + "/".join(
                    f"{_max_over_ranks(e):.3g}" for e in errs
                )
                if two_shot_ref:
                    sg = _sglang(a, variant, world, two_shot=True)
                    same = all(_equal(g, s) for g, s in zip(got, sg))
                    assert same, (
                        f"differs from SGLang's two-shot: rows={rows} {variant}"
                    )
                    line += " | two-shot bitwise"
                one = _sglang(a, variant, world, False) if one_shot_plane else None
                if one is None:
                    line += " | one-shot n/a"
                else:
                    stream_err = (got[0].float() - one[0].float()).abs().max().item()
                    assert (
                        stream_err <= NUMERICS.tolerances["two_shot.residual.max_abs"]
                    ), f"streams differ from one-shot: {rows} {variant}"
                    note = " | one-shot streams bitwise"
                    if len(got) > 1:
                        diff = _bits(got[1]).view(torch.int16) != _bits(one[1]).view(
                            torch.int16
                        )
                        n = int(diff.sum())
                        ulps = _bf16_ulps(got[1], one[1]) if n else 0
                        assert ulps <= NUMERICS.tolerances["two_shot.normed.ulp"], (
                            rows,
                            variant,
                            ulps,
                        )
                        n_all = int(_max_over_ranks(n))
                        note += f", norm {'bitwise' if n_all == 0 else f'{n_all} elems differ (<=1 ulp)'}"
                    if len(got) > 2:
                        q_same = _equal(got[2], one[2]) and _equal(got[3], one[3])
                        if NUMERICS.tolerances["two_shot.mxfp8.max_abs"] == 0:
                            assert q_same, (rows, "MXFP8 differs from one-shot")
                        # quant bytes may follow a differing normalized element only
                        if not q_same:
                            rows_diff = diff.any(dim=1)
                            qd = (
                                (_bits(got[2]) != _bits(one[2]))
                                .view(rows, HIDDEN)
                                .any(dim=1)
                            )
                            assert not (qd & ~rows_diff).any(), (
                                rows,
                                "quant differs off the norm diffs",
                            )
                        q_all = _max_over_ranks(0.0 if q_same else 1.0) == 0.0
                        note += f", quant {'bitwise' if q_all else 'differs where norm differs'}"
                    line += note
                log(line, flush=True)

        # 50 back-to-back calls, mixed row counts and variants: epochs and markers cycle
        order = [(rows, v) for v in VARIANTS for rows in ROWS]
        seq = [order[(i * 7) % len(order)] for i in range(50)]
        outs = []
        for i, key in enumerate(seq):
            if rank == i % world:
                torch.cuda._sleep(5_000)
            outs.append(_run(hop, expected[key][0], key[1]))
        torch.cuda.synchronize()
        for key, got in zip(seq, outs):
            for g, e in zip(got, expected[key][1]):
                assert _equal(g, e), f"back-to-back differs: {key}"
        log("50 back-to-back calls: bitwise equal to the first calls", flush=True)

        # one CUDA graph of hops at several row counts, replayed with a late rank
        keys = [(rows, v) for rows in (49, 192, 384, 128) for v in VARIANTS]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
            captured = [_run(hop, expected[k][0], k[1]) for k in keys]
        torch.cuda.current_stream().wait_stream(stream)
        for replay in range(20):
            for got in captured:
                for g in got:
                    g.view(torch.uint8).zero_()
            if rank == replay % world:
                torch.cuda._sleep(50_000)
            graph.replay()
            torch.cuda.synchronize()
            for key, got in zip(keys, captured):
                for g, e in zip(got, expected[key][1]):
                    assert _equal(g, e), f"graph replay {replay} differs: {key}"
            # eager calls between replays keep the epochs moving
            _run(hop, expected[(96, "moe")][0], "moe")
        torch.cuda.synchronize()
        log("CUDA graph (16 hops) x 20 replays: bitwise equal", flush=True)


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4,), timeout=2400)

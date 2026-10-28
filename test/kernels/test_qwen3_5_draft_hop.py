# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The MTP draft's hop (``qwen3_5/sm103/draft/hop.py``, ``qwen3_5/draft/hop.py``) on 4
SM100-family GPUs (GB300) with NVLink multicast, on the decode hop's mailbox:

- add_norm: the consume equals the fp32 rank-order sum rounded to bf16 followed by
  SGLang's ``gemma_fused_add_rmsnorm`` (the stock draft's add-norm), bit for bit;
- mnnvl: its residual output equals bf16(bf16(sum) + residual) bit for bit, and its norm
  stays within one bf16 ulp of an fp32 reference (its bits against FlashInfer's legacy
  MNNVL kernel are checked with SGLang branch nvidia/qwen35-agentx-gb300, not here);
- rows 1, 2 and 7, both entry releases, a late rank in turn, hops interleaved with the
  decode hop's own, so the three mailbox stages cycle;
- the draft o_proj GEMV equals an fp32 emulation of its summation order, bit for bit, and
  its pushed output followed by a consume-only all-reduce equals publish + consume;
- with the attention output gate, the GEMV (plain and pushing) equals SGLang's gate
  kernel followed by the plain GEMV, bit for bit;
- a draft step's hops replayed from a CUDA graph with fresh inputs equal the eager run.

Run with ``python test/kernels/test_qwen3_5_draft_hop.py --num-gpu 4``.
``test_protocol_copies`` needs no GPU.
"""

import os
import re
from pathlib import Path

import pytest
import torch

HIDDEN, EPS, MAX_M, O_PROJ_K = 4096, 1e-6, 7, 1024
SM103 = Path(__file__).resolve().parents[2] / "sgl_nvidia_mega" / "qwen3_5" / "sm103"

gpu = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10),
    reason="Requires SM100/SM103",
)


def _protocol(path: Path) -> str:
    text = path.read_text()
    start = text.index("#include <cuda_bf16.h>")
    end = text.index("}  // namespace q35hop\n") + len("}  // namespace q35hop\n")
    return text[start:end]


def test_protocol_copies():
    """The draft kernels' copy of the decode hop's Site, Ctrl and mailbox protocol is
    q35_hop.cuh's, verbatim."""
    want = _protocol(SM103 / "hop" / "src" / "q35_hop.cuh")
    for name in ("draft_hop.cu", "draft_gemv.cu"):
        assert _protocol(SM103 / "draft" / "src" / name) == want, name
    # the draft consume's Args extends hop.cu's field by field
    fields = re.compile(r"struct Args \{(.*?)\n\};", re.S)
    hop_args = fields.search((SM103 / "hop" / "src" / "hop.cu").read_text()).group(1)
    draft_args = fields.search((SM103 / "draft" / "src" / "draft_hop.cu").read_text())
    assert draft_args.group(1).startswith(hop_args)


@pytest.fixture(scope="module")
def rank():
    from sglang.srt.distributed import parallel_state as ps
    from sglang.test.test_utils import publish_build_topology

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


@pytest.fixture(scope="module")
def draft(rank):
    """The decode hop as the target builds it (7 rows), and the draft hop on it."""
    from sglang.srt.distributed import parallel_state as ps

    from sgl_nvidia_mega.qwen3_5.draft.hop import DraftHop
    from sgl_nvidia_mega.qwen3_5.sm103.draft import hop as kernels
    from sgl_nvidia_mega.qwen3_5.sm103.hop import Hop

    group = ps.get_tp_group().device_group
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    hop = Hop(max_m=MAX_M, eps=EPS, group=group, device=device, deadline_s=30.0)
    kernels.build(device.index, int(30.0 * 1e9), gemv_ks=(O_PROJ_K,))
    return DraftHop(hop)


class _Norm:
    """The GemmaRMSNorm fields the draft hop reads."""

    def __init__(self, weight):
        self.weight = weight
        self.gemma_weight = torch.add(weight, 1.0)
        self.variance_epsilon = EPS


def _operands(m, device, rank, seed):
    own = torch.Generator(device=device).manual_seed(seed * 100 + rank)
    common = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape, gen=own):
        return torch.randn(*shape, device=device, generator=gen).bfloat16()

    return dict(
        partial=randn(m, HIDDEN),
        attn=randn(1, O_PROJ_K),
        o_proj=(0.03 * torch.randn(HIDDEN, O_PROJ_K, device=device, generator=own))
        .bfloat16()
        .contiguous(),
        residual=randn(m, HIDDEN, gen=common),
        weight=(0.1 * randn(HIDDEN, gen=common).float()).bfloat16(),
    )


def _rank_sum(partial):
    """The hop's pull: the ranks' partials (-0.0 pushed as +0.0) summed in fp32 from
    rank 0, rounded to bf16."""
    import torch.distributed as dist
    from sglang.srt.distributed import parallel_state as ps

    group = ps.get_tp_group().device_group
    parts = [torch.empty_like(partial) for _ in range(dist.get_world_size(group))]
    dist.all_gather(parts, partial.contiguous(), group=group)
    acc = None
    for p in parts:
        p = torch.where(p.view(torch.int16) == -32768, torch.zeros_like(p), p)
        acc = p.float() if acc is None else acc + p.float()
    return acc.bfloat16()


def _same(got, want):
    for a, b in zip(got, want):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16))


def _gemv_ref(x, w):
    """mkd_draft_gemv's arithmetic: lane l of a row's warp fma-accumulates columns
    (32 j + l) * 8 + e in (j, e) order (a bf16 product is exact in fp32, so add after
    multiply rounds once, as fma), then the xor butterfly 16 .. 1; lane 0 stores."""
    k, n = x.shape[1], w.shape[0]
    xs = x.float().view(k // 256, 32, 8)
    ws = w.float().view(n, k // 256, 32, 8)
    acc = torch.zeros(n, 32, device=x.device)
    for j in range(k // 256):
        for e in range(8):
            acc = acc + ws[:, j, :, e] * xs[j, :, e]
    lanes = torch.arange(32, device=x.device)
    for mask in (16, 8, 4, 2, 1):
        acc = acc + acc[:, lanes ^ mask]
    return acc[:, 0].bfloat16().view(1, n)


@gpu
def test_draft_hop_rounding(rank, draft):
    from sgl_kernel import gemma_fused_add_rmsnorm

    from sgl_nvidia_mega.qwen3_5.sm103.draft import hop as kernels

    world = torch.distributed.get_world_size()
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    hop = draft.hop
    calls = [
        (m, add_norm, early)
        for m in (1, 2, 7)
        for add_norm in (False, True)
        for early in (False, True)
    ]
    with torch.inference_mode():
        for i, (m, add_norm, early) in enumerate(calls):
            x = _operands(m, device, rank, seed=i)
            norm = _Norm(x["weight"])
            summed = _rank_sum(x["partial"])
            if i % (world + 1) == rank:
                torch.cuda._sleep(2_000_000)  # this rank publishes late
            got = kernels.draft_all_reduce_rms_norm(
                hop,
                local_contribution=x["partial"],
                residual=x["residual"],
                gamma=norm.weight if add_norm else norm.gemma_weight,
                eps=EPS,
                weight_bias=1.0 if add_norm else 0.0,
                add_norm=add_norm,
                early_release=early,
            )
            if add_norm:
                want_x, want_r = summed.clone(), x["residual"].clone()
                gemma_fused_add_rmsnorm(want_x, want_r, norm.weight, EPS)
                _same(got, (want_x, want_r))
            else:
                pre = summed + x["residual"]  # bf16 add, as __hadd
                _same(got[1:], (pre,))
                f = pre.float()
                ref = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + EPS)
                ref = (ref * norm.gemma_weight.float()).bfloat16().float()
                # bf16 squares and an approximate rsqrt: at most one bf16 ulp off
                assert ((got[0].float() - ref).abs() <= ref.abs() * 2**-7).all()
            # the decode hop's own hop in between keeps the stage order shared
            y = _operands(m, device, rank, seed=1000 + i)
            hop.all_reduce_residual_rms_norm(
                local_contribution=y["partial"],
                residual=y["residual"],
                gamma=norm.gemma_weight,
            )
        torch.cuda.synchronize()


@gpu
def test_draft_gemv_and_push(rank, draft):
    from sgl_nvidia_mega.qwen3_5.sm103.draft import hop as kernels

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    with torch.inference_mode():
        for seed in range(4):
            x = _operands(1, device, rank, seed=200 + seed)
            norm = _Norm(x["weight"])
            plain = torch.empty(1, HIDDEN, dtype=torch.bfloat16, device=device)
            kernels.draft_gemv(x["attn"], x["o_proj"], plain)
            _same((plain,), (_gemv_ref(x["attn"], x["o_proj"]),))
            want = kernels.draft_all_reduce_rms_norm(
                draft.hop,
                local_contribution=plain,
                residual=x["residual"],
                gamma=norm.gemma_weight,
                eps=EPS,
                weight_bias=0.0,
                add_norm=False,
            )
            pushed = draft.o_proj(x["attn"], x["o_proj"])  # pushes into the hop
            assert draft.has_pushed(pushed)
            got = draft.ffn_input(
                pushed, x["residual"], norm, moe_single_launch=seed % 2 == 1
            )
            assert not draft.has_pushed(pushed)
            _same((pushed,), (plain,))
            _same(got, want)
        torch.cuda.synchronize()


@gpu
def test_draft_gemv_folds_the_output_gate(rank, draft):
    from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul

    from sgl_nvidia_mega.qwen3_5.sm103.draft import hop as kernels

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    with torch.inference_mode():
        for seed in range(4):
            x = _operands(1, device, rank, seed=500 + seed)
            norm = _Norm(x["weight"])
            g = torch.Generator(device=device).manual_seed(600 + seed)
            gate = (
                8.0 * torch.randn(1, O_PROJ_K, device=device, generator=g)
            ).bfloat16()
            gate[0, :4] = torch.tensor([-0.0, 0.0, 100.0, -100.0])
            want = torch.empty(1, HIDDEN, dtype=torch.bfloat16, device=device)
            kernels.draft_gemv(fused_sigmoid_mul(x["attn"], gate), x["o_proj"], want)
            got = torch.empty_like(want)
            kernels.draft_gemv(x["attn"], x["o_proj"], got, gate=gate)
            _same((got,), (want,))
            pushed = draft.o_proj(x["attn"], x["o_proj"], gate=gate)
            draft.ffn_input(pushed, x["residual"], norm, moe_single_launch=False)
            _same((pushed,), (want,))
        torch.cuda.synchronize()


@gpu
def test_draft_step_replays_from_a_cuda_graph(rank, draft):
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    def step(x, norm):
        part = draft.o_proj(x["attn"], x["o_proj"])
        h, res = draft.ffn_input(part, x["residual"], norm, moe_single_launch=True)
        return draft.final_norm(x["partial"], res, norm) + (h,)

    with torch.inference_mode():
        static = [_operands(1, device, rank, seed=300 + i) for i in range(2)]
        norms = [_Norm(x["weight"]) for x in static]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for x, norm in zip(static, norms):
                step(x, norm)  # eager warmup on the capture stream
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = [step(x, norm) for x, norm in zip(static, norms)]
        for replay in range(3):
            for i, x in enumerate(static):
                fresh = _operands(1, device, rank, seed=400 + 10 * replay + i)
                for name, tensor in x.items():
                    tensor.copy_(fresh[name])
                norms[i].gemma_weight.copy_(torch.add(x["weight"], 1.0))
            graph.replay()
            for x, norm, got in zip(static, norms, outputs):
                _same(got, step(x, norm))
        torch.cuda.synchronize()


if __name__ == "__main__":
    from sglang.test.kernels.utils import multigpu_pytest_main

    multigpu_pytest_main(__name__, __file__, num_gpus=(4,))

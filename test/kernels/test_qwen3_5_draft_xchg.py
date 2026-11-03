# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The MTP draft step boundary (``qwen3_5/draft/xchg.py`` on ``qwen3_5/sm103/draft``) on 4
SM100-family GPUs (GB300) with NVLink multicast, against torch on the same inputs:

- the draft head + exchange pick the global argmax of the ranks' shard logits (order keys:
  -0.0 as +0.0, ties to the lowest id; NaN as -1e30 in draft steps, highest after the draft
  extend), write the draft column, advance the positions and the M-RoPE positions, and the
  next step's fc equals the memo row plus the hidden ranks' partials summed in rank order,
  rounded once to bf16, bit for bit;
- the memo rows equal the split fc's partials summed in rank order (checked at
  construction), and the split fc's gather and rank-order sum are exact;
- a whole draft round (five steps, the extend exchange with extend fc) replays from a CUDA
  graph with fresh inputs, equal to its eager run.

The bitwise comparison with the kernels of SGLang branch nvidia/qwen35-agentx-gb300 needs that
branch installed as ``sglang`` and is not part of this test.
Run with ``python test/kernels/test_qwen3_5_draft_xchg.py --num-gpu 4``.
"""

import os

import pytest
import torch
import torch.distributed as dist
from sglang.test.kernels.utils import multigpu_pytest_main

H, EPS, NUM_STEPS, VOCAB = 4096, 1e-6, 6, 1024  # VOCAB rows a rank: a short memo check

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10),
    reason="Requires SM100/SM103",
)


class _Norm:
    """A Gemma RMSNorm's weight and epsilon, and its forward (x * rsqrt(mean x^2 + eps) *
    (1 + w), fp32 statistics, bf16 output) for splits over two rows."""

    def __init__(self, weight):
        self.weight, self.variance_epsilon = weight, EPS

    def __call__(self, x):
        xf = x.float()
        inv = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
        return (xf * inv * (1.0 + self.weight.float())).to(torch.bfloat16)


@pytest.fixture(scope="module")
def ranks():
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    cpu_group = dist.new_group(backend="gloo")
    yield dist.get_rank(), dist.get_world_size(), dist.group.WORLD, cpu_group
    dist.destroy_process_group()


def _gen(seed):
    return torch.Generator(device="cuda").manual_seed(seed)


def _randn(*shape, g, scale=1.0, dtype=torch.bfloat16):
    return (torch.randn(*shape, device="cuda", generator=g) * scale).to(dtype)


@pytest.fixture(scope="module")
def boundary(ranks):
    from sgl_nvidia_mega.qwen3_5.draft.xchg import MkdBoundary

    rank, world, group, cpu_group = ranks
    g = _gen(2026)
    full_embed = _randn(world * VOCAB, H, g=g)
    weights = dict(
        fc=_randn(H, 2 * H, g=g, scale=0.02),
        norm_emb=_randn(H, g=g, scale=0.1),
        norm_hid=_randn(H, g=g, scale=0.1),
        embed=full_embed[rank * VOCAB : (rank + 1) * VOCAB].contiguous(),
        head=_randn(world * VOCAB, H, g=g, scale=0.02)[
            rank * VOCAB : (rank + 1) * VOCAB
        ].contiguous(),
    )
    mkd = MkdBoundary(
        fc_weight=weights["fc"],
        norm_embedding=_Norm(weights["norm_emb"]),
        norm_hidden=_Norm(weights["norm_hid"]),
        embed_shard=weights["embed"],
        vocab_start=rank * VOCAB,
        lm_head_shard=weights["head"],
        embed=lambda ids: full_embed[ids],
        rank=rank,
        world=world,
        device_group=group,
        cpu_group=cpu_group,
        num_steps=NUM_STEPS,
    )
    return mkd, weights


def _all_gather(t, group):
    world = dist.get_world_size(group)
    out = torch.empty((world, *t.shape), dtype=t.dtype, device=t.device)
    dist.all_gather_into_tensor(out, t.contiguous(), group=group)
    return out


def _want_token(local_logits, group, nan_as):
    """The argmax over the ranks' logits concatenated in rank order, ties to the lowest id;
    a NaN counts as ``nan_as``."""
    full = _all_gather(local_logits[0], group).reshape(-1).float()
    full = torch.where(torch.isnan(full), torch.full_like(full, nan_as), full)
    return int(torch.argmax(full))  # the first maximum: the lowest id


def test_step_exchange_picks_the_global_argmax(ranks, boundary):
    rank, world, group, _ = ranks
    mkd, _ = boundary
    n_embed = world // 2
    with torch.inference_mode():
        for seed in range(4):
            g = _gen(100 + seed)
            hidden = _randn(1, H, g=g)
            if seed == 1:  # ties between rows and ranks
                mkd.lm_head_weight.copy_(torch.round(mkd.lm_head_weight * 64) / 64)
            draft_tokens = torch.zeros(1, NUM_STEPS, dtype=torch.int64, device="cuda")
            positions = torch.tensor([77], dtype=torch.int64, device="cuda")
            mrope = torch.arange(6, dtype=torch.int64, device="cuda").view(3, 2).clone()
            step = seed % (NUM_STEPS - 2)  # not the last step: the payload travels
            mkd.begin_step(step, 1)
            mkd.fc()
            mkd.hidden_partial(hidden)
            logits = mkd.head(hidden)
            topk_p, topk_index = mkd.finish_step(
                logits, positions, draft_tokens, step, step + 1, mrope[:, :1]
            )
            torch.cuda.synchronize()
            want = _want_token(logits, group, -1e30)
            assert int(topk_index) == want and float(topk_p) == 1.0
            assert int(draft_tokens[0, step + 1]) == want
            assert int(positions) == 78
            assert mrope[:, 0].tolist() == [1, 3, 5] and mrope[:, 1].tolist() == [
                1,
                3,
                5,
            ]
            # the next fc: the winner's memo row, then the hidden ranks' partials in rank order
            owner, row = divmod(want, VOCAB)
            memo_row = _all_gather(mkd.cand[min(row, VOCAB - 1)], group)[owner].float()
            partials = _all_gather(mkd.own_partial[0], group)[n_embed:]
            acc = memo_row.clone()
            for h in range(world - n_embed):
                acc = acc + partials[h]
            assert torch.equal(
                mkd.fc_buf[0].view(torch.int16), acc.bfloat16().view(torch.int16)
            )
            assert mkd.pending
            # the head's logits against torch (only the fp32 summation order differs)
            ref = (hidden.float() @ mkd.lm_head_weight.float().T).bfloat16()
            torch.testing.assert_close(
                logits.float(), ref.float(), atol=2e-2, rtol=2e-2
            )


@pytest.mark.parametrize("nan_key", ["split", "torch"])
def test_extend_exchange_nan_rules(ranks, boundary, nan_key):
    from sgl_nvidia_mega.qwen3_5.sm103.draft import xchg as kernels

    rank, world, group, _ = ranks
    mkd, _ = boundary
    with torch.inference_mode():
        g = _gen(500 + rank)
        logits = _randn(1, VOCAB, g=g)
        if rank == 1:
            logits[0, 17] = float("nan")
        topk_index = torch.empty(1, 1, dtype=torch.int64, device="cuda")
        topk_p = torch.empty(1, 1, dtype=torch.float32, device="cuda")
        key = (
            kernels.NAN_KEY_SPLIT_ARGMAX
            if nan_key == "split"
            else kernels.NAN_KEY_TORCH_ARGMAX
        )
        kernels.mkd_xchg(
            logits,
            mkd.cand,
            mkd.hot_ids,
            mkd.own_partial,
            mkd.epoch_done,
            mkd.ext_cand,
            mkd.ext_partials,
            topk_index,
            topk_p,
            None,
            None,
            None,
            mkd.mc,
            mkd.uc,
            rank,
            world,
            0,
            key,
            token_only=True,
            scan_part=mkd.scan_part,
        )
        torch.cuda.synchronize()
        want = VOCAB + 17 if nan_key == "torch" else _want_token(logits, group, -1e30)
        assert int(topk_index) == want


@pytest.mark.parametrize("m", [1, 2, 7])
def test_split_fc_gathers_and_sums_in_rank_order(ranks, boundary, m):
    from sgl_nvidia_mega.qwen3_5.sm103.draft import xchg as kernels

    rank, world, group, _ = ranks
    mkd, weights = boundary
    with torch.inference_mode():
        g = _gen(900 + m)
        embeds, hidden = _randn(m, H, g=g), _randn(m, H, g=g)
        partial = _randn(m, H, g=_gen(950 + 7 * rank + m), dtype=torch.float32)
        gathered = mkd._fc_gather(partial)
        want = _all_gather(partial, group).permute(1, 0, 2).reshape(m, world * H)
        assert torch.equal(gathered, want)
        acc = want[:, :H].clone()
        for r in range(1, world):
            acc = acc + want[:, r * H : (r + 1) * H]
        summed = kernels.ordered_partial_sum(
            gathered, world, torch.empty(m, H, dtype=torch.bfloat16, device="cuda")
        )
        assert torch.equal(summed.view(torch.int16), acc.bfloat16().view(torch.int16))
        # the split fc against the replicated fc in fp32 (only the summation order differs)
        got = mkd.split_fc(embeds, hidden)
        x = torch.cat(
            [_Norm(weights["norm_emb"])(embeds), _Norm(weights["norm_hid"])(hidden)], -1
        )
        ref = (x.float() @ weights["fc"].float().T).bfloat16()
        torch.testing.assert_close(got.float(), ref.float(), atol=3e-2, rtol=3e-2)


def test_round_replays_from_a_cuda_graph(ranks, boundary):
    rank, world, group, cpu_group = ranks
    mkd, _ = boundary

    def inputs(seed):
        g = _gen(3000 + seed)
        return dict(
            token=torch.randint(0, world * VOCAB, (1, 1), device="cuda", generator=g),
            hidden0=_randn(1, H, g=g),
            hidden=[_randn(1, H, g=g) for _ in range(NUM_STEPS - 1)],
            ext_logits=_randn(1, VOCAB, g=_gen(4000 + 10 * seed + rank)),
            tokens=torch.zeros(1, NUM_STEPS, dtype=torch.int64, device="cuda"),
            pos=torch.tensor([9], dtype=torch.int64, device="cuda"),
        )

    def run_round(st):
        out = []
        for i in range(NUM_STEPS - 1):
            mkd.begin_step(i, 1)
            out.append(mkd.fc().clone())
            mkd.hidden_partial(st["hidden"][i])
            logits = mkd.head(st["hidden"][i])
            mkd.finish_step(logits, st["pos"], st["tokens"], i, i + 1)
        mkd.end()
        mkd.note_draft_input()
        _, ti = mkd.exchange_extend(st["ext_logits"], st["hidden0"])
        return out + [ti.clone(), mkd.fc_buf.clone(), st["tokens"], st["pos"]]

    with torch.inference_mode():
        eager = []
        for seed in range(3):
            st = inputs(seed)
            mkd.fill_fc(st["token"], st["hidden0"])
            eager.append([t.clone() for t in run_round(st)])
        static = inputs(0)
        mkd.fill_fc(static["token"], static["hidden0"])
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run_round(static)
        torch.cuda.current_stream().wait_stream(stream)
        dist.barrier(group=cpu_group)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run_round(static)
        torch.cuda.synchronize()
        dist.barrier(group=cpu_group)
        for seed in range(3):
            fresh = inputs(seed)
            for key in ("token", "hidden0", "ext_logits", "tokens", "pos"):
                static[key].copy_(fresh[key])
            for i in range(NUM_STEPS - 1):
                static["hidden"][i].copy_(fresh["hidden"][i])
            mkd.fill_fc(static["token"], static["hidden0"])
            graph.replay()
            torch.cuda.synchronize()
            for got, want in zip(captured, eager[seed]):
                assert torch.equal(
                    got.view(torch.int16) if got.dtype == torch.bfloat16 else got,
                    want.view(torch.int16) if want.dtype == torch.bfloat16 else want,
                )
            dist.barrier(group=cpu_group)


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4,))

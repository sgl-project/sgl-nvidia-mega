# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The DSpark draft's Markov chain slots (sm103/markov.py), each against the chain it
replaces, both CUDA graphs of one draft block (noise forked onto the side stream when a
row samples) replayed over the same Gaussian inputs, greedy, sampled and mixed
temperatures, through the direct exchange and the rank-local fold:

- markov_lowlat (converged pollers, acquire loads, the direct exchange to 4 rows at
  world 4 and 2 at world 8) against the current kernel: tokens and kept logits bit for
  bit;
- markov_tail (the chain writes SGLang's verify ids ``cat([anchor, draft tokens])`` and
  advances the draft-noise offset) against the chain followed by those two launches:
  tokens, logits, verify ids and offset bit for bit, rows past the block untouched;
- markov_shard_logits (only this rank's vocab columns of the step logits kept) against
  the full chain (markov_lowlat and markov_tail on, as served): tokens, verify ids and
  offset bit for bit, the kept logits on this rank's columns bit for bit, the other
  columns untouched.

The same on every rank of a torchrun job (world 8: two GB300 trays):
``torchrun --nnodes 2 --nproc_per_node 4 -m pytest -k ranks ...``.
"""

import os

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41.sm103 import markov

VOCAB, RANK, STEPS, MAX_BS = 129_280, 256, 5, 8
KEY = 0x1234_5678_9ABC_DEF
TEMPERATURE_SETS = (
    (0.0,) * MAX_BS,
    (1.0,) * MAX_BS,
    (0.7, 1.3, 1.0, 0.0, 0.5, 2.0, 0.9, 1.1),
)
ROWS = (1, 3, 8)  # direct exchange (world 4: 1 and 3) and rank-local fold
BLOCKS = 40
STALE = -123.0
# slot: (reference chain, slot chain, which logits compare)
SLOTS = {
    "markov_lowlat": ({}, {"lowlat": True}, "kept"),
    "markov_tail": ({"lowlat": True}, {"lowlat": True, "tail": True}, "all"),
    "markov_shard_logits": (
        {"lowlat": True, "tail": True},
        {"lowlat": True, "tail": True, "shard_logits": True},
        "shard",
    ),
}

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in markov.SUPPORTED_CAPABILITIES,
    reason="needs an SM100/SM103 GPU",
)


def _make(group, rank, world, w1, w2, kw):
    return markov.ShardedMarkov(
        embed_weight=w1,
        head_weight=w2,
        vocab_size=VOCAB,
        num_steps=STEPS,
        cache_dtype=torch.float32,
        group=group,
        rank=rank,
        world=world,
        max_bs=MAX_BS,
        deadline_s=20.0,
        keeps_logits=True,
        **kw,
    )


def _compare(device, group, rank, world, seed, slot) -> dict:
    ref_kw, slot_kw, logits = SLOTS[slot]
    g = torch.Generator().manual_seed(seed)  # the same weights on every rank
    w1 = (torch.randn(VOCAB, RANK, generator=g) * 0.05).to(device, torch.bfloat16)
    w2 = (torch.randn(VOCAB, RANK, generator=g) * 0.05).to(device, torch.bfloat16)
    chains = [_make(group, rank, world, w1, w2, kw) for kw in (ref_kw, slot_kw)]
    vocab_local = VOCAB // world
    shard = slice(rank * vocab_local, (rank + 1) * vocab_local)
    foreign = torch.ones(VOCAB, dtype=torch.bool, device=device)
    foreign[shard] = False
    counts = {"blocks": 0, "mismatch": 0, "distinct": 0}
    for temps in TEMPERATURE_SETS:
        temperatures = torch.tensor(temps, device=device)
        for rows in ROWS:
            if rows > min(c.max_rows for c in chains):
                continue
            sampled = any(t != 0.0 for t in temps[:rows])
            keeps = (temperatures[:rows] != 0).view(rows, 1, 1)
            base = torch.zeros(
                rows, STEPS, vocab_local, dtype=torch.bfloat16, device=device
            )
            anchor = torch.zeros(rows, dtype=torch.int64, device=device)
            out, graphs = [], []
            for chain, kw in zip(chains, (ref_kw, slot_kw)):
                tail = kw.get("tail", False)
                o = dict(
                    tokens=torch.full((MAX_BS, STEPS), -7, device=device).long(),
                    cache=torch.full((MAX_BS, STEPS, VOCAB), STALE, device=device),
                    verify=torch.full((MAX_BS, STEPS + 1), -9, device=device).long(),
                    offset=torch.zeros(1, dtype=torch.int64, device=device),
                )
                out.append(o)

                def block(chain=chain, o=o, tail=tail):
                    if sampled:
                        chain.fork_noise(rows=rows, key=KEY, offset=o["offset"])
                        chain.join_noise()
                    chain.sample(
                        base_logits=base,
                        anchor_tokens=anchor,
                        temperatures=temperatures,
                        logits_cache=o["cache"] if sampled else None,
                        draft_tokens=o["tokens"],
                        verify_ids=o["verify"] if tail else None,
                        noise_offset=o["offset"] if tail and sampled else None,
                    )
                    if not tail:
                        # the launches the tail replaces (draft proposal, SGLang's concat)
                        if sampled:
                            o["offset"].add_(1)
                        torch.cat(
                            [anchor.view(rows, 1), o["tokens"][:rows]],
                            dim=1,
                            out=o["verify"][:rows],
                        )

                side = torch.cuda.Stream(device)
                side.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(side):
                    block()
                torch.cuda.current_stream(device).wait_stream(side)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=side):
                    block()
                graphs.append(graph)
            first = set()
            for i in range(BLOCKS):
                gen = torch.Generator(device=device).manual_seed(seed * 100_003 + i)
                full = torch.randn(rows, STEPS, VOCAB, generator=gen, device=device) * 3
                base.copy_(full[:, :, shard])
                anchor.copy_(
                    torch.randint(0, VOCAB, (rows,), generator=gen, device=device)
                )
                for o, graph in zip(out, graphs):
                    o["offset"].fill_(1000 + 7 * i)
                    o["cache"].fill_(STALE)
                    graph.replay()
                a, b = out
                counts["blocks"] += 1
                bad = not torch.equal(a["tokens"], b["tokens"])
                bad |= not torch.equal(a["verify"], b["verify"])
                bad |= not torch.equal(a["offset"], b["offset"])
                bad |= not bool((b["verify"][rows:] == -9).all())
                if sampled:
                    bad |= int(b["offset"]) != 1000 + 7 * i + 1
                    ca, cb = a["cache"], b["cache"]
                    if logits == "all":
                        bad |= not torch.equal(ca, cb)
                    elif logits == "kept":
                        bad |= bool(((ca[:rows] != cb[:rows]) & keeps).any())
                    else:
                        bad |= not torch.equal(ca[:, :, shard], cb[:, :, shard])
                        bad |= not bool((cb[:, :, foreign] == STALE).all())
                counts["mismatch"] += int(bad)
                first.add(int(a["tokens"][0, 0]))
            counts["distinct"] += len(first)
    torch.cuda.synchronize(device)
    return counts


@needs_gpu
@pytest.mark.parametrize("slot", list(SLOTS))
def test_one_rank_slot_is_the_chain_it_replaces(slot):
    counts = _compare(torch.device("cuda"), None, 0, 1, 3, slot)
    assert counts["mismatch"] == 0, counts
    assert counts["distinct"] > counts["blocks"] // 4, counts  # the blocks differ


_TORCHRUN = "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1


@pytest.fixture(scope="module")
def ranks():
    import torch.distributed as dist

    world, rank = int(os.environ["WORLD_SIZE"]), int(os.environ["RANK"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world)
    group = dist.new_group(list(range(world)), backend="gloo")
    yield device, group, rank, world
    dist.barrier(group=group)
    dist.destroy_process_group()


@needs_gpu
@pytest.mark.skipif(not _TORCHRUN, reason="runs on every rank of a torchrun job")
@pytest.mark.parametrize("slot", list(SLOTS))
def test_ranks_slot_is_the_chain_it_replaces(ranks, slot):
    device, group, rank, world = ranks
    counts = _compare(device, group, rank, world, 4, slot)
    print(f"rank {rank} world {world} {slot}: {counts}", flush=True)
    assert counts["mismatch"] == 0, counts
    assert counts["distinct"] > counts["blocks"] // 4, counts

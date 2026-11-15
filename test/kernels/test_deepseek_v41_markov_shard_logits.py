# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot markov_shard_logits: the sharded Markov chain built with ``shard_logits`` keeps
only this rank's vocab columns of the step logits (no peer staging, no publishing fence,
no epilogue copy). Against the full chain (markov_tail and the low-latency exchange on,
as served), both CUDA graphs of one draft block replayed over the same inputs: tokens,
verify ids and offset bit for bit, the kept logits on this rank's columns bit for bit,
the other columns untouched, at 1-8 rows (narrow kernel) and past 8 (wide), greedy,
sampled and mixed temperatures. The same on every rank of a torchrun job (world 8: two
GB300 trays).

``torchrun --nnodes 2 --nproc_per_node 4 -m pytest -k ranks ...`` for world 8.
"""

import os
import sys

import pytest
import torch

from sgl_nvidia_mega.deepseek_v41.sm103 import markov

VOCAB = 129_280
RANK = 256
STEPS = 5
MAX_BS = 16
KEY = 0x1234_5678_9ABC_DEF
TEMPERATURE_SETS = (
    (0.0,) * MAX_BS,
    (1.0,) * MAX_BS,
    (0.7, 1.3, 1.0, 0.0, 0.5, 2.0, 0.9, 1.1) * 2,
)
DEADLINE_S = 20.0
BLOCKS = 40
ROWS = (1, 2, 3, 5, 8, 9, 16)
STALE = -123.0

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in markov.SUPPORTED_CAPABILITIES,
    reason="needs an SM100/SM103 GPU",
)


def _make(group, rank, world, w1, w2, *, shard_logits):
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
        deadline_s=DEADLINE_S,
        keeps_logits=True,
        lowlat=True,
        tail=True,
        shard_logits=shard_logits,
    )


def _compare(device, group, rank: int, world: int, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)  # the same weights on every rank
    w1 = (torch.randn(VOCAB, RANK, generator=g) * 0.05).to(device, torch.bfloat16)
    w2 = (torch.randn(VOCAB, RANK, generator=g) * 0.05).to(device, torch.bfloat16)
    chains = {
        only: _make(group, rank, world, w1, w2, shard_logits=only)
        for only in (False, True)
    }
    vocab_local = VOCAB // world
    shard = slice(rank * vocab_local, (rank + 1) * vocab_local)
    foreign = torch.ones(VOCAB, dtype=torch.bool, device=device)
    foreign[shard] = False
    counts = {"blocks": 0, "mismatch": 0, "distinct": 0}
    for temps in TEMPERATURE_SETS:
        temperatures = torch.tensor(temps, device=device)
        for rows in ROWS:
            if rows > chains[True].max_rows:
                continue
            sampled = any(t != 0.0 for t in temps[:rows])
            base = torch.zeros(
                rows, STEPS, vocab_local, dtype=torch.bfloat16, device=device
            )
            anchor = torch.zeros(rows, dtype=torch.int64, device=device)
            out, graphs = {}, {}
            for only, chain in chains.items():
                out[only] = dict(
                    tokens=torch.full(
                        (MAX_BS, STEPS), -7, dtype=torch.int64, device=device
                    ),
                    cache=torch.full((MAX_BS, STEPS, VOCAB), STALE, device=device),
                    verify=torch.full(
                        (MAX_BS, STEPS + 1), -9, dtype=torch.int64, device=device
                    ),
                    offset=torch.zeros(1, dtype=torch.int64, device=device),
                )
                o = out[only]

                def block(chain=chain, o=o):
                    if sampled:
                        chain.fork_noise(rows=rows, key=KEY, offset=o["offset"])
                        chain.join_noise()
                    chain.sample(
                        base_logits=base,
                        anchor_tokens=anchor,
                        temperatures=temperatures,
                        logits_cache=o["cache"] if sampled else None,
                        draft_tokens=o["tokens"],
                        verify_ids=o["verify"],
                        noise_offset=o["offset"] if sampled else None,
                    )

                side = torch.cuda.Stream(device)
                side.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(side):
                    block()
                torch.cuda.current_stream(device).wait_stream(side)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=side):
                    block()
                graphs[only] = graph
            first = set()
            for i in range(BLOCKS):
                gen = torch.Generator(device=device).manual_seed(seed * 100_003 + i)
                full = torch.randn(rows, STEPS, VOCAB, generator=gen, device=device) * 3
                base.copy_(full[:, :, shard])
                anchor.copy_(
                    torch.randint(0, VOCAB, (rows,), generator=gen, device=device)
                )
                for only in chains:
                    out[only]["offset"].fill_(1000 + 7 * i)
                    out[only]["cache"].fill_(STALE)
                    graphs[only].replay()
                a, b = out[False], out[True]
                counts["blocks"] += 1
                bad = not torch.equal(a["tokens"], b["tokens"])
                bad |= not torch.equal(a["verify"], b["verify"])
                bad |= not torch.equal(a["offset"], b["offset"])
                if sampled:
                    bad |= not torch.equal(
                        a["cache"][:, :, shard], b["cache"][:, :, shard]
                    )
                    bad |= not bool((b["cache"][:, :, foreign] == STALE).all())
                counts["mismatch"] += int(bad)
                first.add(int(a["tokens"][0, 0]))
            counts["distinct"] += len(first)
    torch.cuda.synchronize(device)
    return counts


@needs_gpu
def test_one_rank_shard_logits_match_the_chain():
    counts = _compare(torch.device("cuda"), None, 0, 1, seed=3)
    assert counts["mismatch"] == 0, counts


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
def test_ranks_shard_logits_match_the_chain(ranks):
    device, group, rank, world = ranks
    counts = _compare(device, group, rank, world, seed=4)
    print(f"rank {rank} world {world}: {counts}", flush=True)
    assert counts["mismatch"] == 0, counts
    assert counts["distinct"] > counts["blocks"] // 4, counts


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))

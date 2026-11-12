# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot markov_tail: the sharded Markov chain built with ``tail`` writes, in its own
launch, what the two launches after it wrote: SGLang's verify ids
``cat([anchor, draft tokens])`` and the draft-noise offset advanced by one. Against the
chain without the tail (the shipped kernel, low-latency exchange on and off), both CUDA
graphs of one draft block replayed over the same inputs: tokens, kept logits, verify ids
and offset bit for bit, at 1-8 rows, greedy, sampled and mixed temperatures, BF16 base
logits, with the narrow (direct / fold) kernels and, past 8 rows, the wide one. Rows past
the block's are left untouched. The same on every rank of a torchrun job (world 8: two
GB300 trays).

``python -m pytest test/kernels/test_deepseek_v41_markov_tail.py -s``;
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
MAX_BS = 16  # past the narrow kernel's 8 rows: the wide kernel too
KEY = 0x1234_5678_9ABC_DEF
TEMPERATURE_SETS = (
    (0.0,) * MAX_BS,
    (1.0,) * MAX_BS,
    (0.7, 1.3, 1.0, 0.0, 0.5, 2.0, 0.9, 1.1) * 2,
)
DEADLINE_S = 20.0
BLOCKS = 60
ROWS = (1, 2, 3, 5, 8, 9, 16)

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in markov.SUPPORTED_CAPABILITIES,
    reason="needs an SM100/SM103 GPU",
)


def _make(group, rank, world, w1, w2, *, lowlat, tail):
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
        lowlat=lowlat,
        tail=tail,
    )


def _compare(device, group, rank: int, world: int, seed: int, lowlat: bool) -> dict:
    g = torch.Generator().manual_seed(seed)  # the same weights on every rank
    w1 = (torch.randn(VOCAB, RANK, generator=g) * 0.05).to(device, torch.bfloat16)
    w2 = (torch.randn(VOCAB, RANK, generator=g) * 0.05).to(device, torch.bfloat16)
    chains = {
        tail: _make(group, rank, world, w1, w2, lowlat=lowlat, tail=tail)
        for tail in (False, True)
    }
    assert chains[True].max_rows >= max(r for r in ROWS if r <= chains[True].max_rows)
    vocab_local = VOCAB // world
    shard = slice(rank * vocab_local, (rank + 1) * vocab_local)
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
            out = {}
            graphs = {}
            for tail, chain in chains.items():
                out[tail] = dict(
                    tokens=torch.full(
                        (MAX_BS, STEPS), -7, dtype=torch.int64, device=device
                    ),
                    cache=torch.zeros(MAX_BS, STEPS, VOCAB, device=device),
                    verify=torch.full(
                        (MAX_BS, STEPS + 1), -9, dtype=torch.int64, device=device
                    ),
                    offset=torch.zeros(1, dtype=torch.int64, device=device),
                )
                o = out[tail]

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
                graphs[tail] = graph
            first = set()
            for i in range(BLOCKS):
                gen = torch.Generator(device=device).manual_seed(seed * 100_003 + i)
                full = torch.randn(rows, STEPS, VOCAB, generator=gen, device=device) * 3
                base.copy_(full[:, :, shard])
                anchor.copy_(
                    torch.randint(0, VOCAB, (rows,), generator=gen, device=device)
                )
                for tail in chains:
                    out[tail]["offset"].fill_(1000 + 7 * i)
                    graphs[tail].replay()
                a, b = out[False], out[True]
                counts["blocks"] += 1
                bad = not torch.equal(a["tokens"], b["tokens"])
                bad |= not torch.equal(a["verify"], b["verify"])
                bad |= not torch.equal(a["offset"], b["offset"])
                bad |= sampled and not torch.equal(a["cache"], b["cache"])
                bad |= not torch.equal(
                    b["verify"][rows:], torch.full_like(b["verify"][rows:], -9)
                )
                if sampled:
                    bad |= int(b["offset"]) != 1000 + 7 * i + 1
                counts["mismatch"] += int(bad)
                first.add(int(a["tokens"][0, 0]))
            counts["distinct"] += len(first)
    torch.cuda.synchronize(device)
    return counts


@needs_gpu
@pytest.mark.parametrize("lowlat", [False, True], ids=["current", "lowlat"])
def test_one_rank_tail_is_the_chain_and_its_two_launches(lowlat):
    counts = _compare(torch.device("cuda"), None, 0, 1, seed=3, lowlat=lowlat)
    assert counts["mismatch"] == 0, counts
    assert counts["distinct"] > counts["blocks"] // 4, counts


@needs_gpu
def test_tail_refuses_outputs_without_the_build():
    device = torch.device("cuda")
    g = torch.Generator().manual_seed(0)
    w = (torch.randn(VOCAB, RANK, generator=g) * 0.05).to(device, torch.bfloat16)
    chain = _make(None, 0, 1, w, w, lowlat=False, tail=False)
    with pytest.raises(AssertionError):
        chain.sample(
            base_logits=torch.zeros(
                1, STEPS, VOCAB, dtype=torch.bfloat16, device=device
            ),
            anchor_tokens=torch.zeros(1, dtype=torch.int64, device=device),
            temperatures=torch.zeros(MAX_BS, device=device),
            logits_cache=None,
            draft_tokens=torch.zeros(1, STEPS, dtype=torch.int64, device=device),
            verify_ids=torch.zeros(1, STEPS + 1, dtype=torch.int64, device=device),
        )


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
@pytest.mark.parametrize("lowlat", [False, True], ids=["current", "lowlat"])
def test_ranks_tail_is_the_chain_and_its_two_launches(ranks, lowlat):
    device, group, rank, world = ranks
    counts = _compare(device, group, rank, world, seed=4, lowlat=lowlat)
    print(f"rank {rank} world {world} lowlat {lowlat}: {counts}", flush=True)
    assert counts["mismatch"] == 0, counts
    assert counts["distinct"] > counts["blocks"] // 4, counts


@needs_gpu
@pytest.mark.skipif(not _TORCHRUN, reason="runs on every rank of a torchrun job")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs a second local GPU")
def test_ranks_graphs_capture_on_the_rank_device(ranks):
    """torch.cuda.graph's default capture stream is created once per process, on the
    device current at its first use: cuda:0 when the one-rank cases ran first. The
    ranks' graphs capture on their own device whatever that stream's device is."""
    device, group, rank, world = ranks
    assert hasattr(torch.cuda.graph, "default_capture_stream")
    saved = torch.cuda.graph.default_capture_stream
    other = torch.device("cuda", (device.index + 1) % torch.cuda.device_count())
    torch.cuda.graph.default_capture_stream = torch.cuda.Stream(other)
    try:
        counts = _compare(device, group, rank, world, seed=5, lowlat=False)
    finally:
        torch.cuda.graph.default_capture_stream = saved
    assert counts["mismatch"] == 0, counts


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))

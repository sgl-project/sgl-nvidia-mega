# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot engram_publish on every rank (torchrun), against the prefetch without it.

Stand-in Engram layers 1 and 14 at DeepSeek-V4.1-Flash's shapes (per-rank table
shards summed by SGLang's TP all-reduce, a real MXFP8 WKV [25600, 6144] on FlashInfer's
GEMM). A step runs the prefetch as the executor does: the lookups, layer 0's reduce,
then every layer start up to 14 (the later layer's columns at layer 10, each Engram
layer's collect at its layer), with stand-in layer work (sleeps and an NCCL
all-reduce) on the main stream. From CUDA graphs replayed with fresh token ids, with
the side stream or one rank late in turn, and eagerly:

- the collected rows of both layers equal the all-reduce gather's rows of the prefetch
  without the slot by value, and SGLang's whole WKV GEMM (``wkv(embed(ids))``) bit for
  bit;
- consecutive replays never read a previous step's words (the token ids change every
  replay, so a stale row would differ).

Run with ``python test/kernels/test_deepseek_v41_engram_publish.py --num-gpu 4 -s``.
"""

import os
from types import SimpleNamespace

import pytest
import torch

N, K, HASH_COLS, SHARD_ROWS = 25600, 6144, 12, 2048
DIM = K // HASH_COLS
ENGRAM_LAYERS = (1, 14)
REPLAYS = 40  # per scenario
DEADLINE_S = 30.0


def _dist() -> bool:
    return (
        "RANK" in os.environ
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 10
    )


needs_dist = pytest.mark.skipif(
    not _dist(),
    reason="run via torchrun: python test/kernels/test_deepseek_v41_engram_publish.py "
    "--num-gpu 4 -s",
)


@pytest.fixture(scope="module")
def env():
    from sglang.srt.distributed import parallel_state as ps
    from sglang.test.test_utils import publish_build_topology

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    ps.set_custom_all_reduce(True)
    ps.init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
    )
    publish_build_topology(tp_size=world_size, world_rank=rank)
    ps.initialize_model_parallel()
    yield SimpleNamespace(rank=rank, tp=world_size, group=ps.get_tp_group())
    ps.destroy_model_parallel()
    ps.destroy_distributed_environment()


class _Embed:
    """SGLang's per-rank Engram table: this rank's rows, zero for the rest."""

    def __init__(self, shard, row_start):
        self.weight, self.row_start, self.rows, self.dim = (
            shard,
            row_start,
            len(shard),
            DIM,
        )

    def _owned_rows(self, indices):
        local = indices - self.row_start
        owned = (local >= 0) & (local < self.rows)
        rows = self.weight[local.clamp(0, self.rows - 1)]
        return torch.where(owned[..., None], rows, torch.zeros_like(rows))

    def _reduce_owned_rows(self, values):
        from sglang.srt.distributed import tensor_model_parallel_all_reduce

        return tensor_model_parallel_all_reduce(values)


def _engrams(rank):
    from flashinfer import block_scale_interleave

    out = {}
    for h, layer in enumerate(ENGRAM_LAYERS):
        gen = torch.Generator(device="cuda").manual_seed(100 + h)
        weight = (torch.randn((N, K), device="cuda", generator=gen) * 0.5).to(
            torch.float8_e4m3fn
        )
        scales = torch.randint(118, 126, (N, K // 32), device="cuda", generator=gen)
        swizzled = block_scale_interleave(scales.to(torch.uint8).contiguous())
        wkv = SimpleNamespace(weight=weight, weight_scale_inv_swizzled=swizzled)
        local = torch.Generator(device="cuda").manual_seed(1000 * h + rank)
        shard = torch.randn(
            (SHARD_ROWS, DIM), device="cuda", generator=local
        ).bfloat16()
        # a few -0.0 rows: the exchange keeps them, the all-reduce makes them +0.0
        shard[::97] = -0.0
        out[layer] = SimpleNamespace(
            embed=_Embed(shard, rank * SHARD_ROWS), wkv=wkv, layer_hash_index=h
        )
    return out


def _hash_ids(ids, tp):
    k = torch.arange(len(ENGRAM_LAYERS) * HASH_COLS, device=ids.device)
    h = ids.long()[:, None] * 7919 + k * 104729
    return (h % (tp * SHARD_ROWS)).view(-1, len(ENGRAM_LAYERS), HASH_COLS)


def _step(env, prefetch, ids, out, late):
    """One step's prefetch as the executor runs it, the layers' work stood in by
    sleeps and an NCCL all-reduce; ``out`` [layers, rows, n] gets the gate's rows."""
    import torch.distributed as dist

    stream = prefetch.stream
    if late == "side":
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            torch.cuda._sleep(200_000)
    if late == "main":
        torch.cuda._sleep(200_000)
    prefetch.issue(_hash_ids(ids, env.tp))
    busy = torch.ones(4096, device="cuda")
    dist.all_reduce(busy, group=env.group.device_group)
    prefetch.reduce()
    for i in range(prefetch.gather_at, max(ENGRAM_LAYERS) + 1):
        torch.cuda._sleep(5_000)
        if prefetch.exchange is not None:
            prefetch.at_layer(i)
        elif i == prefetch.gather_at:
            prefetch.gather()
        if i in ENGRAM_LAYERS:
            if prefetch.exchange is not None:
                rows = prefetch._rows_of(i, ids.size(0))
            else:
                rows = prefetch.kv[prefetch.index[i]]
            out[ENGRAM_LAYERS.index(i)].copy_(rows)


def _graph(env, prefetch, ids, out, late):
    from sglang.srt.distributed import parallel_state as ps

    for _ in range(2):  # warm up eagerly, as SGLang's capture does
        _step(env, prefetch, ids, out, "none")
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with ps.graph_capture() as ctx:
        with torch.cuda.graph(graph, stream=ctx.stream):
            _step(env, prefetch, ids, out, late)
    return graph


def _inline(engrams, hash_ids):
    """SGLang's inline rows: the reduced lookup, then the whole WKV GEMM."""
    from sglang.srt.layers.quantization.fp8_utils import (
        flashinfer_mxfp8_blockscaled_linear,
    )

    rows = []
    for layer in ENGRAM_LAYERS:
        e = engrams[layer]
        emb = e.embed._reduce_owned_rows(
            e.embed._owned_rows(hash_ids[:, e.layer_hash_index])
        )
        rows.append(
            flashinfer_mxfp8_blockscaled_linear(
                input=emb.flatten(-2),
                weight=e.wkv.weight,
                weight_scale=e.wkv.weight_scale_inv_swizzled,
                backend="cute-dsl",
                pin_tactic=True,
            )
        )
    return torch.stack(rows)


@needs_dist
@pytest.mark.parametrize("rows", [1, 6, 8])
def test_the_exchange_gives_sglangs_rows(env, rows):
    from sgl_nvidia_mega.deepseek_v41.engram import prefetch as ep

    engrams = _engrams(env.rank)
    first = min(engrams)
    stock = ep.EngramPrefetch(engrams, env.group, first)
    publish = ep.EngramPrefetch(engrams, env.group, first, DEADLINE_S)
    assert publish.project_at == {1: None, 14: 10}
    ids = torch.zeros(rows, dtype=torch.int64, device="cuda")
    shape = (len(ENGRAM_LAYERS), rows, N)
    want = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
    got = torch.empty_like(want)
    delays = ["none", "side"] + [f"rank{r}" for r in range(env.tp)]
    reference = _graph(env, stock, ids, want, "none")
    graphs = {
        late: _graph(
            env,
            publish,
            ids,
            got,
            "main" if late == f"rank{env.rank}" else late,
        )
        for late in delays
    }
    gen = torch.Generator().manual_seed(rows * 3)
    checked = 0
    for late, graph in graphs.items():
        for _ in range(REPLAYS):
            ids.copy_(torch.randint(0, 129280, (rows,), generator=gen))
            got.fill_(float("nan"))
            reference.replay()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(got, want), late
            inline = _inline(engrams, _hash_ids(ids, env.tp))
            torch.cuda.synchronize()
            assert torch.equal(got.view(torch.int16), inline.view(torch.int16)), late
            checked += 1
    # eager steps (as build runs them) after the graphs: the epochs stay in step
    for _ in range(3):
        ids.copy_(torch.randint(0, 129280, (rows,), generator=gen))
        rows_by_layer = publish.run(_hash_ids(ids, env.tp))
        inline = _inline(engrams, _hash_ids(ids, env.tp))
        torch.cuda.synchronize()
        for j, layer in enumerate(ENGRAM_LAYERS):
            assert torch.equal(
                rows_by_layer[layer].view(torch.int16), inline[j].view(torch.int16)
            )
    assert checked == len(delays) * REPLAYS


if __name__ == "__main__":
    from sglang.test.kernels.utils import multigpu_pytest_main

    multigpu_pytest_main(__name__, __file__, num_gpus=(4,), timeout=2400)

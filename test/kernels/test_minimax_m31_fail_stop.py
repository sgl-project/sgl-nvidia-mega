# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fail-stop waits of the MiniMax-M3.1 kernels on 4 SM100-family GPUs (GB300): when one
rank never joins a call, the ranks that wait on it trap within their deadline instead of
hanging: MegaAttention's TP4 mailbox, the token-parallel MoE's exchange and the draft
backbone's TP4 sums. The operands are synthetic: only the waits are under test.

Each case runs its own torchrun, since a faulted rank leaves its CUDA context unusable.
Run with ``python test/kernels/test_minimax_m31_fail_stop.py``.
"""

import os
import subprocess
import sys
import textwrap
import time

import pytest
import torch

FAULT_DEADLINE_S = 2.0
WORKER = textwrap.dedent(
    """
    import os, sys, time
    from datetime import timedelta
    import torch
    import torch.distributed as dist

    case, deadline = sys.argv[1], float(sys.argv[2])
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    dev = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(dev)
    dist.init_process_group("cpu:gloo,cuda:nccl", timeout=timedelta(minutes=5), device_id=dev)
    group, cpu = dist.group.WORLD, dist.new_group(backend="gloo")
    ns = int(deadline * 1e9)
    g = torch.Generator(device=dev).manual_seed(rank)
    fp8, u8, bf16 = torch.float8_e4m3fn, torch.uint8, torch.bfloat16

    def randn(shape, dtype, std=1.0):
        return (torch.randn(shape, generator=g, device=dev) * std).to(dtype)

    def bytes_(shape):
        return torch.randint(0, 256, shape, generator=g, dtype=u8, device=dev)

    def full(shape, value, dtype=u8):
        return torch.full(shape, value, dtype=dtype, device=dev)

    if case == "mega_attention":
        from sgl_nvidia_mega.minimax_m31.sm103.attention import mega_attention as ma

        pages, kv_len = 16, 1000
        weights = ma.AttentionWeights(
            projection=randn((2560, 6144), fp8, 0.05),
            projection_sf=full((2560 * 192,), 120),
            output_projection=randn((6144, 2048), fp8, 0.05),
            output_projection_sf=full((6144 * 64,), 120),
            **{n: full((6144,), 0, bf16) for n in ("input_norm", "post_attention_norm")},
            **{n: full((128,), 0, bf16) for n in ("q_norm", "k_norm", "qi_norm", "ki_norm")},
        )
        one = 0x38  # E4M3 1.0
        cache = ma.AttentionCache(
            **{n: bytes_((pages, 1, 128, 64)) for n in ("k", "v", "ki")},
            **{n: full((pages, 1, 128, 8), one).view(fp8) for n in ("k_scale", "v_scale", "ki_scale")},
        )
        positions = torch.arange(kv_len - 8, kv_len, device=dev)
        table = torch.arange(pages, dtype=torch.int32, device=dev)
        cos_sin = torch.cat((full((kv_len, 32), 1, bf16), full((kv_len, 32), 0, bf16)), 1)
        step = ma.AttentionStep(
            positions=positions, slot_mapping=positions.clone(),
            kv_len=full((1,), kv_len, torch.int32), kv_indices=table.view(1, -1),
            page_table=table.clone(), cos_sin=cos_sin,
        )
        x = randn((8, 6144), bf16)
        kernel = ma.MegaAttention(
            ma.create_workspace(group, dev, ns), weights, cache, step, ma.AttentionScratch(dev),
            x, torch.empty_like(x), torch.empty_like(x), wait_timeout_ns=ns,
        )
    elif case == "moe":
        from sgl_nvidia_mega.minimax_m31.sm103.moe import token_parallel as tp

        tensors = {}
        for name, dtype, shape in tp.TENSORS:
            if name.endswith("_sf"):
                tensors[name] = full(shape, 120) if dtype is u8 else full(shape, 0x30).view(fp8)
            elif dtype is u8:
                tensors[name] = bytes_(shape)
            elif name.endswith("alpha"):
                tensors[name] = full(shape, 0.01, torch.float32)
            else:
                tensors[name] = randn(shape, dtype, 0.05)
        kernel = tp.TokenParallelMoe(
            tp.create_arena(group, dev), wait_timeout_ns=ns,
            output=torch.empty((8, 6144), dtype=bf16, device=dev), **tensors,
        )
    elif case == "draft":
        from sgl_nvidia_mega.minimax_m31.sm103.draft import backbone as bb

        pages, kv_len, vocab = 8, 600, 1024
        layers = []
        for _ in range(bb.LAYERS):
            fields = {}
            for name, dtype, shape in bb.LAYER_TENSORS[:-2]:
                if dtype is bf16:
                    fields[name] = full(shape, 0, bf16)
                elif dtype is u8:
                    fields[name] = full(shape, 120)
                else:
                    fields[name] = randn(shape, dtype, 0.05)
            layers.append(bb.DraftLayerWeights(**fields))
        caches = [randn((pages * 128, 128), bf16) for _ in range(2 * bb.LAYERS)]
        positions = torch.arange(kv_len - 7, kv_len, device=dev)
        inputs = bb.DraftBackboneInputs(
            input_ids=torch.arange(7, device=dev), positions=positions,
            slot_mapping=positions.clone(),
            page_table=torch.arange(pages, dtype=torch.int32, device=dev),
            kv_len=full((1,), kv_len, torch.int32),
            cos_sin=torch.cat((torch.ones(kv_len, 32), torch.zeros(kv_len, 32)), 1).to(dev),
            k_caches=caches[: bb.LAYERS], v_caches=caches[bb.LAYERS :],
        )
        weights = bb.DraftBackboneWeights(
            layers=layers, embed=randn((vocab, 6144), bf16), vocab_start=rank * vocab,
            final_norm=full((6144,), 0, bf16), eps=1e-6,
        )
        kernel = bb.DraftBackbone(
            bb.create_arena(group, dev), weights, inputs,
            torch.empty((7, 6144), dtype=bf16, device=dev), wait_timeout_ns=ns,
        )
    kernel.run(); torch.cuda.synchronize()  # every rank joins once: built and healthy
    dist.barrier(cpu)
    print(f"RANK{rank} HEALTHY", flush=True)
    if rank == world - 1:
        time.sleep(4 * deadline + 20)  # never joins the second call
        os._exit(0)
    t0 = time.time()
    try:
        kernel.run(); torch.cuda.synchronize()
        print(f"RANK{rank} NO-FAULT", flush=True)
    except Exception as e:
        print(f"RANK{rank} FAULTED after {time.time() - t0:.1f} s: {type(e).__name__}", flush=True)
    os._exit(0)
    """
)


def _gpu4() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 10
        and torch.cuda.device_count() >= 4
    )


@pytest.mark.skipif(not _gpu4(), reason="needs 4 SM100-family GPUs")
@pytest.mark.parametrize("case", ["mega_attention", "moe", "draft"])
def test_a_missing_rank_faults_its_peers_in_time(case, tmp_path):
    script = tmp_path / "worker.py"
    script.write_text(WORKER)
    port = 29500 + (os.getpid() + 37 * len(case)) % 2000
    t0 = time.time()
    out = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=4",
         f"--master_port={port}", str(script), case, str(FAULT_DEADLINE_S)],
        capture_output=True, text=True, timeout=900,
    )  # fmt: skip
    log = out.stdout + out.stderr
    assert log.count("HEALTHY") == 4, log[-4000:]
    # the waiting ranks trap (a CUDA error); none returns as if the call had completed
    assert "NO-FAULT" not in log, log[-4000:]
    faulted = [line for line in log.splitlines() if "FAULTED after" in line]
    assert len(faulted) == 3, log[-4000:]
    for line in faulted:
        assert (
            float(line.split("after ")[1].split(" s")[0]) < 4 * FAULT_DEADLINE_S + 10
        ), line
    assert time.time() - t0 < 600


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))

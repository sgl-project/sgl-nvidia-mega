# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot wo_b_publish_split: muLLM's split-K wo_b that publishes into the Lamport mailbox
(sm103/attention/wo_b_split.py) on SM100-family GPUs (GB300) with NVLink multicast, at
any world size (attention TP8 over two trays: 8 ranks; one tray: 4).

- Its rows against an FP64 reference of the same MXFP8 operands and against FlashInfer's
  low-latency GEMM (the current hop A path), 1-8 rows: within tolerance (the FP32 sums are
  ordered differently, so not bitwise).
- The mailbox it writes: the hop's consumer on the split's published rows equals the same
  consumer on the same rows published by the hop's separate publish kernel, bit for bit,
  over sequences that mix the split, the publishing GEMM and the separate publish across
  the three generations, with each rank late in turn, and from one CUDA graph replayed
  with a late rank.

Run with ``torchrun --nproc_per_node=4 [--nnodes=2 ...] -m pytest
test/kernels/test_deepseek_v41_wo_b_split.py -s``.
"""

import os
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.distributed import parallel_state as ps
from sglang.test.test_utils import publish_build_topology

pytestmark = [
    pytest.mark.skipif(
        "RANK" not in os.environ,
        reason="run via torchrun (see the module docstring)",
    ),
    pytest.mark.skipif(
        not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10),
        reason="Requires SM100/SM103",
    ),
]

# DeepSeek-V4.1-Flash: hidden 5120; wo_b's K per rank at attention TP8 (1 group of 1024),
# and TP4's 2048
HIDDEN, HC = 5120, 4
KS = (1024, 2048)
ROWS = (1, 2, 5, 6, 8)


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


def block_linear(k, device, gen):
    """An MXFP8 wo_b as SGLang loads DeepSeek-V4.1's (E4M3 codes, UE8M0 scales uniform
    over 32x32 blocks, FlashInfer's 128x4 swizzle) and its FP64 dequant."""
    from sgl_nvidia_mega.deepseek_v41.sm103.attention import q_cluster

    codes = (
        (torch.randn(HIDDEN, k, device=device, generator=gen) * 64)
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
    )
    exp = torch.randint(-14, -6, (HIDDEN // 32, k // 32), device=device, generator=gen)
    row_scales = (exp + 127).to(torch.uint8).repeat_interleave(32, dim=0)
    dequant = codes.double() * torch.exp2(
        exp.double().repeat_interleave(32, 0).repeat_interleave(32, 1)
    )
    linear = SimpleNamespace(
        weight=codes.contiguous(),
        weight_scale_inv_swizzled=q_cluster.swizzle_scales(row_scales).contiguous(),
        bias=None,
    )
    return linear, dequant


def activation(rows, k, device, gen):
    """The fused wo_a's MXFP8 rows (FlashInfer's quantizer, 128x4 scales) and their
    FP64 dequant."""
    from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize

    from sgl_nvidia_mega.deepseek_v41.sm103.attention import q_cluster

    x = torch.randn(rows, k, device=device, generator=gen).bfloat16()
    if rows > 1:
        x[-1] = 0  # zero outputs, whose negative zeros the publish clears
    q, sf = flashinfer_mxfp8_quantize(x, is_sf_swizzled_layout=True, alignment=32)
    scales = q_cluster.unswizzle_scales(sf.reshape(-1), max(rows, 1), k)[:rows]
    dequant = q.view(torch.float8_e4m3fn).double() * torch.exp2(
        scales.double().repeat_interleave(32, 1) - 127
    )
    return (q, sf.reshape(-1)), dequant


def flashinfer_gemm(linear, qa):
    from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll

    a, a_scale = qa
    return mxfp8_ll.linear(a, a_scale, linear.weight, linear.weight_scale_inv_swizzled)


def mixes(rows, device, rank):
    gen = torch.Generator(device=device).manual_seed(7 + rank)
    return (
        torch.randn(rows, HC, HIDDEN, device=device, generator=gen).bfloat16(),
        torch.rand(rows, HC, device=device, generator=gen),
        torch.randn(rows, HC, HC, device=device, generator=gen) * 0.1,
        torch.rand(rows, HC, device=device, generator=gen),
        torch.randn(HIDDEN, device=device, generator=gen).bfloat16(),
    )


def rel_rms(a, b):
    return (
        (a.double() - b.double()).square().mean().sqrt()
        / b.double().square().mean().sqrt()
    ).item()


def test_wo_b_split_rows_and_mailbox(rank):
    from sgl_nvidia_mega.deepseek_v41.sm103.all_reduce_mhc import AllReduceMHC
    from sgl_nvidia_mega.deepseek_v41.sm103.runner import SplitPublisher
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
        split = SplitPublisher(arm, "test")
        publisher = PublishingMxfp8Linear(arm)
        gen = torch.Generator(device=device).manual_seed(1000 + rank)

        def boundary(x, mix):
            residual, post, comb, pre, norm_weight = mix
            return arm(x, residual, post, comb, pre, norm_weight, 1e-6)

        for k in KS:
            layers = [block_linear(k, device, gen) for _ in range(2)]
            for linear, _ in layers:
                assert split.add(linear) == ""
            for rows in ROWS:
                publisher.precompile(layers[0][0], rows)
                acts = [activation(rows, k, device, gen) for _ in layers]
                mix = mixes(rows, device, rank)
                # rows: FP64 reference and the current GEMM, within tolerance
                for (linear, w64), (qa, x64) in zip(layers, acts):
                    y = split(linear, qa)
                    assert y is not None and y.shape == (rows, HIDDEN)
                    boundary(y, mix)  # consume the publish
                    ref = x64 @ w64.T
                    cur = flashinfer_gemm(linear, qa)
                    e_split, e_cur = rel_rms(y, ref), rel_rms(cur, ref)
                    if rank == 0:
                        print(
                            f"k={k} rows={rows}: rel RMS vs fp64 split {e_split:.2e} "
                            f"current {e_cur:.2e}; split vs current {rel_rms(y, cur):.2e}, "
                            f"bitwise equal {torch.equal(y.view(torch.int16), cur.view(torch.int16))}"
                        )
                    assert e_split < 4e-3, (k, rows, e_split)
                    assert e_split < 1.5 * e_cur + 1e-3, (k, rows, e_split, e_cur)
                    torch.testing.assert_close(
                        y.float(), cur.float(), atol=2e-2, rtol=2e-2
                    )
                expected = []
                for (linear, _), (qa, _) in zip(layers, acts):
                    x = split(linear, qa)
                    boundary(x, mix)
                    expected.append(x.clone())
                # 0: split, 1: publishing GEMM, 2: separate publish of the split's rows
                pattern = (0, 0, 2, 1, 0, 2, 2, 0, 1, 0, 0)
                for late in range(tp_size):
                    for index, path in enumerate(pattern):
                        li = index % len(layers)
                        linear, qa = layers[li][0], acts[li][0]
                        if rank == late:
                            torch.cuda._sleep(20_000)
                        if path == 0:
                            x = split(linear, qa)
                            assert torch.equal(
                                x.view(torch.int16), expected[li].view(torch.int16)
                            ), f"rows differ run to run: k={k} rows={rows}"
                            got = boundary(x, mix)
                            reference = boundary(expected[li].clone(), mix)
                        elif path == 1:
                            x = publisher(linear, publisher.prepare(linear, qa))
                            got = boundary(x, mix)
                            reference = boundary(x.clone(), mix)
                        else:
                            got = boundary(expected[li].clone(), mix)
                            x = split(linear, qa)
                            reference = boundary(x, mix)
                        for a, b in zip(got, reference):
                            assert torch.equal(
                                a.view(torch.int16), b.view(torch.int16)
                            ), (
                                f"boundary differs: k={k} rows={rows} late={late} "
                                f"step={index} path={path}"
                            )
                    torch.cuda.synchronize()

                # one CUDA graph of split and separate boundaries, replayed with a late rank
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
                    captured = []
                    for index, path in enumerate(pattern):
                        li = index % len(layers)
                        if path == 0:
                            x = split(layers[li][0], acts[li][0])
                        else:
                            x = expected[li].clone()
                        captured.append(boundary(x, mix))
                torch.cuda.current_stream().wait_stream(stream)
                references = [
                    [
                        t.clone()
                        for t in boundary(expected[index % len(layers)].clone(), mix)
                    ]
                    for index in range(len(pattern))
                ]
                for replay in range(6):
                    if rank == replay % tp_size:
                        torch.cuda._sleep(50_000)
                    graph.replay()
                    torch.cuda.synchronize()
                    for index, outputs in enumerate(captured):
                        for a, b in zip(outputs, references[index]):
                            assert torch.equal(
                                a.view(torch.int16), b.view(torch.int16)
                            ), (
                                f"graph boundary differs: k={k} rows={rows} "
                                f"replay={replay} step={index}"
                            )
                torch.cuda.synchronize()
                del graph
        torch.distributed.barrier(group=ps.get_tp_group().cpu_group)

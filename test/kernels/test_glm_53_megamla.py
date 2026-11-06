# SPDX-License-Identifier: Apache-2.0
"""GLM5 MLA request shapes against an independent native-weight reference.

Reference budgets below validate the standalone kernels, not service NUMERICS.
Set GLM5_REQUIRE_KERNELS=1 to treat an incomplete source installation as a failure.
"""

from __future__ import annotations

import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from sgl_nvidia_mega.glm_5.attention.step import cache_views
from sgl_nvidia_mega.glm_5.plan import MLA_CONFIGS, ExecutionPlan
from sgl_nvidia_mega.glm_5.runtime.state import KernelState
from sgl_nvidia_mega.glm_5.runtime.weights import MlaWeights
from sgl_nvidia_mega.glm_5.sm103 import mla
from sgl_nvidia_mega.glm_5.sm103.jit import KernelSourcesMissing, source_bundle


def _require_core(plan, world):
    try:
        source_bundle(plan.variant("mla"))
    except KernelSourcesMissing as error:
        if os.environ.get("GLM5_REQUIRE_KERNELS") == "1":
            pytest.fail(str(error))
        pytest.skip(str(error))
    if not torch.cuda.is_available() or torch.cuda.device_count() < world:
        pytest.skip(f"needs {world} CUDA GPU(s)")
    if any(torch.cuda.get_device_capability(i) != (10, 3) for i in range(world)):
        pytest.skip("the initial GLM5 kernel target is SM103")


def _fp8(shape, generator, device):
    return (torch.randn(shape, generator=generator, device=device) * 0.1).to(
        torch.float8_e4m3fn
    )


def _case(plan, device, rank, layout):
    t, h, heads, capacity = plan.rows, 6144, 64 // plan.tp_size, 2112 * plan.batch_size
    common = torch.Generator(device=device).manual_seed(41)
    local = torch.Generator(device=device).manual_seed(71 + rank)
    rand = lambda shape: torch.randn(shape, generator=common, device=device)
    scale = lambda shape: (
        torch.rand(shape, generator=common, device=device) * 0.01 + 0.015
    )
    gamma = lambda n: torch.rand(n, generator=common, device=device) * 0.2 + 0.9
    weights = MlaWeights(
        gamma(h),
        _fp8((2048, h), common, device),
        scale((16, 48)),
        _fp8((576, h), common, device),
        scale((5, 48)),
        gamma(2048),
        _fp8((heads * 256, 2048), local, device),
        scale((heads * 2, 16)),
        gamma(512),
        _fp8((heads * 448, 512), local, device),
        scale((heads * 7 // 2, 4)),
        _fp8((h, heads * 256), local, device),
        scale((48, heads * 2)),
    )
    storage = _fp8((capacity, 576), common, device)
    kv, pe = cache_views(storage)
    positions = torch.cat(
        [
            torch.arange(
                (request + 1) * 2112 - plan.draft_tokens,
                (request + 1) * 2112,
                dtype=torch.int64 if layout == "sglang" else torch.int32,
                device=device,
            )
            for request in range(plan.batch_size)
        ]
    )
    phase = rand((capacity, 32)) * 0.02
    if layout == "sglang":
        rope = torch.cat((phase.cos(), phase.sin()), dim=-1)
        rope_positions = torch.arange(17, 17 + t, dtype=torch.int64, device=device)
    else:
        kv, pe = kv.contiguous(), pe.contiguous()
        rope = torch.stack((phase[:t].cos(), phase[:t].sin()), dim=-1).reshape(t, 64)
        rope_positions = None
    return {
        "x": rand(plan.shape).bfloat16(),
        "weights": weights,
        "selected_indices": torch.stack(
            [
                torch.randperm(
                    2112, generator=common, device=device, dtype=torch.int32
                )[:2048]
                + (token // plan.draft_tokens) * 2112
                for token in range(t)
            ]
        ),
        "selected_counts": torch.full((t,), 2048, dtype=torch.int32, device=device),
        "kv_cache": kv,
        "pe_cache": pe,
        "cache_positions": positions,
        "rope": rope,
        "rope_positions": rope_positions,
        "residual": rand(plan.shape).bfloat16(),
        "local_output": torch.empty(plan.shape, dtype=torch.bfloat16, device=device),
        "final_output": torch.empty(plan.shape, dtype=torch.bfloat16, device=device),
    }


def _linear(x, weight, scales):
    rows, cols = weight.shape
    blocks = cols // 128
    partials = torch.einsum(
        "tkb,mkb->tmk",
        x.float().reshape(-1, blocks, 128),
        weight.float().reshape(rows, blocks, 128),
    )
    expanded = scales.repeat_interleave(128, dim=0)[:rows]
    return (partials * expanded[None]).sum(dim=-1)


def _rms(x, gamma, dtype):
    return (
        x.float()
        * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5)
        * gamma
    ).to(dtype)


def _dequant(weight, scales):
    return (
        weight.float()
        * scales.repeat_interleave(128, 0).repeat_interleave(128, 1)[
            : weight.shape[0], : weight.shape[1]
        ]
    )


def _rope(x, cosine, sine, dtype):
    pairs = x.float().reshape(*x.shape[:-1], 32, 2)
    first = pairs[..., 0] * cosine - pairs[..., 1] * sine
    second = pairs[..., 0] * sine + pairs[..., 1] * cosine
    return torch.stack((first, second), dim=-1).flatten(-2).to(dtype)


@torch.inference_mode()
def _reference(case, dtype):
    """Native block-scale projections, RoPE, sparse softmax and absorbed V/O."""
    w, x = case["weights"], case["x"].reshape(-1, 6144)
    heads = w.q_b_weight.shape[0] // 256
    normalized = _rms(x, w.x_gamma, dtype)
    q_lora = _linear(normalized, w.q_a_weight, w.q_a_scale_inv).to(dtype)
    kv_a = _linear(normalized, w.kv_a_with_mqa_weight, w.kv_a_with_mqa_scale_inv).to(
        dtype
    )
    q = _linear(_rms(q_lora, w.q_gamma, dtype), w.q_b_weight, w.q_b_scale_inv)
    q = q.to(dtype).reshape(-1, heads, 256)
    if case["rope_positions"] is None:
        pairs = case["rope"].reshape(-1, 32, 2)
        cosine, sine = pairs[..., 0], pairs[..., 1]
    else:
        table = case["rope"].index_select(0, case["rope_positions"])
        cosine, sine = table[:, :32], table[:, 32:]
    kv_rows = _rms(kv_a[:, :512], w.kv_gamma, dtype).to(torch.float8_e4m3fn).float()
    pe_rows = _rope(kv_a[:, 512:], cosine, sine, dtype).to(torch.float8_e4m3fn).float()
    kv, pe = case["kv_cache"].float().clone(), case["pe_cache"].float().clone()
    positions = case["cache_positions"].long()
    kv[positions], pe[positions] = kv_rows, pe_rows
    kv_b = _dequant(w.kv_b_weight, w.kv_b_scale_inv).reshape(heads, 448, 512)
    q_latent = torch.einsum("thc,hcd->thd", q[..., :192].float(), kv_b[:, :192]).to(
        dtype
    )
    q_pe = _rope(q[..., 192:], cosine[:, None], sine[:, None], dtype)
    outputs = []
    for token, count in enumerate(case["selected_counts"].tolist()):
        if count:
            indices = case["selected_indices"][token, :count].long()
            selected_kv, selected_pe = kv[indices], pe[indices]
            scores = (
                q_latent[token].float() @ selected_kv.T
                + q_pe[token].float() @ selected_pe.T
            ) * (1 / 16)
            latent = (scores.softmax(-1) @ selected_kv).to(dtype)
        else:
            latent = torch.zeros((heads, 512), dtype=dtype, device=x.device)
        outputs.append(torch.einsum("hd,hvd->hv", latent.float(), kv_b[:, 192:]))
    absorbed = torch.stack(outputs).to(dtype).flatten(1)
    local = (
        _linear(absorbed, w.o_proj_weight, w.o_proj_scale_inv)
        .bfloat16()
        .reshape_as(case["x"])
    )
    return local, kv_rows, pe_rows


def _final_reference(local, residual, world):
    partials = [torch.empty_like(local) for _ in range(world)]
    if world > 1:
        dist.all_gather(partials, local)
    else:
        partials[0] = local
    total = torch.zeros_like(local, dtype=torch.float32)
    for partial in partials:
        total += partial.float()
    return (total.bfloat16().float() + residual.float()).bfloat16()


def _assert_case(case, state, expected, world, generation):
    local, expected_kv, expected_pe = expected
    torch.testing.assert_close(case["local_output"], local, rtol=2e-2, atol=2e-3)
    if torch.any(case["selected_counts"] > 0):
        error = (case["local_output"].float() - local.float()).norm()
        assert error <= 0.05 * local.float().norm().clamp_min(1e-8)
        assert torch.count_nonzero(case["local_output"]) > 0
    empty = case["selected_counts"] == 0
    if torch.any(empty):
        assert torch.count_nonzero(case["local_output"].reshape(-1, 6144)[empty]) == 0
    torch.testing.assert_close(
        case["final_output"],
        _final_reference(case["local_output"], case["residual"], world),
        rtol=0,
        atol=0,
    )
    pos = case["cache_positions"].long()
    for actual, wanted in (
        (case["kv_cache"].float()[pos], expected_kv),
        (case["pe_cache"].float()[pos], expected_pe),
    ):
        assert torch.isfinite(actual).all()
        assert torch.all((actual - wanted).abs() <= wanted.abs() * 0.125 + 0.03125)
    assert torch.equal(state.epochs, torch.full_like(state.epochs, generation))


@torch.inference_mode()
def _worker(rank, world, draft_tokens, tp_size, batch_size, layout, port):
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("highest")
    if world > 1:
        dist.init_process_group(
            "nccl",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=world,
            timeout=timedelta(seconds=120),
        )
    try:
        kernel = mla.load(
            draft_tokens=draft_tokens, tp_size=tp_size, batch_size=batch_size
        )
        state = KernelState.allocate(kernel.metadata, device)
        case = _case(kernel.plan, device, rank, layout)
        bound = kernel.bind(
            case["weights"],
            state=state,
            kv_cache=case["kv_cache"],
            pe_cache=case["pe_cache"],
            layout=layout,
        )
        assert bound.weights is case["weights"]
        assert bound.kv_cache.data_ptr() == case["kv_cache"].data_ptr()
        dynamic = {
            name: tensor
            for name, tensor in case.items()
            if name not in ("weights", "kv_cache", "pe_cache")
        }
        # Capture the very first execution, with no eager warmup.
        first_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(first_graph):
            bound(**dynamic)
        dtype = torch.bfloat16 if kernel.native_cvt else torch.float16
        counts = (
            (2048,),
            (1, 63, 64, 65, 127, 128, 129, 2048),
            (0, 127, 1024, 2048),
            (17, 129, 511, 1537),
            (2048, 65, 0, 1025),
        )
        for generation, values in enumerate(counts, 1):
            case["selected_counts"].copy_(
                torch.tensor(
                    [values[i % len(values)] for i in range(kernel.plan.rows)],
                    dtype=torch.int32,
                    device=device,
                )
            )
            if generation > 1:
                case["x"].mul_(0.9).add_(0.01)
            expected = _reference(case, dtype)
            if generation == 1:
                first_graph.replay()
            else:
                bound(**dynamic)
            torch.cuda.synchronize(device)
            _assert_case(case, state, expected, world, generation)
        # Read capture's epoch baseline; every replay must advance it exactly once.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            bound(**dynamic)
        torch.cuda.synchronize(device)
        baseline = state.epochs.cpu()
        assert torch.all(baseline == baseline[0])
        for generation in (int(baseline[0]) + 1, int(baseline[0]) + 2):
            case["x"].mul_(0.9).add_(0.02)
            expected = _reference(case, dtype)
            graph.replay()
            torch.cuda.synchronize(device)
            _assert_case(case, state, expected, world, generation)
    finally:
        if world > 1:
            dist.destroy_process_group()


@pytest.mark.parametrize(("draft_tokens", "tp_size", "batch_size"), MLA_CONFIGS)
@pytest.mark.parametrize("layout", ("sglang", "legacy"))
@pytest.mark.parametrize("mode", ("local", "tp"))
def test_mla(draft_tokens, tp_size, batch_size, layout, mode):
    plan = ExecutionPlan(draft_tokens, tp_size, batch_size)
    world = tp_size if mode == "tp" else 1
    _require_core(plan, world)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    args = (world, draft_tokens, tp_size, batch_size, layout, port)
    if world == 1:
        _worker(0, *args)
    else:
        torch.multiprocessing.spawn(_worker, args=args, nprocs=world, join=True)

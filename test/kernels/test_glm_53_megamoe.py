# SPDX-License-Identifier: Apache-2.0
"""GLM5 native FP8 MoE request shapes, with the PyTorch reference in this file.

GLM5_REQUIRE_KERNELS=1 treats an incomplete source installation as an error.
The native block-scale reference validates standalone kernel outputs.
"""

from __future__ import annotations

import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from sgl_nvidia_mega.glm_5.plan import MOE_CONFIGS, ExecutionPlan
from sgl_nvidia_mega.glm_5.runtime.state import KernelState
from sgl_nvidia_mega.glm_5.runtime.weights import MoeWeights
from sgl_nvidia_mega.glm_5.sm103 import moe
from sgl_nvidia_mega.glm_5.sm103.jit import KernelSourcesMissing, source_bundle


def _require_core(plan, world):
    try:
        source_bundle(plan.variant("moe"))
    except KernelSourcesMissing as error:
        if os.environ.get("GLM5_REQUIRE_KERNELS") == "1":
            pytest.fail(str(error))
        pytest.skip(str(error))
    if not torch.cuda.is_available() or torch.cuda.device_count() < world:
        pytest.skip(f"needs {world} CUDA GPU(s)")
    if any(torch.cuda.get_device_capability(i) != (10, 3) for i in range(world)):
        pytest.skip("the initial GLM5 kernel target is SM103")


_ACTIVE_EXPERTS = (0, 1, 31, 32, 63, 64, 127, 128, 191, 192, 254, 255)


def _expert_weights(shape, generator, device):
    """Bound temporary memory; inactive experts stay zero but preserve native ABI."""
    result = torch.empty(shape, dtype=torch.float8_e4m3fn, device=device)
    result.view(torch.uint8).zero_()
    for expert in (*_ACTIVE_EXPERTS, 256):
        values = torch.randn(
            shape[1:], dtype=torch.bfloat16, generator=generator, device=device
        )
        result[expert].copy_((values * 0.5).to(torch.float8_e4m3fn))
    return result


def _case(plan, device, rank):
    t, inter = plan.rows, 2048 // plan.tp_size
    common = torch.Generator(device=device).manual_seed(113)
    local = torch.Generator(device=device).manual_seed(173 + rank)
    rand = lambda shape: torch.randn(shape, generator=common, device=device)
    scale = lambda shape: (
        torch.rand(shape, generator=local, device=device) * 0.01 + 0.015
    )
    bias = torch.full((256,), -64.0, dtype=torch.float32, device=device)
    bias[list(_ACTIVE_EXPERTS)] = torch.linspace(
        0.0, 0.01, len(_ACTIVE_EXPERTS), device=device
    )
    weights = MoeWeights(
        torch.rand(6144, generator=common, device=device) * 0.2 + 0.9,
        (rand((256, 6144)) * 0.01).bfloat16(),
        bias,
        _expert_weights((257, 2 * inter, 6144), local, device),
        scale((257, 2 * inter // 128, 48)),
        _expert_weights((257, 6144, inter), local, device),
        scale((257, 48, inter // 128)),
    )
    return {
        "x": rand(plan.shape).bfloat16(),
        "weights": weights,
        "residual": rand(plan.shape).bfloat16(),
        "local_output": torch.empty(plan.shape, dtype=torch.bfloat16, device=device),
        "final_output": torch.empty(plan.shape, dtype=torch.bfloat16, device=device),
        "selected_indices": torch.empty(
            (plan.batch_size, plan.draft_tokens, 8), dtype=torch.int32, device=device
        ),
        "selected_probs": torch.empty(
            (plan.batch_size, plan.draft_tokens, 8), dtype=torch.float32, device=device
        ),
    }


def _block_matmul(x, weight, scale):
    """Apply one FP32 block scale after each 128-wide dot-product partial."""
    rows, cols = weight.shape
    partials = torch.einsum(
        "mkb,kb->mk",
        weight.float().reshape(rows, cols // 128, 128),
        x.float().reshape(cols // 128, 128),
    )
    return (partials * scale.repeat_interleave(128, dim=0)).sum(-1)


@torch.inference_mode()
def _reference(case):
    """Routed 0..255 then shared 256, Top8 sigmoid, FP32 sum then BF16 local."""
    x, weights = case["x"].reshape(1, -1, 6144), case["weights"]
    normalized = (
        x.float()
        * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5)
        * weights.rms_gamma
    ).bfloat16()
    raw = F.linear(normalized.float(), weights.router_weight.float()).sigmoid()
    raw = torch.nan_to_num(raw, nan=0.0)
    ranking = torch.nan_to_num(
        raw + weights.router_bias, nan=-torch.finfo(torch.float32).max
    )
    selected = ranking.argsort(dim=-1, descending=True, stable=True)[..., :8]
    probs = raw.gather(-1, selected)
    total = probs.sum(-1, keepdim=True)
    probs = torch.where(total > 0, probs * (2.5 / total), torch.zeros_like(probs))
    inter = weights.up_gate_weight.shape[1] // 2
    output = []
    for token in range(x.shape[1]):
        accumulator = torch.zeros(6144, dtype=torch.float32, device=x.device)
        experts = [256, *selected[0, token].tolist()]
        for slot, expert in enumerate(experts):
            gate = _block_matmul(
                normalized[0, token],
                weights.up_gate_weight[expert, :inter],
                weights.up_gate_scale[expert, : inter // 128],
            )
            up = _block_matmul(
                normalized[0, token],
                weights.up_gate_weight[expert, inter:],
                weights.up_gate_scale[expert, inter // 128 :],
            )
            activated = (F.silu(gate) * up).bfloat16()
            down = _block_matmul(
                activated, weights.down_weight[expert], weights.down_scale[expert]
            )
            accumulator += down if slot == 0 else down * probs[0, token, slot - 1]
        output.append(accumulator.bfloat16())
    return (
        torch.stack(output).reshape_as(case["x"]),
        selected.int().reshape_as(case["selected_indices"]),
        probs.reshape_as(case["selected_probs"]),
    )


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


def _assert_case(case, state, expected, world, generation, *, emit_local=True):
    local, indices, probs = expected
    torch.testing.assert_close(case["selected_indices"], indices, rtol=0, atol=0)
    torch.testing.assert_close(case["selected_probs"], probs, rtol=2e-4, atol=2e-5)
    if emit_local:
        torch.testing.assert_close(case["local_output"], local, rtol=1e-2, atol=2e-3)
    # Nullable output still computes a fresh payload; use the independent partial.
    torch.testing.assert_close(
        case["final_output"],
        _final_reference(
            case["local_output"] if emit_local else local, case["residual"], world
        ),
        rtol=0 if emit_local else 1e-2,
        atol=0 if emit_local else 2e-2,
    )
    assert torch.equal(state.epochs, torch.full_like(state.epochs, generation))


@torch.inference_mode()
def _worker(rank, world, draft_tokens, tp_size, batch_size, port):
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
        kernel = moe.load(
            draft_tokens=draft_tokens, tp_size=tp_size, batch_size=batch_size
        )
        state = KernelState.allocate(kernel.metadata, device)
        case = _case(kernel.plan, device, rank)
        bound = kernel.bind(case["weights"], state=state)
        assert bound.weights is case["weights"]
        dynamic = {name: tensor for name, tensor in case.items() if name != "weights"}
        first_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(first_graph):
            bound(**dynamic)
        for generation in range(1, 6):
            if generation > 1:
                case["x"].mul_(0.9).add_(0.01 * generation)
                case["residual"].mul_(0.8).add_(0.01)
            expected = _reference(case)
            if generation == 1:
                first_graph.replay()
            else:
                bound(**dynamic)
            torch.cuda.synchronize(device)
            _assert_case(case, state, expected, world, generation)
        case["x"].mul_(0.9).add_(0.013)
        case["residual"].add_(0.25)
        expected = _reference(case)
        nullable = {**dynamic, "local_output": None}
        bound(**nullable)
        torch.cuda.synchronize(device)
        _assert_case(case, state, expected, world, 6, emit_local=False)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            bound(**dynamic)
        torch.cuda.synchronize(device)
        baseline = state.epochs.cpu()
        assert torch.all(baseline == baseline[0])
        for generation in (int(baseline[0]) + 1, int(baseline[0]) + 2):
            case["x"].mul_(0.9).add_(0.02)
            expected = _reference(case)
            graph.replay()
            torch.cuda.synchronize(device)
            _assert_case(case, state, expected, world, generation)
    finally:
        if world > 1:
            dist.destroy_process_group()


@pytest.mark.parametrize(("draft_tokens", "tp_size", "batch_size"), MOE_CONFIGS)
@pytest.mark.parametrize("mode", ("local", "tp"))
def test_moe(draft_tokens, tp_size, batch_size, mode):
    plan = ExecutionPlan(draft_tokens, tp_size, batch_size)
    world = tp_size if mode == "tp" else 1
    _require_core(plan, world)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    args = (world, draft_tokens, tp_size, batch_size, port)
    if world == 1:
        _worker(0, *args)
    else:
        torch.multiprocessing.spawn(_worker, args=args, nprocs=world, join=True)

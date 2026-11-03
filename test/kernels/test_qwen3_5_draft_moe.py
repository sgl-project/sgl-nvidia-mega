# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The draft package's single-launch MoE (sm103/draft/moe.py) on one SM100-family GPU
(GB300), at the TP8 draft shape (hidden 4096, 512 experts of intermediate 128, top-10):
- the top-10 is exactly the kernel's own selection rule on the logits it published
  (bf16 order keys, ties to the lowest id), and the weights its renormalized softmax;
- logits, weights and output match a float64 reference with the kernel's rounding
  points (only the fp32 summation order differs);
- repeated calls and CUDA-graph replays give the same bytes;
- DraftMoe runs a two-token batch as two single launches.
Bitwise equality with the SGLang kernel it is ported from (on SGLang branch
nvidia/qwen35-agentx-gb300) is checked with that branch installed as ``sglang``, not here."""

from types import SimpleNamespace

import pytest
import torch

H, I, E, K = 4096, 128, 512, 10


def _gpu_ready() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        from cuda.bindings import nvrtc  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _gpu_ready(), reason="needs an SM100-family GPU")


@pytest.fixture(scope="module")
def weights():
    """One TP8 draft MoE block's bf16 weights in checkpoint layout."""
    g = torch.Generator(device="cuda").manual_seed(5)

    def randn(*shape, std):
        return (torch.randn(*shape, device="cuda", generator=g) * std).to(
            torch.bfloat16
        )

    return {
        "router_w": randn(E, H, std=0.02),
        "w13": randn(E, 2 * I, H, std=0.02),
        "w2": randn(E, H, I, std=0.05),
        "w13_shared": randn(2 * I, H, std=0.02),
        "w2_shared": randn(H, I, std=0.05),
        "gate_w": randn(H, std=0.02),
    }


def _x(seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(1000 + seed)
    return torch.randn(1, H, device="cuda", generator=g).to(torch.bfloat16)


def _outputs():
    return (
        torch.empty(1, K, dtype=torch.float32, device="cuda"),
        torch.empty(1, K, dtype=torch.int32, device="cuda"),
        torch.empty(1, H, dtype=torch.bfloat16, device="cuda"),
    )


def _run(w, x, scratch, outs):
    from sgl_nvidia_mega.qwen3_5.sm103.draft import moe

    topk_weights, topk_ids, out = outs
    moe.draft_moe_fused(
        x,
        w["router_w"],
        w["w13"],
        w["w2"],
        w["w13_shared"],
        w["w2_shared"],
        w["gate_w"],
        scratch,
        topk_weights,
        topk_ids,
        out,
    )
    return outs


def _kernel_logits(scratch) -> torch.Tensor:
    """The fp32 logits the call published (replica 0 of the tagged words)."""
    words = scratch[0][:E]
    return (words & 0xFFFFFFFF).to(torch.int32).view(torch.float32)


def _order_keys(logits: torch.Tensor) -> torch.Tensor:
    """draft_router_kernel's selection keys: bf16 order key, then the lower id."""
    v = torch.where(logits == logits, logits, torch.full_like(logits, -1e30))
    b = (v.to(torch.bfloat16).view(torch.int16).to(torch.int64)) & 0xFFFF
    b = torch.where(b == 0x8000, torch.zeros_like(b), b)
    ordered = torch.where((b & 0x8000) != 0, (~b) & 0xFFFF, b | 0x8000)
    ids = torch.arange(E, device=logits.device, dtype=torch.int64)
    return (ordered << 16) | (0xFFFF - ids)


def _r(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.bfloat16).double()


def _reference(w, x, ids, weights):
    """float64 dots with the kernel's bf16 rounding points."""
    xd = x.double().view(-1)
    acts = []
    for s in range(K + 1):
        rows = w["w13_shared"] if s == K else w["w13"][int(ids[s])]
        gate, up = _r(rows[:I].double() @ xd), _r(rows[I:].double() @ xd)
        acts.append(_r(gate / (1.0 + torch.exp(-gate)) * up))
    routed = torch.zeros(H, dtype=torch.float64, device="cuda")
    for s in range(K):
        routed = routed + _r(
            (w["w2"][int(ids[s])].double() @ acts[s]) * float(weights[s])
        )
    shared = _r(w["w2_shared"].double() @ acts[K])
    gate = torch.sigmoid(w["gate_w"].double() @ xd)
    return _r(gate * shared + _r(routed))


def test_selection_weights_and_output(weights):
    from sgl_nvidia_mega.qwen3_5.sm103.draft import moe

    scratch = moe.draft_moe_fused_scratch(E, K, I, torch.device("cuda"))
    for seed in range(4):
        x = _x(seed)
        topk_weights, topk_ids, out = _run(weights, x, scratch, _outputs())
        torch.cuda.synchronize()
        assert int(scratch[2]) == seed + 1  # the call epoch
        logits = _kernel_logits(scratch)
        assert torch.equal(
            (scratch[0][:E] >> 32).to(torch.int32),
            torch.full_like(logits, seed + 1).int(),
        )
        # the logits are bf16 values within a bf16 step of the float64 dot
        ref_logits = weights["router_w"].double() @ x.double().view(-1)
        assert torch.equal(logits.to(torch.bfloat16).float(), logits)
        assert (
            logits.double() - ref_logits
        ).abs().max() <= 2**-7 * ref_logits.abs().max()
        # the top-10 is the kernel's own rule on its own logits, exactly
        want = torch.topk(_order_keys(logits), K).values
        assert torch.equal(topk_ids.view(-1).long(), 0xFFFF - (want & 0xFFFF))
        probs = torch.softmax(logits.double(), dim=0)[topk_ids.view(-1).long()]
        assert torch.allclose(
            topk_weights.view(-1).double(), probs / probs.sum(), rtol=1e-5
        )
        ref = _reference(weights, x, topk_ids.view(-1), topk_weights.view(-1))
        err = (out.view(-1).double() - ref).abs()
        assert (
            err.max() <= 2e-2 * ref.abs().max()
            and err.mean() <= 2e-3 * ref.abs().mean()
        )


def test_repeat_and_graph_replay(weights):
    from sgl_nvidia_mega.qwen3_5.sm103.draft import moe

    scratch = moe.draft_moe_fused_scratch(E, K, I, torch.device("cuda"))
    eager = {}
    for seed in range(3):
        a = [t.clone() for t in _run(weights, _x(seed), scratch, _outputs())]
        b = [t.clone() for t in _run(weights, _x(seed), scratch, _outputs())]
        assert all(torch.equal(p, q) for p, q in zip(a, b))
        eager[seed] = a
    x = _x(0).clone()
    outs = _outputs()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run(weights, x, scratch, outs)
    for _ in range(2):
        for seed in range(3):
            x.copy_(_x(seed))
            graph.replay()
            torch.cuda.synchronize()
            assert all(torch.equal(p, q) for p, q in zip(outs, eager[seed]))


def test_draft_moe_two_tokens(weights):
    """DraftMoe on a loaded block's weights: two tokens are two launches in order."""
    # The draft package needs SGLang's NVIDIA megakernel contract.
    pytest.importorskip("sglang.kernels.plugin_kernels.nvidia.spec")
    from sgl_nvidia_mega.qwen3_5.draft.moe import DraftMoe

    mlp = SimpleNamespace(
        gate=SimpleNamespace(weight=weights["router_w"]),
        experts=SimpleNamespace(w13_weight=weights["w13"], w2_weight=weights["w2"]),
        shared_expert=SimpleNamespace(
            gate_up_proj=SimpleNamespace(weight=weights["w13_shared"]),
            down_proj=SimpleNamespace(weight=weights["w2_shared"]),
        ),
        shared_expert_gate=SimpleNamespace(weight=weights["gate_w"].view(1, H)),
        topk=SimpleNamespace(topk_config=SimpleNamespace(top_k=K)),
    )
    block = DraftMoe(mlp, torch.device("cuda"))
    assert block.ready(1) and block.ready(2) and block.single_launch(2)
    assert not block.ready(0) and not block.ready(3)
    x = torch.cat([_x(7), _x(8)])
    got = block(x)
    from sgl_nvidia_mega.qwen3_5.sm103.draft import moe

    scratch = moe.draft_moe_fused_scratch(E, K, I, torch.device("cuda"))
    for m in range(2):
        *_, out = _run(weights, x[m : m + 1], scratch, _outputs())
        assert torch.equal(got[m : m + 1], out)


def test_rejects_bad_operands(weights):
    from sgl_nvidia_mega.qwen3_5.sm103.draft import moe

    scratch = moe.draft_moe_fused_scratch(E, K, I, torch.device("cuda"))
    with pytest.raises(ValueError):
        _run(weights, _x(0).float(), scratch, _outputs())
    with pytest.raises(ValueError):
        _run(weights, torch.cat([_x(0), _x(1)]), scratch, _outputs())
    with pytest.raises(ValueError):
        moe.draft_moe_fused(
            _x(0),
            weights["router_w"],
            weights["w13"],
            weights["w2"],
            weights["w13_shared"],
            weights["w2_shared"],
            weights["gate_w"],
            scratch,
            *_outputs(),
            hop_site=torch.zeros(7, dtype=torch.int64),
        )

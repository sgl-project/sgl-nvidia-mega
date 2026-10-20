# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DeepSeek-V4.1 hop B, the MoE finalize into the Lamport all-reduce + mHC, on 4
SM100-family GPUs (GB300) with NVLink multicast, at 1-8 rows.

- ``AllReduceMHC.finalize`` with MXFP8 output and residual-only against SGLang's
  one-shot push kernels (streams, normalized row, MXFP8 values and scales), and its
  MXFP8 output against FlashInfer's ``mxfp8_quantize`` of its own normalized row;
- the early PDL triggers (after the poll; hop B's at the start) change no output bit;
- hop A and both finalize variants interleaved on one instance, with late ranks and
  from a replayed CUDA graph, stay bitwise stable and match torch;
- a PDL dependent that spins before its wait, or does not spin, reads the final
  outputs (a dependent that reads before its wait is the control).

Run with ``python test/kernels/test_deepseek_v41_hop_b.py --num-gpu 4 -s``.
"""

import os
from types import SimpleNamespace

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import pytest
import torch
import torch.distributed as dist
from cutlass import Int32, Int64
from sglang.srt.distributed import parallel_state as ps
from sglang.test.kernels.utils import multigpu_pytest_main
from sglang.test.test_utils import publish_build_topology

from sgl_nvidia_mega.common.sync import (
    current_cu_stream,
    global_timer_ns,
    make_fake_dynamic_compact_tensor,
    to_cute_dynamic,
)
from sgl_nvidia_mega.deepseek_v41.spec import NUMERICS

HIDDEN, HC, TOP_K, EPS = 5120, 4, 6, 1e-6
ROWS = tuple(range(1, 9))
SPIN_NS = 30_000


def _gpu4() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 10
        and torch.cuda.device_count() >= 4
    )


pytestmark = pytest.mark.skipif(not _gpu4(), reason="needs 4 SM100-family GPUs")


@pytest.fixture(scope="module")
def env():
    from sglang.kernels.ops.communication import all_reduce_fusion
    from sglang.srt.distributed.comm_cleanup import register_comm_cleanup
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )

    from sgl_nvidia_mega.deepseek_v41.sm103.all_reduce_mhc import AllReduceMHC

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
    tp = ps.get_tp_group()
    device = torch.device(f"cuda:{local_rank}")
    # SGLang's one-shot kernels stage through this communicator's push plane.
    comm = CustomAllReduceV2(
        tp.cpu_group, device, max_pull_size=1 << 20, max_push_size=2 << 20
    )
    assert not comm.disabled and comm.has_multicast
    all_reduce_fusion.register_comm(comm.obj)
    register_comm_cleanup(comm)
    with torch.inference_mode():
        hops = {
            early: AllReduceMHC(
                hidden_size=HIDDEN,
                hc_mult=HC,
                max_num_tokens=16,
                top_k=TOP_K,
                device=device,
                group=tp.device_group,
                quantize_layer_input=True,
                residual_only=True,
                early_trigger=early == "poll",
                hop_b_start_trigger=early == "start",
            )
            for early in ("end", "poll", "start")
        }
    yield SimpleNamespace(
        rank=rank,
        tp=world_size,
        device=device,
        group=tp.device_group,
        cpu_group=tp.cpu_group,
        hops=hops,
    )
    ps.destroy_model_parallel()
    ps.destroy_distributed_environment()


def _inputs(rows, seed, rank, device, shared_zero=False):
    """One hop's inputs: routing, mixes and norm weight equal on every rank (as in
    TP), GEMM2 rows, shared expert and hop A's x per rank; ~15% of a rank's routes
    point at an expert it does not hold."""
    common = torch.Generator().manual_seed(seed)
    local = torch.Generator().manual_seed(seed * 31 + rank + 1)
    padded = rows * TOP_K + 8
    perm = torch.randperm(padded, generator=common)[: rows * TOP_K].view(rows, TOP_K)
    held = torch.rand(rows, TOP_K, generator=local) >= 0.15
    shared = torch.randn(rows, HIDDEN, generator=local) * 0.5
    tensors = dict(
        idx=torch.where(held, perm, -1).int(),
        weights=torch.rand(rows, TOP_K, generator=common),
        gemm2=torch.randn(padded, HIDDEN, generator=local).bfloat16(),
        shared=(torch.zeros_like(shared) if shared_zero else shared).bfloat16(),
        x=torch.randn(rows, HIDDEN, generator=local).bfloat16(),
        residual=torch.randn(rows, HC, HIDDEN, generator=common).bfloat16(),
        post=torch.rand(rows, HC, generator=common),
        comb=torch.randn(rows, HC, HC, generator=common) * 0.1,
        pre=torch.rand(rows, HC, generator=common),
        norm_weight=torch.randn(HIDDEN, generator=common).bfloat16(),
    )
    return SimpleNamespace(rows=rows, **{k: v.to(device) for k, v in tensors.items()})


def _mxfp8_buffers(rows, device):
    from sgl_nvidia_mega.deepseek_v41.sm103.all_reduce_mhc import mxfp8_scale_bytes

    return (
        torch.empty(rows, HIDDEN, dtype=torch.float8_e4m3fn, device=device),
        torch.empty(mxfp8_scale_bytes(rows, HIDDEN), dtype=torch.uint8, device=device),
    )


def _run(hop, kind, inp):
    """One hop of ``kind``: its outputs, MXFP8 buffers last."""
    mix = (inp.residual, inp.post, inp.comb)
    if kind == "residual":
        return hop.finalize(
            inp.gemm2, inp.weights, inp.idx, inp.shared, *mix, None, None, EPS
        )[:1]
    mx = _mxfp8_buffers(inp.rows, inp.x.device) if kind.endswith("quant") else None
    norm = (inp.pre, inp.norm_weight, EPS)
    if kind.startswith("a"):
        out = hop(inp.x, *mix, *norm, mxfp8_out=mx)
    else:
        out = hop.finalize(
            inp.gemm2, inp.weights, inp.idx, inp.shared, *mix, *norm, mxfp8_out=mx
        )
    return tuple(out) + (mx or ())


def _sglang(kind, inp, tp):
    from sglang.kernels.ops.communication import all_reduce_mhc as sgl

    routing = (inp.gemm2, inp.idx.flatten(), inp.weights, TOP_K, inp.shared)
    mix = (inp.residual, inp.post, inp.comb)
    if kind == "residual":
        return sgl.moe_finalize_all_reduce_mhc_post(*routing, *mix, world_size=tp)[1:]
    return sgl.moe_finalize_all_reduce_mhc_post_combine_norm_quant(
        *routing, *mix, inp.pre, inp.norm_weight, EPS, world_size=tp
    )[1:]


def _local_finalize(inp):
    rows = inp.gemm2.float()[inp.idx.clamp(min=0).long()]
    weights = torch.where(inp.idx >= 0, inp.weights, 0.0)
    routed = (rows * weights.unsqueeze(-1)).sum(1).bfloat16()
    return (routed.float() + inp.shared.float()).bfloat16()


def _torch_boundary(x, inp, group):
    y = x.float()
    dist.all_reduce(y, group=group)
    y = y.bfloat16().float()
    streams = inp.post.unsqueeze(-1) * y.unsqueeze(1) + (
        inp.comb.unsqueeze(-1) * inp.residual.float().unsqueeze(2)
    ).sum(dim=1)
    streams = streams.bfloat16()
    collapsed = (inp.pre.unsqueeze(-1) * streams.float()).sum(dim=1)
    normed = collapsed * torch.rsqrt(collapsed.square().mean(-1, keepdim=True) + EPS)
    return streams, (normed * inp.norm_weight.float()).bfloat16()


def _check_torch(kind, inp, out, group):
    x = inp.x if kind.startswith("a") else _local_finalize(inp)
    streams, normed = _torch_boundary(x, inp, group)
    torch.testing.assert_close(out[0].float(), streams.float(), atol=3e-2, rtol=2e-2)
    if kind != "residual":
        torch.testing.assert_close(out[1].float(), normed.float(), atol=6e-2, rtol=3e-2)


def _same(a, b):
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def _ordered(t):
    """Integers whose distance is the ULP distance (sign-magnitude floats)."""
    if t.dtype == torch.bfloat16:
        bits = t.view(torch.int16).int()
        return torch.where(bits < 0, -(bits & 0x7FFF), bits)
    bits = t.view(torch.uint8).int()
    return torch.where(bits >= 0x80, -(bits & 0x7F), bits)


def _diff(a, b, scales=False):
    """(fraction of equal elements, max ULPs where |b| >= 1e-3 or max scale
    exponent step, max abs)."""
    if scales:
        delta = (a.int() - b.int()).abs()
        return (delta == 0).float().mean().item(), delta.max().item(), 0.0
    ulps = (_ordered(a) - _ordered(b)).abs()
    abs_err = (a.float() - b.float()).abs().max().item()
    ulps_big = torch.where(b.float().abs() >= 1e-3, ulps, 0).max().item()
    return (ulps == 0).float().mean().item(), ulps_big, abs_err


@torch.inference_mode()
@pytest.mark.parametrize("shared_zero", [False, True])
def test_finalize_matches_sglang_one_shot(env, shared_zero):
    from flashinfer import mxfp8_quantize

    hop = env.hops["end"]
    worst = {}
    for rows in ROWS:
        inp = _inputs(rows, 100 + rows, env.rank, env.device, shared_zero)
        streams, normed, values, scales = _run(hop, "quant", inp)
        # The MXFP8 output is FlashInfer's quantize of the BF16 row, bit for bit.
        fi_values, fi_scales = mxfp8_quantize(normed, is_sf_swizzled_layout=True)
        assert _same(values, fi_values), f"MXFP8 values differ: rows={rows}"
        assert _same(scales, fi_scales.reshape(-1)), f"scales differ: rows={rows}"
        # The quantized, unquantized and residual-only variants agree bitwise.
        plain = _run(hop, "norm", inp)
        residual_only = _run(hop, "residual", inp)
        assert _same(plain[0], streams) and _same(plain[1], normed)
        assert _same(residual_only[0], streams)
        _check_torch("quant", inp, (streams, normed), env.group)

        sgl_quant = _sglang("quant", inp, env.tp)
        sgl_streams = _sglang("residual", inp, env.tp)[0]
        torch.cuda.synchronize()
        assert _same(sgl_streams, sgl_quant[0])
        stats = {
            "streams": _diff(streams, sgl_quant[0]),
            "normed": _diff(normed, sgl_quant[1]),
            "values": _diff(values, sgl_quant[2]),
            "scales": _diff(scales, sgl_quant[3], scales=True),
        }
        for name, (equal, ulps, abs_err) in stats.items():
            print(
                f"HOPB r{env.rank} sglang shared_zero={shared_zero} rows={rows} "
                f"{name}: equal={equal:.6f} max_ulp={ulps} max_abs={abs_err:.4g}"
            )
            prev = worst.get(name, (1.0, 0, 0.0))
            worst[name] = (
                min(prev[0], equal),
                max(prev[1], ulps),
                max(prev[2], abs_err),
            )
    print(f"HOPB r{env.rank} sglang shared_zero={shared_zero} worst={worst}")
    # The mHC post rounds one product in another order than SGLang's: a few
    # stream values per 10^5 differ by an ULP, which moves their row's RMS.
    assert worst["streams"][0] >= 0.9999, worst
    assert worst["streams"][1] <= NUMERICS.tolerances["hop_b.residual.ulp"], worst
    assert worst["normed"][0] >= 0.999, worst
    assert worst["normed"][1] <= NUMERICS.tolerances["hop_b.normed.ulp"], worst
    assert worst["values"][1] <= NUMERICS.tolerances["hop_b.mxfp8.ulp"], worst
    assert worst["scales"][2] <= NUMERICS.tolerances["hop_b.mxfp8_scales.max_abs"], (
        worst
    )


@torch.inference_mode()
def test_early_trigger_changes_no_bit(env):
    for rows in ROWS:
        inp = _inputs(rows, 200 + rows, env.rank, env.device)
        for kind in ("a", "a_quant", "norm", "quant", "residual"):
            late, *early = (_run(env.hops[e], kind, inp) for e in env.hops)
            for trigger in early:
                for a, b in zip(late, trigger):
                    assert _same(a, b), f"early trigger differs: rows={rows} {kind}"
    torch.cuda.synchronize()


def _interleaved_plan(steps):
    plan = []
    for step in range(steps):
        rows = ROWS[step % len(ROWS)]
        kinds = ("a", "quant", "residual", "a_quant", "norm")
        plan += [(kind, rows, step % 2) for kind in kinds[: 3 + step % 3]]
    return plan


@torch.inference_mode()
@pytest.mark.parametrize("early", ["end", "poll", "start"])
def test_hop_a_and_finalize_interleave_on_one_instance(env, early):
    hop = env.hops[early]
    inputs = {
        (rows, s): _inputs(rows, 300 + 10 * rows + s, env.rank, env.device)
        for rows in ROWS
        for s in (0, 1)
    }
    plan = _interleaved_plan(20)
    assert len(plan) >= 50 and len(plan) % 3

    def run(plan, late=None):
        outs = []
        for kind, rows, s in plan:
            if late == env.rank:
                torch.cuda._sleep(20_000)
            outs.append(_run(hop, kind, inputs[rows, s]))
        return outs

    # Reference: one hop at a time, every rank in step.
    reference = []
    for item in plan:
        reference += run([item])
        torch.cuda.synchronize()
        dist.barrier(group=env.cpu_group)
    for (kind, rows, s), out in zip(plan, reference):
        _check_torch(kind, inputs[rows, s], out, env.group)

    def compare(outs, label):
        for index, (got, ref) in enumerate(zip(outs, reference[: len(outs)])):
            for a, b in zip(got, ref):
                assert _same(a, b), f"{label}: step {index} {plan[index]}"

    for late in (None, *range(env.tp)):
        outs = run(plan, late)
        torch.cuda.synchronize()
        compare(outs, f"back-to-back late={late}")

    # A graph of 22 hops (not a multiple of the 3 generations), replayed.
    sub = plan[:22]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        captured = run(sub)
    torch.cuda.current_stream().wait_stream(stream)
    for replay in range(20):
        graph.replay()
        torch.cuda.synchronize()
        compare(captured, f"graph replay {replay}")
        # Eager hops between replays move the generation the graph starts from.
        if replay % 4 == 3:
            compare(run(plan[:1]), "eager between replays")
    torch.cuda.synchronize()


@cute.jit
def _copy_words(
    src: cute.Tensor, dst: cute.Tensor, count: Int32, stride: cutlass.Constexpr[int]
):
    block, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    index = block * _PdlSnapshot.THREADS + tidx
    while index < count:
        dst[index] = src[index]
        index = index + stride


class _PdlSnapshot:
    """A PDL dependent that copies four buffers after a ``spin_ns`` prologue and
    its griddepcontrol.wait; ``read_first`` copies before the wait (the control)."""

    THREADS = 256
    BLOCKS = 16

    def __init__(self, spin_ns: int, read_first: bool = False) -> None:
        self.spin_ns = spin_ns
        self.read_first = read_first

    @cute.jit
    def __call__(
        self, s0, d0, s1, d1, s2, d2, s3, d3, n0, n1, n2, n3, stream: cuda.CUstream
    ):
        self.kernel(s0, d0, s1, d1, s2, d2, s3, d3, n0, n1, n2, n3).launch(
            grid=(self.BLOCKS, 1, 1),
            block=(self.THREADS, 1, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(self, s0, d0, s1, d1, s2, d2, s3, d3, n0, n1, n2, n3):
        if cutlass.const_expr(self.spin_ns > 0):
            start = global_timer_ns()
            now = start
            while now - start < Int64(self.spin_ns):
                now = global_timer_ns()
        if cutlass.const_expr(not self.read_first):
            cute.arch.griddepcontrol_wait()
        stride = self.BLOCKS * self.THREADS
        _copy_words(s0, d0, n0, stride)
        _copy_words(s1, d1, n1, stride)
        _copy_words(s2, d2, n2, stride)
        _copy_words(s3, d3, n3, stride)
        if cutlass.const_expr(self.read_first):
            # Keeps the stream's PDL chain: later kernels still follow the hop.
            cute.arch.griddepcontrol_wait()


_SNAPSHOTS = {}


def _snapshot(sources, spin_ns, read_first=False):
    """Launch the dependent on the current stream; returns the copies."""
    key = (spin_ns, read_first)
    if key not in _SNAPSHOTS:
        fakes = [
            make_fake_dynamic_compact_tensor(Int32, alignment=16, divisibility=4)
            for _ in range(8)
        ]
        _SNAPSHOTS[key] = cute.compile(
            _PdlSnapshot(spin_ns, read_first),
            *fakes,
            *[Int32(4)] * 4,
            current_cu_stream(),
        )
    copies = [torch.empty_like(s) for s in sources]
    args = []
    for s, d in zip(sources, copies):
        for t in (s, d):
            words = t.view(torch.uint8).flatten().view(torch.int32)
            args.append(to_cute_dynamic(words, 16, divisibility=4))
    counts = [Int32(s.numel() * s.element_size() // 4) for s in sources]
    _SNAPSHOTS[key](*args, *counts, current_cu_stream())
    return copies


@torch.inference_mode()
def test_pdl_dependent_reads_final_outputs(env):
    controls = {}
    for early in env.hops:
        hop = env.hops[early]
        for spin_ns, read_first in ((SPIN_NS, False), (0, False), (0, True)):
            for rows in ROWS:
                inputs = [
                    _inputs(rows, 400 + 10 * rows + s, env.rank, env.device)
                    for s in range(3)
                ]
                stale = 0
                for it in range(12):
                    out = _run(hop, "quant", inputs[it % 3])
                    copies = _snapshot(out, spin_ns, read_first)
                    torch.cuda.synchronize()
                    stale += sum(not _same(a, b) for a, b in zip(out, copies))
                    del out, copies
                if read_first:
                    controls[early, rows] = stale
                else:
                    assert stale == 0, f"early={early} spin={spin_ns} rows={rows}"
                dist.barrier(group=env.cpu_group)
    print(f"HOPB r{env.rank} read-before-wait control, stale buffers of 48: {controls}")


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4,), timeout=2400)

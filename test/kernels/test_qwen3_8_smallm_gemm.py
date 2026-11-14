# SPDX-License-Identifier: Apache-2.0
"""The small-M GEMM routes (runtime/smallm_gemm.py) against SGLang main's stock BF16
dispatch (its cutedsl backend) on random weights at Qwen3.8's TP1 shapes, on an SM107
GPU.

A tolerance where the source claims none: out_proj split-K against main's TGV (relative
L2 <= 1e-3 at every routed row count, the source's load bound; it measured 1.2-1.7e-4)
and the TGV down_proj against main's cuBLAS (declared, not measured). Bitwise for the
fused act, as the source claims: against main's gate_up dispatch followed by
act_and_mul as built before SGLang #48942 (compiled here from that source) on seeded
rows and on every finite (gate, up) BF16 pair, with a div.full copy as the control that
must differ; and against main's own Qwen2MoeMLP (gate_up + the current SiluAndMul) on
random operands, the source's dispatch test. Every route also replays bitwise under a
CUDA graph. The bitwise routes (runtime/sm107_bf16_smallm.py) pass the source's load
check against main's dispatch at all 87 keys, and a key of each kind matches it bit for
bit eagerly and under graph replay, as one kernel where main runs cuBLAS split-K + reduce.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sgl_nvidia_mega.qwen3_8.kernels.ops.gemm import splitk_act_mul as act
from sgl_nvidia_mega.qwen3_8.runtime import smallm_gemm

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the small-M routes are SM107 only",
)

BF16 = torch.bfloat16
N, K = smallm_gemm.OUT_PROJ
INTER, HIDDEN = smallm_gemm.INTER, smallm_gemm.HIDDEN
ROWS = 16
LAUNCHES = HIDDEN // ROWS  # 160 one-hot launches cover every K column
PAIRS = HIDDEN * INTER

_FAST_MATH_ACT_CUDA = r"""
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

__global__ void silu_and_mul_kernel(const __nv_bfloat16* gate_up, __nv_bfloat16* out, int64_t rows, int64_t inter) {
  const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= rows * inter) return;
  const int64_t r = i / inter, c = i - r * inter;
  const float gate = __bfloat162float(gate_up[r * 2 * inter + c]);
  const float up = __bfloat162float(gate_up[r * 2 * inter + inter + c]);
  out[i] = __float2bfloat16(gate / (1.0f + expf(-gate)) * up);
}

void silu_and_mul(torch::Tensor gate_up, torch::Tensor out) {
  const int64_t rows = gate_up.size(0), inter = gate_up.size(1) / 2;
  const int64_t threads = 256, blocks = (rows * inter + threads - 1) / threads;
  silu_and_mul_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(gate_up.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), rows, inter);
}
"""


def _fast_math_silu_and_mul():
    """SGLang's act_and_mul as built before #48942: gate / (1 + expf(-gate)) * up."""
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        name="qwen3_8_fused_act_reference_silu_and_mul",
        cpp_sources="void silu_and_mul(torch::Tensor gate_up, torch::Tensor out);",
        cuda_sources=_FAST_MATH_ACT_CUDA,
        functions=["silu_and_mul"],
        extra_cuda_cflags=["--use_fast_math"],
    ).silu_and_mul


FULL_DIVIDE_PTX = act.SILU_MUL_PTX.replace("div.approx.ftz.f32", "div.full.ftz.f32")


def _bits(t):
    return t.contiguous().view(torch.int16)


def _rel_l2(got, want):
    return float((got.float() - want.float()).norm() / want.float().norm())


def _capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


class _St:
    def __init__(self):
        self.ran_ops = []

    def ran(self, op, impl):
        self.ran_ops.append((op, impl))


@pytest.fixture(scope="module")
def env():
    from sglang.srt.layers.linear import RowParallelLinear
    from sglang.srt.layers.quantization import unquant
    from sglang.srt.model_loader.utils import set_default_torch_dtype
    from sglang.srt.models.qwen2_moe import Qwen2MoeMLP
    from sglang.test.layer_ut_utils import init_single_process_dist

    exec_ctx = SimpleNamespace(
        kernel=SimpleNamespace(bf16_gemm_backend="cutedsl"),
        deterministic=SimpleNamespace(enable_deterministic_inference=False),
    )
    platform = SimpleNamespace(is_sm100=True, device_sm=107)
    with (
        mock.patch.object(unquant, "get_exec", lambda: exec_ctx),
        mock.patch.object(unquant, "get_platform", lambda: platform),
    ):
        unquant.initialize_bf16_gemm_config()
    init_single_process_dist()
    g = torch.Generator(device="cuda").manual_seed(7)
    with set_default_torch_dtype(BF16):
        mlp = Qwen2MoeMLP(
            HIDDEN, INTER, "silu", quant_config=None, reduce_results=False
        )
        proj = RowParallelLinear(
            K, N, bias=False, quant_config=None, reduce_results=False
        )
    mlp, proj = mlp.cuda(), proj.cuda()
    with torch.no_grad():
        for layer in (mlp.gate_up_proj, mlp.down_proj, proj):
            layer.weight.copy_(
                torch.randn(layer.weight.shape, generator=g, device="cuda") * 0.02
            )

    def stock_gemm(x, w):
        return unquant._bf16_gemm_dispatch_impl(x, w, None)

    device = torch.device("cuda", torch.cuda.current_device())
    reason = smallm_gemm.install(
        [proj], [mlp], smallm_gemm.DEFAULT, device=device, stock_gemm=stock_gemm
    )
    assert reason == ""
    assert smallm_gemm._act_passed
    yield SimpleNamespace(
        mlp=mlp, proj=proj, stock_gemm=stock_gemm, act=_fast_math_silu_and_mul(), g=g
    )


def _executor(draft=False):
    return SimpleNamespace(smallm_gemm=smallm_gemm.DEFAULT, draft=draft)


def _stock_pair(env, x, w, out):
    """main's gate_up dispatch, then act_and_mul as built before #48942."""
    env.act(env.stock_gemm(x, w), out)
    return out


def test_out_proj_split_k_within_the_bound(env, record_property):
    routed = smallm_gemm._out_proj_tactics(smallm_gemm.DEFAULT)
    worst = 0.0
    with torch.inference_mode():
        for rows in range(1, 34):
            x = torch.randn(rows, K, generator=env.g, device="cuda").to(BF16)
            st = _St()
            got = smallm_gemm.out_proj(_executor(), env.proj, x, st, "gdn.out_proj")
            if rows not in routed:
                assert got is None and st.ran_ops == [], rows
                continue
            assert st.ran_ops == [("gdn.out_proj", "splitk")], rows
            # RowParallelLinear.forward at TP1 is its quant method's apply
            err = _rel_l2(got, env.proj.quant_method.apply(env.proj, x))
            worst = max(worst, err)
            assert err <= 1e-3, (rows, err)
            state = {}
            graph = _capture(
                lambda: state.update(
                    out=smallm_gemm.out_proj(_executor(), env.proj, x, _St(), "o")
                )
            )
            state["out"].fill_(float("nan"))
            graph.replay()
            assert torch.equal(_bits(state["out"]), _bits(got)), (rows, "graph")
            del graph
    record_property("out_proj_max_rel_l2", worst)


def test_fused_act_bitwise_to_the_pre_48942_pair(env):
    g = torch.Generator(device="cuda").manual_seed(1234)
    w = (torch.randn(2 * INTER, HIDDEN, generator=g, device="cuda") * 0.02).to(BF16)
    w[600, 5], w[INTER + 600, 5] = (
        -87.5,
        -80.0,
    )  # stock gives +0 there; div.full does not
    wp = act.permute_gate_up(w)
    anchor = torch.empty(ROWS, 2 * INTER, dtype=BF16, device="cuda")
    x = torch.zeros(ROWS, HIDDEN, dtype=BF16, device="cuda")
    x[torch.arange(ROWS), torch.arange(ROWS)] = 1.0
    ref = _stock_pair(env, x, w, torch.empty(ROWS, INTER, dtype=BF16, device="cuda"))
    assert int(_bits(ref)[5, 600]) == 0
    for ptx, same in ((act.SILU_MUL_PTX, True), (FULL_DIVIDE_PTX, False)):
        got = act.run(
            x, wp, torch.full_like(ref, float("nan")), anchor, silu_mul_ptx=ptx
        )
        differing = int((_bits(ref) != _bits(got)).sum())
        assert (differing == 0) if same else (differing >= 1), (ptx, differing)

    for kind in ("randn", "outliers", "hot"):
        for rows in range(1, ROWS + 1):
            x = torch.randn(rows, HIDDEN, generator=g, device="cuda")
            if kind == "outliers":
                x[:, torch.randint(0, HIDDEN, (8,), generator=g, device="cuda")] *= 30.0
            elif kind == "hot":
                x = x * 8.0
            x = x.to(BF16)
            ref = torch.empty(rows, INTER, dtype=BF16, device="cuda")
            ref = _stock_pair(env, x, w, ref)
            got = torch.full_like(ref, float("nan"))
            act.run(x, wp, got, anchor[:rows])
            assert torch.equal(_bits(ref), _bits(got)), (kind, rows, "eager")
            graph = _capture(lambda: act.run(x, wp, got, anchor[:rows]))
            got.fill_(float("nan"))
            graph.replay()
            assert torch.equal(_bits(ref), _bits(got)), (kind, rows, "graph")
            del graph


class _OneHot:
    """One weight build = 2,560 x 640 (gate, up) pairs; pair q lands at flat output
    index q: launch c's row m is one-hot at K column 16c + m, so gate_up[m, r] =
    W[r, 16c + m]."""

    def __init__(self, env):
        self.W = torch.zeros(2 * INTER, HIDDEN, dtype=BF16, device="cuda")
        self.Wp = torch.zeros_like(self.W)
        rows = torch.arange(ROWS, device="cuda")
        self.x = []
        for c in range(LAUNCHES):
            x = torch.zeros(ROWS, HIDDEN, dtype=BF16, device="cuda")
            x[rows, ROWS * c + rows] = 1.0
            self.x.append(x)
        anchor = torch.empty(ROWS, 2 * INTER, dtype=BF16, device="cuda")
        self.out = {
            name: torch.empty(LAUNCHES, ROWS, INTER, dtype=BF16, device="cuda")
            for name in ("ref", "approx", "full")
        }
        ptx = {"approx": act.SILU_MUL_PTX, "full": FULL_DIVIDE_PTX}
        self.graphs = [
            _capture(
                lambda: [
                    _stock_pair(env, self.x[c], self.W, self.out["ref"][c])
                    for c in range(LAUNCHES)
                ]
            )
        ] + [
            _capture(
                lambda name=name: [
                    act.run(
                        self.x[c],
                        self.Wp,
                        self.out[name][c],
                        anchor,
                        silu_mul_ptx=ptx[name],
                    )
                    for c in range(LAUNCHES)
                ]
            )
            for name in ("approx", "full")
        ]

    def load(self, gate, up):
        self.W[:INTER].copy_(gate.view(HIDDEN, INTER).t())
        self.W[INTER:].copy_(up.view(HIDDEN, INTER).t())
        # = permute_gate_up
        self.Wp.view(20, 2, 32, HIDDEN).copy_(
            self.W.view(2, 20, 32, HIDDEN).transpose(0, 1)
        )

    def run(self):
        for out in self.out.values():
            out.fill_(float("nan"))
        for graph in self.graphs:
            graph.replay()

    def differ(self, name):
        return _bits(self.out["ref"]).view(-1) != _bits(self.out[name]).view(-1)


def test_every_finite_pair_bitwise(env, record_property):
    h = _OneHot(env)
    assert torch.equal(h.Wp, act.permute_gate_up(h.W))
    v = torch.arange(-32768, 32768, dtype=torch.int32, device="cuda")
    v = v.to(torch.int16).view(BF16)
    finite = v[torch.isfinite(v.float())]
    n = finite.numel()
    total = n * n
    q0 = torch.arange(PAIRS, dtype=torch.int64, device="cuda")
    counts = torch.zeros(2, dtype=torch.int64, device="cuda")
    for b in range(-(-total // PAIRS)):
        q = q0 + b * PAIRS
        valid = q < total
        q = torch.where(valid, q, torch.zeros_like(q))
        h.load(finite[q // n], finite[q % n])
        h.run()
        counts[0] += (h.differ("approx") & valid).sum()
        counts[1] += (h.differ("full") & valid).sum()
    approx, full = counts.tolist()
    record_property("pairs", total)
    record_property("full_divide_differ", full)
    assert approx == 0
    assert (
        full > 0
    )  # the control fires: the comparison sees a divide that is not the stock one


def test_shared_expert_matches_main_mlp(env):
    """``shared_expert`` on the draft (no TGV) equals main's Qwen2MoeMLP.forward,
    eagerly and under graph replay; the fused kernel runs at 1-16 rows and not at 17."""
    mlp, g = env.mlp, env.g
    with torch.inference_mode():
        for rows in range(1, ROWS + 2):
            x = torch.randn(rows, HIDDEN, generator=g, device="cuda").to(BF16)
            st = _St()
            got = smallm_gemm.shared_expert(_executor(draft=True), mlp, x, st)
            if rows > ROWS:
                assert got is None and st.ran_ops == []
                continue
            assert st.ran_ops == [("moe.shared_expert", "act_splitk")], rows
            assert torch.equal(_bits(mlp(x)), _bits(got)), (rows, "eager")
            state = {}
            graph = _capture(
                lambda: state.update(
                    out=smallm_gemm.shared_expert(_executor(draft=True), mlp, x, _St())
                )
            )
            x.copy_(torch.randn(rows, HIDDEN, generator=g, device="cuda").to(BF16))
            state["out"].fill_(float("nan"))
            graph.replay()
            assert torch.equal(_bits(mlp(x)), _bits(state["out"])), (rows, "graph")
            del graph


def test_tgv_down_on_the_target_at_4_8_16(env, record_property):
    mlp, g = env.mlp, env.g
    worst = 0.0
    with torch.inference_mode():
        for rows in range(1, ROWS + 2):
            x = torch.randn(rows, HIDDEN, generator=g, device="cuda").to(BF16)
            for draft in (False, True):
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CUDA]
                ) as prof:
                    got = smallm_gemm.shared_expert(_executor(draft), mlp, x, _St())
                    torch.cuda.synchronize()
                tgv = any("Tgv" in e.name for e in prof.events())
                assert tgv == (not draft and rows in (4, 8, 16)), (rows, draft)
                if tgv:
                    err = _rel_l2(got, mlp(x))
                    worst = max(worst, err)
                    assert err <= 1e-3, (rows, err)
    record_property("down_tgv_max_rel_l2", worst)


def test_the_build_check_catches_a_non_stock_divide(env):
    """The build's act check passes on the shipped epilogue and fails on a div.full
    copy (only the edge pair shows the difference)."""
    real_run = smallm_gemm._k.act.run

    def full_divide_run(*args, **kwargs):
        return real_run(*args, **{**kwargs, "silu_mul_ptx": FULL_DIVIDE_PTX})

    device = torch.device("cuda", torch.cuda.current_device())
    assert smallm_gemm._check_act(env.stock_gemm, device)
    with mock.patch.object(smallm_gemm._k.act, "run", full_divide_run):
        assert not smallm_gemm._check_act(env.stock_gemm, device)


# (M, N, K) of each route kind and shape: router and index_qk on both cluster tactics,
# the shared gate_up, TGV at its own pick and at a fixed tactic, the 2-CTA TGV tile
BITWISE_KEYS = (
    (12, 512, 2560),
    (24, 512, 2560),
    (12, 640, 2560),
    (32, 640, 2560),
    (8, 1280, 2560),
    (20, 16480, 2560),
    (28, 16480, 2560),
    (68, 13312, 2560),
    (64, 2560, 6144),
)


def test_bitwise_routes_match_main_dispatch(env, record_property):
    from sgl_nvidia_mega.qwen3_8.runtime import sm107_bf16_smallm as sr

    assert sr.install(device_sm=107, splitk_enabled=True, cutedsl_available=True)
    # every key, bit for bit: ROUTES against main's dispatch, SOURCE_SPLITK_ROUTES against
    # main's split-K kernel at the source's tactic
    assert sr.precompile(stock_gemm=env.stock_gemm, source_gemm=sr._source_splitk)
    g = torch.Generator(device="cuda").manual_seed(11)
    names = {}
    with torch.inference_mode():
        for m, n, k in BITWISE_KEYS:
            w = (torch.randn(n, k, generator=g, device="cuda") * 0.02).to(BF16)
            x = torch.randn(m, k, generator=g, device="cuda").to(BF16)
            want = env.stock_gemm(x, w)
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA]
            ) as prof:
                got = sr.gemm(x, w)
                torch.cuda.synchronize()
            kernels = [
                e.name
                for e in prof.events()
                if e.device_type == torch.autograd.DeviceType.CUDA
            ]
            assert len(kernels) == 1, ((m, n, k), kernels)
            names[f"{m}x{n}x{k}"] = kernels[0]
            assert torch.equal(_bits(got), _bits(want)), ((m, n, k), "eager")
            state = {}
            graph = _capture(lambda: state.update(out=sr.gemm(x, w)))
            state["out"].fill_(float("nan"))
            graph.replay()
            assert torch.equal(_bits(state["out"]), _bits(want)), ((m, n, k), "graph")
            del graph
    assert all(
        "dense_bf16_gemm_sm100_splitk" in name
        for key, name in names.items()
        if key.split("x")[1] in ("512", "640", "1280")
    ), names
    record_property("route_kernels", names)

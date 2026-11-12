# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slot gemm2_pipe (and its register-pool build): the routed GEMM2 on muLLM's reference-pipeline +
prelude build (gemm2_pipe.cu, gemm2_pipe_strided.cu) against the default GEMM2
(gemm2.cu / gemm2_strided.cu), bit for bit, on one SM100-family GPU (GB300).

Both pairs run the same route kernel and GEMM1 on the same weights and rows, so the
intermediate is the same; every routed slot's GEMM2 row must be equal bit for bit:
- routings of 6 rows over D = 8..36 distinct experts, one expert chosen by every row,
  all-distinct experts, and experts over the 8-row tile (several tiles each);
- 1-48 rows; the target layer at TP4 (384 local experts) and at EP2 (192 of 384, either
  window), the DSpark draft MoE (128, and 64 of 128);
- eager and replayed from a CUDA graph with a shared-expert-sized chain on a second
  stream (forked before the route kernel, joined after GEMM2);
- a late route record: its producer releases PDL dependents at entry and writes the
  record only after a delay, on the same stream and, under capture, on a second stream
  joined before GEMM1 (the programmatic edge a join leaves a PDL launch). GEMM2 reads the
  record before its grid-dependency wait (its weight loader, and the prelude at entry);
  a GEMM2 that read it early would see the poisoned (empty) record and leave its rows NaN.
Run once per grid: SGL_NVIDIA_MEGA_MOE_STRIDED_GEMM=0 (cluster launch control) and =1.
The register-pool cases also check the built kernel's register count.
"""

import ctypes
import os
import sys
import zlib

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_deepseek_v41_moe as single  # noqa: E402

from sgl_nvidia_mega.deepseek_v41.sm103.moe import (  # noqa: E402
    routed_expert_gemm as reg,
)

needs_moe = single.needs_moe
BUILDS = {
    "pipe": {"gemm2_pipe": True},
    "pool": {"gemm2_pipe": True, "gemm2_registers": reg.GEMM2_POOL_REGISTERS},
}


@pytest.fixture(scope="module")
def target_layer():
    return single._shuffled_layer(torch.Generator().manual_seed(2), 384)


@pytest.fixture(scope="module")
def draft_layer():
    return single._shuffled_layer(torch.Generator().manual_seed(3), 128)


def _exact_routing(rows, top_k, distinct, experts, generator, every_row=None):
    """``rows`` x ``top_k`` ids over exactly ``distinct`` experts (each row distinct
    ids); ``every_row``: one expert that every row takes."""
    pool = torch.randperm(experts, generator=generator)[:distinct].tolist()
    if every_row is not None:
        pool = [every_row] + [e for e in pool if e != every_row][: distinct - 1]
    for _ in range(1000):
        ids = []
        for r in range(rows):
            first = [pool[0]] if every_row is not None else []
            rest = [e for e in pool if e not in first]
            pick = torch.randperm(len(rest), generator=generator)[: top_k - len(first)]
            ids.append(first + [rest[i] for i in pick.tolist()])
        # cover every pool expert: deal the missing ones into free slots
        flat = [e for row in ids for e in row]
        missing = [e for e in pool if e not in flat]
        for e in missing:
            for r in torch.randperm(rows, generator=generator).tolist():
                row = ids[r]
                for k in range(len(row) - 1, -1, -1):
                    if every_row is not None and row[k] == pool[0]:
                        continue
                    if flat.count(row[k]) > 1 and e not in row:
                        flat.remove(row[k])
                        row[k] = e
                        flat.append(e)
                        break
                else:
                    continue
                break
        t = torch.tensor(ids, dtype=torch.int32)
        if t.unique().numel() == distinct:
            return t
    raise RuntimeError(f"no routing of {rows}x{top_k} over {distinct} experts")


def _pairs(layer, top_k, build, *, ep_rank=None, ep=2):
    """(default pair, slot pair) on ``layer`` (or its EP window)."""
    router = layer[0].size(0)
    if ep_rank is None:
        weights, offset = layer, 0
    else:
        weights, offset = single._ep_half(layer, ep_rank, ep)
    common = dict(
        top_k=top_k,
        max_rows=reg.SERVED_MAX_ROWS,
        router_experts=router,
        expert_offset=offset,
    )
    return (
        reg.RoutedExpertGemm(*weights, **common),
        reg.RoutedExpertGemm(*weights, **common, **BUILDS[build]),
    )


def _run(gemm, x_q, x_s, ids):
    """Route + GEMM1 + GEMM2 into a NaN-filled output; the routed slots' rows."""
    rows, top_k = ids.shape
    ctas = reg.max_ctas(rows, top_k, gemm.experts)
    permuted, record = gemm._route(ids, rows, ctas)
    out = torch.full(
        (rows * top_k * reg.TILE, reg.HIDDEN), float("nan"), dtype=torch.bfloat16,
        device=ids.device,
    )  # fmt: skip
    gemm._gemms(x_q, x_s, record, ctas, gemm2_rows=out)
    return out, permuted


def _gathered(out, permuted):
    kept = permuted >= 0
    return out[permuted.clamp(min=0).long()][kept].view(torch.int16), kept


def _assert_bitwise(ref, got, where):
    (a, kept_a), (b, kept_b) = ref, got
    assert torch.equal(kept_a, kept_b), where
    assert not torch.isnan(a.view(torch.bfloat16)).any(), f"{where}: reference NaN"
    mismatched = (a != b).any(-1).sum().item()
    assert mismatched == 0, f"{where}: {mismatched} of {a.size(0)} rows differ"


def _check(pair, ids, generator, where):
    default, slot = pair
    x_q, x_s = single._quantized_rows(ids.size(0), generator)
    ids = ids.cuda()
    ref = _gathered(*_run(default, x_q, x_s, ids))
    got = _gathered(*_run(slot, x_q, x_s, ids))
    torch.cuda.synchronize()
    _assert_bitwise(ref, got, where)


@needs_moe
@pytest.mark.parametrize("build", list(BUILDS))
def test_register_count(target_layer, build):
    pair = _pairs(target_layer, 6, build)
    cu = reg._driver()
    (regs,) = reg._check(
        cu.cuFuncGetAttribute(
            cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS,
            pair[1].gemm2_pipe.function,
        ),
        "cuFuncGetAttribute",
    )
    want = reg.GEMM2_POOL_REGISTERS if build == "pool" else 128
    assert regs <= want, regs
    print(f"gemm2 {build}: {regs} registers per thread")


@needs_moe
@pytest.mark.parametrize("build", list(BUILDS))
@pytest.mark.parametrize("window", [None, 0, 1])  # TP4 (all 384) or the EP2 halves
@pytest.mark.parametrize(
    "case",
    [f"d{d}" for d in range(8, 37, 4)] + ["d6", "every_row", "all_distinct"],
)
def test_target_six_rows(target_layer, build, window, case):
    gen = torch.Generator().manual_seed(zlib.crc32(f"{case} {window}".encode()))
    if case == "every_row":
        every = int(torch.randint(384, (1,), generator=gen))
        ids = _exact_routing(6, 6, 24, 384, gen, every_row=every)
    elif case == "all_distinct":
        ids = _exact_routing(6, 6, 36, 384, gen)
    else:
        ids = _exact_routing(6, 6, int(case[1:]), 384, gen)
    _check(_pairs(target_layer, 6, build, ep_rank=window), ids, gen, f"{case} {window}")


@needs_moe
@pytest.mark.parametrize("build", list(BUILDS))
@pytest.mark.parametrize("window", [None, 1])
@pytest.mark.parametrize("rows", [1, 2, 3, 5, 7, 8, 9, 12, 16, 24, 32, 47, 48])
@pytest.mark.parametrize("routing", ["random", "wide", "all_distinct"])
def test_target_rows(target_layer, build, window, rows, routing):
    """wide: six experts take every row (rows > 8: several 8-row tiles each)."""
    gen = torch.Generator().manual_seed(rows * 7 + (window or 0) + len(routing))
    if routing == "wide":
        ids = single._routing(rows, 6, gen, 384, 6)
    elif routing == "all_distinct":
        ids = torch.randperm(384, generator=gen)[: rows * 6].view(rows, 6)
        ids = ids.to(torch.int32)
    else:
        ids = single._routing(rows, 384, gen, 384, 6)
    _check(
        _pairs(target_layer, 6, build, ep_rank=window), ids, gen, f"{rows} {routing}"
    )


@needs_moe
@pytest.mark.parametrize("build", list(BUILDS))
@pytest.mark.parametrize("window", [None, 0, 1])  # all 128, or 64 of 128
@pytest.mark.parametrize("rows", [1, 5, 10, 16, 48])
@pytest.mark.parametrize("distinct", [3, 12, 128])
def test_draft(draft_layer, build, window, rows, distinct):
    gen = torch.Generator().manual_seed(rows * 13 + distinct + (window or 0))
    ids = single._routing(rows, distinct, gen, 128, 3)
    _check(
        _pairs(draft_layer, 3, build, ep_rank=window), ids, gen, f"{rows} {distinct}"
    )


# ------------------------------------------------- graph replay beside a side chain


class SideChain:
    """A shared-expert-sized chain on its own weights (row-tile MXFP8 quant, 1152 x 5120
    gate_up on the low-latency GEMM, clamped SwiGLU, MXFP8 quant, 5120 x 640 down)."""

    def __init__(self, rows, generator):
        from sgl_nvidia_mega.deepseek_v41.sm103 import mxfp8_ll

        self.ll = mxfp8_ll

        def weight(n, k):
            w = (torch.randn(n, k, generator=generator) * 0.05).cuda()
            s = torch.randint(120, 128, (n * k // 32,), dtype=torch.uint8,
                              generator=generator).cuda()  # fmt: skip
            return w.to(torch.float8_e4m3fn), s

        self.gate_up = weight(2 * reg.SLICE, reg.HIDDEN)
        self.down = weight(reg.HIDDEN, reg.PADDED)
        self.x = torch.randn(rows, reg.HIDDEN, generator=generator).to(torch.bfloat16)
        self.x = self.x.cuda()
        mxfp8_ll.precompile(self.gate_up[0], [rows])
        mxfp8_ll.precompile(self.down[0], [rows])

    def __call__(self):
        from sgl_nvidia_mega.deepseek_v41 import sglang_ops
        from sgl_nvidia_mega.deepseek_v41.sm103.attention.q_norm_quant import (
            mxfp8_row_tiles,
        )

        x_q, x_s = mxfp8_row_tiles(self.x)
        gu = self.ll.linear(x_q, x_s, *self.gate_up)
        mid = torch.empty(self.x.size(0), reg.SLICE, dtype=torch.bfloat16,
                          device=gu.device)  # fmt: skip
        sglang_ops.silu_and_mul_clamp(gu, mid, 10.0)
        h = torch.nn.functional.pad(mid, (0, reg.PADDED - reg.SLICE))
        return self.ll.linear(*mxfp8_row_tiles(h.contiguous()), *self.down)


def _replayed(gemm, x_q, x_s, ids, side):
    """Route + pair in a captured graph (side chain forked before, joined after),
    replayed three times after the output is cleared."""
    alt = torch.cuda.Stream()

    def body():
        main = torch.cuda.current_stream()  # under capture: the capture stream
        alt.wait_stream(main)
        with torch.cuda.stream(alt):
            side()
        result = _run(gemm, x_q, x_s, ids)
        main.wait_stream(alt)
        return result

    body()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, permuted = body()
    for _ in range(3):
        out.fill_(float("nan"))
        graph.replay()
    torch.cuda.synchronize()
    return _gathered(out, permuted)


@needs_moe
@pytest.mark.parametrize("build", list(BUILDS))
@pytest.mark.parametrize("window", [None, 1])
@pytest.mark.parametrize(
    "rows, distinct", [(6, 8), (6, 24), (6, 36), (16, 6), (48, 384)]
)
def test_graph_replay_beside_side_chain(target_layer, build, window, rows, distinct):
    gen = torch.Generator().manual_seed(500 + rows + distinct)
    default, slot = _pairs(target_layer, 6, build, ep_rank=window)
    side = SideChain(rows, gen)
    x_q, x_s = single._quantized_rows(rows, gen)
    ids = single._routing(rows, distinct, gen, 384, 6).cuda()
    ref = _gathered(*_run(default, x_q, x_s, ids))
    _assert_bitwise(ref, _replayed(default, x_q, x_s, ids, side), "default replay")
    _assert_bitwise(ref, _replayed(slot, x_q, x_s, ids, side), "slot replay")


# ------------------------------------------------------------------ late route record

_LATE = r"""
extern "C" __global__ void late_copy(const int* src, int* dst, int n,
                                     unsigned long long delay_ns) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); } while (t - t0 < delay_ns);
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];
}
"""


def _late_kernel():
    major, minor = torch.cuda.get_device_capability()
    return reg._load(
        reg.compile_source(_LATE, "late_copy", f"sm_{major}{minor}a"),
        "late_copy",
        (128, 1, 1),
        0,
    )


def _late_launch(kernel, src, dst, delay_us):
    reg._launch(
        kernel,
        (1, 1, 1),
        [
            reg._pointer(src),
            reg._pointer(dst),
            ctypes.c_int32(src.numel()),
            ctypes.c_uint64(int(delay_us * 1000)),
        ],
        torch.cuda.current_stream().cuda_stream,
    )


@needs_moe
@pytest.mark.parametrize("build", list(BUILDS))
@pytest.mark.parametrize("mode", ["eager", "graph_join"])
def test_late_route_record(target_layer, build, mode):
    rows, top_k = 6, 6
    gen = torch.Generator().manual_seed(77)
    default, slot = _pairs(target_layer, top_k, build)
    late = _late_kernel()
    x_q, x_s = single._quantized_rows(rows, gen)
    ids = _exact_routing(rows, top_k, 24, 384, gen).cuda()
    ctas = reg.max_ctas(rows, top_k, 384)
    ref = _gathered(*_run(default, x_q, x_s, ids))
    permuted, good = slot._route(ids, rows, ctas)
    torch.cuda.synchronize()
    record = torch.zeros_like(good)  # poisoned: zero tiles
    out = torch.empty((rows * top_k * reg.TILE, reg.HIDDEN), dtype=torch.bfloat16,
                      device="cuda")  # fmt: skip
    if mode == "eager":
        for _ in range(3):
            record.zero_()
            out.fill_(float("nan"))
            _late_launch(late, good, record, 50.0)
            slot._gemms(x_q, x_s, record, ctas, gemm2_rows=out)
            torch.cuda.synchronize()
            _assert_bitwise(ref, _gathered(out, permuted), "eager late record")
        return
    alt = torch.cuda.Stream()

    def body():
        main = torch.cuda.current_stream()  # under capture: the capture stream
        alt.wait_stream(main)
        with torch.cuda.stream(alt):
            _late_launch(late, good, record, 50.0)
        main.wait_stream(alt)  # GEMM1 is a PDL launch right after the join
        slot._gemms(x_q, x_s, record, ctas, gemm2_rows=out)

    body()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    for _ in range(5):
        record.zero_()
        out.fill_(float("nan"))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        _assert_bitwise(ref, _gathered(out, permuted), "graph late record")

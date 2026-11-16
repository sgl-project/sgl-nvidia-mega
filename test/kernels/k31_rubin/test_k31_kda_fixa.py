# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The "fix A" mode: the KDA in-projection (``SplitKGemm``, fenced_trigger modes 1-3) releases
its PDL dependents only from a thread that executed ``griddepcontrol.wait``, behind
``fence.acq_rel.gpu`` (``.release`` before a later hardening), so the early KDA verify's pre-wait reads (slot table, scratch rows,
checkpoint rows, conv taps) are ordered after every write the projection's wait orders
(PTX 9.7.15.14). ``kda.verify.projection_fence()`` picks the mode (EARLY_FENCE_MODE whenever the
verify runs ``early``). Shapes: ``K31_TEST_HEADS`` (6 = TP16, 12 = TP8), Breq 1 x Q8.

Tests (tolerances stated before any run):
- ``test_projection_fence_bitwise``: each fenced mode vs shipped projection output BITWISE equal
  (eager and CUDA-graph replay); all within test_k31_projection's |out - ref| <= 1e-3 + 2^-7 |ref|
  of the FP32 ``x @ W.T`` reference. Catches a fix-A placement that breaks the GEMM.
- ``test_early_verify_ordering``: chain W -> P -> V on one stream, every launch PDL. W
  triggers its dependents at entry, sleeps ``DELAY_NS`` on %globaltimer, then writes the fresh
  checkpoint state, conv taps, slot table, scratch rows and the projection input x over stale
  values (stale slot -> a NaN-poisoned state/conv slot, stale x = NaN). P = the projection built
  through ``projection_fence()``; V = the verify with ``early+wide+rs+lt``. Required: V's
  output and conv windows BITWISE equal a host-synchronized run of the same kernels on the
  fresh data, which is bitwise equal to the shipped (non-early) verify and within
  test_kda_verify's .02 + .02|ref| of the independent FP32 reference ``_ref_attnkda``. Cases:
  ``shortk`` (K 512, 128.8.1.4: 4 K tiles <= 4 stages, so the shipped weight-warp
  trigger fires before any wait), ``hl5`` (K 7168, 64.8.2.12) and ``default`` (providers'
  tactic for 8 rows); fence ``fmode`` 1-3 forced or ``auto`` (projection_fence() under early).
  Eager and graph. DELAY_NS (20 ms) exceeds the host time to enqueue P and V in eager mode.
- ``test_delay_harness_sensitivity``: positive control. The same chain with the test-only
  kda.verify.projection_fence patched to 0 (the unfenced trigger under early) and ``shortk`` MUST show a mismatch (V started before W's
  writes); if it ever matches, the harness can no longer see a premature start and the test fails.

Mutations (each must make this file fail; fixed harness, suite VALID): ``fa_act_trigger_first``,
``fa_idle_trigger_first``, ``fa_late_trigger_first`` (modes 1/2/3: the fenced trigger before
that thread's PDL wait), ``fa_keep_weight_trigger`` (modes 1/2: the weight warp keeps its unfenced
trigger), ``fa_coupling_off`` (early no longer turns fix A on), ``fa_drop_act_wait`` (the
activation warp skips its PDL wait: P reads x before W writes it).
"""

import os

import pytest
import torch

import _harness as H
import _ref_attnkda as R
import test_k31_kda as T

# Anchors re-made unique: the K-16 fence and the K-42 trigger barrier now sit between the
# activation warp's wait and its trigger, and weight_warp_pair (act_fuse) and epilog_warp_act repeat weight_warp's trigger
# tail and epilog_warp's reduction, so the old anchors matched 0 or 2 times (an aborted mutated import, never a real probe).
# Where a block is repeated, the mutation edits each copy through its own unique context (row0 / up0 TMA operand).
_ACT_WAIT = (
    "        cute.arch.griddepcontrol_wait()\n"
    "        if cutlass.const_expr(self.k16_fence):  # K-16: B was written generically upstream; TMA reads it\n"
    "            _fence_proxy_async_global()\n"
)
_ACT_TRIGGER = (
    "        if cutlass.const_expr(self.fenced_trigger == 1):\n"
    "            _trigger_barrier(False)  # K-42: after the MMA warp's TMEM alloc\n"
    "            _fenced_launch_dependents()\n"
)
_IDLE_WAIT = "                cute.arch.griddepcontrol_wait()\n"
_IDLE_TRIGGER = (
    "                _trigger_barrier(False)  # K-42: after the MMA warp's TMEM alloc\n"
    "                _fenced_launch_dependents()\n"
)
_W_TAIL = (
    "a_map, k, {r})\n"
    "            if stage == stages - 1:\n"
    "                empty_phase = empty_phase ^ 1\n\n"
    "        if cutlass.const_expr(self.fenced_trigger == 3):\n"
)
_LATE = "            cute.arch.griddepcontrol_wait()\n            _fenced_launch_dependents()\n"
_LATE_SWAP = "            _fenced_launch_dependents()\n            cute.arch.griddepcontrol_wait()\n"
_KEEP = "        elif cutlass.const_expr(self.fenced_trigger == 0):\n"
_REDUCE_TAIL = (
    "                    else:\n"
    "                        cute.arch.mbarrier_arrive_and_expect_tx(bar_reduce, expected)\n"
    "                _wait_cluster(bar_reduce, Int32(0))\n"
    "                for peer in cutlass.range_constexpr(self.split_k - {n}):\n"
    "                    for i in cutlass.range_constexpr(vpt):\n"
    "                        rAcc[i] = rAcc[i] + mailbox[peer * vpp + epi_tid * vpt + i]\n\n"
    "        if split_rank == OWNER_RANK:\n"
    "            rD.store(rAcc.load().to(c_dtype))\n"
    "            # Preserve TMEM coordinates"
)
_FX4 = (
    "    compiled = {c}(kernel, *args, options=f\"--gpu-arch={{arch}}\")\n"
    "    overflow = {o}\n"
    "    if overflow is not None:  # FX-4: fail closed"
)

MUTATIONS = {
    "fa_act_trigger_first": (
        "kimi_k31/k3/cutedsl/gemm/splitk.py",
        _ACT_WAIT + _ACT_TRIGGER,
        _ACT_TRIGGER + _ACT_WAIT,
    ),
    "fa_idle_trigger_first": (
        "kimi_k31/k3/cutedsl/gemm/splitk.py",
        _IDLE_WAIT + _IDLE_TRIGGER,
        _IDLE_TRIGGER + _IDLE_WAIT,
    ),
    "fa_late_trigger_first": [
        ("kimi_k31/k3/cutedsl/gemm/splitk.py", _W_TAIL.format(r=r) + _LATE, _W_TAIL.format(r=r) + _LATE_SWAP)
        for r in ("row0", "up0")
    ],
    "fa_keep_weight_trigger": [
        ("kimi_k31/k3/cutedsl/gemm/splitk.py", _W_TAIL.format(r=r) + _LATE + _KEEP, _W_TAIL.format(r=r) + _LATE + "        else:\n")
        for r in ("row0", "up0")
    ],
    "fa_coupling_off": (
        "kimi_k31/k3/cutedsl/kda/verify.py",
        '    return EARLY_FENCE_MODE if "early" in verify_opt() else 0\n',
        "    return 0\n",
    ),
    # Sanitizer control, not a bug: early's main-warp cluster wait moved before the gate wait, so racecheck
    # (which does not co-schedule a cluster) can finish the early verify; the same edit as test_k31_kda's entry.
    "ctl_early_cw_first": [
        (
            "kimi_k31/k3/cutedsl/kda/kernel.py",  # re-anchored (opt chunk changed the line)
            "                ready = None if self.chunk else operands(0)\n",
            "                if cutlass.const_expr(self.early):\n                    cute.arch.cluster_wait()\n"
            "                ready = None if self.chunk else operands(0)\n",
        ),
        (  # = test_k31_kda's entry (the wait now sits under a "not self.tsp" guard, same effect)
            "kimi_k31/k3/cutedsl/kda/kernel.py",
            "                        if cutlass.const_expr(self.early and start == 0 and t == 0):  # noqa: SIM102\n",
            "                        if cutlass.const_expr(False):  # noqa: SIM102\n",
        ),
    ],
    "fa_drop_act_wait": (
        "kimi_k31/k3/cutedsl/gemm/splitk.py",
        _ACT_WAIT,
        _ACT_WAIT.replace("        cute.arch.griddepcontrol_wait()\n", "", 1),
    ),
    # Each must fail test_projection_fence_bitwise_16rows.
    "t16_drop_peer": (  # split-K reduce skips the peer partial (epilog_warp's copy; = test_k31_projection's drop_peer)
        "kimi_k31/k3/cutedsl/gemm/splitk.py",
        _REDUCE_TAIL.format(n=1),
        _REDUCE_TAIL.format(n=2),
    ),
    # K-23 knob: must fail test_projection_k23_arm_bitwise (as a hang).
    "k23_splitk_tx_double": (  # the owner arms bar_reduce for twice the peers' bytes (epilog_warp's copy)
        "kimi_k31/k3/cutedsl/gemm/splitk.py",
        "                        _arm_cluster(bar_reduce, expected)\n" + _REDUCE_TAIL.format(n=1),
        "                        _arm_cluster(bar_reduce, 2 * expected)\n" + _REDUCE_TAIL.format(n=1),
    ),
    # FX-4: must fail test_projection_smem_budget (SplitKGemm's check; TM64 has its own).
    "fx4_skip": (  # the SMEM check never reports an overflow
        "kimi_k31/k3/cutedsl/gemm/projection.py",
        _FX4.format(c="_DSL_COMPILE", o="smem_overflow(compiled)"),
        _FX4.format(c="_DSL_COMPILE", o="None"),
    ),
    # KDA-7 (FX-4 under a compile cache): must fail test_projection_smem_budget_aot.
    "fx4c_routed": (  # projections compile through cute.experimental.compile again (what K31MK_KERNEL_CACHE wraps)
        "kimi_k31/k3/cutedsl/gemm/projection.py",
        _FX4.format(c="_DSL_COMPILE", o="smem_overflow(compiled)"),
        _FX4.format(c="cute.experimental.compile", o="smem_overflow(compiled)"),
    ),
    "fx4c_noir_pass": (  # an object without host IR passes FX-4 unchecked
        "kimi_k31/k3/cutedsl/gemm/projection.py",
        "        raise RuntimeError(\"FX-4: this SplitKGemm has no host IR to read",
        "        return None\n        raise RuntimeError(\"FX-4: this SplitKGemm has no host IR to read",
    ),
    "t16_ovr_rows": (  # K31MK_PROJ_TACTICS override ignores its rows key
        "kimi_k31/k3/cutedsl/gemm/projection.py",
        "table[kind, int(rows)] = tactic",
        "table[kind, 8] = tactic",
    ),
}

Q, D, HEADS = 8, R.D, R.HEADS
EARLY = "early+wide+rs+lt"
DELAY_NS = 20_000_000
MODES = ("auto", "1", "2", "3")
# shortk: K 512 = 4 K tiles <= 4 stages, so the shipped weight-warp trigger fires before any wait; M tile 128 keeps
# the grid small (26 CTAs at 6 heads, 50 at 12, each holding a whole SM's SMEM) so the verify's 8-CTA clusters can
# become resident while the writer sleeps (64.8.1.8 at K 1024 left no free 8-SM slot at 12 heads on GB300).
SHORT_K = 512
CASES = {"shortk": (SHORT_K, (128, 8, 1, 4)), "hl5": (7168, (64, 8, 2, 12)), "default": (7168, None)}

needs_gpu = T.needs_gpu


class _Env:
    """Set env vars for a block (verify knobs and the fence rule are read at prepare)."""

    def __init__(self, **kv):
        self.kv, self.old = kv, {}

    def __enter__(self):
        for k, v in self.kv.items():
            self.old[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return self

    def __exit__(self, *exc):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


_W = {}


def _writer():
    """DelayedCopy: triggers dependents at entry, waits for its predecessor, spins ``delay``
    ns on %globaltimer, then copies five int32 buffers src -> dst (grid-stride)."""
    if "w" in _W:
        return _W["w"]
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int64
    from cutlass.cute.runtime import make_fake_stream

    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.host import dynamic_fake
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.primitives import global_timer_ns, launch

    CTAS, THREADS = 64, 256

    class DelayedCopy:
        @cute.jit
        def __call__(self, s0, d0, s1, d1, s2, d2, s3, d3, s4, d4, delay: Int64, stream):
            args = (s0, d0, s1, d1, s2, d2, s3, d3, s4, d4, delay, stream)
            launch(self.kernel, args, CTAS, THREADS, pdl=True)

        @cute.kernel
        def kernel(self, args):
            s0, d0, s1, d1, s2, d2, s3, d3, s4, d4, delay = args
            cute.arch.griddepcontrol_launch_dependents()
            cute.arch.griddepcontrol_wait()
            t0 = global_timer_ns()
            now = global_timer_ns()
            while now - t0 < delay:
                now = global_timer_ns()
            bid, _, _ = cute.arch.block_idx()
            tid, _, _ = cute.arch.thread_idx()
            first = bid * THREADS + tid
            for src, dst in ((s0, d0), (s1, d1), (s2, d2), (s3, d3), (s4, d4)):
                for i in cutlass.range(first, src.shape[0], CTAS * THREADS, unroll=1):
                    dst[i] = src[i]

    fakes = [dynamic_fake(1, torch.int32) for _ in range(10)]
    _W["w"] = cute.compile(DelayedCopy(), *fakes, Int64(0), make_fake_stream())
    return _W["w"]


def _words(t):
    return t.contiguous().view(-1).view(torch.int32)


def _setup(case, g):
    """Weights, fresh and stale inputs for one chain (Breq 1)."""
    k, _ = CASES[case]
    layer = T._layer(g)
    cap, ring_len, scratch_rows = 5, 16, 3
    conv_fresh, state_fresh = T._pools(g, cap)
    good, bad = 1, 3  # fresh slot table -> good; stale -> bad (a NaN-poisoned slot)
    conv_stale, state_stale = conv_fresh.clone(), state_fresh.clone()
    conv_stale[bad].view(torch.int16).fill_(-1)  # BF16 NaN
    state_stale[bad].view(torch.int32).fill_(-1)  # FP32 NaN
    conv_stale[good].view(torch.int16).fill_(-1)  # the stale copy of the good slot is NaN too
    state_stale[good].view(torch.int32).fill_(-1)
    weight = H.randn((R.PITCH, k), k**-0.5, g)
    x_fresh = H.randn((Q, k), 1.0, g)
    return dict(
        layer=layer, cap=cap, ring_len=ring_len, scratch_rows=scratch_rows, k=k, weight=weight,
        fresh=dict(state=state_fresh, conv=conv_fresh, slots=torch.tensor([good], dtype=torch.int32, device="cuda"),
                   scratch=torch.tensor([2], dtype=torch.int32, device="cuda"), x=x_fresh),
        stale=dict(state=state_stale, conv=conv_stale, slots=torch.tensor([bad], dtype=torch.int32, device="cuda"),
                   scratch=torch.tensor([0], dtype=torch.int32, device="cuda"),
                   x=torch.full_like(x_fresh, float("nan"))),
    )


def _projection(s, case):
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.gemm.projection import PreparedProjection, projection_tactic
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.kda.verify import projection_fence

    tac = CASES[case][1] or projection_tactic("kda", Q, torch.device("cuda", torch.cuda.current_device()))
    proj = PreparedProjection(s["weight"], {Q: tac}, fenced_trigger=projection_fence()).prepare()
    return proj, tac


def _buffers(s):
    """Live buffers the chain reads (filled with stale values before each run)."""
    L = s["ring_len"]
    live = {k: v.clone() for k, v in s["stale"].items()}
    rings = dict(
        rawv=torch.zeros((s["cap"], HEADS, L, 128), dtype=torch.bfloat16, device="cuda"),
        rawk=torch.zeros((s["cap"], HEADS, L, 128), dtype=torch.bfloat16, device="cuda"),
        log_g=torch.zeros((s["cap"], HEADS, L, 128), dtype=torch.float32, device="cuda"),
        beta=torch.zeros((s["cap"], HEADS, L), dtype=torch.float32, device="cuda"),
    )
    windows = torch.zeros((s["scratch_rows"], Q, 3, 3 * D), dtype=torch.bfloat16, device="cuda")
    pools = T._NS(conv=live["conv"].transpose(-1, -2), temporal=live["state"], conv_windows=windows.transpose(-1, -2),
                  **rings)
    projected = torch.empty((Q, R.PITCH), dtype=torch.bfloat16, device="cuda")
    out = torch.empty((Q, D), dtype=torch.bfloat16, device="cuda")
    return live, pools, windows, projected, out


def _reset(s, live, windows, projected, out):
    for k, v in s["stale"].items():
        live[k].copy_(v)
    for t in (windows, projected, out):
        H.poison(t)
    torch.cuda.synchronize()


def _core(s, pools, opt):
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.kda.prepared import decode_conv_weight
    from sgl_nvidia_mega.kimi_k31.verify_kda import PreparedKdaVerify

    layer = s["layer"]
    weights = T._NS(
        conv_weight=decode_conv_weight(layer.conv1d.weight), has_bias=False, f_b=layer.f_b_proj.weight,
        a_log=layer.A_log, dt_bias=layer.dt_bias, norm_weight=layer.o_norm.weight.float(),
        lower_bound=T.LOWER, scale=128**-0.5, norm_eps=T.EPS,
    )
    with _Env(K31MK_KDA_VERIFY_OPT=opt):
        return PreparedKdaVerify(weights, ((1, Q, R.PITCH, 0),), torch.cuda.current_device()).prepare_caches(pools)


def _stages(s, live, pools, projected, out, proj, core, *, delay_ns):
    """The chain's three launches (W, P, V) as separate callables."""
    from cutlass import Int64

    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.host import dynamic_view, stream

    fresh, pairs = s["fresh"], []
    for key in ("state", "conv", "slots", "scratch", "x"):
        pairs += [dynamic_view(_words(fresh[key])), dynamic_view(live[key].view(-1).view(torch.int32))]
    step = T._NS(state_slots=live["slots"], scratch_rows=live["scratch"], query_width=Q)
    w = _writer()
    return (
        lambda: w(*pairs, Int64(delay_ns), stream(live["x"].device)),
        lambda: proj.run(live["x"], projected),
        lambda: core.run(projected, out, pools=pools, step=step),
    )


def _synced_reference(s, case, opt):
    """W, P, V on the fresh data with a host synchronize after each (no delay): out, windows, projected."""
    live, pools, windows, projected, out = _buffers(s)
    _reset(s, live, windows, projected, out)
    proj, _ = _projection(s, case)
    core = _core(s, pools, opt)
    for stage in _stages(s, live, pools, projected, out, proj, core, delay_ns=0):
        stage()
        torch.cuda.synchronize()
    return out.clone(), windows.clone(), projected.clone()


def _mode_env(fmode):
    return dict(K31MK_KDA_VERIFY_OPT=EARLY,
                K31MK_KDA_PROJ_FENCE=None if fmode == "auto" else fmode)


def _run_chain(s, case, opt, mode, env):
    with _Env(**env):
        live, pools, windows, projected, out = _buffers(s)
        proj, tac = _projection(s, case)
        core = _core(s, pools, opt)
        stages = _stages(s, live, pools, projected, out, proj, core, delay_ns=DELAY_NS)

        def run():
            for stage in stages:
                stage()

        _reset(s, live, windows, projected, out)
        if mode == "eager":
            run()
            torch.cuda.synchronize()
        else:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                run()  # warm-up outside capture
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=side):
                run()
            _reset(s, live, windows, projected, out)
            graph.replay()
            torch.cuda.synchronize()
        return out.clone(), windows.clone(), projected.clone(), tac, proj.fenced_trigger


def _row(s):
    return int(s["fresh"]["scratch"][0])


def _same(a, b):
    return bool(torch.equal(a.view(torch.int16), b.view(torch.int16)))


@needs_gpu
@pytest.mark.parametrize("case", list(CASES))
def test_projection_fence_bitwise(case):
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.gemm.projection import PreparedProjection, projection_tactic

    g = H.generator(6100)
    k, tac = CASES[case]
    tac = tac or projection_tactic("kda", Q, torch.device("cuda", torch.cuda.current_device()))
    weight = H.randn((R.PITCH, k), k**-0.5, g)
    x = H.randn((Q, k), 1.0, g)
    ref = (x.float() @ weight.float().T).to(torch.bfloat16)
    outs = {}
    for fmode in (0, 1, 2, 3):
        proj = PreparedProjection(weight, {Q: tac}, fenced_trigger=fmode).prepare()
        out = torch.empty((Q, R.PITCH), dtype=torch.bfloat16, device="cuda")
        H.poison(out)
        proj.run(x, out)
        torch.cuda.synchronize()
        H.check(f"fixa.proj.{case}.mode{fmode}", out, ref, 1e-3, 2**-7)
        graph_out = torch.empty_like(out)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            proj.run(x, graph_out)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            proj.run(x, graph_out)
        H.poison(graph_out)
        graph.replay()
        torch.cuda.synchronize()
        assert _same(graph_out, out), f"{case} mode {fmode}: graph replay differs from eager"
        outs[fmode] = out
    for fmode in (1, 2, 3):
        assert _same(outs[fmode], outs[0]), f"{case}: fenced mode {fmode} output differs from shipped"


# Line stated before any GPU run: the override
# K31MK_PROJ_TACTICS=kda:16=64.16.2.10 (Breq 2 rounds) resolves to that tactic at 16 rows and leaves the 8-row tactic
# alone; at 16 rows its output matches the FP32 reference (as above), is bitwise across fence modes 0-3 and eager vs
# graph, and is bitwise to the default 16-row tactic's output (both split 2: the same K partition and reduction order).
# Catches: an override that ignores its rows key (t16_ovr_rows), a dropped split-K partial (t16_drop_peer), a fence mode
# that changes data.
T16 = (64, 16, 2, 10)


@needs_gpu
def test_projection_fence_bitwise_16rows():
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.gemm.projection import PreparedProjection, projection_tactic

    dev, rows, k = torch.device("cuda", torch.cuda.current_device()), 2 * Q, 7168
    with _Env(K31MK_PROJ_TACTICS="kda:16=" + ".".join(map(str, T16))):
        assert tuple(projection_tactic("kda", rows, dev)) == T16, "16-row override not applied"
        over8 = tuple(projection_tactic("kda", Q, dev))
    with _Env(K31MK_PROJ_TACTICS=None):
        default16 = tuple(projection_tactic("kda", rows, dev))
        assert over8 == tuple(projection_tactic("kda", Q, dev)), "the 16-row override moved the 8-row tactic"
    assert default16 != T16 and default16[2] == T16[2], f"default 16-row tactic {default16}: comparison void"
    g = H.generator(6150)
    weight = H.randn((R.PITCH, k), k**-0.5, g)
    x = H.randn((rows, k), 1.0, g)
    ref = (x.float() @ weight.float().T).to(torch.bfloat16)
    outs = {}
    for tac in (default16, T16):
        for fmode in (0, 1, 2, 3):
            proj = PreparedProjection(weight, {rows: tac}, fenced_trigger=fmode).prepare()
            out = torch.empty((rows, R.PITCH), dtype=torch.bfloat16, device="cuda")
            H.poison(out)
            proj.run(x, out)
            torch.cuda.synchronize()
            H.check(f"fixa.proj16.{'.'.join(map(str, tac))}.mode{fmode}", out, ref, 1e-3, 2**-7)
            graph_out = torch.empty_like(out)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                proj.run(x, graph_out)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=side):
                proj.run(x, graph_out)
            H.poison(graph_out)
            graph.replay()
            torch.cuda.synchronize()
            assert _same(graph_out, out), f"{tac} mode {fmode}: graph replay differs from eager"
            outs[tac, fmode] = out
    for key, out in outs.items():
        assert _same(out, outs[default16, 0]), f"{key}: differs from {default16} mode 0"


# FX-4 (option B). Line stated before any GPU run: a SplitKGemm
# tactic whose SMEM use exceeds the launch allocation raises ValueError at compile / prepare, before any launch, naming
# the DSL's counts: 128.8.7.6 (233,608 B used of 232,448: the DSL's runtime message in b-l31 P1b) and 128.16.4.6
# (245,896 B: t2's compile warning); the served / candidate tactics 64.8.2.12, 128.8.2.6, 128.16.2.6, 64.16.2.10 and
# 128.8.4.6 compile without error. No kernel is launched here. Catches: a check that never fires (fx4_skip), one that
# rejects tactics that fit.
SMEM_OVER = {(128, 8, 7, 6): 233608, (128, 16, 4, 6): 245896}
SMEM_FIT = ((64, 8, 2, 12), (128, 8, 2, 6), (128, 16, 2, 6), (64, 16, 2, 10), (128, 8, 4, 6))


@needs_gpu
def test_projection_smem_budget():
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.gemm import projection as P

    arch = P.architecture(torch.device("cuda", torch.cuda.current_device()))
    for tac in SMEM_FIT:
        P.compile_projection(tac, torch.bfloat16, arch)
    for tac, used in SMEM_OVER.items():
        with pytest.raises(ValueError, match=f"{used}"):
            P.compile_projection(tac, torch.bfloat16, arch)


# KDA-7: every SplitKGemm instantiation a served boot compiles (TP16, from the plugin's tables:
# KDA in-proj 1-7 / 8 / 16 / 32 / 64 rows at fix A mode 3 with the served kda:8 / kda:16 overrides, MLA in-proj, dense
# gate-up / down, shared gate-up (FP32 out) / down at shared width 384, the drafter's x-dgemm v3 table), plus the
# two over-cap tactics. Over-cap counts are independent records: 245,896 = a runtime warning's, 233,608 = a separate probe's.
SMEM_SERVED = (
    ((128, 8, 2, 6), torch.bfloat16, 3), ((64, 8, 2, 12), torch.bfloat16, 3), ((64, 16, 2, 10), torch.bfloat16, 3),
    ((64, 32, 2, 9), torch.bfloat16, 3), ((64, 64, 2, 6), torch.bfloat16, 3),
    ((128, 8, 4, 6), torch.bfloat16, 0), ((128, 16, 4, 5), torch.bfloat16, 0), ((64, 32, 2, 8), torch.bfloat16, 0),
    ((64, 64, 2, 6), torch.bfloat16, 0),
    ((64, 8, 4, 8), torch.bfloat16, 0), ((64, 16, 2, 8), torch.bfloat16, 0), ((64, 64, 2, 5), torch.bfloat16, 0),
    ((128, 8, 1, 6), torch.bfloat16, 0), ((128, 16, 1, 6), torch.bfloat16, 0), ((128, 32, 1, 5), torch.bfloat16, 0),
    ((64, 64, 1, 5), torch.bfloat16, 0),
    ((64, 8, 4, 8), torch.float32, 0), ((64, 16, 4, 10), torch.float32, 0), ((64, 32, 4, 8), torch.float32, 0),
    ((128, 8, 1, 3), torch.bfloat16, 0), ((64, 16, 1, 3), torch.bfloat16, 0), ((64, 32, 1, 3), torch.bfloat16, 0),
    ((64, 64, 1, 3), torch.bfloat16, 0),
    ((128, 8, 7, 5), torch.bfloat16, 0), ((64, 8, 4, 10), torch.bfloat16, 0), ((64, 8, 4, 3), torch.bfloat16, 0),
    ((64, 8, 1, 6), torch.bfloat16, 0), ((64, 8, 1, 7), torch.bfloat16, 0),
)


@needs_gpu
def test_projection_smem_budget_aot(tmp_path, monkeypatch):
    """FX-4 keeps its verdict while cute.experimental.compile serves AOT-loaded objects, as K31MK_KERNEL_CACHE does on a
    hit (its store / load calls: dump_to_object -> cute.runtime.load_module(path)[prefix]; no host IR): fitting
    tactics compile to IR-bearing objects, over-cap ones raise ValueError, and an IR-less object is refused."""
    H.load_plugin(MUTATIONS)
    import cutlass.cute as cute
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.gemm import projection as P

    arch = P.architecture(torch.device("cuda", torch.cuda.current_device()))
    real, served = cute.experimental.compile, []

    def aot(fn, *args, **kwargs):
        prefix = f"k31fx4c_{len(served)}"
        path = tmp_path / f"{prefix}.o"
        path.write_bytes(real(fn, *args, **kwargs).dump_to_object(prefix))
        served.append(prefix)
        return cute.runtime.load_module(str(path))[prefix]

    P.compile_projection.cache_clear()
    monkeypatch.setattr(cute.experimental, "compile", aot)
    try:
        for tac, dtype, fenced in SMEM_SERVED:
            code = P.compile_projection(tac, dtype, arch, fenced)
            assert code.ir_module is not None, (tac, dtype, fenced)
        for tac, used in SMEM_OVER.items():
            with pytest.raises(ValueError, match=f"232448, {used}"):
                P.compile_projection(tac, torch.bfloat16, arch)
        # engagement: the stand-in's object has no host IR, and FX-4 refuses it
        blob = P.compile_projection((64, 8, 2, 12), torch.bfloat16, arch, 3).dump_to_object("k31fx4c_probe")
        (tmp_path / "k31fx4c_probe.o").write_bytes(blob)
        loaded = cute.runtime.load_module(str(tmp_path / "k31fx4c_probe.o"))["k31fx4c_probe"]
        assert loaded.ir_module is None
        with pytest.raises(RuntimeError, match="no host IR"):
            P.smem_overflow(loaded)
    finally:
        P.compile_projection.cache_clear()


@needs_gpu
@pytest.mark.parametrize("mode", ("eager", "graph"))
@pytest.mark.parametrize("fmode", MODES)
@pytest.mark.parametrize("case", list(CASES))
def test_early_verify_ordering(case, fmode, mode):
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.kda.verify import EARLY_FENCE_MODE

    g = H.generator(6200 + list(CASES).index(case))
    s = _setup(case, g)
    with _Env(**_mode_env(fmode)):
        ref_out, ref_win, ref_proj = _synced_reference(s, case, EARLY)
    with _Env(K31MK_KDA_PROJ_FENCE=None):
        stock_out, stock_win, _ = _synced_reference(s, case, "")
    assert _same(ref_out, stock_out), f"{case}: early verify differs from the shipped verify on the same data"
    # Independent FP32 reference on the fresh data (test_kda_verify's output gate).
    fresh = s["fresh"]
    r_out, _, r_windows = R.kda_verify(
        ref_proj, fresh["conv"], fresh["state"], fresh["slots"], fresh["scratch"], Q, ring_len=s["ring_len"],
        **T._ref_kwargs(s["layer"]),
    )
    H.check(f"fixa.order.{case}.ref", ref_out, r_out, 0.02, 0.02)
    out, win, proj_out, tac, fenced = _run_chain(s, case, EARLY, mode, _mode_env(fmode))
    H.record({"kind": "fixa_order", "case": case, "fmode": fmode, "mode": mode, "tactic": list(tac), "fenced": fenced,
              "out_equal": _same(out, ref_out), "win_equal": _same(win[_row(s)], ref_win[_row(s)]),
              "proj_equal": _same(proj_out, ref_proj)})
    want = EARLY_FENCE_MODE if fmode == "auto" else int(fmode)
    assert fenced == want and fenced in (1, 2, 3), f"{case}: projection fence mode {fenced}, want {want} under {EARLY}"
    assert _same(proj_out, ref_proj), f"{case}/{fmode}/{mode}: projection read its input before the writer finished"
    assert _same(out, ref_out), f"{case}/{fmode}/{mode}: early verify output differs (read before the writer finished)"
    assert _same(win[_row(s)], ref_win[_row(s)]), f"{case}/{fmode}/{mode}: conv window differs (stale conv taps)"


@needs_gpu
@pytest.mark.parametrize("mode", ("eager", "graph"))
def test_delay_harness_sensitivity(mode):
    """Positive control: shipped (unfenced) trigger + early + short K must read stale data."""
    H.load_plugin(MUTATIONS)
    case = "shortk"
    g = H.generator(6200 + list(CASES).index(case))
    s = _setup(case, g)
    with _Env(K31MK_KDA_VERIFY_OPT=EARLY, K31MK_KDA_PROJ_FENCE=None):
        ref_out, _, _ = _synced_reference(s, case, EARLY)
    env = dict(K31MK_KDA_VERIFY_OPT=EARLY, K31MK_KDA_PROJ_FENCE=None)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.kda import verify as KV

    shipped_fence = KV.projection_fence
    KV.projection_fence = lambda: 0
    try:
        out, _, _, tac, fenced = _run_chain(s, case, EARLY, mode, env)
    finally:
        KV.projection_fence = shipped_fence
    stale = not _same(out, ref_out)
    H.record({"kind": "fixa_sensitivity", "mode": mode, "tactic": list(tac), "fenced": fenced, "stale_seen": stale})
    assert fenced == 0, "the test-only control did not turn fix A off"
    assert stale, f"{mode}: unfenced short-K chain matched: the delay harness no longer sees a premature start"

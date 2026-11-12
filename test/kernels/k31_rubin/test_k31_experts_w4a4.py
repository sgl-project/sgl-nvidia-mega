# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K3.1 W4A4 routed experts (``mkv2/experts_w4a4.py``, MXFP4 weights x MXFP4 activations; a lossy option behind
``K31MK_EXPERTS_W4A4``, default off) at TP16 (the c=1 MK path) and TP8 (the MKv2 engine, same seeds).

Method under test (the reference implements it independently, FP64): per 32 elements along K, UE8M0 scale =
2^ceil(log2(amax / 6)) (x: FP32 amax clamped at 1e-10; hidden: amax 0 -> 2^0), elements rounded to nearest E2M1 with
ties to the even code, saturating at 6. FC1 input = the FP32 latent, hidden = SiTU (FP64, rounded to FP32) of FC1.

Checks on every call (pre-registered before any run; tolerances never loosened):
  C1  hidden: the kernel's published hidden images (codes + scale bytes) of every unit, decoded, vs the reference
      hidden. A differing scale byte must have |log2(amax / 6) - round(.)| ln 2 <= 2^-8; a differing code (scale
      equal) must be one E2M1 step, same sign, at an element within 2^-8 (relative) of the two codes' midpoint.
  C2  teacher-forced FC2: the BF16 expert rows vs the FP64 FC2 of the kernel's OWN decoded hidden: Frobenius rel-L2
      <= 1e-2, every row <= 3e-2, all finite (outputs NaN-poisoned before every call).
  C3  end-to-end: the rows whose hidden equals the reference's (no C1 difference in that token's hidden of that
      expert) vs the full FP64 reference: Frobenius <= 1e-2, every row <= 3e-2; rows with a C1 difference are recorded.
Each case: 2 eager calls + 2 CUDA-graph replays with new inputs.

Mutations (each must make a base-passing test fail; which check should catch it in brackets):
  m4_nibble      the x plan packs each pair's two codes in the wrong nibbles                               [C1]
  m4_scale_id    every MMA reads scale ID 0                                                                [C1, C2]
  m4_hscale      hidden values scaled by 2^-(e+1) while the published scale says e                         [C1]
  m4_swizzle     the hidden image is written without the 64B-swizzle row bits                              [C1]
  m4_chunk2      TP8 FC2 K 256-383 reads hidden chunk 1                                                    [C2]
  m4_xstage      FC1 reads x at half the stage stride                                                      [C1]
  m4_noacquire   the FC2 loader does not wait for the hidden scale flags (epilogue publishes 3 us late)    [C2]
  m4_early_flag  the scale flag is released before the image words are stored (stores delayed 3 us)        [C2]
  m16_overlap    TP16 FC2 K 128-191 reads the lower half of W2 chunk 1 (elements 64-127)                    [C2]
  m16_hchunk     TP16 FC2 K 128-191 reads hidden chunk 0                                                   [C2]
  m16_tile       the FC1 epilogue writes FC1 tiles >= 2 into the wrong hidden chunk                        [C1]
  m4_hexp        the hidden scale exponent is one too small (kernel self-consistent: C2 cannot see it)     [C1]
  m4_situ        SiTU operands swapped (kernel self-consistent: C2 cannot see it)                          [C1]
  m4_w2scale     FC2 tile 1 of every pair reads tile 0's weight scales (hidden correct: C1 cannot see it)  [C2]
  m16_mask       the TP16 poll also checks the never-published scale bytes 6-7 (a hang without the poll's 2 s
                 deadline, hole H3: must fail-stop, not hang)                                              [deadline]

Timing (``K31_TIME=1``, ``test_time_w4a4``): the standard timing protocol at B8 (16 routes from 4 disjoint pools of 112
experts, E_d ~79, CUDA graph of 16 calls, median of 20 replays; ``W4_POOL`` / ``KW_TIME_SEED`` vary the routes) with arms
(``KW_TIME_ARMS``) ``cal`` (the TP8 planned anchor), ``a16`` (the shipped TP16 engine as the MK runs it at B8),
``b16`` (W4A4 TP16, xone, plan in the call, at its own grid ``W4A4_FREE``),
``b16n`` (W4A4 TP16 without xone, info), ``a16g`` / ``b16g`` (a16 / b16 at the TP16 bundle's grid SM count, info),
``b16@128`` (b16 at 128 CTAs, the per-CTA arrays' lower bound; FC1 tile balance; info), ``a16@128`` (the shipped engine at 128, fairness),
in the order A..Z Z..A A..Z Z..A (4 runs per arm).
"""

import os
import statistics

import pytest
import torch

import _harness as H
import _ref_moe as R
import test_k31_experts as TE
import test_k31_experts_tp16 as T16

M4 = "kimi_k31/k3/cutedsl/mkv2/experts_w4a4.py"
_FLAG = ("            store_release_u16(image + self.h4 + lane * 16 + 2 * chunk,"
         " Uint16((Uint32(scale_token) & 255) | (Uint32(scale_high) & 255) << 8))\n")
MUTATIONS = {
    "m4_nibble": (M4, '" ".join(f"cvt.rn.satfinite.e2m1x2.f32 q{i},${2 * i + 2},${2 * i + 1};" for i in range(4))',
                  '" ".join(f"cvt.rn.satfinite.e2m1x2.f32 q{i},${2 * i + 1},${2 * i + 2};" for i in range(4))'),
    "m4_scale_id": (M4, '        code.append(f"mov.b32 id,{idesc4(sid, sid, sv)};")\n',
                    '        code.append(f"mov.b32 id,{idesc4(0, 0, sv)};")\n'),
    "m4_hscale": (M4, "            v = h[t] * mx_rescale(scale)\n", "            v = h[t] * mx_rescale(scale + 1)\n"),
    "m4_swizzle": (M4, "            piece = (2 * (chunk % 2) + ew // 2) ^ (lane >> 1 & 3)\n",
                   "            piece = (2 * (chunk % 2) + ew // 2) ^ (lane & 3)\n"),
    "m4_chunk2": (M4, "        steps += [(2048 + j * 2, 64 + j * 2, 8, 8, j * 2) for j in range(2)]\n",
                  "        steps += [(2048 + j * 2, 32 + j * 2, 8, 8, j * 2) for j in range(2)]\n"),
    "m4_xstage": (M4, "tmem + acc * 8, a0 + slot * (SLOT // 16) + s * 1024, x0 + ks * 64,",
                  "tmem + acc * 8, a0 + slot * (SLOT // 16) + s * 1024, x0 + ks * 32,"),
    "m4_noacquire": [
        (M4, "            ready = cute.arch.vote_all_sync(not bad)\n", "            ready = cute.arch.vote_all_sync(True)\n"),
        (M4, "        cute.arch.sync_warp()\n        image = self.image(workspace, parity, expert)\n",
         "        cute.arch.sync_warp()\n        sleep_ns(3000)\n        image = self.image(workspace, parity, expert)\n"),
    ],
    "m4_early_flag": [
        (M4, "        cute.arch.sync_warp()\n        image = self.image(workspace, parity, expert)\n        if lane < 8:\n",
         "        cute.arch.sync_warp()\n        image = self.image(workspace, parity, expert)\n"
         "        if (ew == 0) & (lane < 8):\n" + _FLAG + "        sleep_ns(3000)\n        if lane < 8:\n"),
        (M4, "        if (ew == 0) & (lane < 8):\n            pair = (Uint32(scale_token) & 255) | (Uint32(scale_high) & 255) << 8\n"
             "            store_release_u16(image + self.h4 + lane * 16 + 2 * chunk, Uint16(pair))\n", ""),
    ],
    "m16_overlap": (M4, "        steps += [(1024 + 2, 32, 4, 4, 0)]\n", "        steps += [(1024 + 0, 32, 4, 4, 0)]\n"),
    "m16_hchunk": (M4, "        steps += [(1024 + 2, 32, 4, 4, 0)]\n", "        steps += [(1024 + 2, 0, 4, 4, 0)]\n"),
    "m16_tile": (M4, "            store_words(image + chunk // 2 * 512 + lane * 64 + piece * 16 + 8 * (ew % 2), data)\n",
                 "            store_words(image + chunk // 3 * 512 + lane * 64 + piece * 16 + 8 * (ew % 2), data)\n"),
    "m4_hexp": (M4, "    return Int32(127) if amax == 0 else mx4_exponent(amax)\n",
                "    return Int32(127) if amax == 0 else mx4_exponent(amax) - 1\n"),
    "m4_situ": (M4, "            h[t] = situ(gate, values[t], *scalars) if up else Float32(0)\n",
                "            h[t] = situ(values[t], gate, *scalars) if up else Float32(0)\n"),
    "m4_w2scale": (M4, "asf0 + slot * (SLOT_SF // 16) + t * self.fc2_scale_units),",
                   "asf0 + slot * (SLOT_SF // 16) + 0 * t * self.fc2_scale_units),"),
    # TP16 hidden poll checks the never-published bytes 6-7 (always 0xFF): spins forever without the 2 s deadline
    "m16_mask": (M4, "                mask = Uint32(65535)\n", "                mask = Uint32(4294967295)\n"),
}

needs_gpu = TE.needs_gpu

C_FRO, C_ROW = 1e-2, 3e-2  # the T1 bounds of test_k31_experts.py, unchanged
NEAR = 2.0 ** -8  # C1 window (L1's, unchanged)
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


# ------------------------------------------------------------------------------ reference (FP64)
def mx4_exp(amax, hidden):
    """Biased UE8M0 exponent of 2^ceil(log2(amax / 6)), exactly: amax = m 2^e (m in [0.5, 1)), 6 = 0.75 x 2^3, so the
    exponent is e - 3 + (m > 0.75). x: FP32 amax clamped at 1e-10; hidden: amax 0 -> 127."""
    a = amax.float()
    m, e = torch.frexp(a.clamp(min=1e-10) if not hidden else a)
    out = (e.double() - 3 + (m > 0.75).double() + 127).clamp(0, 254)
    if hidden:
        out = torch.where(a == 0, torch.full_like(out, 127.0), out)
    return out


def e2m1_rne(v):
    grid = torch.tensor(E2M1, dtype=torch.float64, device=v.device)
    mids = (grid[1:] + grid[:-1]) / 2
    a = v.abs().unsqueeze(-1)
    idx = (a > mids).sum(-1)
    tie = (a == mids).any(-1)
    idx = torch.where(tie & (idx % 2 == 1), idx + 1, idx)
    return torch.sign(v) * grid[idx]


def mx4_qdq(x, hidden=False):
    """(dequantized FP64 values, biased exponents [.., K/32]) of the MXFP4 method."""
    shape = x.shape
    g = x.float().reshape(*shape[:-1], shape[-1] // 32, 32)
    e = mx4_exp(g.abs().amax(-1), hidden)
    s = torch.exp2(e - 127.0).unsqueeze(-1)
    return (e2m1_rne(g.double() / s) * s).reshape(shape), e


def weights_for(tp):
    """(logical, host-layout) weights: TP8 L1's (``_ref_moe.shared_weights``), TP16 ``test_k31_experts_tp16``'s."""
    return R.shared_weights() if tp == 8 else T16.weights16()


# ------------------------------------------------------------------------------ engines
def engine4(tp, batch, knobs=(), deadline_ns=2_000_000_000):
    """``deadline_ns``: the hidden poll fail-stops after 2 s in the correctness tests (hole H3); the timing arms pass 0
    (production: no check)."""
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.mkv2 import experts_w4a4 as W4
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import planned as PLN

    _, prepared = weights_for(tp)
    flags = dict(xone="xone" in knobs, deadline_ns=deadline_ns)
    ex = W4.W4A4Experts(*prepared, beta=R.BETA, linear_beta=R.LIN, activation_floor=None, **flags)
    assert ex.tp == tp
    buffers = PLN.scratch(tp, torch.device("cuda"))
    out = torch.empty((batch * 16, R.H_LAT), dtype=torch.bfloat16, device="cuda")
    ex.prepare(batch, buffers, out)
    ex.scratch_buffers = buffers
    return ex, out


def image_parity(ex):
    """Hidden-image parity of the last call: its experts call counter (``ws_counter``) is call + 1 afterwards."""
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe.experts import ExpertsGeometry

    at = ExpertsGeometry(ex.tp).ws_counter
    count = int(ex.scratch_buffers["workspace"][at:at + 4].view(torch.int32)[0])
    return (count - 1) & 1


def decode_hidden(ex, parity, unit):
    """The kernel's published hidden of ``unit``: (E2M1 values [8, I] FP64, biased scale bytes [8, I / 32])."""
    inter, h4 = (384, 1536) if ex.tp == 8 else (192, 1024)
    ws = ex.scratch_buffers["workspace"]
    base = (parity * 128 + unit) * (h4 + 128)
    img = ws[base:base + h4 + 128]
    k = torch.arange(inter, device="cuda")
    t = torch.arange(8, device="cuda").view(8, 1)
    byte = k // 2
    piece = (byte % 64) // 16 ^ (t >> 1 & 3)
    raw = img[byte // 64 * 512 + t * 64 + piece * 16 + byte % 16].long()
    code = torch.where(k % 2 == 0, raw & 15, raw >> 4)
    grid = torch.tensor(E2M1, dtype=torch.float64, device="cuda")
    vals = torch.where(code >= 8, -grid[code & 7], grid[code & 7])
    scales = img[h4 + t * 16 + torch.arange(inter // 32, device="cuda")].long()
    return vals, scales


def check_call(name, ex, logical, x, ids, rows):
    """C1-C3 on one call (rows: [B, 16, 3584] BF16)."""
    w13, s13, w2, s2 = logical
    inter, batch = w13.shape[1] // 2, x.shape[0]
    nb = inter // 32
    parity = image_parity(ex)
    # Units: capacity <= 2 runs one unit per routing pair (``ExpertsKernel.route``), else the u-th distinct expert.
    order = ids.flatten().tolist() if batch <= 2 else list(dict.fromkeys(ids.flatten().tolist()))
    xq, _ = mx4_qdq(x)
    grid = torch.tensor(E2M1, dtype=torch.float64, device="cuda")
    teacher = torch.full(rows.shape, float("nan"), dtype=torch.float64, device="cuda")
    ref = torch.full(rows.shape, float("nan"), dtype=torch.float64, device="cuda")
    differs = torch.zeros(rows.shape[:2], dtype=torch.bool, device="cuda")
    flips = bad_codes = bad_scales = scale_diff = 0
    for u, e in enumerate(order):
        kval, kscale = decode_hidden(ex, parity, u)
        kval, kscale = kval[:batch], kscale[:batch]
        h1 = xq @ R.deq_mxfp4(w13[e], s13[e]).T
        a = R.situ(h1[:, :inter], h1[:, inter:]).float().double()  # FP64 SiTU rounded to FP32 (the epilogue's)
        amax = a.view(batch, nb, 32).abs().amax(-1)
        rexp = mx4_exp(amax, hidden=True)
        sdiff = kscale != rexp.long()
        ratio = (amax / 6.0).clamp(min=1e-300).log2()
        near_pow2 = (ratio - ratio.round()).abs() * 0.6931 <= NEAR
        bad_scales += int((sdiff & ~near_pow2).sum())
        scale_diff += int(sdiff.sum())
        v = a / torch.exp2(rexp - 127.0).repeat_interleave(32, dim=1)
        rq = e2m1_rne(v)
        diff = (kval != rq) & ~sdiff.repeat_interleave(32, dim=1)
        if int(diff.sum()):
            ki = (kval.abs().unsqueeze(-1) == grid).float().argmax(-1)
            ri = (rq.abs().unsqueeze(-1) == grid).float().argmax(-1)
            lo = torch.minimum(ki, ri).clamp(max=6)
            mid = (grid[lo] + grid[lo + 1]) / 2
            ok = (((ki - ri).abs() == 1) & ((v.abs() - mid).abs() <= NEAR * mid)
                  & (torch.sign(kval) * torch.sign(rq) >= 0))
            bad_codes += int((diff & ~ok).sum())
            flips += int(diff.sum())
        token_differs = diff.any(-1) | sdiff.any(-1)
        w2d = R.deq_mxfp4(w2[e], s2[e])
        mine = (kval * torch.exp2(kscale.double() - 127).repeat_interleave(32, dim=1)) @ w2d.T
        theirs = (rq * torch.exp2(rexp - 127.0).repeat_interleave(32, dim=1)) @ w2d.T
        where = (ids == e).nonzero()
        teacher[where[:, 0], where[:, 1]] = mine[where[:, 0]]
        ref[where[:, 0], where[:, 1]] = theirs[where[:, 0]]
        differs[where[:, 0], where[:, 1]] = token_differs[where[:, 0]]
    c1 = {"units": len(order), "codes": len(order) * batch * inter, "flips": flips, "unexplained_codes": bad_codes,
          "scale_diff": scale_diff, "unexplained_scales": bad_scales}
    out = rows.double()
    finite = bool(torch.isfinite(rows.float()).all())
    per_row = lambda o, r: ((o - r).norm(dim=-1) / r.norm(dim=-1).clamp(min=1e-300))  # noqa: E731
    c2_rows = per_row(out, teacher)
    c2 = {"fro": R.rel_l2(out, teacher), "row_max": float(c2_rows.max())}
    keep = ~differs
    c3_rows = per_row(out, ref)
    some = bool(keep.any())  # (a broken hidden can make every row differ: C1 then fails first)
    c3 = {"rows_same_hidden": int(keep.sum()), "fro": R.rel_l2(out[keep], ref[keep]) if some else 0.0,
          "row_max": float(c3_rows[keep].max()) if some else 0.0, "rows_hidden_differs": int(differs.sum()),
          "row_max_hidden_differs": float(c3_rows[differs].max()) if bool(differs.any()) else 0.0,
          "row_max_all": float(c3_rows.max())}
    if bool(differs.any()):
        c3["differ_rows"] = [(int(t), int(s), round(float(c3_rows[t, s]), 5)) for t, s in differs.nonzero()][:16]
    stats = {"kind": "check", "item": name, "finite": finite, "c1": c1, "c2": c2, "c3": c3}
    print("CHECK", stats, flush=True)
    H.record(stats)
    assert finite, f"{name}: non-finite rows"
    assert bad_codes == 0 and bad_scales == 0, f"{name}: C1 {c1}"
    assert c2["fro"] <= C_FRO and c2["row_max"] <= C_ROW, f"{name}: C2 {c2}"
    assert c3["fro"] <= C_FRO and c3["row_max"] <= C_ROW, f"{name}: C3 {c3}"
    return stats


def route(batch, g, case):
    if case == "random":
        return R.random_route(batch, g)
    if case == "distinct":
        return R.random_route(batch, g, distinct=True)
    pool = torch.randperm(R.E_ALL, generator=g, device="cuda")[:16]  # "few": all tokens on one set
    ids = pool.view(1, 16).expand(batch, 16).contiguous().int()
    w = torch.rand(batch, 16, generator=g, device="cuda") + 0.1
    return ids, (w / w.sum(-1, keepdim=True)).float().contiguous()


CASES = [(b, "random") for b in (1, 2, 3, 4, 5, 8)] + [(8, "distinct"), (8, "few")]
TPS = [int(t) for t in os.environ.get("KW_TPS", "16,8").split(",")]


@needs_gpu
@pytest.mark.parametrize("tp", TPS, ids=lambda t: f"tp{t}")
@pytest.mark.parametrize("batch,case", CASES, ids=lambda v: str(v))
def test_w4a4(batch, case, tp):
    """C1-C3 on 2 eager calls + 2 graph replays. TP8 uses L1's seeds (test_mkv2_experts.test_w4a4), so its Rubin run
    re-runs the 3 calls that failed L1's T1 row bound (b4.random call3, b8.distinct call0, b8.random call2)."""
    H.load_plugin(MUTATIONS)
    logical, _ = weights_for(tp)
    ex, out = engine4(tp, batch, tuple(k for k in os.environ.get("KW_KNOBS", "xone").split(",") if k))
    g = H.generator((700 if tp == 8 else 1600) + 10 * batch + len(case))
    x = torch.empty((batch, R.H_LAT), device="cuda")
    ids = torch.empty((batch, 16), dtype=torch.int32, device="cuda")
    run = lambda: ex.run(x, ids)  # noqa: E731
    graph = None
    for call in range(4):
        x.copy_(TE.latent(batch, g))
        i, _ = route(batch, g, case)
        ids.copy_(i)
        H.poison(out)
        if call < 2:
            run()
        else:
            graph = graph or TE.graph_of(run)
            H.poison(out)
            graph.replay()
        torch.cuda.synchronize()
        check_call(f"w4a4.tp{tp}.b{batch}.{case}.call{call}", ex, logical, x, ids, out.view(batch, 16, -1))
    H.record({"kind": "resources", "item": f"w4a4.tp{tp}.b{batch}.{case}", "grid": ex.grid_for(batch),
              "knobs": str(ex.knobs), **H.function_resources(ex.prepared[batch][3]), **H.device_facts()})


# ------------------------------------------------------------------------------ timing (Rubin)
def time_routes(batch, gt):
    size = int(os.environ.get("W4_POOL", "112"))  # 112 -> ~79 distinct at B8
    pools = torch.randperm(R.E_ALL, generator=gt, device="cuda")[: 4 * size].view(4, size)
    return [R.random_route(batch, gt, pool=pools[i % 4]) for i in range(16)]


def arm(name, batch):
    """(runner, its rows view, reference check) of a timing arm. Suffix ``g``: the experts grid derived for H.bundle_grid_sms() SMs
    (``K31MK_EXPERTS_GRID_SMS``, arm G) instead of the device's."""
    if name.endswith("g") or "@" in name:  # ``@N``: an N-CTA grid (info arms: FC1 tile balance at TP16)
        base, sms = (name[:-1], str(H.bundle_grid_sms())) if name.endswith("g") else name.split("@")
        knob = "K31MK_EXPERTS_W4A4_GRID" if base.startswith("b16") else "K31MK_EXPERTS_GRID_SMS"
        os.environ[knob] = sms
        try:
            return arm(base, batch)
        finally:
            del os.environ[knob]
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import planned as PLN

    if name == "cal":  # L1's anchor: the TP8 planned engine, 6 slots, x plan (s6x; s5x on GB300)
        logical, prepared = weights_for(8)
        stages = 6 if H.is_sm107() else 5
        ex = PLN.PlannedExperts(*prepared, beta=R.BETA, linear_beta=R.LIN, activation_floor=None, stages=stages,
                                xplan=True, batches="1-8:1")
        out = torch.empty((batch * 16, R.H_LAT), dtype=torch.bfloat16, device="cuda")
        ex.prepare(batch, PLN.scratch(8, torch.device("cuda")), out)
        return ex, out, lambda x, ids, rows: TE.check_rows(f"time.cal.b{batch}", rows, R.expert_rows(logical, x, ids))
    if name == "a16":  # the shipped TP16 engine, as the MK runs it at B8 (non-publish, streamed producer)
        logical, prepared = weights_for(16)
        ex, out = T16.persistent(prepared, batch)
        import _ref_moe16 as M

        return ex, out, lambda x, ids, rows: TE.check_rows(f"time.{name}.b{batch}", rows, M.expert_rows(logical, x, ids))
    knobs = {"b16": ("xone",), "b16n": ()}[name]
    ex, out = engine4(16, batch, knobs, deadline_ns=0)
    logical, _ = weights_for(16)
    return ex, out, lambda x, ids, rows: check_call(f"time.{name}.b{batch}", ex, logical, x, ids, rows)


@needs_gpu
def test_time_w4a4():
    """Pre-registered Rubin timing: arms of ``KW_TIME_ARMS``, A..Z Z..A A..Z Z..A on this GPU; each arm
    first passes its correctness check on the first timing route."""
    if os.environ.get("K31_TIME") != "1" or os.environ.get("K31_MUTATION"):
        pytest.skip("timing")
    H.load_plugin(MUTATIONS)
    batch = 8
    names = os.environ.get("KW_TIME_ARMS", "cal,a16,b16,b16n,a16g,b16g,b16@128,a16@128").split(",")
    gt = H.generator(int(os.environ.get("KW_TIME_SEED", 900 + batch)))
    routes = time_routes(batch, gt)
    xs = [TE.latent(batch, gt) for _ in routes]
    runners = {}
    for name in names:
        ex, out, check = arm(name, batch)
        H.poison(out)
        ex.run(xs[0], routes[0][0])
        torch.cuda.synchronize()
        check(xs[0], routes[0][0], out.view(batch, 16, -1))
        runners[name] = ex
    distinct = sum(int(torch.unique(r[0]).numel()) for r in routes) / len(routes)
    calls = {k: [lambda ex=ex, x=x, r=r: ex.run(x, r[0]) for x, r in zip(xs, routes)] for k, ex in runners.items()}
    got = {k: [] for k in names}
    for k in names + names[::-1] + names + names[::-1]:
        got[k].append(H.cold_time_us(calls[k])["us_median"])
    for k in names:
        row = {"kind": "time", "item": f"w4a4.time.b{batch}.{k}", "us_median": statistics.median(got[k]),
               "us_runs": got[k], "us_spread": max(got[k]) - min(got[k]), "distinct_mean": distinct,
               "grid": runners[k].grid_for(batch), **H.device_facts()}
        H.record(row)
        print("TIME", row, flush=True)


# ------------------------------------------------------------------------------ x quantizer vs lossy-evals' recipe
@needs_gpu
@pytest.mark.parametrize("batch", (1, 8), ids=lambda b: f"b{b}")
def test_plan4_vs_lossy_recipe(batch):
    """For the accuracy gate (a real W4A4 kernel inherits the lossy-line accuracy result only with a bitwise
    quantizer test vs lossy-evals' reference recipe). The W4A4 x plan (``W4A4PlanKernel``, FP32 latent) decoded and
    dequantized must equal, value for value, lossy-evals' ``mx_fake_quant(x, "mxfp4", amax_floor=1e-10)`` (the Triton
    function the HW4 vehicle ran, loaded from ``K31_LOSSY_MXFQ`` = that host tree's mx_fake_quant.py; the test FAILS when
    the path is missing, it never skips) and this file's independent FP64 ``mx4_qdq``. Inputs: random rows with
    outliers, an all-zero block, tiny (1e-30) and huge (1e30) blocks, exact E2M1 midpoints. -0 == +0 (values compared)."""
    H.load_plugin(MUTATIONS)
    import importlib.util

    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.host import stream, to_cute
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.mkv2 import experts_w4a4 as W4
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import planned as PLN

    path = os.environ.get("K31_LOSSY_MXFQ", "")
    assert path and os.path.exists(path), "K31_LOSSY_MXFQ must name lossy-evals' mx_fake_quant.py"
    spec = importlib.util.spec_from_file_location("lossy_mx_fake_quant", path)
    ref_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref_mod)
    g = H.generator(1400 + batch)
    x = torch.randn((batch, R.H_LAT), generator=g, device="cuda")
    x[:, ::97] *= 8.0
    x[:, :32] = 0.0
    x[:, 32:64] *= 1e-30
    x[:, 64:96] *= 1e30
    mids = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.75, -2.5], device="cuda")
    x[:, 96:96 + mids.numel()] = mids
    x[:, 96 + 31] = 6.0  # the block's amax 6: scale 2^0, so the midpoints are exact ties
    plan = torch.full((PLN.PLAN_BYTES,), 0xA5, dtype=torch.uint8, device="cuda")
    W4.compile_plan4(batch, torch.cuda.current_device())(to_cute(x, 16), to_cute(plan, 16), stream(x.device))
    torch.cuda.synchronize()
    codes = plan[: batch * 1792].view(batch, 1792)
    lo, hi = (codes & 15).long(), (codes >> 4).long()
    nib = torch.stack((lo, hi), -1).reshape(batch, R.H_LAT)
    grid = torch.tensor(E2M1, dtype=torch.float64, device="cuda")
    vals = torch.where(nib >= 8, -grid[nib & 7], grid[nib & 7])
    sc = plan[W4.PLAN4_XSF: W4.PLAN4_XSF + batch * 448].view(batch, 28, 16)[:, :, :4].reshape(batch, 112)
    kernel = (vals.view(batch, 112, 32) * torch.exp2(sc.double() - 127.0).unsqueeze(-1)).reshape(batch, R.H_LAT)
    lossy = ref_mod.mx_fake_quant(x, "mxfp4", out=torch.empty((batch, R.H_LAT), dtype=torch.float32, device="cuda"),
                                  amax_floor=1e-10).double()
    mine, _ = mx4_qdq(x)
    bad_lossy = int((kernel != lossy).sum())
    bad_mine = int((kernel != mine).sum())
    rec = {"kind": "check", "item": f"w4a4.plan4_vs_lossy.b{batch}", "cells_vs_lossy_recipe": bad_lossy,
           "cells_vs_fp64_ref": bad_mine, "cells": kernel.numel()}
    H.record(rec)
    print("CHECK", rec, flush=True)
    assert bad_lossy == 0 and bad_mine == 0, rec

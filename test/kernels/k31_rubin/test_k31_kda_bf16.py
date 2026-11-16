# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""BF16 KDA recurrent state (default off; ``--mamba-ssm-dtype bfloat16``, a lossy
option): ``KdaSplit(bf16_state=True)`` at the K3.1 per-rank shapes (``K31_TEST_HEADS=6``: TP16).

The BF16 kernel widens every loaded state value to FP32 exactly and keeps all math in FP32, so on the
SAME state values it must equal the FP32-state kernel bit for bit; the only new rounding is the BF16
store of an updated state (non-ReplaySSM decode), which must be round-to-nearest-even of the FP32
kernel's new state. Lines (stated before any GPU run):

- verify (ReplaySSM, ``kimi_k31.verify_kda.PreparedKdaVerify``), Breq 1/2/4/8/16 x Q8 (8 / 16 run the SM107
  2-CTA-cluster schedule ``SM107_VERIFY``), served opt string and the base path: BF16-pool run vs FP32-pool run on ``state16.float()``: output, raw V / K, log-g and beta
  rings and conv windows bitwise equal; the BF16 checkpoint and conv bitwise unchanged; the BF16 run's
  output vs the independent FP32 torch reference (``_ref_attnkda.kda_verify`` on ``state16.float()``)
  within the suite's output gate (atol / rtol 0.02 / 0.02).
- decode (``kda.prepared.PreparedKda``), rows 1 / 2 / 4 / 8: output bitwise equal to the FP32 run on
  ``state16.float()``; the new BF16 state bitwise equal to ``torch.bfloat16`` (RNE) of the FP32 run's new
  state (independent rounding); slots not written untouched; conv bitwise equal to the FP32 run's.
  Rows 32 (6 heads: FP32 runs the staged early-3 schedule ``SCHEDULE[6]`` 32-64, BF16 falls back to early
  1, so the two runs need not share an FP32 order): the bitwise comparisons are recorded only; required:
  output vs the independent torch reference ``_ref_attnkda.kda_decode`` (on ``state16.float()``) within
  0.02 / 0.02, new BF16 state vs the reference's new state within 1e-3 + 2^-8 |ref| (FP32 gate + one BF16
  rounding), conv bitwise equal to the reference's.
- sensitivity control (verify): a +1 BF16 ulp change of one head's 16 state rows of request 0 in the BF16
  pool changes at least one output cell (the bitwise comparison can see a one-ulp state difference).

Mutations (each must make this file fail): ``bf16_swap`` (state halves widened in the wrong order),
``bf16_rz`` (stores round toward zero), ``bf16_store_swap`` (stored halves swapped).
"""

import os

import pytest
import torch

import _harness as H
import _ref_attnkda as R

MUTATIONS = {
    "bf16_swap": (
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        "values = (*unpack_bf16(hraw[p, j, 0]), *unpack_bf16(hraw[p, j, 1]))",
        "values = (*unpack_bf16(hraw[p, j, 1]), *unpack_bf16(hraw[p, j, 0]))",
    ),
    "bf16_rz": (
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        'words += "cvt.rn.bf16x2.f32 w0,$2,$1; cvt.rn.bf16x2.f32 w1,$4,$3; "',
        'words += "cvt.rz.bf16x2.f32 w0,$2,$1; cvt.rz.bf16x2.f32 w1,$4,$3; "',
    ),
    "bf16_store_swap": (
        "kimi_k31/k3/cutedsl/kda/kernel.py",
        'words += "cvt.rn.bf16x2.f32 w0,$2,$1; cvt.rn.bf16x2.f32 w1,$4,$3; "',
        'words += "cvt.rn.bf16x2.f32 w0,$1,$2; cvt.rn.bf16x2.f32 w1,$3,$4; "',
    ),
}

Q = 8
D, HEADS = R.D, R.HEADS
LOWER, EPS = -5.0, 1e-6
SERVED_OPT = "early+wide+rs+lt+tsp+dmc+chunk+cg+cs"

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU",
)


class _NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _layer(g):
    conv_w = (torch.randn((3 * D, 1, 4), generator=g, device="cuda") * 0.5).float()
    return _NS(
        conv1d=_NS(weight=conv_w, bias=None),
        dt_bias=(torch.randn((D,), generator=g, device="cuda") * 0.5).float(),
        A_log=(torch.randn((HEADS,), generator=g, device="cuda") * 0.5 - 1.0).float(),
        f_b_proj=_NS(weight=H.randn((D, 128), 0.1, g)),
        o_norm=_NS(weight=(1.0 + H.randn((128,), 0.2, g)).to(torch.bfloat16), eps=EPS),
        gate_lower_bound=LOWER,
    )


def _ref_kwargs(layer):
    return dict(
        conv_w=layer.conv1d.weight, f_b=layer.f_b_proj.weight, a_log=layer.A_log,
        dt_bias=layer.dt_bias, norm_w=layer.o_norm.weight, lower_bound=LOWER, eps=EPS,
    )


def _state16(g, slots):
    return (torch.randn((slots, HEADS, 128, 128), generator=g, device="cuda") * 0.1).to(torch.bfloat16)


def _bits(t):
    return t.view(torch.int16) if t.element_size() == 2 else t.view(torch.int32)


def _same(a, b):
    return torch.equal(_bits(a), _bits(b))


def _verify_case(requests, opt, state16, conv_phys, layer, proj, slots, scratch):
    """One ReplaySSM verify on its own rings / windows; returns (out, rings, windows, pools)."""
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.kda.prepared import decode_conv_weight
    from sgl_nvidia_mega.kimi_k31.verify_kda import PreparedKdaVerify

    cap, ring_len, scratch_rows = state16.shape[0], 16, 2 * requests + 1
    rawv = torch.zeros((cap, HEADS, ring_len, 128), dtype=torch.bfloat16, device="cuda")
    rawk = torch.zeros_like(rawv)
    log_g = torch.zeros((cap, HEADS, ring_len, 128), dtype=torch.float32, device="cuda")
    beta = torch.zeros((cap, HEADS, ring_len), dtype=torch.float32, device="cuda")
    windows = torch.zeros((scratch_rows, Q, 3, 3 * D), dtype=torch.bfloat16, device="cuda")
    for t in (rawv, rawk, log_g, beta, windows):
        H.poison(t)
    weights = _NS(
        conv_weight=decode_conv_weight(layer.conv1d.weight), has_bias=False, f_b=layer.f_b_proj.weight,
        a_log=layer.A_log, dt_bias=layer.dt_bias, norm_weight=layer.o_norm.weight.float(),
        lower_bound=LOWER, scale=128**-0.5, norm_eps=EPS,
    )
    pools = _NS(conv=conv_phys.transpose(-1, -2), temporal=state16, rawv=rawv, rawk=rawk, log_g=log_g, beta=beta,
                conv_windows=windows.transpose(-1, -2))
    old = os.environ.get("K31MK_KDA_VERIFY_OPT")
    os.environ["K31MK_KDA_VERIFY_OPT"] = opt
    try:
        core = PreparedKdaVerify(weights, ((requests, Q, R.PITCH, 0),), torch.cuda.current_device()).prepare_caches(pools)
    finally:
        if old is None:
            os.environ.pop("K31MK_KDA_VERIFY_OPT", None)
        else:
            os.environ["K31MK_KDA_VERIFY_OPT"] = old
    out = torch.empty((requests * Q, D), dtype=torch.bfloat16, device="cuda")
    H.poison(out)
    core.run(proj, out, pools=pools, step=_NS(state_slots=slots, scratch_rows=scratch, query_width=Q))
    torch.cuda.synchronize()
    return out, (rawv, rawk, log_g, beta), windows, core


@needs_gpu
@pytest.mark.parametrize("opt", (SERVED_OPT, "wide+rs"))
@pytest.mark.parametrize("requests", (1, 2, 4, 8, 16))
def test_kda_verify_bf16_state(requests, opt):
    H.load_plugin(MUTATIONS)
    g = H.generator(7100 + requests)
    layer = _layer(g)
    cap = 2 * requests + 3
    conv_phys = H.randn((cap, 3, 3 * D), 1.0, g)
    state16 = _state16(g, cap)
    s16_0, conv0 = state16.clone(), conv_phys.clone()
    rows = requests * Q
    proj = H.randn((rows, R.PITCH), 1.0, g)
    slots = torch.randperm(cap, generator=torch.Generator().manual_seed(requests))[:requests].to(torch.int32).cuda()
    scratch = torch.randperm(2 * requests + 1, generator=torch.Generator().manual_seed(7 + requests))[:requests]
    scratch = scratch.to(torch.int32).cuda()
    if requests >= 4:
        slots[2] = -1  # one padded request
    item = f"kda.verify.bf16.breq{requests}.{opt}"
    out16, rings16, win16, core16 = _verify_case(requests, opt, state16, conv_phys, layer, proj, slots, scratch)
    k16 = core16.provider.kernels[requests, Q]
    state32 = s16_0.float()
    out32, rings32, win32, _ = _verify_case(requests, opt, state32, conv_phys.clone(), layer, proj, slots, scratch)
    assert _same(out16, out32), f"{item}: output differs from the FP32-state kernel on the same values"
    for name, a, b in zip(("rawv", "rawk", "log_g", "beta"), rings16, rings32):
        assert _same(a, b), f"{item}: ring {name} differs"
    assert _same(win16, win32), f"{item}: conv windows differ"
    assert _same(state16, s16_0), f"{item}: BF16 checkpoint modified"
    assert _same(conv_phys, conv0), f"{item}: persistent conv modified"
    ref_out, _, _ = R.kda_verify(proj, conv0, state32, slots, scratch, Q, ring_len=16, **_ref_kwargs(layer))
    H.check(item + ".out_vs_ref", out16, ref_out, 0.02, 0.02)
    H.record({"kind": "resources", "item": item, **H.function_resources(k16), **H.device_facts()})
    # Sensitivity control: +1 BF16 ulp on head 0's first 16 state rows of request 0's checkpoint.
    s0 = int(slots[0])
    bumped = s16_0.clone()
    bumped[s0, 0, :16].view(torch.int16).add_(1)
    out_b, _, _, _ = _verify_case(requests, opt, bumped, conv0.clone(), layer, proj, slots, scratch)
    changed = int((out_b.view(torch.int16) != out16.view(torch.int16)).sum())
    H.record({"kind": "check", "item": item + ".ulp_control", "changed_cells": changed})
    assert changed > 0, f"{item}: a +1 ulp state change moved no output cell (comparison blind)"


@needs_gpu
@pytest.mark.parametrize("rows", (1, 2, 4, 8, 32))
def test_kda_decode_bf16_state(rows):
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.kda.prepared import PreparedKda, decode_mode

    dev = torch.device("cuda", torch.cuda.current_device())
    mode = decode_mode(rows, dev)
    g = H.generator(7200 + rows)
    layer = _layer(g)
    cap = 2 * rows + 2
    conv0 = H.randn((cap + 1, 3, 3 * D), 1.0, g)
    s16_0 = _state16(g, cap + 1)
    proj = H.randn((rows, R.PITCH), 1.0, g)
    perm = torch.randperm(cap, generator=torch.Generator().manual_seed(rows))[:rows] + 1
    slots = perm.to(torch.int32).cuda()
    if rows >= 4:
        slots[rows // 2] = 0  # one padding row
    item = f"kda.decode.bf16.b{rows}.mode{mode}"
    results = {}
    for name, state in (("bf16", s16_0.clone()), ("fp32", s16_0.float())):
        core = PreparedKda(layer, {rows: (R.PITCH, mode)}, False)
        conv = conv0.clone()
        core.prepare_caches((conv, state))
        out = torch.empty((rows, D), dtype=torch.bfloat16, device="cuda")
        H.poison(out)
        core.run(proj, out, caches=(conv, state), slots=slots)
        torch.cuda.synchronize()
        results[name] = (out, conv, state)
        if name == "bf16":
            H.record({"kind": "resources", "item": item, **H.function_resources(core.kernels[rows]), **H.device_facts()})
    (o16, c16, st16), (o32, c32, st32) = results["bf16"], results["fp32"]
    rne = st32.to(torch.bfloat16)  # torch's FP32 -> BF16 is round-to-nearest-even
    bad = int((st16.view(torch.int16) != rne.view(torch.int16)).sum())
    out_cells = int((o16.view(torch.int16) != o32.view(torch.int16)).sum())
    H.record({"kind": "check", "item": item + ".state_rne", "bad_cells": bad, "out_cells_vs_fp32": out_cells,
              "moved_cells": int((st16.view(torch.int16) != s16_0.view(torch.int16)).sum())})
    if rows <= 8:
        assert out_cells == 0, f"{item}: output differs from the FP32-state kernel on the same values"
        assert _same(c16, c32), f"{item}: conv state differs"
        assert bad == 0, f"{item}: {bad} BF16 state cells != RNE(FP32 kernel state)"
    else:
        ref_out, ref_conv, ref_state = R.kda_decode(proj, conv0, s16_0.float(), slots, **_ref_kwargs(layer))
        H.check(item + ".out_vs_ref", o16, ref_out, 0.02, 0.02)
        H.check(item + ".state_vs_ref", st16.float(), ref_state, 1e-3, 2**-8)
        assert _same(c16, ref_conv), f"{item}: conv state not bit-exact vs reference"
    written = [int(s) for s in slots.tolist() if s > 0]
    untouched = [s for s in range(cap + 1) if s not in written]
    for s in untouched:
        assert _same(st16[s], s16_0[s]), f"{item}: slot {s} (not written) changed"
    assert int((st16.view(torch.int16) != s16_0.view(torch.int16)).sum()) > 0, f"{item}: no state written"

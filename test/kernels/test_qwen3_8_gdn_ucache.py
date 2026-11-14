# SPDX-License-Identifier: Apache-2.0
"""The u-cache verify (runtime/gdn_ucache.py) on SM100 / SM103 / SM107.

``UcacheVerify.core`` with the plugin's rings and cursor commit
(``gdn_ucache_state.py``) over many verify steps with random accept counts, padded
rows and folds, against the source verifier's reference: an fp32 token-by-token
recurrence that advances each slot by exactly its accepted tokens (the bounds are the
source test's). Then, on SM107, the claims the port rests on: the flush options
change no bit, the fused conv is the separate conv's bits (SGLang's and the hoisted
copy, which agree bit for bit), the fused norm is within a bf16 rounding of SGLang's
gated norm, and a captured graph replays the eager bits. On any of them, at Qwen3.8's
verify shapes, the separate norm's verify launch (16 rows x 8 warps a program) is SGLang's
gated norm bit for bit, and the source's retiled verify conv is the hoisted conv's and
SGLang's conv's bits.
"""

import os

# As in serving (the server-args hook): FlashInfer fixes the materialize kernel's
# state dtype at the first flashinfer.gdn_kernels import; the verify must not care.
os.environ.setdefault("GDN_UCACHE_STATE_DTYPE", "fp16")

import msgspec  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

CAPABILITY = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
if CAPABILITY not in ((10, 0), (10, 3), (10, 7)):
    pytest.skip(
        "the u-cache flush runs on SM100, SM103 or SM107", allow_module_level=True
    )
if CAPABILITY == (10, 7):
    pytest.importorskip("cutlass")
else:
    pytest.importorskip("flashinfer.gdn_kernels.gdn_replay_mtp_umma")
from sgl_nvidia_mega.qwen3_8.kernels.ops.attention.linear import (  # noqa: E402
    gdn_ucache_state as ucache,
)
from sgl_nvidia_mega.qwen3_8.runtime import gdn_ucache as U  # noqa: E402

SM107 = CAPABILITY == (10, 7)
needs_sm107 = pytest.mark.skipif(not SM107, reason="the SM107 flush's options")
DEVICE = torch.device("cuda")
H, HV, K, V = 4, 12, 128, 128
NUM_DRAFT_TOKENS = 4
NUM_SLOTS = 8
# Real rows on slots 3, 5, 1, 6; slot 0 (the reserved padding slot) and -1 are padding.
CACHE_INDICES = [3, 0, 5, -1, 1, 6]
# Under by_request_row: each row's request-table row (padded rows' are never read).
REQUEST_ROWS = [2, 9, 7, 0, 4, 1]
NUM_REQUEST_ROWS = 10
NUM_STEPS = 32
# The source test's bounds against a ring-free fp32 recurrence: one verify from
# the initial fp16 state, over the whole horizon, and relative to the largest output.
FIRST_STEP_TOL = 8e-3
DRIFT_TOL = 1.6e-2
REL_TOL = 1e-2
# An in-place weight update at this step must reach the verify's gate parameters.
DT_BIAS_UPDATE_STEP = NUM_STEPS // 2


def _fp32_window(state, q, k, v, a, b, A_log, dt_bias):
    """Runs one row's T tokens from ``state`` [HV, V, K]; returns the per-token
    outputs [T, HV, V] and the states after each token [T, HV, V, K]."""
    group = v.shape[1] // q.shape[1]
    q_hat = F.normalize(q.float(), dim=-1).repeat_interleave(group, dim=1) * K**-0.5
    k_hat = F.normalize(k.float(), dim=-1).repeat_interleave(group, dim=1)
    log_decay = -torch.exp(A_log.float()) * F.softplus(a.float() + dt_bias.float())
    beta = torch.sigmoid(b.float())
    outputs, states = [], []
    for t in range(q.shape[0]):
        state = state * torch.exp(log_decay[t])[:, None, None]
        pred = torch.einsum("hvk,hk->hv", state, k_hat[t])
        u = (v[t].float() - pred) * beta[t][:, None]
        state = state + u[:, :, None] * k_hat[t][:, None, :]
        outputs.append(torch.einsum("hvk,hk->hv", state, q_hat[t]))
        states.append(state)
    return torch.stack(outputs), torch.stack(states)


def _rings(
    *,
    by_request_row=False,
    k_dtype=torch.bfloat16,
    max_tokens=NUM_DRAFT_TOKENS,
    h=H,
    hv=HV,
    slots=NUM_SLOTS,
):
    buffers = ucache.allocate_gdn_ucache(
        num_layers=2,
        num_slots=NUM_REQUEST_ROWS if by_request_row else slots,
        num_k_heads=h,
        num_v_heads=hv,
        head_k_dim=K,
        head_v_dim=V,
        max_num_draft_tokens=max_tokens,
        device="cuda",
        key_ring_fp16=False,
        by_request_row=by_request_row,
    )
    if buffers.k_ring.dtype != k_dtype:
        buffers = msgspec.structs.replace(buffers, k_ring=buffers.k_ring.to(k_dtype))
    return buffers


def _steps(seed, steps, batch, T, h=H, hv=HV):
    """Each step's served inputs: q / k / v views of packed [q | k | v] rows, b / a of
    [b | a] rows, and the accept counts."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)

    def randn(*shape, scale=1.0):
        return torch.randn(*shape, generator=gen, device=DEVICE) * scale

    out = []
    for _ in range(steps):
        rows = batch * T
        qkv = torch.cat(
            (randn(rows, h * K), randn(rows, h * K), randn(rows, hv * V, scale=0.5)),
            dim=1,
        ).to(torch.bfloat16)
        q, k, v = torch.split(qkv, [h * K, h * K, hv * V], dim=1)
        gates = torch.cat((randn(rows, hv), randn(rows, hv, scale=0.5)), dim=1).to(
            torch.bfloat16
        )
        accept = torch.randint(
            1, T + 1, (batch,), generator=gen, device=DEVICE, dtype=torch.int32
        )
        out.append(
            dict(
                q=q.view(1, rows, h, K),
                k=k.view(1, rows, h, K),
                v=v.view(1, rows, hv, V),
                b=gates[:, :hv],
                a=gates[:, hv:],
                accept=accept,
            )
        )
    return out


def _run(
    options,
    *,
    by_request_row=False,
    k_dtype=torch.bfloat16,
    seed=0,
    truth=True,
    steps=NUM_STEPS,
):
    """Drives one layer's verify (layer 0 of a 2-layer pool) for ``steps`` steps;
    returns the outputs, the final state, the rings and the cursors."""
    T = NUM_DRAFT_TOKENS
    batch = len(CACHE_INDICES)
    buffers = _rings(by_request_row=by_request_row, k_dtype=k_dtype)
    # Poisoned rings: a record read outside the live window turns into NaN.
    for ring in (buffers.k_ring, buffers.u_ring, buffers.g_ring):
        ring.fill_(float("nan"))
    gen = torch.Generator(device=DEVICE).manual_seed(seed + 1000)
    state = (
        torch.randn(NUM_SLOTS, HV, V, K, generator=gen, device=DEVICE) * 0.5
    ).half()
    reference = state.float().clone()
    A_log = (
        torch.full((HV,), -3.0, device=DEVICE)
        + (torch.randn(HV, generator=gen, device=DEVICE) * 0.3).abs()
    )
    dt_bias = (torch.randn(HV, generator=gen, device=DEVICE) * 0.5).to(torch.bfloat16)
    cache_indices = torch.tensor(CACHE_INDICES, dtype=torch.int32, device=DEVICE)
    request_rows = torch.tensor(REQUEST_ROWS, dtype=torch.int64, device=DEVICE)
    live = [r for r, slot in enumerate(CACHE_INDICES) if slot > 0]
    uv = U.UcacheVerify(
        device=DEVICE, first_layer_id=0, by_request_row=by_request_row, options=options
    )
    outputs, worst, worst_first, ref_max = [], 0.0, 0.0, 0.0
    folds = {r: 0 for r in live}
    for step, x in enumerate(_steps(seed, steps, batch, T)):
        if step == DT_BIAS_UPDATE_STEP:
            dt_bias.copy_((torch.randn(HV, generator=gen, device=DEVICE) * 0.5))
        hist_before = buffers.hist_len.clone()
        out = uv.core(
            0,
            x["q"],
            x["k"],
            x["v"],
            x["a"],
            x["b"],
            A_log,
            dt_bias,
            state,
            buffers.k_ring[0],
            buffers.u_ring[0],
            buffers.g_ring[0],
            buffers,
            cache_indices,
            T,
            request_rows,
        ).view(batch, T, HV, V)
        outputs.append(out.clone())
        ucache.commit_gdn_ucache_cursors(
            buffers=buffers,
            slots=cache_indices,
            num_accept_tokens=x["accept"],
            request_rows=request_rows if by_request_row else None,
        )
        if not truth:
            continue
        for r in range(batch):
            if r not in live:
                assert out[r].abs().max().item() == 0.0
                continue
            slot = CACHE_INDICES[r]
            tokens = slice(r * T, (r + 1) * T)
            y_ref, states = _fp32_window(
                reference[slot],
                x["q"][0, tokens],
                x["k"][0, tokens],
                x["v"][0, tokens],
                x["a"][tokens],
                x["b"][tokens],
                A_log,
                dt_bias,
            )
            err = (out[r].float() - y_ref).abs().max().item()
            worst = max(worst, err)
            if step == 0:
                worst_first = max(worst_first, err)
            ref_max = max(ref_max, y_ref.abs().max().item())
            reference[slot] = states[int(x["accept"][r]) - 1]
            ring_row = REQUEST_ROWS[r] if by_request_row else slot
            folds[r] += int(hist_before[ring_row] >= buffers.flush_min)
    torch.cuda.synchronize()
    if truth:
        report = (
            f"|y - fp32 truth|: first step {worst_first:.2e}, horizon {worst:.2e}, "
            f"relative {worst / ref_max:.2%} of max |y| {ref_max:.3f}"
        )
        print(report)
        assert worst_first < FIRST_STEP_TOL, report
        assert worst < DRIFT_TOL, report
        assert worst / ref_max < REL_TOL, report
        assert all(count > 0 for count in folds.values()), folds
    return outputs, state, buffers


OPTIONS = [U.PLAIN] + ([U.DEFAULT] if SM107 else [])


@pytest.mark.parametrize(
    "options", OPTIONS, ids=lambda o: "default" if o is U.DEFAULT else "plain"
)
def test_random_accept_matches_fp32_recurrence(options):
    _run(options)


@needs_sm107
@pytest.mark.parametrize("k_dtype", [torch.bfloat16, torch.float16])
def test_request_rows_and_key_ring_match_fp32_recurrence(k_dtype):
    _run(U.DEFAULT, by_request_row=True, k_dtype=k_dtype)


def _same(a, b, what):
    assert torch.equal(a, b) or (
        a.dtype.is_floating_point
        and torch.equal(torch.nan_to_num(a, nan=0.0), torch.nan_to_num(b, nan=0.0))
        and torch.equal(a.isnan(), b.isnan())
    ), what


@needs_sm107
def test_flush_options_change_no_bit():
    """strided q/k/v, PDL and flush-first (and rings by request row) are launch
    choices: outputs, the folded state, the rings and the cursors are the plain
    launch's bits."""
    plain = _run(U.PLAIN, truth=False, steps=12)
    for options, by_request_row in (
        (U.Options(fused_norm=False), False),
        (U.Options(pdl=False, flush_first=False, fused_norm=False), False),
        (U.Options(strided_qkv=False, pdl=False, fused_norm=False), False),
        (U.Options(fused_norm=False), True),
    ):
        got = _run(options, by_request_row=by_request_row, truth=False, steps=12)
        for step, (x, y) in enumerate(zip(plain[0], got[0])):
            _same(x, y, f"{options} output, step {step}")
        _same(plain[1], got[1], f"{options} state")
        if not by_request_row:
            for ring in ("k_ring", "u_ring", "g_ring", "hist_len", "cache_base"):
                _same(
                    getattr(plain[2], ring), getattr(got[2], ring), f"{options} {ring}"
                )


def _conv_inputs(batch, T, h, hv, seed):
    """Projection rows with gaps (the q|k|v columns of a q|k|v|z row), the conv
    weights / state and the deduplicated step windows, as the pool lays them out."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    D = (2 * h + hv) * K
    rows = batch * T
    proj = torch.randn(rows, D + hv * V, generator=gen, device=DEVICE).to(
        torch.bfloat16
    )
    weight = (torch.randn(D, 4, generator=gen, device=DEVICE) * 0.5).to(torch.bfloat16)
    conv_state = torch.randn(NUM_SLOTS + batch, D, 3, generator=gen, device=DEVICE).to(
        torch.bfloat16
    )
    phys = torch.zeros(batch, D, T + 2, dtype=torch.bfloat16, device=DEVICE)
    window = phys.as_strided((batch, T, D, 3), ((T + 2) * D, 1, T + 2, 1))
    return proj, proj[:, :D], weight, conv_state, phys, window


def _attn(weight, h, hv):
    from types import SimpleNamespace

    return SimpleNamespace(
        conv_weights=weight,
        bias=None,
        activation="silu",
        num_k_heads=h,
        num_v_heads=hv,
        q_dim=h * K,
        k_dim=h * K,
        v_dim=hv * V,
        num_q_heads=h,
        head_q_dim=K,
        head_k_dim=K,
        head_v_dim=V,
    )


def test_hoisted_conv_is_sglang_conv():
    batch, T = 6, NUM_DRAFT_TOKENS
    slots = torch.tensor([3, 0, 5, -1, 1, 6], dtype=torch.int32, device=DEVICE)
    rows = torch.arange(batch, dtype=torch.int32, device=DEVICE)
    results = []
    for hoisted in (False, True):
        proj, mixed, weight, conv_state, phys, window = _conv_inputs(batch, T, H, HV, 7)
        y = U._conv(
            _attn(weight, H, HV),
            mixed,
            conv_state,
            window,
            slots,
            rows,
            batch,
            T,
            hoisted,
        )
        results.append((y, conv_state, phys))
    live = [r for r, s in enumerate(slots.tolist()) if s >= 0]
    for (y0, s0, w0), (y1, s1, w1) in zip(results[:1], results[1:]):
        _same(y0.view(batch, T, -1)[live], y1.view(batch, T, -1)[live], "conv output")
        _same(s0, s1, "conv state")
        _same(w0[live], w1[live], "conv windows")


# Verify requests (x 4 draft rows: 4 to 320 rows)
VERIFY_BATCHES = (1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 32, 44, 48, 52, 56, 60, 64)
VERIFY_BATCHES += (68, 72, 80)


@pytest.mark.parametrize("batch", VERIFY_BATCHES)
def test_retiled_conv_is_the_hoisted_and_sglang_conv(batch, monkeypatch):
    """SGLANG_ENABLE_GDN_VERIFY_CONV_RETILE at Qwen3.8's shapes (16 key heads, 48 value
    heads): the source's retiled kernel writes the bits of the hoisted conv (the separate
    conv before it) and of SGLang's causal_conv1d_update: the outputs, the conv states
    and the deduplicated step windows, with padded rows."""
    from sgl_nvidia_mega.qwen3_8.kernels.ops.attention.gdn_front import conv as front

    h, hv, T = 16, 48, NUM_DRAFT_TOKENS
    retiled = []
    launch = front.launch_retiled

    def spy(**kw):
        retiled.append(kw["config"])
        return launch(**kw)

    monkeypatch.setattr(front, "launch_retiled", spy)
    # one state slot a row (two rows on one slot would race); every 5th row is padding
    slots = torch.arange(1, batch + 1, dtype=torch.int32, device=DEVICE)
    slots[3::5] = -1
    rows = torch.arange(batch, dtype=torch.int32, device=DEVICE)
    live = [r for r, s in enumerate(slots.tolist()) if s >= 0]

    def run(retile, hoisted, perturb=False):
        proj, mixed, weight, conv_state, phys, window = _conv_inputs(
            batch, T, h, hv, batch
        )
        if perturb:
            mixed[live[-1] * T + 1, 7] += 4
        y = U._conv(
            _attn(weight, h, hv),
            mixed,
            conv_state,
            window,
            slots,
            rows,
            batch,
            T,
            hoisted,
            retile=retile,
        )
        torch.cuda.synchronize()
        return y.view(batch, T, -1)[live], conv_state, phys[live]

    got = run(True, True)
    assert retiled == [front.RETILE_DEFAULT]
    for name, ref in (("hoisted", run(False, True)), ("SGLang's", run(False, False))):
        for what, a, b in zip(("output", "conv state", "windows"), got, ref):
            _same(a, b, f"retiled vs {name} conv at batch {batch}: {what}")
    # the comparison sees the rows: one changed x element changes its own request only
    moved = run(True, True, perturb=True)
    assert not torch.equal(moved[0][-1], got[0][-1]), "a perturbed row must change"
    _same(moved[0][:-1], got[0][:-1], "the other requests' outputs")


def test_retiled_conv_on_the_dense_window_is_sglang_conv():
    """The per-step window layout (no deduplication) takes the retiled kernel's
    per-step stores: SGLang's bits."""
    h, hv, T, batch = 16, 48, NUM_DRAFT_TOKENS, 6
    slots = torch.tensor([3, 0, 5, -1, 1, 6], dtype=torch.int32, device=DEVICE)
    rows = torch.arange(batch, dtype=torch.int32, device=DEVICE)
    live = [r for r, s in enumerate(slots.tolist()) if s >= 0]
    results = []
    for retile in (False, True):
        proj, mixed, weight, conv_state, _, _ = _conv_inputs(batch, T, h, hv, 11)
        window = torch.zeros(
            batch, T, mixed.shape[1], 3, dtype=torch.bfloat16, device=DEVICE
        )
        y = U._conv(
            _attn(weight, h, hv),
            mixed,
            conv_state,
            window,
            slots,
            rows,
            batch,
            T,
            False,
            retile=retile,
        )
        torch.cuda.synchronize()
        results.append((y.view(batch, T, -1)[live], conv_state, window[live]))
    for what, a, b in zip(("output", "conv state", "windows"), *results):
        _same(a, b, f"retiled vs SGLang's conv, per-step windows: {what}")


def _flush_batch(h, hv, T):
    """The smallest batch whose launch runs one CTA per (key head, row): the fused
    conv's case."""
    from sgl_nvidia_mega.qwen3_8.kernels.ops.attention.linear.gdn_ucache_rubin import (
        gdn_replay_mtp_umma_rubin as R,
    )

    for batch in range(1, 257):
        _, mixed, weight, conv_state, _, window = _conv_inputs(batch, T, h, hv, 0)
        fc = R.GdnFusedConv(
            x=mixed.view(batch, T, -1),
            weight=weight,
            conv_state=conv_state,
            window=window,
            window_indices=torch.zeros(batch, dtype=torch.int32, device=DEVICE),
        )
        if R.fused_conv_unsupported_reason(fc, num_k_heads=h, num_v_heads=hv) is None:
            return batch
    pytest.skip("no batch takes the fused conv")


def _verify_once(*, h, hv, batch, T, fused_conv, fused_norm, seed=3):
    """One first-layer verify on fresh copies: the separate conv + flush (+ SGLang's
    gated norm) or the fused launch. Returns everything it writes."""
    from types import SimpleNamespace

    from sglang.kernels.ops.attention.fla.layernorm_gated import layernorm_fn

    proj, mixed, weight, conv_state, phys, window = _conv_inputs(batch, T, h, hv, seed)
    gen = torch.Generator(device=DEVICE).manual_seed(seed + 1)
    rows = batch * T
    gates = torch.randn(rows, 2 * hv, generator=gen, device=DEVICE).to(torch.bfloat16)
    z = proj[:, (2 * h + hv) * K :].view(rows, hv, V)
    # one state slot a row (a slot verified twice in a launch would race)
    slots = torch.arange(1, batch + 1, dtype=torch.int32, device=DEVICE)
    slots[1] = -1
    buffers = _rings(h=h, hv=hv, slots=NUM_SLOTS + batch)
    torch.manual_seed(seed)
    for ring in (buffers.k_ring, buffers.u_ring):
        ring.copy_(torch.randn_like(ring, dtype=torch.float32).to(ring.dtype))
    buffers.g_ring.copy_(-torch.rand_like(buffers.g_ring) * 0.1)
    buffers.hist_len.copy_(torch.arange(buffers.hist_len.numel(), device=DEVICE) % 14)
    state = (torch.randn(NUM_SLOTS + batch, hv, V, K, device=DEVICE) * 0.5).half()
    A_log = torch.full((hv,), -3.0, device=DEVICE)
    dt_bias = torch.zeros(hv, dtype=torch.bfloat16, device=DEVICE)
    # the layer's RMSNormGated (norm before a sigmoid gate, no groups, no bias)
    norm = SimpleNamespace(
        weight=(torch.rand(V, device=DEVICE) + 0.5).to(torch.bfloat16),
        eps=1e-6,
        bias=None,
        group_size=None,
        norm_before_gate=True,
        activation="sigmoid",
    )
    options = U.Options(fused_conv=fused_conv, fused_norm=fused_norm)
    uv = U.UcacheVerify(device=DEVICE, first_layer_id=0, options=options)
    attn = _attn(weight, h, hv)
    window_rows = torch.arange(batch, dtype=torch.int32, device=DEVICE)
    fc = uv.fused_conv(
        attn=attn,
        mixed_qkv=mixed,
        conv_state=conv_state,
        window=window,
        window_indices=window_rows,
        batch_size=batch,
        draft_token_num=T,
    )
    if fused_conv:
        assert fc is not None
        convolved = torch.empty(
            (rows, mixed.shape[1]), dtype=mixed.dtype, device=DEVICE
        )
    else:
        convolved = U._conv(
            attn, mixed, conv_state, window, slots, window_rows, batch, T, True
        )
    q, k, v = U._split_qkv(attn, convolved, True)
    fn = uv.fused_norm(
        norm=norm, z=z, num_tokens=rows, num_draft_tokens=T, num_v_heads=hv
    )
    if fused_norm and fn is None:
        pytest.skip(f"the fused norm does not apply at batch {batch} on this GPU")
    assert fn is None or fused_norm
    out = uv.core(
        0,
        q,
        k,
        v,
        gates[:, hv:],
        gates[:, :hv],
        A_log,
        dt_bias,
        state,
        buffers.k_ring[0],
        buffers.u_ring[0],
        buffers.g_ring[0],
        buffers,
        slots,
        T,
        None,
        fc,
        fn,
    )
    if not fused_norm:
        # RMSNormGated.forward_cuda's call
        out = layernorm_fn(
            out.reshape(-1, V),
            norm.weight,
            None,
            z=z,
            eps=norm.eps,
            group_size=None,
            norm_before_gate=True,
            is_rms_norm=True,
            activation="sigmoid",
        ).reshape(rows, hv, V)
    torch.cuda.synchronize()
    live = [r for r in range(batch) if int(slots[r]) >= 0]
    return dict(
        out=out.reshape(batch, T, hv, V)[live].clone(),
        convolved=convolved.view(batch, T, -1)[live].clone(),
        conv_state=conv_state[slots[live].long()].clone(),
        windows=phys[live].clone(),
        state=state.clone(),
        k_ring=buffers.k_ring.clone(),
        u_ring=buffers.u_ring.clone(),
        g_ring=buffers.g_ring.clone(),
    )


@needs_sm107
def test_fused_conv_is_the_separate_conv():
    h, hv, T = 16, 48, NUM_DRAFT_TOKENS
    batch = _flush_batch(h, hv, T)
    separate = _verify_once(
        h=h, hv=hv, batch=batch, T=T, fused_conv=False, fused_norm=False
    )
    fused = _verify_once(
        h=h, hv=hv, batch=batch, T=T, fused_conv=True, fused_norm=False
    )
    for name in separate:
        _same(separate[name], fused[name], f"fused conv at batch {batch}: {name}")


@needs_sm107
@pytest.mark.parametrize("batch", [3, 16, 48])
def test_fused_norm_against_sglang_gated_norm(batch):
    h, hv, T = 16, 48, NUM_DRAFT_TOKENS
    separate = _verify_once(
        h=h, hv=hv, batch=batch, T=T, fused_conv=False, fused_norm=False
    )
    fused = _verify_once(
        h=h, hv=hv, batch=batch, T=T, fused_conv=False, fused_norm=True
    )
    for name in ("state", "k_ring", "u_ring", "g_ring", "conv_state", "windows"):
        _same(separate[name], fused[name], f"fused norm: {name}")
    a, b = separate["out"].float(), fused["out"].float()
    diff = (a - b).abs()
    print(
        f"batch {batch}: fused norm vs SGLang's gated norm: {int((diff > 0).sum())} of "
        f"{diff.numel()} elements differ, max {diff.max().item():.3e}"
    )
    torch.testing.assert_close(b, a, atol=1.6e-2, rtol=1e-2)


# Verify rows (requests x 4 draft rows): the small verifies of the low-concurrency
# points and the high-concurrency ones (to 80 requests)
VERIFY_ROWS = (4, 8, 12, 16, 24, 32, 48, 64, 80, 96, 128, 160, 176, 192, 208, 224, 240)
VERIFY_ROWS += (256, 272, 288, 304, 320)


class _GatedNorm:
    """The layer's RMSNormGated (norm before a sigmoid gate, no groups, no bias);
    calling it is RMSNormGated.forward_cuda's call."""

    def __init__(self, weight):
        self.weight, self.eps, self.bias, self.group_size = weight, 1e-6, None, None
        self.norm_before_gate, self.activation = True, "sigmoid"

    def __call__(self, x, z):
        from sglang.kernels.ops.attention.fla.layernorm_gated import layernorm_fn

        return layernorm_fn(
            x,
            self.weight,
            self.bias,
            z=z,
            eps=self.eps,
            group_size=self.group_size,
            norm_before_gate=self.norm_before_gate,
            is_rms_norm=True,
            activation=self.activation,
        )


def _verify_norm_inputs(rows, seed, h=16, hv=48):
    """The flush output [rows * HV, V], the strided gate view z [rows, HV, V] of the
    projection rows (q | k | v | z) and the layer's norm."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    D = (2 * h + hv) * K
    proj = torch.randn(rows, D + hv * V, generator=gen, device=DEVICE).to(
        torch.bfloat16
    )
    core = torch.randn(rows * hv, V, generator=gen, device=DEVICE).to(torch.bfloat16)
    weight = (torch.rand(V, generator=gen, device=DEVICE) + 0.5).to(torch.bfloat16)
    return core, proj[:, D:].view(rows, hv, V), _GatedNorm(weight)


@pytest.mark.parametrize("rows", VERIFY_ROWS)
def test_verify_norm_launch_is_sglang_gated_norm(rows, monkeypatch):
    """SGLANG_ENABLE_GDN_VERIFY_NORM_RETILE: wherever SGLang's launch holds >= 2 rows a
    program, the verify norm runs on 16 rows x 8 warps a program, each 128-wide row on one
    warp's 16 lanes x 8 elements as SGLang's launch: SGLang's bits. Elsewhere SGLang's own
    launch runs."""
    from sglang.kernels.ops.attention.fla.layernorm_gated import calc_rows_per_block

    from sgl_nvidia_mega.qwen3_8.kernels.ops.attention.linear import (
        gdn_prefill_norm as norm_mod,
    )

    launches = []
    launch = norm_mod._layer_norm_fwd

    def spy(*args, **kw):
        launches.append(kw.get("launch_cfg"))
        return launch(*args, **kw)

    monkeypatch.setattr(norm_mod, "_layer_norm_fwd", spy)
    hv = 48
    core, z, norm = _verify_norm_inputs(rows, seed=rows)
    stock_rows = calc_rows_per_block(rows * hv, DEVICE)
    got = U._gated_norm(U.DEFAULT, norm, core, z)
    want = norm(core, z)
    torch.cuda.synchronize()
    assert launches == ([(16, 8)] if stock_rows >= 2 else []), (rows, stock_rows)
    if rows >= 176:  # every high-concurrency verify takes it
        assert launches == [(16, 8)]
    _same(got, want, f"verify norm, {rows} rows ({stock_rows} rows a stock program)")
    # the comparison sees every row: a changed element changes its row only
    r = rows * hv // 2 + 1
    perturbed = core.clone()
    perturbed[r, 5] += 4
    moved = U._gated_norm(U.DEFAULT, norm, perturbed, z)
    torch.cuda.synchronize()
    assert not torch.equal(moved[r], got[r]), "a perturbed row must change"
    _same(moved[:r], got[:r], "rows before the perturbed one")
    _same(moved[r + 1 :], got[r + 1 :], "rows after the perturbed one")


@needs_sm107
def test_graph_replay_is_eager():
    """The prep and two layers' flushes captured once replay the eager bits (the
    shared output buffer, gate refresh tables and row order are built before)."""
    T, batch = NUM_DRAFT_TOKENS, len(CACHE_INDICES)
    x = _steps(5, 1, batch, T)[0]
    cache_indices = torch.tensor(CACHE_INDICES, dtype=torch.int32, device=DEVICE)
    A_log = torch.full((HV,), -3.0, device=DEVICE)
    dt_bias = torch.full((HV,), 0.25, dtype=torch.bfloat16, device=DEVICE)
    buffers = _rings()
    torch.manual_seed(5)
    for ring in (buffers.k_ring, buffers.u_ring):
        ring.copy_(torch.randn_like(ring, dtype=torch.float32).to(ring.dtype))
    buffers.g_ring.copy_(-torch.rand_like(buffers.g_ring) * 0.1)
    buffers.hist_len.copy_(torch.tensor([0, 5, 13, 2, 9, 7, 1, 4], device=DEVICE))
    state0 = (torch.randn(2, NUM_SLOTS, HV, V, K, device=DEVICE) * 0.5).half()
    snapshot = [t.clone() for t in (buffers.k_ring, buffers.u_ring, buffers.g_ring)]
    state = state0.clone()
    uv = U.UcacheVerify(device=DEVICE, first_layer_id=0)
    uv.warm_up(
        max_batch_size=batch,
        num_draft_tokens=T,
        state=state[0],
        k_ring=buffers.k_ring[0],
        u_ring=buffers.u_ring[0],
        g_ring=buffers.g_ring[0],
        flush_min=buffers.flush_min,
    )

    def forward():
        outs = []
        for layer in (0, 1):
            outs.append(
                uv.core(
                    layer,
                    x["q"],
                    x["k"],
                    x["v"],
                    x["a"],
                    x["b"],
                    A_log,
                    dt_bias,
                    state[layer],
                    buffers.k_ring[layer],
                    buffers.u_ring[layer],
                    buffers.g_ring[layer],
                    buffers,
                    cache_indices,
                    T,
                ).clone()
            )
        return outs

    def reset():
        state.copy_(state0)
        for t, s in zip((buffers.k_ring, buffers.u_ring, buffers.g_ring), snapshot):
            t.copy_(s)

    eager = forward()
    eager_state = state.clone()
    reset()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        forward()  # the capture warmup
    torch.cuda.current_stream().wait_stream(stream)
    reset()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = forward()
    reset()
    graph.replay()
    torch.cuda.synchronize()
    for layer, (a, b) in enumerate(zip(eager, captured)):
        _same(a, b, f"layer {layer} output")
    _same(eager_state, state, "state")

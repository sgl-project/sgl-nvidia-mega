# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Independent FP32 PyTorch references for the K3/K3.1 attention residual (AttnRes) and
KDA decode / ReplaySSM verify steps, written from the stock SGLang model math, not from
the megakernel source:

- AttnRes: ``sglang.srt.layers.attn_residual`` (``aggregate_stream_torch``, the fast TMA
  path's contract: BF16 combined weight ``cw = bf16(norm_w * proj_w)``, FP32 mixture, FP32
  output RMSNorm; ``_mix_fused`` for the FP32-cw aux stream) and ``AttnResidual.forward``
  (prefix = bf16(prefix + hidden), bank row snapshot at block-write layers).
- KDA: ``sglang.kernels.ops.attention.kda_fused_decode`` (causal conv1d update, width 4,
  SiLU, l2-normalized q/k with eps 1e-6 and scale 128^-0.5, sigmoid beta, "safe" gate
  ``g = lower_bound * sigmoid(exp(A_log) * (a + dt_bias))`` with ``a = bf16(f_a @ W_fb^T)``,
  per-key-channel decay ``exp(g)``, delta rule, sigmoid-gated output RMSNorm), and the
  ReplaySSM verify contract (state/conv untouched; raw K/V rings in BF16, log-g and beta
  rings in FP32, post-token conv windows).
"""

import os

import torch

H = 7168


def rms(x, eps):
    return torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def attnres(delta, prefix, blocks, *, nb, has_delta, write, norm_w, query_w, out_w, eps, out_eps):
    """Returns (out bf16 [T,H], new prefix bf16 [T,H], blocks after the write)."""
    new = (prefix.float() + delta.float()).to(torch.bfloat16) if has_delta else delta.clone()
    blocks = blocks.clone()
    if write >= 0:
        blocks[:, write] = new
    if nb == 0:
        out = new.float() * rms(new.float(), out_eps) * out_w.float()
        return out.to(torch.bfloat16), new, blocks
    cw = (norm_w.float() * query_w.reshape(-1).float()).to(torch.bfloat16).float()
    rows = torch.cat([blocks[:, :nb].float(), new.float().unsqueeze(1)], dim=1)  # [T,R,H]
    score = (rows * cw).sum(-1) * rms(rows, eps).squeeze(-1)
    p = torch.softmax(score, dim=-1)
    mixed = (p.unsqueeze(-1) * rows).sum(1)
    out = mixed * rms(mixed, out_eps) * out_w.float()
    return out.to(torch.bfloat16), new, blocks


def attnres_aux_mixture(new_prefix, blocks, *, nb, norm_w, query_w, eps):
    """Pre-norm stream mixture with the FP32 combined weight (stock ``_mix_fused``)."""
    if nb == 0:
        return new_prefix.clone()
    cw = norm_w.float() * query_w.reshape(-1).float()
    rows = torch.cat([blocks[:, :nb].float(), new_prefix.float().unsqueeze(1)], dim=1)
    score = (rows * cw).sum(-1) * rms(rows, eps).squeeze(-1)
    p = torch.softmax(score, dim=-1)
    return (p.unsqueeze(-1) * rows).sum(1).to(torch.bfloat16)


# ----------------------------------------------------------------------------- KDA

HEADS, DIM = int(os.environ.get("K31_TEST_HEADS", "12")), 128  # 12: TP8, 6: TP16 per-rank shapes
D = HEADS * DIM  # 1536 per TP8 rank, 768 per TP16 rank
PITCH = 4 * D + 128 + HEADS + (-(4 * D + 128 + HEADS) % 16)  # packed KDA projection rows: TP8 6288, TP16 3216


def kda_gate_inputs(proj, f_b):
    """Split the packed [T, 6288] projection: qkv [T,3D], output gate [T,D], f_a [T,128],
    beta [T,12]; ``a = bf16(f_a @ f_b^T)`` like the stock BF16 F_B projection."""
    qkv = proj[:, : 3 * D]
    gate = proj[:, 3 * D : 4 * D]
    f_a = proj[:, 4 * D : 4 * D + 128]
    beta = proj[:, 4 * D + 128 : 4 * D + 128 + HEADS]
    a = (f_a.float() @ f_b.float().T).to(torch.bfloat16)
    return qkv, gate, a, beta


def kda_token(window, conv_w, a_row, beta_row, gate_row, state, *, a_log, dt_bias, norm_w, lower_bound, eps):
    """One token. window: [4, 3D] FP32 (3 taps oldest first, then x); conv_w [3D, 4] FP32;
    state [12, 128 (value), 128 (key)] FP32, updated in place. Returns (out [D] FP32,
    raw k [D], raw v [D], log decay [12, 128], beta [12])."""
    conv = (window * conv_w.T).sum(0)  # [3D]
    act = conv * torch.sigmoid(conv)
    q, k, v = (act[i * D : (i + 1) * D].view(HEADS, DIM) for i in range(3))
    qn = q * torch.rsqrt(q.pow(2).sum(-1, keepdim=True) + 1e-6) * DIM**-0.5
    kn = k * torch.rsqrt(k.pow(2).sum(-1, keepdim=True) + 1e-6)
    b = torch.sigmoid(beta_row.float())  # [12]
    g = lower_bound * torch.sigmoid(
        torch.exp(a_log.float()).view(HEADS, 1) * (a_row.float().view(HEADS, DIM) + dt_bias.float().view(HEADS, DIM))
    )  # [12, 128 keys]
    state.mul_(torch.exp(g).unsqueeze(1))
    pred = torch.einsum("hvk,hk->hv", state, kn)
    v_new = (v - pred) * b.unsqueeze(1)
    state.add_(v_new.unsqueeze(2) * kn.unsqueeze(1))
    o = torch.einsum("hvk,hk->hv", state, qn)
    o = o * rms(o, eps) * norm_w.float() * torch.sigmoid(gate_row.float().view(HEADS, DIM))
    return o.reshape(-1), k.reshape(-1), v.reshape(-1), g, b


def kda_decode(proj, conv, state, slots, *, conv_w, f_b, a_log, dt_bias, norm_w, lower_bound, eps):
    """Decode (one token per row) against pools conv [S, 3, 3D] BF16 and state
    [S, 12, 128, 128] FP32; rows with slot <= 0 are padding (output 0, nothing written).
    conv_w: [3D, 1, 4] FP32. Returns (out bf16 [T,D], new conv, new state)."""
    conv, state = conv.clone(), state.clone()
    qkv, gate, a, beta = kda_gate_inputs(proj, f_b)
    w = conv_w.reshape(3 * D, 4).float()
    out = torch.zeros(proj.shape[0], D, dtype=torch.bfloat16, device=proj.device)
    for t in range(proj.shape[0]):
        s = int(slots[t])
        if s <= 0:
            continue
        window = torch.cat([conv[s].float(), qkv[t].float().unsqueeze(0)], 0)
        st = state[s]
        o, *_ = kda_token(window, w, a[t], beta[t], gate[t], st, a_log=a_log, dt_bias=dt_bias, norm_w=norm_w, lower_bound=lower_bound, eps=eps)
        out[t] = o.to(torch.bfloat16)
        conv[s] = torch.cat([conv[s][1:], qkv[t].unsqueeze(0)], 0)
    return out, conv, state


def kda_verify(proj, conv, state, slots, scratch, q, *, conv_w, f_b, a_log, dt_bias, norm_w, lower_bound, eps, ring_len):
    """ReplaySSM verify: requests r run tokens r*q .. r*q+q-1 in order from checkpoint
    state[slots[r]] and conv taps conv[slots[r]] (negative slot or scratch row = padding:
    output 0). State and conv stay unchanged. Returns (out bf16 [R*q, D], rings
    {rawv, rawk [S,12,L,128] bf16, log_g [S,12,L,128] f32, beta [S,12,L] f32} entries per
    (slot, t), windows {scratch_row: [q, 3, 3D] bf16 post-token taps})."""
    qkv, gate, a, beta = kda_gate_inputs(proj, f_b)
    w = conv_w.reshape(3 * D, 4).float()
    rows = proj.shape[0]
    out = torch.zeros(rows, D, dtype=torch.bfloat16, device=proj.device)
    rings, windows = {}, {}
    for r in range(rows // q):
        s, row = int(slots[r]), int(scratch[r])
        if s < 0 or row < 0:
            continue
        st = state[s].clone()
        seq = torch.cat([conv[s].float(), qkv[r * q : (r + 1) * q].float()], 0)  # [3+q, 3D]
        win = torch.empty(q, 3, 3 * D, dtype=torch.bfloat16, device=proj.device)
        for t in range(q):
            i = r * q + t
            o, k, v, g, b = kda_token(seq[t : t + 4], w, a[i], beta[i], gate[i], st, a_log=a_log, dt_bias=dt_bias, norm_w=norm_w, lower_bound=lower_bound, eps=eps)
            out[i] = o.to(torch.bfloat16)
            rings[(s, t)] = (v.view(HEADS, DIM).to(torch.bfloat16), k.view(HEADS, DIM).to(torch.bfloat16), g, b)
            win[t] = seq[t + 1 : t + 4].to(torch.bfloat16)
        windows[row] = win
    return out, rings, windows

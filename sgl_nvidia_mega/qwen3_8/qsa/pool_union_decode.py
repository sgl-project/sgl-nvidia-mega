# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang (Apache-2.0, SGLang Team):
# python/sglang/srt/layers/attention/qsa/pool_union_decode.py, through an earlier commit
# of this plugin. Changed: no switch or launch knob is read at import, the caller owns
# the workspace and the split count, and the split chooser and engagement log are gone.
# The kernels are unchanged.
"""QSA target-verify sparse attention read from an FP8 (E4M3) KV pool, each request's
union of selected tokens once (not bitwise the stock chain).

Replaces fused_kv_prepare (write the new K/V, pack each row's <= 2051 selected tokens)
and trtllm decode over the packed rows with four launches chained by PDL:
    kernel 1  _qsa_write_new_kv (pool_decode.py) copies the new K/V into the pool, or
              _qsa_write_new_kv_e4m3 casts BF16 rows to E4M3 in the write;
    kernel U  _qsa_union_tokens grid (requests,): sorts the W rows' selected blocks,
              keeps one entry per block with a row bitmask, resolves each token's pool
              slot, appends the rows' uncompressed tails and counts each row's
              counted-but-invalid entries (score 0, V 0, as in the packed rows);
    kernel 2  _qsa_pool_union_splitk grid (requests, kv_head, splits): the request's
              W * GQ q heads against a split of its token list, each q row masked to its
              own row's bit; E4M3 -> BF16 is exact, no k/v scale (stock bmm2_scale 1.0);
    kernel 3  _qsa_pool_union_merge merges the split partials.
Per row the attended positions are pool_decode.py's (fused_kv_prepare EXPAND + CHAIN with
visible_page_size); only the FP32 accumulation order and tiling differ (P is rounded to
BF16 before PV). pool_union_ws.py replaces kernel 2 with a warp-specialized kernel.
"""

import torch
import triton
import triton.language as tl

from .pool_decode import LAST_KERNEL, LOG2E, _qsa_write_new_kv, write_new_kv_e4m3


@triton.jit
def _qsa_union_tokens(
    IDX,
    IDX_STRIDE,
    LENGTH,
    QPOS,
    LOC,
    MAP,
    MAP_STRIDE,
    REQ,
    TS,
    TM,
    NTOK,
    NINV,
    SCR,
    IB: tl.constexpr,
    W: tl.constexpr,
    RATIO: tl.constexpr,
    NB: tl.constexpr,
    NT: tl.constexpr,
    TB: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    r = tl.program_id(0)
    i = tl.arange(0, NB)
    j = i // IB
    c = i % IB
    in_rows = j < W
    row = r * W + j
    length = tl.load(LENGTH + row, mask=in_rows, other=0)
    active = tl.load(LOC + row, mask=in_rows, other=0) != 0
    take = in_rows & (c < tl.minimum(length // RATIO, IB))
    blk = tl.load(IDX + row.to(tl.int64) * IDX_STRIDE + c, mask=take, other=-1)
    use = take & active & (blk >= 0)
    # invalid tokens of the compressed part: a -1 entry, or a block's tokens at or past the length
    inv = tl.where(
        take,
        tl.where(
            blk < 0,
            RATIO,
            tl.minimum(RATIO, tl.maximum(0, blk * RATIO + RATIO - length)),
        ),
        0,
    )
    key = tl.where(use, blk * 8 + j, 1073741824)  # unused entries sort last
    key = tl.sort(key)
    base = SCR + r.to(tl.int64) * NB
    tl.store(base + i, key)
    tl.debug_barrier()
    kb = key // 8
    used = key != 1073741824
    prev = tl.load(base + i - 1, mask=i > 0, other=-8)
    head = used & (kb != prev // 8)
    one = tl.full([NB], 1, tl.int32)
    mask = tl.where(used, one << (key % 8), 0)
    for dlt in tl.static_range(1, W):
        nxt = tl.load(base + i + dlt, mask=i + dlt < NB, other=1073741824)
        same = (nxt != 1073741824) & (nxt // 8 == kb)
        mask = mask | tl.where(same, one << (nxt % 8), 0)
    slot = tl.cumsum(head.to(tl.int32), 0) - 1
    ucnt = tl.sum(head.to(tl.int32), 0)

    req = tl.load(REQ + r * W).to(tl.int64)
    map_base = MAP + req * MAP_STRIDE
    ts_base = TS + r.to(tl.int64) * NT
    tm_base = TM + r.to(tl.int64) * NT
    for t in tl.static_range(RATIO):
        pos = kb * RATIO + t
        below = tl.zeros([NB], tl.int32)
        for jj in tl.static_range(W):
            below = below | tl.where(pos < tl.load(LENGTH + r * W + jj), 1 << jj, 0)
        tok_slot = tl.load(map_base + pos, mask=head, other=0)
        tl.store(ts_base + slot * RATIO + t, tok_slot, mask=head)
        tl.store(tm_base + slot * RATIO + t, mask & below, mask=head)

    # uncompressed tails: row j attends tail_start_j .. min(qpos_j + 1, len_j) - 1 (at most
    # RATIO - 1 tokens); a chain's rows keep them within TB positions of the first tail start
    tbase = 1073741824
    for jj in tl.static_range(W):
        q_j = tl.load(QPOS + r * W + jj)
        act_j = tl.load(LOC + r * W + jj) != 0
        tbase = tl.minimum(
            tbase, tl.where(act_j, (q_j + 1) // RATIO * RATIO, 1073741824)
        )
    tp = tbase + tl.arange(0, TB)
    tmask = tl.zeros([TB], tl.int32)
    for jj in tl.static_range(W):
        q_j = tl.load(QPOS + r * W + jj)
        len_j = tl.load(LENGTH + r * W + jj)
        act_j = tl.load(LOC + r * W + jj) != 0
        ts_j = (q_j + 1) // RATIO * RATIO
        te_j = tl.minimum(tl.minimum(q_j + 1, len_j), ts_j + RATIO - 1)
        tmask = tmask | tl.where(act_j & (tp >= ts_j) & (tp < te_j), 1 << jj, 0)
    keep = tmask != 0
    tidx = tl.cumsum(keep.to(tl.int32), 0) - 1
    tslot = tl.load(map_base + tp, mask=keep, other=0)
    tl.store(ts_base + ucnt * RATIO + tidx, tslot, mask=keep)
    tl.store(tm_base + ucnt * RATIO + tidx, tmask, mask=keep)
    tl.store(NTOK + r, ucnt * RATIO + tl.sum(keep.to(tl.int32), 0))

    # per row: invalid compressed tokens, or every counted token of an inactive row (loc 0)
    jr = tl.arange(0, NB // IB)
    rrow = r * W + jr
    rin = jr < W
    rlen = tl.load(LENGTH + rrow, mask=rin, other=0)
    rq = tl.load(QPOS + rrow, mask=rin, other=0)
    ract = tl.load(LOC + rrow, mask=rin, other=0) != 0
    n = tl.minimum(rlen // RATIO, IB) * RATIO
    tail_start = (rq + 1) // RATIO * RATIO
    tail_count = tl.maximum(0, tl.minimum(rq + 1, rlen) - tail_start)
    inv_rows = tl.sum(tl.reshape(inv, (NB // IB, IB)), 1)
    tl.store(NINV + rrow, tl.where(ract, inv_rows, n + tail_count), mask=rin)


@triton.jit
def _qsa_pool_union_splitk(
    Q,
    K,
    V,
    TS,
    TM,
    NTOK,
    NINV,
    M_OUT,
    L_OUT,
    ACC_OUT,
    qk_scale,
    Q_STRIDE,
    H: tl.constexpr,
    D: tl.constexpr,
    GQ: tl.constexpr,
    QP: tl.constexpr,
    W: tl.constexpr,
    NT: tl.constexpr,
    SPLITS: tl.constexpr,
    BN: tl.constexpr,
    KV_FP8: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    r = tl.program_id(0)
    head = tl.program_id(1)
    split = tl.program_id(2)
    d = tl.arange(0, D)
    g = tl.arange(0, QP)
    jq = g // GQ
    qv = jq < W
    row = r * W + jq
    q = tl.load(
        Q
        + row.to(tl.int64)[:, None] * Q_STRIDE
        + (head * GQ + g % GQ)[:, None] * D
        + d[None, :],
        mask=qv[:, None],
        other=0.0,
    )
    ntok = tl.load(NTOK + r)
    chunk = tl.cdiv(tl.cdiv(ntok, SPLITS), BN) * BN
    start = split * chunk
    end = tl.minimum(start + chunk, ntok)
    ts_base = TS + r.to(tl.int64) * NT
    tm_base = TM + r.to(tl.int64) * NT

    m_i = tl.full([QP], float("-inf"), tl.float32)
    l_i = tl.zeros([QP], tl.float32)
    acc = tl.zeros([QP, D], tl.float32)
    for off in range(start, end, BN):
        t = off + tl.arange(0, BN)
        tin = t < end
        tm = tl.load(tm_base + t, mask=tin, other=0)
        # a listed token that no row attends is never read (no stale slot can reach PV)
        live = tin & (tm != 0)
        tok_slot = tl.load(ts_base + t, mask=live, other=0).to(tl.int64)
        kvoff = tok_slot[:, None] * (H * D) + head * D + d[None, :]
        k = tl.load(K + kvoff, mask=live[:, None], other=0.0)
        v = tl.load(V + kvoff, mask=live[:, None], other=0.0)
        if KV_FP8:
            k = k.to(Q.dtype.element_ty)  # E4M3 -> BF16 is exact
            v = v.to(Q.dtype.element_ty)
        s = tl.dot(q, tl.trans(k)) * qk_scale
        s = tl.where(((tm[None, :] >> jq[:, None]) & 1) != 0, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.math.exp2(m_i - m_safe)
        p = tl.math.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    if split == 0:
        # counted-but-invalid entries: score q . 0 = 0, V 0
        ninv = tl.load(NINV + row, mask=qv, other=0).to(tl.float32)
        has = ninv > 0
        m_new = tl.where(has, tl.maximum(m_i, 0.0), m_i)
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.math.exp2(m_i - m_safe)
        l_i = l_i * alpha + tl.where(has, ninv * tl.math.exp2(0.0 - m_safe), 0.0)
        acc = acc * alpha[:, None]
        m_i = m_new

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
    part = (r * H + head) * SPLITS + split
    tl.store(M_OUT + part * QP + g, m_i, mask=qv)
    tl.store(L_OUT + part * QP + g, l_i, mask=qv)
    live_q = qv & (m_i > float("-inf"))
    tl.store(
        ACC_OUT + part.to(tl.int64) * (QP * D) + g[:, None] * D + d[None, :],
        acc,
        mask=live_q[:, None],
    )


@triton.jit
def _qsa_pool_union_merge(
    M_IN,
    L_IN,
    ACC_IN,
    O,
    O_STRIDE,
    H: tl.constexpr,
    D: tl.constexpr,
    GQ: tl.constexpr,
    QP: tl.constexpr,
    W: tl.constexpr,
    SPLITS: tl.constexpr,
    SB: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    pid = tl.program_id(0)
    qh_all = H * GQ
    row = pid // qh_all
    qh = pid % qh_all
    head = qh // GQ
    g = (row % W) * GQ + qh % GQ
    base = (row // W * H + head) * SPLITS
    s_all = tl.arange(0, SB)
    d = tl.arange(0, D)
    mx = tl.full([SB], float("-inf"), tl.float32)
    for s0 in tl.static_range(0, SPLITS, SB):
        s = s0 + s_all
        m = tl.load(M_IN + (base + s) * QP + g, mask=s < SPLITS, other=float("-inf"))
        mx = tl.maximum(mx, m)
    M = tl.max(mx, 0)
    M = tl.where(M == float("-inf"), 0.0, M)
    L = tl.zeros([SB], tl.float32)
    acc = tl.zeros([D], tl.float32)
    for s0 in tl.static_range(0, SPLITS, SB):
        s = s0 + s_all
        m = tl.load(M_IN + (base + s) * QP + g, mask=s < SPLITS, other=float("-inf"))
        l = tl.load(L_IN + (base + s) * QP + g, mask=s < SPLITS, other=0.0)
        w = tl.math.exp2(m - M)
        L += w * l
        live = (s < SPLITS) & (m > float("-inf"))
        a = tl.load(
            ACC_IN + ((base + s).to(tl.int64) * QP + g)[:, None] * D + d[None, :],
            mask=live[:, None],
            other=0.0,
        )
        acc += tl.sum(a * w[:, None], 0)
    Ls = tl.sum(L, 0)
    out = tl.where(Ls > 0, acc / Ls, 0.0)
    tl.store(O + row.to(tl.int64) * O_STRIDE + qh * D + d, out.to(O.dtype.element_ty))


def union_buffers(rows, width, ib, ratio):
    """(nb, tb, nt): the sort width, the tail window and one request's token-list capacity."""
    nb = triton.next_power_of_2(width * ib)
    tb = max(16, triton.next_power_of_2(width + 2 * ratio))
    return nb, tb, nb * ratio + tb


def qsa_pool_union_decode(
    q,
    k_cache,
    v_cache,
    k_new,
    v_new,
    loc,
    req_to_token,
    row_requests,
    block_indices,
    sequence_lens,
    query_positions,
    width,
    scale,
    ratio=4,
    out=None,
    workspace=None,
    splits=None,
    waves=None,
    bn=128,
    num_warps=8,
    num_stages=2,
    pdl=True,
    write_new_kv=True,
    union_warps=8,
):
    """Target-verify sparse attention read from the pool, each request's union of tokens once.

    Rows are requests' groups of ``width`` consecutive rows with chain positions, as a topk-1
    verify lays them out (row_requests constant within a group). q [rows, QH, 256] BF16 (or
    [rows, QH * 256]); k/v_cache [slots, H, 256] BF16 or FP8 (E4M3); an FP8 pool takes
    k/v_new in its dtype (kernel 1 copies their bytes) or BF16 (cast in the write);
    write_new_kv=False when the new K/V were stored before this call. ``union_warps`` is kernel U's warp count (its output
    does not depend on it). Returns [rows, QH * 256] in q's dtype.
    """
    rows, ib = block_indices.shape
    kv_heads, dim = k_cache.shape[1], k_cache.shape[2]
    q3 = q.view(rows, -1, dim)
    qh = q3.shape[1]
    gq = qh // kv_heads
    assert rows % width == 0
    assert q3.stride(2) == 1 and q3.stride(1) == dim
    assert k_new.stride(-1) == 1 and k_new.shape[-1] * k_new.shape[-2] == kv_heads * dim
    kv_fp8 = k_cache.dtype == torch.float8_e4m3fn
    if kv_fp8:
        assert k_new.dtype == v_new.dtype and k_cache.dtype == v_cache.dtype
        assert k_new.dtype in (k_cache.dtype, torch.bfloat16)
        assert q.dtype == torch.bfloat16
    if out is None or workspace is None:
        raise ValueError("union launch requires prepared workspace and caller output")
    if rows == 0:
        return out
    requests = rows // width
    qp = max(16, triton.next_power_of_2(width * gq))
    nb, tb, nt = union_buffers(rows, width, ib, ratio)
    if splits is None:
        raise ValueError("union requires caller-resolved split count")
    tok_slots, tok_masks, scratch, ntok, ninv, m_buf, l_buf, acc_buf = workspace
    pdl_kw = {"launch_pdl": True} if pdl else {}
    if write_new_kv:
        nk = k_new.reshape(rows, kv_heads * dim) if k_new.dim() == 3 else k_new
        nv = v_new.reshape(rows, kv_heads * dim) if v_new.dim() == 3 else v_new
        if nk.dtype != k_cache.dtype:  # BF16 rows, cast to E4M3 in the write
            write_new_kv_e4m3(
                k_cache, v_cache, nk, nv, loc, rows, kv_heads, dim, pdl, pdl_kw
            )
        else:
            _qsa_write_new_kv[(rows, kv_heads)](
                k_cache,
                v_cache,
                nk,
                nv,
                loc,
                nk.stride(0),
                nv.stride(0),
                kv_heads,
                dim,
                pdl,
                num_warps=1,
                **pdl_kw,
            )
    _qsa_union_tokens[(requests,)](
        block_indices,
        block_indices.stride(0),
        sequence_lens,
        query_positions,
        loc,
        req_to_token,
        req_to_token.stride(0),
        row_requests,
        tok_slots,
        tok_masks,
        ntok,
        ninv,
        scratch,
        ib,
        width,
        ratio,
        nb,
        nt,
        tb,
        pdl,
        num_warps=union_warps,
        **pdl_kw,
    )
    LAST_KERNEL["union"] = _qsa_pool_union_splitk[(requests, kv_heads, splits)](
        q3,
        k_cache,
        v_cache,
        tok_slots,
        tok_masks,
        ntok,
        ninv,
        m_buf,
        l_buf,
        acc_buf,
        float(scale) * LOG2E,
        q3.stride(0),
        kv_heads,
        dim,
        gq,
        qp,
        width,
        nt,
        splits,
        bn,
        kv_fp8,
        pdl,
        num_warps=num_warps,
        num_stages=num_stages,
        **pdl_kw,
    )
    sb = min(16, triton.next_power_of_2(splits))
    _qsa_pool_union_merge[(rows * qh,)](
        m_buf,
        l_buf,
        acc_buf,
        out,
        out.stride(0),
        kv_heads,
        dim,
        gq,
        qp,
        width,
        splits,
        sb,
        pdl,
        num_warps=2,
        **pdl_kw,
    )
    return out

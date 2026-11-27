"""Read-split context parallelism ("CP-lite") for TRT-LLM MHA decode-kernel attention.

When the TP size exceeds the KV head count, each KV head is replicated on the ``cp``
ranks of its KV group (``cp = tp // kv_heads``) and every rank reads the whole KV of its
head. CP-lite keeps the KV cache as it is (every rank still writes every token) and
splits the read: page ``p`` of a request's context prefix is read by group rank
``p % cp`` only, for every query head of the group. Per call:

1. ``_cp_gather_q``: all-gather of the group's FP8 queries through peer stores.
2. TRT-LLM GenMHA (FlashInfer) over the owned prefix pages, the ``q_len`` rows folded
   with the group heads into one query row (no causal effect on the prefix), with LSE.
3. ``_cp_a2a_merge``: each rank pushes every owner its heads' partials, then merges its
   own heads' partials with the causal ``q_len x q_len`` block of this step's rows.

The exchange buffer is one symmetric-memory allocation on the attention TP group; each
CTA signals its peers with self-resetting CAS flags, so calls stay graph capturable.

For multi-request calls, the stock and CP paths are captured together. The batch's KV
lengths select the path on the device from fixed, measured thresholds. On the exact path,
the stock kernel runs beside the gather and partial kernels on a second stream.
"""

from __future__ import annotations

import logging
import weakref

import torch
import triton
import triton.language as tl

from sgl_nvidia_mega.qwen3_5 import CP_LITE_HEAD_DIM, cp_lite_topology

logger = logging.getLogger(__name__)

PAGE, HEAD_DIM = 64, CP_LITE_HEAD_DIM
_MAX_ROWS = 896  # widest call: rows = requests * q_len (TP8 verify at B128: 896)
_PRECISE_MAX_BATCH = 64
_KINDS = 2  # flag regions: q gather, partial exchange
# exact path's fused partial: E4M3 P with shift 8 against the exact running max (lazy rescale threshold 0), so the
# row max maps to 2^8, exact in E4M3
_EXACT_P_SHIFT, _EXACT_RESCALE_THRESHOLD = 8.0, 0.0
_LL_PWORDS = 520  # LL words of one (row, source) partial: 4 heads x 128 bf16 pairs + 4 LSE, padded to 16 B
_PUSHED_MERGE_SLEEP_NS = 64


def _lam_q_layout(base: int, cp: int, q_heads: int) -> tuple[int, int]:
    off = (base + 15) // 16 * 16
    return off, off + _MAX_ROWS * cp * (q_heads * HEAD_DIM // 4) * 4


_LSE_STATS_Q_SLOTS, _LSE_STATS_GUARD = 256, 1 << 20


def _lse_workspace_bytes(rows: int, heads: int, sms: int) -> int:
    stats = 8 * rows * heads * _LSE_STATS_Q_SLOTS + _LSE_STATS_GUARD
    return stats + sms * 128 * (256 * 2 * 2 + 8)


@triton.jit
def _flat_tid():
    return tl.inline_asm_elementwise(
        "mov.u32 $0, %tid.x;", "=r", [], dtype=tl.uint32, is_pure=True, pack=1
    )


@triton.jit
def _sync_threads():
    tl.inline_asm_elementwise(
        "bar.sync 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
    )


@triton.jit
def _send_release(addrs):
    # set the peer's flag 0 -> 1 once its previous signal is consumed
    tl.inline_asm_elementwise(
        """
        {
            .reg .u32 %t;
            .reg .pred %p;
            send:
                atom.global.release.sys.cas.b32 %t, [$1], 0, 1;
                setp.eq.u32 %p, %t, 0;
                @!%p bra send;
        }
        """,
        "=r, l",
        [addrs],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _wait_acquire(addrs):
    # consume this rank's flag 1 -> 0
    tl.inline_asm_elementwise(
        """
        {
            .reg .u32 %t;
            .reg .pred %p;
            wait:
                atom.global.sys.acquire.cas.b32 %t, [$1], 1, 0;
                setp.eq.u32 %p, %t, 1;
                @!%p bra wait;
        }
        """,
        "=r, l",
        [addrs],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _send_release_dl(addrs, t_end):
    tl.inline_asm_elementwise(
        """
        {
            .reg .u32 %t, %n, %ev;
            .reg .pred %p;
            .reg .b64 %g, %z;
            mov.u32 %n, 0;
            send:
                atom.global.release.sys.cas.b32 %t, [$1], 0, 1;
                setp.eq.u32 %p, %t, 0;
                @%p bra sent;
                add.u32 %n, %n, 1;
                and.b32 %ev, %n, 255;
                setp.ne.u32 %p, %ev, 0;
                @%p bra send;
                mov.u64 %g, %globaltimer;
                setp.le.u64 %p, %g, $2;
                @%p bra send;
                mov.u64 %z, 0;
                st.volatile.global.u32 [%z], 0;
                trap;
            sent:
            mov.u32 $0, 0;
        }
        """,
        "=r, l, l",
        [addrs, (addrs.to(tl.int64) & 0) + t_end],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _wait_acquire_dl(addrs, t_end):
    # _wait_acquire bounded by t_end, as _send_release_dl
    tl.inline_asm_elementwise(
        """
        {
            .reg .u32 %t, %n, %ev;
            .reg .pred %p;
            .reg .b64 %g, %z;
            mov.u32 %n, 0;
            wait:
                atom.global.sys.acquire.cas.b32 %t, [$1], 1, 0;
                setp.eq.u32 %p, %t, 1;
                @%p bra got;
                add.u32 %n, %n, 1;
                and.b32 %ev, %n, 255;
                setp.ne.u32 %p, %ev, 0;
                @%p bra wait;
                mov.u64 %g, %globaltimer;
                setp.le.u64 %p, %g, $2;
                @%p bra wait;
                mov.u64 %z, 0;
                st.volatile.global.u32 [%z], 0;
                trap;
            got:
            mov.u32 $0, 0;
        }
        """,
        "=r, l, l",
        [addrs, (addrs.to(tl.int64) & 0) + t_end],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _group_barrier(
    peers, me, flags_off, slot, CP: tl.constexpr, t_end=0, DL: tl.constexpr = False
):
    c = tl.arange(0, CP)
    peer = tl.load(peers + c)
    own = tl.load(peers + me)
    send = (peer + flags_off + (slot * CP + me) * 4).to(tl.pointer_type(tl.uint32))
    wait = (own + flags_off + (slot * CP + c) * 4).to(tl.pointer_type(tl.uint32))
    if _flat_tid() < CP:
        if DL:
            _send_release_dl(send, t_end)
            _wait_acquire_dl(wait, t_end)
        else:
            _send_release(send)
            _wait_acquire(wait)


@triton.jit
def _gate_on(
    seq_lens, nb, thr, NB: tl.constexpr, GMAX: tl.constexpr, CG: tl.constexpr = False
):
    """CP for this call: the sum (GMAX: the max) of the batch's KV lengths reaches thr. Every rank of the group
    reads the same lengths, so all of them take the same branch."""
    k = tl.arange(0, NB)
    L = tl.load(seq_lens + k, mask=k < nb, other=0, cache_modifier=".cg" if CG else "")
    if GMAX:
        return tl.max(L, axis=0) >= thr
    else:
        return tl.sum(L, axis=0) >= thr


@triton.jit
def _gate_lens(
    seq_lens,
    nb,
    thr,
    stock_lens,
    pre_lens,
    QL: tl.constexpr,
    NB: tl.constexpr,
    GMAX: tl.constexpr,
    ZERO: tl.constexpr,
):
    """Once per forward: the stock call's lengths (L with the gate off, q_len keys with it on) and the exact
    path's prefix lengths (max(L - QL, 0) on, 0 off)."""
    k = tl.arange(0, NB)
    m = k < nb
    L = tl.load(seq_lens + k, mask=m, other=0)
    on = _gate_on(seq_lens, nb, thr, NB, GMAX)
    on_keys = 0 if ZERO else tl.minimum(L, QL)
    tl.store(stock_lens + k, tl.where(on, on_keys, L), mask=m)
    tl.store(pre_lens + k, tl.where(on, tl.maximum(L - QL, 0), 0), mask=m)


@triton.jit
def _cp_tables(
    page_table,
    pt_stride,
    seq_lens,
    tab,
    tab_stride,
    lens,
    ncols,
    rank,
    nb,
    thr,
    QL: tl.constexpr,
    CP: tl.constexpr,
    PG: tl.constexpr,
    BLOCK: tl.constexpr,
    GATE: tl.constexpr,
    NB: tl.constexpr,
    GMAX: tl.constexpr,
):
    """This rank's compact page table and token count over the prefix [0, L - QL); with GATE and the gate
    off, no owned page and no key."""
    b = tl.program_id(0)
    cb = tl.program_id(1)
    L = tl.load(seq_lens + b)
    prefix = tl.maximum(L - QL, 0)
    n_pages = (prefix + PG - 1) // PG
    n_own = tl.maximum((n_pages - rank + CP - 1) // CP, 0)
    on = 1
    if GATE:
        on = _gate_on(seq_lens, nb, thr, NB, GMAX).to(tl.int32)
        n_own = n_own * on
    j = cb * BLOCK + tl.arange(0, BLOCK)
    m = j < ncols
    pg = tl.load(
        page_table + b * pt_stride + j * CP + rank, mask=m & (j < n_own), other=0
    )
    tl.store(tab + b * tab_stride + j, pg, mask=m)
    if cb == 0:
        last_tok = prefix - (n_pages - 1) * PG
        tail = (n_own > 0) & ((n_own - 1) * CP + rank == n_pages - 1)
        n = n_own * PG - tl.where(tail, PG - last_tok, 0)
        # no owned page: one dummy key, dropped by the merge; gate off: no key (the merge exits)
        tl.store(lens + b, tl.maximum(n, on))


@triton.jit
def _cp_chunk_tables(
    tab,
    tstride,
    lens,
    base_lens,
    vtab,
    vlens,
    vstart,
    vcnt,
    nb,
    nv,
    ragged,
    C: tl.constexpr,
    KP: tl.constexpr,
    NB: tl.constexpr,
    PG: tl.constexpr,
):
    """Once per forward, program (r, j), chunked fold partial: with a ragged batch (longest owned length
    x B >= ragged x the total, and the longest over C pages), chunk j of request r's owned pages into dense row
    start_r + j of vtab ([nv, C]) and no key for the one-row-per-request call; otherwise every chunk row empty and that
    call keeps its lengths. Row r * KP + j is cleared when it lies past the last chunk."""
    r = tl.program_id(0)
    j = tl.program_id(1)
    k = tl.arange(0, NB)
    L = tl.load(lens + k, mask=k < nb, other=0)
    pg = (L + PG - 1) // PG
    mp = tl.max(pg, axis=0)
    # Chunk rows should not merely repeat a one-row call.
    on = ((mp * nb).to(tl.float32) >= ragged * tl.sum(pg, axis=0).to(tl.float32)) & (
        mp > C
    )
    n = tl.where(on, (pg + C - 1) // C, 0)
    start = tl.cumsum(n, axis=0) - n
    total = tl.sum(n, axis=0)
    my_n = tl.sum(tl.where(k == r, n, 0), axis=0)
    my_start = tl.sum(tl.where(k == r, start, 0), axis=0)
    my_pg = tl.sum(tl.where(k == r, pg, 0), axis=0)
    my_L = tl.sum(tl.where(k == r, L, 0), axis=0)
    z = r * KP + j
    if (z < nv) & (z >= total):
        tl.store(vlens + z, 0)
    if j == 0:
        tl.store(vstart + r, my_start)
        tl.store(vcnt + r, my_n)
        tl.store(base_lens + r, tl.where(on, 0, my_L))
    if j < my_n:
        v = my_start + j
        cols = tl.arange(0, C)
        src = j * C + cols
        m = src < my_pg
        tl.store(
            vtab + v * C + cols,
            tl.load(tab + r * tstride + src, mask=m, other=0),
            mask=m,
        )
        tl.store(vlens + v, tl.minimum(tl.maximum(my_L - j * (C * PG), 0), C * PG))


@triton.jit
def _cp_chunk_q(q, qv, vstart, vcnt, N, BLK: tl.constexpr):
    """Request r's query into its chunk rows (program (r, j): row vstart_r + j when j < vcnt_r)."""
    r = tl.program_id(0)
    j = tl.program_id(1)
    if j < tl.load(vcnt + r):
        v = tl.load(vstart + r) + j
        for c in range(tl.cdiv(N, BLK)):
            w = c * BLK + tl.arange(0, BLK)
            m = w < N
            tl.store(
                qv + v.to(tl.int64) * N + w,
                tl.load(q + r.to(tl.int64) * N + w, mask=m),
                mask=m,
            )


@triton.jit
def _cp_chunk_comb(
    po_v, pl_v, vstart, vcnt, po, pl, H, HB: tl.constexpr, D: tl.constexpr
):
    """One partial per request from its chunk rows (trtllm-gen LSE is base 2); no write for a request without chunk
    rows (the one-row-per-request call wrote it)."""
    r = tl.program_id(0)
    n = tl.load(vcnt + r)
    if n > 0:
        h = tl.program_id(1) * HB + tl.arange(0, HB)
        hm = h < H
        d = tl.arange(0, D)
        m = tl.full([HB], float("-inf"), tl.float32)
        s = tl.zeros([HB], tl.float32)
        acc = tl.zeros([HB, D], tl.float32)
        v0 = tl.load(vstart + r)
        for j in range(n):
            v = v0 + j
            lj = tl.load(pl_v + v * H + h, mask=hm, other=0.0)
            oj = tl.load(
                po_v + (v * H + h[:, None]).to(tl.int64) * D + d[None, :],
                mask=hm[:, None],
                other=0.0,
            )
            mn = tl.maximum(m, lj)
            a = tl.exp2(m - mn)
            w = tl.exp2(lj - mn)
            acc = acc * a[:, None] + oj.to(tl.float32) * w[:, None]
            s = s * a + w
            m = mn
        ok = s > 0
        tl.store(
            po + (r * H + h[:, None]).to(tl.int64) * D + d[None, :],
            (acc / tl.where(ok, s, 1.0)[:, None]).to(tl.bfloat16),
            mask=hm[:, None],
        )
        tl.store(
            pl + r * H + h,
            tl.where(ok, m + tl.log2(tl.where(ok, s, 1.0)), float("-inf")),
            mask=hm,
        )


@triton.jit
def _cp_gather_q(
    q,
    q_stride,
    peers,
    me,
    q_off,
    flags_off,
    CP: tl.constexpr,
    WORDS: tl.constexpr,
    tmo=0,
    DEADLINE: tl.constexpr = False,
):
    i = tl.program_id(0)
    w = tl.arange(0, WORDS)
    x = tl.load(q + i * q_stride + w)
    for c in tl.static_range(CP):
        base = tl.load(peers + c) + q_off + (i * CP + me) * WORDS * 4
        tl.store(base.to(tl.pointer_type(tl.int32)) + w, x)
    _sync_threads()
    if DEADLINE:
        _group_barrier(peers, me, flags_off, i, CP, _t_end(tmo), True)
    else:
        _group_barrier(peers, me, flags_off, i, CP)


@triton.jit
def _cp_a2a_merge(
    po,
    plse,
    q,
    q_stride,
    k_cache,
    v_cache,
    kv_page_stride,
    kv_tok_stride,
    table,
    t_stride,
    seq_lens,
    out,
    out_stride,
    peers,
    me,
    p_off,
    l_off,
    flags_off,
    slot0,
    bmm1,
    bmm2,
    QL: tl.constexpr,
    HQ: tl.constexpr,
    CP: tl.constexpr,
    D: tl.constexpr,
    PG: tl.constexpr,
    tmo=0,
    DEADLINE: tl.constexpr = False,
):
    i = tl.program_id(0)
    h = tl.arange(0, HQ)
    d = tl.arange(0, D)
    hd = h[:, None] * D + d[None, :]
    # push owner c its heads' partial O (bf16) and LSE (fp32, base 2)
    for c in tl.static_range(CP):
        x = tl.load(po + (i * CP * HQ + c * HQ) * D + hd)
        lse = tl.load(plse + i * CP * HQ + c * HQ + h)
        base = tl.load(peers + c)
        dst = (base + p_off).to(tl.pointer_type(tl.bfloat16)) + (i * CP + me) * HQ * D
        tl.store(dst + hd, x)
        ldst = (base + l_off).to(tl.pointer_type(tl.float32)) + (i * CP + me) * HQ
        tl.store(ldst + h, lse)
    _sync_threads()
    if DEADLINE:
        _group_barrier(peers, me, flags_off, slot0 + i, CP, _t_end(tmo), True)
    else:
        _group_barrier(peers, me, flags_off, slot0 + i, CP)
    _sync_threads()
    own = tl.load(peers + me)
    own_p = (own + p_off).to(tl.pointer_type(tl.bfloat16)) + i * CP * HQ * D
    own_l = (own + l_off).to(tl.pointer_type(tl.float32)) + i * CP * HQ
    b = i // QL
    t = i % QL
    L = tl.load(seq_lens + b)
    # the causal block of this step's rows, for this rank's heads
    j = tl.arange(0, 8)
    pos = L - QL + j
    valid = (j <= t) & (pos >= 0)
    pos = tl.maximum(pos, 0)
    page = tl.load(table + b * t_stride + pos // PG, mask=valid, other=0)
    koff = page.to(tl.int64) * kv_page_stride + (pos % PG) * kv_tok_stride
    kk = tl.load(
        k_cache + koff[:, None] + d[None, :], mask=valid[:, None], other=0.0
    ).to(tl.float32)
    vv = tl.load(
        v_cache + koff[:, None] + d[None, :], mask=valid[:, None], other=0.0
    ).to(tl.float32)
    qv = tl.load(q + i * q_stride + hd).to(tl.float32)
    s = tl.sum(qv[:, None, :] * kk[None, :, :], axis=2) * bmm1
    s = tl.where(valid[None, :], s, float("-inf"))
    md = tl.max(s, axis=1)
    md0 = tl.where(md > float("-inf"), md, 0.0)
    p = tl.exp(s - md0[:, None])
    ld = tl.sum(p, axis=1)
    od = (
        tl.sum(p[:, :, None] * vv[None, :, :], axis=1)
        / tl.maximum(ld, 1e-30)[:, None]
        * bmm2
    )
    lse_d = tl.where(ld > 0, md0 + tl.log(ld), float("-inf"))
    # rank c owns a prefix page iff the prefix has more than c pages
    n_pages = (tl.maximum(L - QL, 0) + PG - 1) // PG
    m = lse_d
    for c in tl.static_range(CP):
        lc = tl.load(own_l + c * HQ + h, cache_modifier=".cg") * 0.6931471805599453
        lc = tl.where(n_pages > c, lc, float("-inf"))
        m = tl.maximum(m, lc)
    m0 = tl.where(m > float("-inf"), m, 0.0)
    w = tl.where(lse_d > float("-inf"), tl.exp(lse_d - m0), 0.0)
    acc = od * w[:, None]
    for c in tl.static_range(CP):
        lc = tl.load(own_l + c * HQ + h, cache_modifier=".cg") * 0.6931471805599453
        wc = tl.where(n_pages > c, tl.exp(lc - m0), 0.0)
        oc = tl.load(own_p + c * HQ * D + hd, cache_modifier=".cg").to(tl.float32)
        acc += oc * wc[:, None]
        w += wc
    o = acc / tl.where(w > 0, w, 1.0)[:, None]
    tl.store(out + i * out_stride + hd, o.to(tl.bfloat16))


@triton.jit
def _ll_pack(flag, w):
    return flag | (w.to(tl.int64) & 0xFFFFFFFF)


@triton.jit
def _st_relaxed_sys_b32(addr, val):
    tl.inline_asm_elementwise(
        "st.relaxed.sys.global.b32 [$1], $2; mov.u32 $0, 0;",
        "=r,l,r",
        [addr, val],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _st_relaxed_sys_b64(addr, val):
    tl.inline_asm_elementwise(
        "st.relaxed.sys.global.b64 [$1], $2; mov.u32 $0, 0;",
        "=r,l,l",
        [addr, val],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _lam_rearm(addr):
    _st_relaxed_sys_b32(addr, (addr & 0).to(tl.int32) - 1)


@triton.jit
def _ll_spin(addr, e):
    # one thread per word: volatile reads until the word's high half is e
    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p;
            .reg .b32 %lo, %hi;
            spin:
            ld.volatile.global.b64 $0, [$1];
            mov.b64 {%lo, %hi}, $0;
            setp.ne.b32 %p, %hi, $2;
            @%p bra spin;
        }
        """,
        "=l,l,r",
        args=[addr, e],
        dtype=tl.int64,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _gtimer(dep):
    return tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer; // $1",
        "=l,r",
        [dep],
        dtype=tl.int64,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _t_end(tmo):
    return _gtimer(tl.program_id(0)) + tmo.to(tl.int64)


@triton.jit
def _fail_late(t_end, dep):
    tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p;
            .reg .b64 %t, %z;
            mov.u64 %t, %globaltimer;
            setp.gt.u64 %p, %t, $1;
            mov.u64 %z, 0;
            @%p st.volatile.global.u32 [%z], 0;
            @%p trap;
            mov.u32 $0, 0;
        }
        """,
        "=r,l",
        [(dep & 0).to(tl.int64) + t_end],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _ll_spin_dl(addr, e, t_end):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p;
            .reg .b32 %lo, %hi, %n, %ev;
            .reg .b64 %t, %z;
            mov.b32 %n, 0;
            spin:
            ld.volatile.global.b64 $0, [$1];
            mov.b64 {%lo, %hi}, $0;
            setp.eq.b32 %p, %hi, $2;
            @%p bra done;
            add.u32 %n, %n, 1;
            and.b32 %ev, %n, 255;
            setp.ne.u32 %p, %ev, 0;
            @%p bra spin;
            mov.u64 %t, %globaltimer;
            setp.le.u64 %p, %t, $3;
            @%p bra spin;
            mov.u64 %z, 0;
            st.volatile.global.u32 [%z], 0;
            trap;
            done:
        }
        """,
        "=l,l,r,l",
        args=[addr, e, t_end],
        dtype=tl.int64,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _ll_wait(src, n: tl.constexpr, e, t_end=0, DL: tl.constexpr = False):
    # each thread waits for the words it returns: with a block-wide count, a warp could leave holding a stale copy of
    # a word another warp also held (polls narrower than the CTA); the barrier keeps every warp before the epoch store
    if DL:
        v = _ll_spin_dl(src + tl.arange(0, n), e, t_end)
    else:
        v = _ll_spin(src + tl.arange(0, n), e)
    if n > 32:
        tl.debug_barrier()
    return v


@triton.jit
def _ll_wait_plain(src, n: tl.constexpr, e, t_end=0, DL: tl.constexpr = False):
    k = tl.arange(0, n)
    v = tl.load(src + k, volatile=True)
    if DL:
        ok = tl.min(((v >> 32) == e).to(tl.int32), axis=0)
        while ok == 0:
            _fail_late(t_end, ok)
            v = tl.load(src + k, volatile=True)
            ok = tl.min(((v >> 32) == e).to(tl.int32), axis=0)
    else:
        while tl.min(((v >> 32) == e).to(tl.int32), axis=0) == 0:
            v = tl.load(src + k, volatile=True)
    return v


@triton.jit
def _ll_wait_plain_ee(addr, ee, src_l, e, t_end=0, DL: tl.constexpr = False):
    k1 = tl.arange(0, 1)
    v = tl.load(addr, volatile=True)
    lv = tl.load(src_l + k1, volatile=True)
    ok = tl.min(
        tl.min(((v >> 32) == ee).to(tl.int32), axis=1) * ((lv >> 32) == e).to(tl.int32),
        axis=0,
    )
    while ok == 0:
        if DL:
            _fail_late(t_end, ok)
        v = tl.load(addr, volatile=True)
        lv = tl.load(src_l + k1, volatile=True)
        ok = tl.min(
            tl.min(((v >> 32) == ee).to(tl.int32), axis=1)
            * ((lv >> 32) == e).to(tl.int32),
            axis=0,
        )
    return v, lv


@triton.jit
def _nsleep(dep, NS: tl.constexpr):
    # nanosleep, tied to ``dep`` so the compiler keeps it in the loop
    tl.inline_asm_elementwise(
        "nanosleep.u32 $1; mov.u32 $0, 0;",
        "=r,r",
        [(dep & 0).to(tl.int32) + NS],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _ll_wait_all(
    daddr, laddr, e, SLEEP: tl.constexpr, t_end=0, DL: tl.constexpr = False
):
    v = tl.load(daddr, volatile=True)
    lv = tl.load(laddr, volatile=True)
    ok = tl.min(
        tl.min(((v >> 32) == e).to(tl.int32), axis=1) * ((lv >> 32) == e).to(tl.int32),
        axis=0,
    )
    while ok == 0:
        if DL:
            _fail_late(t_end, ok)
        if SLEEP > 0:
            _nsleep(lv, SLEEP)
        v = tl.load(daddr, volatile=True)
        lv = tl.load(laddr, volatile=True)
        ok = tl.min(
            tl.min(((v >> 32) == e).to(tl.int32), axis=1)
            * ((lv >> 32) == e).to(tl.int32),
            axis=0,
        )
    return v, lv


@triton.jit
def _cp_gather_q_ll(
    q,
    q_stride,
    peers,
    me,
    ll_off,
    q_all,
    cnt,
    seq_lens,
    pre_lens,
    stock_lens,
    nb,
    thr,
    CP: tl.constexpr,
    WORDS: tl.constexpr,
    QL: tl.constexpr,
    FOLD: tl.constexpr,
    GATE: tl.constexpr,
    NB: tl.constexpr,
    GMAX: tl.constexpr,
    GWRITE: tl.constexpr,
    TRIG: tl.constexpr = False,
    WAIT: tl.constexpr = False,
    qv=None,
    vstart=None,
    vcnt=None,
    CHUNK: tl.constexpr = False,
    tmo=0,
    DEADLINE: tl.constexpr = False,
):
    """LL variant of _cp_gather_q: every 4-byte word travels with the call's epoch in one 8-byte store.
    FOLD: q_all is written source-major per request ([B, CP, QL, heads, D]) for the prefix-only fused partial,
    and pre_lens[b] = max(L - QL, 0).
    GATE: exits with the gate off; GWRITE: first writes the stock call's lengths (L with the gate off, q_len with it
    on) and, FOLD, pre_lens = 0 with the gate off (else _gate_lens wrote them).
    TRIG (no GWRITE): releases the dependent grid at entry; the fused partial reads only lengths, tables and prefix KV
    before its griddepcontrol.wait, none of them written here.
    WAIT: launched as a programmatic dependent of q's writer, which writes none of what the partial reads before its
    wait either; every access here follows the wait."""
    if TRIG:
        tl.extra.cuda.gdc_launch_dependents()
    if WAIT:
        _gdc_wait_clobber()
    i = tl.program_id(0)
    if GATE:
        on = _gate_on(seq_lens, nb, thr, NB, GMAX)
        if GWRITE and i % QL == 0:
            Lb = tl.load(seq_lens + i // QL)
            tl.store(stock_lens + i // QL, tl.where(on, tl.minimum(Lb, QL), Lb))
            if FOLD:
                tl.store(pre_lens + i // QL, tl.where(on, tl.maximum(Lb - QL, 0), 0))
        if on.to(tl.int32) == 0:
            return
    e = tl.load(cnt + i) + 1
    flag = e.to(tl.int64) << 32
    w = tl.arange(0, WORDS)
    x = _ll_pack(flag, tl.load(q + i * q_stride + w))
    for c in tl.static_range(CP):
        dst = tl.load(peers + c) + ll_off + (i * CP + me) * WORDS * 8
        _st_relaxed_sys_b64(dst + w.to(tl.int64) * 8, x)
    own = tl.load(peers + me) + ll_off
    if DEADLINE:
        t_end = _t_end(tmo)
    else:
        t_end = 0
    for c in tl.static_range(CP):
        v = _ll_wait(
            (own + (i * CP + c) * WORDS * 8).to(tl.pointer_type(tl.int64)),
            WORDS,
            e,
            t_end,
            DEADLINE,
        )
        if FOLD:
            slot = ((i // QL) * CP + c) * QL + i % QL
        else:
            slot = i * CP + c
        val = (v & 0xFFFFFFFF).to(tl.int32)
        tl.store(q_all + slot * WORDS + w, val)
        if CHUNK:
            v0 = tl.load(vstart + i // QL)
            for j in range(tl.load(vcnt + i // QL)):
                tl.store(
                    qv
                    + ((v0 + j) * (QL * CP) + (i % QL) * CP + c).to(tl.int64) * WORDS
                    + w,
                    val,
                )
    if FOLD:  # noqa: SIM102
        if not GATE:  # noqa: SIM102
            if i % QL == 0:
                tl.store(
                    pre_lens + i // QL,
                    tl.maximum(tl.load(seq_lens + i // QL) - QL, 0),
                )
    tl.store(cnt + i, e)


@triton.jit
def _cp_gather_q_wait(
    peers,
    me,
    ll_off,
    q_all,
    cnt,
    seq_lens,
    pre_lens,
    CP: tl.constexpr,
    WORDS: tl.constexpr,
    QL: tl.constexpr,
    TRIG: tl.constexpr = True,
    LAM: tl.constexpr = False,
    tmo=0,
    DEADLINE: tl.constexpr = False,
):
    if TRIG:
        tl.extra.cuda.gdc_launch_dependents()
    _gdc_wait_clobber()
    i = tl.program_id(0)
    e = tl.load(cnt + i) + 1
    w = tl.arange(0, WORDS)
    cc = tl.arange(0, CP)
    own = tl.load(peers + me) + ll_off
    if DEADLINE:
        t_end = _t_end(tmo)
    else:
        t_end = 0
    if LAM:
        laddr = (own + (i * CP + cc) * WORDS * 4)[:, None] + w[None, :].to(tl.int64) * 4
        src = laddr.to(tl.pointer_type(tl.int32))
    else:
        src = (own + (i * CP + cc) * WORDS * 8).to(tl.pointer_type(tl.int64))
        src = src[:, None] + w[None, :]
    v = tl.load(src, volatile=True)
    if LAM:
        done = tl.min(tl.min((v != -1).to(tl.int32), axis=1), axis=0)
    else:
        done = tl.min(tl.min(((v >> 32) == e).to(tl.int32), axis=1), axis=0)
    while done == 0:
        if DEADLINE:
            _fail_late(t_end, done)
        v = tl.load(src, volatile=True)
        if LAM:
            done = tl.min(tl.min((v != -1).to(tl.int32), axis=1), axis=0)
        else:
            done = tl.min(tl.min(((v >> 32) == e).to(tl.int32), axis=1), axis=0)
    slot = ((i // QL) * CP + cc) * QL + i % QL
    if LAM:
        tl.store(q_all + slot[:, None] * WORDS + w[None, :], v)
    else:
        tl.store(
            q_all + slot[:, None] * WORDS + w[None, :], (v & 0xFFFFFFFF).to(tl.int32)
        )
    if i % QL == 0:
        tl.store(pre_lens + i // QL, tl.maximum(tl.load(seq_lens + i // QL) - QL, 0))
    tl.store(cnt + i, e)
    if LAM:
        _lam_rearm(laddr)


class _WaitOnlyGather:
    def __getitem__(self, grid):
        def run(
            q,
            q_stride,
            peers,
            me,
            ll_off,
            q_all,
            cnt,
            seq_lens,
            pre_lens,
            stock_lens,
            nb,
            thr,
            CP,
            WORDS,
            QL,
            FOLD,
            GATE,
            NB,
            GMAX,
            GWRITE,
            TRIG=False,
            WAIT=False,
            num_warps=4,
            launch_pdl=False,
            LAM=False,
            tmo=0,
            DEADLINE=False,
        ):
            assert FOLD and WAIT and launch_pdl and not GATE and not GWRITE, (
                "the wait half serves the ungated exact call only"
            )
            assert CP * WORDS >= 2 * num_warps * 32, (
                "the CTA-wide wait needs >= 2 words a thread"
            )
            return _cp_gather_q_wait[grid](
                peers,
                me,
                ll_off,
                q_all,
                cnt,
                seq_lens,
                pre_lens,
                CP=CP,
                WORDS=WORDS,
                QL=QL,
                TRIG=TRIG,
                num_warps=num_warps,
                launch_pdl=True,
                **({"LAM": True} if LAM else {}),
                **({"tmo": tmo, "DEADLINE": True} if DEADLINE else {}),
            )

        return run


_WAIT_ONLY_GATHER = _WaitOnlyGather()

_TMO_NS = 0


def set_wait_deadline(deadline_s: float | None) -> None:
    global _TMO_NS
    if deadline_s is not None and deadline_s < 0:
        raise ValueError(f"wait deadline {deadline_s} s")
    _TMO_NS = int(round(float(deadline_s) * 1e9)) if deadline_s else 0


def _dl_args() -> dict:
    """The wait deadline's launch arguments (none while it is off)."""
    if not _TMO_NS:
        return {}
    return {"tmo": _TMO_NS, "DEADLINE": True}


def _cp_gather_q_ll_or_wait(cp, rows: int):
    pushed, cp._q_pushed = cp._q_pushed, None
    cp._q_pushed_lam = False
    gathered, cp._q_gathered = cp._q_gathered, None
    if pushed is None:
        assert gathered is None, "slot cp_q_gather: a gathered call is a pushed call"
        return _cp_gather_q_ll
    assert pushed == rows, (pushed, rows)
    if gathered is not None:
        # slot cp_q_gather: the q writer landed q_all, the prefix lengths and the epochs
        assert gathered == rows, (gathered, rows)
        return None
    return _WAIT_ONLY_GATHER


@triton.jit
def _cp_a2a_merge_ll(
    po,
    plse,
    po_sr,
    po_sc,
    pl_sr,
    pl_sc,
    q,
    q_stride,
    k_cache,
    v_cache,
    kv_page_stride,
    kv_tok_stride,
    table,
    t_stride,
    seq_lens,
    out,
    out_stride,
    peers,
    me,
    ll_off,
    cnt,
    bmm1,
    bmm2,
    nb,
    thr,
    QL: tl.constexpr,
    HQ: tl.constexpr,
    CP: tl.constexpr,
    D: tl.constexpr,
    PG: tl.constexpr,
    NWP: tl.constexpr,
    GATE: tl.constexpr,
    NB: tl.constexpr,
    GMAX: tl.constexpr,
    tmo=0,
    DEADLINE: tl.constexpr = False,
):
    """LL variant of _cp_a2a_merge; po / plse are read as int32 words, row / member strides po_sr / po_sc (words)
    and pl_sr / pl_sc. Online LSE merge of the causal block and the CP partials as they arrive.
    GATE: exits with the gate off (out keeps the stock call's rows)."""
    i = tl.program_id(0)
    if GATE:  # noqa: SIM102
        if _gate_on(seq_lens, nb, thr, NB, GMAX).to(tl.int32) == 0:
            return
    e = tl.load(cnt + i) + 1
    flag = e.to(tl.int64) << 32
    NO: tl.constexpr = HQ * D // 2
    hw = tl.arange(0, NO)
    h = tl.arange(0, HQ)
    d = tl.arange(0, D)
    hd = h[:, None] * D + d[None, :]
    for c in tl.static_range(CP):
        w = _ll_pack(flag, tl.load(po + i * po_sr + c * po_sc + hw))
        lw = _ll_pack(flag, tl.load(plse + i * pl_sr + c * pl_sc + h))
        dst = tl.load(peers + c) + ll_off + (i * CP + me) * NWP * 8
        _st_relaxed_sys_b64(dst + hw.to(tl.int64) * 8, w)
        _st_relaxed_sys_b64(dst + (NO + h).to(tl.int64) * 8, lw)
    b = i // QL
    t = i % QL
    L = tl.load(seq_lens + b)
    j = tl.arange(0, 8)
    pos = L - QL + j
    valid = (j <= t) & (pos >= 0)
    pos = tl.maximum(pos, 0)
    page = tl.load(table + b * t_stride + pos // PG, mask=valid, other=0)
    koff = page.to(tl.int64) * kv_page_stride + (pos % PG) * kv_tok_stride
    kk = tl.load(
        k_cache + koff[:, None] + d[None, :], mask=valid[:, None], other=0.0
    ).to(tl.float32)
    vv = tl.load(
        v_cache + koff[:, None] + d[None, :], mask=valid[:, None], other=0.0
    ).to(tl.float32)
    qv = tl.load(q + i * q_stride + hd).to(tl.float32)
    s = tl.sum(qv[:, None, :] * kk[None, :, :], axis=2) * bmm1
    s = tl.where(valid[None, :], s, float("-inf"))
    md = tl.max(s, axis=1)
    md0 = tl.where(md > float("-inf"), md, 0.0)
    p = tl.exp(s - md0[:, None])
    ld = tl.sum(p, axis=1)
    acc = tl.sum(p[:, :, None] * vv[None, :, :], axis=1) * bmm2
    m = tl.where(ld > 0, md0, float("-inf"))
    wsum = ld
    n_pages = (tl.maximum(L - QL, 0) + PG - 1) // PG
    own = tl.load(peers + me) + ll_off
    if DEADLINE:
        t_end = _t_end(tmo)
    else:
        t_end = 0
    for c in tl.static_range(CP):
        src = (own + (i * CP + c) * NWP * 8).to(tl.pointer_type(tl.int64))
        v = _ll_wait(src, NO, e, t_end, DEADLINE)
        lv = _ll_wait(src + NO, HQ, e, t_end, DEADLINE)
        lc = (lv & 0xFFFFFFFF).to(tl.int32).to(
            tl.float32, bitcast=True
        ) * 0.6931471805599453
        lc = tl.where(n_pages > c, lc, float("-inf"))
        wd = (v & 0xFFFFFFFF).to(tl.uint32)
        lo = (wd << 16).to(tl.float32, bitcast=True)
        hi = (wd & 0xFFFF0000).to(tl.float32, bitcast=True)
        oc = tl.reshape(tl.join(lo, hi), (HQ, D))
        mn = tl.maximum(m, lc)
        mn0 = tl.where(mn > float("-inf"), mn, 0.0)
        a_old = tl.where(m > float("-inf"), tl.exp(m - mn0), 0.0)
        a_new = tl.where(lc > float("-inf"), tl.exp(lc - mn0), 0.0)
        acc = acc * a_old[:, None] + oc * a_new[:, None]
        wsum = wsum * a_old + a_new
        m = mn
    o = acc / tl.where(wsum > 0, wsum, 1.0)[:, None]
    tl.store(out + i * out_stride + hd, o.to(tl.bfloat16))
    tl.store(cnt + i, e)


@triton.jit
def _cp_a2a_merge4_ll(
    po,
    plse,
    out,
    out_stride,
    peers,
    me,
    ll_off,
    cnt,
    HQ: tl.constexpr,
    CP: tl.constexpr,
    D: tl.constexpr,
    NWP: tl.constexpr,
    tmo=0,
    DEADLINE: tl.constexpr = False,
):
    """_cp_a2a_merge_ll for fused partials, which already hold the causal block (O = 0 / LSE = -inf where a rank
    sees no key): push, then a base-2 online LSE merge of the CP partials only."""
    i = tl.program_id(0)
    e = tl.load(cnt + i) + 1
    flag = e.to(tl.int64) << 32
    NO: tl.constexpr = HQ * D // 2
    hw = tl.arange(0, NO)
    h = tl.arange(0, HQ)
    d = tl.arange(0, D)
    for c in tl.static_range(CP):
        w = _ll_pack(flag, tl.load(po + (i * CP * HQ + c * HQ) * (D // 2) + hw))
        lw = _ll_pack(flag, tl.load(plse + i * CP * HQ + c * HQ + h))
        dst = tl.load(peers + c) + ll_off + (i * CP + me) * NWP * 8
        _st_relaxed_sys_b64(dst + hw.to(tl.int64) * 8, w)
        _st_relaxed_sys_b64(dst + (NO + h).to(tl.int64) * 8, lw)
    m = tl.full([HQ], float("-inf"), tl.float32)
    wsum = tl.zeros([HQ], dtype=tl.float32)
    acc = tl.zeros([HQ, D], dtype=tl.float32)
    own = tl.load(peers + me) + ll_off
    if DEADLINE:
        t_end = _t_end(tmo)
    else:
        t_end = 0
    for c in tl.static_range(CP):
        src = (own + (i * CP + c) * NWP * 8).to(tl.pointer_type(tl.int64))
        v = _ll_wait(src, NO, e, t_end, DEADLINE)
        lv = _ll_wait(src + NO, HQ, e, t_end, DEADLINE)
        lc = (lv & 0xFFFFFFFF).to(tl.int32).to(tl.float32, bitcast=True)
        wd = (v & 0xFFFFFFFF).to(tl.uint32)
        lo = (wd << 16).to(tl.float32, bitcast=True)
        hi = (wd & 0xFFFF0000).to(tl.float32, bitcast=True)
        oc = tl.reshape(tl.join(lo, hi), (HQ, D))
        mn = tl.maximum(m, lc)
        mn0 = tl.where(mn > float("-inf"), mn, 0.0)
        a_old = tl.where(m > float("-inf"), tl.exp2(m - mn0), 0.0)
        a_new = tl.where(lc > float("-inf"), tl.exp2(lc - mn0), 0.0)
        acc = acc * a_old[:, None] + oc * a_new[:, None]
        wsum = wsum * a_old + a_new
        m = mn
    o = acc / tl.where(wsum > 0, wsum, 1.0)[:, None]
    tl.store(out + i * out_stride + h[:, None] * D + d[None, :], o.to(tl.bfloat16))
    tl.store(cnt + i, e)


@triton.jit
def _gdc_wait_clobber():
    # griddepcontrol.wait with a memory clobber: no load may be hoisted above it
    tl.inline_asm_elementwise(
        "griddepcontrol.wait; // dummy $0",
        "=r,~{memory}",
        [],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _causal_block_h(
    q,
    q_stride,
    i,
    h,
    L,
    k_cache,
    v_cache,
    kv_page_stride,
    kv_tok_stride,
    table,
    t_stride,
    bmm1,
    bmm2,
    QL: tl.constexpr,
    D: tl.constexpr,
    PG: tl.constexpr,
    CG: tl.constexpr = False,
):
    """Head h of row i: the fp32 causal block over this step's QL keys as _cp_a2a_merge_ll starts its merge, i.e.
    unnormalized O (x bmm2) [D], the natural-log max [1] (-inf: no key) and the row sum [1]."""
    b = i // QL
    t = i % QL
    d = tl.arange(0, D)
    j = tl.arange(0, 8)
    pos = L - QL + j
    valid = (j <= t) & (pos >= 0)
    pos = tl.maximum(pos, 0)
    CM: tl.constexpr = ".cg" if CG else ""
    page = tl.load(
        table + b * t_stride + pos // PG, mask=valid, other=0, cache_modifier=CM
    )
    koff = page.to(tl.int64) * kv_page_stride + (pos % PG) * kv_tok_stride
    kk = tl.load(
        k_cache + koff[:, None] + d[None, :],
        mask=valid[:, None],
        other=0.0,
        cache_modifier=CM,
    ).to(tl.float32)
    vv = tl.load(
        v_cache + koff[:, None] + d[None, :],
        mask=valid[:, None],
        other=0.0,
        cache_modifier=CM,
    ).to(tl.float32)
    qv = tl.load(q + i * q_stride + h * D + d, cache_modifier=CM).to(tl.float32)
    s = tl.sum(qv[None, :] * kk, axis=1) * bmm1
    s = tl.where(valid, s, float("-inf"))
    md = tl.max(s, axis=0)
    md0 = tl.where(md > float("-inf"), md, 0.0)
    p = tl.exp(s - md0)
    ld = tl.sum(p, axis=0)
    acc = tl.sum(p[:, None] * vv, axis=0) * bmm2
    m = tl.where(ld > 0, md0, float("-inf"))
    return acc, m + tl.zeros([1], tl.float32), ld + tl.zeros([1], tl.float32)


@triton.jit
def _cp_merge_h_ll(
    po,
    plse,
    po_sr,
    po_sc,
    pl_sr,
    pl_sc,
    q,
    q_stride,
    k_cache,
    v_cache,
    kv_page_stride,
    kv_tok_stride,
    table,
    t_stride,
    seq_lens,
    out,
    out_stride,
    peers,
    me,
    ll_off,
    cnt,
    bmm1,
    bmm2,
    nb,
    thr,
    QL: tl.constexpr,
    HQ: tl.constexpr,
    CP: tl.constexpr,
    D: tl.constexpr,
    PG: tl.constexpr,
    NWP: tl.constexpr,
    NB: tl.constexpr,
    GMAX: tl.constexpr,
    WAIT: tl.constexpr,
    EARLY: tl.constexpr = False,
    po_v=None,
    pl_v=None,
    vstart=None,
    vcnt=None,
    CHUNK: tl.constexpr = False,
    gate=None,
    gate_stride=0,
    o_proj_input_scale=None,
    o_proj_input=None,
    o_proj_input_stride=0,
    GATE_EPILOGUE: tl.constexpr = False,
    PLAIN_WAIT: tl.constexpr = False,
    PUSH_ROT: tl.constexpr = False,
    CG: tl.constexpr = False,
    NOPUSH: tl.constexpr = False,
    MSLEEP: tl.constexpr = 0,
    HILO: tl.constexpr = False,
    F16: tl.constexpr = False,
    GATE_BF16: tl.constexpr = False,
    TRIG: tl.constexpr = False,
    tmo=0,
    DEADLINE: tl.constexpr = False,
):
    """_cp_a2a_merge_ll (gated) with one program per (row, head): pushes the head's partials, computes its causal
    block, then merges the CP partials as they arrive. One epoch per (row, head) in cnt, in an LL region of its own
    (the row merge's epochs differ). WAIT: launched as a programmatic dependent.
    EARLY (with WAIT): the causal block before the wait, which then guards only the partials;
    its predecessor releases it only after its own wait, so the step's q / K / V (written upstream of that wait) are
    complete.
    CHUNK (fold layout): a request with chunk rows sends the base-2 LSE combine of its rows' partials
    (po_v [nv, QL * CP * HQ, D] bf16, pl_v [nv, QL * CP * HQ] fp32) instead of po / plse.
    GATE_EPILOGUE also writes the gated, statically scaled FP8 input of o_proj."""
    tl.static_assert(
        not NOPUSH or (WAIT and PLAIN_WAIT and not CHUNK and (HILO or (EARLY and CG))),
        "NOPUSH: the early, plain-wait, L2-read merge of the partial's push, or a HILO merge",
    )
    tl.static_assert(not (HILO and CHUNK), "HILO: no fold layout")
    tl.static_assert(not TRIG or WAIT, "TRIG: a programmatic dependent")
    tl.static_assert(
        not (F16 and (HILO or CHUNK)),
        "F16 partials: no residual words, no fold layout",
    )
    tl.static_assert(
        not (GATE_BF16 and GATE_EPILOGUE),
        "GATE_BF16: the bf16 output gate, no FP8 input",
    )
    i = tl.program_id(0)
    h = tl.program_id(1)
    if TRIG:
        tl.extra.cuda.gdc_launch_dependents()
    if WAIT and not EARLY:
        _gdc_wait_clobber()
    CGM: tl.constexpr = ".cg" if CG else ""
    if _gate_on(seq_lens, nb, thr, NB, GMAX, CG).to(tl.int32) == 0:
        if WAIT and EARLY:
            _gdc_wait_clobber()
        return
    one = tl.arange(0, 1)
    e = tl.load(cnt + i * HQ + h, cache_modifier=CGM) + 1
    flag = e.to(tl.int64) << 32
    if HILO:
        R = tl.num_programs(0)
        e_lo = tl.load(cnt + (R + i) * HQ + h, cache_modifier=CGM) + 1
        flag_lo = e_lo.to(tl.int64) << 32
    DW: tl.constexpr = D // 2
    NO: tl.constexpr = HQ * D // 2
    H: tl.constexpr = QL * CP * HQ
    w = tl.arange(0, DW)
    L = tl.load(seq_lens + i // QL, cache_modifier=CGM)
    if EARLY:
        acc, m, wsum = _causal_block_h(
            q,
            q_stride,
            i,
            h,
            L,
            k_cache,
            v_cache,
            kv_page_stride,
            kv_tok_stride,
            table,
            t_stride,
            bmm1,
            bmm2,
            QL,
            D,
            PG,
            CG,
        )
        if WAIT and not NOPUSH:
            _gdc_wait_clobber()
    if CHUNK:
        n_v = tl.load(vcnt + i // QL)
        v0 = tl.load(vstart + i // QL)
    for c in tl.static_range(0 if NOPUSH else CP):
        if PUSH_ROT:
            cs = (me + c) % CP
        else:
            cs = c
        if CHUNK:
            if n_v > 0:
                hh = (i % QL) * (CP * HQ) + cs * HQ + h
                mx = tl.full([1], float("-inf"), tl.float32)
                sw = tl.zeros([1], tl.float32)
                acc_lo = tl.zeros([DW], tl.float32)
                acc_hi = tl.zeros([DW], tl.float32)
                for j in range(n_v):
                    vr = v0 + j
                    lj = tl.load(pl_v + vr * H + hh + one)
                    wd = tl.load(po_v + (vr * H + hh).to(tl.int64) * DW + w).to(
                        tl.uint32, bitcast=True
                    )
                    mn = tl.maximum(mx, lj)
                    a = tl.exp2(mx - mn)
                    b2 = tl.exp2(lj - mn)
                    acc_lo = acc_lo * a + (wd << 16).to(tl.float32, bitcast=True) * b2
                    acc_hi = (
                        acc_hi * a + (wd & 0xFFFF0000).to(tl.float32, bitcast=True) * b2
                    )
                    sw = sw * a + b2
                    mx = mn
                inv = 1.0 / tl.where(sw > 0, sw, 1.0)
                lo16 = (acc_lo * inv).to(tl.bfloat16).to(tl.int16, bitcast=True).to(
                    tl.int32
                ) & 0xFFFF
                hi16 = (acc_hi * inv).to(tl.bfloat16).to(tl.int16, bitcast=True).to(
                    tl.int32
                ) & 0xFFFF
                lse2 = tl.where(
                    sw > 0,
                    mx + tl.log2(tl.where(sw > 0, sw, 1.0)),
                    float("-inf"),
                )
                ow = _ll_pack(flag, lo16 | (hi16 << 16))
                lw = _ll_pack(flag, lse2.to(tl.int32, bitcast=True))
            else:
                ow = _ll_pack(flag, tl.load(po + i * po_sr + cs * po_sc + h * DW + w))
                lw = _ll_pack(flag, tl.load(plse + i * pl_sr + cs * pl_sc + h + one))
        else:
            ow = _ll_pack(flag, tl.load(po + i * po_sr + cs * po_sc + h * DW + w))
            lw = _ll_pack(flag, tl.load(plse + i * pl_sr + cs * pl_sc + h + one))
        dst = tl.load(peers + cs) + ll_off + (i * CP + me) * NWP * 8
        _st_relaxed_sys_b64(dst + (h * DW + w).to(tl.int64) * 8, ow)
        _st_relaxed_sys_b64(dst + (NO + h + one).to(tl.int64) * 8, lw)
        if HILO:
            owl = _ll_pack(
                flag_lo, tl.load(po + (R + i) * po_sr + cs * po_sc + h * DW + w)
            )
            dst_lo = tl.load(peers + cs) + ll_off + ((R + i) * CP + me) * NWP * 8
            _st_relaxed_sys_b64(dst_lo + (h * DW + w).to(tl.int64) * 8, owl)
    if not EARLY:
        acc, m, wsum = _causal_block_h(
            q,
            q_stride,
            i,
            h,
            L,
            k_cache,
            v_cache,
            kv_page_stride,
            kv_tok_stride,
            table,
            t_stride,
            bmm1,
            bmm2,
            QL,
            D,
            PG,
        )
    n_pages = (tl.maximum(L - QL, 0) + PG - 1) // PG
    own = tl.load(peers + me) + ll_off
    if DEADLINE:
        t_end = _t_end(tmo)
    else:
        t_end = 0
    if NOPUSH and not HILO:
        cc = tl.arange(0, CP)
        sbase = own + (i * CP + cc) * NWP * 8
        daddr = (sbase[:, None] + (h * DW + w[None, :]) * 8).to(
            tl.pointer_type(tl.int64)
        )
        laddr = (sbase + (NO + h) * 8).to(tl.pointer_type(tl.int64))
        vall, lall = _ll_wait_all(daddr, laddr, e, MSLEEP, t_end, DEADLINE)
    for c in tl.static_range(CP):
        src = (own + (i * CP + c) * NWP * 8).to(tl.pointer_type(tl.int64))
        if NOPUSH and not HILO:
            v = tl.sum(tl.where(cc[:, None] == c, vall, 0), axis=0)
            lv = tl.sum(tl.where(cc == c, lall, 0), axis=0) + tl.zeros([1], tl.int64)
        elif HILO:
            two = tl.arange(0, 2)
            base = tl.where(
                two == 0,
                own + (i * CP + c) * NWP * 8,
                own + ((R + i) * CP + c) * NWP * 8,
            )
            addr = (base[None, :] + (h * DW + w[:, None]) * 8).to(
                tl.pointer_type(tl.int64)
            )
            ee = tl.where(two == 0, e, e_lo)[None, :] + w[:, None] * 0
            if PLAIN_WAIT:
                vv, lv = _ll_wait_plain_ee(addr, ee, src + NO + h, e, t_end, DEADLINE)
            else:
                if DEADLINE:
                    vv = _ll_spin_dl(addr, ee, t_end)
                else:
                    vv = _ll_spin(addr, ee)
                tl.debug_barrier()
                lv = _ll_wait(src + NO + h, 1, e, t_end, DEADLINE)
            v, vl = tl.split(vv)
        elif PLAIN_WAIT:
            v = _ll_wait_plain(src + h * DW, DW, e, t_end, DEADLINE)
            lv = _ll_wait_plain(src + NO + h, 1, e, t_end, DEADLINE)
        else:
            v = _ll_wait(src + h * DW, DW, e, t_end, DEADLINE)
            lv = _ll_wait(src + NO + h, 1, e, t_end, DEADLINE)
        lc = (lv & 0xFFFFFFFF).to(tl.int32).to(
            tl.float32, bitcast=True
        ) * 0.6931471805599453
        lc = tl.where(n_pages > c, lc, float("-inf"))
        wd = (v & 0xFFFFFFFF).to(tl.uint32)
        if F16:
            lo = (wd & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(
                tl.float32
            ) * bmm2
            hi = (wd >> 16).to(tl.uint16).to(tl.float16, bitcast=True).to(
                tl.float32
            ) * bmm2
        else:
            lo = (wd << 16).to(tl.float32, bitcast=True)
            hi = (wd & 0xFFFF0000).to(tl.float32, bitcast=True)
        oc = tl.reshape(tl.join(lo, hi), (D,))
        if HILO:
            wl = (vl & 0xFFFFFFFF).to(tl.uint32)
            oc = oc + tl.reshape(
                tl.join(
                    (wl << 16).to(tl.float32, bitcast=True),
                    (wl & 0xFFFF0000).to(tl.float32, bitcast=True),
                ),
                (D,),
            )
        mn = tl.maximum(m, lc)
        mn0 = tl.where(mn > float("-inf"), mn, 0.0)
        a_old = tl.where(m > float("-inf"), tl.exp(m - mn0), 0.0)
        a_new = tl.where(lc > float("-inf"), tl.exp(lc - mn0), 0.0)
        if PLAIN_WAIT:
            # the update as explicit FMAs, so its rounding does not depend on the wait's code
            acc = tl.fma(acc, a_old, oc * a_new)
            wsum = tl.fma(wsum, a_old, a_new)
        else:
            acc = acc * a_old + oc * a_new
            wsum = wsum * a_old + a_new
        m = mn
    o = (acc / tl.where(wsum > 0, wsum, 1.0)).to(tl.bfloat16)
    if NOPUSH and EARLY:
        _gdc_wait_clobber()
    d = tl.arange(0, D)
    if GATE_BF16:
        g = tl.load(gate + i * gate_stride + h * D + d, cache_modifier=CGM).to(
            tl.float32
        )
        o = (o.to(tl.float32) * tl.sigmoid(g)).to(tl.bfloat16)
    tl.store(out + i * out_stride + h * D + d, o)
    if GATE_EPILOGUE:
        g = tl.load(gate + i * gate_stride + h * D + d, cache_modifier=CGM).to(
            tl.float32
        )
        gated = (o.to(tl.float32) * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
        scale_inv = 1.0 / tl.load(o_proj_input_scale).to(tl.float32)
        tl.store(
            o_proj_input + i * o_proj_input_stride + h * D + d,
            tl.clamp(gated * scale_inv, -448.0, 448.0).to(tl.float8e4nv),
        )
    tl.store(cnt + i * HQ + h + one, e)
    if HILO:
        tl.store(cnt + (R + i) * HQ + h + one, e_lo)


@triton.jit
def _cp_merge_h_ll_plain(
    po,
    plse,
    po_sr,
    po_sc,
    pl_sr,
    pl_sc,
    q,
    q_stride,
    k_cache,
    v_cache,
    kv_page_stride,
    kv_tok_stride,
    table,
    t_stride,
    seq_lens,
    out,
    out_stride,
    peers,
    me,
    ll_off,
    cnt,
    bmm1,
    bmm2,
    nb,
    thr,
    QL: tl.constexpr,
    HQ: tl.constexpr,
    CP: tl.constexpr,
    D: tl.constexpr,
    PG: tl.constexpr,
    NWP: tl.constexpr,
    NB: tl.constexpr,
    GMAX: tl.constexpr,
    WAIT: tl.constexpr,
    po_v=None,
    pl_v=None,
    vstart=None,
    vcnt=None,
    CHUNK: tl.constexpr = False,
    tmo=0,
    DEADLINE: tl.constexpr = False,
):
    """R11's plain non-EARLY merge, kept separate from EARLY/gate epilogues.

    The separate JIT function is intentional: it gives draft calls the same
    compile-time program as R11 instead of relying on dead-code elimination in
    the feature-rich shared merge.
    """
    i = tl.program_id(0)
    h = tl.program_id(1)
    if WAIT:
        _gdc_wait_clobber()
    if _gate_on(seq_lens, nb, thr, NB, GMAX).to(tl.int32) == 0:
        return
    one = tl.arange(0, 1)
    e = tl.load(cnt + i * HQ + h) + 1
    flag = e.to(tl.int64) << 32
    DW: tl.constexpr = D // 2
    NO: tl.constexpr = HQ * D // 2
    H: tl.constexpr = QL * CP * HQ
    w = tl.arange(0, DW)
    if CHUNK:
        n_v = tl.load(vcnt + i // QL)
        v0 = tl.load(vstart + i // QL)
    for c in tl.static_range(CP):
        if CHUNK:
            if n_v > 0:
                hh = (i % QL) * (CP * HQ) + c * HQ + h
                mx = tl.full([1], float("-inf"), tl.float32)
                sw = tl.zeros([1], tl.float32)
                acc_lo = tl.zeros([DW], tl.float32)
                acc_hi = tl.zeros([DW], tl.float32)
                for j in range(0, n_v):
                    vr = v0 + j
                    lj = tl.load(pl_v + vr * H + hh + one)
                    wd = tl.load(po_v + (vr * H + hh).to(tl.int64) * DW + w).to(
                        tl.uint32, bitcast=True
                    )
                    mn = tl.maximum(mx, lj)
                    a = tl.exp2(mx - mn)
                    b2 = tl.exp2(lj - mn)
                    acc_lo = acc_lo * a + (wd << 16).to(tl.float32, bitcast=True) * b2
                    acc_hi = (
                        acc_hi * a + (wd & 0xFFFF0000).to(tl.float32, bitcast=True) * b2
                    )
                    sw = sw * a + b2
                    mx = mn
                inv = 1.0 / tl.where(sw > 0, sw, 1.0)
                lo16 = (acc_lo * inv).to(tl.bfloat16).to(tl.int16, bitcast=True).to(
                    tl.int32
                ) & 0xFFFF
                hi16 = (acc_hi * inv).to(tl.bfloat16).to(tl.int16, bitcast=True).to(
                    tl.int32
                ) & 0xFFFF
                lse2 = tl.where(
                    sw > 0,
                    mx + tl.log2(tl.where(sw > 0, sw, 1.0)),
                    float("-inf"),
                )
                ow = _ll_pack(flag, lo16 | (hi16 << 16))
                lw = _ll_pack(flag, lse2.to(tl.int32, bitcast=True))
            else:
                ow = _ll_pack(flag, tl.load(po + i * po_sr + c * po_sc + h * DW + w))
                lw = _ll_pack(flag, tl.load(plse + i * pl_sr + c * pl_sc + h + one))
        else:
            ow = _ll_pack(flag, tl.load(po + i * po_sr + c * po_sc + h * DW + w))
            lw = _ll_pack(flag, tl.load(plse + i * pl_sr + c * pl_sc + h + one))
        dst = tl.load(peers + c) + ll_off + (i * CP + me) * NWP * 8
        _st_relaxed_sys_b64(dst + (h * DW + w).to(tl.int64) * 8, ow)
        _st_relaxed_sys_b64(dst + (NO + h + one).to(tl.int64) * 8, lw)
    L = tl.load(seq_lens + i // QL)
    acc, m, wsum = _causal_block_h(
        q,
        q_stride,
        i,
        h,
        L,
        k_cache,
        v_cache,
        kv_page_stride,
        kv_tok_stride,
        table,
        t_stride,
        bmm1,
        bmm2,
        QL,
        D,
        PG,
    )
    n_pages = (tl.maximum(L - QL, 0) + PG - 1) // PG
    own = tl.load(peers + me) + ll_off
    if DEADLINE:
        t_end = _t_end(tmo)
    else:
        t_end = 0
    for c in tl.static_range(CP):
        src = (own + (i * CP + c) * NWP * 8).to(tl.pointer_type(tl.int64))
        v = _ll_wait_plain(src + h * DW, DW, e, t_end, DEADLINE)
        lv = _ll_wait_plain(src + NO + h, 1, e, t_end, DEADLINE)
        lc = (lv & 0xFFFFFFFF).to(tl.int32).to(
            tl.float32, bitcast=True
        ) * 0.6931471805599453
        lc = tl.where(n_pages > c, lc, float("-inf"))
        wd = (v & 0xFFFFFFFF).to(tl.uint32)
        lo = (wd << 16).to(tl.float32, bitcast=True)
        hi = (wd & 0xFFFF0000).to(tl.float32, bitcast=True)
        oc = tl.reshape(tl.join(lo, hi), (D,))
        mn = tl.maximum(m, lc)
        mn0 = tl.where(mn > float("-inf"), mn, 0.0)
        a_old = tl.where(m > float("-inf"), tl.exp(m - mn0), 0.0)
        a_new = tl.where(lc > float("-inf"), tl.exp(lc - mn0), 0.0)
        acc = acc * a_old + oc * a_new
        wsum = wsum * a_old + a_new
        m = mn
    o = acc / tl.where(wsum > 0, wsum, 1.0)
    tl.store(out + i * out_stride + h * D + tl.arange(0, D), o.to(tl.bfloat16))
    tl.store(cnt + i * HQ + h + one, e)


class CpLite:
    """One process's exchange buffer and scratch, shared by the Qwen3.5 target and
    draft slots; built collectively on the plugin binding's TP group."""

    _instance: CpLite | None = None
    _q_pushed: int | None = None
    merge_cg: bool = False
    lam_q: bool = False
    lamq_off: int = 0
    lam_q_end: int = 0
    _q_pushed_lam: bool = False
    o_push: bool = False
    _ll_tab_t: torch.Tensor | None = None
    _q_gathered: int | None = None
    _gate_bf16: torch.Tensor | None = None
    _gate_bf16_done: bool = False

    def __init__(
        self, group, tp_rank: int, tp_size: int, cp: int, q_heads: int, device
    ):
        import torch.distributed._symmetric_memory as symm_mem

        self.cp, self.hq, self.device = cp, q_heads, torch.device(device)
        self.me = tp_rank % cp
        g0 = tp_rank - self.me
        self.min_bs = 2
        self.q_bytes = _MAX_ROWS * cp * q_heads * HEAD_DIM  # fp8
        self.p_off = self.q_bytes
        self.p_bytes = _MAX_ROWS * cp * q_heads * HEAD_DIM * 2  # bf16
        self.l_off = self.p_off + self.p_bytes
        self.l_bytes = _MAX_ROWS * cp * q_heads * 4
        self.f_off = self.l_off + self.l_bytes
        self.llq_off = self.f_off + _KINDS * _MAX_ROWS * cp * 4
        self.llp_off = self.llq_off + _MAX_ROWS * cp * (q_heads * HEAD_DIM // 4) * 8
        total = self.llp_off + _MAX_ROWS * cp * _LL_PWORDS * 8
        self.cp_ungated = True
        self.gather_pdl = self.cp_ungated
        self.early_merge = self.cp_ungated
        self.merge_h = True
        self.fold_mh = True
        self.fold_chunk = {33: 128, 41: 64, 65: 128}
        self.fold_chunk_q1 = {65: 128}
        self.fold_ragged = 1.8
        self.fold_fuse = True
        self.llh_off = total
        total += _MAX_ROWS * cp * _LL_PWORDS * 8
        if self.lam_q:
            # filled with the sentinel below, before the rendezvous and the group barrier
            self.lamq_off, self.lam_q_end = _lam_q_layout(total, cp, q_heads)
            total = self.lam_q_end
        self.ll = q_heads * HEAD_DIM // 2 + q_heads <= _LL_PWORDS
        self.cnt = torch.zeros(_KINDS, _MAX_ROWS, dtype=torch.int32, device=self.device)
        self.sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        self.fused = self.ll
        self.exact = self.fused
        if self.fused:
            from sgl_nvidia_mega.qwen3_5.sm103.attention import q35_cp_fused

            # grid-barrier counters and workspace: persistent, allocated before any capture
            q35_cp_fused.q35_cp_fused_prepare(self.device, self.sms)
        if symm_mem.get_backend(self.device) is None:
            symm_mem.set_backend("NCCL")
        self.buf = symm_mem.empty((total,), dtype=torch.uint8, device=self.device)
        self.buf.zero_()
        if self.lam_q:
            self.buf[self.lamq_off : self.lam_q_end].fill_(0xFF)
        torch.cuda.synchronize(self.device)
        handle = symm_mem.rendezvous(self.buf, group)
        assert handle.rank == tp_rank, (handle.rank, tp_rank)
        self.peers = torch.tensor(
            [int(p) for p in handle.buffer_ptrs[g0 : g0 + cp]],
            dtype=torch.int64,
            device=self.device,
        )
        self._handle = handle
        if cp <= 4:
            self._ll_tab()
        self.q_all = (
            self.buf[: self.q_bytes]
            .view(torch.float8_e4m3fn)
            .view(_MAX_ROWS, cp * q_heads, HEAD_DIM)
        )
        rows = _MAX_ROWS
        self.po = torch.empty(
            rows, cp * q_heads, HEAD_DIM, dtype=torch.bfloat16, device=self.device
        )
        self.plse = torch.empty(
            rows, cp * q_heads, dtype=torch.float32, device=self.device
        )
        self.counter = torch.zeros(1 << 20, dtype=torch.int32, device=self.device)
        # exact path: the fused kernel's [cp, rows, heads, D] partials and the prefix lengths the gather writes
        self.po_x = torch.empty(
            cp, rows, q_heads, HEAD_DIM, dtype=torch.bfloat16, device=self.device
        )
        self.pl_x = torch.empty(
            cp, rows, q_heads, dtype=torch.float32, device=self.device
        )
        self.pre_lens = torch.zeros(rows, dtype=torch.int32, device=self.device)
        self.min_kv = {}
        self.gate_max = False
        self.stock_lens = torch.zeros(rows, dtype=torch.int32, device=self.device)
        self.gate_overlap = 1
        self.cp_main = True
        self.cp_pdl = self.cp_main
        self._gate_epilogue = None
        self._gate_epilogue_done = False
        self.cnt_h = torch.zeros(
            _MAX_ROWS * q_heads, dtype=torch.int32, device=self.device
        )
        self._side = torch.cuda.Stream(device=self.device)
        self._ws = None
        self._ws_bytes: dict = {}
        self._ws_captured: list = []
        self._ws_in_graph = False
        if self.min_kv and self.gate_overlap:
            self._own_workspace()
        # (lengths, b, q_len, threshold) -> [(stock lengths, prefix lengths), forward, last layer id that used them]
        self._glens: dict = {}
        self._logged: set = set()
        # (page table, lengths, q_len) -> [tables, owned lengths, forward, last layer id that used them]
        self._tabs: dict = {}
        torch.distributed.barrier(group)
        logger.info(
            "TRT-L1 CP-lite: tp rank %d, KV group of %d (group rank %d), min batch %d, LL %s, fused %s",
            tp_rank,
            cp,
            self.me,
            self.min_bs,
            self.ll,
            self.fused,
        )
        logger.info(
            "TRT-L1 CP-lite exact (prefix-only fused partial + fp32 causal block): %s",
            self.exact,
        )
        logger.info(
            "TRT-L18 CP-lite current-stream CP: %s, PDL: %s, per-head merge: %s, ungated exact: %s, gather PDL: %s, early merge: %s",
            self.cp_main,
            self.cp_pdl,
            self.merge_h,
            self.cp_ungated,
            self.gather_pdl,
            self.early_merge,
        )
        logger.info(
            "TRT-L1 CP-lite KV gate: minimum mean KV per request %s", self.min_kv
        )
        if self.lam_q:
            logger.info(
                "NVIDIA megakernel: Qwen3.5 CP-lite Lamport q region of %d B at %d, "
                "sentinel-filled (slot cp_lam_q)",
                self.lam_q_end - self.lamq_off,
                self.lamq_off,
            )

    def offer_gate_epilogue(
        self, gate: torch.Tensor, scale: torch.Tensor, out: torch.Tensor
    ) -> None:
        """Offer the next exact CP-lite call the attention gate and FP8 o_proj input."""
        self._gate_epilogue = (gate, scale, out)
        self._gate_epilogue_done = False

    def q_push_args(
        self,
        rows: int,
        b: int,
        q_len: int,
        gather: bool = False,
        seq_lens: torch.Tensor | None = None,
    ) -> dict | None:
        if not (
            self.ll
            and self.cp_ungated
            and self.exact
            and not self.min_kv
            and b >= self.min_bs
            and rows == b * q_len
            and rows <= _MAX_ROWS
            and self._exact_supported(q_len, b)
        ):
            return None
        lam = bool(self.lam_q)
        if lam and self.lam_q_end <= self.lamq_off:
            raise RuntimeError(
                "slot cp_lam_q: CP-lite was built without its Lamport regions"
            )
        self._q_pushed = rows
        self._q_pushed_lam = lam
        args = {
            "peers": self.peers,
            "me": self.me,
            "off": self.lamq_off if lam else self.llq_off,
            "cnt": self.cnt[0],
            "cp": self.cp,
            "words": self.hq * HEAD_DIM // 4,
            **({"lam": True} if lam else {}),
        }
        if gather and (
            lam
            and seq_lens is not None
            and seq_lens.shape[0] == b
            and seq_lens.dtype == torch.int32
        ):
            self._q_gathered = rows
            args["gather"] = {
                "q_all": self.q_all.view(torch.int32),
                "seq_lens": seq_lens,
                "pre_lens": self.pre_lens,
                "ql": q_len,
                # the in-front gather's poll takes the wait deadline (set_wait_deadline)
                **({"tmo": _TMO_NS} if _TMO_NS else {}),
            }
        return args

    def take_gate_epilogue(self) -> bool:
        """Return whether the last CP-lite call wrote the offered o_proj input."""
        done = self._gate_epilogue_done
        self._gate_epilogue = None
        self._gate_epilogue_done = False
        return done

    def offer_gate_bf16(self, gate: torch.Tensor) -> None:
        self._gate_bf16 = gate
        self._gate_bf16_done = False

    def take_gate_bf16(self) -> bool:
        """Return whether the last CP-lite call applied the offered gate, and drop the offer."""
        done = self._gate_bf16_done
        self._gate_bf16 = None
        self._gate_bf16_done = False
        return done

    def _gate_thr(self, b: int) -> int | None:
        """The device threshold of a b-request call (on the length sum, or with GATE_MAX on the max), or None."""
        keys = [k for k in self.min_kv if k <= b]
        if not keys:
            return None
        t = self.min_kv[max(keys)]
        return t if self.gate_max else t * b

    @classmethod
    def maybe_create(cls, binding, device) -> CpLite | None:
        """Build the Qwen3.5 CP state collectively on the attention TP group.

        The plugin spec pins the global Q/KV head counts and head dimension. Keeping
        those model-specific facts here avoids a runtime-side prebuild hook. Every
        rank calls this from plugin build before graph variants are validated.
        """
        if cls._instance is not None:
            return cls._instance
        tp, rank = binding.tp_size, binding.tp_rank
        topology = cp_lite_topology(tp, kernel_head_dim=HEAD_DIM)
        if topology is None:
            return None
        cp, q_heads = topology
        group = binding.tp_device_group
        cls._instance = cls(group, rank, tp, cp, q_heads, device)
        return cls._instance

    def _tables(self, page_table, seq_lens, q_len, forward_batch, layer_id, gate):
        """Once per forward: reused only by later layers of the same forward, so every capture and
        every single-layer draft step computes its own."""
        b = seq_lens.shape[0]
        ncols = triton.cdiv(page_table.shape[1], self.cp)
        key = (
            page_table.data_ptr(),
            page_table.shape,
            seq_lens.data_ptr(),
            b,
            q_len,
            gate["GATE"],
            gate["thr"],
        )
        ent = self._tabs.get(key)
        if ent is not None and ent[2]() is forward_batch and layer_id > ent[3]:
            ent[3] = layer_id
            return ent[0], ent[1], ncols * PAGE
        if len(self._tabs) > 64:
            self._tabs.clear()
        tab = torch.empty(b, ncols, dtype=torch.int32, device=self.device)
        lens = torch.empty(b, dtype=torch.int32, device=self.device)
        self._tabs[key] = [tab, lens, weakref.ref(forward_batch), layer_id]
        _cp_tables[(b, triton.cdiv(ncols, 256))](
            page_table,
            page_table.stride(0),
            seq_lens,
            tab,
            tab.stride(0),
            lens,
            ncols,
            self.me,
            **gate,
            QL=q_len,
            CP=self.cp,
            PG=PAGE,
            BLOCK=256,
        )
        return tab, lens, ncols * PAGE

    def attend(
        self,
        q: torch.Tensor,
        kv_cache,
        page_table: torch.Tensor,
        seq_lens: torch.Tensor,
        q_len: int,
        bmm1_scale,
        bmm2_scale,
        workspace: torch.Tensor,
        layer,
        forward_batch,
        stock=None,
        min_bs: int | None = None,
        early_merge: bool | None = None,
        plain_wait: bool = False,
        merge_h: bool = False,
        out: torch.Tensor | None = None,
        precise: bool = False,
        p2f16: bool = False,
        hilo_push: bool = False,
        hilo_early: bool = False,
        hilo_trig: bool = False,
    ) -> torch.Tensor | None:
        """CP-lite attention of this rank's heads: [rows, heads, head_dim] bf16, or None
        where the call is not served (decided by shapes only, so every rank agrees).
        stock(seq_lens, out): the caller's own kernel call into out; with it and a KV gate for this batch size,
        both are captured and the device picks one per call.
        min_bs: override the smallest batch this call serves; a caller that picks CP by
        its own host-side KV length can pass 1.
        early_merge: override the exact ungated partial/merge ordering for this call;
        without a fused gate epilogue, False also selects the R11 plain merge for
        exact and non-exact fold calls; None keeps the shared default."""
        rows = q.shape[0]
        b = seq_lens.shape[0]
        if (
            q.dtype != torch.float8_e4m3fn
            or q.dim() != 3
            or q.shape[1] != self.hq
            or q.shape[2] != HEAD_DIM
            or layer.tp_k_head_num != 1
            or layer.sliding_window_size not in (None, -1)
            or rows != b * q_len
            or b < (self.min_bs if min_bs is None else min_bs)
            or rows > _MAX_ROWS
            or not isinstance(bmm1_scale, float)
            or not isinstance(bmm2_scale, float)
        ):
            return None
        if out is not None and not (
            out.dtype == torch.bfloat16
            and tuple(out.shape) == (rows, self.hq, HEAD_DIM)
            and out.is_contiguous()
        ):
            return None
        if (b, q_len) not in self._logged:
            self._logged.add((b, q_len))
            logger.info(
                "TRT-L1 CP-lite engaged: %d requests x %d rows (layer %d)",
                b,
                q_len,
                layer.layer_id,
            )
        k_cache, v_cache = kv_cache
        q = q if q.is_contiguous() else q.contiguous()
        words = self.hq * HEAD_DIM // 4
        exact = self.exact and self._exact_supported(q_len, b)
        # Every path but the non-exact fused one can be gated; exact batches above
        # the fused kernel's limit take the TRT-LLM partial.
        thr = (
            self._gate_thr(b)
            if stock is not None and self.ll and (self.exact or not self.fused)
            else None
        )
        gate = {
            "nb": b,
            "thr": 0 if thr is None else thr,
            "GATE": thr is not None,
            "NB": triton.next_power_of_2(b),
            "GMAX": self.gate_max,
        }
        into = out
        if into is not None and not (
            exact and thr is None and self.ll and self.cp_ungated
        ):
            # only the exact ungated path writes into the caller's tensor; the others copy
            res = self.attend(
                q,
                kv_cache,
                page_table,
                seq_lens,
                q_len,
                bmm1_scale,
                bmm2_scale,
                workspace,
                layer,
                forward_batch,
                stock=stock,
                min_bs=min_bs,
                early_merge=early_merge,
                plain_wait=plain_wait,
                merge_h=merge_h,
                precise=precise,
                p2f16=p2f16,
                hilo_push=hilo_push,
                hilo_early=hilo_early,
                hilo_trig=hilo_trig,
            )
            if res is None:
                return None
            into.copy_(res)
            return into
        out = None
        if thr is not None and self.gate_overlap:
            return self._gated_overlap(
                q,
                k_cache,
                v_cache,
                page_table,
                seq_lens,
                q_len,
                b,
                rows,
                bmm1_scale,
                bmm2_scale,
                forward_batch,
                layer.layer_id,
                stock,
                gate,
                exact,
            )
        if exact and thr is None and self.ll and self.cp_ungated:
            return self._ungated_main(
                q,
                k_cache,
                v_cache,
                page_table,
                seq_lens,
                q_len,
                b,
                rows,
                bmm1_scale,
                bmm2_scale,
                gate,
                early_merge=early_merge,
                # precise calls poll with the plain wait (slot cp_merge_plain_wait's)
                plain_wait=plain_wait or precise,
                merge_h=merge_h,
                out=into,
                hilo=precise and b <= _PRECISE_MAX_BATCH and 2 * rows <= _MAX_ROWS,
                p_hilo=(precise or p2f16) and b <= _PRECISE_MAX_BATCH,
                f16=p2f16 and not precise and b <= _PRECISE_MAX_BATCH,
                hilo_push=hilo_push,
                hilo_early=hilo_early,
                hilo_trig=hilo_trig,
            )
        if self.ll:
            _cp_gather_q_ll[(rows,)](
                q.view(torch.int32),
                words,
                self.peers,
                self.me,
                self.llq_off,
                self.q_all.view(torch.int32),
                self.cnt[0],
                seq_lens,
                self.pre_lens,
                self.stock_lens,
                **gate,
                CP=self.cp,
                WORDS=words,
                QL=q_len,
                FOLD=exact,
                GWRITE=True,
                num_warps=4,
                **_dl_args(),
            )
            if thr is not None:
                out = torch.empty(
                    rows, self.hq, HEAD_DIM, dtype=torch.bfloat16, device=q.device
                )
                stock(self.stock_lens[:b], out)
        else:
            _cp_gather_q[(rows,)](
                q.view(torch.int32),
                words,
                self.peers,
                self.me,
                0,
                self.f_off,
                CP=self.cp,
                WORDS=words,
                num_warps=4,
                **_dl_args(),
            )
        if exact:
            return self._exact(
                q,
                k_cache,
                v_cache,
                page_table,
                seq_lens,
                q_len,
                b,
                rows,
                bmm1_scale,
                bmm2_scale,
                out,
                gate,
            )
        if self.fused and not self.exact:
            out = self._fused(
                k_cache,
                v_cache,
                page_table,
                seq_lens,
                q_len,
                b,
                rows,
                bmm1_scale,
                bmm2_scale,
            )
            if out is not None:
                return out
        self._fold_partial(
            k_cache,
            v_cache,
            page_table,
            seq_lens,
            q_len,
            b,
            rows,
            bmm1_scale,
            bmm2_scale,
            forward_batch,
            layer.layer_id,
            gate,
            workspace,
        )
        if out is None:
            out = torch.empty(
                rows, self.hq, HEAD_DIM, dtype=torch.bfloat16, device=q.device
            )
        if self.ll and self.fold_mh:
            fold_gate = gate if gate["GATE"] else {**gate, "thr": 0}
            self._fold_merge_h(
                q,
                k_cache,
                v_cache,
                page_table,
                seq_lens,
                q_len,
                rows,
                bmm1_scale,
                bmm2_scale,
                out,
                fold_gate,
                plain=early_merge is False,
            )
        elif self.ll:
            self._fold_merge(
                q,
                k_cache,
                v_cache,
                page_table,
                seq_lens,
                q_len,
                rows,
                bmm1_scale,
                bmm2_scale,
                out,
                gate,
            )
        else:
            _cp_a2a_merge[(rows,)](
                self.po,
                self.plse,
                q,
                q.stride(0),
                k_cache,
                v_cache,
                k_cache.stride(0),
                k_cache.stride(2),
                page_table,
                page_table.stride(0),
                seq_lens,
                out,
                out.stride(0),
                self.peers,
                self.me,
                self.p_off,
                self.l_off,
                self.f_off,
                _MAX_ROWS,
                bmm1_scale,
                bmm2_scale,
                QL=q_len,
                HQ=self.hq,
                CP=self.cp,
                D=HEAD_DIM,
                PG=PAGE,
                num_warps=8,
                **_dl_args(),
            )
        return out

    def _exact_supported(self, q_len, b):
        """The fused partial (q_len folded into the heads) takes <= MAX_BATCH verify / MAX_BATCH_Q1 draft requests;
        wider batches take the trtllm-gen partial."""
        from sgl_nvidia_mega.qwen3_5.sm103.attention import q35_cp_fused as F

        return F.q35_cp_fused_supported(
            1,
            self.cp,
            q_len * self.hq,
            b,
            self.sms,
            max_batch=F.MAX_BATCH if q_len > 1 else F.MAX_BATCH_Q1,
        )

    def _exact(
        self,
        q,
        k_cache,
        v_cache,
        page_table,
        seq_lens,
        q_len,
        b,
        rows,
        bmm1_scale,
        bmm2_scale,
        out,
        gate,
    ):
        """The fused partial on the context prefix (q_len folded into the heads: one query row of q_len x heads per
        request, no causal effect), then the trtllm-gen path's merge with the causal block in fp32."""
        self._exact_partial(
            k_cache,
            v_cache,
            page_table,
            self.pre_lens,
            q_len,
            b,
            rows,
            bmm1_scale,
            bmm2_scale,
        )
        return self._exact_merge(
            q,
            k_cache,
            v_cache,
            page_table,
            seq_lens,
            q_len,
            rows,
            bmm1_scale,
            bmm2_scale,
            out,
            gate,
        )

    def _gate_lens_fwd(
        self, seq_lens, b, q_len, thr, forward_batch, layer_id, zero=False
    ):
        """Once per forward, make the stock and exact-prefix lengths.

        With ``zero``, the stock call gets no key while CP is selected.
        """
        key = (seq_lens.data_ptr(), b, q_len, thr, zero)
        ent = self._glens.get(key)
        if ent is not None and ent[1]() is forward_batch and layer_id > ent[2]:
            ent[2] = layer_id
            return ent[0]
        if len(self._glens) > 64:
            self._glens.clear()
        lens = torch.empty(2, b, dtype=torch.int32, device=self.device)
        self._glens[key] = [(lens[0], lens[1]), weakref.ref(forward_batch), layer_id]
        _gate_lens[(1,)](
            seq_lens,
            b,
            thr,
            lens[0],
            lens[1],
            QL=q_len,
            NB=triton.next_power_of_2(b),
            GMAX=self.gate_max,
            ZERO=zero,
        )
        return lens[0], lens[1]

    def _gated_overlap(
        self,
        q,
        k_cache,
        v_cache,
        page_table,
        seq_lens,
        q_len,
        b,
        rows,
        bmm1_scale,
        bmm2_scale,
        forward_batch,
        layer_id,
        stock,
        gate,
        exact,
    ):
        """Gated call: stock(lengths, out) on this stream beside gather + partial on the side stream (the
        TRT-LLM partial with its own workspace), then the merge after the streams join."""
        if self.cp_main and exact and self.gate_overlap == 1:
            return self._gated_main(
                q,
                k_cache,
                v_cache,
                page_table,
                seq_lens,
                q_len,
                b,
                rows,
                bmm1_scale,
                bmm2_scale,
                forward_batch,
                layer_id,
                stock,
                gate,
            )
        fold_mh = self.fold_mh and not exact
        stock_lens, pre_lens = self._gate_lens_fwd(
            seq_lens,
            b,
            q_len,
            gate["thr"],
            forward_batch,
            layer_id,
            zero=fold_mh,
        )
        out = torch.empty(
            rows, self.hq, HEAD_DIM, dtype=torch.bfloat16, device=self.device
        )
        main = torch.cuda.current_stream()
        self._side.wait_stream(main)
        stock(stock_lens, out)
        words = self.hq * HEAD_DIM // 4
        fused = None
        with torch.cuda.stream(self._side):
            if fold_mh and self.fold_fuse:
                # Fill chunk-row queries in the gather and combine them in the merge.
                tabs = self._tables(
                    page_table,
                    seq_lens,
                    q_len,
                    forward_batch,
                    layer_id,
                    gate,
                )
                chunk = self._chunk_tables(
                    tabs[0], tabs[1], b, forward_batch, layer_id, q_len
                )
                if chunk is not None:
                    nv, h = chunk[1].shape[0], q_len * self.cp * self.hq
                    fused = (
                        chunk,
                        torch.empty(nv, h, HEAD_DIM, dtype=q.dtype, device=self.device),
                        torch.empty(
                            nv,
                            h,
                            HEAD_DIM,
                            dtype=torch.bfloat16,
                            device=self.device,
                        ),
                        torch.empty(nv, h, dtype=torch.float32, device=self.device),
                        tabs,
                    )
            _cp_gather_q_ll[(rows,)](
                q.view(torch.int32),
                words,
                self.peers,
                self.me,
                self.llq_off,
                self.q_all.view(torch.int32),
                self.cnt[0],
                seq_lens,
                pre_lens,
                self.stock_lens,
                **gate,
                CP=self.cp,
                WORDS=words,
                QL=q_len,
                FOLD=exact,
                GWRITE=False,
                num_warps=4,
                **(
                    {}
                    if fused is None
                    else {
                        "qv": fused[1].view(torch.int32),
                        "vstart": fused[0][3],
                        "vcnt": fused[0][4],
                        "CHUNK": True,
                    }
                ),
                **_dl_args(),
            )
            if exact:
                self._exact_partial(
                    k_cache,
                    v_cache,
                    page_table,
                    pre_lens,
                    q_len,
                    b,
                    rows,
                    bmm1_scale,
                    bmm2_scale,
                )
            else:
                self._fold_partial(
                    k_cache,
                    v_cache,
                    page_table,
                    seq_lens,
                    q_len,
                    b,
                    rows,
                    bmm1_scale,
                    bmm2_scale,
                    forward_batch,
                    layer_id,
                    gate,
                    self._own_workspace(
                        triton.cdiv(page_table.shape[1], self.cp), q_len
                    ),
                    fused,
                )
        main.wait_stream(self._side)
        merge = (
            self._exact_merge
            if exact
            else (self._fold_merge_h if fold_mh else self._fold_merge)
        )
        merge_args = (
            q,
            k_cache,
            v_cache,
            page_table,
            seq_lens,
            q_len,
            rows,
            bmm1_scale,
            bmm2_scale,
            out,
            gate,
        )
        if fused is None:
            return merge(*merge_args)
        return merge(*merge_args, fused=fused)

    def _gated_main(
        self,
        q,
        k_cache,
        v_cache,
        page_table,
        seq_lens,
        q_len,
        b,
        rows,
        bmm1_scale,
        bmm2_scale,
        forward_batch,
        layer_id,
        stock,
        gate,
    ):
        """Gated exact call (TRT-L18): gather, fused partial and merge (all exit with the gate off) on this stream. The
        stock kernel would hold SMs the partial's grid barrier waits for, so it runs on the side stream after the
        partial; it gets no key with the gate on (then writes nothing), so it and the merge write out in disjoint cases."""
        stock_lens, pre_lens = self._gate_lens_fwd(
            seq_lens, b, q_len, gate["thr"], forward_batch, layer_id, zero=True
        )
        out = torch.empty(
            rows, self.hq, HEAD_DIM, dtype=torch.bfloat16, device=self.device
        )
        words = self.hq * HEAD_DIM // 4
        _cp_gather_q_ll[(rows,)](
            q.view(torch.int32),
            words,
            self.peers,
            self.me,
            self.llq_off,
            self.q_all.view(torch.int32),
            self.cnt[0],
            seq_lens,
            pre_lens,
            self.stock_lens,
            **gate,
            CP=self.cp,
            WORDS=words,
            QL=q_len,
            FOLD=True,
            GWRITE=False,
            TRIG=self.cp_pdl,
            num_warps=4,
            **_dl_args(),
        )
        self._exact_partial(
            k_cache,
            v_cache,
            page_table,
            pre_lens,
            q_len,
            b,
            rows,
            bmm1_scale,
            bmm2_scale,
            pdl=self.cp_pdl,
        )
        main = torch.cuda.current_stream()
        self._side.wait_stream(main)
        with torch.cuda.stream(self._side):
            stock(stock_lens, out)
        if self.merge_h:
            self._merge_h(
                q,
                k_cache,
                v_cache,
                page_table,
                seq_lens,
                q_len,
                rows,
                bmm1_scale,
                bmm2_scale,
                out,
                gate,
            )
        else:
            self._exact_merge(
                q,
                k_cache,
                v_cache,
                page_table,
                seq_lens,
                q_len,
                rows,
                bmm1_scale,
                bmm2_scale,
                out,
                gate,
            )
        main.wait_stream(self._side)
        return out

    def _ungated_main(
        self,
        q,
        k_cache,
        v_cache,
        page_table,
        seq_lens,
        q_len,
        b,
        rows,
        bmm1_scale,
        bmm2_scale,
        gate,
        early_merge=None,
        plain_wait=False,
        merge_h=False,
        out=None,
        hilo=False,
        p_hilo=False,
        f16=False,
        hilo_push=False,
        hilo_early=False,
        hilo_trig=False,
    ):
        r11 = early_merge is False and not merge_h and self._gate_epilogue is None
        if hilo and r11:
            hilo = False
        assert not (f16 and hilo), "f16: no residual words"
        if f16 and r11:
            f16 = p_hilo = False
        hl = hilo or (p_hilo and not f16)
        early = self.early_merge if early_merge is None else early_merge
        words = self.hq * HEAD_DIM // 4
        pushed = self._q_pushed is not None
        q_lam = pushed and self._q_pushed_lam
        gather = _cp_gather_q_ll_or_wait(self, rows)
        if q_lam and ("lam_q", rows) not in self._logged:
            self._logged.add(("lam_q", rows))
            logger.info(
                "NVIDIA megakernel: Qwen3.5 slot cp_lam_q moves the pushed q rows as "
                "Lamport words at %d rows",
                rows,
            )
        if gather is not None:
            gather[(rows,)](
                q.view(torch.int32),
                words,
                self.peers,
                self.me,
                self.lamq_off if q_lam else self.llq_off,
                self.q_all.view(torch.int32),
                self.cnt[0],
                seq_lens,
                self.pre_lens,
                self.stock_lens,
                **gate,
                CP=self.cp,
                WORDS=words,
                QL=q_len,
                FOLD=True,
                GWRITE=False,
                TRIG=True,
                WAIT=self.gather_pdl,
                num_warps=4,
                launch_pdl=self.gather_pdl,
                **({"LAM": True} if q_lam else {}),
                **_dl_args(),
            )
        epilogue = self._gate_epilogue
        if epilogue is not None and not (
            tuple(epilogue[0].shape) == (rows, self.hq * HEAD_DIM)
            and epilogue[0].stride(1) == 1
            and tuple(epilogue[2].shape) == (rows, self.hq * HEAD_DIM)
            and epilogue[2].stride(1) == 1
        ):
            epilogue = None
        plain = early_merge is False and epilogue is None and not merge_h
        draft_hilo = hilo and merge_h and not early and epilogue is None and plain_wait
        # slot draft_gate_fold: that merge applies the offered attention output gate
        gate_bf16 = self._gate_bf16
        if gate_bf16 is not None and not (
            draft_hilo
            and gate_bf16.dtype == torch.bfloat16
            and tuple(gate_bf16.shape) == (rows, self.hq * HEAD_DIM)
            and gate_bf16.stride(1) == 1
        ):
            gate_bf16 = None
        o_push = (
            self.o_push
            and pushed
            and plain_wait
            and early
            and not plain
            and not hl
            and b <= 64
        )
        hilo_nopush = bool(hilo_push) and draft_hilo and b <= 64
        hilo_early = hilo_nopush and bool(hilo_early)
        hilo_trig = hilo_early and bool(hilo_trig) and q_len == 1
        if hilo_nopush:
            o_push = True
            if ("hilo_push", rows) not in self._logged:
                self._logged.add(("hilo_push", rows))
                logger.info(
                    "NVIDIA megakernel: Qwen3.5 slot draft_o_push: CP-lite's partial "
                    "pushes the (O, LSE) and residual words of its precise merge at %d rows "
                    "(merge %s)",
                    rows,
                    (
                        "early, wait at its end, o_proj released at entry: options "
                        "draft_merge_early, draft_hilo_trig"
                        if hilo_trig
                        else "early, wait at its end: option draft_merge_early"
                    )
                    if hilo_early
                    else "after its wait",
                )
        elif hilo_push and ("hilo_push", rows) not in self._logged:
            self._logged.add(("hilo_push", rows))
            logger.warning(
                "NVIDIA megakernel: Qwen3.5 slot draft_o_push: the call of %d rows is not "
                "the precise multi-request call; its merge pushes",
                rows,
            )
        if o_push and not hilo_nopush and ("o_push", rows) not in self._logged:
            self._logged.add(("o_push", rows))
            logger.info(
                "NVIDIA megakernel: Qwen3.5 slot cp_o_push: CP-lite's partial pushes "
                "the (O, LSE) words of its merge at %d rows",
                rows,
            )
        self._exact_partial(
            k_cache,
            v_cache,
            page_table,
            seq_lens,
            q_len,
            b,
            rows,
            bmm1_scale,
            bmm2_scale,
            pdl=True,
            len_sub=q_len,
            early_trigger=early and not o_push,
            o_push=o_push,
            o_hilo=hilo,
            p_hilo=p_hilo,
            o_f16=f16,
            # the precise draft merge after its wait: released at the partial's end
            late_trigger=hilo_early or not hilo_nopush,
        )
        if (hilo or p_hilo) and ("hilo", rows, f16, o_push) not in self._logged:
            self._logged.add(("hilo", rows, f16, o_push))
            logger.info(
                "TRT-L1 CP-lite precise call at %d rows: two-term E4M3 P %s, BF16 + "
                "residual partials %s, FP16 partials %s (partial pushes %s)",
                rows,
                p_hilo,
                hilo,
                f16,
                o_push,
            )
        if out is None:
            out = torch.empty(
                rows, self.hq, HEAD_DIM, dtype=torch.bfloat16, device=self.device
            )
        self._merge_h(
            q,
            k_cache,
            v_cache,
            page_table,
            seq_lens,
            q_len,
            rows,
            bmm1_scale,
            bmm2_scale,
            out,
            {**gate, "thr": 0},
            pdl=True,
            early=early or hilo_early,
            plain=plain,
            gate_epilogue=epilogue,
            plain_wait=plain_wait,
            push_rot=pushed,
            nopush=o_push,
            hilo=hilo,
            f16=f16,
            gate_bf16=gate_bf16,
            trig=hilo_trig,
        )
        if gate_bf16 is not None:
            self._gate_bf16_done = True
            if ("gate_bf16", rows) not in self._logged:
                self._logged.add(("gate_bf16", rows))
                logger.info(
                    "NVIDIA megakernel: Qwen3.5 slot draft_gate_fold: CP-lite's precise "
                    "merge applies the attention output gate at %d rows",
                    rows,
                )
        if epilogue is not None:
            self._gate_epilogue_done = True
            if ("gate_epilogue", rows) not in self._logged:
                self._logged.add(("gate_epilogue", rows))
                logger.info(
                    "TRT-L1 CP-lite merge writes the gated FP8 o_proj input at %d rows",
                    rows,
                )
        return out

    def _merge_h(
        self,
        q,
        k_cache,
        v_cache,
        page_table,
        seq_lens,
        q_len,
        rows,
        bmm1_scale,
        bmm2_scale,
        out,
        gate,
        pdl=None,
        early=False,
        plain=False,
        gate_epilogue=None,
        plain_wait=False,
        push_rot=False,
        nopush=False,
        hilo=False,
        f16=False,
        gate_bf16=None,
        trig=False,
    ):
        pdl = self.cp_pdl if pdl is None else pdl
        hq = self.hq
        epilogue_args = (
            {}
            if gate_epilogue is None
            else {
                "gate": gate_epilogue[0],
                "gate_stride": gate_epilogue[0].stride(0),
                "o_proj_input_scale": gate_epilogue[1],
                "o_proj_input": gate_epilogue[2],
                "o_proj_input_stride": gate_epilogue[2].stride(0),
                "GATE_EPILOGUE": True,
            }
        )
        merge_kernel = _cp_merge_h_ll_plain if plain else _cp_merge_h_ll
        merge_options = {} if plain else {"EARLY": early and pdl, **epilogue_args}
        if (self.merge_cg or nopush) and early and pdl and not plain:
            merge_options["CG"] = True
        if nopush and not early:
            assert hilo and pdl and plain_wait and not plain, (
                "NOPUSH after the wait: the precise call's per-head merge"
            )
            merge_options["NOPUSH"] = True
        elif nopush:
            assert early and pdl and plain_wait and not plain, (
                "NOPUSH: the early plain-wait merge"
            )
            merge_options.update(NOPUSH=True, MSLEEP=_PUSHED_MERGE_SLEEP_NS)
        if hilo:
            assert not plain and 2 * rows <= _MAX_ROWS, (
                "HILO: the per-head merge, rows <= _MAX_ROWS / 2"
            )
            merge_options["HILO"] = True
        if f16:
            assert not plain and not hilo, "F16: the per-head merge, no residual words"
            merge_options["F16"] = True
        if trig:
            assert hilo and early and nopush and pdl and not plain, (
                "TRIG: the EARLY pushed precise merge"
            )
            merge_options["TRIG"] = True
        if gate_bf16 is not None:
            assert hilo and (not early or nopush) and gate_epilogue is None, (
                "GATE_BF16: the precise call's per-head merge, the gate after its wait"
            )
            merge_options.update(
                gate=gate_bf16, gate_stride=gate_bf16.stride(0), GATE_BF16=True
            )
        if plain_wait and not plain:
            assert HEAD_DIM // 2 == 4 * 32, (
                "PLAIN_WAIT needs one LL word a merge thread"
            )
            merge_options["PLAIN_WAIT"] = True
            if push_rot:
                # slot cp_q_push's call: the sources start their pushes on different members
                merge_options["PUSH_ROT"] = True
            if ("plain_wait", rows) not in self._logged:
                self._logged.add(("plain_wait", rows))
                logger.info(
                    "NVIDIA megakernel: Qwen3.5 slot cp_merge_plain_wait polls "
                    "CP-lite's merge with the plain wait at %d rows",
                    rows,
                )
        merge_kernel[(rows, hq)](
            self.po_x.view(torch.int32),
            self.pl_x.view(torch.int32),
            hq * HEAD_DIM // 2,
            self.po_x.stride(0) // 2,
            hq,
            self.pl_x.stride(0),
            q,
            q.stride(0),
            k_cache,
            v_cache,
            k_cache.stride(0),
            k_cache.stride(2),
            page_table,
            page_table.stride(0),
            seq_lens,
            out,
            out.stride(0),
            self.peers,
            self.me,
            self.llh_off,
            self.cnt_h,
            bmm1_scale,
            bmm2_scale,
            gate["nb"],
            gate["thr"],
            QL=q_len,
            HQ=hq,
            CP=self.cp,
            D=HEAD_DIM,
            PG=PAGE,
            NWP=_LL_PWORDS,
            NB=gate["NB"],
            GMAX=gate["GMAX"],
            WAIT=pdl,
            num_warps=4,
            launch_pdl=pdl,
            **merge_options,
            **_dl_args(),
        )

    def _own_workspace(self, ncols: int = 0, q_len: int = 1):
        capturing = (
            self.device.type == "cuda" and torch.cuda.is_current_stream_capturing()
        )
        need = self._fold_ws_bytes(ncols, q_len)
        if self._ws is None or self._ws.numel() < need:
            if self._ws is not None:
                if capturing or self._ws_in_graph:
                    self._ws_captured.append(self._ws)
                elif self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
            self._ws = torch.empty(need, dtype=torch.uint8, device=self.device)
            self._ws_in_graph = False
            logger.info(
                "CP-lite own trtllm-gen workspace: %.1f MiB (%d owned pages a request, q_len %d)",
                need / (1 << 20),
                ncols,
                q_len,
            )
        self._ws_in_graph |= capturing
        return self._ws

    def _fold_ws_bytes(self, ncols: int, q_len: int) -> int:
        spec = self.fold_chunk_q1 if q_len == 1 else self.fold_chunk
        key = (ncols, q_len, tuple(sorted(spec.items())))
        if key not in self._ws_bytes:
            rows = 0
            for b in range(1, _MAX_ROWS // q_len + 1):
                c = self._chunk_cols(b, q_len)
                k = ncols // c if c and ncols > c and ncols % c == 0 else 1
                rows = max(rows, b * k)
            heads = q_len * self.cp * self.hq
            self._ws_bytes[key] = _lse_workspace_bytes(rows, heads, self.sms)
        return self._ws_bytes[key]

    def _chunk_cols(self, b: int, q_len: int) -> int:
        spec = self.fold_chunk_q1 if q_len == 1 else self.fold_chunk
        keys = [k for k in spec if k <= b]
        return spec[max(keys)] if keys else 0

    def _fold_partial(
        self,
        k_cache,
        v_cache,
        page_table,
        seq_lens,
        q_len,
        b,
        rows,
        bmm1_scale,
        bmm2_scale,
        forward_batch,
        layer_id,
        gate,
        workspace,
        fused=None,
    ):
        """trtllm-gen over this rank's owned prefix pages for every group head, the q_len rows folded into the
        heads (q_len_per_req 1), into po / plse with LSE."""
        import flashinfer

        if fused is None:
            tab, lens, max_kv = self._tables(
                page_table, seq_lens, q_len, forward_batch, layer_id, gate
            )
        else:
            tab, lens, max_kv = fused[4]
        h = q_len * self.cp * self.hq
        query = self.q_all[:rows].view(b, h, HEAD_DIM)
        out = self.po[:rows].view(b, h, HEAD_DIM)
        lse = self.plse[:rows].view(b, h)
        chunk = (
            self._chunk_tables(tab, lens, b, forward_batch, layer_id, q_len)
            if fused is None
            else fused[0]
        )
        flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=query,
            kv_cache=(k_cache, v_cache),
            workspace_buffer=workspace,
            block_tables=tab,
            seq_lens=lens if chunk is None else chunk[0],
            max_seq_len=max_kv,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            out=out,
            lse=lse,
            q_len_per_req=1,
            multi_ctas_kv_counter_buffer=self.counter,
        )
        if chunk is None:
            return
        # ragged batch: the call above got no key; the chunk rows (empty otherwise) and the combine
        _, vtab, vlens, vstart, vcnt = chunk
        nv, C = vtab.shape
        n = h * HEAD_DIM // 4
        if fused is None:
            qv = torch.empty(nv, h, HEAD_DIM, dtype=query.dtype, device=self.device)
            po_v = torch.empty(
                nv, h, HEAD_DIM, dtype=torch.bfloat16, device=self.device
            )
            pl_v = torch.empty(nv, h, dtype=torch.float32, device=self.device)
            _cp_chunk_q[(b, nv // b)](
                query.view(torch.int32),
                qv.view(torch.int32),
                vstart,
                vcnt,
                n,
                BLK=1024,
            )
        else:  # the gather wrote the chunk rows' queries; the merge combines their partials
            qv, po_v, pl_v = fused[1:4]
        flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=qv,
            kv_cache=(k_cache, v_cache),
            workspace_buffer=self._own_workspace(tab.shape[1], q_len),
            block_tables=vtab,
            seq_lens=vlens,
            max_seq_len=C * PAGE,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            out=po_v,
            lse=pl_v,
            q_len_per_req=1,
            multi_ctas_kv_counter_buffer=self.counter,
        )
        if fused is None:
            _cp_chunk_comb[(b, triton.cdiv(h, 16))](
                po_v, pl_v, vstart, vcnt, out, lse, h, HB=16, D=HEAD_DIM
            )

    def _chunk_tables(self, tab, lens, b, forward_batch, layer_id, q_len=7):
        """Build the one-row and chunk-row tables once per forward.

        Draft calls use their own chunk policy. Ragged batches clear the one-row
        lengths and use dense ``[B * K, C]`` chunk rows instead.
        """
        C = self._chunk_cols(b, q_len)
        ncols = tab.shape[1]
        if not C or ncols <= C or ncols % C:
            return None
        key = ("chunk", tab.data_ptr(), lens.data_ptr(), b)
        ent = self._tabs.get(key)
        if ent is not None and ent[1]() is forward_batch and layer_id > ent[2]:
            ent[2] = layer_id
            return ent[0]
        K = ncols // C
        nv = b * K
        base_lens = torch.empty(b, dtype=torch.int32, device=self.device)
        vtab = torch.empty(nv, C, dtype=torch.int32, device=self.device)
        vlens = torch.empty(nv, dtype=torch.int32, device=self.device)
        vstart = torch.empty(b, dtype=torch.int32, device=self.device)
        vcnt = torch.empty(b, dtype=torch.int32, device=self.device)
        _cp_chunk_tables[(b, K)](
            tab,
            tab.stride(0),
            lens,
            base_lens,
            vtab,
            vlens,
            vstart,
            vcnt,
            b,
            nv,
            float(self.fold_ragged),
            C=C,
            KP=K,
            NB=triton.next_power_of_2(b),
            PG=PAGE,
        )
        out = (base_lens, vtab, vlens, vstart, vcnt)
        self._tabs[key] = [out, weakref.ref(forward_batch), layer_id]
        if (b, "chunk") not in self._logged:
            self._logged.add((b, "chunk"))
            logger.info(
                "CP-lite fold partial: %d requests, chunk rows of %d pages (%d rows) when the longest "
                "owned length x B >= %.2f x the total",
                b,
                C,
                nv,
                self.fold_ragged,
            )
        return out

    def _fold_merge(
        self,
        q,
        k_cache,
        v_cache,
        page_table,
        seq_lens,
        q_len,
        rows,
        bmm1_scale,
        bmm2_scale,
        out,
        gate,
    ):
        gh_w = self.cp * self.hq * HEAD_DIM // 2
        _cp_a2a_merge_ll[(rows,)](
            self.po.view(torch.int32),
            self.plse.view(torch.int32),
            gh_w,
            self.hq * HEAD_DIM // 2,
            self.cp * self.hq,
            self.hq,
            q,
            q.stride(0),
            k_cache,
            v_cache,
            k_cache.stride(0),
            k_cache.stride(2),
            page_table,
            page_table.stride(0),
            seq_lens,
            out,
            out.stride(0),
            self.peers,
            self.me,
            self.llp_off,
            self.cnt[1],
            bmm1_scale,
            bmm2_scale,
            **gate,
            QL=q_len,
            HQ=self.hq,
            CP=self.cp,
            D=HEAD_DIM,
            PG=PAGE,
            NWP=_LL_PWORDS,
            num_warps=8,
            **_dl_args(),
        )
        return out

    def _fold_merge_h(
        self,
        q,
        k_cache,
        v_cache,
        page_table,
        seq_lens,
        q_len,
        rows,
        bmm1_scale,
        bmm2_scale,
        out,
        gate,
        fused=None,
        plain=False,
    ):
        """Merge the fold layout with one program per row and query head."""
        hq = self.hq
        merge_kernel = _cp_merge_h_ll_plain if plain else _cp_merge_h_ll
        merge_kernel[(rows, hq)](
            self.po.view(torch.int32),
            self.plse.view(torch.int32),
            self.cp * hq * HEAD_DIM // 2,
            hq * HEAD_DIM // 2,
            self.cp * hq,
            hq,
            q,
            q.stride(0),
            k_cache,
            v_cache,
            k_cache.stride(0),
            k_cache.stride(2),
            page_table,
            page_table.stride(0),
            seq_lens,
            out,
            out.stride(0),
            self.peers,
            self.me,
            self.llh_off,
            self.cnt_h,
            bmm1_scale,
            bmm2_scale,
            gate["nb"],
            gate["thr"],
            QL=q_len,
            HQ=hq,
            CP=self.cp,
            D=HEAD_DIM,
            PG=PAGE,
            NWP=_LL_PWORDS,
            NB=gate["NB"],
            GMAX=gate["GMAX"],
            WAIT=False,
            num_warps=4,
            **(
                {}
                if fused is None
                else {
                    "po_v": fused[2].view(torch.int32),
                    "pl_v": fused[3],
                    "vstart": fused[0][3],
                    "vcnt": fused[0][4],
                    "CHUNK": True,
                }
            ),
            **_dl_args(),
        )
        return out

    def _exact_partial(
        self,
        k_cache,
        v_cache,
        page_table,
        pre_lens,
        q_len,
        b,
        rows,
        bmm1_scale,
        bmm2_scale,
        pdl=False,
        len_sub=0,
        early_trigger=False,
        o_push=False,
        o_hilo=False,
        p_hilo=False,
        o_f16=False,
        late_trigger=True,
    ):
        from sgl_nvidia_mega.qwen3_5.sm103.attention import q35_cp_fused as F

        cp, hq = self.cp, self.hq
        push = {}
        if o_push:
            push = dict(
                o_push=True,
                ll=self._ll_tab(),
                cnt_h=self.cnt_h,
                ql_real=q_len,
                hq_real=hq,
                late_trigger=bool(late_trigger),
            )
        F.q35_cp_fused_partial(
            self.q_all[:rows].view(b, cp, q_len * hq, HEAD_DIM).permute(1, 0, 2, 3),
            k_cache,
            page_table,
            pre_lens,
            1,
            self.me,
            cp,
            bmm1_scale,
            bmm2_scale,
            self.po_x[:, :rows].view(cp, b, q_len * hq, HEAD_DIM),
            self.pl_x[:, :rows].view(cp, b, q_len * hq),
            sm_count=self.sms,
            enable_pdl=pdl,
            v_pool=v_cache,
            p_shift=_EXACT_P_SHIFT,
            rescale_threshold=_EXACT_RESCALE_THRESHOLD,
            len_sub=len_sub,
            early_trigger=early_trigger,
            **push,
            **({"o_hilo": True} if o_hilo else {}),
            **({"p_hilo": True} if p_hilo else {}),
            **({"o_f16": True} if o_f16 else {}),
        )

    def _ll_tab(self) -> torch.Tensor:
        if self._ll_tab_t is None:
            assert self.cp <= 4
            t = torch.zeros(4, dtype=torch.int64, device=self.device)
            t[: self.cp] = self.peers + self.llh_off
            self._ll_tab_t = t
        return self._ll_tab_t

    def _exact_merge(
        self,
        q,
        k_cache,
        v_cache,
        page_table,
        seq_lens,
        q_len,
        rows,
        bmm1_scale,
        bmm2_scale,
        out,
        gate,
    ):
        hq = self.hq
        if out is None:
            out = torch.empty(
                rows, hq, HEAD_DIM, dtype=torch.bfloat16, device=self.device
            )
        _cp_a2a_merge_ll[(rows,)](
            self.po_x.view(torch.int32),
            self.pl_x.view(torch.int32),
            hq * HEAD_DIM // 2,
            self.po_x.stride(0) // 2,
            hq,
            self.pl_x.stride(0),
            q,
            q.stride(0),
            k_cache,
            v_cache,
            k_cache.stride(0),
            k_cache.stride(2),
            page_table,
            page_table.stride(0),
            seq_lens,
            out,
            out.stride(0),
            self.peers,
            self.me,
            self.llp_off,
            self.cnt[1],
            bmm1_scale,
            bmm2_scale,
            **gate,
            QL=q_len,
            HQ=hq,
            CP=self.cp,
            D=HEAD_DIM,
            PG=PAGE,
            NWP=_LL_PWORDS,
            num_warps=8,
            **_dl_args(),
        )
        return out

    def _fused(
        self,
        k_cache,
        v_cache,
        page_table,
        seq_lens,
        q_len,
        b,
        rows,
        bmm1_scale,
        bmm2_scale,
    ):
        """The fused partial (TRT-LLM's kernel) on the gathered queries, then the 4-way merge; None where the kernel
        does not take the shape (decided by shapes only)."""
        from sgl_nvidia_mega.qwen3_5.sm103.attention import q35_cp_fused as F

        if not F.q35_cp_fused_supported(q_len, self.cp, self.hq, b, self.sms):
            return None
        cp, hq = self.cp, self.hq
        F.q35_cp_fused_partial(
            self.q_all[:rows].view(rows, cp, hq, HEAD_DIM).permute(1, 0, 2, 3),
            k_cache,
            page_table,
            seq_lens,
            q_len,
            self.me,
            cp,
            bmm1_scale,
            bmm2_scale,
            self.po[:rows].view(rows, cp, hq, HEAD_DIM).permute(1, 0, 2, 3),
            self.plse[:rows].view(rows, cp, hq).permute(1, 0, 2),
            sm_count=self.sms,
            enable_pdl=False,
            v_pool=v_cache,
        )
        out = torch.empty(rows, hq, HEAD_DIM, dtype=torch.bfloat16, device=self.device)
        _cp_a2a_merge4_ll[(rows,)](
            self.po.view(torch.int32),
            self.plse.view(torch.int32),
            out,
            out.stride(0),
            self.peers,
            self.me,
            self.llp_off,
            self.cnt[1],
            HQ=hq,
            CP=cp,
            D=HEAD_DIM,
            NWP=_LL_PWORDS,
            num_warps=8,
            **_dl_args(),
        )
        return out

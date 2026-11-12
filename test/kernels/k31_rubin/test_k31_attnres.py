# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Seven-CTA cluster AttnRes (``attn_res.prepared.PreparedAttnRes``) at the K3.1 shapes
(H = 7168, up to 8 saved blocks, rows 1-8/16/32/64), against the independent FP32 torch
reference ``_ref_attnkda.attnres`` (stock ``sglang.srt.layers.attn_residual`` contract).

Variants (each at every row count, sharded over the node's GPUs):
- ``layer0``: no saved blocks, no delta (embeddings), bank row 0 written.
- ``nb1``, ``nb3w`` (block-write layer: the new prefix also goes to bank row 3), ``nb8``.
- ``final``: the output AttnRes over 8 blocks.
- ``ring_*``: ``run_ring`` reading the MoE tail's out mailbox [1, 64, 7168] (Lamport,
  sentinel word 0x80000000), with +0.0 values present; the read rows must be re-armed to
  the sentinel, the rest untouched. ``ring_pre``: rows published before the launch (no
  wait). ``ring_side``: published by a copy kernel on another stream launched 10 ms (host
  time) after the consumer, which is spinning by then.
  ``ring_pdl``: the serving pattern: a PDL predecessor kernel on the same stream releases
  its dependents at entry, waits ~200 us, then publishes the rows with st.relaxed.sys (the
  unicast form of the MoE tail's multimem.st publish); the consumer launches early and polls.
- ``aux`` / ``auxmix``: ``bind_aux`` copies of the updated prefix, and the stock FP32-cw
  pre-norm mixture (``AuxMixture``) for DFlash aux capture.

``test_attnres_fold`` (``K31MK_AUX_FOLD=1``: the aux mixture computed inside ClusterAttnRes instead
of AuxMixture's score + combine kernels). A pure fold: unfused and fold runs on identical inputs
must agree BITWISE on the aux mixture, the normalized output, the new prefix and the bank, at the
serving capture shapes (nb 1, 2, 5, 6, 7, 8; the final AttnRes over 8 blocks; a block-write row at
index nb; a write inside rows 0..nb-1, which must keep the unfused kernels; a Lamport ring input).
The independent FP32 reference checks above apply too. The aux mixture's agreement with stock
Triton ``_mix_fused`` (host tree via ``K31_HOST_PYTHON``) is recorded, not asserted.

Tolerances (stated before any run):
- new prefix, written bank row, aux prefix copy: bit-exact (both compute bf16(fp32 add)).
- normalized output and aux mixture: |out - ref| <= 1e-3 + 2^-7 |ref| per cell. The
  kernel works in FP32 (fast exp/rsqrt/rcp, cluster-order sums) and rounds once to BF16,
  so it differs from the FP32 reference by one BF16 rounding (2^-9) plus ~1e-5 relative.
- re-armed mailbox words: exactly 0x80000000; unread mailbox rows and other bank rows: exact.

Mutations (each must make the suite fail): ``score_rms`` (every block's score uses block
0's RMS), ``drop_delta`` (odd columns lose the delta in the prefix update), ``mix_weight``
(the new prefix enters the mixture without its softmax weight), ``out_rms`` (output RMS
over 3584 instead of 7168 columns), ``no_rearm`` (the ring is not re-armed), ``aux_row``
(the aux mixture uses row 0's weight for every row). Fold mutations: ``fold_lane`` (the
odd score warp's sum lands in the even slot), ``fold_cw`` (FP32 combined weight without the
query factor), ``fold_chunk`` (chunk sums added in reverse CTA order), ``fold_tree`` (8-warp
tree in another order), ``fold_prefix`` (the prefix row takes block 0's score), ``fold_carry``
(the prefix row's words mixed in for every row), ``fold_guard`` (fold despite a block write
inside rows 0..nb-1), ``fold_extra`` (the prefix row's dot slot carries its sum of squares),
``fold_butterfly`` (one butterfly step skipped), ``fold_overlap_drop`` (the overlapped variant,
K31MK_AUX_FOLD=2, never computes the mixture).
"""

import os
import time

import pytest
import torch

import _harness as H
import _ref_attnkda as R

MUTATIONS = {
    "score_rms": (
        "kimi_k31/k3/cutedsl/attn_res/kernel.py",
        "logits[b] = total[0, self.nb + b] * rstd(total[0, b], eps)",
        "logits[b] = total[0, self.nb + b] * rstd(total[0, 0], eps)",
    ),
    "drop_delta": (
        "kimi_k31/k3/cutedsl/attn_res/kernel.py",
        "lo, hi = v[OLD, 2 * j] + lo, v[OLD, 2 * j + 1] + hi",
        "lo, hi = v[OLD, 2 * j] + lo, v[OLD, 2 * j + 1]",
    ),
    "mix_weight": (
        "kimi_k31/k3/cutedsl/attn_res/kernel.py",
        "result[0, i] = fma(alpha, mixture[0, i], ep * v[NEW, i])",
        "result[0, i] = fma(alpha, mixture[0, i], v[NEW, i])",
    ),
    "out_rms": (
        "kimi_k31/k3/cutedsl/attn_res/kernel.py",
        "return cute.math.rsqrt(fma(sum_sq, Float32(1 / 7168), eps), fastmath=True)",
        "return cute.math.rsqrt(fma(sum_sq, Float32(1 / 3584), eps), fastmath=True)",
    ),
    "no_rearm": (
        "kimi_k31/k3/cutedsl/attn_res/kernel.py",
        "rearm(base + slot * delta.shape[1] * 7168)",
        "None",
    ),
    "aux_row": (
        "kimi_k31/k3/cutedsl/attn_res/aux.py",
        "acc[2 * j] = fma(weights[row], bf16(word), acc[2 * j])",
        "acc[2 * j] = fma(weights[0], bf16(word), acc[2 * j])",
    ),
    "fold_lane": (
        "kimi_k31/k3/cutedsl/attn_res/kernel.py",
        "red[0, row, 2 * warp + lane // 16] = s",
        "red[0, row, 2 * warp] = s",
    ),
    "fold_cw": (
        "kimi_k31/k3/cutedsl/attn_res/kernel.py",
        "return [aux_mul(bf16(nw[i // 2], i % 2), bf16(qw[i // 2], i % 2)) for i in range(8)]",
        "return [bf16(nw[i // 2], i % 2) for i in range(8)]",
    ),
    "fold_chunk": (
        "kimi_k31/k3/cutedsl/attn_res/kernel.py",
        "total = add_rn(total, buf[peer * 16 + index])",
        "total = add_rn(total, buf[(6 - peer) * 16 + index])",
    ),
    "fold_tree": (
        "kimi_k31/k3/cutedsl/attn_res/kernel.py",
        "left = add_rn(add_rn(r[0], r[4]), add_rn(r[2], r[6]))",
        "left = add_rn(add_rn(r[0], r[2]), add_rn(r[4], r[6]))",
    ),
    "fold_prefix": (
        "kimi_k31/k3/cutedsl/attn_res/kernel.py",
        "            raw = prefix_score\n",
        "            raw = scores[0]\n",
    ),
    "fold_carry": (
        "kimi_k31/k3/cutedsl/attn_res/kernel.py",
        "                    word = blockw[row, j]\n",
        "                    word = prefix_words[j]\n",
    ),
    "fold_guard": (
        "kimi_k31/k3/cutedsl/attn_res/kernel.py",
        "and nb > 0 and not 0 <= write < nb)",
        "and nb > 0)",
    ),
    "fold_extra": (
        "kimi_k31/k3/cutedsl/attn_res/kernel.py",
        "value = aux_tree(bufs[4][0], tid - count, self.nb)",
        "value = aux_tree(bufs[4][0], 0, self.nb)",
    ),
    "fold_overlap_drop": (
        "kimi_k31/k3/cutedsl/attn_res/kernel.py",
        "        if cutlass.const_expr(aux_args is not None):\n            self.aux_mix(*aux_args)\n",
        "        if cutlass.const_expr(False):\n            self.aux_mix(*aux_args)\n",
    ),
    "fold_butterfly": (
        "kimi_k31/k3/cutedsl/attn_res/kernel.py",
        "        for step in cutlass.range_constexpr(4):\n            for b in cutlass.range_constexpr(2):",
        "        for step in cutlass.range_constexpr(3):\n            for b in cutlass.range_constexpr(2):",
    ),
}

ROWS = (1, 2, 3, 4, 8, 16, 32, 64)
# (name, nb, has_delta, write, final)
VARIANTS = {
    "layer0": (0, False, 0, False),
    "nb1": (1, True, -1, False),
    "nb3w": (3, True, 3, False),
    "nb8": (8, True, -1, False),
    "final": (8, True, -1, True),
    "ring_pre": (4, True, -1, False),
    "ring_pdl": (4, True, -1, False),
    "aux": (3, True, 3, False),
    "auxmix": (3, True, -1, False),
    "auxmix0": (0, False, 0, False),
    # Last in each shard: a deadline fail-stop poisons the process's CUDA context.
    "ring_side": (4, True, -1, False),
}
EPS = 1e-6
DEADLINE_NS = 4_000_000_000  # a missed publish traps after 4 s instead of hanging
SENTINEL = -0x80000000  # int32 view of 0x80000000

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU",
)


class _NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _norm(g, scale=0.2):
    return _NS(weight=(1.0 + H.randn((7168,), scale, g)).to(torch.bfloat16), variance_epsilon=EPS)


def _target(g, nb, write, final):
    if final:
        return _NS(
            output_attn_res_norm=_norm(g),
            output_attn_res_proj=_NS(weight=H.randn((1, 7168), 0.02, g)),
            norm=_norm(g),
            num_attn_res_blocks=nb,
        )
    return _NS(
        self_attention_res_norm=_norm(g),
        self_attention_res_proj=_NS(weight=H.randn((1, 7168), 0.02, g)),
        input_layernorm=_norm(g),
        prev_valid_blocks=nb,
        is_block_write_layer=write >= 0,
        block_write_idx=max(write, 0),
    )


def _weights(t, final):
    if final:
        return t.output_attn_res_norm, t.output_attn_res_proj, t.norm
    return t.self_attention_res_norm, t.self_attention_res_proj, t.input_layernorm


def _canonical(x):
    """Producers canonicalize -0.0 to +0.0 (BF16 0x8000 can form the sentinel)."""
    x = x.clone()
    x.view(torch.int16)[x.view(torch.int16) == -0x8000] = 0
    return x


_PUBLISHERS = {}


def _publisher(delay_ns):
    """A PDL primary that releases its dependents at entry, waits ``delay_ns`` on
    %globaltimer, then stores rows src[t] into mailbox[0, t] (st.relaxed.sys, 16 B)."""
    if delay_ns in _PUBLISHERS:
        return _PUBLISHERS[delay_ns]
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int64
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.primitives import at, global_timer_ns, launch, load_words, store_words

    class DelayedPublisher:
        @cute.jit
        def __call__(self, src: cute.Tensor, dst: cute.Tensor, stream):
            # Not a PDL secondary itself: back-to-back pairs then serialize at each
            # publisher, so consumer k+1 cannot launch (and poll the slot) before consumer k
            # has re-armed it; within a pair the consumer still launches early (PDL).
            launch(self.kernel, (src, dst, stream), src.shape[0], 128, pdl=False)

        @cute.kernel
        def kernel(self, args):
            src, dst = args
            token, _, _ = cute.arch.block_idx()
            tid, _, _ = cute.arch.thread_idx()
            cute.arch.griddepcontrol_launch_dependents()
            # Publish only after the predecessor (e.g. the previous consumer, which
            # re-arms this slot) completes, as the MoE tail does.
            cute.arch.griddepcontrol_wait()
            start = global_timer_ns()
            now = global_timer_ns()
            while now - start < Int64(delay_ns):
                now = global_timer_ns()
            for k in cutlass.range_constexpr(7):
                col = (k * 128 + tid) * 8
                words = load_words(at(src, token, col), count=4, mutable=True)
                store_words(at(dst, 0, token, col), words, scope="sys")

    _PUBLISHERS[delay_ns] = DelayedPublisher()
    return _PUBLISHERS[delay_ns]


_PUBLISH_CODE = {}


def _publish_fn(delay_ns, src, mailbox):
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.host import stream, to_cute

    views = (to_cute(src), to_cute(mailbox))
    key = (delay_ns, tuple(src.shape))
    if key not in _PUBLISH_CODE:
        _PUBLISH_CODE[key] = cute.compile(_publisher(delay_ns), *views, make_fake_stream())
    code = _PUBLISH_CODE[key]
    return lambda: code(*views, stream(src.device))


def _inputs(g, rows, nb):
    delta = _canonical(H.randn((rows, 7168), 1.0, g))
    delta[:, 5::97] = 0.0  # +0.0 halves next to nonzero halves in published words
    prefix = H.randn((rows, 7168), 2.0, g)
    blocks = H.randn((rows, 8, 7168), 2.0, g)
    return delta, prefix, blocks


@needs_gpu
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_attnres(variant, rows):
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.attn_res.prepared import PreparedAttnRes

    nb, has_delta, write, final = VARIANTS[variant]
    g = H.generator(2000 + rows * 31 + sorted(VARIANTS).index(variant))
    target = _target(g, nb, write, final)
    prep = PreparedAttnRes.for_target(target, deadline_ns=DEADLINE_NS, final=final, has_delta=has_delta)
    norm, query, out_norm = _weights(target, final)
    delta, prefix0, blocks0 = _inputs(g, rows, nb)
    ref_out, ref_prefix, ref_blocks = R.attnres(
        delta, prefix0, blocks0, nb=nb, has_delta=has_delta, write=write,
        norm_w=norm.weight, query_w=query.weight, out_w=out_norm.weight, eps=EPS, out_eps=EPS,
    )
    prefix, blocks = prefix0.clone(), blocks0.clone()
    out = torch.empty((rows, 7168), dtype=torch.bfloat16, device="cuda")
    H.poison(out)
    item = f"attnres.{variant}.b{rows}"
    aux = None
    mailbox = None
    if variant.startswith("ring"):
        mailbox = torch.empty((1, 64, 7168), dtype=torch.bfloat16, device="cuda")
        mailbox.view(torch.int32).fill_(SENTINEL)
        sentinel_rows = mailbox[0, rows:].clone()
        prep.bind_ring(mailbox)
        if variant == "ring_pre":
            mailbox[0, :rows].copy_(delta)
            prep.run_ring(prefix, blocks, out)
        elif variant == "ring_pdl":
            _publish_fn(200_000, delta, mailbox)()
            prep.run_ring(prefix, blocks, out)  # PDL secondary: starts early, polls
        else:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            prep.run_ring(prefix, blocks, out)  # polls the armed mailbox
            time.sleep(0.01)  # host-side delay: the consumer is spinning by now
            with torch.cuda.stream(side):
                mailbox[0, :rows].copy_(delta)
        torch.cuda.synchronize()
        words = mailbox[0, :rows].view(torch.int32)
        assert bool((words == SENTINEL).all()), f"{item}: {int((words != SENTINEL).sum())} words not re-armed"
        assert torch.equal(mailbox[0, rows:].view(torch.int32), sentinel_rows.view(torch.int32))
    elif variant.startswith("aux"):
        prep.bind_aux(mixture=variant.startswith("auxmix"))
        aux = torch.empty((rows, 7168), dtype=torch.bfloat16, device="cuda")
        H.poison(aux)
        prep.run(delta, prefix, blocks, out, aux=aux)
    else:
        prep.run(delta, prefix, blocks, out)
    torch.cuda.synchronize()
    code = prep.ring[0] if variant.startswith("ring") else (prep.aux[0] if aux is not None else prep.code)
    try:
        H.record({"kind": "resources", "item": item, **H.function_resources(code), **H.device_facts()})
    except Exception as e:  # aux+mixture modules hold two kernels
        H.record({"kind": "resources", "item": item, "error": repr(e)[:200]})
    assert torch.equal(prefix.view(torch.int16), ref_prefix.view(torch.int16)), f"{item}: prefix not bit-exact"
    assert torch.equal(blocks.view(torch.int16), ref_blocks.view(torch.int16)), f"{item}: bank not bit-exact"
    H.check(item, out, ref_out, 1e-3, 2**-7)
    if aux is not None:
        if variant == "aux":
            assert torch.equal(aux.view(torch.int16), ref_prefix.view(torch.int16)), f"{item}: aux prefix copy"
        else:
            ref_aux = R.attnres_aux_mixture(ref_prefix, ref_blocks, nb=nb, norm_w=norm.weight, query_w=query.weight, eps=EPS)
            H.check(item + ".mixture", aux, ref_aux, 1e-3, 2**-7)

    if os.environ.get("K31_TIME") == "1" and not os.environ.get("K31_MUTATION") and variant == "ring_pdl":
        # Per call: a zero-delay PDL publisher, then the consumer; minus the publisher alone.
        copies = 32
        bufs = [(_inputs(g, rows, nb), torch.empty((rows, 7168), dtype=torch.bfloat16, device="cuda")) for _ in range(copies)]
        pubs = [_publish_fn(0, b[0][0], mailbox) for b in bufs]
        both = [lambda b=b, p=p: (p(), prep.run_ring(b[0][1], b[0][2], b[1])) for b, p in zip(bufs, pubs)]
        tp, tb = H.cold_time_us(pubs), H.cold_time_us(both)
        nbytes = rows * 7168 * 2 * (nb + 4) + 3 * 7168 * 2
        t = {"us_median": tb["us_median"] - tp["us_median"], "us_publisher": tp["us_median"], "us_pair": tb["us_median"],
             "calls_per_graph": copies, "reps": tb["reps"]}
        fl = H.floor_us(nbytes)
        H.record({"kind": "time", "item": item, **t, **fl})
        print("TIME", item, t, fl, flush=True)
    if os.environ.get("K31_TIME") == "1" and not os.environ.get("K31_MUTATION") and variant == "ring_pre":
        # Per call: publish the rows (a copy kernel), then the consumer; minus the copy alone.
        copies = 32
        bufs = [(_inputs(g, rows, nb), torch.empty((rows, 7168), dtype=torch.bfloat16, device="cuda")) for _ in range(copies)]
        pub = [lambda b=b: mailbox[0, :rows].copy_(b[0][0]) for b in bufs]
        both = [lambda b=b: (mailbox[0, :rows].copy_(b[0][0]), prep.run_ring(b[0][1], b[0][2], b[1])) for b in bufs]
        tp, tb = H.cold_time_us(pub), H.cold_time_us(both)
        nbytes = rows * 7168 * 2 * (nb + 4) + 3 * 7168 * 2
        t = {"us_median": tb["us_median"] - tp["us_median"], "us_publish_copy": tp["us_median"], "us_with_copy": tb["us_median"],
             "calls_per_graph": copies, "reps": tb["reps"]}
        fl = H.floor_us(nbytes)
        H.record({"kind": "time", "item": item, **t, **fl})
        print("TIME", item, t, fl, flush=True)
    if os.environ.get("K31_TIME") == "1" and not os.environ.get("K31_MUTATION") and not variant.startswith("ring"):
        nbytes = rows * 7168 * 2 * (nb + 4 + (write >= 0) + (aux is not None)) + 3 * 7168 * 2
        copies = H.copies_for(nbytes, cap=64)
        bufs = [(_inputs(g, rows, nb), torch.empty((rows, 7168), dtype=torch.bfloat16, device="cuda")) for _ in range(copies)]
        auxes = [torch.empty((rows, 7168), dtype=torch.bfloat16, device="cuda") if aux is not None else None for _ in range(copies)]
        calls = [lambda b=b, a=a: prep.run(b[0][0], b[0][1], b[0][2], b[1], aux=a) for b, a in zip(bufs, auxes)]
        t = H.cold_time_us(calls)
        fl = H.floor_us(nbytes)
        H.record({"kind": "time", "item": item, **t, **fl})
        print("TIME", item, t, fl, flush=True)


# (name, nb, has_delta, write, final, ring): the serving capture consumers (score grids [8, nb + 1]
# = nb 1, 2, 5, 6, 7 and the final AttnRes over 8 blocks), a block write at row nb, a write inside
# rows 0..nb-1 (the fold must fall back), and the Lamport ring input the serving path uses.
FOLD_VARIANTS = {
    "f1": (1, True, -1, False, False),
    "f2": (2, True, -1, False, False),
    "f5": (5, True, -1, False, False),
    "f6": (6, True, -1, False, False),
    "f7": (7, True, -1, False, False),
    "f8fin": (8, True, -1, True, False),
    "f3w3": (3, True, 3, False, False),
    "f3w1": (3, True, 1, False, False),
    "f5ring": (5, True, -1, False, True),
    "f8ring": (8, True, -1, False, True),
}


def _stock_mix(prefix, blocks, nb, norm, query):
    """Stock Triton ``_mix_fused`` from the host tree, or None when it is not importable."""
    try:
        from sglang.srt.layers.attn_residual import _mix_fused
    except Exception as e:  # no host tree on the path
        H.record({"kind": "stock_mix", "error": repr(e)[:200]})
        return None
    proj = _NS(weight=query.weight)
    return _mix_fused(prefix, blocks, nb, proj, norm)


def _prepared_aux(target, final, ring, fold):
    """fold: 0 unfused (AuxMixture kernels), 1 folded, 2 folded and overlapped (K31MK_AUX_FOLD)."""
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.attn_res.prepared import PreparedAttnRes

    prep = PreparedAttnRes.for_target(target, deadline_ns=DEADLINE_NS, final=final, has_delta=True)
    old = os.environ.get("K31MK_AUX_FOLD")
    os.environ["K31MK_AUX_FOLD"] = str(int(fold))
    try:
        if ring is not None:
            prep.bind_ring(ring)
        prep.bind_aux(mixture=True)
    finally:
        if old is None:
            os.environ.pop("K31MK_AUX_FOLD")
        else:
            os.environ["K31MK_AUX_FOLD"] = old
    return prep


@needs_gpu
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("variant", list(FOLD_VARIANTS))
def test_attnres_fold(variant, rows):
    H.load_plugin(MUTATIONS)
    nb, has_delta, write, final, ring = FOLD_VARIANTS[variant]
    g = H.generator(7000 + rows * 31 + sorted(FOLD_VARIANTS).index(variant))
    target = _target(g, nb, write, final)
    norm, query, out_norm = _weights(target, final)
    delta, prefix0, blocks0 = _inputs(g, rows, nb)
    mailbox = None
    if ring:
        mailbox = torch.empty((1, 64, 7168), dtype=torch.bfloat16, device="cuda")
        mailbox.view(torch.int32).fill_(SENTINEL)
    preps = {fold: _prepared_aux(target, final, mailbox, fold) for fold in (0, 1, 2)}
    got = {}
    for fold, prep in preps.items():
        prefix, blocks = prefix0.clone(), blocks0.clone()
        out = torch.empty((rows, 7168), dtype=torch.bfloat16, device="cuda")
        aux = torch.empty((rows, 7168), dtype=torch.bfloat16, device="cuda")
        H.poison(out, aux)
        if ring:
            mailbox[0, :rows].copy_(delta)
            prep.run_ring(prefix, blocks, out, aux=aux)
        else:
            prep.run(delta, prefix, blocks, out, aux=aux)
        torch.cuda.synchronize()
        if ring:
            assert bool((mailbox[0, :rows].view(torch.int32) == SENTINEL).all()), f"fold={fold}: ring not re-armed"
        got[fold] = (out, prefix, blocks, aux)
    item = f"attnres_fold.{variant}.b{rows}"
    ref_out, ref_prefix, ref_blocks = R.attnres(
        delta, prefix0, blocks0, nb=nb, has_delta=has_delta, write=write,
        norm_w=norm.weight, query_w=query.weight, out_w=out_norm.weight, eps=EPS, out_eps=EPS,
    )
    ref_aux = R.attnres_aux_mixture(ref_prefix, ref_blocks, nb=nb, norm_w=norm.weight, query_w=query.weight, eps=EPS)
    stock = _stock_mix(ref_prefix, ref_blocks, nb, norm, query)
    # A block write inside rows 0..nb-1 is outside ClusterAttnRes's contract (it reads the saved
    # blocks at launch): that variant only checks that the fold falls back to the unfused kernels.
    in_contract = not 0 <= write < nb
    for fold, (out, prefix, blocks, aux) in got.items():
        if not in_contract:
            continue
        tag = f"{item}.{('unfused', 'fold', 'fold2')[fold]}"
        assert torch.equal(prefix.view(torch.int16), ref_prefix.view(torch.int16)), f"{tag}: prefix not bit-exact"
        assert torch.equal(blocks.view(torch.int16), ref_blocks.view(torch.int16)), f"{tag}: bank not bit-exact"
        H.check(tag, out, ref_out, 1e-3, 2**-7)
        H.check(tag + ".mixture", aux, ref_aux, 1e-3, 2**-7)
        if stock is not None:
            same = (aux.view(torch.int16) == stock.view(torch.int16)).float().mean().item()
            H.record({"kind": "stock_mix", "item": tag, "bitwise_fraction": same})
            print("STOCK_MIX", tag, same, flush=True)
    names = ("output", "prefix", "bank", "aux mixture")
    for mode in (1, 2):
        for name, a, b in zip(names, got[0], got[mode]):
            diff = int((a.view(torch.int16) != b.view(torch.int16)).sum())
            H.record({"kind": "fold_bitwise", "item": f"{item}.fold{mode}.{name}", "differing": diff, "cells": a.numel()})
            assert diff == 0, f"{item}: fold mode {mode} {name} differs from unfused in {diff}/{a.numel()} cells"

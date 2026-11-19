# SPDX-License-Identifier: Apache-2.0
"""The PLE kernels runtime/ple.py runs, against SGLang main's stock code on a CUDA GPU.

Every one is claimed bitwise by the source: the target-verify layout
(``qwen4_ple_verify_prepare``) against main's ``_prepare_ple_batch`` torch branch with
the state slot taken from the request table (the source's rule), the one-launch verify
conv against main's ``fused_qwen4_verify_conv`` (prepare + F.conv1d + finish, with the
intermediate windows), the gate after the reduction against main's eager chain, the
extend conv over the token rows and its state snapshots against main's padded
``F.conv1d`` path (passes main runs on the native depthwise kernel), the chunked conv
against one ``F.conv1d``, and the multi-row host-table gather against main's one-row
kernel.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="the PLE kernels need a CUDA GPU",
)

BF16 = torch.bfloat16


def _bits(a, b):
    return a.shape == b.shape and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


@pytest.mark.parametrize("virtual", [False, True])
@pytest.mark.parametrize("requests,width", [(1, 4), (37, 4), (256, 3)])
def test_verify_layout_is_the_torch_branch(requests, width, virtual):
    from sgl_nvidia_mega.qwen3_8.kernels.ops.embeddings import (
        qwen4_ple_verify_prepare as prep,
    )

    torch.manual_seed(requests)
    dev, eos, ctx = "cuda", 151645, 2
    rows = requests * width
    tokens = torch.randint(0, 150000, (rows,), device=dev)
    tokens[::7] = eos
    lengths = torch.randint(0, width + 1, (requests,), device=dev, dtype=torch.int32)
    req = torch.randint(1, 512, (requests,), device=dev)
    locations = torch.randint(1, 1 << 20, (rows,), device=dev)
    locations.view(requests, width)[::5] = 0  # graph padding rows
    mapping = torch.randperm(1024, device=dev)[:512].to(torch.int32)
    v2p = torch.randperm(1024, device=dev).to(torch.int32) if virtual else None
    history = torch.randint(0, 150000, (1024, ctx), device=dev)
    assert prep.can_prepare(
        tokens, lengths, req, locations, mapping, v2p, history, width, ctx + 1
    )
    out = prep.prepare_verify_layout(
        tokens, lengths, req, locations, mapping, v2p, history, width, eos
    )

    # _prepare_ple_batch's target-verify branch (SGLang main), slots from the table
    positions = torch.arange(rows, device=dev)
    ref_lengths = lengths.long()
    req_indices = torch.div(positions, width, rounding_mode="floor")
    offsets = positions - req_indices * width
    valid_tokens = offsets < ref_lengths.index_select(0, req_indices)
    slots = mapping[req].long()
    if virtual:
        slots = v2p[slots]
    valid = ref_lengths.ne(0) & locations.reshape(requests, width).ne(0).any(dim=1)
    slots = torch.where(valid, slots, torch.zeros_like(slots))
    padded = tokens.new_full((requests, width), eos)
    padded[req_indices, offsets] = torch.where(
        valid_tokens, tokens, tokens.new_full((), eos)
    )
    context = torch.cat([history.index_select(0, slots.long()), padded], dim=1)
    contexts = context.unfold(1, ctx + 1, 1)[req_indices, offsets]

    assert _bits(out.lengths, ref_lengths)
    assert _bits(out.req_indices, req_indices)
    assert _bits(out.token_offsets, offsets)
    assert _bits(out.valid_tokens, valid_tokens)
    assert out.state_indices.dtype == (torch.int32 if virtual else torch.int64)
    assert torch.equal(out.state_indices.long(), slots.long())
    assert _bits(out.ngram_context, context)
    assert _bits(out.contexts, contexts)


@pytest.mark.parametrize("state_dtype", [BF16, torch.float32])
@pytest.mark.parametrize("intermediate", [False, True])
@pytest.mark.parametrize("requests,width", [(1, 4), (48, 4), (130, 3)])
def test_verify_conv_one_kernel_is_bitwise_main(
    requests, width, intermediate, state_dtype
):
    from sglang.kernels.ops.mamba.qwen4_short_conv import fused_qwen4_verify_conv

    from sgl_nvidia_mega.qwen3_8.kernels.ops.mamba import qwen4_verify_conv

    torch.manual_seed(width * requests)
    channels, k, dilation = 10240, 4, 3
    state_len = (k - 1) * dilation
    rows = requests * width
    x = torch.randn(rows, channels, device="cuda", dtype=BF16)
    residual = torch.randn_like(x)
    weight = torch.randn(channels, 1, k, device="cuda", dtype=BF16)
    state = torch.randn(requests + 9, channels, state_len, device="cuda").to(
        state_dtype
    )
    indices = torch.randperm(requests + 9, device="cuda")[:requests]
    valid = torch.rand(rows, device="cuda") > 0.2
    cache = (
        torch.zeros(
            requests + 2, width, channels, state_len, device="cuda", dtype=state_dtype
        )
        if intermediate
        else None
    )
    ref_cache = cache.clone() if intermediate else None
    args = (x, residual, weight, state, indices, valid, width, dilation)
    ref = fused_qwen4_verify_conv(*args, ref_cache)
    new = qwen4_verify_conv.fused_qwen4_verify_conv(*args, cache, one_kernel=True)
    assert _bits(new, ref)
    if intermediate:
        assert _bits(cache, ref_cache)


def _eager_gate(gate, value):
    # Qwen4ExpPLELayer.forward (SGLang main), after the BF16 reduction
    import math

    gate = gate / math.sqrt(2560)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    gate = torch.sigmoid(gate)
    return gate * value.unsqueeze(-2)


@pytest.mark.parametrize("tokens", [1, 33, 16384])
def test_gate_post_is_bitwise_the_eager_chain(tokens):
    from sgl_nvidia_mega.qwen3_8.kernels.ops.elementwise import qwen4_gate_post as g

    device = torch.device("cuda", torch.cuda.current_device())
    assert g.is_qwen4_gate_post_ready(device), g._POST_STATE["reason"]
    torch.manual_seed(tokens)
    key = torch.randn(tokens, 4, 2560, device="cuda", dtype=BF16)
    query = torch.randn_like(key)
    value = torch.randn(tokens, 2560, device="cuda", dtype=BF16)
    gate = (key * query).sum(dim=-1, keepdim=True)
    assert g.can_fuse_qwen4_gate_post(gate, value)
    assert _bits(g.fused_qwen4_gate_post(gate, value), _eager_gate(gate, value))


def _extend_batch(lengths):
    lengths_t = torch.tensor(lengths, device="cuda")
    req = torch.cat([torch.full((n,), i, device="cuda") for i, n in enumerate(lengths)])
    off = torch.cat([torch.arange(n, device="cuda") for n in lengths])
    return SimpleNamespace(
        mode=SimpleNamespace(name="EXTEND"),
        use_decode_fast_path=False,
        lengths=lengths_t,
        row_width=max(lengths),
        req_indices=req,
        token_offsets=off,
    )


@pytest.mark.parametrize("lengths", [[1], [700, 3, 0, 129], [4096, 17]])
def test_extend_conv_is_bitwise_mains_padded_conv(lengths):
    from sgl_nvidia_mega.qwen3_8.kernels.ops.mamba import qwen4_ple_fused as fused

    torch.manual_seed(len(lengths))
    channels, k, dilation = 2560, 4, 3
    state_len = (k - 1) * dilation
    batch = _extend_batch(lengths)
    x = torch.randn(sum(lengths), channels, device="cuda", dtype=BF16)
    weight = torch.randn(channels, 1, k, device="cuda", dtype=BF16)
    conv_state = torch.randn(len(lengths) + 3, channels, state_len, device="cuda")
    conv_state = conv_state.to(BF16)
    indices = torch.randperm(len(lengths) + 3, device="cuda")[: len(lengths)]
    track_offsets = (batch.lengths // 2).clamp(min=0)
    assert fused.conv_engages(
        x,
        batch,
        lengths,
        weight=weight,
        conv_state=conv_state,
        channels=channels,
        state_len=state_len,
    )
    state = conv_state.index_select(0, indices).to(dtype=x.dtype)

    # Qwen4ExpPLELayer._short_conv's padded path (SGLang main), before SiLU
    padded = x.new_zeros((len(lengths), batch.row_width, channels))
    padded[batch.req_indices, batch.token_offsets] = x
    conv_input = torch.cat([state, padded.transpose(1, 2)], dim=-1)
    ref = F.conv1d(conv_input, weight, dilation=dilation, groups=channels)
    ref = ref.transpose(1, 2)[batch.req_indices, batch.token_offsets]
    cols = torch.arange(state_len, device="cuda")

    def gather_at(offsets):
        return conv_input.gather(
            2,
            (offsets.unsqueeze(1) + cols.unsqueeze(0))
            .unsqueeze(1)
            .expand(-1, channels, -1),
        )

    conv, (next_state, track) = fused.fused_short_conv_extend(
        x,
        state,
        weight,
        batch,
        dilation=dilation,
        state_offsets=(batch.lengths, track_offsets),
    )
    assert _bits(conv, ref)
    assert _bits(next_state, gather_at(batch.lengths))
    assert _bits(track, gather_at(track_offsets))


def test_chunked_conv_is_bitwise_one_conv():
    from sgl_nvidia_mega.qwen3_8.kernels.ops.mamba.ple_short_conv import (
        chunked_packed_short_conv,
    )

    torch.manual_seed(3)
    lengths, channels, k, dilation = [5000, 1, 2047], 2560, 4, 3
    state_len = (k - 1) * dilation
    conv_input = torch.randn(
        len(lengths), channels, state_len + max(lengths), device="cuda", dtype=BF16
    )
    weight = torch.randn(channels, 1, k, device="cuda", dtype=BF16)
    full = F.conv1d(conv_input, weight, dilation=dilation, groups=channels)
    full = full.transpose(1, 2)
    ref = torch.cat([full[i, :n] for i, n in enumerate(lengths)])
    out = chunked_packed_short_conv(
        conv_input, weight, lengths, dilation=dilation, chunk_size=1024
    )
    assert _bits(out, ref)


@pytest.mark.parametrize("dtype", [BF16, torch.float8_e4m3fn])
@pytest.mark.parametrize("rows,programs", [(32, 64), (2, 0), (128, 1)])
def test_gather_rows_is_bitwise_the_stock_gather(dtype, rows, programs):
    import triton
    from sglang.srt.models.qwen4_exp import _gather_ple_embedding_from_pinned_kernel

    from sgl_nvidia_mega.qwen3_8.kernels.ops.embeddings.qwen4_ple_gather import (
        gather_rows,
    )

    torch.manual_seed(rows)
    vocab, dim, start, end = 70000, 80, 1000, 61000
    table = torch.randn(end - start, dim).to(dtype).pin_memory()
    ids = torch.randint(0, vocab, (16384 * 2 + 3,), device="cuda")
    block_d = triton.next_power_of_2(dim)
    ref = torch.empty(ids.numel(), dim, device="cuda", dtype=BF16)
    _gather_ple_embedding_from_pinned_kernel[(ids.numel(),)](
        table.data_ptr(),
        ids,
        ref,
        embedding_dim=dim,
        tp_vocab_start=start,
        tp_vocab_end=end,
        is_fp8=dtype == torch.float8_e4m3fn,
        BLOCK_D=block_d,
    )
    out = torch.empty_like(ref)
    gather_rows(
        weight_ptr=table.data_ptr(),
        flat_ids=ids,
        output=out,
        embedding_dim=dim,
        tp_vocab_start=start,
        tp_vocab_end=end,
        is_fp8=dtype == torch.float8_e4m3fn,
        block_d=block_d,
        rows=rows,
        programs=programs,
    )
    torch.cuda.synchronize()
    assert _bits(out, ref)

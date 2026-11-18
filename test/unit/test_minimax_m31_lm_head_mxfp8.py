# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU checks of the MiniMax-M3.1 target's opt-in slot lm_head_mxfp8: the slot list
(opt-in, needs slot mega), its field of the per-width plan, the numerics the spec
declares with it and its toolchain; the kernel's shape rules on the TP4 shard (N 50016,
K 6144: one variant, no padded weight row, no scale of a padded row read) and the
quantizer's rules; the stand-in through SGLang's own
``LogitsProcessor._compute_lm_head``; the heads the load refuses; the build (the copy,
the check against SGLang's projection, the ranks' agreement); the target's build and
summary, the executor and slot mega's step, which hand the logits processor the
stand-in; and the parity switch's line. The kernel itself is in
test/kernels/test_minimax_m31_lm_head_mxfp8.py.

The tests that drive SGLang's logits processor skip where it does not import."""

import contextlib
import logging
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

try:
    from sglang.kernels.plugin_kernels.nvidia.spec import NumericsKind, Step
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.megakernels.spec import NumericsKind, Step

from sgl_nvidia_mega.minimax_m31 import ROWS, logits
from sgl_nvidia_mega.minimax_m31 import moe as m31_moe
from sgl_nvidia_mega.minimax_m31 import plan as m31_plan
from sgl_nvidia_mega.minimax_m31 import runtime as m31_runtime
from sgl_nvidia_mega.minimax_m31 import spec as m31_spec
from sgl_nvidia_mega.minimax_m31.moe import moe_nofill as mn
from sgl_nvidia_mega.minimax_m31.moe import moe_swap
from sgl_nvidia_mega.minimax_m31.runtime import agree as m31_agree
from sgl_nvidia_mega.minimax_m31.runtime import executor as m31_executor
from sgl_nvidia_mega.minimax_m31.runtime import parity as m31_parity
from sgl_nvidia_mega.minimax_m31.sm103 import host, nvcc
from sgl_nvidia_mega.minimax_m31.sm103.logits import lm_head_mxfp8 as lh

SLOT = "lm_head_mxfp8"
# MiniMax M3.1-Flash on TP4: this rank's shard of the 200,064-token vocabulary
SHARD, HIDDEN = 50_016, 6144
VERIFY_WIDTHS = tuple(range(8, 129, 8))
BOUND = m31_spec.LM_HEAD_TOLERANCES[f"{SLOT}.logits.rel"]
# a small head with the shard's remainders: 96 rows past a multiple of 128, whole
# 16-row tiles
VOCAB, DIM = 224, 256


def _weight(n=VOCAB, k=DIM, seed=0):
    torch.manual_seed(seed)
    return (torch.randn(n, k) * 0.02).to(torch.bfloat16)


# ------------------------------------------------------------------ the slot list


def test_the_slot_is_opt_in_and_needs_slot_mega():
    env = m31_runtime.SLOTS_ENV
    assert SLOT in m31_runtime.OPT_IN_SLOTS
    assert SLOT not in m31_runtime.SHIPPED_SLOTS
    assert SLOT not in m31_runtime.requested_slots({})
    listed = m31_runtime.requested_slots({env: f"mega,markov,{SLOT}"})
    assert listed == {"mega", "markov", SLOT}
    # a slot of the target's plan, not the draft's, and not bitwise
    assert m31_runtime.target_slots(listed) == {"mega", SLOT}
    assert SLOT not in m31_runtime.DRAFT_SLOTS
    assert SLOT not in m31_runtime.BITWISE_SLOTS
    assert m31_runtime.NEEDS[SLOT] == ("mega",)
    assert m31_runtime.slot_reasons(listed) == []
    (needs,) = m31_runtime.slot_reasons(frozenset({"moe_nofill", SLOT}))
    assert f"slot '{SLOT}' needs slot 'mega'" in needs
    (unknown,) = m31_runtime.slot_reasons(frozenset({"lm_head_fp8"}))
    assert "unknown slot 'lm_head_fp8'" in unknown and SLOT in unknown


def test_the_plan_gives_it_megas_step_only():
    every = frozenset(VERIFY_WIDTHS)
    facts = m31_plan.Facts(
        mega=frozenset({8}), moe_nofill=every, moe_swap=every, lm_head_mxfp8=every
    )
    plan = m31_plan.plan(VERIFY_WIDTHS, facts)
    assert plan[8] == m31_plan.Slots(mega=True, lm_head_mxfp8=True)
    assert not any(plan[w].lm_head_mxfp8 for w in VERIFY_WIDTHS[1:])
    assert (
        m31_plan.describe(plan)
        == "8: mega, lm_head_mxfp8; 16-128: moe_nofill, moe_swap"
    )
    # without slot mega's step there is nothing for it to close
    no_mega = m31_plan.Facts(moe_nofill=every, lm_head_mxfp8=frozenset({8}))
    assert not m31_plan.slots(8, no_mega).lm_head_mxfp8
    # and off unless its build serves the width
    assert m31_plan.slots(8, m31_plan.Facts(mega=frozenset({8}))) == m31_plan.Slots(
        mega=True
    )


def test_the_spec_declares_its_tolerance_with_megas():
    both = m31_spec._numerics(frozenset({"mega", SLOT}))
    assert both.kind is NumericsKind.TOLERANCE
    assert dict(both.tolerances) == {
        **m31_spec.MEGA_TOLERANCES,
        **m31_spec.LM_HEAD_TOLERANCES,
    }
    assert m31_spec.LM_HEAD_TOLERANCES == {"lm_head_mxfp8.logits.rel": 4e-2}
    # not requested: the shipped list's numerics are slot mega's alone
    shipped = m31_spec._numerics(frozenset(m31_runtime.SHIPPED_SLOTS))
    assert dict(shipped.tolerances) == m31_spec.MEGA_TOLERANCES


def test_its_toolchain_is_the_bindings_and_any_nvcc():
    def reasons(slots, missing=(), nvcc_reason=""):
        with (
            patch.object(m31_spec, "_has_module", lambda name: name not in missing),
            patch.object(nvcc, "why_not", return_value=nvcc_reason) as why_not,
        ):
            return m31_spec.toolchain_reasons(frozenset(slots)), why_not

    found, why_not = reasons({SLOT})
    assert found == []
    why_not.assert_called_once_with(None)  # no version floor: QMUL4 is slot mega's
    _, why_not = reasons({"mega", SLOT})
    why_not.assert_called_once_with(nvcc.QMUL4_VERSION)
    assert reasons({SLOT}, missing=("cuda.bindings",))[0] == [
        "needs the CUDA Python bindings"
    ]
    assert reasons({SLOT}, nvcc_reason="no CUDA toolkit (nvcc) found")[0] == [
        "no CUDA toolkit (nvcc) found"
    ]


def test_an_fp32_lm_head_server_is_refused_only_with_the_slot():
    fp32 = SimpleNamespace(enable_fp32_lm_head=True)
    bf16 = SimpleNamespace(enable_fp32_lm_head=False)
    with_slot = frozenset({"mega", SLOT})
    assert m31_spec.lm_head_config_reasons(bf16, with_slot) == []
    (reason,) = m31_spec.lm_head_config_reasons(fp32, with_slot)
    assert "--enable-fp32-lm-head" in reason and f"slot {SLOT}" in reason
    assert m31_spec.lm_head_config_reasons(fp32, frozenset({"mega"})) == []


# ------------------------------------------------- the kernel's shapes, on the shard


def test_the_tp4_shard_is_one_variant_and_302_mib():
    """N 50016 = 3126 whole 16-row tiles, K 6144 = 48 chunks of 128: two warps a tile,
    four chunks ahead, on B300's 148 SMs and GB300's 152; the copy is the codes and
    the scales of N padded to 128 rows."""
    assert SHARD % 128 == 96 and SHARD % lh.TILE == 0 and HIDDEN % lh.CHUNK == 0
    for sms in (148, 152):
        v = lh.variant(SHARD, HIDDEN, fp8=True, sms=sms)
        assert v == lh.Variant(HIDDEN, True, 2, 4)
        assert lh.variant(SHARD, HIDDEN, fp8=False, sms=sms) == lh.Variant(
            HIDDEN, False, 2, 2
        )
    # the rows of x and the split's partial sums, within SM100's 227 KiB a CTA
    assert lh.shared_bytes(v) == 8 * (HIDDEN + 8) * 2 + 4096 < 227 * 1024
    assert lh._pad128(SHARD) == SHARD + 32
    size = SHARD * HIDDEN + (SHARD + 32) * HIDDEN // lh.BLOCK
    assert size == 316_907_520 and round(size / 2**20, 1) == 302.2
    with pytest.raises(ValueError, match="does not split"):
        lh.variant(SHARD, 6144 + 128, fp8=True, sms=148)


def _scale_offset(row: int, block: int, k: int) -> int:
    """lm_head_mxfp8.cu's scale_offset: where the kernel reads a row's block scale."""
    cols = k // lh.BLOCK
    return (
        ((row >> 7) * (cols >> 2) + (block >> 2)) * 512
        + (row & 31) * 16
        + ((row >> 5) & 3) * 4
        + (block & 3)
    )


@pytest.mark.parametrize("n", [VOCAB, 1000])
def test_no_tile_reads_a_padded_row(n):
    """The kernel's tiles are ceil(N / 16); a tile's rows past N read row N - 1. Every
    scale it reads is the row's own: the bytes of the rows the swizzle pads to 128
    are never addressed, whatever they hold."""
    q = lh.quantize(_weight(n))
    blocks = DIM // lh.BLOCK
    scales = lh.unswizzle(q.scales, n, DIM)
    rows = {
        min(tile * lh.TILE + g, n - 1)
        for tile in range(-(-n // lh.TILE))
        for g in range(lh.TILE)
    }
    assert rows == set(range(n))
    read = set()
    for row in rows:
        for block in range(blocks):
            offset = _scale_offset(row, block, DIM)
            assert q.scales[offset] == scales[row, block]
            read.add(offset)
    padded = {
        _scale_offset(row, block, DIM)
        for row in range(n, lh._pad128(n))
        for block in range(blocks)
    }
    assert padded and not padded & read
    assert len(read) + len(padded) == q.scales.numel()
    # poisoned, the padded rows change nothing the kernel multiplies
    poisoned = q.scales.clone()
    poisoned[sorted(padded)] = 0xFF
    assert torch.equal(
        lh.dequantize(lh.Mxfp8Weight(q.codes, poisoned)), lh.dequantize(q)
    )
    assert q.codes.shape == (n, DIM)  # the codes are not padded


def test_each_block_takes_the_smallest_scale_that_fits_448():
    w = _weight()
    w[1, :32] = 0
    w[2, 32:64] = 448.0  # exactly 448 * 2^0
    w[3, 64:96] = 456.0  # just past 448 (449 is 448 in BF16)
    q = lh.quantize(w)
    e = lh.unswizzle(q.scales, *q.shape).to(torch.int64) - 127
    amax = w.float().view(VOCAB, -1, lh.BLOCK).abs().amax(-1)
    nonzero = amax > 0
    assert torch.all(amax[nonzero] <= 448 * torch.exp2(e[nonzero].double()))
    assert torch.all(amax[nonzero] > 448 * torch.exp2(e[nonzero].double() - 1))
    assert e[2, 1] == 0 and e[3, 2] == 1
    assert q.codes.float().abs().max() <= lh.E4M3_MAX
    assert torch.all(q.codes[1, :32].float() == 0)
    assert not lh.scale_reason(q) and lh.scale_reason(lh.quantize(_weight() * 1e6))


def test_codes_round_to_the_nearest_e4m3_value_and_dequantize_exactly():
    w = _weight()
    q = lh.quantize(w)
    e = lh.unswizzle(q.scales, *q.shape).to(torch.float32) - 127
    scaled = torch.ldexp(w.float().view(VOCAB, -1, lh.BLOCK), -e[..., None])
    codes = q.codes.float().view_as(scaled)
    # an E4M3 step at |v|: 2^(floor(log2 |v|) - 3), 2^-9 among the subnormals
    step = torch.exp2(torch.floor(torch.log2(scaled.abs().clamp_min(2**-6))) - 3)
    assert torch.all((scaled - codes).abs() <= step / 2)
    dq = lh.dequantize(q)
    exact = torch.ldexp(codes, e[..., None]).view(VOCAB, DIM)
    assert torch.equal(dq.float(), exact)
    assert torch.equal(lh.dequantize(lh.quantize(dq)), dq)


def test_the_copys_projection_is_within_the_declared_bound_of_the_bf16_one():
    """The quantizer's error on Gaussian data at the shard's K, without the kernel
    (which multiplies the dequantized weight exactly): inside the declared bound."""
    w = _weight(2048, HIDDEN)
    dq = lh.dequantize(lh.quantize(w))
    x = torch.randn(ROWS, HIDDEN).to(torch.bfloat16)
    theirs = x.float() @ w.float().T
    rel = ((x.float() @ dq.float().T - theirs).abs().max() / theirs.abs().max()).item()
    assert 0 < rel < BOUND, rel


def test_the_kernel_shape_rules():
    assert lh.unsupported_reason(_weight()) == ""
    assert "multiple of 128" in lh.unsupported_reason(_weight(k=192))
    assert "BF16" in lh.unsupported_reason(_weight().float())
    assert "BF16" in lh.unsupported_reason(_weight()[0])


def test_a_launch_does_not_compile():
    """The kernels build in precompile only: a launch that finds none raises."""
    v = lh.variant(SHARD, HIDDEN, fp8=True, sms=148)
    with patch.dict(lh._KERNELS, {}, clear=True):
        with pytest.raises(RuntimeError, match="a launch does not compile"):
            lh._kernel(torch.device("cuda", 0), v)
    # and the stream takes CUDA rows only
    with pytest.raises(ValueError, match="x must be BF16"):
        lh.matmul(torch.zeros(ROWS, DIM, dtype=torch.bfloat16), lh.quantize(_weight()))


# --------------------------------------------------- through SGLang's own projection


@pytest.fixture
def sglang_head():
    """SGLang's logits processor and an LM head as it loads one (the BF16 shard on its
    unquantized method); skips where SGLang's module does not import."""
    try:
        from sglang.srt.layers.logits_processor import LogitsProcessor
        from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
    except (ImportError, OSError) as e:
        pytest.skip(f"SGLang's logits processor does not import here: {e}")

    def model(**over):
        proc = LogitsProcessor.__new__(LogitsProcessor)
        torch.nn.Module.__init__(proc)
        proc.use_fp32_lm_head = proc.use_fp32_lm_head_output = False
        proc.rl_on_policy_target = None
        for name, value in over.items():
            setattr(proc, name, value)
        head = SimpleNamespace(
            weight=_weight(), quant_method=UnquantizedEmbeddingMethod()
        )
        return SimpleNamespace(lm_head=head, logits_processor=proc)

    return model


@pytest.fixture
def stream(monkeypatch):
    """The MXFP8 stream on CPU (x @ dequantize(q).T, as the kernel multiplies);
    records each call's rows and each build."""
    calls, built = [], []

    def matmul(x, q, *, out=None):
        calls.append(x.shape[0])
        return (x.float() @ lh.dequantize(q).float().T).to(torch.bfloat16)

    monkeypatch.setattr(lh, "matmul", matmul)
    monkeypatch.setattr(lh, "precompile", lambda n, k, device: built.append((n, k)))
    monkeypatch.setattr(logits, "_cpu_group", lambda: None)
    return SimpleNamespace(calls=calls, built=built)


def _rows(rows=ROWS, k=DIM, seed=1):
    torch.manual_seed(seed)
    return torch.randn(rows, k).to(torch.bfloat16)


def test_sglang_routes_the_stand_in_to_the_stream_and_its_own_head_to_its_gemm(
    sglang_head, stream
):
    from sglang.srt.layers.logits_processor import should_apply_lm_head_quant_method

    model = sglang_head()
    head = model.lm_head
    stand_in = logits.Mxfp8LmHead(head, lh.quantize(head.weight))
    assert should_apply_lm_head_quant_method(stand_in, stand_in.quant_method)
    assert not should_apply_lm_head_quant_method(head, head.quant_method)
    # the stand-in is SGLang's module to every other reader
    assert stand_in.weight is head.weight
    assert stand_in.quant_method is not head.quant_method
    with pytest.raises(AttributeError):
        stand_in.set_lora
    x = _rows()
    proc = model.logits_processor
    theirs = proc._compute_lm_head(x, head)
    assert stream.calls == []
    ours = proc._compute_lm_head(x, stand_in)
    assert stream.calls == [ROWS]
    assert ours.dtype == theirs.dtype == torch.bfloat16
    assert ours.shape == theirs.shape == (ROWS, VOCAB)
    rel = (
        (ours.float() - theirs.float()).abs().max() / theirs.float().abs().max()
    ).item()
    assert 0 < rel <= BOUND


@pytest.mark.parametrize(
    "change, want",
    [
        (lambda m: setattr(m.logits_processor, "use_fp32_lm_head", True), "FP32"),
        (
            lambda m: setattr(m.logits_processor, "use_fp32_lm_head_output", True),
            "FP32",
        ),
        (
            lambda m: setattr(m.logits_processor, "rl_on_policy_target", "fsdp"),
            "on-policy",
        ),
        (
            lambda m: (
                setattr(m.lm_head, "set_lora", 1) or setattr(m.lm_head, "apply_lora", 1)
            ),
            "LoRA",
        ),  # fmt: skip
        (
            lambda m: setattr(m.lm_head, "quant_method", SimpleNamespace(apply=print)),
            "quant method",
        ),  # fmt: skip
        (lambda m: setattr(m.lm_head, "weight", m.lm_head.weight.half()), "float16"),
        (
            lambda m: setattr(m.lm_head, "weight", _weight(k=192)),
            "K 192 is not a multiple of 128",
        ),
        (
            lambda m: setattr(m.lm_head, "weight", _weight(k=2 * DIM)[:, ::2]),
            "not contiguous",
        ),
        (lambda m: setattr(m.lm_head, "weight", None), "has no weight"),
    ],
    ids=[
        "fp32",
        "fp32_output",
        "on_policy",
        "lora",
        "quantized",
        "fp16",
        "k",
        "strided",
        "no_weight",
    ],
)
def test_a_head_sglang_projects_another_way_is_refused(sglang_head, change, want):
    model = sglang_head()
    assert logits.head_reasons(model) == []
    change(model)
    reasons = logits.head_reasons(model)
    assert any(want in r for r in reasons), reasons


def test_build_makes_the_copy_and_checks_it_against_sglangs_projection(
    sglang_head, stream, caplog
):
    model = sglang_head()
    slots = frozenset({"mega", SLOT})
    # not requested, or no width of one request: nothing is built
    assert logits.build(model, VERIFY_WIDTHS, frozenset({"mega"})) is None
    assert logits.build(model, (16, 24), slots) is None
    assert stream.built == [] and stream.calls == []
    caplog.set_level(logging.INFO, logger=logits.__name__)
    head = logits.build(model, VERIFY_WIDTHS, slots)
    assert isinstance(head, logits.Mxfp8LmHead) and head.lm_head is model.lm_head
    # the kernel of this shard, built at the build; then the check's one projection
    assert stream.built == [(VOCAB, DIM)] and stream.calls == [ROWS]
    assert 0 < head.rel <= BOUND
    copy = head.quant_method.copy
    assert torch.equal(lh.dequantize(copy), lh.dequantize(lh.quantize(_weight())))
    size = VOCAB * DIM + lh._pad128(VOCAB) * DIM // lh.BLOCK
    assert head.device_bytes() == size
    assert head.summary() == (
        f"the 8-row step's LM head streams an MXFP8 copy of this rank's "
        f"[{VOCAB}, {DIM}] shard, {size / 2**20:.1f} MiB of device memory"
    )
    (line,) = [r.getMessage() for r in caplog.records if SLOT in r.getMessage()]
    assert head.summary() in line and f"{head.rel:.2e}" in line


def test_build_refuses_logits_outside_the_bound_and_heads_it_cannot_serve(
    sglang_head, stream, monkeypatch
):
    slots = frozenset({"mega", SLOT})
    good = lh.matmul
    monkeypatch.setattr(lh, "matmul", lambda x, q, **kw: good(x, q, **kw) * 1.05)
    with pytest.raises(ValueError, match="bound 4.00e-02"):
        logits.build(sglang_head(), VERIFY_WIDTHS, slots)
    monkeypatch.setattr(lh, "matmul", lambda x, q, **kw: good(x, q, **kw).float())
    with pytest.raises(ValueError, match="gives torch.float32"):
        logits.build(sglang_head(), VERIFY_WIDTHS, slots)
    monkeypatch.setattr(lh, "matmul", good)
    with pytest.raises(ValueError, match="FP32"):
        logits.build(sglang_head(use_fp32_lm_head=True), VERIFY_WIDTHS, slots)
    model = sglang_head()
    model.lm_head.weight = model.lm_head.weight * 1e6
    with pytest.raises(ValueError, match="a block scale of 2\\^"):
        logits.build(model, VERIFY_WIDTHS, slots)


@pytest.mark.parametrize("others", [None, "ValueError: K 192 is not a multiple of 128"])
def test_every_rank_raises_when_one_ranks_build_fails(
    sglang_head, stream, monkeypatch, others
):
    """The build is agreed through sm103/host.py: a rank whose own build passed raises
    with the failed rank's error."""
    import torch.distributed as dist

    seen = []

    def all_gather_object(errors, error, group):
        seen.append((error, group))
        errors[:] = [error, others, None, None]

    monkeypatch.setattr(logits, "_cpu_group", lambda: "cpu group")
    monkeypatch.setattr(dist, "get_world_size", lambda group: 4)
    monkeypatch.setattr(dist, "all_gather_object", all_gather_object)
    build = lambda: logits.build(  # noqa: E731
        sglang_head(), VERIFY_WIDTHS, frozenset({"mega", SLOT})
    )
    if others is None:
        assert isinstance(build(), logits.Mxfp8LmHead)
    else:
        with pytest.raises(RuntimeError) as caught:
            build()
        assert str(caught.value) == f"building slot {SLOT} failed: rank 1: {others}"
    assert seen == [(None, "cpu group")]


def test_the_stream_takes_no_bias_and_logs_its_capture_once(monkeypatch, caplog):
    seen = []
    monkeypatch.setattr(lh, "matmul", lambda x, q, **kw: seen.append(q) or "logits")
    method = logits.Mxfp8LmHeadMethod("copy")
    caplog.set_level(logging.INFO, logger=logits.__name__)
    x = torch.zeros(ROWS, DIM, dtype=torch.bfloat16)
    assert method.apply(None, x) == "logits" and seen == ["copy"]
    with pytest.raises(ValueError, match="without an embedding bias"):
        method.apply(None, x, torch.zeros(VOCAB))
    # the rows as SGLang's GEMM takes them: cast to BF16, and dense for the kernel
    rows = []
    monkeypatch.setattr(lh, "matmul", lambda x, q, **kw: rows.append(x))
    method.apply(None, x)
    method.apply(None, torch.ones(ROWS, 2 * DIM)[:, ::2])
    assert rows[0] is x
    assert rows[1].dtype == torch.bfloat16 and rows[1].is_contiguous()
    assert rows[1].shape == (ROWS, DIM) and bool((rows[1] == 1).all())
    monkeypatch.setattr(logits, "_capturing", lambda: True)
    for _ in range(3):
        method.apply(None, x)
    lines = [r.getMessage() for r in caplog.records if SLOT in r.getMessage()]
    assert lines == [f"NVIDIA megakernel: {SLOT} captured at 8 rows"]


# ------------------------------------------------------------- the target's build


@contextlib.contextmanager
def _builds(slots, *, head=None, mega=None):
    """``runtime.build`` with ``slots`` requested, the MoE slots serving every width,
    slot mega's step ``mega`` and slot lm_head_mxfp8's build returning ``head``."""
    from sgl_nvidia_mega.minimax_m31.runtime import mega as m31_mega

    agreed = []
    with (
        patch.object(m31_runtime, "SLOTS", frozenset(slots)),
        patch.object(m31_moe, "build", return_value=mn.Bound(widths=VERIFY_WIDTHS)),
        patch.object(
            moe_swap, "build", return_value=moe_swap.Bound(widths=VERIFY_WIDTHS)
        ),
        patch.object(m31_agree, "agree_tp", lambda value, what: agreed.append(value)),
        patch.object(m31_mega, "MegaStep", mega or Mock()),
        patch.object(logits, "build", return_value=head) as build,
    ):
        yield build, agreed


VERIFY = SimpleNamespace(widths=VERIFY_WIDTHS, tokens_per_request=8, binding=None)
TARGET = ("mega", "moe_nofill", "moe_swap")


def test_the_target_builds_it_plans_it_at_8_rows_and_states_its_memory():
    head = Mock()
    head.summary.return_value = "an MXFP8 copy, 302.2 MiB of device memory"
    mega = Mock()
    mega.return_value.summary.return_value = "MegaAttention"
    slots = frozenset({*TARGET, SLOT})
    with _builds(slots, head=head, mega=mega) as (build, agreed):
        built = m31_runtime.build("model", "hf", VERIFY)
    build.assert_called_once_with("model", VERIFY_WIDTHS, slots)
    assert built.bound[SLOT] is head
    assert built.slot_plan[8] == m31_plan.Slots(mega=True, lm_head_mxfp8=True)
    assert not any(built.slot_plan[w].lm_head_mxfp8 for w in VERIFY_WIDTHS[1:])
    # every rank agrees on the widths it serves
    (facts,) = agreed
    assert facts.lm_head_mxfp8 == frozenset({8}) and facts.mega == frozenset({8})
    assert built.summary().endswith(
        "slots by width 8: mega, lm_head_mxfp8; 16-128: moe_nofill, moe_swap; mega: "
        "MegaAttention; lm_head_mxfp8: an MXFP8 copy, 302.2 MiB of device memory"
    )


def test_the_default_list_builds_no_copy():
    with _builds(TARGET) as (build, agreed):
        built = m31_runtime.build("model", "hf", VERIFY)
    # the slot's own build decides: it returns None unless the slot is listed
    build.assert_called_once_with("model", VERIFY_WIDTHS, frozenset(TARGET))
    assert built.bound[SLOT] is None
    assert built.slot_plan[8] == m31_plan.Slots(mega=True)
    assert agreed[0].lm_head_mxfp8 == frozenset()
    assert SLOT not in built.summary()


def test_without_megas_step_the_slot_is_refused_and_no_copy_is_made():
    """The slot list's check refuses it at argument resolution; a build that got past
    it refuses too, before any copy."""
    with (
        _builds({"moe_nofill", "moe_swap", SLOT}) as (build, _),
        pytest.raises(RuntimeError) as caught,
    ):
        m31_runtime.build("model", "hf", VERIFY)
    build.assert_not_called()
    assert f"requested slots ['{SLOT}'] serve no width" in str(caught.value)
    assert f"{SLOT}: slot mega's step is not built)" in str(caught.value)


def test_the_load_names_a_head_the_slot_cannot_serve(sglang_head, monkeypatch):
    """runtime.model_disable_reasons, which SGLang calls on every rank at load: the
    head's reasons under the slot's name, only while it is requested."""
    from sglang.srt.models.minimax_m3 import MiniMaxM3Model

    m = MiniMaxM3Model.__new__(MiniMaxM3Model)
    torch.nn.Module.__init__(m)
    m.layers, m.start_layer, m.end_layer = [], 0, 0
    m.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    fp32 = sglang_head(use_fp32_lm_head_output=True)
    model = SimpleNamespace(model=m, is_mrope_enabled=False, **vars(fp32))
    assert m31_runtime.model_disable_reasons(model, None, frozenset()) == []
    (reason,) = m31_runtime.model_disable_reasons(model, None, frozenset({SLOT}))
    assert reason == f"slot {SLOT}: SGLang computes the LM head's logits in FP32"
    model.logits_processor.use_fp32_lm_head_output = False
    assert m31_runtime.model_disable_reasons(model, None, frozenset({SLOT})) == []


# ------------------------------------------------------------------- the step


def test_the_executor_hands_megas_step_the_stand_in_where_the_plan_says():
    mega = Mock()
    mega.run.return_value = "mega"
    plan = {
        8: m31_plan.Slots(mega=True, lm_head_mxfp8=True),
        16: m31_plan.Slots(),
    }
    built = m31_executor.MegakernelDecode(
        None, widths=(8, 16), plan=plan, bound={SLOT: "stand-in"}, mega=mega
    )
    step = Step(mode="target_verify", rows=8)
    assert built._forward(step, "ids", "pos", "fb") == "mega"
    mega.run.assert_called_once_with(step, "ids", "pos", "fb", lm_head="stand-in")
    # without the slot at that width: SGLang's head (no argument)
    mega.reset_mock()
    built.slot_plan = {8: m31_plan.Slots(mega=True)}
    built._forward(step, "ids", "pos", "fb")
    mega.run.assert_called_once_with(step, "ids", "pos", "fb")


def _mega_step():
    """Slot mega's step over fakes with no layer: the embedding, the hand-off of the
    residual stream, the final norm and the logits processor."""
    from sgl_nvidia_mega.minimax_m31.runtime import mega as m31_mega

    seen = []
    lm = SimpleNamespace(
        start_layer=0,
        end_layer=0,
        layers=[],
        layers_to_capture=[],
        embed_tokens=lambda ids: ids.float()[:, None].expand(-1, 4),
        norm=lambda x: x + 1,
    )

    def logits_processor(input_ids, hidden, lm_head, fb, aux):
        seen.append((lm_head, hidden.clone(), aux))
        return "output"

    model = SimpleNamespace(
        model=lm,
        lm_head="SGLang's head",
        capture_aux_hidden_states=False,
        logits_processor=logits_processor,
    )
    step = m31_mega.MegaStep.__new__(m31_mega.MegaStep)
    step.model, step.lm = model, lm
    step.layer_ids, step.first_moe, step.attention, step.moe = [], 0, [], []
    step.layer_input = torch.zeros(ROWS, 4)
    step._bind = lambda view, forward_batch: None
    return step, seen


def test_megas_step_closes_on_the_stand_in_and_on_sglangs_head_without_it():
    try:
        from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
    except (ImportError, OSError) as e:
        pytest.skip(f"SGLang's residual stream does not import here: {e}")

    stream_of = lambda fb: SimpleNamespace(  # noqa: E731
        pending=None, export=lambda hidden: (hidden, None)
    )
    ids = torch.arange(ROWS)
    request = SimpleNamespace(view=None)
    for stand_in in (None, Mock()):
        step, seen = _mega_step()
        with (
            patch.object(residual_batch, "start", lambda fb: None),
            patch.object(residual_batch, "clear", lambda fb: None),
            patch.object(residual_batch, "stream_of", stream_of),
        ):
            if stand_in is None:
                out = step.run(request, ids, ids, "fb")
            else:
                out = step.run(request, ids, ids, "fb", lm_head=stand_in)
        assert out == "output"
        ((head, hidden, aux),) = seen
        assert aux is None
        assert torch.equal(hidden, ids.float()[:, None].expand(-1, 4) + 1)
        if stand_in is None:
            assert head == "SGLang's head"
        else:
            assert head is stand_in
            # the parity switch's hook, on the final-normed rows the processor gets
            ((model, rows),) = [c.args for c in stand_in.log_parity.call_args_list]
            assert model is step.model and torch.equal(rows, hidden)


# ------------------------------------------------------------- the parity switch


@pytest.mark.parametrize("value, lines", [("0", 0), ("", 0), ("1", 1)])
def test_the_parity_switch_logs_the_stream_against_sglangs_projection(
    monkeypatch, caplog, value, lines
):
    """One greppable line for this rank's shard: the stream against SGLang's BF16
    projection of the step's own hidden rows (its rel against the declared bound)."""
    head = SimpleNamespace(weight=_weight())
    stand_in = logits.Mxfp8LmHead(head, lh.quantize(head.weight))
    dq = lh.dequantize(stand_in.quant_method.copy)
    asked = []

    def compute(hidden, lm_head):
        asked.append(lm_head)
        weight = dq if lm_head is stand_in else head.weight
        return (hidden.float() @ weight.float().T).to(torch.bfloat16)

    model = SimpleNamespace(
        lm_head=head, logits_processor=SimpleNamespace(_compute_lm_head=compute)
    )
    monkeypatch.setenv(m31_parity.PARITY_ENV, value)
    caplog.set_level(logging.INFO, logger=m31_parity.logger.name)
    stand_in.log_parity(model, _rows())
    found = [
        r.getMessage() for r in caplog.records if "megakernel parity" in r.getMessage()
    ]
    assert len(found) == lines
    if lines:
        assert asked == [head, stand_in]
        assert found[0].startswith(
            f"NVIDIA megakernel parity {SLOT} shard logits rows=8 "
        )
        rel = float(found[0].split(" rel=")[1].split()[0])
        assert 0 < rel <= BOUND
    # a captured step runs no check
    monkeypatch.setattr(logits, "_capturing", lambda: True)
    asked.clear()
    stand_in.log_parity(model, _rows())
    assert asked == []


def test_the_host_launch_asks_for_a_dependent_launch_only_when_told(monkeypatch):
    """The stream is the package's one dependent (PDL) launch: the cooperative
    kernels' launches carry no such attribute."""
    launched = []

    class Attribute:
        def __init__(self):
            self.id = None
            self.value = SimpleNamespace()

    cu = SimpleNamespace(
        CUlaunchConfig=lambda: SimpleNamespace(),
        CUlaunchAttribute=Attribute,
        CUlaunchAttributeID=SimpleNamespace(
            CU_LAUNCH_ATTRIBUTE_COOPERATIVE="cooperative",
            CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION="dependent",
        ),
        CUstream=lambda stream: stream,
        cuLaunchKernelEx=lambda config, function, parameters, extra: (
            launched.append([a.id for a in config.attrs]) or (0,)
        ),
    )
    monkeypatch.setattr(host, "driver", lambda: cu)
    for kwargs in ({}, {"cooperative": True}, {"dependent": True}):
        host.launch("function", (1, 1, 1), (32, 1, 1), 0, [], 0, **kwargs)
    assert launched == [[], ["cooperative"], ["dependent"]]

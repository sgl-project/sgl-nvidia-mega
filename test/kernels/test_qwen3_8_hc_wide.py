# SPDX-License-Identifier: Apache-2.0
"""The wide HC kernels (runtime/hc_wide.py) against SGLang main's stock
``GatedResidual`` on random weights, 33..4096 rows of 4 x 2560 BF16 streams (rank 320),
on an SM10x GPU.

Bitwise where the source claims it: the extend write-back
(``hc_combine_extend``) and the wide combine-norm (``hc_combine_apply_norm_wide``)
against main's ``combine`` (``hc_combine``) and next ``hc_norm``
(``grouped_gemma_rmsnorm``), and the one-launch boundary's combined streams. A
tolerance elsewhere: ``hc_mix_wide`` (split-K order of the down GEMM) and the
boundary's normalized streams, gate totals and mixed input (rounding points and K
order). The tolerances are declared, not measured; record the actual errors.
"""

import dataclasses

import pytest
import torch
import torch.nn.functional as F

from sgl_nvidia_mega.qwen3_8.runtime import hc_wide
from sgl_nvidia_mega.qwen3_8.runtime.hc_wide import HIGH_CONCURRENCY as HIGH
from sgl_nvidia_mega.qwen3_8.runtime.hc_wide import MID_CONCURRENCY as MID

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="the wide HC kernels are SM10x only",
)

HC, HS, RANK = 4, 2560, 320
WIDTH = HC * HS
EPS = 1e-6
ROWS = [36, 64, 128, 512, 4096]
BF16 = torch.bfloat16


@pytest.fixture(scope="module", autouse=True)
def built():
    for limits in (HIGH, MID):
        assert hc_wide.build(limits=limits) == ""


def _module(seed, combine=True):
    from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionConfig

    config = HyperConnectionConfig(
        hc_count=HC,
        hidden_size=HS,
        params_dtype=BF16,
        hc_lowrank=RANK,
        rms_norm_eps=EPS,
        hc_per_branch_norm=True,
    )
    hc = GatedResidual(config, use_mix=True, use_combine=combine)
    hc = hc.to(device="cuda", dtype=BF16)
    torch.manual_seed(seed)
    with torch.no_grad():
        hc.hc_norm.weight.normal_(0, 0.1)
        hc.input_mix_weight_down.weight.normal_(0, 0.02)
        hc.input_mix_weight_up.weight.normal_(0, 0.02)
        if combine:
            hc.block_inject_weight.weight.normal_(0, 0.02)
    return hc


def _randn(seed, *shape):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(shape, device="cuda", dtype=BF16, generator=g)


def _reference_mix(x, hc):
    """GatedResidual._mix_compute in FP64."""
    x = x.double()
    t = F.silu(F.linear(x, hc.input_mix_weight_down.weight.double()) / HC)
    u = torch.sigmoid(F.linear(t, hc.input_mix_weight_up.weight.double()))
    return (u.unflatten(-1, (HC, HS)) * x.unflatten(-1, (HC, HS))).mean(dim=-2)


def _close(out, ref, rtol, atol, record_property, name):
    error = (out.double() - ref.double()).abs().max().item()
    record_property(f"{name}_max_abs", error)
    torch.testing.assert_close(out.double(), ref.double(), rtol=rtol, atol=atol)


@pytest.mark.parametrize("rows", ROWS)
def test_mix(rows, record_property):
    hc = _module(1)
    streams = _randn(10, rows, WIDTH)
    stock_mixed, (_, stock_normed, _) = hc.mix(streams)
    mixed, normed, partials = hc_wide.mix(hc, streams, limits=HIGH)
    torch.cuda.synchronize()
    assert torch.equal(normed, stock_normed)  # the same grouped_gemma_rmsnorm
    assert partials is None
    if rows > 512:
        assert hc_wide.mix_route(hc, normed, limits=HIGH) == "stock"
        assert torch.equal(mixed, stock_mixed)
        return
    assert hc_wide.mix_route(hc, normed, limits=HIGH) == "hc_mix_wide"
    _close(mixed, _reference_mix(normed, hc), 1e-2, 5e-3, record_property, "fp64")
    _close(mixed, stock_mixed, 2e-2, 1e-2, record_property, "stock")


@pytest.mark.parametrize("rows", ROWS)
def test_extend_write_back_is_bitwise(rows):
    hc, nxt = _module(1), _module(2)
    residual = _randn(10, rows, WIDTH)
    block, block2 = _randn(11, rows, HS), _randn(12, rows, HS)
    normed = hc.hc_norm(residual)
    stock = hc.combine(block, (residual, normed, None))
    stock_normed = nxt.hc_norm(stock)
    stock_next = nxt.combine(block2, (stock, stock_normed, None))

    route = hc_wide.write_route(
        hc, block, residual, normed, None, nxt.hc_norm, limits=HIGH
    )
    assert route == "hc_combine_extend"
    combined, normalized, totals = hc_wide.write(
        hc,
        block,
        residual,
        normed,
        None,
        nxt.hc_norm,
        nxt.block_inject_weight.weight,
        limits=HIGH,
    )
    assert torch.equal(combined, stock)
    assert torch.equal(normalized, stock_normed)
    assert totals.shape == (rows, HC) and totals.dtype == torch.float32
    # The next write-back takes the gate totals in place of the normalized streams
    out, no_norm, no_totals = hc_wide.write(
        nxt, block2, combined, normalized, totals, None, limits=HIGH
    )
    assert no_norm is None and no_totals is None
    assert torch.equal(out, stock_next)


@pytest.mark.parametrize("rows", ROWS)
def test_mid_concurrency_write_back_is_bitwise(rows):
    hc, nxt = _module(1), _module(2)
    residual = _randn(10, rows, WIDTH)
    block = _randn(11, rows, HS)
    stock_mixed, (_, normed_ref, _) = hc.mix(residual)
    stock = hc.combine(block, (residual, normed_ref, None))
    stock_normed = nxt.hc_norm(stock)

    mixed, normed, partials = hc_wide.mix(hc, residual, limits=MID)
    wide = rows <= MID.fused_combine_max_rows
    assert (partials is not None) == wide  # the side-stream gate
    route = hc_wide.write_route(
        hc, block, residual, normed, partials, nxt.hc_norm, limits=MID
    )
    assert route == ("hc_combine_apply_norm_wide" if wide else "hc_combine_extend")
    combined, normalized, _ = hc_wide.write(
        hc,
        block,
        residual,
        normed,
        partials,
        nxt.hc_norm,
        nxt.block_inject_weight.weight,
        limits=MID,
    )
    # Before a PLE layer: no next norm
    alone, no_norm, _ = hc_wide.write(
        hc, block, residual, normed, partials, None, limits=MID
    )
    torch.cuda.synchronize()
    assert torch.equal(mixed, stock_mixed)  # MID keeps main's mix
    assert torch.equal(combined, stock)
    assert torch.equal(normalized, stock_normed)
    assert no_norm is None
    assert torch.equal(alone, stock)


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("gate", [True, False])
def test_one_launch_boundary(rows, gate, record_property):
    before, writer, reader = _module(1), _module(2), _module(3)
    residual0 = _randn(10, rows, WIDTH)
    block0, block = _randn(11, rows, HS), _randn(12, rows, HS)
    # The write-back before hands the writer's read its gate totals
    residual, normed, totals = hc_wide.write(
        before,
        block0,
        residual0,
        before.hc_norm(residual0),
        None,
        writer.hc_norm,
        writer.block_inject_weight.weight,
        limits=HIGH,
    )
    inject = reader.block_inject_weight.weight if gate else None
    reason = hc_wide.boundary_reason(
        writer, block, residual, totals, reader, inject, limits=HIGH
    )
    if rows > HIGH.boundary_ks_max_rows:
        assert reason
        return
    assert reason == ""
    # Flag 4 writes the normalized streams, so they can be compared
    forced = dataclasses.replace(HIGH, boundary_ks_flags=HIGH.boundary_ks_flags | 4)
    combined, normalized, next_totals, mixed = hc_wide.boundary(
        writer, block, residual, totals, reader, inject, limits=forced
    )
    _, skipped, _, mixed_default = hc_wide.boundary(
        writer, block, residual, totals, reader, inject, limits=HIGH
    )
    separate, separate_normed, separate_totals = hc_wide.write(
        writer, block, residual, normed, totals, reader.hc_norm, inject, limits=HIGH
    )
    stock = writer.combine(block, (residual, normed, None))
    stock_mixed, (_, stock_normed, _) = reader.mix(stock)
    torch.cuda.synchronize()
    assert torch.equal(combined, stock) and torch.equal(separate, stock)
    assert torch.equal(separate_normed, stock_normed)
    _close(normalized, stock_normed, 1e-2, 1e-2, record_property, "normalized")
    _close(mixed, stock_mixed, 2e-2, 1e-2, record_property, "mixed")
    _close(mixed_default, stock_mixed, 2e-2, 1e-2, record_property, "mixed_default")
    # Skipped exactly when the next write-back reads the totals instead
    assert (skipped.numel() == 0) == gate
    if gate:
        gates = 2 * torch.sigmoid(next_totals / HC)
        reference = 2 * torch.sigmoid(separate_totals / HC)
        _close(gates, reference, 0, 5e-3, record_property, "gate")
    else:
        assert next_totals is None


def test_one_launch_boundary_into_the_final_mix(record_property):
    rows = 64
    before, writer = _module(1), _module(2)
    mixer = _module(3, combine=False)
    residual0 = _randn(10, rows, WIDTH)
    block0, block = _randn(11, rows, HS), _randn(12, rows, HS)
    residual, normed, totals = hc_wide.write(
        before,
        block0,
        residual0,
        before.hc_norm(residual0),
        None,
        writer.hc_norm,
        writer.block_inject_weight.weight,
        limits=HIGH,
    )
    assert hc_wide.boundary_reason(writer, block, residual, totals, mixer) == ""
    combined, normalized, next_totals, mixed = hc_wide.boundary(
        writer, block, residual, totals, mixer
    )
    stock = writer.combine(block, (residual, normed, None))
    stock_mixed, (_, stock_normed, _) = mixer.mix(stock)
    torch.cuda.synchronize()
    assert next_totals is None and normalized.numel() > 0
    assert torch.equal(combined, stock)
    _close(normalized, stock_normed, 1e-2, 1e-2, record_property, "normalized")
    _close(mixed, stock_mixed, 2e-2, 1e-2, record_property, "mixed")

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""``K31MK_ATTNRES_CARVEOUT_ROWS``: the AttnRes carve-out (``K31MK_ATTNRES_CARVEOUT``) only at the listed rows.

Lines (stated before any run):
(1) CARVEOUT=100 + ROWS "1-32:1": the function ``PreparedAttnRes.run`` actually calls (recorded at the call) reads
    CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT 100 at 32 rows and the driver default d0 (!= 100) at 64 rows;
    the ring (``bind_ring``) variant's code picks the same way (read through its row pick; the ring is not launched);
(2) both plain launches' output, new prefix and bank are bitwise the knob-off ones (32 and 64 rows, nb 3, delta, no write);
(3) ROWS unset with CARVEOUT=100: the 64-row launch calls a function reading 100 (today's behaviour kept);
(4) a row outside 1-128 or a value other than 0 / 1 raises ValueError.

Mutations (each must make the test fail): ``rows_ignored`` (the carve-out code at every row) fails (1) at 64 rows;
``rows_inverted`` (the knob-off code at listed rows) fails (1) at 32 rows; ``launch_no_pick`` (``_launch`` calls the
table object itself) fails (1) (the call raises).
"""

import pytest
import torch

import _harness as H

MUTATIONS = {
    "rows_ignored": (
        "kimi_k31/k3/cutedsl/attn_res/prepared.py",
        "return self.on if rows in self.rows else self.off",
        "return self.on",
    ),
    "rows_inverted": (
        "kimi_k31/k3/cutedsl/attn_res/prepared.py",
        "return self.on if rows in self.rows else self.off",
        "return self.off if rows in self.rows else self.on",
    ),
    "launch_no_pick": (
        "kimi_k31/k3/cutedsl/attn_res/prepared.py",
        "            code = code.pick(out.shape[0])",
        "            pass",
    ),
}

NB = 3
DEADLINE_NS = 4_000_000_000

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU",
)


class _NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _target(g):
    norm = lambda: _NS(weight=(1.0 + H.randn((7168,), 0.2, g)).to(torch.bfloat16), variance_epsilon=1e-6)  # noqa: E731
    return _NS(self_attention_res_norm=norm(), self_attention_res_proj=_NS(weight=H.randn((1, 7168), 0.02, g)),
               input_layernorm=norm(), prev_valid_blocks=NB, is_block_write_layer=False, block_write_idx=0)


def _carveout(code, device_index):
    import cuda.bindings.driver as cuda
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.host import resolve_kernel

    _, function = resolve_kernel(getattr(code, "compiled", code), device_index)
    attr = cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
    status, value = cuda.cuFuncGetAttribute(attr, function)
    assert status == cuda.CUresult.CUDA_SUCCESS, status
    return value


class _Rec:
    """Forwards a launch to ``inner`` and records it (so the test reads the function the launch really called)."""

    def __init__(self, inner, log):
        self.inner, self.log = inner, log

    def __call__(self, *args):
        self.log.append(self.inner)
        return self.inner(*args)


def _run(prep, rows, inputs):
    delta, prefix, blocks = (t[:rows].clone() for t in inputs)
    out = torch.empty((rows, 7168), dtype=torch.bfloat16, device="cuda")
    H.poison(out)
    prep.run(delta, prefix, blocks, out)
    torch.cuda.synchronize()
    return out, prefix, blocks


@needs_gpu
def test_attnres_carveout_rows(monkeypatch):
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.attn_res import prepared as P

    dev = torch.cuda.current_device()
    P.compile_consumer.cache_clear()
    g = H.generator(5151)
    target = _target(g)
    inputs = (H.randn((64, 7168), 1.0, g), H.randn((64, 7168), 2.0, g), H.randn((64, 8, 7168), 2.0, g))
    mailbox = torch.empty((1, 64, 7168), dtype=torch.bfloat16, device="cuda")

    def prepared():
        prep = P.PreparedAttnRes.for_target(target, deadline_ns=DEADLINE_NS)
        prep.bind_ring(mailbox)
        return prep

    monkeypatch.delenv("K31MK_ATTNRES_CARVEOUT", raising=False)
    monkeypatch.delenv("K31MK_ATTNRES_CARVEOUT_ROWS", raising=False)
    off = prepared()
    d0 = _carveout(off.code, dev)
    assert d0 != 100, "driver default carve-out is already 100: the check cannot see the knob"
    base = {rows: _run(off, rows, inputs) for rows in (32, 64)}

    monkeypatch.setenv("K31MK_ATTNRES_CARVEOUT", "100")
    monkeypatch.setenv("K31MK_ATTNRES_CARVEOUT_ROWS", "1-32:1")
    on = prepared()
    assert isinstance(on.code, P.RowCarveout) and isinstance(on.ring[0], P.RowCarveout), "ROWS set: no row table built"
    log = []
    on.code = P.RowCarveout(_Rec(on.code.on, log), _Rec(on.code.off, log), on.code.rows)
    for rows, want in ((32, 100), (64, d0)):
        log.clear()
        got = _run(on, rows, inputs)
        assert len(log) == 1, f"{rows} rows: {len(log)} recorded launches"
        value = _carveout(log[0], dev)
        assert value == want, f"ROWS 1-32:1, plain launch at {rows} rows: carve-out {value}, want {want}"
        ring_value = _carveout(on.ring[0].pick(rows), dev)
        assert ring_value == want, f"ROWS 1-32:1, ring code at {rows} rows: carve-out {ring_value}, want {want}"
        for name, a, b in zip(("out", "prefix", "blocks"), got, base[rows]):
            assert torch.equal(a.view(torch.int16), b.view(torch.int16)), f"{name} differs at {rows} rows"

    monkeypatch.delenv("K31MK_ATTNRES_CARVEOUT_ROWS", raising=False)
    every = prepared()
    assert not isinstance(every.code, P.RowCarveout)
    log = []
    every.code = _Rec(every.code, log)
    _run(every, 64, inputs)
    assert len(log) == 1 and _carveout(log[0], dev) == 100, "ROWS unset: the carve-out must apply at every row (today's behaviour)"
    H.record({"kind": "check", "item": "attnres_carveout_rows", "default": d0, "rows": "1-32:1"})

    for bad in ("129:1", "8:2"):
        monkeypatch.setenv("K31MK_ATTNRES_CARVEOUT_ROWS", bad)
        with pytest.raises(ValueError):
            P.PreparedAttnRes.for_target(target, deadline_ns=DEADLINE_NS)

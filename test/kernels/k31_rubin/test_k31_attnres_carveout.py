# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""``K31MK_ATTNRES_CARVEOUT=N``: the preferred shared-memory carve-out of the AttnRes kernels
(``attn_res.prepared.PreparedAttnRes``), a launch preference set as a driver function attribute.

Lines (stated before any run):
(1) knob unset: the compiled AttnRes function's CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT reads the driver
    default d0 (a fresh knob-off compile, before any knob compile in the process), and d0 != 100 (else (2) proves nothing);
(2) knob 100: the function the AttnRes launches reads 100, for the plain and the ring (``bind_ring``) variants;
(3) after (2), a knob-off PreparedAttnRes of the same key in the same process reads d0 again and launches a different
    compiled object than the knob-on one (a cached compile does not carry the attribute across knob settings);
(4) the knob-on output, new prefix and bank are bitwise the knob-off ones (64 rows, nb 3, delta, no write);
(5) a value outside 0-100 raises ValueError.

Mutations (each must make the test fail): ``carveout_ignored`` (the knob compiles but never sets the attribute) fails (2);
``carveout_shared_key`` (the knob reuses the knob-off cache entry) fails (3).
"""

import pytest
import torch

import _harness as H

MUTATIONS = {
    "carveout_ignored": (
        "kimi_k31/k3/cutedsl/attn_res/prepared.py",
        "return prepare_shared_carveout(code, self.device.index, value)",
        "return code",
    ),
    "carveout_shared_key": (
        "kimi_k31/k3/cutedsl/attn_res/prepared.py",
        "*extra, carveout=value",
        "*extra",
    ),
}

ROWS, NB = 64, 3
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


def _run(prep, inputs):
    delta, prefix, blocks = (t.clone() for t in inputs)
    out = torch.empty((ROWS, 7168), dtype=torch.bfloat16, device="cuda")
    H.poison(out)
    prep.run(delta, prefix, blocks, out)
    torch.cuda.synchronize()
    return out, prefix, blocks


@needs_gpu
def test_attnres_carveout(monkeypatch):
    H.load_plugin(MUTATIONS)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.attn_res import prepared as P

    dev = torch.cuda.current_device()
    P.compile_consumer.cache_clear()
    g = H.generator(5150)
    target = _target(g)
    inputs = (H.randn((ROWS, 7168), 1.0, g), H.randn((ROWS, 7168), 2.0, g), H.randn((ROWS, 8, 7168), 2.0, g))
    mailbox = torch.empty((1, 64, 7168), dtype=torch.bfloat16, device="cuda")

    def prepared():
        prep = P.PreparedAttnRes.for_target(target, deadline_ns=DEADLINE_NS)
        prep.bind_ring(mailbox)
        return prep

    monkeypatch.delenv("K31MK_ATTNRES_CARVEOUT", raising=False)
    off = prepared()
    d0 = _carveout(off.code, dev)
    assert d0 != 100, "driver default carve-out is already 100: the knob check cannot see the knob"
    assert _carveout(off.ring[0], dev) == d0
    base = _run(off, inputs)

    monkeypatch.setenv("K31MK_ATTNRES_CARVEOUT", "100")
    on = prepared()
    got = (_carveout(on.code, dev), _carveout(on.ring[0], dev))
    assert got == (100, 100), f"knob 100: plain / ring carve-out {got}"
    knob = _run(on, inputs)

    monkeypatch.delenv("K31MK_ATTNRES_CARVEOUT", raising=False)
    again = prepared()
    got = (_carveout(again.code, dev), _carveout(again.ring[0], dev))
    assert got == (d0, d0), f"knob-off after a knob-on compile: carve-out {got}, want the default {d0}"
    assert getattr(again.code, "compiled", again.code) is not getattr(on.code, "compiled", on.code)

    for name, a, b in zip(("out", "prefix", "blocks"), knob, base):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16)), f"{name} differs with the knob on"
    H.record({"kind": "check", "item": "attnres_carveout", "default": d0, "knob": 100})

    monkeypatch.setenv("K31MK_ATTNRES_CARVEOUT", "101")
    with pytest.raises(ValueError):
        P.PreparedAttnRes.for_target(target, deadline_ns=DEADLINE_NS)

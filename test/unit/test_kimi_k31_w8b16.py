# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""W8 (TP8 / DCP8) verify at Breq 16 (128 rows), stage 1: the contract admits Breq 16 only when a boot lists it, and the
DCP verify executor's prologue buffer holds the boot's largest verify width. CPU only. Expected values are hand-derived
from the contract (rows = Breq x 8) and from comm.cuh's prologue check (local slots >= rows)."""

from types import SimpleNamespace

import pytest


def test_w8b16_contract_breq16_only_when_listed():
    from sgl_nvidia_mega.kimi_k31.verify_contract import served_request_batches, verify_keys

    assert served_request_batches({}) == (1, 2, 4)  # knob unset: the c=1 build, unchanged
    assert served_request_batches({"K31MK_VERIFY_BATCHES": "1,2,4,8"}) == (1, 2, 4, 8)  # today's W8 kits, unchanged
    assert served_request_batches({"K31MK_VERIFY_BATCHES": "1,2,4,8,16"}) == (1, 2, 4, 8, 16)
    assert verify_keys(8, (1, 2, 4, 8, 16))[-1] == (16, 8, 128)
    # Wide tier (port of ad05010 with exact keys): Breq 9-15 are admitted with their own keys (no padding to 16);
    # 17+ stay refused (32 / 64 are later HT stages).
    assert served_request_batches({"K31MK_VERIFY_BATCHES": "1,2,4,8,12"}) == (1, 2, 4, 8, 12)
    assert verify_keys(8, (1, 2, 4, 8, 12))[-1] == (12, 8, 96)
    for bad in ("1,2,4,8,16,32", "1,2,4,8,16,64", "1,2,4,8,17"):
        with pytest.raises(ValueError):
            served_request_batches({"K31MK_VERIFY_BATCHES": bad})


def _executor(cap):
    # dcp/verify.py imports the DSA verify build, the index projection and the chain executor, which pull GPU-only
    # modules (triton, the host tree). They are not under test: stand-ins replace them while dcp/verify.py is loaded
    # from its real source, so DcpVerifyExecutor.__init__ and local_slot_rows run as written.
    import importlib.util
    import pathlib
    import sys
    import types

    import sgl_nvidia_mega.kimi_k31 as k31
    import sgl_nvidia_mega.kimi_k31.dcp  # noqa: F401 (docstring-only package: the relative imports' parent)

    class _Base:
        def __init__(self, *args, **kwargs):
            pass

    stubs = {
        "sgl_nvidia_mega.kimi_k31.dsa": types.ModuleType("sgl_nvidia_mega.kimi_k31.dsa"),
        "sgl_nvidia_mega.kimi_k31.dsa.indexer": types.ModuleType("sgl_nvidia_mega.kimi_k31.dsa.indexer"),
        "sgl_nvidia_mega.kimi_k31.dsa.verify": types.ModuleType("sgl_nvidia_mega.kimi_k31.dsa.verify"),
        "sgl_nvidia_mega.kimi_k31.verify_executor": types.ModuleType("sgl_nvidia_mega.kimi_k31.verify_executor"),
    }
    stubs["sgl_nvidia_mega.kimi_k31.dsa"].__path__ = []
    stubs["sgl_nvidia_mega.kimi_k31.dsa.indexer"].IndexProjection = stubs["sgl_nvidia_mega.kimi_k31.dsa.indexer"].IndexWeights = _Base
    stubs["sgl_nvidia_mega.kimi_k31.dsa.verify"].BuiltSparseVerify = stubs["sgl_nvidia_mega.kimi_k31.dsa.verify"].SparseVerifyStep = _Base
    stubs["sgl_nvidia_mega.kimi_k31.verify_executor"].K31VerifyExecutor = _Base
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        src = pathlib.Path(k31.__file__).parent / "dcp" / "verify.py"
        spec = importlib.util.spec_from_file_location("sgl_nvidia_mega.kimi_k31.dcp._w8b16_verify", src)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert issubclass(module.DcpVerifyExecutor, _Base)  # the stand-in base, not the real chain
        return module.DcpVerifyExecutor(None, None, (), comm=SimpleNamespace(cap=cap, device="cpu"))
    finally:
        for name, old in saved.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def test_w8b16_dcp_local_slots_hold_the_largest_width():
    # The prologue needs local_slots.size(0) >= rows (comm.cuh): a Breq 16 step has 128 rows.
    for cap, want in ((8, 64), (32, 64), (64, 64), (128, 128)):
        ex = _executor(cap)
        assert tuple(ex.local_slots.shape) == (want,), (cap, tuple(ex.local_slots.shape))
        assert int(ex.local_slots.min()) == -1 and int(ex.local_slots.max()) == -1  # unwritten slots read as 'other owner'
    ex = _executor(128)
    assert ex.local_slots[:128].numel() == 128  # the run() slice of a 128-row step is not truncated

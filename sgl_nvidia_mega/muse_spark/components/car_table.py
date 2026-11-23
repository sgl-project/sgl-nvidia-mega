# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``car_table`` (env ``MUSE_VR_CAR_TABLE=1``, default off): standalone custom all-reduce at 8 ranks uses
``2shot_push`` instead of ``1shot_push`` for mid-size messages.

SGLang's JIT custom all-reduce (v2) picks its algorithm from a per-(architecture, world size) table
(``device_communicators/configs/custom_all_reduce_v2.py``). On the devices listed in ``SWEPT_DEVICES`` the 8-rank entry
sends every message up to its one-shot ceiling to ``1shot_push`` and leaves ``2shot_push`` disabled. In a CUDA-graph
chain of dependent all-reduces of ``[M, 8192]`` bf16 at 8 ranks over two hosts, ``2shot_push`` measured faster than
``1shot_push`` above the default threshold below (swept at world 8;
tunable with ``MUSE_VR_CAR_TABLE_ONE_SHOT_KB``).

This component wraps ``CustomAllReduceV2.custom_all_reduce``: when the stock dispatch picks ``1shot_push`` on an
8-rank communicator on a swept device for a message larger than ``MUSE_VR_CAR_TABLE_ONE_SHOT_KB`` (default 384), the
call runs SGLang's ``2shot_push`` kernel instead (its slot, ``ceil(n / 8)`` rounded to 16 B vectors, always fits the
push workspace the stock table sized for ``1shot_push`` at that size). Nothing else changes: the stock table, the workspace
sizes, ``should_custom_ar`` and the ``uses_one_shot_push`` / ``uses_two_shot_push`` answers that fused kernels (the
Muse Spark finalize all-reduce, ``ar_attn_boundary``) consult stay stock, so those fused paths engage exactly as
before. Explicit ``override_algo`` calls are left alone.

With ``MUSE_VR_CAR_TABLE_PULL_MAX_KB=<n>`` (default 0 = off) the same swap also applies to messages the stock table
sends to ``2shot_pull`` (multicast or not) up to ``n`` KB, as long as the ``2shot_push`` slot fits the push workspace.
Unlike the ``1shot_push`` swap this one is not bitwise: the multicast pull reduces in the NVLink switch, the push kernel
on the SMs in rank order (a reduction-order/precision difference; the test compares both against an fp64 sum). ``tests/dist_car_table.py`` checks that the two algorithms
give bitwise identical results on the sizes the component moves.

Engagement requires the sha256 of the table file and of the communicator to match the pinned revision. On a device
not in ``SWEPT_DEVICES`` every call runs the stock dispatch; sweep there first and add the device's key.

4 ranks (``MUSE_VR_CAR_TABLE_W4=1`` with ``MUSE_VR_CAR_TABLE=1``; default off, so 8-rank lines are unchanged): the same
wrapper acts on 4-rank communicators with its own thresholds: ``1shot_push`` above ``MUSE_VR_CAR_TABLE_W4_ONE_SHOT_KB``
(default 1024) and ``2shot_pull`` (multicast or not) up to ``MUSE_VR_CAR_TABLE_W4_PULL_MAX_KB`` (default 16384) run
``2shot_push``, the row SGLang's later stock table uses at 4 ranks on this device class (1-shot to 1 MiB, then 2shot_push to
16 MiB). In a CUDA-graph chain of dependent ``[M, 8192]`` and ``[M, 4096]`` all-reduces + boundary kernels at 4 ranks,
``2shot_push`` measured faster above about 1 MiB and below about 0.75 MiB slower. A non-multicast ``2shot_pull`` and
``2shot_push`` both sum in source-rank order in fp32 (bitwise outputs); the multicast pull reduces in the switch (values
compared against an fp64 sum, as at 8 ranks).
"""

from __future__ import annotations

import hashlib
import logging
import os

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/ops/communication/all_reduce.py": ("AllReduceAlgo",),
    "srt/distributed/device_communicators/configs/custom_all_reduce_v2.py": (
        "two_shot_push_slot_bytes",
    ),
}


logger = logging.getLogger("muse_vr_plugin")

NAME = "car_table"
SGLANG_PINS = {
    "sglang/srt/distributed/device_communicators/configs/custom_all_reduce_v2.py": (
        "194512e1a61907198836d5f720d4c9110172abc456168aa37e2df0aba34ace8b",
        "eb510c72ab31df1c681e48ce2958db03f91a860ccbf152c0cfa6cbbf4495c91f",
    ),
    "sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py": (
        "277da88e2837e585f3ebdf4122906d29ea7a96cc30b981f7a028f6f63b6cf001",
        "f7b23a55019734be7b2f1958e20cc1f78a31251a035f2d75885911aa4a46e6b5",
    ),
}
WORLD = 8
# Per communicator width: (1shot threshold env, default KB, max KB, pull-range env, default pull max KB, opt-in env or None).
WIDTHS = {
    8: (
        "MUSE_VR_CAR_TABLE_ONE_SHOT_KB",
        384,
        2496,
        "MUSE_VR_CAR_TABLE_PULL_MAX_KB",
        0,
        None,
    ),
    4: (
        "MUSE_VR_CAR_TABLE_W4_ONE_SHOT_KB",
        1024,
        10048,
        "MUSE_VR_CAR_TABLE_W4_PULL_MAX_KB",
        16384,
        "MUSE_VR_CAR_TABLE_W4",
    ),
}
# Device keys (hash of the device name and compute capability, as ``moe_tactic.device_key``) the world-8 sweep ran on.
SWEPT_DEVICES = frozenset({"c233982307dc30bd"})
_state = {"engaged": None}


def _one_shot_bytes(world: int = WORLD) -> int:
    env, default, top = WIDTHS[world][:3]
    kb = int(os.environ.get(env, str(default)))
    if not 16 <= kb <= top:
        raise ValueError(f"{env}={kb} outside 16..{top}")
    return kb * 1024


def _pull_max_bytes(world: int = WORLD) -> int:
    env, default = WIDTHS[world][3:5]
    kb = int(os.environ.get(env, str(default)))
    if kb < 0:
        raise ValueError(f"{env}={kb} < 0")
    return kb * 1024


def _widths() -> dict:
    """{world: (1shot threshold bytes, pull max bytes)} for the widths this process acts on."""
    out = {}
    for w, spec in WIDTHS.items():
        opt_in = spec[5]
        if opt_in is None or os.environ.get(opt_in, "0").strip() == "1":
            out[w] = (_one_shot_bytes(w), _pull_max_bytes(w))
    return out


def device_key(index: int) -> str:
    """Key of a device: a hash of its name and compute capability (same formula as ``moe_tactic.device_key``)."""
    import torch

    major, minor = torch.cuda.get_device_capability(index)
    return hashlib.sha256(
        f"{torch.cuda.get_device_name(index)}|{major}.{minor}".encode()
    ).hexdigest()[:16]


def install() -> None:
    import sglang.srt.distributed.device_communicators.custom_all_reduce_v2 as car
    import torch
    from sglang.kernels.ops.communication.all_reduce import AllReduceAlgo
    from sglang.srt.distributed.device_communicators.configs.custom_all_reduce_v2 import (
        two_shot_push_slot_bytes,
    )

    widths = _widths()
    cls = car.CustomAllReduceV2
    stock = cls.custom_all_reduce
    counted: set = set()
    arch: list = []  # resolved on the first call (register() runs before CUDA is initialized in some processes)

    def custom_all_reduce(self, input):  # noqa: A002 - SGLang's parameter name
        if not arch:
            index = (
                input.device.index
                if input.device.index is not None
                else torch.cuda.current_device()
            )
            arch.append(device_key(index) in SWEPT_DEVICES)
        if (
            not arch[0]
            or self.world_size not in widths
            or self.override_algo is not None
        ):
            return stock(self, input)
        one_shot, pull_max = widths[self.world_size]
        nbytes = input.numel() * input.element_size()
        if nbytes <= one_shot or nbytes % 16:
            return stock(self, input)
        cfg = self._pick_config(nbytes, self._can_use_graph())
        if cfg is None:
            return stock(self, input)
        if cfg.algo == AllReduceAlgo.ONE_SHOT_PUSH:
            src = "1shot"
        elif cfg.algo == AllReduceAlgo.TWO_SHOT_PULL and nbytes <= pull_max:
            src = "2shot_pull_mc" if cfg.use_multicast else "2shot_pull"
        else:
            return stock(self, input)
        if two_shot_push_slot_bytes(nbytes, self.world_size) > self.max_push_size:
            return stock(self, input)
        key = (
            f"2shot_push:from={src}:M={input.shape[0] if input.dim() > 1 else nbytes}"
            + ("" if self.world_size == WORLD else f":w={self.world_size}")
        )
        if key not in counted:
            counted.add(key)
            bed.count(NAME, key)
        return car.custom_all_reduce(
            self.obj,
            input,
            algo=AllReduceAlgo.TWO_SHOT_PUSH,
            graph_params=None,
            use_multicast=False,
        )

    custom_all_reduce.__wrapped__ = stock  # type: ignore[attr-defined]
    cls.custom_all_reduce = custom_all_reduce


def register() -> bool:
    if _state["engaged"] is not None:
        return _state["engaged"]
    if not bed.enabled(NAME):
        _state["engaged"] = False
        return False
    for rel, sha in SGLANG_PINS.items():
        if not bed.pinned(rel, sha):
            bed.report(NAME, False, reason=f"pin mismatch: {rel}")
            _state["engaged"] = False
            return False
    try:
        install()
    except Exception as e:  # noqa: BLE001 - leave SGLang stock
        bed.report(NAME, False, reason=f"install failed: {type(e).__name__}: {e}")
        _state["engaged"] = False
        return False
    w = _widths()
    bed.report(
        NAME,
        True,
        world=WORLD,
        one_shot_kb=_one_shot_bytes() // 1024,
        pull_max_kb=_pull_max_bytes() // 1024,
        widths={k: [v[0] // 1024, v[1] // 1024] for k, v in w.items()},
    )
    _state["engaged"] = True
    return True

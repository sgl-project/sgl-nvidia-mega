# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``gemm_pick`` (env ``MUSE_VR_GEMM_PICK=1``, default off): SGLang's TGV-or-cuBLAS choice for chosen bf16 GEMM
shapes and row ranges.

SGLang's bf16 GEMM dispatch (``cutedsl`` backend, the default on this platform without deterministic inference) runs a
GEMM ``[M, K] x [N, K]^T`` on the CuTe-DSL TGV kernel when its shape rule ``use_cutedsl_bf16_gemm(m, n, k)`` answers
True and on cuBLAS (``torch.nn.functional.linear``) otherwise. This component overrides the rule's answer for a list of
entries ``(N, K, kernel, row ranges)``; every other shape and row count keeps the stock rule, and the split-K table the
dispatch consults first is untouched. Both kernels are SGLang's own, called through SGLang's own entry points: TGV with
its own tile choice and launch mode, cuBLAS through ``F.linear``.

For the shapes these entries are meant for (the replicated MoE pooler and unpooler and the 4-way qkvg slice of Muse
Spark), the two kernels reduce K in one pass in the same order wherever cuBLAS picks a kernel without split-K, so they
give bitwise identical outputs there (``tests/test_gemm_pick.py`` checks each row count of each entry against the stock
dispatch on several input sets). Which row counts that covers depends on the cuBLAS build: from 56 rows up with the
branch layout's library; with SGLang main's image the pooler only from 136 rows up. Where it holds, the choice only
changes speed; the ranges are tunables, chosen with
``bench/bench_gemm_pick.py`` and a layer-chain A/B. The default entry puts the 4-way qkvg projection on cuBLAS at 136-256
rows, where it measured faster in a layer-chain A/B on SGLang main's image (bitwise there); the other shapes keep the
stock rule unless an entry is given.

Entries (env ``MUSE_VR_GEMM_PICK_RULES``): ``N x K : tgv|cublas : lo-hi[,lo-hi...]``, entries separated by ``;``, every
range inside 56..512. An entry for the pooler shape is dropped (and reported) where its rows overlap the ``pool_tgv``
component's range while that component is engaged.

Mechanism: the shape rule is wrapped at its definition and where the dispatch has bound it, as ``pool_tgv`` does.
Engagement is guarded by sha256 pins on the two SGLang files involved; on drift it stays off and logs.
"""

from __future__ import annotations

import logging
import os

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {}


logger = logging.getLogger("muse_vr_plugin")

NAME = "gemm_pick"
MIN_ROWS = 56  # below this row count cuBLAS splits K for some of these shapes (not TGV's reduction order)
MAX_ROWS = 512
POOL_SHAPE = (4096, 8192)
DEFAULT_RULES = "4608x8192:cublas:136-256"

# sha256 of files relative to the directory that holds the ``sglang`` package: every validated revision (the branch
# layout's, then SGLang main's, whose rule, entry points and dispatch binding are unchanged).
SGLANG_PINS = {
    # use_cutedsl_bf16_gemm (shape rule) and the TGV entry points
    "sglang/kernels/ops/gemm/cutedsl_bf16_gemm.py": (
        "c95c27b173142a1eaa94c25480f7e7efbd04bec3f900138f3335b4360a86c29d",
        "265a9444d0b980aaa46d1bd4113374dbf38ac86aa66235802fbe603f665bcec5",
    ),
    # initialize_bf16_gemm_config binds the rule; _bf16_gemm_dispatch_impl consults the split-K table, then the rule
    "sglang/srt/layers/quantization/unquant.py": (
        "f428022195466613e7e988265a1090862a5aa0557bb7a063c0c172f9bd3988c5",
        "7285d3e9e9fa881c5ecd97d093a8f59e11e69f26d5e3e7006925149f624a2f9c",
    ),
}

_state: dict = {"engaged": None, "rules": (), "orig_use": None}


def parse_rules(text: str) -> tuple:
    """``"8192x4096:cublas:64-128;..."`` -> ``((8192, 4096, False, ((64, 128),)), ...)``; the bool is "use TGV"."""
    rules = []
    for item in filter(None, (t.strip() for t in text.split(";"))):
        shape, kernel, ranges = (t.strip() for t in item.split(":"))
        n, k = (int(v) for v in shape.lower().split("x"))
        if kernel not in ("tgv", "cublas"):
            raise ValueError(f"kernel {kernel!r} (tgv or cublas)")
        rng = []
        for r in ranges.split(","):
            lo, hi = (int(v) for v in r.strip().split("-"))
            if lo < MIN_ROWS or hi > MAX_ROWS or hi < lo:
                raise ValueError(
                    f"range {lo}-{hi}: needs {MIN_ROWS} <= low <= high <= {MAX_ROWS}"
                )
            rng.append((lo, hi))
        rules.append((n, k, kernel == "tgv", tuple(rng)))
    if len({(n, k) for n, k, _, _ in rules}) != len(rules):
        raise ValueError("one entry per shape")
    return tuple(rules)


def pick(m: int, n: int, k: int, rules=None):
    """The override for a GEMM of ``m`` rows and weight ``[n, k]``: True (TGV), False (cuBLAS) or None (stock rule)."""
    for rn, rk, tgv, rng in _state["rules"] if rules is None else rules:
        if n == rn and k == rk and any(lo <= m <= hi for lo, hi in rng):
            return tgv
    return None


def _drop_pool_overlap(rules: tuple) -> tuple:
    try:
        from . import pool_tgv
    except ImportError:  # a tree without the pool_tgv component
        return rules
    if not pool_tgv._state.get("engaged"):
        return rules
    plo, phi = pool_tgv._state["range"]
    out = []
    for n, k, tgv, rng in rules:
        if (n, k) == POOL_SHAPE:
            keep = tuple((lo, hi) for lo, hi in rng if hi < plo or lo > phi)
            if keep != rng:
                bed.count(NAME, "pooler rows dropped: pool_tgv owns them")
                logger.warning(
                    "gemm_pick: pooler rows overlapping pool_tgv %d-%d dropped",
                    plo,
                    phi,
                )
            if not keep:
                continue
            rng = keep
        out.append((n, k, tgv, rng))
    return tuple(out)


def install(rules: tuple) -> None:
    import sglang.kernels.ops.gemm.cutedsl_bf16_gemm as G
    import sglang.srt.layers.quantization.unquant as U

    orig_use = G.use_cutedsl_bf16_gemm

    def use_cutedsl_bf16_gemm(m: int, n: int, k: int) -> bool:
        v = pick(m, n, k, rules)
        return orig_use(m, n, k) if v is None else v

    G.use_cutedsl_bf16_gemm = use_cutedsl_bf16_gemm
    # the dispatch binds the rule when its backend is configured; rebind if that already happened
    if getattr(U, "_use_cutedsl_bf16_gemm", None) is orig_use:
        U._use_cutedsl_bf16_gemm = use_cutedsl_bf16_gemm
    _state.update(rules=rules, orig_use=orig_use)


def engage() -> bool:
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
        rules = _drop_pool_overlap(
            parse_rules(os.environ.get("MUSE_VR_GEMM_PICK_RULES", DEFAULT_RULES))
        )
    except ValueError as e:
        bed.report(NAME, False, reason=f"bad MUSE_VR_GEMM_PICK_RULES ({e})")
        _state["engaged"] = False
        return False
    try:
        install(rules)
    except Exception as e:  # noqa: BLE001
        bed.report(NAME, False, reason=f"install failed: {type(e).__name__}: {e}")
        _state["engaged"] = False
        return False
    text = ";".join(
        f"{n}x{k}:{'tgv' if t else 'cublas'}:{','.join(f'{lo}-{hi}' for lo, hi in r)}"
        for n, k, t, r in rules
    )
    bed.report(NAME, True, rules=text)
    _state["engaged"] = True
    return True

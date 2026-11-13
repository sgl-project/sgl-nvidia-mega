# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Bounded mbarrier waits for the ``moe_x_kernels`` kernels.

CuTe DSL pipelines block in ``cute.arch.mbarrier_wait``. While one of this package's kernels is traced, that function
is replaced by a wait that issues at most ``MAX_POLLS`` ``mbarrier.try_wait`` attempts, each with a suspend-time hint
of ``SUSPEND_HINT_NS`` (PTX ISA, mbarrier.try_wait: a thread whose phase has not completed may be suspended up to the
hint), so a wait that never completes ends after at most MAX_POLLS x SUSPEND_HINT_NS; the thread then executes
``trap``. A synchronization bug therefore ends the kernel with a launch error (the host side then stops using these
kernels for the process) instead of leaving a kernel spinning on the device. No barrier instruction is added; the poll
loop runs only in the threads that already wait.

Test only: ``MUSE_VR_MOE_X_TEST_WAIT_SCALE`` (an integer in [1, 1_000_000], default 1) multiplies every poll bound of
this package (``scaled_bound``), for runs under instrumentation (sanitizers), where a correct wait can take longer than
the bounds allow. Unset or 1, the kernel text is unchanged; any other value is announced on stderr at import.
Test only: ``MUSE_VR_MOE_X_TEST_PLAIN_WAITS=1`` compiles the kernels with CuTe DSL's own (unbounded) pipeline waits instead
of the bounded form (a diagnostic build for sanitizer runs; announced on stderr at import). Unset, nothing changes.
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

WAIT_SCALE_ENV = "MUSE_VR_MOE_X_TEST_WAIT_SCALE"
_U32_MAX = 0xFFFFFFFF


def wait_scale(value: str | None = None) -> int:
    """Multiplier of every poll bound (``WAIT_SCALE_ENV``, or ``value`` when given); 1 when unset or empty."""
    raw = (os.environ.get(WAIT_SCALE_ENV, "") if value is None else value).strip()
    if not raw:
        return 1
    scale = int(raw)
    if not 1 <= scale <= 1_000_000:
        raise ValueError(f"{WAIT_SCALE_ENV}={raw}: expected an integer in [1, 1000000]")
    return scale


def scaled_bound(base: int, scale: int) -> int:
    """A poll bound multiplied by ``scale``, capped to the u32 counter the polls count in."""
    return min(base * scale, _U32_MAX)


TEST_WAIT_SCALE = wait_scale()
if TEST_WAIT_SCALE != 1:
    print(
        f"muse_vr_plugin: {WAIT_SCALE_ENV}={TEST_WAIT_SCALE} (test only): moe_x wait bounds x {TEST_WAIT_SCALE}",
        file=sys.stderr,
        flush=True,
    )
PLAIN_WAITS_ENV = "MUSE_VR_MOE_X_TEST_PLAIN_WAITS"
TEST_PLAIN_WAITS = os.environ.get(PLAIN_WAITS_ENV, "").strip() == "1"
if TEST_PLAIN_WAITS:
    print(
        f"muse_vr_plugin: {PLAIN_WAITS_ENV}=1 (test only): moe_x pipeline waits are CuTe DSL's own (unbounded)",
        file=sys.stderr,
        flush=True,
    )

# Bound of one wait: MAX_POLLS attempts, each suspending at most SUSPEND_HINT_NS while the phase is incomplete.
MAX_POLLS = scaled_bound(4000, TEST_WAIT_SCALE)
SUSPEND_HINT_NS = 2_500_000

_ASM = (
    "{\n\t"
    ".reg .pred p;\n\t"
    ".reg .u32 n;\n\t"
    "mov.u32 n, 0;\n\t"
    "WAIT_LOOP:\n\t"
    f"mbarrier.try_wait.parity.shared::cta.b64 p, [$0], $1, {SUSPEND_HINT_NS};\n\t"
    "@p bra WAIT_DONE;\n\t"
    "add.u32 n, n, 1;\n\t"
    f"setp.lt.u32 p, n, {MAX_POLLS};\n\t"
    "@p bra WAIT_LOOP;\n\t"
    "trap;\n\t"
    "WAIT_DONE:\n\t"
    "}\n"
)


@dsl_user_op
def bounded_mbarrier_wait(mbar_ptr, phase, *, loc=None, ip=None) -> None:
    """Drop-in for ``cute.arch.mbarrier_wait`` (CTA-shared mbarrier, parity wait) with a bounded number of attempts."""
    addr = llvm.ptrtoint(T.i32(), mbar_ptr.to_llvm_ptr(loc=loc, ip=ip), loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [addr, cutlass.Int32(phase).ir_value(loc=loc, ip=ip)],
        _ASM,
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


_lock = threading.RLock()


@contextlib.contextmanager
def bounded_waits():
    """Trace (compile) kernels inside this context to give every pipeline wait the bound (no change with the test-only
    plain-waits build)."""
    with _lock:
        if TEST_PLAIN_WAITS:
            yield
            return
        orig = cute.arch.mbarrier_wait
        cute.arch.mbarrier_wait = bounded_mbarrier_wait
        try:
            yield
        finally:
            cute.arch.mbarrier_wait = orig

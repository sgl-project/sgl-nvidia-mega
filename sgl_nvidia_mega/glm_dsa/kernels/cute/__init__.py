# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang Team
"""CuTe DSL kernels for GLM-5.x DSA decode and verify at 1-16 rows per step.

Kernels only: each engine's integration code decides eligibility, owns every
buffer and calls the family ``api`` modules; nothing here reads the environment
or imports an inference engine. ``cutlass_supported()`` is the version gate the
engines call before engaging any kernel: on a cutlass-dsl version this package
was not validated on, the engine declines to its stock path with one warning (it
never raises, also not on explicit opt-in).
"""

from __future__ import annotations

__all__ = ["VALIDATED_CUTLASS_DSL", "cutlass_supported"]

# cutlass-dsl versions the bodies, gates and SASS evidence were taken on.
VALIDATED_CUTLASS_DSL: tuple[str, ...] = ("4.8.0",)


def cutlass_supported() -> tuple[bool, str]:
    """(ok, reason): False when cutlass is missing or not a validated version."""
    try:
        import cutlass
    except Exception as e:  # noqa: BLE001
        return False, f"cutlass-dsl not importable ({type(e).__name__})"
    ver = str(getattr(cutlass, "__version__", ""))
    if ver not in VALIDATED_CUTLASS_DSL:
        validated = ", ".join(VALIDATED_CUTLASS_DSL)
        return False, (
            f"cutlass-dsl {ver or '<unknown>'} not validated (validated: {validated})"
        )
    return True, ""

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CuTe DSL kernels for GLM-5.x DSA decode and verify at 1-16 rows per step.

Kernels only: the plugin's glue decides where they run, owns every buffer and
calls the family ``api`` modules; nothing here reads the environment or imports
SGLang. ``cutlass_supported()`` is the version gate the plugin checks before
engaging any kernel: on a cutlass-dsl version this package was not validated on,
the plugin's spec refuses the flag on it.
"""

from __future__ import annotations

__all__ = ["VALIDATED_CUTLASS_DSL", "cutlass_supported"]

# cutlass-dsl versions the kernels were validated on.
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

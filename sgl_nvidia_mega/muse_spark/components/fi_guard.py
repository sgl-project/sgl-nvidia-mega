# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""FlashInfer version guard of the MoE components that call or build against FlashInfer without an SGLang source pin.

Components: ``moe_ld``, ``moe_ld_ops``, ``moe_ldw``, ``moe_ldtp``, ``lb_km2``, ``bf16_etp``, ``ep_balance``, ``bf16_ldw``. Their unit
records were taken on the FlashInfer versions listed in ``CLEARED``. ``check(component)`` (once per process and
component, at install or first build) reads the installed FlashInfer version and the batched-GEMM artifact it names
(``flashinfer.artifacts``: ``ArtifactPath.TRTLLM_GEN_BMM`` and ``CheckSumHash.TRTLLM_GEN_BMM``), compares them with
``ARTIFACTS`` for that version, and hashes the staged manifest (``<cubin dir>/<artifact path>/checksums.txt``), which
must equal the listed checksum. The class it records (install record ``fi_guard_<component>``) is:

- ``match``: the version is cleared for the component, and the artifact path, its checksum and the staged manifest are
  the listed ones;
- ``unverified``: anything else (version not cleared, unlisted version, other artifact, manifest missing or different).

``unverified`` raises at install when ``MUSE_VR_REQUIRE_ENGAGED=1``, unless the run names the version cleared
(``MUSE_VR_FI_CLEARED=<version>[,<version>...]``, set only by a run spec that cleared that version for these components)
or declares itself informational (``MUSE_VR_FI_INFO=1``; recorded, never a result). Without
``MUSE_VR_REQUIRE_ENGAGED=1`` the component stays as it is and the record says ``unverified``.

The JIT-built kernels need no FlashInfer define of their own: SGLang's JIT build cache already keys every build by the
FlashInfer package version and the content of every header it includes. (A define, even an unused one, changed the
generated code's register assignment in a measured build, so the build flags stay as they are.)
"""

from __future__ import annotations

import functools
import hashlib
import os
from pathlib import Path

ENV_REQUIRE = "MUSE_VR_REQUIRE_ENGAGED"
ENV_CLEARED = "MUSE_VR_FI_CLEARED"
ENV_INFO = "MUSE_VR_FI_INFO"

COMPONENTS = (
    "moe_ld",
    "moe_ld_ops",
    "moe_ldw",
    "moe_ldtp",
    "lb_km2",
    "bf16_etp",
    "ep_balance",
    "bf16_ldw",
)

# FlashInfer version -> (batched-GEMM artifact path, sha256 of its checksums.txt manifest), as each release names them
ARTIFACTS = {
    "0.7.1": (
        "4e73ccb74f333ecfebeac52abf3ebe1fa2dd34b3/batched_gemm-b738138-6923fec/",
        "8190fcb70b7661bf11cc5247b98d7fc55153386f969175c196a9e29d5801c7a1",
    ),
    "0.7.2": (
        "35cb99413a3ebce2e87a03fc04196e528e22c8d5/batched_gemm-fdba669-7e3d0a6/",
        "66b467a012dcefadbb56c993796397d81d2768a70b3dcd6235fe9e125b07971f",
    ),
}

# versions each component's unit records were taken on (others are cleared per run with MUSE_VR_FI_CLEARED)
CLEARED = {c: ("0.7.1",) for c in COMPONENTS}
# FlashInfer 0.7.2 (SGLang main image): unit records bitwise to its routed MoE at MMA-K auto (K = 128)
_CLEARED_072 = ("moe_ld", "moe_ld_ops", "moe_ldw", "bf16_ldw")
for _c in _CLEARED_072:
    CLEARED[_c] = ("0.7.1", "0.7.2")

_recorded: dict[str, str] = {}


@functools.lru_cache(maxsize=None)
def state() -> dict:
    """Installed FlashInfer version, the batched-GEMM artifact it names, the staged manifest's digest and the verdict on
    the artifact (``match`` or the reason it is not)."""
    st = {
        "flashinfer": "unknown",
        "bmm": "-",
        "bmm_sum": "-",
        "bmm_staged": "-",
        "artifact": "unknown",
    }
    try:
        import flashinfer

        st["flashinfer"] = str(getattr(flashinfer, "__version__", "unknown"))
        from flashinfer import artifacts

        st["bmm"] = str(artifacts.ArtifactPath.TRTLLM_GEN_BMM)
        st["bmm_sum"] = str(artifacts.CheckSumHash.TRTLLM_GEN_BMM)
    except Exception as e:  # noqa: BLE001 - recorded as unverified
        st["artifact"] = f"flashinfer artifacts unreadable: {type(e).__name__}"
        return st
    try:
        from flashinfer.jit import env as fi_env

        manifest = Path(fi_env.FLASHINFER_CUBIN_DIR) / st["bmm"] / "checksums.txt"
        st["bmm_staged"] = (
            hashlib.sha256(manifest.read_bytes()).hexdigest()
            if manifest.is_file()
            else "missing"
        )
    except Exception as e:  # noqa: BLE001 - recorded as unverified
        st["bmm_staged"] = f"unreadable: {type(e).__name__}"
    want = ARTIFACTS.get(st["flashinfer"])
    if want is None:
        st["artifact"] = "version not listed"
    elif (st["bmm"], st["bmm_sum"]) != want:
        st["artifact"] = "artifact differs from the listed one"
    elif st["bmm_staged"] != want[1]:
        st["artifact"] = (
            "staged manifest differs"
            if len(st["bmm_staged"]) == 64
            else f"staged manifest {st['bmm_staged']}"
        )
    else:
        st["artifact"] = "match"
    return st


def _env_cleared() -> set[str]:
    return {v.strip() for v in os.environ.get(ENV_CLEARED, "").split(",") if v.strip()}


def classify(component: str) -> tuple[str, str]:
    """(class, cleared_by) for ``component`` under the installed FlashInfer: ``match`` / ``unverified``, and ``code``,
    ``run`` (MUSE_VR_FI_CLEARED) or ``-``."""
    if component not in COMPONENTS:
        raise ValueError(f"fi_guard: unknown component {component!r}")
    st = state()
    ver = st["flashinfer"]
    by = (
        "code"
        if ver in CLEARED[component]
        else ("run" if ver in _env_cleared() else "-")
    )
    cls = "match" if by != "-" and st["artifact"] == "match" else "unverified"
    return cls, by


def check(component: str) -> str:
    """Record the class once per process and component; raise for ``unverified`` under MUSE_VR_REQUIRE_ENGAGED=1 unless
    the run is declared informational (MUSE_VR_FI_INFO=1)."""
    cls, by = classify(component)
    info = os.environ.get(ENV_INFO, "0").strip() == "1"
    if component not in _recorded:
        _recorded[component] = cls
        from . import bed

        st = state()
        bed.report(
            f"fi_guard_{component}",
            cls == "match",
            cls=cls,
            flashinfer=st["flashinfer"],
            cleared_by=by,
            artifact=st["artifact"],
            bmm=st["bmm"],
            bmm_sum=st["bmm_sum"][:16],
            bmm_staged=st["bmm_staged"][:16],
            info=info,
        )
    if cls != "match" and os.environ.get(ENV_REQUIRE, "0").strip() == "1" and not info:
        st = state()
        raise RuntimeError(
            f"fi_guard: {component} is 'unverified' on FlashInfer {st['flashinfer']} "
            f"(artifact: {st['artifact']}; cleared versions {sorted(CLEARED[component])}, run-cleared "
            f"{sorted(_env_cleared())}) and {ENV_REQUIRE}=1; set {ENV_CLEARED} only for a version the run "
            f"spec cleared, or {ENV_INFO}=1 for an informational run"
        )
    return cls

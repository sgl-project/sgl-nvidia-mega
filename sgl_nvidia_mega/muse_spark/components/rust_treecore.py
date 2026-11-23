# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``rust_treecore`` (env ``MUSE_VR_RUST_TREECORE=1``, default off): lets SGLang's Rust TreeCore load on the
pinned image, whose torch build is outside the Rust TreeCore's static support range, for one validated pair of torch
build and bundled Rust module only.

SGLang's default radix TreeCore backend is Rust (``SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND``, default ``rust``). On the
pinned image it never engages, for two independent reasons:

1. a static torch-version range (``rust_extensions/torch_build.py`` ``_MIN_SUPPORTED_TORCH`` .. ``_MAX_SUPPORTED_TORCH``)
   is checked by the backend registry (``tree_core_registry._rust_fallback_reason``, which then selects the Python
   TreeCore) and again by the extension loader (``rust_tree_core/extension.py``, which raises), and the image's torch
   minor version is above the range;
2. with ``SGLANG_RUST_BUILD_MODE=auto`` (the default), a source checkout without a Rust toolchain does not trust the
   module bundled in the package and also falls back to Python.

When every pin matches (the running torch build, the bundled module file, and the SGLang sources this mechanism relies
on), this component does two things in each process that loads plugins:

- it raises ``_MAX_SUPPORTED_TORCH`` to the running (major, minor), so the registry and the loader accept this build;
- it imports the bundled module under its own name. The registry and the loader both use a module that is already in
  ``sys.modules`` as it is, so ``SGLANG_RUST_BUILD_MODE`` stays unchanged and the other Rust extensions keep their own
  loading rules.

Backend selection stays SGLang's own: the run sets ``SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=rust`` (or leaves it
unset). On any pin mismatch or import error the component changes nothing and SGLang keeps its stock fallback.

Engagement: after each TreeCore is built the component records its class on that process (``tree_core=<class>``;
engaged means the Rust core was built). With ``MUSE_VR_REQUIRE_ENGAGED=1`` a TreeCore of another class, built while this
component is enabled, raises at cache construction, so the server does not start on the wrong core.

The TreeCore is host-side cache bookkeeping (prefix matching, insertion, locking, eviction of KV page indices). Both
cores implement the same interface and are checked against each other by SGLang's shared parity tests; this component
adds no arithmetic and no kernel.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import logging
import os
import re
import sys

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
}


logger = logging.getLogger(__name__)

NAME = "rust_treecore"  # env MUSE_VR_RUST_TREECORE
ENV_REQUIRE = "MUSE_VR_REQUIRE_ENGAGED"
MODULE = "sglang.srt.mem_cache.rust_tree_core.mem_cache"
RUST_CORE_CLASSES = ("RustUnifiedTreeCore",)
HOOK_TARGET = "sglang.srt.mem_cache.unified_cache.tree_core_registry.create_tree_core"

# The validated torch build: torch.__version__ and torch.version.git_version of the pinned image.
TORCH_VERSION = "2.15.0.dev20261005+cu134"
TORCH_GIT = "7bf52b3a9782d3440a17eb3f00983795f3b12af0"
# sha256 of the bundled module file (sglang/srt/mem_cache/rust_tree_core/mem_cache.cpython-312-aarch64-linux-gnu.so).
MODULE_SHA256 = "4537c75d00d1a46b2b6fd8b4afb1aa5c0dcae6e2cd56ac22b519e18b857e0390"
# SGLang sources whose behaviour this mechanism relies on (range check, registry fallback, loader shortcut, hook target).
PINS = {
    "sglang/srt/rust_extensions/torch_build.py": "36fce352fe9b3aa1b699f50f0a2ed09fa0d8a612d061384e7e177faf92b4126e",
    "sglang/srt/mem_cache/unified_cache/tree_core_registry.py": (
        "ab08c6584ab8d0b354a1ec616c81dd5745933a5c585a4c4b9a6c8b8f1c124118"
    ),
    "sglang/srt/mem_cache/rust_tree_core/extension.py": "bf6ed9f4eac2d01990388e79dd230769ee388211dbd9e09b5834c6fde981e550",
    "sglang/srt/mem_cache/rust_tree_core/__init__.py": "7f4c3105ec06d2ac587aacfde92517b6d49da95b6f2b3940b41aee22258d0fff",
    "sglang/srt/rust_extensions/loader.py": "0c25123a7e1c051a7f285abdb0b4ac6f4591bcf2cefe07b100bb55bc24137c6d",
    "sglang/srt/mem_cache/unified_radix_cache.py": "d9193bcb9e69073ba2dc337b2e8bdae7f9c94ca68d16a09d67ce0e2bf18822bd",
}

_state = {"armed": False, "info": {}}


def enabled() -> bool:
    return bed.enabled(NAME)


def _require() -> bool:
    return os.environ.get(ENV_REQUIRE, "").strip() == "1"


def _sha256(path: str) -> str | None:
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return None


def check_pins() -> tuple[bool, str, dict]:
    """(ok, reason, info). Reads files and versions only; changes nothing."""
    import torch

    info = {"torch": str(torch.__version__)}
    git = getattr(getattr(torch, "version", None), "git_version", None)
    if str(torch.__version__) != TORCH_VERSION or git != TORCH_GIT:
        return (
            False,
            f"torch {torch.__version__} ({git}) is not the validated build {TORCH_VERSION}",
            info,
        )
    for rel, want in PINS.items():
        if not bed.pinned(rel, want):
            return False, f"{rel} differs from the pinned revision", info
    spec = importlib.util.find_spec(MODULE)
    origin = getattr(spec, "origin", None) if spec is not None else None
    if not origin:
        return False, f"{MODULE} is not bundled in this installation", info
    got = _sha256(origin)
    info["module_sha16"] = (got or "<unreadable>")[:16]
    if got != MODULE_SHA256:
        return (
            False,
            f"bundled module sha256 {(got or '<unreadable>')[:16]} is not the validated {MODULE_SHA256[:16]}",
            info,
        )
    return True, "ok", info


def _running_major_minor() -> tuple[int, int]:
    import torch

    m = re.match(r"^(\d+)\.(\d+)", str(torch.__version__))
    return int(m.group(1)), int(m.group(2))


def arm() -> tuple[bool, str, dict]:
    """Check the pins and, only if they all match, widen the range and import the bundled module."""
    ok, why, info = check_pins()
    if not ok:
        return False, why, info
    import sglang.srt.rust_extensions.torch_build as torch_build

    running = _running_major_minor()
    before = tuple(torch_build._MAX_SUPPORTED_TORCH)
    if not (tuple(torch_build._MIN_SUPPORTED_TORCH) <= running):
        return (
            False,
            f"running torch {running} is below the Rust TreeCore minimum",
            info,
        )
    if before < running:
        torch_build._MAX_SUPPORTED_TORCH = running
    info["range_max"] = f"{before} -> {tuple(torch_build._MAX_SUPPORTED_TORCH)}"
    try:
        mod = importlib.import_module(MODULE)
    except Exception as e:  # noqa: BLE001 - restore the stock range and step aside
        torch_build._MAX_SUPPORTED_TORCH = before
        return (
            False,
            f"importing the bundled module failed: {type(e).__name__}: {e}",
            info,
        )
    if (
        _sha256(getattr(mod, "__file__", "") or "") != MODULE_SHA256
        or sys.modules.get(MODULE) is not mod
    ):
        torch_build._MAX_SUPPORTED_TORCH = before
        sys.modules.pop(MODULE, None)
        return False, "the imported module is not the pinned file", info
    return True, "ok", info


def _after_create(result, *args, **kwargs):
    """AFTER hook on create_tree_core: record which core this process built; refuse a non-Rust core under REQUIRE."""
    cls = type(result).__name__
    name = kwargs.get("name", args[0] if args else None)
    is_rust = cls in RUST_CORE_CLASSES
    bed.report(
        NAME,
        is_rust,
        phase="tree_core",
        tree_core=cls,
        requested=name,
        armed=_state["armed"],
        **_state["info"],
    )
    logger.warning(
        "muse_vr_plugin: %s tree_core=%s requested=%s pid=%d",
        NAME,
        cls,
        name,
        os.getpid(),
    )
    if not is_rust and _require():
        raise RuntimeError(
            f"muse_vr_plugin: {NAME} is enabled but this process built a {cls} TreeCore (requested {name!r}); "
            "set SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=rust or turn the component off"
        )
    return result


def register() -> bool:
    if not enabled():
        return False
    ok, why, info = arm()
    _state["info"] = info
    _state["armed"] = ok
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    # Registered even when arming failed: the record then shows the core actually built, and under
    # MUSE_VR_REQUIRE_ENGAGED=1 the cache build refuses the stock fallback instead of serving on it.
    HookRegistry.register(HOOK_TARGET, _after_create, HookType.AFTER)
    if not ok:
        bed.report(NAME, False, phase="armed", reason=why, **info)
        return False
    bed.report(NAME, True, phase="armed", **info)
    return True

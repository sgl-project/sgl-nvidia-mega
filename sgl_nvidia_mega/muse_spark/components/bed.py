# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Shared helpers for plugin components: env gating, SGLang source pins and engagement records.

``enabled(name)``
    A component is opted in with ``MUSE_VR_<NAME>=1`` (default off). ``MUSE_VR_PLUGIN_DISABLE=<name>[,<name>...]``
    forces it off even when opted in.
``pinned(relpath, sha256_hex)``
    True when the installed SGLang source file (``relpath`` relative to the directory that holds the ``sglang``
    package, e.g. ``sglang/srt/models/muse_spark.py``) has exactly this sha256. The package is located without
    importing it.
``report(name, engaged, **info)``
    Once per process and component: a WARNING-level log line ``muse_vr_plugin: <name> engaged=<bool> ...`` and, when
    ``MUSE_VR_ENGAGE_DIR`` is set, ``<dir>/install_<pid>.json`` holding every decision this process made. WARNING,
    because SGLang loads plugins before it configures logging in its worker processes, so INFO records emitted from
    ``register()`` are dropped by the logging module's last-resort handler.
``flag(name)`` / ``env_int(var, default)``
    Cached forms for per-call (hot eager path) checks: the environment is read once per process and name, since a
    server's environment does not change while it runs.
``count(name, key)``
    Host-side tally of fast-path call sites (e.g. per CUDA-graph capture shape), written to the same JSON record.
    Calls replayed from a captured graph do not reach the host and are not counted.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger("muse_vr_plugin")

_lock = threading.Lock()
_installs: dict[str, dict] = {}
_counts: dict[str, dict[str, int]] = {}


def disabled(name: str) -> bool:
    return name in {
        x.strip()
        for x in os.environ.get("MUSE_VR_PLUGIN_DISABLE", "").split(",")
        if x.strip()
    }


def enabled(name: str) -> bool:
    """Opt-in gate: MUSE_VR_<NAME>=1 and not listed in MUSE_VR_PLUGIN_DISABLE."""
    v = os.environ.get("MUSE_VR_" + name.upper(), "0").strip().lower()
    return v in ("1", "true", "yes", "on") and not disabled(name)


@functools.lru_cache(maxsize=None)
def flag(name: str) -> bool:
    """``enabled(name)``, read once per process."""
    return enabled(name)


@functools.lru_cache(maxsize=None)
def env_int(var: str, default: int) -> int:
    """Integer environment value (``default`` when unset or malformed), read once per process."""
    try:
        return int(os.environ.get(var, str(default)))
    except ValueError:
        return default


def sglang_root() -> Path | None:
    """Directory that contains the ``sglang`` package (found without importing it)."""
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(list(spec.submodule_search_locations)[0]).resolve().parent


def file_sha256(relpath: str) -> str | None:
    """sha256 of the pinned source ``relpath``, read at its installed location (``sgl_compat.relpath``)."""
    from . import sgl_compat

    root = sglang_root()
    if root is None:
        return None
    try:
        return hashlib.sha256(
            (root / sgl_compat.relpath(relpath)).read_bytes()
        ).hexdigest()
    except OSError:
        return None


def sha_ok(got: str | None, want) -> bool:
    """``want``: one sha256 hex digest, or a tuple of the digests of every validated release of that source; a tuple
    entry ``"absent"`` accepts a release that does not have the source at all (a reference the component does not
    import, removed upstream)."""
    if got is None:
        return not isinstance(want, str) and "absent" in want
    return got == want if isinstance(want, str) else got in want


def sha_label(want) -> str:
    return want[:16] if isinstance(want, str) else "|".join(w[:16] for w in want)


def pinned(relpath: str, sha256_hex) -> bool:
    """``sha256_hex``: one digest, or a tuple of the digests of every validated revision of that source."""
    got = file_sha256(relpath)
    if not sha_ok(got, sha256_hex):
        logger.warning(
            "muse_vr_plugin: %s sha256 %s does not match the pin %s",
            relpath,
            (got or "<missing>")[:16],
            sha_label(sha256_hex),
        )
        return False
    return True


def _dump() -> None:
    d = os.environ.get("MUSE_VR_ENGAGE_DIR")
    if not d:
        return
    try:
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"install_{os.getpid()}.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(
                {
                    "pid": os.getpid(),
                    "t": time.time(),
                    "installs": _installs,
                    "counts": _counts,
                },
                f,
                indent=1,
            )
        os.replace(tmp, path)
    except OSError as e:
        logger.warning(
            "muse_vr_plugin: cannot write engagement record to %s (%s)", d, e
        )


def report(name: str, engaged: bool, **info) -> None:
    """Record an install decision; logs once per process and component."""
    with _lock:
        first = name not in _installs
        _installs[name] = dict(info, engaged=bool(engaged), t=time.time())
        _dump()
    if first:
        extra = " ".join(f"{k}={v}" for k, v in info.items())
        logger.warning(
            "muse_vr_plugin: %s engaged=%s pid=%d %s",
            name,
            bool(engaged),
            os.getpid(),
            extra,
        )


def count(name: str, key: str = "calls", n: int = 1) -> None:
    with _lock:
        c = _counts.setdefault(name, {})
        new_key = key not in c
        c[key] = c.get(key, 0) + n
        if new_key:  # write on new keys only: no file I/O on hot eager paths
            _dump()


def shadow_result(name: str, ok: bool) -> None:
    """Tally a shadow (debug) comparison. Mismatches are tallied under a new key on the first one; matches also leave a
    ``match>=2^k`` key each time the tally reaches a power of two, so the engagement record shows progress."""
    key = "shadow:match" if ok else "shadow:MISMATCH"
    count(name, key)
    n = _counts.get(name, {}).get(key, 0)
    if ok and n & (n - 1) == 0:
        count(name, f"shadow:match>={n}")


def installs() -> dict:
    return {k: dict(v) for k, v in _installs.items()}


def counts() -> dict:
    return {k: dict(v) for k, v in _counts.items()}

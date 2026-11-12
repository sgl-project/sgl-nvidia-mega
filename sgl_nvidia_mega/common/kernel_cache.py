# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Persistent cache of the plugin's compiled CuTe DSL kernels: ``K31MK_KERNEL_CACHE=<dir>`` (default off).

cutlass-dsl 4.8 keeps nothing across processes on this path: ``cute.compile`` forces ``no_cache``
(``base_dsl/compiler.py``, ``CompileCallable._compile``), so the DSL's own file cache never runs and every
process traces and compiles every kernel again. ``compile`` exports each kernel it builds through the CuTe ABI
AOT path (``dump_to_object``); later processes load it with ``cute.runtime.load_module`` and skip both the trace
and the compile.

What decides the binary, and where each input is covered:
- **Key** (computed before tracing; anything the encoder cannot encode compiles uncached):
  - every file of this package and the CuTe DSL's Python sources by content (DSL libraries by size and mtime),
    FlashInfer's Python sources, the versions of the DSL, torch, CUDA Python and FlashInfer distributions, the
    Python, torch and CUDA driver versions;
  - the environment the plugin and the DSL read (``K31MK_*``, ``SGLANG_NVIDIA_MEGAKERNEL*``, ``CUTE_DSL_*``,
    ``CUTLASS_*``), read at every compile (a process may change it after its first compile);
  - the placement: torch.distributed rank and world size, ``RANK`` / ``LOCAL_RANK`` / ``WORLD_SIZE`` /
    ``LOCAL_WORLD_SIZE`` and the current CUDA device index, so no rank loads another rank's kernel;
  - the device (name, compute capability, i.e. the target arch, SM count, opt-in shared memory, L2 size);
  - the compiler, the kernel object and every argument and option, by value.
- **Trace record** (stored with the entry, checked at every load; a mismatch compiles again): every source
  file whose code ran while the kernel was traced and compiled, with its sha256, and every plain module-level
  value (numbers, strings, enums, dtypes and containers of them) that this code names and its module
  defines, with its value. Module values are recorded for this package and for code outside installed
  packages; installed packages (the DSL, torch, the standard library, site-packages) keep their own
  bookkeeping, and their sources are still recorded. This covers code outside the hashed trees and module
  values set at run time.
- **Entry check at load**: schema, key and every key part equal to this process's; the object's sha256 equal
  to the one recorded; the SM version of every cubin embedded in the object equal to this device's.
Entries are immutable: written to a private directory and published by one ``rename`` to
``<dir>/<key>/<variant>`` (the variant names the trace record and the object). Writers never touch a published
entry, so processes sharing a directory (one node or a shared filesystem) never see a partial or mixed entry;
of two racing writers, the second rename fails and its copy is dropped. Variants of one key coexist (e.g. two
source trees sharing a directory); a load takes the first that passes every check.

``K31MK_KERNEL_CACHE_VERIFY=1`` also traces every hit and raises unless its IR equals the IR recorded with
the entry: the audit for inputs no key part covers (state reached through calls into other objects, or class
attributes changed at run time)."""

from __future__ import annotations

import enum
import errno
import functools
import hashlib
import importlib
import importlib.metadata
import json
import logging
import os
import shutil
import site
import socket
import struct
import sys
import sysconfig
import tempfile
import time
import types
from pathlib import Path

import cutlass
import cutlass.cute as cute
import torch

logger = logging.getLogger(__name__)

# The DSL's own compilers, captured before install() replaces the module attributes.
_CUTE_COMPILE = cute.compile
_EXPERIMENTAL_COMPILE = cute.experimental.compile
# install() routes the compile calls made from modules under this prefix (kimi_k31 and its K3 base kimi_k31/k3).
ROUTED_CALLERS = "sgl_nvidia_mega.kimi_k31"

ROOT = os.environ.get("K31MK_KERNEL_CACHE") or None
VERIFY = os.environ.get("K31MK_KERNEL_CACHE_VERIFY", "0") == "1"
SCHEMA = "k31mk-kernel-cache-2"
ENV_PREFIXES = ("K31MK_", "SGLANG_NVIDIA_MEGAKERNEL", "CUTE_DSL_", "CUTLASS_")
ENV_IGNORED = ("K31MK_KERNEL_CACHE", "K31MK_KERNEL_CACHE_VERIFY")
PLACEMENT_ENV = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
# Packages whose sources the key hashes: only their classes and functions may appear in a key.
PACKAGES = ("sgl_nvidia_mega", "cutlass", "flashinfer")
DISTRIBUTIONS = (
    "nvidia-cutlass-dsl",
    "nvidia-cutlass-dsl-libs-base",
    "nvidia-cutlass-dsl-libs-core",
    "nvidia-cutlass-dsl-libs-cu12",
    "nvidia-cutlass-dsl-libs-cu13",
    "torch",
    "cuda-python",
    "cuda-bindings",
    "flashinfer-python",
)
PLAIN_LIMIT = 4096  # leaves; a larger module-level container is not recorded
STATS = dict(
    hit=0,
    miss=0,
    stored=0,
    raced=0,
    rejected=0,
    uncacheable=0,
    unstored=0,
    load_failed=0,
    verified=0,
    key_s=0.0,
    lookup_s=0.0,
    load_s=0.0,
    compile_s=0.0,
    store_s=0.0,
    verify_s=0.0,
)
REASONS: dict[str, int] = {}


class Uncacheable(Exception):
    pass


class Rejected(Exception):
    """An entry that fails a check at load: it is skipped and the kernel compiles again. A damaged entry
    (wrong key, key parts, object or arch) is also moved aside, so a good copy can be published in its place;
    an entry recorded for other sources or module values stays (it is valid for that state)."""

    def __init__(self, reason, damaged=True):
        super().__init__(reason)
        self.damaged = damaged


def compile(fn, *args, compiler=None, **kwargs):
    """``cute.compile(fn, *args, **kwargs)`` (or ``compiler``, e.g. ``cute.experimental.compile``),
    served from and stored to ``K31MK_KERNEL_CACHE`` when it is set."""
    compiler = _CUTE_COMPILE if compiler is None else compiler
    if ROOT is None:
        return compiler(fn, *args, **kwargs)
    start = time.perf_counter()
    try:
        key, parts = _key(compiler, fn, args, kwargs)
    except Uncacheable as e:
        _count("uncacheable", str(e))
        return compiler(fn, *args, **kwargs)
    finally:
        STATS["key_s"] += time.perf_counter() - start
    kdir = Path(ROOT) / key
    start = time.perf_counter()
    found = _lookup(kdir, key, parts)
    STATS["lookup_s"] += time.perf_counter() - start
    if found is not None:
        loaded, meta = found
        STATS["hit"] += 1
        if VERIFY:
            _verify(compiler, fn, args, kwargs, key, meta["ir"])
        return loaded
    STATS["miss"] += 1
    ir = []
    hook = lambda _o, m, _n: ir.append(_ir_digest(m))
    start = time.perf_counter()
    with _Record() as record:
        compiled = compiler(fn, *args, trace_finalize_hooks=hook, **kwargs)
    STATS["compile_s"] += time.perf_counter() - start
    start = time.perf_counter()
    try:
        _publish(kdir, key, parts, record.result(), compiled, ir[-1])
    except Exception as e:  # the compiled function is returned either way
        _count("unstored", f"{type(e).__name__}: {e}"[:160])
    STATS["store_s"] += time.perf_counter() - start
    return compiled


class _Routed:
    """Stands in for ``cute.compile`` / ``cute.experimental.compile`` after install(): a call made from a
    module under ROUTED_CALLERS goes through compile(); any other caller gets the DSL's compiler unchanged."""

    def __init__(self, compiler):
        self.compiler = compiler

    def __call__(self, fn, *args, **kwargs):
        caller = sys._getframe(1).f_globals.get("__name__", "")
        if caller.startswith(ROUTED_CALLERS):
            return compile(fn, *args, compiler=self.compiler, **kwargs)
        return self.compiler(fn, *args, **kwargs)

    def __getattr__(self, name):  # e.g. to_precompiled_mlir, or cute.compile[...] options
        return getattr(self.compiler, name)

    def __getitem__(self, item):
        return self.compiler[item]


def install():
    """With K31MK_KERNEL_CACHE set, serve every compile the plugin's kimi_k3 / kimi_k31 modules make from the
    cache, through one hook here instead of an edit at each compile site. Without it, nothing changes."""
    if ROOT is None or isinstance(cute.compile, _Routed):
        return
    cute.compile = _Routed(_CUTE_COMPILE)
    cute.experimental.compile = _Routed(_EXPERIMENTAL_COMPILE)
    logger.info("K31MK_KERNEL_CACHE=%s: compiles from %s.* are cached", ROOT, ROUTED_CALLERS)


def summary():
    return dict(STATS, root=ROOT, verify=VERIFY, reasons=dict(REASONS))


def log_summary(what):
    if ROOT is not None:
        logger.info(
            "K31MK_KERNEL_CACHE after %s: %s", what, json.dumps(summary(), default=str)
        )


def _count(kind, reason):
    STATS[kind] += 1
    reason = f"{kind}: {reason}"
    if reason not in REASONS:
        logger.info("K31MK_KERNEL_CACHE %s", reason)
    REASONS[reason] = REASONS.get(reason, 0) + 1


# ---- key ----------------------------------------------------------------------------------------------------


def _key(compiler, fn, args, kwargs):
    if compiler is _CUTE_COMPILE:
        name = "cute.compile"
    elif compiler is _EXPERIMENTAL_COMPILE:
        name = "cute.experimental.compile"
    else:
        raise Uncacheable(f"compiler {compiler!r}")
    if "--enable-tvm-ffi" in str(kwargs.get("options", "")):
        raise Uncacheable(
            "TVM-FFI functions load as tvm_ffi.Function, not as the compiled type"
        )
    device = torch.cuda.current_device()
    parts = dict(
        schema=SCHEMA,
        static=_static_key(),
        env=_env_key(),
        placement=_placement(device),
        device=_device_key(device),
        compiler=name,
    )
    out = [json.dumps(parts, sort_keys=True)]
    _encode((fn, args, sorted(kwargs.items())), out, {})
    return hashlib.sha256("\n".join(out).encode()).hexdigest(), parts


@functools.cache
def _static_key():
    h = hashlib.sha256(SCHEMA.encode())
    parts = STATS["static_s"] = {}
    clock = time.perf_counter()

    def lap(name):
        nonlocal clock
        parts[name] = round(time.perf_counter() - clock, 3)
        clock = time.perf_counter()

    _tree(h, Path(__file__).resolve().parents[1], lambda p: True)
    lap("plugin")
    _tree(h, _dsl_root(), lambda p: p.suffix == ".py")
    lap("dsl")
    try:
        import flashinfer

        root = Path(flashinfer.__file__).resolve().parent
        _tree(
            h, root, lambda p: True, pattern="*.py", skip=root / "data"
        )  # data/: CUDA sources and headers
    except ImportError:
        h.update(b"no flashinfer")
    lap("flashinfer")
    for name in DISTRIBUTIONS:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = "-"
        h.update(f"{name}={version}\n".encode())
    lap("versions")
    from cuda.bindings import driver

    status, version = driver.cuDriverGetVersion()
    h.update(
        f"python={sys.version}\ntorch={torch.__version__}/{torch.version.cuda}\n".encode()
    )
    h.update(f"driver={status}/{version}\n".encode())
    return h.hexdigest()


def _dsl_root():
    return Path(cutlass.__file__).resolve().parents[2]  # nvidia_cutlass_dsl/dsl_packages/cutlass


def _env_key():
    """The environment the plugin and the DSL read, at this compile."""
    env = sorted(
        (k, v)
        for k, v in os.environ.items()
        if k.startswith(ENV_PREFIXES) and k not in ENV_IGNORED
    )
    return hashlib.sha256(repr(env).encode()).hexdigest()


def _placement(device):
    """Which rank this process is. A kernel that bakes the rank in through state the encoder does not see
    (torch.distributed, the launcher's environment, the device index) still gets an entry of its own."""
    dist = torch.distributed
    group = None
    if dist.is_available() and dist.is_initialized():
        group = (dist.get_rank(), dist.get_world_size())
    return repr((group, tuple(os.environ.get(k) for k in PLACEMENT_ENV), device))


def _tree(h, root, by_content, pattern="*", skip=None):
    """Files under ``root`` (except ``skip``): their content where ``by_content``, else their size and mtime."""
    for path in sorted(root.rglob(pattern)):
        if (
            "__pycache__" in path.parts
            or not path.is_file()
            or (skip and path.is_relative_to(skip))
        ):
            continue
        rel = path.relative_to(root).as_posix().encode()
        if by_content(path):
            h.update(b"F" + rel + b"\0" + hashlib.sha256(path.read_bytes()).digest())
        else:
            stat = path.stat()
            h.update(b"S" + rel + b"\0%d:%d" % (stat.st_size, stat.st_mtime_ns))


@functools.cache
def _device_key(index):
    p = torch.cuda.get_device_properties(index)
    return repr(
        (
            p.name,
            p.major,
            p.minor,
            p.multi_processor_count,
            p.shared_memory_per_block_optin,
            p.L2_cache_size,
        )
    )


def _qualified(obj):
    module = getattr(obj, "__module__", None) or ""
    if module.split(".")[0] not in PACKAGES:
        raise Uncacheable(
            f"{module}.{getattr(obj, '__qualname__', type(obj).__name__)} is outside {PACKAGES}"
        )
    return f"{module}.{obj.__qualname__}"


def _encode(value, out, memo):
    """Append a deterministic encoding of ``value`` to ``out``; raise Uncacheable for what it cannot
    encode by value. ``memo`` (id -> (number, object)) numbers each container and object at its first
    visit, so a later visit (shared objects, cycles) encodes as a reference: the key records which
    arguments are the same object, which tracing can depend on."""
    kind = type(value)
    if value is None or kind in (bool, int, float, complex, str, bytes):
        out.append(f"{kind.__name__}:{value!r}")
        return
    if isinstance(value, cutlass.Numeric):  # e.g. Int32(0): its type and Python value
        if type(value.value) not in (bool, int, float):
            raise Uncacheable(f"{kind.__name__} without a Python value")
        out.append(f"numeric:{_qualified(kind)}:{value.value!r}")
        return
    if isinstance(value, type):
        builtin = value.__module__ == "builtins"
        out.append(f"type:{value.__qualname__ if builtin else _qualified(value)}")
        return
    if isinstance(value, enum.Enum):
        out.append(f"enum:{_qualified(kind)}.{value.name}")
        return
    if isinstance(value, (torch.dtype, torch.device)):
        out.append(f"torch:{value}")
        return
    if isinstance(value, torch.Tensor):
        raise Uncacheable("torch.Tensor")
    if id(value) in memo:
        out.append(f"ref:{memo[id(value)][0]}")
        return
    memo[id(value)] = (
        len(memo),
        value,
    )  # holding the object keeps its id unique while encoding
    if kind in (tuple, list):
        out.append(f"{kind.__name__}:{len(value)}")
        for item in value:
            _encode(item, out, memo)
    elif kind is dict:
        out.append(f"dict:{len(value)}")
        for (
            k,
            v,
        ) in value.items():  # insertion order: deterministic for the same construction
            _encode(k, out, memo)
            _encode(v, out, memo)
    elif kind in (set, frozenset):
        items = []
        for item in value:
            encoded = []
            _encode(item, encoded, memo)
            items.append("\n".join(encoded))
        out.append(f"{kind.__name__}:{len(items)}")
        out.extend(sorted(items))
    elif isinstance(value, functools.partial):
        out.append("partial")
        _encode((value.func, value.args, value.keywords), out, memo)
    elif isinstance(value, types.MethodType):
        out.append("method")
        _encode((value.__self__, value.__func__), out, memo)
    elif isinstance(value, types.FunctionType):
        while hasattr(value, "__wrapped__"):  # @cute.jit: the function it traces
            value = value.__wrapped__
        out.append(f"function:{_qualified(value)}")
        cells = tuple(_cell(c) for c in value.__closure__ or ())
        _encode((value.__defaults__, value.__kwdefaults__, cells), out, memo)
    else:
        name = _qualified(kind)
        slots = [
            s
            for c in kind.__mro__
            for s in _slots(c)
            if s not in ("__dict__", "__weakref__")
        ]
        if not slots and not hasattr(value, "__dict__"):
            raise Uncacheable(f"{name} has no attributes to encode")
        attrs = dict(vars(value)) if hasattr(value, "__dict__") else {}
        attrs.update((s, getattr(value, s)) for s in slots if hasattr(value, s))
        out.append(f"object:{name}:{len(attrs)}")
        for attr in sorted(attrs):
            out.append(attr)
            _encode(attrs[attr], out, memo)


def _cell(cell):
    try:
        return cell.cell_contents
    except ValueError:  # an empty cell
        return ("k31mk", "empty cell")


def _slots(cls):
    slots = cls.__dict__.get("__slots__", ())
    return (slots,) if isinstance(slots, str) else slots


# ---- trace record -------------------------------------------------------------------------------------------

_MONITOR = getattr(sys, "monitoring", None)
_ACTIVE: list = []
_TOOL = None


def _plain(value, budget=None):
    """Encoding of a plain value (numbers, strings, enums, dtypes, and tuples / lists / sets / dicts of
    them), or None for anything else (functions, classes, modules, objects, tensors, large containers)."""
    budget = [PLAIN_LIMIT] if budget is None else budget
    budget[0] -= 1
    if budget[0] < 0:
        return None
    kind = type(value)
    if value is None or kind in (bool, int, float, complex, str, bytes):
        return f"{kind.__name__}:{value!r}"
    if isinstance(value, enum.Enum):
        return f"enum:{kind.__module__}.{kind.__qualname__}.{value.name}"
    if isinstance(value, torch.dtype):
        return f"torch:{value}"
    if kind in (tuple, list, set, frozenset):
        items = [_plain(v, budget) for v in value]
        if any(i is None for i in items):
            return None
        if kind in (set, frozenset):
            items.sort()
        return f"{kind.__name__}[{','.join(items)}]"
    if kind is dict:
        items = []
        for k, v in value.items():
            ek, ev = _plain(k, budget), _plain(v, budget)
            if ek is None or ev is None:
                return None
            items.append(f"{ek}={ev}")
        return f"dict[{','.join(items)}]"
    return None


@functools.cache
def _library_roots():
    """Installed code, whose module values are its own bookkeeping, not inputs of our kernels."""
    roots = {str(_dsl_root()), str(Path(torch.__file__).resolve().parent)}
    paths = sysconfig.get_paths()
    for name in ("stdlib", "platstdlib", "purelib", "platlib"):
        if paths.get(name):
            roots.add(paths[name])
    try:
        roots.update(site.getsitepackages())
    except AttributeError:
        pass
    resolved = set()
    for r in roots:
        resolved.add(r)
        resolved.add(str(Path(r).resolve()))
    return tuple(r.rstrip("/") + "/" for r in resolved)


@functools.cache
def _plugin_root():
    return str(Path(__file__).resolve().parents[1]).rstrip("/") + "/"


def _records_values(filename):
    if filename.startswith("<"):  # frozen standard-library modules and generated code
        return False
    return filename.startswith(_plugin_root()) or not filename.startswith(
        _library_roots()
    )


class _Record:
    """Sources and plain module values touched while one kernel compiles. With ``sys.monitoring``
    (PY_START), each code object reports once per record, then its event is disabled until the next
    record; without it, a profile hook on this thread."""

    def __init__(self):
        self.files = set()
        self.codes = set()
        self.globals = {}
        self.ok = True

    def __enter__(self):
        global _TOOL
        if _MONITOR is None:
            sys.setprofile(_profile)
        else:
            if _TOOL is None:
                for tool in (3, 4):  # unnamed ids (0-2, 5: debugger, coverage, profiler, optimizer)
                    try:
                        _MONITOR.use_tool_id(tool, "k31mk-kernel-cache")
                    except ValueError:
                        continue
                    _MONITOR.register_callback(
                        tool, _MONITOR.events.PY_START, _on_start
                    )
                    _TOOL = tool
                    break
            if _TOOL is None:
                self.ok = False  # nothing can be recorded: the entry is not stored
                return self
            _MONITOR.set_events(_TOOL, _MONITOR.events.PY_START)
            _MONITOR.restart_events()
        _ACTIVE.append(self)
        return self

    def __exit__(self, *exc):
        if self in _ACTIVE:
            _ACTIVE.remove(self)
        if _MONITOR is None:
            if not _ACTIVE:
                sys.setprofile(None)
        elif _TOOL is not None and not _ACTIVE:
            _MONITOR.set_events(_TOOL, 0)
        return False

    def add(self, code, frame):
        if code in self.codes:
            return
        self.codes.add(code)
        filename = code.co_filename
        self.files.add(filename)
        if not _records_values(filename):
            return
        g = frame.f_globals
        if g is globals():  # this module's own bookkeeping (the record's exit runs while it records)
            return
        module = g.get("__name__")
        # Only names the module itself defines: a namespace built for a traced function (the DSL's
        # preprocessor) may add names of its own, which no later process could look up.
        namespace = getattr(sys.modules.get(module), "__dict__", None) if isinstance(module, str) else None
        if namespace is None:
            return
        for name in code.co_names:
            if name in g and name in namespace and (module, name) not in self.globals:
                encoded = _plain(g[name])
                if encoded is not None:
                    self.globals[(module, name)] = encoded

    def result(self):
        if not self.ok:
            raise RuntimeError("no free sys.monitoring tool id to record the trace")
        files = {}
        for name in sorted(self.files):
            digest = _file_digest(name)
            if digest is not None:
                files[name] = digest
        return dict(
            files=files,
            globals=sorted([m, n, v] for (m, n), v in self.globals.items()),
        )


def _on_start(code, _offset):
    if _ACTIVE:
        frame = sys._getframe(1)
        for record in _ACTIVE:
            record.add(code, frame)
    return _MONITOR.DISABLE


def _profile(frame, event, _arg):
    if event == "call":
        for record in _ACTIVE:
            record.add(frame.f_code, frame)


@functools.cache
def _file_digest(name):
    """sha256 of a source file, or None for code without one (``<string>``, frozen modules). Read once
    per process: the code a process runs does not change under it."""
    try:
        with open(name, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def _check_trace(trace):
    for name, digest in trace["files"].items():
        if _file_digest(name) != digest:
            raise Rejected(f"source changed: {name}", damaged=False)
    for module, name, encoded in trace["globals"]:
        if module not in sys.modules:  # imported during the trace: import it as the trace would
            try:
                importlib.import_module(module)
            except Exception as e:
                raise Rejected(f"module {module} does not import: {type(e).__name__}", damaged=False)
        namespace = getattr(sys.modules.get(module), "__dict__", None)
        if namespace is None or name not in namespace:
            raise Rejected(f"module value missing: {module}.{name}", damaged=False)
        if _plain(namespace[name]) != encoded:
            raise Rejected(f"module value changed: {module}.{name}", damaged=False)


# ---- entries ------------------------------------------------------------------------------------------------


def _ir_digest(module):
    return hashlib.sha256(
        module.operation.get_asm(enable_debug_info=False).encode()
    ).hexdigest()


def _cubin_sms(blob):
    """SM versions of the CUDA ELF images (e_machine EM_CUDA = 190) embedded in an exported object. The
    CUDA ELF ABI version (e_ident[8]) 8 keeps the SM in e_flags bits 8-15, older versions in bits 0-7."""
    sms = []
    start = 0
    while True:
        at = blob.find(b"\x7fELF", start)
        if at < 0:
            return sms
        start = at + 4
        if len(blob) < at + 64 or blob[at + 4] != 2:  # ELFCLASS64 headers are 64 bytes
            continue
        if struct.unpack_from("<H", blob, at + 18)[0] != 190:
            continue
        flags = struct.unpack_from("<I", blob, at + 48)[0]
        sms.append((flags >> 8) & 0xFF if blob[at + 8] >= 8 else flags & 0xFF)


def _device_sm():
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    return major * 10 + minor


def _lookup(kdir, key, parts):
    """The first published variant of ``key`` that passes every check, as (function, meta), or None."""
    try:
        variants = sorted(os.listdir(kdir))
    except FileNotFoundError:
        return None
    for variant in variants:
        if variant.startswith("."):  # a writer's private directory
            continue
        try:
            return _open_entry(kdir / variant, key, parts)
        except Rejected as e:
            _count("rejected", str(e)[:200])
            if e.damaged:
                try:
                    os.rename(kdir / variant, kdir / f".bad-{variant}-{os.getpid()}-{time.time_ns()}")
                except OSError:
                    pass  # another process moved it first
        except Exception as e:  # unreadable or unloadable: skipped, the kernel compiles again
            _count("load_failed", f"{type(e).__name__}: {e}"[:200])
    return None


def _open_entry(path, key, parts):
    start = time.perf_counter()
    dfd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    ofd = None
    try:
        with os.fdopen(os.open("meta.json", os.O_RDONLY, dir_fd=dfd)) as f:
            meta = json.load(f)
        if meta.get("schema") != SCHEMA or meta.get("key") != key:
            raise Rejected("schema or key differs")
        if meta.get("parts") != parts:
            raise Rejected("key parts differ")
        _check_trace(meta["trace"])
        ofd = os.open("object.o", os.O_RDONLY, dir_fd=dfd)
        with os.fdopen(os.dup(ofd), "rb") as f:
            blob = f.read()
        if hashlib.sha256(blob).hexdigest() != meta["object_sha256"]:
            raise Rejected("object sha256 differs from its entry")
        sms = _cubin_sms(blob)
        if sms != meta["cubin_sms"] or any(sm != _device_sm() for sm in sms):
            raise Rejected(f"cubin SM {sms} vs device {_device_sm()}")
        # load_module reads the inode just hashed: entries are never written after they are published.
        loaded = _load(f"/proc/self/fd/{ofd}", meta)
    finally:
        if ofd is not None:
            os.close(ofd)
        os.close(dfd)
    STATS["load_s"] += time.perf_counter() - start
    return loaded, meta


def _convention(fn, prefix=None):
    """What a call of ``fn`` depends on besides the object code (a loaded entry's host function
    carries the export prefix in its symbol name)."""
    a = fn.execution_args
    m = a._meta
    name = fn.function_name
    if prefix is not None:
        name = name.removeprefix(f"_mlir_{prefix}__mlir_ciface_")
    return json.dumps(
        dict(
            signature=str(a.signature),
            names=m.all_names,
            types=[repr(t) for t in m.annotated_types],
            numeric=m.numeric_flags,
            count=m.arg_count,
            specs=a._pointer_address_arg_specs,
            none=a._compiled_with_none,
            function=name,
            kernels=json.loads(json.dumps(fn.kernel_info)),
            extra=fn.kernel_extra_args,
            added=fn.total_added_arguments,
            gpu=fn.has_gpu_module,
            spec=fn._pyir_spec is None,
        ),
        sort_keys=True,
        default=repr,
    )


def _publish(kdir, key, parts, trace, compiled, ir):
    prefix = "k31mk_" + key[:24]
    blob = compiled.dump_to_object(prefix)
    sms = _cubin_sms(blob)
    if compiled.has_gpu_module and not sms:
        raise RuntimeError("no CUDA ELF image found in the exported object")
    meta = dict(
        schema=SCHEMA,
        key=key,
        parts=parts,
        prefix=prefix,
        ir=ir,
        object_sha256=hashlib.sha256(blob).hexdigest(),
        cubin_sms=sms,
        trace=trace,
        specs=compiled.execution_args._pointer_address_arg_specs,
        none=compiled.execution_args._compiled_with_none,
        extra=compiled.kernel_extra_args,
        added=compiled.total_added_arguments,
        writer=dict(host=socket.gethostname(), pid=os.getpid(), time=time.time()),
    )
    variant = hashlib.sha256(
        json.dumps([trace, meta["object_sha256"]], sort_keys=True).encode()
    ).hexdigest()[:24]
    os.makedirs(ROOT, mode=0o700, exist_ok=True)  # makedirs gives only the leaf its mode: the root first
    os.makedirs(kdir, mode=0o700, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".tmp-", dir=kdir))
    try:
        (tmp / "object.o").write_bytes(blob)
        reloaded = _load(str(tmp / "object.o"), meta)
        if _convention(reloaded, prefix) != _convention(compiled):
            raise RuntimeError(
                f"reloaded convention differs: {_convention(reloaded, prefix)} != {_convention(compiled)}"
            )
        (tmp / "meta.json").write_text(json.dumps(meta))
        try:
            os.rename(tmp, kdir / variant)  # publishes the whole entry at once
        except OSError as e:
            if e.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            STATS["raced"] += 1  # another process published this variant first
        else:
            STATS["stored"] += 1
            tmp = None
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)


def _load(path, meta):
    module = cute.runtime.load_module(str(path))
    fn = module[meta["prefix"]]
    args = fn.execution_args
    args._pointer_address_arg_specs = meta["specs"]
    args._compiled_with_none = meta["none"]
    args._has_pointer_address_arg_specs = any(
        spec is not None for spec in meta["specs"]
    )
    fn.kernel_extra_args = dict(meta["extra"])
    fn.total_added_arguments = meta["added"]
    fn.k31mk_module = (
        module  # the engine that holds the code lives as long as the function
    )
    return fn


def _verify(compiler, fn, args, kwargs, key, expected):
    start = time.perf_counter()
    ir = []
    hook = lambda _o, m, _n: ir.append(_ir_digest(m))
    try:
        compiler.to_precompiled_mlir(fn, *args, trace_finalize_hooks=hook, **kwargs)
    except Exception:
        if not ir:
            raise
    STATS["verify_s"] += time.perf_counter() - start
    if ir[-1] != expected:
        raise RuntimeError(
            f"K31MK_KERNEL_CACHE: entry {key} is stale: the traced IR differs from the IR it was built from"
        )
    STATS["verified"] += 1

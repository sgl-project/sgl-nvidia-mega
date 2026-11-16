# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""``K31MK_KERNEL_CACHE`` (``common/kernel_cache.py``): an input that changes a kernel never loads a stale entry,
and an entry that does not match this process (another rank, arch, kernel, or a damaged file) is never served.

Every "boot" is a fresh Python process over a copy of the plugin with a probe module added, against one cache
directory per test. Probe kernels write ``y = x + d`` over 128 int32 values, where ``d`` is a sum of integers the
kernel reads while it is traced (a constructor constant, a constant in the probe's source, a variable read at
import, a helper in a module outside the plugin, a module value set at run time, the rank, a variable read at trace
time, a class attribute). The expected ``d`` is computed here from those integers, independently of the code under
test, and checked exactly on every one of the 128 cells (``y`` is poisoned before the launch).

The probe module sits under ``sgl_nvidia_mega/kimi_k31/k3/`` and calls ``cute.compile`` like every plugin compile site;
the cache serves it only through the single hook ``kernel_cache.install()`` that ``kimi_k31/k3/__init__.py`` runs when
``K31MK_KERNEL_CACHE`` is set (no compile site calls the cache itself).

Tests (what each catches):
- ``constructor``: cold (miss, stored) -> warm (hit, same output) -> a new constructor constant must miss; the cache
  directory it creates, and every directory under it, is mode 700.
- ``module``: a constant edited in the probe's source must miss (kernel source covered).
- ``env``: a ``K31MK_*`` variable read at import must miss.
- ``tensor``: a kernel object holding a tensor is uncacheable (never stored).
- ``out_of_tree_source``: a helper module outside the plugin, edited between boots, must miss.
- ``runtime_global``: a probe module value set at run time before the compile must miss when it changes.
- ``rank_env`` / ``rank_dist``: a kernel that bakes the rank in from ``RANK`` / ``torch.distributed`` (two ranks
  sharing one directory, then ranks swapped across GPUs) must get its own rank's kernel.
- ``trace_env``: a variable set after the process's first compile and read at trace time must miss.
- ``device_key``: entries are not served to a device with another compute capability (arch).
- ``foreign_arch``: an entry whose cubin header names another SM, planted under this device's key, is refused.
- ``swapped_entries``: two entries swapped between keys are refused (the boot computes its own value).
- ``forged_object``: an object replaced by another kernel's object under the same symbol prefix is refused.
- ``verify_class_attr``: a class attribute changed at run time is the documented limit of the key (served stale
  without ``K31MK_KERNEL_CACHE_VERIFY``); with it the boot refuses the entry.
- ``route_scope``: the same probe outside ``kimi_k3`` (the frozen deepseek_v41 code, or any other caller) compiles
  uncached: correct output, no counter moves, no entry.
- ``route_off``: without ``K31MK_KERNEL_CACHE`` the DSL's ``cute.compile`` is untouched and no cache directory appears.
- ``concurrent``: 8 processes publish and read the same keys at once in one directory (node-local, and a shared
  filesystem when ``K31_KC_SHARED_DIR`` is set): every output exact, every published entry complete, no leftovers.

Mutations (each must make its test fail): see ``MUTATIONS``; the target test of each is in ``TARGETS``.
The tests use only the cache's public behaviour (outputs, the counters in ``summary()``, files on disk), so the same
file also runs against the first version of the cache (name-keyed objects, F4), as the control arm.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

import pytest

KC = "common/kernel_cache.py"
MUTATIONS = {
    "key_drops_object_state": (
        KC,
        "        for attr in sorted(attrs):\n",
        "        for attr in sorted(attrs)[:0]:\n",
    ),
    "sources_unchecked": [
        (KC, "    _tree(h, Path(__file__).resolve().parents[1], lambda p: True)\n", ""),
        (KC, "        if _file_digest(name) != digest:\n", "        if False:\n"),
    ],
    "env_unchecked": [
        (KC, "        if k.startswith(ENV_PREFIXES) and k not in ENV_IGNORED\n", "        if False\n"),
        (KC, "        if _plain(namespace[name]) != encoded:\n", "        if False:\n"),
    ],
    "key_encodes_tensors": (
        KC,
        '        raise Uncacheable("torch.Tensor")\n',
        '        out.append("tensor")\n        return\n',
    ),
    "trace_ignores_files": (KC, "        if _file_digest(name) != digest:\n", "        if False:\n"),
    "trace_ignores_globals": (KC, "        if _plain(namespace[name]) != encoded:\n", "        if False:\n"),
    "key_drops_placement": (KC, "        placement=_placement(device),\n", "        placement=None,\n"),
    "env_cached": (KC, "def _env_key():\n", "@functools.cache\ndef _env_key():\n"),
    "key_drops_device": (KC, "        device=_device_key(device),\n", "        device=None,\n"),
    "skip_arch_check": (
        KC,
        '        if sms != meta["cubin_sms"] or any(sm != _device_sm() for sm in sms):\n',
        "        if False:\n",
    ),
    "skip_key_check": (
        KC,
        '        if meta.get("schema") != SCHEMA or meta.get("key") != key:\n',
        "        if False:\n",
    ),
    "skip_object_hash": (
        KC,
        '        if hashlib.sha256(blob).hexdigest() != meta["object_sha256"]:\n',
        "        if False:\n",
    ),
    "verify_off": (KC, "    if ir[-1] != expected:\n", "    if False:\n"),
    "route_none": (KC, "        if caller.startswith(ROUTED_CALLERS):\n", "        if False:\n"),
    "route_everything": (KC, "        if caller.startswith(ROUTED_CALLERS):\n", "        if True:\n"),
    "install_noop": (KC, "    cute.compile = _Routed(_CUTE_COMPILE)\n", ""),
    "install_ignores_root": (
        KC,
        "    if ROOT is None or isinstance(cute.compile, _Routed):\n",
        "    if isinstance(cute.compile, _Routed):\n",
    ),
    "hook_not_called": ("kimi_k31/k3/__init__.py", "    kernel_cache.install()\n", "    pass\n"),
    "root_default_mode": (
        KC,
        "    os.makedirs(ROOT, mode=0o700, exist_ok=True)  # makedirs gives only the leaf its mode: the root first\n",
        "",
    ),
}
TARGETS = {
    "key_drops_object_state": "test_constructor",
    "sources_unchecked": "test_module",
    "env_unchecked": "test_env",
    "key_encodes_tensors": "test_tensor",
    "trace_ignores_files": "test_out_of_tree_source",
    "trace_ignores_globals": "test_runtime_global",
    "key_drops_placement": "test_rank_env or test_rank_dist",
    "env_cached": "test_trace_env",
    "key_drops_device": "test_device_key",
    "skip_arch_check": "test_foreign_arch",
    "skip_key_check": "test_swapped_entries",
    "skip_object_hash": "test_forged_object",
    "verify_off": "test_verify_class_attr",
    "root_default_mode": "test_constructor",
    "route_none": "test_constructor",
    "route_everything": "test_route_scope",
    "install_noop": "test_constructor",
    "install_ignores_root": "test_route_off",
    "hook_not_called": "test_constructor",
}

PROBE = """
import os

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream

import k31probe_ext as ext
from sgl_nvidia_mega.common import kernel_cache
from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.host import fake, stream, to_cute
from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.primitives import launch, thread

ENV_CONST = int(os.environ.get("K31MK_PROBE_ENV", "0"))
RUNTIME = 0  # set by the boot before it compiles (a module value set at run time)


def rank():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", "0"))


class AddConst:
    def __init__(self, c):
        self.c = c

    def constant(self):
        return {module_const} + ENV_CONST + self.c  # a literal in the code, not a module value

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, 1, 128, pdl=False)

    @cute.kernel
    def kernel(self, args):
        x, y = args
        tid, _, _, _ = thread()
        y[tid] = x[tid] + cutlass.Int32(self.constant())


class AddTable(AddConst):
    def __init__(self, table):
        self.table = table

    def constant(self):
        return {module_const} + ENV_CONST + int(self.table[0])


class AddExt(AddConst):
    def constant(self):
        return self.c + ext.value()


class AddRuntime(AddConst):
    def constant(self):
        return self.c + RUNTIME


class AddRank(AddConst):
    def constant(self):
        return self.c + 100 * rank()


class AddTraceEnv(AddConst):
    def constant(self):
        return self.c + int(os.environ.get("K31MK_PROBE_TRACE", "0"))


class AddClassAttr(AddConst):
    SCALE = 1

    def constant(self):
        return self.c * type(self).SCALE


KINDS = dict(const=AddConst, ext=AddExt, runtime=AddRuntime, rank=AddRank, trace_env=AddTraceEnv, cls=AddClassAttr)


def build(kind, value, options=None):
    kernel = AddTable(torch.tensor([value], dtype=torch.int32)) if kind == "table" else KINDS[kind](value)
    spec = lambda: fake((128,), None, torch.int32, 16)
    extra = dict(options=options) if options else dict()
    return cute.compile(kernel, spec(), spec(), make_fake_stream(), **extra)


def run(compiled, x, y):
    compiled(to_cute(x), to_cute(y), stream(x.device))
"""

EXT = "def value():\n    return {ext}\n"


def _boot_main(spec):
    """Child process: build (and run) the probe; print its output and the cache counters."""
    sys.path.insert(0, spec["ext"])
    sys.path.insert(0, spec["root"])
    import torch

    if spec.get("dist"):
        torch.distributed.init_process_group(
            "gloo",
            init_method=f"file://{spec['dist']}",
            rank=spec["rank"],
            world_size=spec["world"],
        )
    if "device" in spec:
        torch.cuda.set_device(spec["device"])
    if spec.get("fake_cc"):  # device properties report another compute capability (and name)
        real = torch.cuda.get_device_properties
        capability = torch.cuda.get_device_capability()  # torch derives it from the properties: keep the real one
        torch.cuda.get_device_capability = lambda device=None: capability

        class Props:
            def __init__(self, p):
                self._p = p

            def __getattr__(self, name):
                return getattr(self._p, name)

        def fake(index=None):
            p = Props(real(index if index is not None else torch.cuda.current_device()))
            p.major, p.minor = spec["fake_cc"]
            p.name = "fake device"
            return p

        torch.cuda.get_device_properties = fake

    import cutlass.cute as cute

    if spec.get("out_of_scope"):
        from sgl_nvidia_mega import _kcache_probe_out as probe
    else:
        from sgl_nvidia_mega.kimi_k31.k3 import _kcache_probe as probe
    from sgl_nvidia_mega.common import kernel_cache

    if spec.get("direct_install"):  # a caller that installs the hook itself, not through kimi_k31/k3/__init__.py
        kernel_cache.install()

    assert Path(probe.__file__).resolve().is_relative_to(Path(spec["root"]).resolve()), probe.__file__
    if spec.get("start_at"):  # concurrent boots start together
        time.sleep(max(0.0, spec["start_at"] - time.time()))
    if spec.get("first_compile"):  # the process's first compile, before the variable below is set
        probe.build("const", 0)
    for k, v in spec.get("set_env", {}).items():
        os.environ[k] = v
    probe.RUNTIME = spec.get("runtime", 0)
    probe.AddClassAttr.SCALE = spec.get("scale", 1)
    if spec.get("forge"):  # write a kernel's object under another entry's symbol prefix, run nothing
        compiled = probe.build(spec["kind"], spec["value"])
        Path(spec["forge"]["path"]).write_bytes(compiled.dump_to_object(spec["forge"]["prefix"]))
        print("BOOT " + json.dumps({"forged": True}), flush=True)
        return
    deltas = []
    for value in spec.get("values", [spec.get("value")]):
        compiled = probe.build(spec["kind"], value, spec.get("options"))
        if not spec.get("run", True):
            continue
        x = torch.arange(128, dtype=torch.int32, device="cuda") * 7 - 300
        y = torch.full_like(x, -(2**31))  # poisoned: a launch that writes nothing fails the exact check
        probe.run(compiled, x, y)
        torch.cuda.synchronize()
        deltas.append((y - x).tolist())
    routed = type(cute.compile).__name__ == "_Routed"
    print("BOOT " + json.dumps({"deltas": deltas, "stats": kernel_cache.summary(), "routed": routed}), flush=True)


class Probe:
    """A plugin copy with the probe module, an out-of-tree helper module, one cache directory, and boots."""

    def __init__(self, tmp, module_const=1, ext=10, cache=None):
        import _harness as H

        plugin = Path(H.load_plugin(MUTATIONS).__file__).resolve().parent  # mutated copy under K31_MUTATION
        self.root = tmp / "root"
        shutil.copytree(plugin, self.root / "sgl_nvidia_mega", ignore=shutil.ignore_patterns("__pycache__"))
        self.ext_dir = tmp / "ext"
        self.ext_dir.mkdir()
        self.cache = Path(cache) if cache else tmp / "cache"
        self.write(module_const)
        self.write_ext(ext)

    def write(self, module_const):
        self.module_const = module_const
        text = PROBE.format(module_const=module_const)
        (self.root / "sgl_nvidia_mega" / "kimi_k31" / "k3" / "_kcache_probe.py").write_text(text)
        (self.root / "sgl_nvidia_mega" / "_kcache_probe_out.py").write_text(text)  # outside the routed modules

    def write_ext(self, ext):
        self.ext = ext
        (self.ext_dir / "k31probe_ext.py").write_text(EXT.format(ext=ext))

    def spec(self, kind, value=None, **extra):
        return dict(root=str(self.root), ext=str(self.ext_dir), kind=kind, value=value, **extra)

    def environ(self, env=None):
        environ = {k: v for k, v in os.environ.items() if not k.startswith(("K31MK_", "CUTE_DSL_"))}
        for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE"):
            environ.pop(k, None)
        environ.update(PYTHONPATH=f"{self.root}:{environ.get('PYTHONPATH', '')}", K31MK_KERNEL_CACHE=str(self.cache))
        for k, v in (env or {}).items():  # None removes a variable
            if v is None:
                environ.pop(k, None)
            else:
                environ[k] = v
        return environ

    def start(self, spec, env=None):
        return subprocess.Popen(
            [sys.executable, __file__, "--boot", json.dumps(spec)],
            env=self.environ(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=lambda: os.umask(0o022),  # a server's usual umask (the test wrappers run with 077)
        )

    @staticmethod
    def finish(proc, expect_ok=True):
        out, err = proc.communicate(timeout=900)
        line = next((l for l in out.splitlines() if l.startswith("BOOT ")), None)
        if not expect_ok:
            return proc.returncode, out, err
        assert proc.returncode == 0 and line, f"boot failed rc={proc.returncode}\n{out[-3000:]}\n{err[-6000:]}"
        return json.loads(line[5:])

    def boot(self, kind, value=None, env=None, expect_ok=True, **extra):
        return self.finish(self.start(self.spec(kind, value, **extra), env), expect_ok)

    def entries(self):
        """{key: [(object, meta), ...]} in either layout: F4 (<key>.o + <key>.json) or <key>/<variant>/."""
        found = {}
        if not self.cache.exists():
            return found
        for p in sorted(self.cache.iterdir()):
            if p.is_file() and p.suffix == ".json":
                found.setdefault(p.stem, []).append((p.with_suffix(".o"), p))
            elif p.is_dir() and not p.name.startswith("."):
                for v in sorted(p.iterdir()):
                    if v.is_dir() and not v.name.startswith("."):
                        found.setdefault(p.name, []).append((v / "object.o", v / "meta.json"))
        return found


def delta(value):
    return [value] * 128


def hit_miss(boot):
    return boot["stats"]["hit"], boot["stats"]["miss"]


def st(boot):
    """The boot's counters and reasons, in full (pytest shortens dicts in assertion messages)."""
    return "\n" + json.dumps(boot["stats"], sort_keys=True, default=str)[:4000]


def test_constructor(tmp_path):
    p = Probe(tmp_path)
    cold = p.boot("const", 3)
    assert cold["deltas"] == [delta(p.module_const + 3)]
    assert (cold["stats"]["miss"], cold["stats"]["stored"], cold["stats"]["hit"]) == (1, 1, 0), st(cold)
    assert len(p.entries()) == 1
    # Compiled code: the directory the cache creates (it did not exist) and every directory under it is mode 700.
    dirs = [p.cache] + [d for d in p.cache.rglob("*") if d.is_dir()]
    assert all(d.stat().st_mode & 0o077 == 0 for d in dirs), [(str(d), oct(d.stat().st_mode)) for d in dirs]
    warm = p.boot("const", 3)
    assert warm["deltas"] == [delta(p.module_const + 3)]
    assert hit_miss(warm) == (1, 0), st(warm)
    changed = p.boot("const", 5)
    assert changed["deltas"] == [delta(p.module_const + 5)]  # a stale load would give 1 + 3
    assert hit_miss(changed) == (0, 1), st(changed)
    assert len(p.entries()) == 2


def test_route_scope(tmp_path):
    p = Probe(tmp_path)
    out = p.boot("const", 3, out_of_scope=True)
    assert out["deltas"] == [delta(p.module_const + 3)]
    s = out["stats"]
    assert out["routed"] and (s["miss"], s["hit"], s["stored"], s["uncacheable"]) == (0, 0, 0, 0), st(out)
    assert p.entries() == {}
    inside = p.boot("const", 3)
    assert (inside["stats"]["miss"], inside["stats"]["stored"]) == (1, 1), st(inside)


def test_route_off(tmp_path):
    p = Probe(tmp_path)
    out = p.boot("const", 3, env={"K31MK_KERNEL_CACHE": None})
    assert out["deltas"] == [delta(p.module_const + 3)]
    assert not out["routed"], "cute.compile replaced without K31MK_KERNEL_CACHE"
    assert out["stats"]["root"] is None and not p.cache.exists(), st(out)
    # install() itself must refuse without a cache root, whoever calls it (kimi_k31/k3/__init__.py guards it too)
    direct = p.boot("const", 3, env={"K31MK_KERNEL_CACHE": None}, direct_install=True)
    assert direct["deltas"] == [delta(p.module_const + 3)]
    assert not direct["routed"], "install() replaced cute.compile without K31MK_KERNEL_CACHE"


def test_module(tmp_path):
    p = Probe(tmp_path, module_const=1)
    assert p.boot("const", 3)["deltas"] == [delta(1 + 3)]
    p.write(module_const=2)
    edited = p.boot("const", 3)
    assert edited["deltas"] == [delta(2 + 3)]  # a stale load would give 1 + 3
    assert hit_miss(edited) == (0, 1), st(edited)


def test_env(tmp_path):
    p = Probe(tmp_path)
    assert p.boot("const", 3)["deltas"] == [delta(1 + 3)]
    changed = p.boot("const", 3, env={"K31MK_PROBE_ENV": "10"})
    assert changed["deltas"] == [delta(1 + 10 + 3)]
    assert hit_miss(changed) == (0, 1), st(changed)


def test_tensor(tmp_path):
    p = Probe(tmp_path)
    first = p.boot("table", 3)
    assert first["deltas"] == [delta(1 + 3)]
    assert (first["stats"]["uncacheable"], first["stats"]["stored"]) == (1, 0), st(first)
    second = p.boot("table", 9)
    assert second["deltas"] == [delta(1 + 9)]  # a stale load would give 1 + 3
    assert p.entries() == {}


def test_out_of_tree_source(tmp_path):
    p = Probe(tmp_path, ext=10)
    assert p.boot("ext", 3)["deltas"] == [delta(10 + 3)]
    warm = p.boot("ext", 3)
    assert warm["deltas"] == [delta(10 + 3)] and hit_miss(warm) == (1, 0), st(warm)
    p.write_ext(20)  # same function name, new body, outside the plugin
    edited = p.boot("ext", 3)
    assert edited["deltas"] == [delta(20 + 3)]  # a stale load would give 10 + 3
    assert hit_miss(edited) == (0, 1), st(edited)


def test_runtime_global(tmp_path):
    p = Probe(tmp_path)
    assert p.boot("runtime", 3, runtime=40)["deltas"] == [delta(40 + 3)]
    warm = p.boot("runtime", 3, runtime=40)
    assert warm["deltas"] == [delta(40 + 3)] and hit_miss(warm) == (1, 0), st(warm)
    changed = p.boot("runtime", 3, runtime=50)
    assert changed["deltas"] == [delta(50 + 3)]  # a stale load would give 40 + 3
    assert hit_miss(changed) == (0, 1), st(changed)


def test_rank_env(tmp_path):
    p = Probe(tmp_path)
    assert p.boot("rank", 3, env={"RANK": "0"})["deltas"] == [delta(3)]
    r1 = p.boot("rank", 3, env={"RANK": "1"})
    assert r1["deltas"] == [delta(103)], st(r1)  # rank 0's kernel would give 3
    r0 = p.boot("rank", 3, env={"RANK": "0"})
    assert r0["deltas"] == [delta(3)] and hit_miss(r0) == (1, 0), st(r0)


def test_rank_dist(tmp_path):
    """Two torch.distributed ranks share the directory (cold, then warm), then swap GPUs when the node has two."""
    import torch

    p = Probe(tmp_path)
    devices = list(range(torch.cuda.device_count()))
    layouts = [[0, 0]] if len(devices) < 2 else [[0, 1], [1, 0]]
    rounds = [("cold", layouts[0]), ("warm", layouts[0])] + [("swapped", l) for l in layouts[1:]]
    for n, (what, gpus) in enumerate(rounds):
        store = tmp_path / f"dist_{n}"
        procs = [
            p.start(p.spec("rank", 3, dist=str(store), rank=r, world=2, device=gpus[r])) for r in range(2)
        ]
        boots = [p.finish(proc) for proc in procs]
        for r, b in enumerate(boots):
            assert b["deltas"] == [delta(3 + 100 * r)], (what, r, b["deltas"][0][:4], st(b))
        if what == "warm":
            assert all(hit_miss(b) == (1, 0) for b in boots), [st(b) for b in boots]


def test_trace_env(tmp_path):
    p = Probe(tmp_path)
    first = p.boot("trace_env", 3, first_compile=True, set_env={"K31MK_PROBE_TRACE": "5"})
    assert first["deltas"] == [delta(5 + 3)]
    second = p.boot("trace_env", 3, first_compile=True, set_env={"K31MK_PROBE_TRACE": "7"})
    assert second["deltas"] == [delta(7 + 3)]  # a key taken at the first compile would serve 5 + 3


def test_device_key(tmp_path):
    import torch

    p = Probe(tmp_path)
    assert p.boot("const", 3)["deltas"] == [delta(1 + 3)]
    major, minor = torch.cuda.get_device_capability()
    other = [10, 7] if (major, minor) != (10, 7) else [10, 3]
    # Same process state, but the device properties report another compute capability: the entry must not be
    # looked up under it (the kernel is only built, not run). get_device_capability, which the load-time arch
    # check reads, keeps the real value, so only the key can refuse the entry (the arch check is test_foreign_arch).
    pretend = p.boot("const", 3, fake_cc=other, run=False)
    assert hit_miss(pretend) == (0, 1), st(pretend)


def _cuda_elf_headers(blob):
    """Offsets of the CUDA ELF images (ELFCLASS64, e_machine EM_CUDA = 190) inside an exported object."""
    found, start = [], 0
    while (at := blob.find(b"\x7fELF", start)) >= 0:
        start = at + 4
        if len(blob) >= at + 64 and blob[at + 4] == 2 and struct.unpack_from("<H", blob, at + 18)[0] == 190:
            found.append(at)
    return found


def _sm_field(blob, at):
    """(shift, sm) of the SM number in e_flags: bits 8-15 for CUDA ELF ABI version >= 8, else bits 0-7."""
    flags = struct.unpack_from("<I", blob, at + 48)[0]
    shift = 8 if blob[at + 8] >= 8 else 0
    return shift, (flags >> shift) & 0xFF


def test_foreign_arch(tmp_path):
    """An entry whose embedded cubin is for another SM, planted under this device's key, is refused.

    cutlass-dsl 4.8 cannot export a cross-compiled function (DSLRuntimeError in dump_to_object, u2), so the foreign
    entry is the native one with the SM field of every embedded cubin header rewritten (sm_103 -> sm_100, or
    sm_107 -> sm_103), and its other fields made consistent with the new bytes, so that only an arch check can
    refuse it. The native object's SM must first read as this device's SM: ground truth for the header format.
    """
    import torch

    p = Probe(tmp_path)
    assert p.boot("const", 3)["deltas"] == [delta(1 + 3)]
    ((key, variants),) = p.entries().items()
    ((obj, meta_path),) = variants
    blob = bytearray(obj.read_bytes())
    headers = _cuda_elf_headers(blob)
    major, minor = torch.cuda.get_device_capability()
    assert headers and all(_sm_field(blob, at)[1] == major * 10 + minor for at in headers), headers
    foreign = 100 if major * 10 + minor != 100 else 103
    for at in headers:
        shift, _ = _sm_field(blob, at)
        flags = struct.unpack_from("<I", blob, at + 48)[0]
        struct.pack_into("<I", blob, at + 48, (flags & ~(0xFF << shift)) | (foreign << shift))
    obj.write_bytes(bytes(blob))
    meta = json.loads(meta_path.read_text())
    if "object_sha256" in meta:  # keep the entry self-consistent (fields of the new cache's entries)
        meta["object_sha256"] = hashlib.sha256(bytes(blob)).hexdigest()
        meta["cubin_sms"] = [foreign] * len(headers)
        meta_path.write_text(json.dumps(meta))
    served = p.boot("const", 3, expect_ok=False)
    rc, out, err = served
    line = next((l for l in out.splitlines() if l.startswith("BOOT ")), None)
    assert rc == 0 and line, f"boot failed rc={rc}\n{err[-3000:]}"
    served = json.loads(line[5:])
    assert served["deltas"] == [delta(1 + 3)], st(served)
    assert hit_miss(served) == (0, 1), st(served)


def test_swapped_entries(tmp_path):
    p = Probe(tmp_path)
    p.boot("const", 3)
    (k3,) = p.entries()
    p.boot("const", 5)
    (k5,) = set(p.entries()) - {k3}
    (o3, m3), = p.entries()[k3]
    (o5, m5), = p.entries()[k5]
    for a, b in ((o3, o5), (m3, m5)):  # k3's entry now holds c = 5's kernel and vice versa
        tmp = a.with_name(a.name + ".swap")
        a.rename(tmp)
        b.rename(a)
        tmp.rename(b)
    served = p.boot("const", 3)
    assert served["deltas"] == [delta(1 + 3)], st(served)  # the swapped entry would give 1 + 5


def test_forged_object(tmp_path):
    p = Probe(tmp_path)
    p.boot("const", 3)
    (k3,) = p.entries()
    (o3, m3), = p.entries()[k3]
    prefix = json.loads(m3.read_text())["prefix"]
    forged = tmp_path / "forged.o"
    p.boot("const", 5, forge=dict(path=str(forged), prefix=prefix))  # c = 5's kernel under k3's symbols
    shutil.copyfile(forged, o3)
    served = p.boot("const", 3)
    assert served["deltas"] == [delta(1 + 3)], st(served)  # the forged object would give 1 + 5


def test_verify_class_attr(tmp_path):
    p = Probe(tmp_path)
    assert p.boot("cls", 3, scale=1)["deltas"] == [delta(3)]
    stale = p.boot("cls", 3, scale=2)
    # The documented limit: a class attribute set at run time is in neither the key nor the trace record.
    assert stale["deltas"] == [delta(3)] and hit_miss(stale) == (1, 0), st(stale)
    rc, out, err = p.boot("cls", 3, scale=2, env={"K31MK_KERNEL_CACHE_VERIFY": "1"}, expect_ok=False)
    assert rc != 0 and "is stale" in err, err[-3000:]
    ok = p.boot("cls", 3, scale=1, env={"K31MK_KERNEL_CACHE_VERIFY": "1"})
    assert ok["deltas"] == [delta(3)] and ok["stats"]["verified"] == 1, st(ok)


def _shared_dir():
    root = os.environ.get("K31_KC_SHARED_DIR")
    return None if not root else Path(root) / f"t{os.getpid()}_{socket.gethostname()}_{time.time_ns()}"


@pytest.mark.parametrize("where", ["local", "shared"])
def test_concurrent(tmp_path, where):
    if where == "shared":
        cache = _shared_dir()
        if cache is None:
            pytest.skip("K31_KC_SHARED_DIR is not set (shared-filesystem case not run)")
    else:
        cache = None
    p = Probe(tmp_path, cache=cache)
    values = [3, 5, 7]
    expect = [delta(1 + v) for v in values]
    try:
        for wave in ("cold", "warm"):
            start_at = time.time() + 25
            procs = [p.start(p.spec("const", values=values, start_at=start_at)) for _ in range(8)]
            boots = [p.finish(proc) for proc in procs]
            for b in boots:
                assert b["deltas"] == expect, st(b)
            if wave == "warm":
                assert all(hit_miss(b) == (3, 0) for b in boots), [st(b) for b in boots]
            entries = p.entries()
            assert len(entries) == 3, list(entries)
            for key, variants in entries.items():
                for obj, meta in variants:
                    assert obj.is_file() and meta.is_file() and json.loads(meta.read_text()), (key, obj)
        leftovers = [q for q in p.cache.rglob(".tmp*")] + [q for q in p.cache.glob("*.tmp*")]
        assert leftovers == [], leftovers
    finally:
        if cache is not None:
            shutil.rmtree(cache, ignore_errors=True)


if __name__ == "__main__" and sys.argv[1:2] == ["--boot"]:
    _boot_main(json.loads(sys.argv[2]))

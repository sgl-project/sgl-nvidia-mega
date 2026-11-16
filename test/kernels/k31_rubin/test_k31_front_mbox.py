# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MBOX (``K31MK_MOE_FRONT_MBOX``, ``moe/front.py`` front_tmem_mbox): CPU checks, no GPU.

Modes (tree = the plugin checkout on PYTHONPATH; tools/rubin_k31/tp16_compile.py is its CPU shim):
  unit                 knob parsing and engagement lines: unset / empty = off, a batch table with rows 9-64 and value 0 / 1,
                       every listed batch must be a K31MK_MOE_FRONT_TMEM batch, one stderr line per engaged batch per process.
  compile MBOX OUT     sm_107a CPU compile (--keep-ptx) of the served FTM front (``compile_front_tmem``: split 64, valid 56, shard 224,
                       hidden 384, promote 2 = the c=8 served form) with mbox MBOX (0 / 1; 'base' = the base tree's
                       signature without the argument); the PTX lands in OUT.
  check OFF ON         ON must drop cluster barrier phases (fewer barrier.cluster.arrive and .wait than OFF, >= 1 left
                       for the entry init) and leave every other instruction class's count unchanged except shared-memory
                       addressing; exit 1 on a miss.
The bitwise / timing evidence is the GPU rig (not included); this file proves the knob reaches the kernel and that the
default-off PTX equals the base tree's (the caller compares normalized md5s)."""

import contextlib
import importlib.util
import io
import os
import re
import sys
import types


def tree_root():
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))


def load_shim():
    spec = importlib.util.spec_from_file_location("tp16c", os.path.join(tree_root(), "tools/rubin_k31/tp16_compile.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if importlib.util.find_spec("flashinfer") is None:  # CPU venv without FlashInfer
        name = "flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk"
        for parent in ("flashinfer", "flashinfer.gemm", "flashinfer.gemm.kernels"):
            sys.modules.setdefault(parent, types.ModuleType(parent))
        src = os.environ.get("FI_SPLITK_SRC")  # compile mode: that module's source file (SplitKTactic, DSMEM store)
        if src:
            spec = importlib.util.spec_from_file_location(name, src)
            stub = importlib.util.module_from_spec(spec)
            sys.modules[name] = stub
            spec.loader.exec_module(stub)
        else:  # unit mode: names imported but unused
            stub = types.ModuleType(name)
            stub.OWNER_RANK, stub.SplitKTactic, stub._store_shared_remote_v4 = 0, object, None
            sys.modules[name] = stub
    import torch

    torch.cuda.get_device_capability = lambda *_: (10, 7)  # CPU node: the compile target is sm_107a (CUTE_DSL_ARCH)
    return module


def _front():
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import front

    return front


FTM = set(range(16, 33))  # K31MK_MOE_FRONT_TMEM=16-32:1 (the served c=8 table)


def _call(value, ftm, used):
    front = _front()
    if value is None:
        os.environ.pop("K31MK_MOE_FRONT_MBOX", None)
    else:
        os.environ["K31MK_MOE_FRONT_MBOX"] = value
    front._mbox_engaged.clear()
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        out = front.front_tmem_mbox(ftm, used)
    return out, [x for x in err.getvalue().splitlines() if x.startswith("K31MK_MOE_FRONT_MBOX engaged")]


def test_unset_and_empty_are_off():
    for value in (None, "", "16-32:0"):
        on, lines = _call(value, FTM, {16, 24, 32})
        assert on == set() and lines == [], (value, on, lines)


def test_table_used_batches_and_one_line_each():
    on, lines = _call("16-32:1", FTM, {16, 24, 32})  # graph batches 16 / 24 / 32 of the 16-32 table
    assert on == {16, 24, 32}, on
    assert [x.split()[2] for x in lines] == ["batch=16", "batch=24", "batch=32"], lines
    front = _front()
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        again = front.front_tmem_mbox(FTM, {16, 24, 32})
    assert again == on and "engaged" not in err.getvalue(), err.getvalue()
    on, lines = _call("24:1", FTM, {16, 24, 32})
    assert on == {24} and len(lines) == 1, (on, lines)


def test_bad_values_raise():
    for value, ftm in (("8:1", {8}), ("65:1", {65}), ("16:2", FTM), ("64:1", FTM), ("16-40:1", FTM)):
        try:
            _call(value, ftm, ftm)
        except ValueError:
            continue
        raise AssertionError(f"no ValueError for {value!r} with FTM batches {sorted(ftm)}")


def run_unit():
    load_shim()
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"UNIT PASS {name}", flush=True)
        except Exception as error:  # noqa: BLE001 - report every test
            failed += 1
            print(f"UNIT FAIL {name}: {type(error).__name__}: {str(error)[:300]}", flush=True)
    os.environ.pop("K31MK_MOE_FRONT_MBOX", None)
    print(f"UNIT {len(tests) - failed}/{len(tests)} passed", flush=True)
    return 1 if failed else 0


def run_compile(mbox, out):
    load_shim()
    front = _front()
    import sgl_nvidia_mega

    print(f"TREE {os.path.dirname(os.path.dirname(os.path.abspath(sgl_nvidia_mega.__file__)))}", flush=True)
    dump = os.environ["CUTE_DSL_DUMP_DIR"]
    import cutlass.cute as cute

    plain = cute.experimental.compile  # compile_front_tmem passes options; CUTE_DSL_KEEP does not reach that path

    def keep(*args, options="", **kw):
        return plain(*args, options=f"{options} --keep-ptx --dump-dir={dump}", **kw)

    cute.experimental.compile = keep
    kwargs = {} if mbox == "base" else {"mbox": int(mbox)}
    front.compile_front_tmem(64, 56, 224, 384, 0, promote=2, **kwargs)
    os.makedirs(out, exist_ok=True)
    moved = 0
    for f in os.listdir(dump):
        if f.endswith(".ptx"):
            os.replace(os.path.join(dump, f), os.path.join(out, f))
            moved += 1
    print(f"COMPILE OK mbox={mbox} files={moved}", flush=True)
    return 0 if moved >= 1 else 1


TOKENS = re.compile(r"\b(barrier\.cluster\.arrive|barrier\.cluster\.wait|st\.async|mbarrier\.try_wait|tcgen05\.mma|ld\.shared|"
                    r"griddepcontrol)")


def census(d):
    """Occurrences of each instruction token in the PTX (several may share one line inside an inline-asm block)."""
    counts = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".ptx"):
            for line in open(os.path.join(d, f)):
                code = line.split("//", 1)[0]
                for tok in TOKENS.findall(code):
                    counts[tok] = counts.get(tok, 0) + 1
    return counts


def run_check(off, on):
    a, b = census(off), census(on)
    arr = lambda c: sum(v for k, v in c.items() if k.startswith("barrier.cluster.arrive"))  # noqa: E731
    wai = lambda c: sum(v for k, v in c.items() if k.startswith("barrier.cluster.wait"))  # noqa: E731
    ok = arr(b) < arr(a) and wai(b) < wai(a) and arr(b) >= 1 and wai(b) >= 1
    print(f"CHECK cluster arrive off/on {arr(a)}/{arr(b)}, wait off/on {wai(a)}/{wai(b)}: {'PASS' if ok else 'FAIL'}", flush=True)
    keep = ("st.async", "mbarrier.try_wait", "tcgen05.mma", "ld.shared", "griddepcontrol")
    for k in keep:
        ka = sum(v for x, v in a.items() if x.startswith(k))
        kb = sum(v for x, v in b.items() if x.startswith(k))
        same = ka == kb
        ok &= same
        print(f"CHECK {k} off/on {ka}/{kb}: {'PASS' if same else 'FAIL'}", flush=True)
    print(f"CHECK {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "unit"
    if mode == "unit":
        sys.exit(run_unit())
    if mode == "compile":
        sys.exit(run_compile(sys.argv[2], sys.argv[3]))
    if mode == "check":
        sys.exit(run_check(sys.argv[2], sys.argv[3]))
    raise SystemExit(f"unknown mode {mode}")

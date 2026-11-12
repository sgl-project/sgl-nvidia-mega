# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K-19 trigger fences (``K31MK_K19_FENCE``, ``common/k19.py``): CPU checks, no GPU.

Three modes (tree = the plugin checkout on PYTHONPATH; tools/rubin_k31/tp16_compile.py is its CPU shim):
  unit                        knob parsing, engagement lines, and the host wiring of every site (no compile):
                              ``oproj.prepared.producer`` sets the fence on the split / tcgen05 producer of
                              its own site only, the fused o-projection's producer stays unfenced,
                              ``tail_runtime.k19_tail`` only for published / two-shot 9-128 rows,
                              ``compile_tail`` refuses the tail fence elsewhere.
  compile FORM [FORM ...]     sm_107a CPU compile of each form (CUTE_DSL_ARCH=sm_107a CUTE_DSL_KEEP=ptx
                              CUTE_DSL_DUMP_DIR=OUT), each form's PTX moved to OUT/<form>/; the env decides
                              which sites are fenced. Forms: see FORMS.
  check OFF_DIR ON_DIR FORM.. per form: ON fences EVERY griddepcontrol.launch_dependents (a fence.acq_rel.gpu
                              before it in its basic block with no memory access in between), OFF leaves at
                              least one trigger unfenced, both have the same trigger count. Exit 1 on a miss.
Fences change no value, so these PTX checks plus the default-off PTX identity against the base tree (the
caller compares normalized md5s) are the correctness evidence; the timing / bitwise read is a TP16 arm."""

import contextlib
import importlib.util
import io
import os
import re
import sys
import time

SITES_OF = {"prod": "oproj_producer", "umma": "oproj_umma", "front": "front_gemm", "r64": "rubin64",
            "rubin8": "rubin_experts", "tail": "tail9"}
FORMS = ("prod_b1", "prod_b16", "umma_b16", "umma_b64t", "front_b8", "front_b16", "front_b64",
         "r64_b16p", "r64_b64t", "rubin8_b8", "tail_b16", "tail_b64t")


def tree_root():
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))


def load_shim():
    """The tree's CPU compile shim (cuInit + CPU-torch accelerator patch); returns the module."""
    spec = importlib.util.spec_from_file_location("tp16c", os.path.join(tree_root(), "tools/rubin_k31/tp16_compile.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- unit

def _fresh(value):
    """Set K31MK_K19_FENCE (None: unset) and forget earlier engagement lines."""
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common import k19

    if value is None:
        os.environ.pop(k19.ENV, None)
    else:
        os.environ[k19.ENV] = value
    k19._engaged.clear()
    return k19


def _lines(fn):
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        result = fn()
    return result, [x for x in err.getvalue().splitlines() if x.startswith("K31MK_K19_FENCE engaged")]


def test_unset_is_off():
    k19 = _fresh(None)
    result, lines = _lines(lambda: [k19.fence(s) for s in k19.SITES])
    assert result == [False] * len(k19.SITES) and lines == [], (result, lines)


def test_named_sites_only_and_one_line_each():
    k19 = _fresh("rubin64+tail9")
    assert k19.sites() == frozenset({"rubin64", "tail9"})
    (a, b, c), lines = _lines(lambda: (k19.fence("rubin64"), k19.fence("rubin64"), k19.fence("front_gemm")))
    assert (a, b, c) == (True, True, False), (a, b, c)
    assert len(lines) == 1 and "site=rubin64 " in lines[0], lines


def test_unknown_names_raise():
    k19 = _fresh("rubin64+bogus")
    for call in (k19.sites, lambda: k19.fence("rubin64")):
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("an unknown K31MK_K19_FENCE site was accepted")
    _fresh(None)
    try:
        k19.fence("bogus")
    except ValueError:
        return
    raise AssertionError("k19.fence accepted an unknown site name")


def test_producer_wiring():
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.oproj import prepared as P
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.oproj.consumer import OprojFused
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.oproj.producer import OprojProducer, OprojUmma

    def build():
        return (P.producer(768, 1, 16, True), P.producer(768, 16, 16, True, kind="mma"),
                P.producer(768, 64, 16, True, kind="umma"), OprojFused(1).producer)

    for env, want in ((None, (False, False, False)), ("oproj_umma", (False, False, True)),
                      ("oproj_producer", (True, True, False)), ("oproj_producer+oproj_umma", (True, True, True))):
        _fresh(env)
        (split, grouped, umma, fused), lines = _lines(build)
        assert type(split) is OprojProducer and type(grouped) is OprojProducer and type(umma) is OprojUmma
        got = (split.k19_fence, grouped.k19_fence, umma.k19_fence)
        assert got == want, (env, got, want)
        assert fused.k19_fence is False, env  # the fused o-projection never takes the split producer's fence
        assert len(lines) == int(want[0]) + int(want[2]), (env, lines)  # one line per engaged site
        if env is None:  # default objects carry no instance attribute (kernel-cache keys unchanged)
            assert "k19_fence" not in vars(split) and "k19_fence" not in vars(umma)


def test_tail_rows():
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import tail_runtime as T

    cases = (("publish", 16, True), ("twoshot", 64, True), ("publish", 9, True), ("twoshot", 128, True),
             ("publish", 8, False), ("front", 16, False), ("front_poll", 16, False), ("publish", 129, False))
    _fresh("tail9")
    for mode, batch, want in cases:
        assert T.k19_tail(mode, batch) is want, (mode, batch)
    _fresh(None)
    assert not any(T.k19_tail(m, b) for m, b, _ in cases)
    _fresh("rubin64")
    assert not any(T.k19_tail(m, b) for m, b, _ in cases)
    for mode, batch in (("publish", 8), ("front", 16)):
        try:
            T.compile_tail(16, batch, 0, mode, k19_tfence=True)
        except ValueError:
            continue
        raise AssertionError(f"compile_tail accepted the tail fence at {mode} B{batch}")


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
    _fresh(None)
    print(f"UNIT {len(tests) - failed}/{len(tests)} passed", flush=True)
    return 1 if failed else 0


# ---------------------------------------------------------------- compile

def compile_form(shim, form):
    batch = int(re.search(r"_b(\d+)", form).group(1))
    deadline = 10**9
    if form.startswith(("prod", "umma")):
        from sgl_nvidia_mega.kimi_k31.k3.cutedsl.oproj.prepared import compile_oproj

        kind = "producer" if batch <= 8 else ("producer_mma" if form.startswith("prod") else "producer_umma")
        return compile_oproj(kind, 768, batch, 16, 1, True, True, False, 64, form.endswith("t"), deadline)
    if form.startswith("front"):
        return shim.compile_generic("publish", batch)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.rubin.moe import experts_runtime as RE

    if form.startswith("r64"):
        sms = int(os.environ["K31_TARGET_SMS"])  # the target's SM count (CPU compile: no GPU to ask)
        dies = tuple(0 if i < sms // 2 else 1 for i in range(sms))
        return RE.compile_rubin64(896, batch, sms - 6, 0, "twoshot" if form.endswith("t") else "push", sms, dies, False, 12)
    if form.startswith("rubin8"):
        sms = int(os.environ["K31_TARGET_SMS"])
        return RE.compile_rubin(896, 8, sms, 0, True, max(160, -(-sms // 32) * 32), polled=True)  # workspace_slots' rule
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import tail_runtime as T

    T.prepare_shared_carveout = lambda code, *_: code  # a driver attribute call (needs a GPU)
    mode = "twoshot" if form.endswith("t") else "publish"
    extra = dict(k19_tfence=True) if getattr(T, "k19_tail", None) and T.k19_tail(mode, batch) else {}
    early = batch in T.early_batches(16)
    umma = batch in T.umma_batches(16)
    return T.compile_tail(16, batch, 0, mode, early=early, umma=umma, deadline_ns=deadline, **extra)


def run_compile(forms):
    shim = load_shim()
    import sgl_nvidia_mega

    print(f"TREE {os.path.dirname(os.path.dirname(os.path.abspath(sgl_nvidia_mega.__file__)))}", flush=True)
    dump = os.environ["CUTE_DSL_DUMP_DIR"]
    failed = 0
    for form in forms:
        t0 = time.time()
        try:
            compile_form(shim, form)
            moved = 0
            for f in os.listdir(dump):
                if f.endswith(".ptx"):
                    os.makedirs(os.path.join(dump, form), exist_ok=True)
                    os.replace(os.path.join(dump, f), os.path.join(dump, form, f))
                    moved += 1
            assert moved >= 1, (form, moved)
            print(f"COMPILE OK {form} files={moved} {time.time() - t0:.1f}s", flush=True)
        except Exception as error:  # noqa: BLE001
            failed += 1
            print(f"COMPILE FAIL {form}: {type(error).__name__}: {str(error)[:400]}", flush=True)
    return 1 if failed else 0


# ---------------------------------------------------------------- check

MEMORY = re.compile(r"^\s*(@!?%\w+\s+)?(ld|st|atom|red|cp|multimem|tcgen05\.(ld|st|cp)|mbarrier)\b")
BLOCK_END = re.compile(r"^\s*(\$?[\w.$]+:|(@!?%\w+\s+)?bra\b|ret\b|\})")


def trigger_census(text):
    """(triggers, fenced triggers) in one PTX text."""
    lines = text.splitlines()
    triggers = fenced = 0
    for i, line in enumerate(lines):
        if "griddepcontrol.launch_dependents" not in line:
            continue
        triggers += 1
        for prev in reversed(lines[:i]):
            stripped = prev.strip()
            if not stripped or stripped.startswith("//"):
                continue
            if "fence.acq_rel.gpu" in stripped:
                fenced += 1
                break
            if BLOCK_END.match(prev) or MEMORY.match(prev) or "griddepcontrol" in stripped:
                break
    return triggers, fenced


def form_census(root, form):
    triggers = fenced = 0
    d = os.path.join(root, form)
    for f in sorted(os.listdir(d)):
        if f.endswith(".ptx"):
            with open(os.path.join(d, f)) as fh:
                t, k = trigger_census(fh.read())
            triggers, fenced = triggers + t, fenced + k
    return triggers, fenced


def run_check(off, on, forms):
    bad = 0
    for form in forms:
        try:
            (t_off, f_off), (t_on, f_on) = form_census(off, form), form_census(on, form)
        except FileNotFoundError as error:
            print(f"CHECK FAIL {form}: missing {error.filename}", flush=True)
            bad += 1
            continue
        ok = t_on == t_off >= 1 and f_on == t_on and f_off < t_off
        bad += not ok
        print(f"CHECK {'PASS' if ok else 'FAIL'} {form}: triggers off/on {t_off}/{t_on}, fenced off/on {f_off}/{f_on}", flush=True)
    print(f"CHECK {len(forms) - bad}/{len(forms)} forms pass", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "unit"
    if mode == "unit":
        sys.exit(run_unit())
    if mode == "compile":
        sys.exit(run_compile(sys.argv[2:] or list(FORMS)))
    if mode == "check":
        sys.exit(run_check(sys.argv[2], sys.argv[3], sys.argv[4:] or list(FORMS)))
    raise SystemExit(f"unknown mode {mode}")

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU contracts for the exact SM-stable roles of verify_mk's megakernel builds (``MSO_MK_I2B_SMID=2``).

* the builds in the vendored library (its Python module, loaded from its file: no build, no GPU) and verify_mk's served
  build and check twin;
* the claim words' readout (``library.i2b_claim``);
* the claim protocol of ``msoMegakernel.cu`` (``i2bClaimStart`` / ``i2bSettle`` / ``i2bSettleSlow``) as a Python
  model, one atomic step per access to the claim words: every interleaving of one launch from every valid state for
  2 and 3 CTAs (by induction, any sequence of launches), random interleavings up to 212 CTAs, and negative controls;
* with clang++: the preprocessed device code routes every ``blockIdx`` of the megakernel to the claimed index.
"""

from __future__ import annotations

import ctypes
import importlib
import itertools
import random
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

import sgl_nvidia_mega

MK = (
    Path(sgl_nvidia_mega.__file__).resolve().parent
    / "muse_spark/sm107/trtllm_ab93b78/megakernel"
)
SOURCE = MK / "msoMegakernel.cu"
SERVED = (
    "lean_icfpoxaft_rt16_ld479_ue10_et5_nb3_bh11_pb15_rp1_rl1_ah903_xp40_km8_lm2_s4"
)
CHECK = SERVED.replace("lean_", "check_", 1).replace("_rl1_", "_rl3_", 1)
SG = SERVED[: -len("_s4")] + "_sg_s4"
SMX = SERVED[: -len("_s4")] + "_smx_sg_s4"
SMXT = SERVED[: -len("_s4")] + "_smxt_sg_s4"
CHECK_SG = CHECK[: -len("_s4")] + "_sg_s4"
CHECK_SMX = CHECK[: -len("_s4")] + "_smx_sg_s4"
SG_KV = ("-DMSO_MK_SG_KV=1", "-DMSO_MK_SG_PAGE=128", "-DMSO_MK_FA_VBOX=0")
I2B = ("-DMSO_MK_I2B_SMID=2",)
NONE = 0xFFFFFFFF


@pytest.fixture(scope="module")
def real():
    """The vendored ``library`` (and its ``schedule``) under a private package name."""
    name = "_verify_mk_i2b_vendor"
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(MK)]
        sys.modules[name] = pkg
    dont = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        return importlib.import_module(f"{name}.library")
    finally:
        sys.dont_write_bytecode = dont


# ---------------------------------------------------------------------------------------------------- builds
def test_i2b_modes(real):
    assert real.I2B_MODES == [SMX, CHECK_SMX, SMXT]
    for sg, i2b, extra in (
        (SG, SMX, I2B),
        (CHECK_SG, CHECK_SMX, I2B),
        (SG, SMXT, (*I2B, "-DMSO_MK_I2B_TEST=1")),
    ):
        src = tuple(real.MODES[sg])[: -len(SG_KV)]
        assert tuple(real.MODES[sg]) == (*src, *SG_KV)
        assert tuple(real.MODES[i2b]) == (*src, *extra, *SG_KV)
        for (
            lever
        ) in real.LEVERS:  # nb / bh / rl and every other lever at the _sg build's level
            assert real.lever_level(i2b, lever) == real.lever_level(sg, lever), (
                i2b,
                lever,
            )
        assert real.compose_mode(i2b, {}) == i2b
    assert (
        sum(1 for flags in real.MODES.values() if any("I2B_SMID=2" in d for d in flags))
        == 3
    )
    # TensorRT-LLM's %smid builds keep value 1
    for name in ("lean_sm_s4", "stlean_sm_s4", "check_sm_s4"):
        assert "-DMSO_MK_I2B_SMID=1" in real.MODES[name], name


def test_served_builds(real):
    """verify_mk's MODE and CHECK_MODE (runtime.py: serving.DEFAULT_MODE with ``_smx_sg_s4``, and its rl 3 check twin)
    are the exact builds of TensorRT-LLM's served build."""
    serving = (MK / "serving.py").read_text()
    assert re.search(r'^DEFAULT_MODE = "([^"]+)"$', serving, re.M).group(1) == SERVED
    mode = SERVED[: -len("_s4")] + "_smx_sg_s4"
    check = mode.replace("lean_", "check_", 1).replace("_rl1_", "_rl3_", 1)
    assert (mode, check) == (SMX, CHECK_SMX)
    assert {mode, check} <= set(real.MODES)


# ---------------------------------------------------------------------------------------------------- readout
class FakeCdll:
    """A loaded MSO_MK_I2B_SMID=2 library: ``msoMkI2bClaim`` fills the 7 words."""

    def __init__(self, words, rc=0):
        self.words, self.rc, self.calls = words, rc, 0

        def fn(out):
            self.calls += 1
            for i, v in enumerate(self.words):
                out[i] = v
            return self.rc

        self.msoMkI2bClaim = fn


def test_i2b_claim_readout(real, monkeypatch):
    synced = []
    monkeypatch.setattr(real.torch.cuda, "synchronize", lambda: synced.append(1))
    cdll = FakeCdll([212 * 240, 0, 240, 0, 0, 212, 4096])
    claim = real.i2b_claim(types.SimpleNamespace(lib=cdll))
    assert claim == {
        "tickets": 212 * 240,
        "misses": 0,
        "launches": 240,
        "partial": 0,
        "off": 0,
        "ctas": 212,
        "cycle": 4096,
    }
    assert synced == [1] and cdll.calls == 1
    assert cdll.msoMkI2bClaim.argtypes == [ctypes.c_void_p]
    assert cdll.msoMkI2bClaim.restype is ctypes.c_int
    # another build: no claim words, no synchronize
    assert real.i2b_claim(types.SimpleNamespace(lib=types.SimpleNamespace())) is None
    assert synced == [1]
    with pytest.raises(AssertionError, match="CUDA error 700"):
        real.i2b_claim(types.SimpleNamespace(lib=FakeCdll([0] * 7, rc=700)))


# ---------------------------------------------------------------------------------------------------- the protocol
class Trap(Exception):
    pass


def claim(hint, g, cycle, undo=True):
    """Thread 0's claim, statement for statement as ``msoMegakernel.cu`` (``i2bClaimStart``, ``i2bSettle``,
    ``i2bSettleSlow``): yields each access to the claim words as one atomic operation, receives the value it read,
    returns the claimed index or raises ``Trap`` (``__trap()``). ``undo=False`` drops the undo (a negative control)."""
    ticket = yield ("inc", "ticket", g * cycle - 1)
    old = (yield ("inc", hint, cycle - 1)) if hint < g else NONE
    k = ticket // g
    if old == k:
        return hint
    j = g - 1
    if old != NONE:
        d = (old + cycle - k) % cycle
        back = (yield ("dec", hint, cycle - 1)) if undo else old + 1
        e = (back + cycle - k) % cycle
        if d < 1 or d > g - 1 or e < 2 or e > g:
            raise Trap(f"d {d}, e {e}")
        j = hint
    yield ("add", "miss", 1)
    k1 = 0 if k + 1 == cycle else k + 1
    for _ in range(g):
        j = 0 if j + 1 == g else j + 1
        if (yield ("cas", j, k, k1)) == k:
            return j
    raise Trap("no counter left at k")


def test_model_is_the_source():
    """``claim`` above transcribes these statements of ``msoMegakernel.cu`` (a changed statement there must change the
    model too)."""
    src = " ".join(SOURCE.read_text().split())
    for statement in (
        'asm volatile("atom.relaxed.gpu.global.inc.u32 %0, [%1], %2;"',
        'asm volatile("atom.relaxed.gpu.global.dec.u32 %0, [%1], %2;"',
        'asm volatile("atom.relaxed.gpu.global.cas.b32 %0, [%1], %2, %3;"',
        "c.ticket = i2bAtomInc(&gI2bWords[kI2bTicketWord], static_cast<uint32_t>(kCtas) * kI2bCycle - 1u);",
        "c.old = hint < static_cast<uint32_t>(kCtas) ? i2bAtomInc(&gI2bIndex[hint * kI2bStride], kI2bCycle - 1u) "
        ": kI2bNone;",
        "uint32_t const k = c.ticket / static_cast<uint32_t>(kCtas);",
        "return c.old == k ? c.hint : i2bSettleSlow(c.hint, c.old, k);",
        "uint32_t j = g - 1u;",
        "if (old != kI2bNone) { uint32_t const d = (old + kI2bCycle - k) % kI2bCycle;",
        "uint32_t const back = i2bAtomDec(&gI2bIndex[hint * kI2bStride], kI2bCycle - 1u);",
        "uint32_t const e = (back + kI2bCycle - k) % kI2bCycle;",
        "if (d < 1u || d > g - 1u || e < 2u || e > g)",
        "{ __trap(); } j = hint; } redRelaxedAdd(&gI2bWords[kI2bMissWord], 1u);",
        "uint32_t const k1 = k + 1u == kI2bCycle ? 0u : k + 1u;",
        "for (uint32_t n = 0; n < g; ++n) { j = j + 1u == g ? 0u : j + 1u;",
        "if (i2bAtomCas(&gI2bIndex[j * kI2bStride], k, k1) == k) { return j; } } __trap(); // no counter left at k",
        "#if MSO_MK_I2B_TEST == 1 return s & ~1u;",
        "I2bClaim const claim = i2bClaimStart(i2bHint);",
        "int const own = static_cast<int>(i2bSettle(claim));",
    ):
        assert statement in src, statement


def rmw(words, op):
    """PTX ``atom.{inc,dec,cas}.u32`` / ``red.add.u32`` on ``words``: the value read."""
    kind, loc, *args = op
    old = words[loc]
    if kind == "inc":
        words[loc] = 0 if old >= args[0] else old + 1
    elif kind == "dec":
        words[loc] = args[0] if old == 0 or old > args[0] else old - 1
    elif kind == "cas":
        words[loc] = args[1] if old == args[0] else old
    else:
        words[loc] = (old + args[0]) & 0xFFFFFFFF
    return old


def valid_words(g, k):
    """The words before launch k (mod cycle) of a build: k everywhere, k * g start tickets taken."""
    return {"ticket": k * g, "miss": 0, **dict.fromkeys(range(g), k)}


def check_launch(words, results, g, cycle, k):
    assert sorted(results) == list(range(g)), results  # every index exactly once
    k1 = (k + 1) % cycle
    # the next launch's state
    assert all(words[i] == k1 for i in range(g)) and words["ticket"] == k1 * g, words


def next_op(hint, g, cycle, history, undo=True):
    """Replay a CTA's claim on the values it read so far: ("op", op) | ("done", index) | ("trap", why)."""
    gen = claim(hint, g, cycle, undo)
    try:
        op = gen.send(None)
        for value in history:
            op = gen.send(value)
        return "op", op
    except StopIteration as stop:
        return "done", stop.value
    except Trap as trap:
        return "trap", str(trap)


def explore(hints, g, cycle, k, undo=True):
    """Every interleaving of one launch's claims (one CTA per hint) from the valid state at launch k; each final state
    is checked. Returns the number of states."""
    start = (
        tuple(sorted(valid_words(g, k).items(), key=str)),
        tuple(() for _ in hints),
    )
    seen, frontier = {start}, [start]
    while frontier:
        nxt = []
        for items, hist in frontier:
            words, results, moved = dict(items), [], False
            for c, hint in enumerate(hints):
                kind, value = next_op(hint, g, cycle, hist[c], undo)
                if kind == "trap":
                    raise Trap(f"{value} (hints {hints}, k {k}, values read {hist})")
                if kind == "done":
                    results.append(value)
                    continue
                moved = True
                after = dict(words)
                old = rmw(after, value)
                state = (
                    tuple(sorted(after.items(), key=str)),
                    (*hist[:c], (*hist[c], old), *hist[c + 1 :]),
                )
                if state not in seen:
                    seen.add(state)
                    nxt.append(state)
            if not moved:
                check_launch(words, results, g, cycle, k)
        frontier = nxt
    return len(seen)


@pytest.mark.parametrize("g", [2, 3])
def test_claim_every_interleaving(g):
    """All hints in 0 .. g (g: out of range) for every CTA, every launch index k of a cycle of g + 1 (the smallest
    allowed: kI2bCycle > kCtas), every interleaving: exact coverage and the next launch's state. By induction over
    launches (the end state is the next launch's valid start state) this holds for any sequence of launches."""
    cycle = g + 1
    states = sum(
        explore(h, g, cycle, k)
        for h in itertools.product(range(g + 1), repeat=g)
        for k in range(cycle)
    )
    assert states > (600 if g == 2 else 100_000)


@pytest.mark.parametrize(
    "hints", [(0, 0, 2, 2), (1, 0, 3, 2), (3, 3, 3, 0), (4, 0, 4, 2)]
)
def test_claim_every_interleaving_four(hints):
    for k in (0, 4):  # a cycle of 5: k = 4 wraps to 0
        explore(hints, 4, 5, k)


def run_launches(g, cycle, hints_of, launches, seed, undo=True, overlap=False):
    """Random interleavings of ``launches`` launches. Launch n + 1's CTAs start after every claim of launch n (the
    PDL order: each CTA consumes its claim before griddepcontrol.launch_dependents), or, with ``overlap`` (a negative
    control), alongside it. Returns each launch's claimed indices by CTA and the words."""
    rng = random.Random(seed)
    words = valid_words(g, 0)
    claimed = []
    pending = []  # [launch, cta, generator, op]
    for n in range(launches):
        gens = [claim(h, g, cycle, undo) for h in hints_of(n, rng)]
        claimed.append([None] * len(gens))
        pending += [[n, c, gen, gen.send(None)] for c, gen in enumerate(gens)]
        if overlap and n + 1 < launches:
            continue  # launch n + 1 joins before launch n's claims end
        while pending:
            item = pending[rng.randrange(len(pending))]
            try:
                item[3] = item[2].send(rmw(words, item[3]))
            except StopIteration as stop:
                claimed[item[0]][item[1]] = stop.value
                pending.remove(item)
        if not overlap:
            check_launch(words, claimed[n], g, cycle, n % cycle)
    return claimed, words


def test_claim_random_schedules():
    g, cycle = 16, 17  # k wraps every 17 launches
    patterns = [
        lambda n, r: list(range(g)),  # %smid = the block index permutation
        lambda n, r: r.sample(range(g), g),  # another permutation every launch
        # collisions and out-of-range hints, moving every launch
        lambda n, r: [r.randrange(g + 4) for _ in range(g)],
        lambda n, r: [0] * g,  # every CTA hints index 0
        lambda n, r: [s & ~1 for s in range(g)],  # MSO_MK_I2B_TEST = 1
    ]
    for p, hints_of in enumerate(patterns):
        run_launches(g, cycle, hints_of, launches=120, seed=p)


def test_claim_served_shape():
    """212 CTAs, kI2bCycle 4096: with %smid a dense permutation every CTA keeps its SM's index (no miss); with the test
    build's hints every CTA ends on its SM's index or its pair's (106 misses a launch)."""
    src = SOURCE.read_text()
    assert re.search(r"constexpr uint32_t kI2bCycle = 4096;", src)
    assert "#define MSO_MK_CTAS 212" in src
    g, cycle = 212, 4096
    claimed, words = run_launches(
        g, cycle, lambda n, r: list(range(g)), launches=6, seed=1
    )
    assert all(c == list(range(g)) for c in claimed) and words["miss"] == 0
    claimed, words = run_launches(
        g, cycle, lambda n, r: [s & ~1 for s in range(g)], launches=6, seed=2
    )
    assert all(c[s] in (s & ~1, s | 1) for c in claimed for s in range(g))
    assert words["miss"] == 6 * g // 2


def test_claim_negative_controls():
    g = 4
    # 1. no undo: the hinted counter ends one ahead after a collision
    with pytest.raises(AssertionError):
        explore((0, 0, 2, 3), g, g + 1, 0, undo=False)
    # 2. a cycle of only kCtas: a counter's steps past k wrap (the undo reads e = 0 when every CTA hints one index)
    with pytest.raises(Trap):
        explore((1, 1, 1), 3, 3, 0)
    # 3. launches overlapping in their claims (the order PDL gives the pre-wait claim, removed): some schedule breaks
    # coverage, the end state or traps
    broken = 0
    for seed in range(200):
        try:
            claimed, words = run_launches(
                g,
                g + 1,
                lambda n, r: [r.randrange(g) for _ in range(g)],
                2,
                seed,
                overlap=True,
            )
            assert all(sorted(c) == list(range(g)) for c in claimed)
            assert all(words[i] == 2 for i in range(g))
        except (Trap, AssertionError):
            broken += 1
    assert broken > 0


# ---------------------------------------------------------------------------------------------------- the source
def _preprocess(real, tmp_path, name, device=True, extra=()):
    """``clang++ -E -P`` of msoMegakernel.cu with build ``name``'s defines, ``extra`` replacing those of the same name
    (empty stub CUDA headers; the device pass as __CUDA_ARCH__ 1070): (return code, stderr, text)."""
    stub = tmp_path / "stubinc"
    stub.mkdir(exist_ok=True)
    for header in ("cuda.h", "cuda_bf16.h", "cuda_fp16.h", "cuda_runtime.h"):
        (stub / header).write_text("")
    names = {e.split("=")[0] for e in extra}
    defs = [d for d in real.MODES[name] if d.split("=")[0] not in names] + list(extra)
    cmd = [
        "clang++",
        "-E",
        "-P",
        "-x",
        "c++",
        "-std=c++20",
        "-I",
        str(stub),
        "-DMSO_MK_HEADERS=0",
        *defs,
    ]
    if device:
        cmd += ["-D__CUDA_ARCH__=1070", "-D__CUDACC__"]
    p = subprocess.run(
        [*cmd, str(SOURCE)], capture_output=True, text=True, timeout=120, check=False
    )
    return p.returncode, p.stderr, p.stdout


def _bodies(text, names):
    """The [start, end) text ranges of the bodies of the functions ``names`` (by brace matching)."""
    spans = []
    for name in names:
        m = re.search(r"\b" + name + r"\([^;{]*\)\s*\{", text)
        assert m, name
        depth, i = 0, m.end() - 1
        while True:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
            if depth == 0:
                break
        spans.append((m.start(), i))
    return spans


def _uses(text):
    return [m.start() for m in re.finditer(r"\bblockIdx\b", text)]


@pytest.mark.skipif(shutil.which("clang++") is None, reason="clang++ not found")
def test_blockidx_routing(real, tmp_path):
    """Value 2: every blockIdx of the megakernel's code (the stage files, the watchdog's records) reads the claimed
    index; only the weight-table and spin kernels keep their own grid's. Value 0 routes nothing."""
    out = {}
    for key, name, extra in (
        ("smx", SMX, ()),
        ("smxt", SMXT, ()),
        ("sm1", SMX, ("-DMSO_MK_I2B_SMID=1",)),  # TensorRT-LLM's value 1
        ("sg", SG, ()),
    ):
        rc, err, text = _preprocess(real, tmp_path, name, extra=extra)
        assert rc == 0 and "error" not in err, err[:400]
        out[key] = text
    raw = len(_uses(out["sg"]))  # value 0: every use reads the hardware index
    assert raw > 20
    assert "i2bLogicalBlockIdx" not in out["sg"] and "gI2bIndex" not in out["sg"]
    for key in ("smx", "smxt"):
        text = out[key]
        own = _bodies(text, ["msoWeightTableKernel", "msoSpinKernel"])
        uses = _uses(text)
        assert len(uses) == 2 and all(any(a <= u < b for a, b in own) for u in uses), (
            uses
        )
        # every other use reads the claimed index (the kernel's own `cta = blockIdx.x` became the hint, and its one read
        # of the claimed index after the first barrier is new)
        assert text.count("::mso_mk::i2bLogicalBlockIdx()") == raw - 2
        # only smid(): the hint and the stamps' diagnostic word
        assert len(re.findall(r"%%smid", text)) == 1
        assert "i2bClaimStart(i2bHint)" in text
        assert "kSmallI2b) = static_cast<uint32_t>(own)" in text
    assert ("s & ~1u" in out["smxt"]) and ("s & ~1u" not in out["smx"])
    # Value 1 defines blockIdx after moe_hops.cuh: the watchdog's two records and the rl logit unit's arena replica
    # (identical replicas, so its values do not change) read the hardware index
    sm1 = out["sm1"]
    early = _bodies(
        sm1,
        ["fail", "logWaiter", "stageRlLogit", "msoWeightTableKernel", "msoSpinKernel"],
    )
    assert len(_uses(sm1)) == 5
    assert all(any(a <= u < b for a, b in early) for u in _uses(sm1))
    assert sm1.count("::mso_mk::i2bLogicalBlockIdx()") == raw - 5
    assert "i2bClaimStart" not in sm1 and "gI2bIndex" not in sm1
    assert len(re.findall(r"%%smid", sm1)) == 2


@pytest.mark.skipif(shutil.which("clang++") is None, reason="clang++ not found")
def test_source_rules(real, tmp_path):
    """The host pass of value 2 carries the claim words and their readout; the #error rules reject other values."""
    rc, err, host = _preprocess(real, tmp_path, SMX, device=False)
    assert rc == 0 and "gI2bIndex[kCtas * kI2bStride]" in host
    assert "int msoMkI2bClaim(unsigned* out)" in host
    rc, err, host = _preprocess(real, tmp_path, SG, device=False)
    assert rc == 0 and "msoMkI2bClaim" not in host and "gI2bWords" not in host
    for name, extra in (
        (SG, ("-DMSO_MK_I2B_SMID=3",)),
        (SG, ("-DMSO_MK_I2B_SMID=1", "-DMSO_MK_I2B_TEST=1")),
        (SMX, ("-DMSO_MK_I2B_TEST=2",)),
    ):
        rc, err, _ = _preprocess(real, tmp_path, name, extra=extra)
        assert rc != 0 and "MSO_MK_I2B_" in err, (extra, err[-300:])
    assert all(b < 128 for b in SOURCE.read_bytes())  # ASCII

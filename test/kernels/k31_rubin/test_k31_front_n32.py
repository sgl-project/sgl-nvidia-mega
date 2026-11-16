# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""N32 (``K31MK_MOE_FRONT_N32``, ``moe/front_tmem.py`` front_n32_enabled): CPU checks, no GPU.

Modes (tree = the plugin checkout on PYTHONPATH; tools/rubin_k31/tp16_compile.py is its CPU shim):
  unit                 knob parsing and the engagement lines: scalar form unset / empty / 0 = off, 1 (N32 tile) / 2 (N32 tile, run-time
                       MMA N) only with K31MK_MOE_FRONT_TMEM batches that are all <= 64 rows (two N32 tiles at 33-64), any other value
                       raises, one stderr line per process; batch-table form (front_n32_for) per batch, rows 9-128 (65-128:
                       value 0 / 1, 1 = two N-exact tiles N = ceil(rows / 2) rounded up to 8, N64 = 0), one line per engaged batch,
                       and the scalar reader refuses it.
  compile N32 OUT      sm_107a CPU compile (--keep-ptx) of the served c=8 FTM front through the unchanged served call
                       (``front.compile_front_tmem``: split 64, valid 56, shard 224, hidden 384, promote 2, mbox 1 = the served form) with
                       K31MK_MOE_FRONT_TMEM=16-32:1 and K31MK_MOE_FRONT_N32 unset ('base' / 0), 1 or 2; the PTX lands in OUT.
  compile nxN OUT      (N = 16 / 24 / 40 / 48 / 56) the N-exact tile (65-128 rows; value 3): the same served call with MoeFrontTmem64 built at n32 = N, mbox 0
                       (boots with 65-128-row FTM batches have no MBOX there); 'nx0' = the same call at N64, mbox 0 (its baseline).
  checknx OFF ON N [B] ON must carry (OFF compiled at N B, default 64; N16 / N24 compare against the N32 mode-1 compile, B 32);
                       owner sums by (row, quad) items: no narrow shared loads beyond OFF's, the same 16-byte load count the M128 N descriptor and no N64 one (OFF the reverse), the same tcgen05.mma / tcgen05.cp /
                       griddepcontrol counts (12 K tiles in tensor memory either way under PROMOTE 2), st.async pushes scaled N / 64, and
                       the pair (N48) or word (N40 / N56) owner-sum loads OFF does not have.
  compile pf OUT / pf1 OUT  K31MK_MOE_FRONT_L2PF=1 on the served call at N64 ('pf') or N32 mode 1 ('pf1').
  compile lean64 / lx64 / lx1 / lx48 OUT  K31MK_MOE_FRONT_LEAN=1 at N64 mbox 0 (lean64), + K31MK_MOE_FRONT_TX576=1 (lx64; lx1 at N32
                       mode 1 mbox 1; lx48 at the N48 N-exact tile, mbox 0); every compile prints 'BUILT smem=<bytes> tmem_tiles=<n> smem_tiles=<n> tmem_cols=<n>'.
  checklx OFF ON TILES ON must use one exclusive 576-column allocation (no non-exclusive alloc / dealloc / relinquish), stage TILES K
                       tiles in tensor memory (tcgen05.cp = 4 x TILES), and keep the MMA / descriptor / push / PDL counts of OFF.
  compile xs1 / xs2 / xs1n48 / xs1mb OUT  K31MK_MOE_FRONT_XSEP=1 / 2 at N64 mbox 0 (the nx0 call), 1 at the N48 N-exact tile (nx48),
                       1 at N64 with mbox 1 (the MID form); BUILT adds 'xsep=<v> x_slot=<list> x_wait=<list>'.
  checkxs OFF ON EXTRA [CL] ON must keep OFF's MMA / copy / descriptor / push / PDL / alloc counts, add EXTRA tcgen05.commit (a_ready per
                       TMEM tile + stage_free per weight stage), 13 tensor loads (x unrolled over 14 K tiles), one ld.shared.b32, more mbarrier.init
                       and CL barrier.cluster.arrive (1 at mbox 0: the MMA warp's cluster phase; 0 at mbox 1).
  compile rf / rfx1 / rfmb OUT  K31MK_MOE_FRONT_RFIRST=1 at nx0, + XSEP 1 (rfx1), at nx0mb (rfmb).
  checkrf OFF ON       ON must carry the same count of every census token as OFF (RFIRST only changes the cluster -> tile index math;
                       the caller checks that the PTX differs).
  checkpf OFF ON COUNT ON must carry COUNT cp.async.bulk.prefetch.tensor instructions (OFF none) and the same counts of every other
                       census token as OFF (loads, MMAs, copies, pushes, PDL): the knob adds the hint and nothing else.
  check OFF ON [MODE]  ON must carry the M128 N32 instruction descriptor (MODE 1, default) or no immediate front descriptor at all
                       (MODE 2: register descriptors), never the N64 one (OFF the reverse), the same number of tcgen05.mma (56 = 14 K
                       tiles x 4), half the st.async partial pushes, more tcgen05.cp, and the same griddepcontrol count; exit 1 on a miss.
The bitwise / timing evidence is the GPU rig (not included); this file proves the knob reaches the kernel and that the
default-off PTX equals the base tree's (the caller compares normalized md5s)."""

import contextlib
import importlib.util
import io
import os
import re
import sys
import types

# kind::f16 instruction descriptors of the front's MMAs (splitk_tmem64: 0x08000490 | (N >> 3) << 17), as the decimal the
# inline asm carries
IDESC = {n: 0x08000490 | (n >> 3) << 17 for n in (64, 32, 16, 24, 40, 48, 56)}


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


def _ft():
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import front_tmem

    return front_tmem


def _front():
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import front

    return front


def _call(value, ftm):
    ft = _ft()
    for key, v in (("K31MK_MOE_FRONT_N32", value), ("K31MK_MOE_FRONT_TMEM", ftm)):
        if v is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = v
    ft._n32_engaged.clear()
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        out = ft.front_n32_enabled()
    return out, [x for x in err.getvalue().splitlines() if x.startswith("K31MK_MOE_FRONT_N32 engaged")]


def test_unset_empty_zero_are_off():
    for value in (None, "", "0"):
        on, lines = _call(value, "16-32:1")
        assert on == 0 and lines == [], (value, on, lines)


def test_on_with_served_table_and_one_line():
    for mode in ("1", "2"):
        on, lines = _call(mode, "16-32:1")  # the served c=8 table
        assert on == int(mode) and len(lines) == 1, (mode, on, lines)
        assert lines[0].split()[2:4] == [f"mode={mode}", "ftm=16-32"], lines
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            again = _ft().front_n32_enabled()
        assert again == int(mode) and "engaged" not in err.getvalue(), err.getvalue()
    on, _ = _call("1", "16-32:1,64:0")  # a value-0 entry is not an FTM batch
    assert on == 1


def test_mid_rows_two_tiles():
    # FTM batches up to 64 rows run two N32 tiles; the engagement line names the whole FTM range
    on, lines = _call("1", "16-64:1")
    assert on == 1 and len(lines) == 1 and lines[0].split()[2:4] == ["mode=1", "ftm=16-64"], (on, lines)


def _for(value, batch):
    ft = _ft()
    if value is None:
        os.environ.pop("K31MK_MOE_FRONT_N32", None)
    else:
        os.environ["K31MK_MOE_FRONT_N32"] = value
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        out = ft.front_n32_for(batch)
    return out, [x for x in err.getvalue().splitlines() if x.startswith("K31MK_MOE_FRONT_N32 engaged")]


def test_table_form_per_batch():
    ft = _ft()
    ft._n32_batches.clear()
    for value in (None, "", "0", "1", "2"):  # scalar form / unset: the class decides (None), no line
        assert _for(value, 24) == (None, []), value
    spec = "16-64:1,24:2,40:0"  # later entries override
    want = {16: 1, 24: 2, 32: 1, 40: 0, 48: 1, 64: 1, 8: 0, 96: 0, 128: 0}
    for b, mode in want.items():
        got, lines = _for(spec, b)
        assert got == mode, (b, got, mode)
        assert len(lines) == (1 if mode else 0), (b, lines)
        if mode:
            assert lines[0].split()[2:4] == [f"mode={mode}", f"batch={b}"], lines
    assert _for(spec, 48) == (1, []), "one engagement line per batch per process"
    for bad in ("16-136:1", "16-32:4", "4-32:1", "16-32:yes", "72:2", "16-128:2"):
        try:
            _for(bad, 16)
        except ValueError:
            continue
        raise AssertionError(f"no ValueError for table {bad!r}")


def test_mid_wide_rows_nexact():
    # 65-128 rows (table only): two N-exact tiles, N = ceil(rows / 2) rounded up to 8 (hand-derived), 0 where that is 64
    ft = _ft()
    ft._n32_batches.clear()
    want = {40: 1, 64: 1, 65: 40, 72: 40, 80: 40, 81: 48, 88: 48, 96: 48, 97: 56, 104: 56, 112: 56, 113: 0, 120: 0, 128: 0}
    for b, n in want.items():
        got, lines = _for("33-128:1", b)
        assert got == n, (b, got, n)
        if b > 64:
            assert len(lines) == (1 if n else 0), (b, lines)
            if n:
                assert lines[0].split()[2:5] == ["mode=1", f"batch={b}", f"ntile={n}"], lines
    assert _for("33-128:1", 96) == (48, []), "one engagement line per batch per process"
    assert _for("33-64:1", 96) == (0, []), "a 65-128-row batch the table does not list stays N64"
    assert _for("33-64:1,96:0", 96) == (0, [])


def test_single_exact_tile_value3():
    # value 3: ONE N-exact tile of rows rounded up to 8 at 9-64 rows (hand-derived; N32 -> mode 1, N64 -> 0), two tiles above 64 rows
    ft = _ft()
    ft._n32_batches.clear()
    want = {9: 16, 16: 16, 17: 24, 24: 24, 25: 1, 32: 1, 33: 40, 40: 40, 41: 48, 48: 48, 56: 56, 57: 0, 64: 0, 72: 40, 96: 48, 104: 56, 128: 0}
    for b, n in want.items():
        got, lines = _for("9-128:3", b)
        assert got == n, (b, got, n)
        if n:
            assert len(lines) == 1 and lines[0].split()[2:5] == ["mode=3", f"batch={b}", f"ntile={32 if n == 1 else n}"], (b, lines)
        else:
            assert lines == [], (b, lines)


def _l2pf(value):
    ft = _ft()
    if value is None:
        os.environ.pop("K31MK_MOE_FRONT_L2PF", None)
    else:
        os.environ["K31MK_MOE_FRONT_L2PF"] = value
    ft._l2pf_engaged.clear()
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        out = ft.front_l2pf()
        out2 = ft.front_l2pf()
    os.environ.pop("K31MK_MOE_FRONT_L2PF", None)
    return out, out2, [x for x in err.getvalue().splitlines() if x.startswith("K31MK_MOE_FRONT_L2PF engaged")]


def test_l2pf_knob():
    # unset / empty / 0 = off and silent; 1 = on with exactly one line per process; anything else raises
    for value in (None, "", "0"):
        assert _l2pf(value) == (0, 0, []), (value, _l2pf(value))
    out, out2, lines = _l2pf("1")
    assert (out, out2, len(lines)) == (1, 1, 1), (out, out2, lines)
    for value in ("2", "yes", "16-32:1", "on"):
        try:
            _l2pf(value)
        except ValueError:
            continue
        raise AssertionError(f"no ValueError for K31MK_MOE_FRONT_L2PF={value!r}")


def test_lean_tx576_knobs():
    ft = _ft()
    for name, fn in (("K31MK_MOE_FRONT_LEAN", ft.front_lean), ("K31MK_MOE_FRONT_TX576", ft.front_tx576)):
        for value, want in ((None, 0), ("", 0), ("0", 0), ("1", 1)):
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
            ft._flag_engaged.discard(name)
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                got, got2 = fn(), fn()
            lines = [x for x in err.getvalue().splitlines() if x.startswith(f"{name} engaged")]
            assert (got, got2, len(lines)) == (want, want, want), (name, value, got, got2, lines)
        for value in ("2", "yes", "16-32:1"):
            os.environ[name] = value
            try:
                fn()
            except ValueError:
                continue
            finally:
                os.environ.pop(name, None)
            raise AssertionError(f"no ValueError for {name}={value!r}")
        os.environ.pop(name, None)


def test_xsep_knob():
    # K31MK_MOE_FRONT_XSEP: unset / empty / 0 = 0, 1 and 2 engage with one stderr line per process, anything else raises
    ft = _ft()
    name = "K31MK_MOE_FRONT_XSEP"
    for value, want in ((None, 0), ("", 0), ("0", 0), ("1", 1), ("2", 2)):
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
        ft._xsep_engaged.clear()
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            got, got2 = ft.front_xsep(), ft.front_xsep()
        lines = [x for x in err.getvalue().splitlines() if x.startswith(f"{name} engaged value=")]
        assert (got, got2, len(lines)) == (want, want, int(want > 0)), (value, got, got2, lines)
    for value in ("3", "yes", "16-32:1", "-1"):
        os.environ[name] = value
        try:
            ft.front_xsep()
        except ValueError:
            continue
        finally:
            os.environ.pop(name, None)
        raise AssertionError(f"no ValueError for {name}={value!r}")


def test_rfirst_knob():
    # K31MK_MOE_FRONT_RFIRST: unset / empty / 0 = 0, 1 engages with one stderr line per process, anything else raises
    ft = _ft()
    name = "K31MK_MOE_FRONT_RFIRST"
    for value, want in ((None, 0), ("", 0), ("0", 0), ("1", 1)):
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
        ft._flag_engaged.discard(name)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            got, got2 = ft.front_rfirst(), ft.front_rfirst()
        lines = [x for x in err.getvalue().splitlines() if x.startswith(f"{name} engaged pid=")]
        assert (got, got2, len(lines)) == (want, want, want), (value, got, got2, lines)
    for value in ("2", "yes", "65-128:1"):
        os.environ[name] = value
        try:
            ft.front_rfirst()
        except ValueError:
            continue
        finally:
            os.environ.pop(name, None)
        raise AssertionError(f"no ValueError for {name}={value!r}")


def test_bad_values_raise():
    # not 0 / 1 / 2; the table form in the scalar reader; on without FTM; on with an FTM batch above two N32 tiles (> 64 rows)
    for value, ftm in (("3", "16-32:1"), ("yes", "16-32:1"), ("16-32:1", "16-32:1"), ("1", None), ("1", "16-32:1,65:1"),
                       ("2", "16-64:1,128:1")):
        try:
            _call(value, ftm)
        except ValueError:
            continue
        raise AssertionError(f"no ValueError for K31MK_MOE_FRONT_N32={value!r} with K31MK_MOE_FRONT_TMEM={ftm!r}")


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
    os.environ.pop("K31MK_MOE_FRONT_N32", None)
    os.environ.pop("K31MK_MOE_FRONT_TMEM", None)
    print(f"UNIT {len(tests) - failed}/{len(tests)} passed", flush=True)
    return 1 if failed else 0


def run_compile(n32, out):
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
    os.environ["K31MK_MOE_FRONT_TMEM"] = "16-32:1"
    ft0 = _ft()
    init0 = ft0.MoeFrontTmem64.__init__

    def init_print(self, *a, **k):
        init0(self, *a, **k)
        xs = f" xsep={self.xsep} x_slot={','.join(map(str, self.x_slot))} x_wait={'|'.join('.'.join(map(str, w)) for w in self.x_wait)}" if getattr(self, 'xsep', 0) else ""
        print(f"BUILT smem={self.smem_bytes} tmem_tiles={self.tmem_tiles} smem_tiles={self.smem_tiles} tmem_cols={getattr(self, 'tmem_cols', 512)}{xs}", flush=True)

    ft0.MoeFrontTmem64.__init__ = init_print
    mbox_nx = 0
    if n32 in ("xs1", "xs2", "xs1n48", "xs1mb"):  # XSEP (MID form: N64 / N48 mbox 0; xs1mb: N64 mbox 1 = MID)
        os.environ["K31MK_MOE_FRONT_XSEP"] = "2" if n32 == "xs2" else "1"
        mbox_nx = int(n32 == "xs1mb")
        n32 = "nx48" if n32 == "xs1n48" else "nx0"
    elif n32 == "nx0mb":  # the XSEP off baseline of xs1mb
        mbox_nx, n32 = 1, "nx0"
    elif n32 in ("rf", "rfx1", "rfmb"):  # RFIRST (nx0; + XSEP 1; mbox 1)
        os.environ["K31MK_MOE_FRONT_RFIRST"] = "1"
        if n32 == "rfx1":
            os.environ["K31MK_MOE_FRONT_XSEP"] = "1"
        mbox_nx, n32 = int(n32 == "rfmb"), "nx0"
    if n32 in ("lean64", "lx64", "lx1", "lx48"):  # LEAN / TX576 (64 / 48: the MID form, mbox 0; 1: c=8 N32 mode 1, mbox 1)
        os.environ["K31MK_MOE_FRONT_LEAN"] = "1"
        if n32 != "lean64":
            os.environ["K31MK_MOE_FRONT_TX576"] = "1"
        n32 = {"lean64": "nx0", "lx64": "nx0", "lx1": "1", "lx48": "nx48"}[n32]
    if n32.startswith("nx"):  # 65-128-row form: MoeFrontTmem64 at n32 = N (0: N64), mbox 0, through the same served call
        ft = _ft()
        base_cls, n = ft.MoeFrontTmem64, int(n32[2:])

        class Nx(base_cls):
            def __init__(self, **kw):
                kw["n32"] = n
                super().__init__(**kw)

        ft.MoeFrontTmem64 = Nx
        os.environ.pop("K31MK_MOE_FRONT_N32", None)
        front.compile_front_tmem(64, 56, 224, 384, 0, promote=2, **({"mbox": 1} if mbox_nx else {}))
        ft.MoeFrontTmem64 = base_cls
    elif n32 in ("1", "2"):
        os.environ["K31MK_MOE_FRONT_N32"] = n32
    elif n32 in ("pf", "pf1"):  # L2PF on the served call at N64 (pf) or N32 mode 1 (pf1)
        os.environ["K31MK_MOE_FRONT_L2PF"] = "1"
        if n32 == "pf1":
            os.environ["K31MK_MOE_FRONT_N32"] = "1"
        else:
            os.environ.pop("K31MK_MOE_FRONT_N32", None)
    else:  # 'base' (the base tree) or 0
        os.environ.pop("K31MK_MOE_FRONT_N32", None)
    if not n32.startswith("nx"):
        front.compile_front_tmem(64, 56, 224, 384, 0, promote=2, mbox=1)  # the served call, unchanged
    os.makedirs(out, exist_ok=True)
    moved = 0
    for f in os.listdir(dump):
        if f.endswith(".ptx"):
            os.replace(os.path.join(dump, f), os.path.join(out, f))
            moved += 1
    print(f"COMPILE OK n32={n32} files={moved}", flush=True)
    return 0 if moved >= 1 else 1


TOKENS = re.compile(r"\b(tcgen05\.commit|mbarrier\.init|tcgen05\.alloc\.exclusive|tcgen05\.dealloc\.exclusive|tcgen05\.alloc\.cta_group|tcgen05\.dealloc\.cta_group|tcgen05\.relinquish_alloc_permit|cp\.async\.bulk\.prefetch\.tensor|cp\.async\.bulk\.tensor|st\.async|tcgen05\.mma|tcgen05\.cp|griddepcontrol|barrier\.cluster\.arrive|ld\.shared\.v2\.b32|ld\.shared\.b32|ld\.shared\.v4\.b32)")


def census(d):
    """Occurrences of each instruction token in the PTX (several may share one line inside an inline-asm block), plus
    the lines that carry each front MMA descriptor."""
    counts = {f"idesc{n}": 0 for n in IDESC}
    for f in sorted(os.listdir(d)):
        if f.endswith(".ptx"):
            for line in open(os.path.join(d, f)):
                code = line.split("//", 1)[0]
                for tok in TOKENS.findall(code):
                    counts[tok] = counts.get(tok, 0) + 1
                if "tcgen05.mma" in code:
                    for n, v in IDESC.items():
                        counts[f"idesc{n}"] += len(re.findall(rf"\b{v}\b", code))
    return counts


def run_check(off, on, mode="1"):
    a, b = census(off), census(on)
    ok = True

    def line(name, good, detail):
        nonlocal ok
        ok &= bool(good)
        print(f"CHECK {name} {detail}: {'PASS' if good else 'FAIL'}", flush=True)

    on_n32 = b["idesc32"] > 0 if mode == "1" else b["idesc32"] == 0  # mode 2: register descriptors, no immediate one
    line("idesc", a["idesc64"] > 0 and a["idesc32"] == 0 and on_n32 and b["idesc64"] == 0,
         f"mode {mode}: off N64/N32 {a['idesc64']}/{a['idesc32']} on N64/N32 {b['idesc64']}/{b['idesc32']}")
    line("tcgen05.mma", a.get("tcgen05.mma", 0) == b.get("tcgen05.mma", 0) > 0,
         f"off/on {a.get('tcgen05.mma', 0)}/{b.get('tcgen05.mma', 0)}")
    line("st.async", a.get("st.async", 0) > 0 and 2 * b.get("st.async", 0) == a.get("st.async", 0),
         f"off/on {a.get('st.async', 0)}/{b.get('st.async', 0)} (on = off / 2: 8 partial quads per row instead of 16)")
    line("griddepcontrol", a.get("griddepcontrol", 0) == b.get("griddepcontrol", 0) > 0,
         f"off/on {a.get('griddepcontrol', 0)}/{b.get('griddepcontrol', 0)}")
    line("tcgen05.cp", b.get("tcgen05.cp", 0) > a.get("tcgen05.cp", 0) > 0,
         f"off/on {a.get('tcgen05.cp', 0)}/{b.get('tcgen05.cp', 0)} (on stages 14 A tiles in tensor memory, off 12)")
    print(f"CHECK {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


def run_checknx(off, on, n, base_n="64"):
    n, base_n = int(n), int(base_n)
    a, b = census(off), census(on)
    ok = True

    def line(name, good, detail):
        nonlocal ok
        ok &= bool(good)
        print(f"CHECK {name} {detail}: {'PASS' if good else 'FAIL'}", flush=True)

    bk = f"idesc{base_n}"
    line("idesc", a[bk] > 0 and a[f"idesc{n}"] == 0 and b[f"idesc{n}"] > 0 and b[bk] == 0,
         f"N{n}: off N{base_n}/N{n} {a[bk]}/{a[f'idesc{n}']} on N{base_n}/N{n} {b[bk]}/{b[f'idesc{n}']}")
    for tok in ("tcgen05.mma", "tcgen05.cp", "griddepcontrol"):
        line(tok, a.get(tok, 0) == b.get(tok, 0) > 0, f"off/on {a.get(tok, 0)}/{b.get(tok, 0)}")
    line("st.async", a.get("st.async", 0) > 0 and base_n * b.get("st.async", 0) == n * a.get("st.async", 0),
         f"off/on {a.get('st.async', 0)}/{b.get('st.async', 0)} (on = off x {n} / {base_n}: {n // 4} partial quads per row instead of {base_n // 4})")
    narrow = ("ld.shared.v2.b32", "ld.shared.b32")
    line("narrow loads", all(b.get(k, 0) <= a.get(k, 0) for k in narrow),
         f"off/on v2 {a.get(narrow[0], 0)}/{b.get(narrow[0], 0)} b32 {a.get(narrow[1], 0)}/{b.get(narrow[1], 0)} (owner sums by (row, quad) items: no narrow loads)")
    v4 = "ld.shared.v4.b32"
    line(v4, b.get(v4, 0) == a.get(v4, 0) > 0, f"off/on {a.get(v4, 0)}/{b.get(v4, 0)} (owner quad loads: 2 rounds x 8 slots = N64's 16, 1 x 8 = N32's 8)")
    print(f"CHECK {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


def run_checklx(off, on, tiles):
    a, b = census(off), census(on)
    tiles = int(tiles)
    ok = True

    def line(name, good, detail):
        nonlocal ok
        ok &= bool(good)
        print(f"CHECK {name} {detail}: {'PASS' if good else 'FAIL'}", flush=True)

    ex = ("tcgen05.alloc.exclusive", "tcgen05.dealloc.exclusive")
    ne = ("tcgen05.alloc.cta_group", "tcgen05.dealloc.cta_group", "tcgen05.relinquish_alloc_permit")
    line("exclusive", all(a.get(t, 0) == 0 and b.get(t, 0) == 1 for t in ex), "off/on " + " ".join(f"{t} {a.get(t, 0)}/{b.get(t, 0)}" for t in ex))
    line("non-exclusive", all(a.get(t, 0) == 1 and b.get(t, 0) == 0 for t in ne), "off/on " + " ".join(f"{t} {a.get(t, 0)}/{b.get(t, 0)}" for t in ne))
    line("tcgen05.cp", b.get("tcgen05.cp", 0) == 4 * tiles, f"off/on {a.get('tcgen05.cp', 0)}/{b.get('tcgen05.cp', 0)} (want on = 4 x {tiles})")
    for tok in ["tcgen05.mma", "st.async", "griddepcontrol", "ld.shared.v4.b32"] + [k for k in sorted(a) if k.startswith("idesc")]:
        line(tok, a.get(tok, 0) == b.get(tok, 0), f"off/on {a.get(tok, 0)}/{b.get(tok, 0)}")
    print(f"CHECK {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


def run_checkxs(off, on, extra, cluster="0"):
    """XSEP: the same MMAs / copies / descriptors / pushes / PDL / allocation as OFF; EXTRA more tcgen05.commit and mbarrier.init."""
    a, b = census(off), census(on)
    extra = int(extra)
    ok = True

    def line(name, good, detail):
        nonlocal ok
        ok &= bool(good)
        print(f"CHECK {name} {detail}: {'PASS' if good else 'FAIL'}", flush=True)

    # the structural deltas XSEP must show (first run: x's TMA loop is unrolled over the 14 K tiles because each tile has its
    # own slot, 1 -> 14 tensor loads; warp 3 reads the tensor-memory address from shared memory once)
    delta = {"tcgen05.commit": extra, "cp.async.bulk.tensor": 13, "ld.shared.b32": 1}
    if int(cluster):  # XSEP v2 at mbox 0: the MMA warp (warp 8) passes the cluster barrier phase too
        delta["barrier.cluster.arrive"] = int(cluster)
    for tok, d in delta.items():
        line(tok, a.get(tok, 0) > 0 and b.get(tok, 0) == a.get(tok, 0) + d, f"off/on {a.get(tok, 0)}/{b.get(tok, 0)} (want on = off + {d})")
    tok = "mbarrier.init"  # the init loops may stay loops in the PTX: more, not an exact count
    line(tok, b.get(tok, 0) > a.get(tok, 0) > 0, f"off/on {a.get(tok, 0)}/{b.get(tok, 0)} (want on > off)")
    for tok in sorted(set(a) | set(b)):
        if tok in delta or tok == "mbarrier.init":
            continue
        line(tok, a.get(tok, 0) == b.get(tok, 0), f"off/on {a.get(tok, 0)}/{b.get(tok, 0)}")
    print(f"CHECK {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


def run_checkrf(off, on):
    """RFIRST: every census token count equal (the cluster -> tile mapping adds index arithmetic only)."""
    a, b = census(off), census(on)
    ok = True
    for tok in sorted(set(a) | set(b)):
        good = a.get(tok, 0) == b.get(tok, 0)
        ok &= good
        print(f"CHECK {tok} off/on {a.get(tok, 0)}/{b.get(tok, 0)}: {'PASS' if good else 'FAIL'}", flush=True)
    print(f"CHECK {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


def run_checkpf(off, on, count):
    a, b = census(off), census(on)
    pf = "cp.async.bulk.prefetch.tensor"
    ok = a.get(pf, 0) == 0 and b.get(pf, 0) == int(count)
    print(f"CHECK prefetch off/on {a.get(pf, 0)}/{b.get(pf, 0)} (want 0/{count}): {'PASS' if ok else 'FAIL'}", flush=True)
    for tok in sorted(set(a) | set(b)):
        if tok == pf:
            continue
        good = a.get(tok, 0) == b.get(tok, 0)
        ok &= good
        print(f"CHECK {tok} off/on {a.get(tok, 0)}/{b.get(tok, 0)}: {'PASS' if good else 'FAIL'}", flush=True)
    print(f"CHECK {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "unit"
    if mode == "unit":
        sys.exit(run_unit())
    if mode == "compile":
        sys.exit(run_compile(sys.argv[2], sys.argv[3]))
    if mode == "check":
        sys.exit(run_check(*sys.argv[2:5]))
    if mode == "checklx":
        sys.exit(run_checklx(*sys.argv[2:5]))
    if mode == "checkpf":
        sys.exit(run_checkpf(*sys.argv[2:5]))
    if mode == "checkxs":
        sys.exit(run_checkxs(*sys.argv[2:6]))
    if mode == "checkrf":
        sys.exit(run_checkrf(*sys.argv[2:4]))
    if mode == "checknx":
        sys.exit(run_checknx(*sys.argv[2:6]))
    raise SystemExit(f"unknown mode {mode}")

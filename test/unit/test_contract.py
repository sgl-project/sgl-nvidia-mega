# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The SGLang names this package reads and the binding code SGLang tolerates it
calling, pinned on CPU against the installed SGLang.

The megakernel lives outside SGLang, so it reads SGLang by name (its specs plug into
``sglang.kernels.plugin_kernels.nvidia``). Every module that imports from
``sglang`` lists what it reads in ``SGLANG_READS``. A refactor that renames one
fails here, naming the file to repair, instead of failing at a GPU server's startup.
A module whose stock step mirrors an SGLang function lists it in ``SGLANG_MIRRORS``
with the digest of its code, so a changed branch fails here too. Code moved out of
SGLang into the package is listed in ``SGLANG_MOVED`` the same way: while SGLang still
defines it, both copies must be the same code; once SGLang drops it, the pin retires.
A DFlash runtime declares the worker names it binds (``DFLASH_BINDINGS``) and the
guard values it serves (``DFLASH_GUARDS``); SGLang's DFlash binding must bind and
guard each by name. A file that moved between SGLang versions is named newest first,
``"new.py|old.py"``; the first one present is checked. A read name such as
"replication|layout" accepts either field.
An explicitly guarded opt-in extension uses ``SGLANG_OPTIONAL_READS``: its
file may be absent, but all declared names must match when it is installed.
The feature's tests must cover rejection when its extension is unavailable.
The checks are static (AST and tokens): no package module is
imported. Run them against every supported SGLang and nightly against main. The
GLM-5.3 package reads SGLang's megakernel DSA binding view, and its draft spec
SGLang's draft capture check marker, and the Kimi K3.1 DFlash runtime SGLang's DFlash
binding; against an SGLang without them, those reads and pins are skipped, naming each.
The MiniMax-M3.1 package needs SGLang support newer than the oldest supported SGLang;
against an SGLang without it its reads and pins are skipped, naming each file.
"""

import ast
import hashlib
import io
import tokenize
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pytest
import sglang

import sgl_nvidia_mega

ROOT = Path(sglang.__file__).resolve().parent  # the installed sglang package
PACKAGE = Path(sgl_nvidia_mega.__file__).resolve().parent


def _files(pattern: str) -> list:
    """The files a ``SGLANG_READS`` key names: its first alternative that exists.
    A key like ``"new.py|old.py"`` tries the first path; if no file matches, tries
    the next, so files that moved between SGLang versions are named newest first."""
    for alternative in pattern.split("|"):
        files = sorted(ROOT.glob(alternative))
        if files:
            return files
    return []


@lru_cache(maxsize=None)
def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


@lru_cache(maxsize=None)
def _provided(path: Path) -> frozenset:
    """What a source file provides: definitions, assignment targets (``x.name = ...``
    too), imported names, parameters as ``f(p)`` and string literals as ``"x"``."""
    names = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            if not isinstance(node, ast.ClassDef):
                params = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
                names.update(f"{node.name}({p.arg})" for p in params)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for t in ast.walk(target):
                    if isinstance(t, ast.Name):
                        names.add(t.id)
                    elif isinstance(t, ast.Attribute):
                        names.add(t.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.add(f'"{node.value}"')
    return frozenset(names)


def _declared(path: Path, kind: str = "SGLANG_READS") -> dict:
    """The module's named source contract, or {} without one."""
    for node in _tree(path).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == kind for t in node.targets
        ):
            return ast.literal_eval(node.value)
    return {}


def _sglang_imports(path: Path) -> set:
    return {
        a.asname or a.name
        for node in ast.walk(_tree(path))
        if isinstance(node, ast.ImportFrom)
        and (node.module or "").startswith("sglang.")
        for a in node.names
    }


READERS = [
    p
    for p in sorted(PACKAGE.rglob("*.py"))
    if _declared(p) or _declared(p, "SGLANG_OPTIONAL_READS") or _sglang_imports(p)
]


def _id(path: Path) -> str:
    return str(path.relative_to(PACKAGE.parent))


@pytest.mark.parametrize("path", READERS, ids=_id)
def test_every_sglang_import_is_declared(path):
    contracts = _declared(path) | _declared(path, "SGLANG_OPTIONAL_READS")
    declared = {n for names in contracts.values() for n in names}
    undeclared = _sglang_imports(path) - declared
    assert not undeclared, f"{_id(path)}: declare {sorted(undeclared)} in SGLANG_READS"


@pytest.mark.parametrize("path", READERS, ids=_id)
def test_sglang_provides_what_the_plugin_reads(path):
    if _id(path).startswith(K31_PACKAGE) and not _has_k31_host():
        pytest.skip(f"{NO_K31_HOST}: {_id(path)}")
    if _id(path).startswith(M31_PACKAGE) and not _has_m31_host():
        pytest.skip(f"{NO_M31_HOST}: {_id(path)}")
    missing, unchecked = [], []
    required = _declared(path)
    optional = _declared(path, "SGLANG_OPTIONAL_READS")
    assert not required.keys() & optional.keys(), (
        "contract cannot be both required and optional"
    )
    for pattern, names in (required | optional).items():
        files = _files(pattern)
        if not files and pattern in optional:
            continue
        assert files, f"{_id(path)}: no file matches any alternative in {pattern!r}"
        provided = frozenset().union(*map(_provided, files))
        skipped = []
        if files == _files(BINDING) and not _has_dsa_view():
            skipped = [n for n in names if n.split("(", 1)[0] in DSA_VIEW_NAMES]
            reason = NO_DSA_VIEW
        if files == _files(DRAFT_CHECK) and not _has_draft_check():
            skipped = [n for n in names if n in DRAFT_CHECK_NAMES]
            reason = NO_DRAFT_CHECK
        if skipped:
            unchecked.append(f"{reason}: {_id(path)} reads {sorted(skipped)}")
            names = [n for n in names if n not in skipped]
        missing += [
            f"{pattern}: {n}"
            for n in names
            if not any(alternative in provided for alternative in n.split("|"))
        ]
    assert not missing, (
        f"SGLang no longer provides what {_id(path)} reads; repair it and its "
        "SGLANG_READS:\n  " + "\n  ".join(missing)
    )
    if unchecked:
        pytest.skip("; ".join(unchecked))


# What SGLang's binding file tolerates the megakernel still calling, writing and
# reading; any change there must be reviewed here
BINDING = "srt/kernel_plugins/nvidia/binding.py|srt/megakernels/binding.py"
PINNED_TOLERATED = {
    "TOLERATED_CALLABLES": (
        "ForwardView.attention",
        "ForwardView.store_cache",
        "ForwardView.swa_out_cache_loc",
        "ForwardView.low_ratio_sources",
        "ForwardView.core_compressor",
        "KdaStepView.track_decode_state",
        "KvPoolView.set_swa_key_buffer_radix_fused_norm_rope",
        "KvPoolView.pool",
        "sglang.kernels.ops",
        "the replaced model's modules, weights and quant methods",
    ),
    "TOLERATED_WRITES": (
        "ForwardView.metadata.q_pad_buffer",
        "ForwardView.metadata.low_ratio_req_indices",
        "ForwardView.metadata.low_ratio_pos_i64",
    ),
    "TOLERATED_READS": (
        "runtime_context getters (get_parallel, get_exec, get_spec, get_platform)",
        "capture-mode queries (get_is_capture_mode, is_in_breakable_cuda_graph)",
        "MegakernelBinding.engram_host_table",
    ),
    "TOLERATED_DSA_CALLABLES": (
        "KvPoolView.set_mla_kv_buffer",
        "DsaView.indexer_metadata",
    ),
    "TOLERATED_DSA_WRITES": (
        "KvPoolView.get_key_buffer",
        "KvPoolView.get_index_k_with_scale_buffer",
    ),
}
# What SGLang's megakernel DSA binding view adds to binding.py (the DSA view and the
# MLA pool members): an SGLang without it skips these reads and pins
DSA_VIEW_PINS = ("TOLERATED_DSA_CALLABLES", "TOLERATED_DSA_WRITES")
DSA_VIEW_NAMES = frozenset(
    {
        "dsa",
        "DsaView",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "kv_cache_dim",
        "physical_page_size",
        "decode_impl",
        "use_mha",
        "kv_cache_dtype",
        "use_fused_topk",
        "topk_backend",
        "dcp_size",
        "index_kpool",
        "hisparse",
        "workspace_buffer",
        "multi_ctas_kv_counter_buffer",
        "indexer_metadata",
        "get_key_buffer",
        "get_index_k_with_scale_buffer",
        "index_page_size",
        "set_mla_kv_buffer",
    }
)
NO_DSA_VIEW = "this SGLang has no megakernel DSA binding view (ForwardView.dsa)"
# The EAGLE draft runners' capture check marker in spec.py
DRAFT_CHECK, DRAFT_CHECK_NAMES = (
    "kernels/plugin_kernels/nvidia/spec.py|srt/megakernels/spec.py",
    ("EAGLE_DRAFT_CAPTURE_CHECK",),
)
NO_DRAFT_CHECK = "this SGLang's EAGLE draft graph runners run no draft capture check"


# The Kimi K3.1 megakernel reads SGLang's K3.1 host bindings (sgl-project/sglang#48414);
# against an SGLang without them its spec refuses every configuration, and its reads
# are skipped here, naming each file.
K31_PACKAGE = "sgl_nvidia_mega/kimi_k31/"
K31_HOST = "srt/kernel_plugins/nvidia/kimi_k31_verify_binding.py"
NO_K31_HOST = "this SGLang has no Kimi K3.1 megakernel host bindings"


def _has_k31_host() -> bool:
    return bool(_files(K31_HOST))


# The MiniMax-M3.1 megakernel mirrors SGLang's M3.1 step code and reads its DSpark
# switches as of sgl-project/sglang#46693, #42855 and #49595; against an older SGLang
# its spec refuses every configuration (spec.host_reasons), and its reads and mirror
# pins are skipped here, naming each file. The newest of the three is the marker.
M31_PACKAGE = "sgl_nvidia_mega/minimax_m31/"
M31_HOST = ("srt/arg_groups/fields/*.py", "speculative_dspark_lm_head_dtype")
NO_M31_HOST = "this SGLang predates the MiniMax-M3.1 support the megakernel is built on"


def _has_m31_host() -> bool:
    pattern, name = M31_HOST
    return name in frozenset().union(*map(_provided, _files(pattern)))


def _has_draft_check() -> bool:
    return DRAFT_CHECK_NAMES[0] in _provided(_files(DRAFT_CHECK)[0])


def _has_dsa_view() -> bool:
    return any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == DSA_VIEW_PINS[0] for t in node.targets
        )
        for node in _tree(_files(BINDING)[0]).body
    )


def _binding_tuple(name: str) -> tuple:
    for node in _tree(_files(BINDING)[0]).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return tuple(ast.literal_eval(node.value))
    raise AssertionError(f"{BINDING} defines no {name}")


@pytest.mark.parametrize("name", sorted(PINNED_TOLERATED))
def test_the_binding_tolerates_exactly_the_pinned_sglang_code(name):
    if name in DSA_VIEW_PINS and not _has_dsa_view():
        pytest.skip(NO_DSA_VIEW)
    assert _binding_tuple(name) == PINNED_TOLERATED[name]


def _binding_functions() -> frozenset:
    """The public functions of SGLang's binding file: its megakernel API (in
    guards.py and runtime.py before sgl-project/sglang#45389), not view members."""
    return frozenset(
        node.name
        for node in _tree(_files(BINDING)[0]).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    )


def test_every_binding_callable_the_plugin_reads_is_tolerated():
    """A ForwardView / KvPoolView / DsaView callable a module declares must be on
    SGLang's tolerated lists (the data fields, the pool's own getters and the
    binding file's public functions need not be)."""
    callables = {
        name.split(".", 1)[1]
        for key in ("TOLERATED_CALLABLES", "TOLERATED_DSA_CALLABLES")
        for name in PINNED_TOLERATED[key]
        if name.startswith(("ForwardView.", "KdaStepView.", "KvPoolView.", "DsaView."))
    }
    data = {
        "ForwardView",
        "HybridPoolView",
        "HybridStepView",
        "PreparationView",
        "aux_hidden_states",
        "block_table",
        "cache",
        "capacity",
        "conv",
        "intermediate_conv_window",
        "intermediate_ssm",
        "kda",
        "layer_mapping",
        "logical_dtype",
        "KvPoolView",
        "DsaView",
        "MegakernelBinding",
        "metadata",
        "mla",
        "page_size",
        "post_capture_active",
        "reserved_capacity",
        "sequence_lengths",
        "serving_capacity",
        "state",
        "state_indices",
        "storage_dtype",
        "temporal",
        "tokens_per_request",
        "verify_scratch_indices",
        "kv_pool",
        "trtllm_attn",
        "uniform_fp8",
        "request_window",
        "write_slots",
        "forward_view",
        "engram_history",
        "engram_host_table",
        # the DSA backend's view: its layout and state, and the pool's MLA facts
        "dsa",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "kv_cache_dim",
        "physical_page_size",
        "decode_impl",
        "use_mha",
        "kv_cache_dtype",
        "use_fused_topk",
        "topk_backend",
        "dcp_size",
        "index_kpool",
        "hisparse",
        "workspace_buffer",
        "multi_ctas_kv_counter_buffer",
        "index_page_size",
        # the Kimi K3.1 binding's hybrid pool, DSA indexer and cache views
        "hybrid",
        "indexer",
        "caches",
    }
    # Startup agreement is a host preparation API, not a borrowed runtime-view
    # callable to replace with data. Keep its exact signature covered separately.
    preparation = {"megakernel_agree"}
    for path in READERS:
        names = {
            n.split("(", 1)[0]
            for pattern, declared in _declared(path).items()
            if _files(pattern) == _files(BINDING)
            for n in declared
        }
        untolerated = names - callables - data - preparation - _binding_functions()
        untolerated = {n for n in untolerated if not n.startswith("get_")}
        assert not untolerated, f"{_id(path)}: {sorted(untolerated)}"


def test_native_preparation_agreement_signature():
    binding = _tree(_files(BINDING)[0])
    agree = next(
        node
        for node in binding.body
        if isinstance(node, ast.FunctionDef) and node.name == "megakernel_agree"
    )
    assert [arg.arg for arg in agree.args.args] == ["value"]
    assert [arg.arg for arg in agree.args.kwonlyargs] == ["what", "group"]
    assert agree.args.kw_defaults[0] is None
    assert ast.literal_eval(agree.args.kw_defaults[1]) is None
    assert agree.args.vararg is None and agree.args.kwarg is None


def _mirrors(path: Path) -> dict:
    """The module's ``SGLANG_MIRRORS``, or {} without one."""
    for node in _tree(path).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "SGLANG_MIRRORS" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    return {}


def mirror_digest(path: Path, qualname: str) -> Optional[str]:
    """The digest of a function's code in ``path`` (``Class.method`` or ``function``):
    its decorators, signature and body as tokens, without comments, blank lines or
    docstring; None when the file does not define it."""
    scope = _tree(path).body
    *owners, name = qualname.split(".")
    for owner in owners:
        cls = next(
            (n for n in scope if isinstance(n, ast.ClassDef) and n.name == owner), None
        )
        if cls is None:
            return None
        scope = cls.body
    node = next(
        (
            n
            for n in scope
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
        ),
        None,
    )
    if node is None:
        return None
    first = min([node.lineno, *(d.lineno for d in node.decorator_list)])
    lines = path.read_text().splitlines(keepends=True)[first - 1 : node.end_lineno]
    doc = node.body[0]
    has_doc = (
        isinstance(doc, ast.Expr)
        and isinstance(doc.value, ast.Constant)
        and isinstance(doc.value.value, str)
    )
    skip = (doc.lineno - first + 1, doc.end_lineno - first + 1) if has_doc else None
    digest = hashlib.sha256()
    for tok in tokenize.generate_tokens(io.StringIO("".join(lines)).readline):
        if tok.type in (tokenize.COMMENT, tokenize.NL):
            continue
        if skip and skip[0] <= tok.start[0] and tok.end[0] <= skip[1]:
            continue
        # indentation depth is structure, not text
        text = "" if tok.type in (tokenize.INDENT, tokenize.NEWLINE) else tok.string
        digest.update(f"{tok.type}:{text}\n".encode())
    return digest.hexdigest()[:16]


MIRRORS = [
    (path, key, digest)
    for path in sorted(PACKAGE.rglob("*.py"))
    for key, digest in _mirrors(path).items()
]


@pytest.mark.parametrize(
    "path, key, digest", MIRRORS, ids=[f"{_id(p)}::{k}" for p, k, _ in MIRRORS]
)
def test_the_mirrored_sglang_code_is_unchanged(path, key, digest):
    if _id(path).startswith(M31_PACKAGE) and not _has_m31_host():
        pytest.skip(f"{NO_M31_HOST}: {_id(path)}")
    pattern, qualname = key.split(":")
    files = _files(pattern)  # "new.py|old.py": the first one present
    assert len(files) == 1, f"{_id(path)}: {pattern} matches {len(files)} files"
    got = mirror_digest(files[0], qualname)
    assert got is not None, f"{_id(path)}: SGLang no longer defines {key}"
    assert got in ((digest,) if isinstance(digest, str) else digest), (
        f"SGLang changed {key}, whose body {_id(path)} mirrors: re-read it, update "
        f"the mirror, then pin {got!r} in its SGLANG_MIRRORS (a tuple where the "
        "supported SGLangs differ)"
    )


MOVED = [
    (path, key, digest)
    for path in sorted(PACKAGE.rglob("*.py"))
    for key, digest in _declared(path, "SGLANG_MOVED").items()
]


@pytest.mark.parametrize(
    "path, key, digest", MOVED, ids=[f"{_id(p)}::{k}" for p, k, _ in MOVED]
)
def test_code_moved_from_sglang_matches_while_sglang_keeps_it(path, key, digest):
    if _id(path).startswith(K31_PACKAGE) and not _has_k31_host():
        pytest.skip(f"{NO_K31_HOST}: {_id(path)}")
    pattern, qualname = key.split(":")
    files = sorted(ROOT.glob(pattern))
    got = mirror_digest(files[0], qualname) if len(files) == 1 else None
    if got is None:
        pytest.skip(f"SGLang no longer defines {key}; {_id(path)} owns it")
    assert got == digest, (
        f"SGLang changed {key}, which {_id(path)} moved here: port the change to "
        f"the package's copy, then pin {got!r} in its SGLANG_MOVED"
    )


# SGLang's DFlash binding (sglang.srt.kernel_plugins.nvidia.kimi_k31_dflash_binding): an SGLang without it
# never builds BuiltK31Verify.dflash_runtime, so the runtime's pins are skipped there.
DFLASH_BINDING = "srt/kernel_plugins/nvidia/kimi_k31_dflash_binding.py"
NO_DFLASH_BINDING = f"this SGLang has no DFlash runtime binding ({DFLASH_BINDING})"


def _assigned(tree: ast.Module, name: str) -> Optional[ast.expr]:
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else []
        if isinstance(node, ast.AnnAssign):
            targets = [node.target]
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            return node.value
    return None


def _strings(node: Optional[ast.expr]) -> frozenset:
    """The string keys of a dict literal, or the strings of a frozenset({...}) call."""
    if isinstance(node, ast.Dict):
        return frozenset(ast.literal_eval(k) for k in node.keys)
    if isinstance(node, ast.Call) and node.args:
        return frozenset(ast.literal_eval(node.args[0]))
    return frozenset()


DFLASH_RUNTIMES = [
    path
    for path in sorted(PACKAGE.rglob("*.py"))
    if _assigned(_tree(path), "DFLASH_BINDINGS") is not None
]


@pytest.mark.parametrize("path", DFLASH_RUNTIMES, ids=_id)
def test_the_dflash_binding_binds_and_guards_what_the_runtime_declares(path):
    files = _files(DFLASH_BINDING)
    if not files:
        pytest.skip(f"{NO_DFLASH_BINDING}: {_id(path)}")
    binding, runtime = _tree(files[0]), _tree(path)
    missing = _strings(_assigned(runtime, "DFLASH_BINDINGS")) - _strings(
        _assigned(binding, "BINDINGS")
    )
    assert not missing, f"{_id(path)}: SGLang does not bind {sorted(missing)}"
    unguarded = _strings(_assigned(runtime, "DFLASH_GUARDS")) - _strings(
        _assigned(binding, "GUARDS")
    )
    assert not unguarded, f"{_id(path)}: SGLang has no guard key {sorted(unguarded)}"


def _built_classes() -> dict:
    """Each package class SGLang may hold as a built megakernel (one with plan,
    forward and summary, its own or a package base's), with what it assigns."""
    classes = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        for node in ast.walk(_tree(path)):
            if not isinstance(node, ast.ClassDef):
                continue
            names, methods = set(), set()
            for item in node.body:
                if isinstance(item, ast.AnnAssign):
                    names |= {
                        n.id for n in ast.walk(item.target) if isinstance(n, ast.Name)
                    }
                elif isinstance(item, ast.Assign):
                    names |= {
                        n.id
                        for target in item.targets
                        for n in ast.walk(target)
                        if isinstance(n, ast.Name)
                    }
                elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.add(item.name)
                    names |= {
                        n.attr
                        for n in ast.walk(item)
                        if isinstance(n, ast.Attribute)
                        and isinstance(n.ctx, ast.Store)
                        and isinstance(n.value, ast.Name)
                        and n.value.id == "self"
                    }
            bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
            classes[node.name] = (path, bases, names, methods)

    def inherited(name: str, index: int) -> set:
        if name not in classes:
            return set()
        entry = classes[name]
        return set(entry[index]).union(*(inherited(b, index) for b in entry[1]))

    return {
        f"{_id(entry[0])}::{name}": inherited(name, 2)
        for name, entry in classes.items()
        if {"plan", "forward", "summary"} <= inherited(name, 3)
    }


BUILT = _built_classes()


@pytest.mark.parametrize("name", sorted(BUILT))
def test_every_built_object_defines_speculative_runtime_hooks(name):
    # SGLang's BuiltMegakernel requires the member (None without a factory) and
    # reads it unguarded under SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.
    assert "speculative_runtime_hooks" in BUILT[name], name


def _plugin_imports_in_sglang() -> list:
    """Each ``from sgl_nvidia_mega.<module> import <name>`` in SGLang's own code."""
    sites = []
    for path in sorted(ROOT.rglob("*.py")):
        if "sgl_nvidia_mega" not in path.read_text(errors="ignore"):
            continue
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "sgl_nvidia_mega."
            ):
                rel = path.relative_to(ROOT)
                sites += [
                    (f"{rel}:{node.lineno}", node.module, a.name) for a in node.names
                ]
    return sites


def _module_level_names(module: str) -> Optional[frozenset]:
    """The names a package module binds at module level (defs, assignments,
    imports, submodules), from its source; None when there is no such module."""
    rel = Path(*module.split(".")[1:])
    for path in (PACKAGE / rel.with_suffix(".py"), PACKAGE / rel / "__init__.py"):
        if not path.is_file():
            continue
        names = set()
        if path.name == "__init__.py":  # a package's submodules import by name too
            names |= {p.stem for p in path.parent.glob("*.py")}
            names |= {p.parent.name for p in path.parent.glob("*/__init__.py")}
        todo = list(_tree(path).body)
        while todo:
            node = todo.pop()
            if isinstance(node, (ast.If, ast.Try)):
                todo += node.body + node.orelse + getattr(node, "finalbody", [])
                todo += [s for h in getattr(node, "handlers", []) for s in h.body]
            elif isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                names.add(node.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                names |= {
                    n.id
                    for t in targets
                    for n in ast.walk(t)
                    if isinstance(n, ast.Name)
                }
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names |= {(a.asname or a.name).split(".")[0] for a in node.names}
        return frozenset(names)
    return None


PLUGIN_IMPORTS = _plugin_imports_in_sglang()


# SGLang's host bindings (the Kimi K3.1 ones) import package modules by path: a move
# or rename here fails at a GPU server's first covered step, so pin each name.
@pytest.mark.parametrize(
    "site, module, name",
    PLUGIN_IMPORTS
    or [
        pytest.param(
            None,
            None,
            None,
            marks=pytest.mark.skip(
                reason="this SGLang imports nothing from the package by name"
            ),
        )
    ],
    ids=[f"{s}:{n}" for s, _, n in PLUGIN_IMPORTS] or None,
)
def test_the_package_defines_what_sglang_imports_from_it(site, module, name):
    names = _module_level_names(module)
    assert names is not None, f"SGLang {site} imports the missing module {module}"
    assert name in names, f"SGLang {site} imports {module}.{name}, which is not defined"

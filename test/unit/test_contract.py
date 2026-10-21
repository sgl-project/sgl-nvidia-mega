# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The SGLang names this package reads and the binding code SGLang tolerates it
calling, pinned on CPU against the installed SGLang.

The megakernel lives outside SGLang, so it reads SGLang by name (its specs plug into
``sglang.srt.megakernels``). Every module that imports from
``sglang`` lists what it reads in ``SGLANG_READS``. A refactor that renames one fails here, naming the file to
repair, instead of failing at a GPU server's startup. A module whose stock step mirrors
the body of an SGLang function lists it in ``SGLANG_MIRRORS`` with the digest of its
code (its tokens without comments or docstring; a tuple of digests where the supported
SGLangs differ), so a changed branch fails here too.
The checks are static (AST and tokens): no module of this package is imported. Run it
against each SGLang this package supports, and nightly against SGLang main.
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


def _declared(path: Path) -> dict:
    """The module's ``SGLANG_READS``, or {} without one."""
    for node in _tree(path).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "SGLANG_READS" for t in node.targets
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
    p for p in sorted(PACKAGE.rglob("*.py")) if _declared(p) or _sglang_imports(p)
]


def _id(path: Path) -> str:
    return str(path.relative_to(PACKAGE.parent))


@pytest.mark.parametrize("path", READERS, ids=_id)
def test_every_sglang_import_is_declared(path):
    declared = {n for names in _declared(path).values() for n in names}
    undeclared = _sglang_imports(path) - declared
    assert not undeclared, f"{_id(path)}: declare {sorted(undeclared)} in SGLANG_READS"


@pytest.mark.parametrize("path", READERS, ids=_id)
def test_sglang_provides_what_the_plugin_reads(path):
    missing = []
    for pattern, names in _declared(path).items():
        files = sorted(ROOT.glob(pattern))
        assert files, f"{_id(path)}: no file matches {pattern}"
        provided = frozenset().union(*map(_provided, files))
        missing += [f"{pattern}: {n}" for n in names if n not in provided]
    assert not missing, (
        f"SGLang no longer provides what {_id(path)} reads; repair it and its "
        "SGLANG_READS:\n  " + "\n  ".join(missing)
    )


# What SGLang's binding file tolerates the megakernel still calling, writing and
# reading; any change there must be reviewed here
BINDING = "srt/megakernels/binding.py"
PINNED_TOLERATED = {
    "TOLERATED_CALLABLES": (
        "ForwardView.attention",
        "ForwardView.store_cache",
        "ForwardView.swa_out_cache_loc",
        "ForwardView.low_ratio_sources",
        "ForwardView.core_compressor",
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
}


def _binding_tuple(name: str) -> tuple:
    for node in _tree(ROOT / BINDING).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return tuple(ast.literal_eval(node.value))
    raise AssertionError(f"{BINDING} defines no {name}")


@pytest.mark.parametrize("name", sorted(PINNED_TOLERATED))
def test_the_binding_tolerates_exactly_the_pinned_sglang_code(name):
    assert _binding_tuple(name) == PINNED_TOLERATED[name]


def test_every_binding_callable_the_plugin_reads_is_tolerated():
    """A ForwardView / KvPoolView callable a module declares must be on SGLang's
    tolerated list (the data fields and the pool's own getters need not be)."""
    callables = {
        name.split(".", 1)[1]
        for name in PINNED_TOLERATED["TOLERATED_CALLABLES"]
        if name.startswith(("ForwardView.", "KvPoolView."))
    }
    data = {
        "ForwardView",
        "KvPoolView",
        "MegakernelBinding",
        "metadata",
        "kv_pool",
        "trtllm_attn",
        "uniform_fp8",
        "request_window",
        "forward_view",
        "engram_history",
        "engram_host_table",
    }
    for path in READERS:
        names = {n.split("(", 1)[0] for n in _declared(path).get(BINDING, ())}
        untolerated = names - callables - data
        untolerated = {n for n in untolerated if not n.startswith("get_")}
        assert not untolerated, f"{_id(path)}: {sorted(untolerated)}"


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
    pattern, qualname = key.split(":")
    files = sorted(ROOT.glob(pattern))
    assert len(files) == 1, f"{_id(path)}: {pattern} matches {len(files)} files"
    got = mirror_digest(files[0], qualname)
    assert got is not None, f"{_id(path)}: SGLang no longer defines {key}"
    assert got in ((digest,) if isinstance(digest, str) else digest), (
        f"SGLang changed {key}, whose body {_id(path)} mirrors: re-read it, update "
        f"the mirror, then pin {got!r} in its SGLANG_MIRRORS (a tuple where the "
        "supported SGLangs differ)"
    )

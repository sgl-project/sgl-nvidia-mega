# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The SGLang names this package reads and the binding code SGLang tolerates it
calling, pinned on CPU against the installed SGLang.

The megakernel lives outside SGLang, so it reads SGLang by name (its specs plug into
``sglang.kernels.plugin_kernels.nvidia``). Every module that imports from
``sglang`` lists what it reads in ``SGLANG_READS``. A refactor that renames one
fails here, naming the file to repair, instead of failing at a GPU server's startup.
A module whose stock step mirrors an SGLang function lists it in ``SGLANG_MIRRORS``
with the digest of its code, so a changed branch fails here too. A file that moved
between SGLang versions is named newest first, ``"new.py|old.py"``; the first one
present is checked. A read name such as "replication|layout" accepts either field.
The checks are static (AST and tokens): no package module is
imported. Run them against every supported SGLang and nightly against main. The
GLM-5.3 package reads SGLang's megakernel DSA binding view, and its draft spec
SGLang's draft capture check marker; against an SGLang without them, those reads and
pins are skipped, naming each.
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


def _files(pattern: str) -> list:
    """The files a ``SGLANG_READS`` key names: its first alternative that exists."""
    for alternative in pattern.split("|"):
        files = sorted(ROOT.glob(alternative))
        if files:
            return files
    return []


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
    missing, unchecked = [], []
    for pattern, names in _declared(path).items():
        files = _files(pattern)
        assert files, f"{_id(path)}: no file matches {pattern}"
        provided = frozenset().union(*map(_provided, files))
        optional = []
        if files == _files(BINDING) and not _has_dsa_view():
            optional = [n for n in names if n.split("(", 1)[0] in DSA_VIEW_NAMES]
            reason = NO_DSA_VIEW
        if files == _files(DRAFT_CHECK) and not _has_draft_check():
            optional = [n for n in names if n in DRAFT_CHECK_NAMES]
            reason = NO_DRAFT_CHECK
        if optional:
            unchecked.append(f"{reason}: {_id(path)} reads {sorted(optional)}")
            names = [n for n in names if n not in optional]
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
        "block_table",
        "cache",
        "capacity",
        "conv",
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
    }
    for path in READERS:
        names = {
            n.split("(", 1)[0]
            for pattern, declared in _declared(path).items()
            if _files(pattern) == _files(BINDING)
            for n in declared
        }
        untolerated = names - callables - data - _binding_functions()
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

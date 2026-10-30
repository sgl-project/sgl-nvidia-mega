# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Exercise the actual import guards without importing CUDA/runtime packages."""

from __future__ import annotations

import ast
import builtins
import unittest
from pathlib import Path
from types import SimpleNamespace

KIMI = Path(__file__).resolve().parents[2] / "sgl_nvidia_mega/kimi_k3"
SPEC = "sglang.kernels.plugin_kernels.nvidia.spec"
BINDING = "sglang.srt.kernel_plugins.nvidia.binding"
ABSENT_NAMESPACES = {
    SPEC: (
        "sglang.kernels",
        "sglang.kernels.plugin_kernels",
        "sglang.kernels.plugin_kernels.nvidia",
        SPEC,
    ),
    BINDING: (
        "sglang.srt.kernel_plugins",
        "sglang.srt.kernel_plugins.nvidia",
        BINDING,
    ),
}


def _guards():
    guards = []
    for filename in ("spec.py", "providers.py", "executor.py"):
        tree = ast.parse((KIMI / filename).read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Try)
                and isinstance(node.body[0], ast.ImportFrom)
                and node.body[0].module in ABSENT_NAMESPACES
            ):
                guards.append((filename, node))
    return guards


def _execute(guard, *, new_error=None, old_error=None, missing_symbol=False):
    newest = guard.body[0]
    names = [alias.name for alias in newest.names]
    new = SimpleNamespace(**{name: object() for name in names})
    old = SimpleNamespace(**{name: object() for name in names})
    if missing_symbol:
        delattr(new, names[0])
    calls = []

    def importer(name, globals=None, locals=None, fromlist=(), level=0):
        calls.append(name)
        if name == newest.module:
            if new_error is not None:
                raise new_error
            return new
        if name in ("sglang.srt.megakernels.spec", "sglang.srt.megakernels.binding"):
            if old_error is not None:
                raise old_error
            return old
        raise AssertionError(f"Unexpected import: {name}")

    namespace = {"__builtins__": {**vars(builtins), "__import__": importer}}
    code = compile(ast.Module(body=[guard], type_ignores=[]), "import-guard", "exec")
    # Execute only our repository's import statement and its fallback handler.
    return lambda: exec(code, namespace), namespace, new, old, calls  # noqa: S102


class TestKimiK3ImportCompatibility(unittest.TestCase):
    def test_all_four_guards_prefer_new_types_when_both_paths_exist(self):
        guards = _guards()
        self.assertEqual(len(guards), 4)  # spec, its annotation, pool and step
        for filename, guard in guards:
            with self.subTest(file=filename, module=guard.body[0].module):
                run, namespace, new, _, calls = _execute(guard)
                run()
                self.assertEqual(calls, [guard.body[0].module])
                for alias in guard.body[0].names:
                    self.assertIs(namespace[alias.name], getattr(new, alias.name))

    def test_each_absent_new_namespace_uses_old_types(self):
        for filename, guard in _guards():
            for absent in ABSENT_NAMESPACES[guard.body[0].module]:
                with self.subTest(file=filename, absent=absent):
                    run, namespace, _, old, calls = _execute(
                        guard, new_error=ModuleNotFoundError(name=absent)
                    )
                    run()
                    self.assertEqual(len(calls), 2)
                    for alias in guard.body[0].names:
                        self.assertIs(namespace[alias.name], getattr(old, alias.name))

    def test_missing_new_symbol_does_not_fall_back(self):
        for filename, guard in _guards():
            with self.subTest(file=filename, module=guard.body[0].module):
                run, _, _, _, calls = _execute(guard, missing_symbol=True)
                with self.assertRaises(ImportError):
                    run()
                self.assertEqual(calls, [guard.body[0].module])

    def test_internal_missing_dependency_does_not_fall_back(self):
        for filename, guard in _guards():
            for absent in (
                None,
                "torch",
                "sglang",
                "sglang.srt.runtime_context",
                SPEC + ".missing",
            ):
                with self.subTest(file=filename, absent=absent):
                    error = ModuleNotFoundError(name=absent)
                    run, _, _, _, calls = _execute(guard, new_error=error)
                    with self.assertRaises(ModuleNotFoundError) as caught:
                        run()
                    self.assertIs(caught.exception, error)
                    self.assertEqual(calls, [guard.body[0].module])

    def test_other_new_import_failure_does_not_fall_back(self):
        for filename, guard in _guards():
            with self.subTest(file=filename, module=guard.body[0].module):
                error = ImportError("incompatible native ABI")
                run, _, _, _, calls = _execute(guard, new_error=error)
                with self.assertRaises(ImportError) as caught:
                    run()
                self.assertIs(caught.exception, error)
                self.assertEqual(calls, [guard.body[0].module])

    def test_failure_of_old_fallback_propagates(self):
        for filename, guard in _guards():
            with self.subTest(file=filename, module=guard.body[0].module):
                error = ModuleNotFoundError(name="old_dependency")
                run, _, _, _, calls = _execute(
                    guard,
                    new_error=ModuleNotFoundError(name=guard.body[0].module),
                    old_error=error,
                )
                with self.assertRaises(ModuleNotFoundError) as caught:
                    run()
                self.assertIs(caught.exception, error)
                self.assertEqual(len(calls), 2)

    def test_source_contracts_name_both_layouts_newest_first(self):
        for filename in ("spec.py", "providers.py", "executor.py"):
            tree = ast.parse((KIMI / filename).read_text())
            declaration = next(
                node.value
                for node in tree.body
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id == "SGLANG_READS"
                    for t in node.targets
                )
            )
            reads = ast.literal_eval(declaration)
            for _, guard in (item for item in _guards() if item[0] == filename):
                newest = (
                    guard.body[0].module.removeprefix("sglang.").replace(".", "/")
                    + ".py"
                )
                oldest = "srt/megakernels/" + newest.rsplit("/", 1)[-1]
                self.assertIn(newest + "|" + oldest, reads)
                for alias in guard.body[0].names:
                    self.assertIn(alias.name, reads[newest + "|" + oldest])


if __name__ == "__main__":
    unittest.main()

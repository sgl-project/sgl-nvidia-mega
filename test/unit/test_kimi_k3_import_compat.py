# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Exercise the SGLang import fallback without loading CUDA/runtime packages."""

from __future__ import annotations

import ast
import builtins
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from sgl_nvidia_mega import _sglang_spec

KIMI = Path(__file__).resolve().parents[2] / "sgl_nvidia_mega/kimi_k3"
BINDING = "sglang.srt.kernel_plugins.nvidia.binding"
ABSENT_NAMESPACES = (
    "sglang.srt.kernel_plugins",
    "sglang.srt.kernel_plugins.nvidia",
    BINDING,
)


def _guards():
    guards = []
    for filename in ("spec.py", "providers.py", "executor.py"):
        tree = ast.parse((KIMI / filename).read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Try)
                and isinstance(node.body[0], ast.ImportFrom)
                and node.body[0].module == BINDING
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
        if name == "sglang.srt.megakernels.binding":
            if old_error is not None:
                raise old_error
            return old
        raise AssertionError(f"Unexpected import: {name}")

    namespace = {"__builtins__": {**vars(builtins), "__import__": importer}}
    code = compile(ast.Module(body=[guard], type_ignores=[]), "import-guard", "exec")
    # Execute only our repository's import statement and its fallback handler.
    return lambda: exec(code, namespace), namespace, new, old, calls  # noqa: S102


class TestKimiK3ImportCompatibility(unittest.TestCase):
    def test_binding_guards_prefer_new_types_when_both_paths_exist(self):
        guards = _guards()
        # Spec annotation, capture-width agreement, pool and step.
        self.assertEqual(len(guards), 4)
        for filename, guard in guards:
            with self.subTest(file=filename, module=guard.body[0].module):
                run, namespace, new, _, calls = _execute(guard)
                run()
                self.assertEqual(calls, [guard.body[0].module])
                for alias in guard.body[0].names:
                    self.assertIs(namespace[alias.name], getattr(new, alias.name))

    def test_each_absent_new_namespace_uses_old_types(self):
        for filename, guard in _guards():
            for absent in ABSENT_NAMESPACES:
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
                BINDING + ".missing",
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


class TestSharedSpecImportCompatibility(unittest.TestCase):
    def setUp(self):
        _sglang_spec.load_spec.cache_clear()
        self.addCleanup(_sglang_spec.load_spec.cache_clear)

    def test_prefers_shared_spec_and_forwards_the_original_type(self):
        module = ModuleType("shared_spec")
        step = object()
        module.Step = step
        with patch.object(_sglang_spec, "import_module", return_value=module) as load:
            self.assertIs(_sglang_spec.Step, step)
            self.assertIs(_sglang_spec.load_spec(), module)
        load.assert_called_once_with("sglang.kernels.plugin_kernels.megakernel.spec")

    def test_falls_back_when_the_spec_namespace_is_absent(self):
        shared, old = _sglang_spec._SPEC_MODULES
        for installed in (old, _sglang_spec._LEGACY_SPEC):
            module = ModuleType(installed)

            def load(name):
                if name == installed:
                    return module
                raise ModuleNotFoundError(name=name.rsplit(".", 1)[0])

            with (
                self.subTest(installed=installed),
                patch.object(
                    _sglang_spec, "import_module", side_effect=load
                ) as importer,
            ):
                self.assertIs(_sglang_spec.load_spec(), module)
                self.assertEqual(importer.call_args_list[0].args, (shared,))
            _sglang_spec.load_spec.cache_clear()

    def test_broken_shared_module_fails_without_trying_legacy(self):
        broken = ModuleNotFoundError(name="torch")
        with patch.object(_sglang_spec, "import_module", side_effect=broken) as load:
            with self.assertRaises(ModuleNotFoundError) as caught:
                _sglang_spec.load_spec()
        self.assertIs(caught.exception, broken)
        load.assert_called_once_with("sglang.kernels.plugin_kernels.megakernel.spec")


if __name__ == "__main__":
    unittest.main()

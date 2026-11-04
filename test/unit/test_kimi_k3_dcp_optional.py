# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Dormant DCP import/activation boundaries, not GPU qualification."""

import builtins
import importlib.util
from unittest.mock import patch

import pytest

from sgl_nvidia_mega.kimi_k3 import dcp_execution, dcp_verify_execution

CASES = (
    (
        dcp_execution,
        "sglang.srt.kernel_plugins.nvidia._dcp_binding",
        "PreparedDcpHybrid",
    ),
    (
        dcp_verify_execution,
        "sglang.srt.kernel_plugins.nvidia._dcp_verify_binding",
        "PreparedDcpVerify",
    ),
)


@pytest.mark.parametrize("module,name,constructor", CASES)
def test_native_optional_boundary(module, name, constructor):
    """Use the real installed framework, without creating substitute modules."""
    if importlib.util.find_spec(name) is None:
        # Reject before inspecting resources or preparing any device state.
        with pytest.raises(
            RuntimeError, match="requires the SGLang DCP binding extension"
        ) as caught:
            getattr(module, constructor)(None, None)
        assert isinstance(caught.value.__cause__, ModuleNotFoundError)
        assert caught.value.__cause__.name == name
    else:
        native = __import__(name, fromlist=["*"])
        declared = next(iter(module.SGLANG_OPTIONAL_READS.values()))
        assert module._native_types() == tuple(
            getattr(native, value) for value in declared
        )


@pytest.mark.parametrize("module,name,constructor", CASES)
def test_missing_extension_is_explained_without_resource_access(
    module, name, constructor
):
    error = ModuleNotFoundError(name=name)
    with (
        patch.object(builtins, "__import__", side_effect=error),
        pytest.raises(RuntimeError, match="ordinary DCP-off") as caught,
    ):
        getattr(module, constructor)(None, None)
    assert caught.value.__cause__ is error


@pytest.mark.parametrize("module,name,constructor", CASES)
@pytest.mark.parametrize(
    "error",
    (
        ModuleNotFoundError(name="torch"),
        ModuleNotFoundError(name="sglang.srt.runtime_context"),
        ImportError("incompatible binding symbol"),
        RuntimeError("extension initialization failed"),
    ),
)
def test_broken_dependency_is_not_misclassified_as_optional(
    module, name, constructor, error
):
    with (
        patch.object(builtins, "__import__", side_effect=error),
        pytest.raises(type(error)) as caught,
    ):
        module._native_types()
    assert caught.value is error


def test_optional_contract_still_checks_present_names(tmp_path, monkeypatch):
    import test_contract as contracts

    path = tmp_path / "reader.py"
    path.write_text('SGLANG_OPTIONAL_READS = {"extension.py": ("RequiredBinding",)}\n')
    monkeypatch.setattr(contracts, "ROOT", tmp_path)
    # Missing extension is allowed, but a present incompatible one is not.
    contracts.test_sglang_provides_what_the_plugin_reads(path)
    (tmp_path / "extension.py").write_text("class WrongBinding: pass\n")
    monkeypatch.setattr(contracts, "_id", lambda path: path.name)
    with pytest.raises(AssertionError, match="RequiredBinding"):
        contracts.test_sglang_provides_what_the_plugin_reads(path)

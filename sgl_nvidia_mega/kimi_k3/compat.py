# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Fail-early ABI checks for Kimi K3's private GPU dependencies."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import inspect
import json
from functools import cache
from pathlib import Path
from typing import Any

_CUTLASS_DSL_VERSION = "4.8.0"
_FLASHINFER_DISTRIBUTIONS = (
    "flashinfer-python",
    "flashinfer-cubin",
    "flashinfer-jit-cache",
)
_SPLITK_FIELDS = ("mma_m", "mma_n", "split_k", "ab_stages")
_REMOTE_STORE_PARAMETERS = (
    "value0",
    "value1",
    "value2",
    "value3",
    "smem_ptr",
    "mbar_ptr",
    "peer_cta_rank_in_cluster",
    "loc",
    "ip",
)
_TRTLLM_MLA_PARAMETERS = frozenset(
    {
        "backend",
        "block_tables",
        "bmm1_scale",
        "bmm2_scale",
        "enable_pdl",
        "kv_cache",
        "kv_lora_rank",
        "max_seq_len",
        "multi_ctas_kv_counter_buffer",
        "out",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "query",
        "seq_lens",
        "workspace_buffer",
    }
)


def _base_version(value: str) -> str:
    return value.split("+", 1)[0]


def _require_attributes(value: Any, names: tuple[str, ...], label: str) -> None:
    missing = tuple(name for name in names if not hasattr(value, name))
    if missing:
        raise RuntimeError(f"{label} is missing required attributes: {missing}")


def _require_files(
    paths: tuple[Path, ...], label: str, *, directory: bool = False
) -> None:
    if not paths:
        raise RuntimeError(f"{label} are absent: ()")
    predicate = Path.is_dir if directory else Path.is_file
    missing = tuple(str(path) for path in paths if not predicate(path))
    if missing:
        raise RuntimeError(f"{label} are absent: {missing}")


@cache
def validate_runtime_compatibility() -> str:
    """Validate the private CuTeDSL/FlashInfer ABI before payload import or compile.

    The returned digest is stable for one process and records the structural ABI
    accepted by this build.  Caching makes repeated graph preparation idempotent.
    """

    cutlass_version = importlib.metadata.version("nvidia-cutlass-dsl")
    if _base_version(cutlass_version) != _CUTLASS_DSL_VERSION:
        raise RuntimeError(
            "Kimi K3 megakernel requires nvidia-cutlass-dsl 4.8.0, found "
            f"{cutlass_version}"
        )

    flashinfer_versions = {
        name: importlib.metadata.version(name) for name in _FLASHINFER_DISTRIBUTIONS
    }
    bases = {_base_version(value) for value in flashinfer_versions.values()}
    if len(bases) != 1:
        raise RuntimeError(
            "Kimi K3 megakernel requires a coherent FlashInfer wheel trio, found "
            f"{flashinfer_versions}"
        )

    import cutlass
    import cutlass.cute as cute
    from cutlass._mlir.dialects import llvm
    from cutlass.cute import experimental as cute_experimental
    from cutlass.cute.arch import store_async_dsmem
    from cutlass.cute.nvgpu import tcgen05
    from cutlass.cutlass_dsl import dsl_user_op

    _require_attributes(cute, ("compile", "jit", "kernel", "sym_int"), "CuTeDSL")
    _require_attributes(
        cute_experimental,
        ("allocate", "jit", "kernel", "make_tmem_layout_acc"),
        "CuTeDSL experimental API",
    )
    _require_attributes(tcgen05, ("CtaGroup",), "CuTeDSL tcgen05 API")
    if not callable(dsl_user_op) or not callable(store_async_dsmem):
        raise RuntimeError("Kimi K3 megakernel CuTeDSL decorators are unavailable")

    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla
    from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import (
        OWNER_RANK,
        SplitKTactic,
        _store_shared_remote_v4,
    )
    from flashinfer.jit import gen_trtllm_gen_fmha_module
    from flashinfer.utils import get_trtllm_gen_multi_ctas_kv_counter_bytes

    if int(OWNER_RANK) != 0:
        raise RuntimeError(f"unexpected FlashInfer split-K owner rank: {OWNER_RANK}")
    fields = tuple(field.name for field in dataclasses.fields(SplitKTactic))
    if fields != _SPLITK_FIELDS:
        raise RuntimeError(f"FlashInfer SplitKTactic fields changed: {fields}")
    remote_store = tuple(inspect.signature(_store_shared_remote_v4).parameters)
    if remote_store != _REMOTE_STORE_PARAMETERS:
        raise RuntimeError(
            f"FlashInfer private split-K store signature changed: {remote_store}"
        )
    mla_parameters = set(
        inspect.signature(trtllm_batch_decode_with_kv_cache_mla).parameters
    )
    missing_mla = tuple(sorted(_TRTLLM_MLA_PARAMETERS - mla_parameters))
    if missing_mla:
        raise RuntimeError(f"FlashInfer TRTLLM MLA signature is missing: {missing_mla}")

    import torch

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    sms = int(properties.multi_processor_count)
    counter_bytes = int(get_trtllm_gen_multi_ctas_kv_counter_bytes(64, 12, sms))
    if counter_bytes <= 0 or counter_bytes % 4:
        raise RuntimeError(
            f"invalid FlashInfer TRTLLM multi-CTA counter size: {counter_bytes}"
        )

    module_spec = gen_trtllm_gen_fmha_module()
    sources = tuple(Path(path).resolve() for path in module_spec.sources)
    artifacts = tuple(Path(item.path).resolve() for item in module_spec.aot_artifacts)
    includes = tuple(
        Path(path).resolve() for path in (module_spec.extra_include_dirs or ())
    )
    _require_files(sources, "FlashInfer TRTLLM-gen wrapper sources")
    _require_files(artifacts, "FlashInfer TRTLLM-gen AOT artifacts")
    _require_files(
        includes,
        "FlashInfer TRTLLM-gen include directories",
        directory=True,
    )

    contract = {
        "cutlass_dsl": cutlass_version,
        "cutlass_module": str(Path(cutlass.__file__).resolve()),
        "llvm_module": llvm.__name__,
        "flashinfer": flashinfer_versions,
        "splitk_fields": fields,
        "remote_store_parameters": remote_store,
        "trtllm_mla_parameters": sorted(_TRTLLM_MLA_PARAMETERS),
        "trtllm_gen_sources": [str(path) for path in sources],
        "trtllm_gen_artifacts": [str(path) for path in artifacts],
        "trtllm_gen_aot_path": str(Path(module_spec.aot_path).resolve()),
        "trtllm_gen_includes": [str(path) for path in includes],
        "trtllm_gen_name": str(module_spec.name),
        "trtllm_gen_cuda_flags": list(module_spec.extra_cuda_cflags or ()),
        "multi_cta_counter_bytes_b64_h12": counter_bytes,
    }
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()

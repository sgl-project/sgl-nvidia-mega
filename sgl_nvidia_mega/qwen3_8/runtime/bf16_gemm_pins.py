# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team): the source's
# python/sglang/srt/layers/bf16_gemm_pins.py (the file format, _parse, _load, the
# runners, _build and gemm) and its call sites: python/sglang/srt/layers/quantization/
# unquant.py (UnquantizedLinearMethod.apply into _bf16_gemm_dispatch_impl),
# python/sglang/srt/layers/logits_processor.py (LogitsProcessor._compute_lm_head) and
# python/sglang/srt/layers/hyperconnection.py (_pinned_mix_compute in GatedResidual.mix);
# with SGLang main's python/sglang/srt/models/qwen3_5.py (_forward_input_proj),
# qwen2_moe.py (Qwen2MoeMLP.forward), qwen4_exp.py (_attn_mix, _ffn_mix, _MixStreams),
# logits_processor.py and hyperconnection.py around them.
# Changed: the file comes from the environment directly, with an optional SHA-256 check.
# The source serves the pins inside SGLang's BF16 dispatch, lm_head GEMM and HC mix;
# here the executor launches those GEMMs itself (``linear``, ``in_proj``, ``mlp``,
# ``logits_processor``, ``gated_mix``) where SGLang would have run them, at TP1. Where
# no pin serves a ``linear`` or ``in_proj`` call, runtime/sm107_bf16_smallm.py's route
# may, as the source's dispatch consults those routes after its pins.
"""Pinned kernels for unquantized BF16 GEMMs at listed (M, N, K) (default off).

``SGLANG_BF16_GEMM_PIN_FILE`` names a JSON file (``.json.py`` so a source overlay can
carry it)::

    {"_metadata": {"flashinfer_version": "<flashinfer.__version__>", "cuda_version": "13.4",
                   "cublas_version": "13.8.1", "gpu": "NVIDIA VR NVL72", "sm": "10.7"},
     "bf16_gemm_pins": [
        {"n": 16480, "k": 2560, "m": 256, "kind": "cublaslt", "algo": "<128 hex>", "workspace": 0},
        {"n": 640, "k": 2560, "m": 256, "kind": "tgv", "tactic": 21},
     ]}

``SGLANG_BF16_GEMM_PIN_FILE_SHA256``, when set, is the file's SHA-256; a file with
another digest is not applied.

Each entry names, for one exact M (a CUDA-graph batch size), a kernel measured faster
than the stock dispatch and bit-equal to it:

* ``"cublaslt"``: a serialized ``cublasLtMatmulAlgo_t`` (opaque, valid for the cuBLAS
  build the metadata names) run through FlashInfer's mm_bf16_cublaslt module (TRANSA=T,
  TRANSB=N, FP32 compute, BF16 output, beta 0). A pin keeps stock's K reduction order.
* ``"tgv"``: SGLang's CuTe DSL TGV kernel at a fixed tactic (it never splits K).

The file applies only under its exact ``_metadata``
(``flashinfer_tactic_pins.tuning_file_applies``). Before a key serves, its first eager
call checks it once (seeded random activations, the layer's weight) against the stock
dispatch (``stock``): a pin that fails to launch or differs in a single bit is dropped
for good and the stock dispatch kept. A first call under CUDA graph capture cannot
check, so that call keeps the stock dispatch. Any other shape or M, a bias, a
non-contiguous or non-BF16 operand, batch-invariant mode or torch.compile tracing takes
the stock dispatch.
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Callable, Optional

import torch

from sgl_nvidia_mega.qwen3_8.runtime import flashinfer_tactic_pins, sm107_bf16_smallm

logger = logging.getLogger(__name__)

SGLANG_READS = {
    "srt/batch_invariant_ops/__init__.py": ("is_batch_invariant_mode_enabled",),
    "srt/layers/quantization/unquant.py": (
        "UnquantizedLinearMethod",
        "get_bf16_gemm_backend",
        "is_cutedsl",
        "is_gemv",
        "_bf16_gemm_dispatch_impl",
    ),
    "kernels/ops/gemm/cutedsl_bf16_gemm.py": (
        "cutedsl_bf16_gemm",
        "_resolve_tactic",
        "_run_tgv",
    ),
    "srt/models/qwen3_5.py": (
        "_forward_input_proj",
        "_fused_in_proj_weight",
        "_fused_in_proj_qkvz_width",
    ),
    "srt/runtime_context.py": ("get_parallel",),
    "srt/models/qwen2_moe.py": (
        "can_run_silu_fp4_quant_fusion",
        "silu_and_mul_fp4_quant",
        "gate_up_proj",
        "down_proj",
        "act_fn",
    ),
    "srt/layers/logits_processor.py": (
        "LogitsProcessor",
        "_compute_lm_head",
        "should_apply_lm_head_quant_method",
        "use_intel_amx_backend",
        "_aiter_lm_head_ok",
        "use_fp32_lm_head",
        "use_fp32_lm_head_output",
        "rl_on_policy_target",
    ),
    "srt/layers/hyperconnection.py": (
        "flydsl_hc_mix_supported",
        "fused_hc_mix",
        "fused_hc_mix_supported",
        "_flydsl_mix_ok",
        "_jit_mix_ok",
        "_mix_up_weight_padded",
        "_gate_stream",
        "hc_norm",
        "hc_count",
        "hidden_size",
        "params_dtype",
        "input_mix_weight_down",
        "input_mix_weight_up",
        "block_inject_weight",
        "hc_lowrank",
        "hc_per_branch_norm",
    ),
    "kernels/ops/gemm/hc_mix_splitk.py": ("hc_mix", "permute_pad_up_weight"),
    "kernels/ops/elementwise/hc_combine_decode.py": ("hc_combine_gate",),
    "srt/models/qwen4_exp.py": (
        "_add_ple_embedding",
        "_hc_norms",
        "attn_input",
        "ffn_input",
        "ple",
        "attn_hyper_connection",
        "mlp_hyper_connection",
    ),
}

# What in_proj, mlp, logits_processor, gated_mix, attn_mix, ffn_mix and MixStreams mirror;
# _pinned_mix_compute mirrors GatedResidual.__init__'s nested _mix_compute.
SGLANG_MIRRORS = {
    "srt/models/qwen3_5.py:Qwen3_5GatedDeltaNet._forward_input_proj": "5269f43d28e6f303",
    "srt/models/qwen2_moe.py:Qwen2MoeMLP.forward": "60b6d1e897d94ed9",
    "srt/layers/logits_processor.py:LogitsProcessor._compute_lm_head": "b4fe8c96270053f7",
    "srt/layers/hyperconnection.py:GatedResidual.mix": "48230e9ab57c695a",
    "srt/models/qwen4_exp.py:Qwen4ExpLayerExtensionMixin._attn_mix": "9c0349424b16d0af",
    "srt/models/qwen4_exp.py:Qwen4ExpLayerExtensionMixin._ffn_mix": "cc60f61162f363c1",
    "srt/models/qwen4_exp.py:_MixStreams.__call__": "e7c5d6730d8ec20b",
}

FILE_ENV = "SGLANG_BF16_GEMM_PIN_FILE"
SHA256_ENV = "SGLANG_BF16_GEMM_PIN_FILE_SHA256"
_SECTION = "bf16_gemm_pins"
_ALGO_BYTES = 64
_KINDS = ("cublaslt", "tgv")
_CHECK_SEED = 0x4442
# _forward_input_proj takes the packed GEMM up to this many rows
_PACKED_IN_PROJ_MAX_ROWS = 1024

_state: Optional[dict] = None  # None: not read yet; {} when off
_runners: dict = {}  # (device, m, n, k) -> callable(x, w) or None (stock)
_lt_module = None
_workspaces: dict = {}
# set by build: the executor's projection calls take the pins (TP1)
_routed = False


def _parse(document: dict, path: str) -> dict:
    section = document.get(_SECTION)
    if not isinstance(section, list):
        raise ValueError(f"{path}: needs a list {_SECTION!r}")
    entries = {}
    for i, entry in enumerate(section):
        where = f"{path}: {_SECTION}[{i}]"
        try:
            m, n, k = int(entry["m"]), int(entry["n"]), int(entry["k"])
            kind = entry["kind"]
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"{where} needs integer m, n, k and a kind") from None
        if kind not in _KINDS or min(m, n, k) <= 0:
            raise ValueError(f"{where}: kind must be one of {_KINDS}, m, n, k positive")
        if kind == "cublaslt":
            algo = bytes.fromhex(entry["algo"])
            if len(algo) != _ALGO_BYTES:
                raise ValueError(f"{where}: algo must be {_ALGO_BYTES} bytes")
            spec = ("cublaslt", algo, int(entry.get("workspace", 32 << 20)))
        else:
            spec = ("tgv", int(entry["tactic"]))
        if (m, n, k) in entries:
            raise ValueError(f"{where}: duplicate key {(m, n, k)}")
        entries[(m, n, k)] = spec
    return entries


def _load() -> dict:
    global _state
    if _state is not None:
        return _state
    _state = {}
    path = os.environ.get(FILE_ENV)
    if not path:
        return _state
    entries = _parse(flashinfer_tactic_pins.read_tuning_file(path), path)
    reason = flashinfer_tactic_pins.digest_reason(path, SHA256_ENV)
    if reason:
        logger.warning("BF16 GEMM pins off: %s", reason)
        return _state
    if not flashinfer_tactic_pins.tuning_file_applies(path):
        logger.warning("BF16 GEMM pins off: %s is for another stack (see above)", path)
        return _state
    _state = {
        "path": path,
        "entries": entries,
        "shapes": {(n, k) for (_, n, k) in entries},
    }
    logger.info(
        "BF16 GEMM pins on: %d keys over %d shapes from %s",
        len(entries),
        len(_state["shapes"]),
        os.path.basename(path),
    )
    return _state


def enabled() -> bool:
    return bool(_load())


def has_shape(n: int, k: int) -> bool:
    state = _load()
    return bool(state) and (n, k) in state["shapes"]


def _cublaslt_runner(device: torch.device, n: int, k: int, algo: bytes, workspace: int):
    global _lt_module
    if _lt_module is None:
        from flashinfer.jit.gemm import gen_mm_bf16_cublaslt_module

        _lt_module = gen_mm_bf16_cublaslt_module().build_and_load()
    module = _lt_module
    key = (device.index, n, k)
    ws = _workspaces.get(key)
    if ws is None or ws.numel() < max(workspace, 1):
        # One per shape: two GEMMs of one shape never run concurrently on two streams here.
        ws = torch.empty(max(workspace, 1), dtype=torch.uint8, device=device)
        _workspaces[key] = ws
    algo_buf = torch.frombuffer(bytearray(algo), dtype=torch.uint8).clone()

    def run(x: torch.Tensor, weight: torch.Tensor, out=None) -> torch.Tensor:
        if out is None:
            out = torch.empty(
                (x.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=x.device
            )
        with torch.cuda.device(x.device):
            handle = torch.cuda.current_blas_handle()
        module.mm_bf16_cublaslt_run_with_algo(
            x, weight, None, out, ws, handle, algo_buf, 0
        )
        return out

    return run


def _tgv_runner(tactic: int):
    from sglang.kernels.ops.gemm import cutedsl_bf16_gemm

    cutedsl_bf16_gemm._resolve_tactic(tactic)  # range check

    def run(x: torch.Tensor, weight: torch.Tensor, out=None) -> torch.Tensor:
        if out is None:
            out = torch.empty(
                (x.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=x.device
            )
        cutedsl_bf16_gemm._run_tgv(x, weight.t(), None, out, True, tactic)
        return out

    return run


def _build(key, spec, weight: torch.Tensor, stock: Callable) -> Optional[Callable]:
    """The pinned runner for ``key`` if it launches and matches ``stock`` bit for bit, else None (for good)."""
    m, n, k = key[1:]
    try:
        if spec[0] == "cublaslt":
            run = _cublaslt_runner(weight.device, n, k, spec[1], spec[2])
        else:
            run = _tgv_runner(spec[1])
        gen = torch.Generator(device=weight.device).manual_seed(_CHECK_SEED + m)
        x = torch.randn((m, k), device=weight.device, generator=gen).to(torch.bfloat16)
        got, ref = run(x, weight), stock(x, weight)
        same = torch.equal(got.view(torch.int16), ref.view(torch.int16))
    except Exception as e:  # noqa: BLE001 - no module, bad algorithm or tactic: stock for good
        logger.warning(
            "BF16 GEMM pin M=%d N=%d K=%d %s refused, stock kept: %r",
            m,
            n,
            k,
            spec[0],
            e,
        )
        return None
    if not same:
        logger.warning(
            "BF16 GEMM pin M=%d N=%d K=%d %s differs from stock, stock kept",
            m,
            n,
            k,
            spec[0],
        )
        return None
    logger.info(
        "BF16 GEMM pin engaged: M=%d N=%d K=%d %s%s",
        m,
        n,
        k,
        spec[0],
        f" tactic {spec[1]}" if spec[0] == "tgv" else "",
    )
    return run


def gemm(
    x: torch.Tensor, weight: torch.Tensor, stock: Callable, out=None
) -> Optional[torch.Tensor]:
    """``x @ weight.T`` on the pinned kernel for this (M, N, K), or None to run the stock dispatch.

    ``stock(x, weight)`` is the dispatch without pins; it is called only once per key, by the check.
    ``out``: a contiguous [M, N] BF16 buffer the kernel writes, as SGLang's lm_head does."""
    from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled

    state = _load()
    if not state or x.ndim != 2 or weight.ndim != 2:
        return None
    # CuTe's DLPack import refuses tensors that require grad (e.g. the HC mix weights)
    x, weight = x.detach(), weight.detach()
    n, k = weight.shape
    if (n, k) not in state["shapes"]:
        return None
    m = x.shape[0]
    spec = state["entries"].get((m, n, k))
    if (
        spec is None
        or x.shape[1] != k
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or (
            out is not None
            and (
                out.dtype != torch.bfloat16
                or tuple(out.shape) != (m, n)
                or not out.is_contiguous()
            )
        )
        or not x.is_cuda
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or torch.compiler.is_compiling()
        or is_batch_invariant_mode_enabled()
    ):
        return None
    key = (x.device.index, m, n, k)
    if key in _runners:
        run = _runners[key]
    elif torch.cuda.is_current_stream_capturing():
        logger.warning(
            "BF16 GEMM pin M=%d N=%d K=%d first seen under graph capture: stock in this graph",
            m,
            n,
            k,
        )
        return None
    else:
        run = _runners[key] = _build(key, spec, weight, stock)
    return None if run is None else run(x, weight, out)


# -- the executor's projection calls -------------------------------------------------


def _stock(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl

    return _bf16_gemm_dispatch_impl(x, weight, None)


def _reaches_dispatch(module, x: torch.Tensor) -> bool:
    """Whether ``UnquantizedLinearMethod.apply`` hands this call to
    ``_bf16_gemm_dispatch_impl`` without a bias (its ROCm-only branches before that
    never run on CUDA)."""
    from sglang.srt.layers.quantization.unquant import (
        UnquantizedLinearMethod,
        get_bf16_gemm_backend,
    )

    backend = get_bf16_gemm_backend()
    return (
        type(module.quant_method) is UnquantizedLinearMethod
        and module.bias is None
        and (backend.is_cutedsl() or backend.is_gemv())
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and module.weight.dtype == torch.bfloat16
        and not module.weight.requires_grad
    )


def routed() -> bool:
    """Whether the executor's projection calls go through ``linear`` / ``in_proj``: pins
    or the SM107 small-M routes on."""
    return _routed or sm107_bf16_smallm.served()


def enable_routes() -> None:
    """Build agreed: the executor's projection calls take the pins."""
    global _routed
    _routed = True


def _dispatch(x: torch.Tensor, weight: torch.Tensor) -> Optional[torch.Tensor]:
    """The source's ``_bf16_gemm_dispatch_impl`` order: a pin, then an SM107 small-M
    route; None leaves the call to SGLang's dispatch."""
    out = gemm(x, weight, _stock) if _routed else None
    if out is None and sm107_bf16_smallm.served():
        out = sm107_bf16_smallm.gemm(x, weight)
    return out


def linear(module, x: torch.Tensor) -> torch.Tensor:
    """``module(x)[0]`` for an executor's TP1 projection call, its GEMM on the pinned
    kernel or the SM107 small-M route where one serves this (M, N, K) and SGLang's
    linear method would have run its BF16 dispatch."""
    if routed() and _reaches_dispatch(module, x):
        out = _dispatch(x, module.weight)
        if out is not None:
            return out
    return module(x)[0]


def in_proj(gdn, x: torch.Tensor):
    """``gdn._forward_input_proj(x)``, its packed (qkvz | ba) GEMM on the pinned kernel
    or the SM107 small-M route where one serves (the CUDA branch that runs it through
    ``bf16_gemm_dispatch``)."""
    if (
        routed()
        and gdn._fused_in_proj_weight is not None
        and x.dtype == torch.bfloat16
        and x.shape[0] <= _PACKED_IN_PROJ_MAX_ROWS
    ):
        out = _dispatch(x, gdn._fused_in_proj_weight)
        if out is not None:
            split = gdn._fused_in_proj_qkvz_width
            return out[:, :split], out[:, split:]
    return gdn._forward_input_proj(x)


def build(model, call_sites: dict) -> str:
    """'' when the file's pins reach at least one of the executor's projection calls
    (``call_sites``: (N, K) -> name), else why not (a malformed file raises). Logs the
    file's shapes no executor call takes."""
    from sglang.srt.runtime_context import get_parallel

    if not os.environ.get(FILE_ENV):
        return f"{FILE_ENV} unset"
    state = _load()
    if not state:
        return f"{os.environ[FILE_ENV]} does not apply here (see the warning above)"
    if get_parallel().tp_size != 1:
        return "TP1 only: a pinned GEMM replaces the whole projection"
    served = sorted(call_sites[s] for s in state["shapes"] if s in call_sites)
    stranded = sorted(state["shapes"] - set(call_sites))
    if stranded:
        logger.warning(
            "BF16 GEMM pins: no executor call takes the shapes (N, K) %s; SGLang's "
            "modules launch them on its own dispatch",
            stranded,
        )
    if not served:
        return f"no executor projection has a pinned shape (file: {sorted(state['shapes'])})"
    logger.info(
        "BF16 GEMM pins: %s (SHA-256 %s) serve the executor's %s, each key checked "
        "bit for bit against SGLang's dispatch at its first eager call",
        state["path"],
        flashinfer_tactic_pins.file_digest(state["path"])[:16],
        ", ".join(served),
    )
    return ""


# -- the source's other pin sites, which the executor launches itself ---------------


def mlp(module, x: torch.Tensor) -> torch.Tensor:
    """``Qwen2MoeMLP.forward`` (a shared expert), its projections through ``linear``."""
    from sglang.srt.models.qwen2_moe import (
        can_run_silu_fp4_quant_fusion,
        silu_and_mul_fp4_quant,
    )

    gate_up = linear(module.gate_up_proj, x)
    if can_run_silu_fp4_quant_fusion(module, gate_up):
        x, _ = module.down_proj(silu_and_mul_fp4_quant(module, gate_up))
        return x
    return linear(module.down_proj, module.act_fn(gate_up))


def _lm_head_matmul(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.matmul(x, weight.T)


def _pinned_lm_head(processor, hidden_states, lm_head, out) -> Optional[torch.Tensor]:
    """The logits of ``LogitsProcessor._compute_lm_head``'s plain BF16 branch on a pin,
    or None (the branch does not apply or no pin serves)."""
    from sglang.srt.layers.logits_processor import (
        _aiter_lm_head_ok,
        should_apply_lm_head_quant_method,
        use_intel_amx_backend,
    )

    weight = getattr(lm_head, "weight", None)
    if not (
        weight is not None
        and not (hasattr(lm_head, "set_lora") and hasattr(lm_head, "apply_lora"))
        and not should_apply_lm_head_quant_method(
            lm_head, getattr(lm_head, "quant_method", None)
        )
        and hidden_states.is_cuda
        and hidden_states.dtype == weight.dtype == torch.bfloat16
        and not (processor.use_fp32_lm_head or processor.use_fp32_lm_head_output)
        and not use_intel_amx_backend(lm_head)
        and processor.rl_on_policy_target is None
        and not _aiter_lm_head_ok(hidden_states, weight)
    ):
        return None
    if out is not None and (
        out.shape != (hidden_states.shape[0], weight.shape[0])
        or out.dtype != weight.dtype
    ):
        out = None
    return gemm(hidden_states, weight, _lm_head_matmul, out=out)


@functools.lru_cache(maxsize=None)
def _logits_view_class(cls):
    def _compute_lm_head(
        self, hidden_states, lm_head, embedding_bias=None, *, out=None
    ):
        logits = _pinned_lm_head(self, hidden_states, lm_head, out)
        if logits is not None:
            return logits
        return cls._compute_lm_head(
            self, hidden_states, lm_head, embedding_bias, out=out
        )

    return type(cls.__name__, (cls,), {"_compute_lm_head": _compute_lm_head})


_logits_views: dict = {}  # id(LogitsProcessor) -> its view


def logits_processor(processor):
    """``processor``, or with the pins routed a view of it whose ``_compute_lm_head``
    takes a pin for its plain BF16 GEMM: the view shares the module's state
    (its ``__dict__``), so the gather, softcap and hidden-states hand-off stay SGLang's."""
    if not _routed:
        return processor
    from sglang.srt.layers.logits_processor import LogitsProcessor

    cls = type(processor)
    if cls._compute_lm_head is not LogitsProcessor._compute_lm_head:
        return processor
    view = _logits_views.get(id(processor))
    if view is None:
        view = object.__new__(_logits_view_class(cls))
        object.__setattr__(view, "__dict__", processor.__dict__)
        _logits_views[id(processor)] = view
    return view


@functools.lru_cache(maxsize=None)
def _pinned_mix_compute():
    """torch.compile of GatedResidual's _mix_compute with its two GEMMs on a custom op
    that serves the pins (the source's bf16_gemm_dispatch); the pointwise ops between
    the GEMMs are _mix_compute's."""
    import torch.nn.functional as F

    @torch.library.custom_op(
        "sgl_nvidia_mega::qwen3_8_pinned_bf16_gemm",
        mutates_args=(),
        schema="(Tensor x, Tensor weight) -> Tensor",
    )
    def pinned(x, weight):
        out = gemm(x, weight, _stock)
        return _stock(x, weight) if out is None else out

    @pinned.register_fake
    def _(x, weight):
        return x.new_empty((*x.shape[:-1], weight.shape[0]))

    def mix_compute(
        hyper_input_normed: torch.Tensor,
        input_mix_weight_down: torch.Tensor,
        input_mix_weight_up: torch.Tensor,
        hc: int,
        hs: int,
    ) -> torch.Tensor:
        input_mix_weight = F.silu(
            pinned(hyper_input_normed, input_mix_weight_down) / hc
        )
        input_mix_weight = pinned(input_mix_weight, input_mix_weight_up)
        input_mix_weight = torch.sigmoid(input_mix_weight)
        input_mix_weight = input_mix_weight.unflatten(-1, (hc, hs))
        output = (input_mix_weight * hyper_input_normed.unflatten(-1, (hc, hs))).mean(
            dim=-2
        )
        return output

    return torch.compile(mix_compute)


def _mix_gemms_pinned(gr) -> bool:
    lowrank, width = gr.config.hc_lowrank, gr.hc_count * gr.hidden_size
    return has_shape(lowrank, width) or has_shape(width, lowrank)


def gated_mix(gr, hyper_input: torch.Tensor, normalized_input=None):
    """``gr.mix(hyper_input, normalized_input=...)`` (an HC read, ``GatedResidual``);
    with the pins routed and gr's mix shapes in the table, the source's pinned
    _mix_compute where SGLang's mix runs _mix_compute."""
    from sgl_nvidia_mega.qwen3_8.runtime import hc_mix_tuned

    # the source's row-aware HC mix tactics serve its split-K mix rows (<= 32; SGLang's <= 24)
    tuned = (
        hyper_input.is_cuda
        and hyper_input.dtype in (torch.bfloat16, torch.float16)
        and 0 < hyper_input.shape[0] <= hc_mix_tuned.MAX_ROWS
        and gr._jit_mix_ok
        and hc_mix_tuned.enabled()
    )
    if not (tuned or (_routed and _mix_gemms_pinned(gr))):
        return gr.mix(hyper_input, normalized_input=normalized_input)
    from sglang.srt.layers.hyperconnection import (
        flydsl_hc_mix_supported,
        fused_hc_mix,
        fused_hc_mix_supported,
    )

    assert hyper_input.shape[-1] == gr.hc_count * gr.hidden_size
    if hyper_input.shape[0] == 0:
        return gr.mix(hyper_input, normalized_input=normalized_input)
    if normalized_input is not None:
        assert gr.config.hc_per_branch_norm
        assert normalized_input.shape == hyper_input.shape
        hyper_input_normed = normalized_input
    elif gr.config.hc_per_branch_norm:
        hyper_input_normed = gr.hc_norm(hyper_input)
    else:
        hyper_input_normed = gr.hc_norm(
            hyper_input.unflatten(-1, (gr.hc_count, gr.hidden_size))
        ).flatten(-2)
    down, up = gr.input_mix_weight_down.weight, gr.input_mix_weight_up.weight
    if (
        gr._jit_mix_ok
        and hyper_input_normed.is_cuda
        and hyper_input_normed.dtype in (torch.bfloat16, torch.float16)
        and hyper_input_normed.shape[0] <= (hc_mix_tuned.MAX_ROWS if tuned else 24)
    ):
        from sglang.kernels.ops.gemm.hc_mix_splitk import (
            hc_mix,
            permute_pad_up_weight,
        )

        if tuned:
            hc_mix = hc_mix_tuned.hc_mix
        if gr._mix_up_weight_padded is None:
            gr._mix_up_weight_padded = permute_pad_up_weight(up, gr.hc_count)
        mixed_input = hc_mix(
            hyper_input_normed,
            down.data,
            gr._mix_up_weight_padded,
            gr.hc_count,
            gr.hidden_size,
        ).to(gr.params_dtype)
    elif flydsl_hc_mix_supported(hyper_input_normed, gr._flydsl_mix_ok()):
        return gr.mix(hyper_input, normalized_input=normalized_input)  # ROCm only
    elif fused_hc_mix_supported(hyper_input_normed, down, up):
        mixed_input = fused_hc_mix(
            hyper_input_normed, down, up, gr.hc_count, gr.hidden_size
        ).to(gr.params_dtype)
    else:
        mixed_input = _pinned_mix_compute()(
            hyper_input_normed, down, up, gr.hc_count, gr.hidden_size
        ).to(gr.params_dtype)
    partials = None
    stream = gr._gate_stream
    if (
        stream is not None
        and hyper_input.shape[0] == 1
        and hyper_input.dtype == torch.bfloat16
        and not torch.compiler.is_compiling()
    ):
        from sglang.kernels.ops.elementwise.hc_combine_decode import hc_combine_gate

        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            partials = hc_combine_gate(
                hyper_input_normed, gr.block_inject_weight.weight
            )
        hyper_input_normed.record_stream(stream)
    return mixed_input, (hyper_input, hyper_input_normed, partials)


def attn_mix(layer, residual: torch.Tensor):
    """``layer._attn_mix(residual)``, through ``gated_mix`` with the pins routed."""
    if not _routed:
        return layer._attn_mix(residual)
    if layer.ple is not None:
        residual = layer._add_ple_embedding(residual)
    normalized_input = (
        layer._hc_norms.attn_input
        if layer._hc_norms is not None and layer.ple is None
        else None
    )
    return gated_mix(layer.attn_hyper_connection, residual, normalized_input)


def ffn_mix(layer, residual: torch.Tensor):
    """``layer._ffn_mix(residual)``, through ``gated_mix`` with the pins routed."""
    if not _routed:
        return layer._ffn_mix(residual)
    normalized_input = (
        layer._hc_norms.ffn_input if layer._hc_norms is not None else None
    )
    return gated_mix(layer.mlp_hyper_connection, residual, normalized_input)


class MixStreams:
    """``_MixStreams`` (the layer stack's terminal read) through ``gated_mix``."""

    def __init__(self, hyper_connection_mixer, normalized_input=None):
        self.hyper_connection_mixer = hyper_connection_mixer
        self.normalized_input = normalized_input

    def __call__(self, hidden_states):
        return gated_mix(
            self.hyper_connection_mixer, hidden_states, self.normalized_input
        )[0]

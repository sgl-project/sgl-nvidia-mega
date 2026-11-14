# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team): the source's
# python/sglang/srt/layers/moe/prefill_autotune.py (_route_one_call,
# autotune_qwen_moe_prefill) and its model hook autotune_prefill_kernels in
# python/sglang/srt/models/qwen4_exp.py. Changed: the tuning call is slot moe_finalize's
# extend-size routed call (runtime/moe.py); the build sets the hook on the target model
# with that slot, instead of an environment switch; TP1 only, so an OOM never raises.
"""A FlashInfer tuning pass of the routed experts at prefill sizes, for slot
``moe_finalize``'s extend-size step.

SGLang's startup autotune runs a decode-shaped forward, so FlashInfer profiles
``trtllm_fp4_block_scale_moe`` only up to the decode rows; every larger prefill call
misses the tactic cache and runs the default tactic. SGLang's
``maybe_flashinfer_autotune_extend`` then calls the model's ``autotune_prefill_kernels``
at the prefill ceiling inside its autotune context; the build sets it on the target
(``install``). It makes one extend-size routed call (``_route_one_call``): the same
deferred call, prequantized as slot ``moe_routing`` quantizes while capturing, so it
stores the cache keys the served prefill calls look up. The router GEMM and the shared
expert are not called. ``SGLANG_FLASHINFER_TACTIC_PIN_FILE``'s prefill table, where it
applies, then settles every entry the pass added
(``flashinfer_tactic_pins.settle_prefill_tactics``), before any graph captures them.
"""

from __future__ import annotations

import functools
import logging
import time

import torch

from sgl_nvidia_mega.qwen3_8.runtime import flashinfer_tactic_pins, moe_routing

logger = logging.getLogger(__name__)

SGLANG_READS = {
    "srt/model_executor/runner/flashinfer_autotune.py": (
        "maybe_flashinfer_autotune_extend",
        '"autotune_prefill_kernels"',
    ),
    "srt/environ.py": ("envs", "SGLANG_FLASHINFER_AUTOTUNE_EXTEND"),
    "srt/models/qwen2_moe.py": ("gate",),
}

# The source's floor: its router-front paths serve at most 16 rows.
_MIN_TOKENS = 17


def reasons(model) -> str:
    """'' when the pass serves this target with slot moe_finalize built (TP1 ModelOpt
    NVFP4 experts on FlashInfer's TRT-LLM runner), else why not."""
    return moe_routing.layer_reasons(_block(model))


def install(model, executor) -> None:
    """Hand SGLang's extend autotune this executor's pass as the model's hook."""
    model.autotune_prefill_kernels = functools.partial(autotune, executor)


def _block(model):
    m = model.model
    return m.layers[m.start_layer].mlp


def _moe_entries(winners: dict) -> set:
    return {key for key in winners if key.custom_op == flashinfer_tactic_pins.MOE_OP}


def _route_one_call(executor, num_tokens: int, dtype: torch.dtype) -> bool:
    """The extend-size routed call on random rows and logits; returns whether its input
    was prequantized."""
    mlp = _block(executor.lm)
    options = executor.moe_routing
    prequant = options is not None and options.prequant_overlap
    device = mlp.gate.weight.device
    with torch.no_grad():
        x = torch.randn(
            (num_tokens, mlp.gate.weight.shape[1]), dtype=dtype, device=device
        )
        # the router output's shape and dtype; the tuner profiles on its own logits
        logits = torch.randn(
            (num_tokens, mlp.gate.weight.shape[0]),
            dtype=mlp.gate.weight.dtype,
            device=device,
        )
        quant = moe_routing.quantize(mlp, x) if prequant else None
        moe_routing.deferred(mlp, x, logits, quant)
        torch.cuda.current_stream().synchronize()
    return prequant


def autotune(executor, num_tokens: int, *, dtype: torch.dtype) -> int:
    """SGLang's ``autotune_prefill_kernels`` hook: the tuning pass, then the prefill
    table; 0 lets SGLang run its own extend pass."""
    from flashinfer.autotuner import AutoTuner
    from sglang.srt.environ import envs

    if num_tokens < _MIN_TOKENS:
        return 0
    # Read before any GPU work: a malformed tuning file fails the boot.
    pins = (
        flashinfer_tactic_pins.active_moe_tactic_pins() if executor.moe_pins else None
    )
    tuner = AutoTuner.get()
    before = _moe_entries(tuner._winner_cache())
    started = time.perf_counter()
    oom = False
    prequant = None
    try:
        prequant = _route_one_call(executor, num_tokens, dtype)
    except torch.OutOfMemoryError:
        oom = True
    finally:
        torch.cuda.empty_cache()  # free the dummy rows before graph capture sizes memory

    with tuner._lock:
        winners = tuner._winner_cache()
        new_keys = _moe_entries(winners) - before
        valid = {}
        if pins is not None and pins.prefill is not None:
            from flashinfer.fused_moe.backends.trtllm.sm100_runner import MoERunner

            valid = flashinfer_tactic_pins.valid_tactics_by_bucket(
                MoERunner.valid_tactics_dict, pins.shape
            )
        rows = flashinfer_tactic_pins.settle_prefill_tactics(
            winners, new_keys, pins=pins, valid=valid
        )
    for bucket, final, tuned, source in rows:
        logger.debug(
            "MoE prefill autotune: bucket=%d tactic=%s tuned=%s source=%s",
            bucket,
            final,
            tuned,
            source,
        )
    from_table = sum(1 for row in rows if row[3].startswith("pin"))
    if pins is not None and pins.prefill is not None and rows and not from_table:
        logger.warning(
            "The tuning file's prefill MoE table (shape %s) matched no tuned entry; "
            "the tuner's picks are kept",
            pins.shape,
        )
    logger.info(
        "MoE prefill autotune: %d tokens, %d buckets tuned this boot, %d set from the "
        "tuning file, %d on the default tactic; prequant=%s, %.1f s%s",
        num_tokens,
        len(rows),
        from_table,
        sum(1 for row in rows if row[1] == flashinfer_tactic_pins.DEFAULT_TACTIC),
        prequant,
        time.perf_counter() - started,
        ", skipped after OOM" if oom else "",
    )
    if oom:
        return 0
    if envs.SGLANG_FLASHINFER_AUTOTUNE_EXTEND.get():
        return 0  # an explicitly requested whole-model extend pass still runs
    return max(1, len(rows))

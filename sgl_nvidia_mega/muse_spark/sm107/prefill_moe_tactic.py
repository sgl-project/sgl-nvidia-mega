# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``prefill_moe_tactic``: pin the FlashInfer trtllm-gen routed-MoE tactic for prefill-sized token counts.

SGLang's Muse Spark expert layers call ``flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe`` (NVFP4 expert
layers) and ``trtllm_bf16_routed_moe`` (the BF16 expert layers), on both the replicated-token path and the
sequence-parallel prefill path, with ``tune_max_num_tokens`` set to the next power of two of the call's token count. FlashInfer's autotuner answers from the tactics it profiled at startup;
the server's warmup only profiles token buckets up to the decode / verify sizes it runs, so a prefill call whose token
count lies above the largest profiled bucket gets the fallback tactic (``-1``): the launcher's heuristic tile and the
kernel list's default config. For the Muse Spark expert shapes that default is slower at prefill token counts than
the choice the autotuner itself makes for its largest profiled bucket (NVFP4: a different first-GEMM configuration;
BF16: a smaller token tile).

This component answers the autotuner query for token counts inside configured ranges with a fixed ``[tile_N, config]``
chosen by an offline sweep on routing captured from served traffic (``bench/bench_moe_tactics.py``). Only the token
tile and the GEMM configuration change; the kernels, the routing and every rounding point stay FlashInfer's, and the
sweep keeps a tactic only when its per-(token, slot) expert outputs are bitwise identical to the fallback tactic's.
A tactic is used only if FlashInfer lists it as a candidate for the call's token count; otherwise the stock choice is
kept. Token counts outside every range always keep the stock choice, so this composes with ``moe_tactic`` (decode
sizes) in either installation order: each hook passes calls it does not own to the previously installed one.

Opt in with ``MUSE_VR_PREFILL_MOE_TACTIC=1``. ``MUSE_VR_PREFILL_MOE_TACTIC_TABLE`` (NVFP4 op) and
``MUSE_VR_PREFILL_MOE_TACTIC_BF16_TABLE`` (BF16 op) replace the built-in ranges, as
``<local_experts>@<lo>-<hi>:<tile_N>/<config>`` entries separated by commas; a range covers ``lo < M <= hi``.
"""

from __future__ import annotations

import logging
import os
import threading

from sgl_nvidia_mega.muse_spark.components import bed

logger = logging.getLogger("muse_vr_plugin")

NAME = "prefill_moe_tactic"
_OP_FP4 = "flashinfer::trtllm_fp4_block_scale_moe"
_OP_BF16 = "flashinfer::trtllm_bf16_moe"

# Muse Spark routed-expert shape the ranges apply to.
TOP_K = 8
HIDDEN = 4096
INTERMEDIATE = 4096
SWIGLU = 3

# {local_experts: [(lo, hi, (tile_N, config)), ...]}, a range covers lo < M <= hi. Chosen by sweeping every
# candidate tactic on routing captured from served traffic: best (or tied) tactic at every swept size, bitwise identical
# expert outputs to the fallback. Ranges: token counts above the largest bucket the server's startup autotune profiles
# for the op, plus the one profiled bucket (768, 1024] whose cached pick measured slower than this tactic on captured
# routing; every other count keeps the autotuner's own answer.
DEFAULT_TABLE: dict[
    int, list[tuple[int, int, tuple[int, int]]]
] = {  # NVFP4 op (swept M 512..16384)
    64: [(768, 1024, (128, 28)), (8192, 65536, (128, 28))],
}
DEFAULT_TABLE_BF16: dict[
    int, list[tuple[int, int, tuple[int, int]]]
] = {  # BF16 op (swept M 640..16384)
    64: [(512, 65536, (128, 0))],
}

_lock = threading.Lock()
_state: dict = {
    "engaged": None,
    "table": {},
    "table_bf16": {},
    "orig": None,
    "require_candidate": True,
}
_candidate_cache: dict = {}


def parse_table(text: str) -> dict[int, list[tuple[int, int, tuple[int, int]]]]:
    """``"64@8192-65536:128/28"`` -> ``{64: [(8192, 65536, (128, 28))]}``; ranges of one table must not overlap."""
    table: dict[int, list[tuple[int, int, tuple[int, int]]]] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        head, tac = item.split(":")
        le, rng = head.split("@")
        lo, hi = (int(v) for v in rng.split("-"))
        tile, cfg = (int(v) for v in tac.split("/"))
        if not 0 <= lo < hi:
            raise ValueError(f"empty range {rng}")
        table.setdefault(int(le), []).append((lo, hi, (tile, cfg)))
    for rows in table.values():
        rows.sort()
        for (_, hi0, _), (lo1, _, _) in zip(rows, rows[1:]):
            if lo1 < hi0:
                raise ValueError("overlapping ranges")
    return table


def format_table(table: dict[int, list[tuple[int, int, tuple[int, int]]]]) -> str:
    return ",".join(
        f"{le}@{lo}-{hi}:{t}/{c}"
        for le in sorted(table)
        for lo, hi, (t, c) in table[le]
    )


def lookup(
    local_experts: int, num_tokens: int, op: str = _OP_FP4
) -> tuple[int, int] | None:
    table = _state["table_bf16"] if op == _OP_BF16 else _state["table"]
    for lo, hi, tac in table.get(int(local_experts), ()):
        if lo < num_tokens <= hi:
            return tac
    return None


def _is_target(runner, op: str) -> bool:
    try:
        table = _state["table_bf16"] if op == _OP_BF16 else _state["table"]
        return (
            int(runner.top_k) == TOP_K
            and int(runner.hidden_size) == HIDDEN
            and int(runner.hidden_size_output) == HIDDEN
            and int(runner.intermediate_size) == INTERMEDIATE
            and int(runner.activation_type) == SWIGLU
            and (op == _OP_BF16 or bool(runner.use_per_token_scaling))
            and int(getattr(runner, "num_fused_shared_experts", 0) or 0) == 0
            and int(runner.num_local_experts) in table
        )
    except (AttributeError, TypeError, ValueError):
        return False


def candidate_tactics(
    runner, num_tokens: int, op: str = _OP_FP4
) -> set[tuple[int, int]]:
    """FlashInfer's own candidate set for this shape (the tactics its autotuner would choose among).

    Cached per op and activation/weight dtypes: the NVFP4 and BF16 runners share one module object."""
    key = (
        op,
        id(runner.moe_op),
        str(runner.dtype_act),
        str(runner.dtype_weights),
        int(runner.num_local_experts),
        int(num_tokens),
    )
    hit = _candidate_cache.get(key)
    if hit is None:
        rows = runner.moe_op.trtllm_get_valid_moe_configs(
            runner.dtype_act,
            runner.dtype_weights,
            runner.fp8_quantization_type,
            runner.top_k,
            runner.hidden_size,
            runner.hidden_size_output,
            runner.intermediate_size,
            runner.num_local_experts,
            runner.activation_type,
            runner.use_shuffled_weight,
            runner.weight_layout,
            runner.use_per_token_scaling,
            int(num_tokens),
            False,
        )
        hit = {(int(r[0]), int(r[1])) for r in rows}
        _candidate_cache[key] = hit
    return hit


def _choose_one(self, custom_op, runners, tuning_config, inputs, **kwargs):
    orig = _state["orig"]
    if (
        custom_op not in (_OP_FP4, _OP_BF16)
        or self.is_tuning_mode
        or not runners
        or not _is_target(runners[0], custom_op)
    ):
        return orig(self, custom_op, runners, tuning_config, inputs, **kwargs)
    runner = runners[0]
    hidden_states = inputs[
        4
    ]  # MoeRunnerInputs field order: output, logits, ids, weights, hidden_states, ...
    want = (
        "torch.uint8" if custom_op == _OP_FP4 else "torch.bfloat16"
    )  # packed NVFP4 / BF16 activations
    if hidden_states is None or str(hidden_states.dtype) != want:
        return orig(self, custom_op, runners, tuning_config, inputs, **kwargs)
    m = int(hidden_states.shape[0])
    tac = lookup(runner.num_local_experts, m, custom_op)
    if tac is None:
        return orig(self, custom_op, runners, tuning_config, inputs, **kwargs)
    ok = True
    if _state["require_candidate"]:
        try:
            ok = tac in candidate_tactics(runner, m, custom_op)
        except Exception as e:  # noqa: BLE001 - any query failure keeps the stock choice
            logger.warning(
                "muse_vr_plugin: prefill_moe_tactic candidate query failed at M=%d (%s); stock kept",
                m,
                e,
            )
            ok = False
    if not ok:
        bed.count(
            NAME, f"stock_not_candidate_{'bf16' if custom_op == _OP_BF16 else 'fp4'}"
        )
        return orig(self, custom_op, runners, tuning_config, inputs, **kwargs)
    # one key per op and tactic (not per M): prefill token counts vary call to call
    kind = "bf16" if custom_op == _OP_BF16 else "fp4"
    bed.count(NAME, f"{kind}_le{runner.num_local_experts}_t{tac[0]}_c{tac[1]}")
    return runner, [int(tac[0]), int(tac[1])]


def install(
    table: dict[int, list[tuple[int, int, tuple[int, int]]]],
    require_candidate: bool = True,
    table_bf16: dict[int, list[tuple[int, int, tuple[int, int]]]] | None = None,
) -> bool:
    """Install the autotuner hook with the NVFP4 ``table`` and the BF16 ``table_bf16`` (also used by the tests)."""
    from flashinfer.autotuner import AutoTuner

    with _lock:
        _state["table"] = {int(le): sorted(rows) for le, rows in table.items()}
        _state["table_bf16"] = {
            int(le): sorted(rows) for le, rows in (table_bf16 or {}).items()
        }
        _state["require_candidate"] = bool(require_candidate)
        _candidate_cache.clear()
        if _state["orig"] is None:
            _state["orig"] = AutoTuner.choose_one
            AutoTuner.choose_one = _choose_one
    return True


def uninstall() -> None:
    """Remove the hook (only when it is still the outermost one; otherwise the table is emptied and calls pass)."""
    from flashinfer.autotuner import AutoTuner

    with _lock:
        if _state["orig"] is not None and AutoTuner.choose_one is _choose_one:
            AutoTuner.choose_one = _state["orig"]
            _state["orig"] = None
        _state["table"] = {}
        _state["table_bf16"] = {}


def engage() -> bool:
    if _state["engaged"] is not None:
        return _state["engaged"]
    if not bed.enabled(NAME):
        _state["engaged"] = False
        return False
    tables = {}
    for var, default in (
        ("MUSE_VR_PREFILL_MOE_TACTIC_TABLE", DEFAULT_TABLE),
        ("MUSE_VR_PREFILL_MOE_TACTIC_BF16_TABLE", DEFAULT_TABLE_BF16),
    ):
        env = os.environ.get(var, "").strip()
        try:
            tables[var] = parse_table(env) if env else default
        except ValueError as e:
            bed.report(NAME, False, reason=f"bad {var} ({e})")
            _state["engaged"] = False
            return False
    table, table_bf16 = (
        tables["MUSE_VR_PREFILL_MOE_TACTIC_TABLE"],
        tables["MUSE_VR_PREFILL_MOE_TACTIC_BF16_TABLE"],
    )
    if not table and not table_bf16:
        bed.report(NAME, False, reason="empty tables")
        _state["engaged"] = False
        return False
    try:
        install(table, table_bf16=table_bf16)
    except Exception as e:  # noqa: BLE001
        bed.report(NAME, False, reason=f"install failed: {e}")
        _state["engaged"] = False
        return False
    bed.report(
        NAME, True, table=format_table(table), table_bf16=format_table(table_bf16)
    )
    _state["engaged"] = True
    return True

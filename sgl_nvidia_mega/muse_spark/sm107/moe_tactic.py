# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``moe_tactic``: pin the FlashInfer trtllm-gen NVFP4 routed-MoE tactic per token count.

SGLang's Muse Spark NVFP4 expert path calls ``flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe`` with
pre-computed (packed) top-k ids. Inside, FlashInfer asks its autotuner for a ``[tile_N, config]`` tactic; when the
autotuner has no entry for the shape it returns the fallback ``-1`` and the launcher picks a heuristic tile and the
kernel list's default config. This component answers that query for the Muse Spark expert shapes with a fixed table
(token count -> ``[tile_N, config]``) chosen by an offline sweep (``bench/bench_moe_tactics.py``). The kernels, the
routing and every rounding point stay FlashInfer's; only the token tile and the GEMM config change. The sweep keeps a
tactic only if its per-(token, slot) expert outputs are bitwise identical to the default tactic's.

Opt in with ``MUSE_VR_MOE_TACTIC=1``. The table is device specific: it is read from
``data/moe_tactic/<key>.json``, where the key is a hash of the device name and compute capability, when the first
routed-MoE call of a process resolves its device; with no file for the device the hook stays installed but every call
keeps the stock choice (recorded as ``moe_tactic:table`` engaged=False). ``MUSE_VR_MOE_TACTIC_TABLE`` replaces the
file, as ``<local_experts>@<M>:<tile_N>/<config>`` entries separated by commas (e.g. ``64@8:8/12,64@16:8/12``). A token
count without an entry uses the entry of the largest smaller token count of the same table, if that tile is still among
FlashInfer's candidate tiles for the count; counts above the largest entry and all other cases keep the stock choice.
An entry is applied only when FlashInfer lists it among its candidate tactics for the call's shape
(``trtllm_get_valid_moe_configs``); otherwise the call keeps the stock choice.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from pathlib import Path

from sgl_nvidia_mega.muse_spark.components import bed

logger = logging.getLogger("muse_vr_plugin")

NAME = "moe_tactic"
_OP = "flashinfer::trtllm_fp4_block_scale_moe"

# Muse Spark routed-expert shape the table applies to.
TOP_K = 8
HIDDEN = 4096
INTERMEDIATE = 4096
SWIGLU = 3

# Per-device tables: {local_experts: {num_tokens: (tile_N, config)}} for the target-verify token counts of the decode
# CUDA graphs (8 x running requests); how each entry was chosen is recorded in the file. Every entry gave
# bitwise-identical expert outputs.
DATA_DIR = Path(__file__).resolve().parent / "data" / "moe_tactic"


def device_key(index: int) -> str:
    """File key of a device: a hash of its name and compute capability."""
    import torch

    major, minor = torch.cuda.get_device_capability(index)
    return hashlib.sha256(
        f"{torch.cuda.get_device_name(index)}|{major}.{minor}".encode()
    ).hexdigest()[:16]


def device_table(index: int) -> dict[int, dict[int, tuple[int, int]]] | None:
    """The packaged table for a device, or None."""
    try:
        data = json.loads((DATA_DIR / f"{device_key(index)}.json").read_text())
    except (OSError, ValueError):
        return None
    return {
        int(le): {int(m): (int(t), int(c)) for m, (t, c) in rows.items()}
        for le, rows in data["table"].items()
    }


_lock = threading.Lock()
# require_candidate: only apply a table entry that is in FlashInfer's own candidate set for the shape (the bench
# turns it off to sweep every tile the launcher accepts).
_state: dict = {
    "engaged": None,
    "table": {},
    "orig": None,
    "require_candidate": True,
    "per_device": False,
}
_candidate_cache: dict = {}


def parse_table(text: str) -> dict[int, dict[int, tuple[int, int]]]:
    """``"64@8:8/12,64@16:16/3"`` -> ``{64: {8: (8, 12), 16: (16, 3)}}``."""
    table: dict[int, dict[int, tuple[int, int]]] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        head, tac = item.split(":")
        le, m = head.split("@")
        tile, cfg = tac.split("/")
        table.setdefault(int(le), {})[int(m)] = (int(tile), int(cfg))
    return table


def format_table(table: dict[int, dict[int, tuple[int, int]]]) -> str:
    return ",".join(
        f"{le}@{m}:{t}/{c}"
        for le in sorted(table)
        for m, (t, c) in sorted(table[le].items())
    )


def _is_target(runner) -> bool:
    try:
        return (
            int(runner.top_k) == TOP_K
            and int(runner.hidden_size) == HIDDEN
            and int(runner.hidden_size_output) == HIDDEN
            and int(runner.intermediate_size) == INTERMEDIATE
            and int(runner.activation_type) == SWIGLU
            and bool(runner.use_per_token_scaling)
            and int(runner.num_fused_shared_experts) == 0
            and int(runner.num_local_experts) in _state["table"]
        )
    except (AttributeError, TypeError, ValueError):
        return False


def candidate_tactics(runner, num_tokens: int) -> set[tuple[int, int]]:
    """FlashInfer's own candidate set for this shape (the tactics its autotuner would choose among)."""
    key = (id(runner.moe_op), int(runner.num_local_experts), int(num_tokens))
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


def lookup(local_experts: int, num_tokens: int) -> tuple[int, int] | None:
    rows = _state["table"].get(int(local_experts))
    if not rows:
        return None
    if num_tokens in rows:
        return rows[num_tokens]
    if num_tokens > max(
        rows
    ):  # beyond the swept range (e.g. prefill chunks): keep the stock choice
        return None
    lower = [m for m in rows if m < num_tokens]
    return rows[max(lower)] if lower else None


def _resolve_device_table(inputs) -> None:
    """Load the device's packaged table at the first routed-MoE call of the process (the device is known then)."""
    with _lock:
        if not _state["per_device"]:
            return
        _state["per_device"] = False
        hs = inputs[4] if len(inputs) > 4 else None
        index = hs.device.index if hs is not None and hs.is_cuda else None
        table = device_table(index) if index is not None else None
        _state["table"] = table or {}
        _candidate_cache.clear()
    bed.report(
        NAME + ":table",
        bool(table),
        key=device_key(index) if index is not None else None,
        entries=format_table(table) if table else "",
    )


def _choose_one(self, custom_op, runners, tuning_config, inputs, **kwargs):
    orig = _state["orig"]
    if custom_op == _OP and _state["per_device"]:
        _resolve_device_table(inputs)
    if (
        custom_op != _OP
        or self.is_tuning_mode
        or not runners
        or not _is_target(runners[0])
    ):
        return orig(self, custom_op, runners, tuning_config, inputs, **kwargs)
    runner = runners[0]
    hidden_states = inputs[
        4
    ]  # MoeRunnerInputs field order: output, logits, ids, weights, hidden_states, ...
    if (
        hidden_states is None or str(hidden_states.dtype) != "torch.uint8"
    ):  # packed NVFP4 activations only
        return orig(self, custom_op, runners, tuning_config, inputs, **kwargs)
    m = int(hidden_states.shape[0])
    tac = lookup(runner.num_local_experts, m)
    if tac is None:
        bed.count(NAME, "stock_no_entry")
        return orig(self, custom_op, runners, tuning_config, inputs, **kwargs)
    ok = True
    if _state["require_candidate"]:
        try:
            ok = tac in candidate_tactics(runner, m)
        except Exception as e:  # noqa: BLE001 - any query failure keeps the stock choice
            logger.warning(
                "muse_vr_plugin: moe_tactic candidate query failed at M=%d (%s); stock tactic kept",
                m,
                e,
            )
            ok = False
    if not ok:
        bed.count(NAME, f"stock_not_candidate_m{m}")
        return orig(self, custom_op, runners, tuning_config, inputs, **kwargs)
    bed.count(NAME, f"le{runner.num_local_experts}_m{m}_t{tac[0]}_c{tac[1]}")
    return runner, [int(tac[0]), int(tac[1])]


def install(
    table: dict[int, dict[int, tuple[int, int]]] | None, require_candidate: bool = True
) -> bool:
    """Install the autotuner hook with ``table`` (also used by the bench and the tests); ``None`` = the device's
    packaged table, resolved at the first routed-MoE call."""
    from flashinfer.autotuner import AutoTuner

    with _lock:
        _state["per_device"] = table is None
        _state["table"] = (
            {} if table is None else {int(le): dict(rows) for le, rows in table.items()}
        )
        _state["require_candidate"] = bool(require_candidate)
        _candidate_cache.clear()
        if _state["orig"] is None:
            _state["orig"] = AutoTuner.choose_one
            AutoTuner.choose_one = _choose_one
    return True


def uninstall() -> None:
    from flashinfer.autotuner import AutoTuner

    with _lock:
        if _state["orig"] is not None:
            AutoTuner.choose_one = _state["orig"]
            _state["orig"] = None
        _state["table"] = {}
        _state["per_device"] = False


def engage() -> bool:
    if _state["engaged"] is not None:
        return _state["engaged"]
    if not bed.enabled(NAME):
        _state["engaged"] = False
        return False
    env = os.environ.get("MUSE_VR_MOE_TACTIC_TABLE", "").strip()
    try:
        table = parse_table(env) if env else None  # None: the device's packaged table
    except ValueError as e:
        bed.report(NAME, False, reason=f"bad MUSE_VR_MOE_TACTIC_TABLE ({e})")
        _state["engaged"] = False
        return False
    if table is not None and not table:
        bed.report(NAME, False, reason="empty table")
        _state["engaged"] = False
        return False
    try:
        install(table)
    except Exception as e:  # noqa: BLE001
        bed.report(NAME, False, reason=f"install failed: {e}")
        _state["engaged"] = False
        return False
    bed.report(
        NAME,
        True,
        table=format_table(table) if table is not None else "per-device file",
    )
    _state["engaged"] = True
    return True

# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Derived from SGLang (https://github.com/sgl-project/sglang), Apache License 2.0; see include/LICENSE.sglang.
# _attention_layers follows the attention-layer traversal of NVFP4KVCacheMethod.load_scales_from_model
# (python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py at the pinned revision).
# Modified by NVIDIA: the traversal is condensed and returns the attention layers instead of loading checkpoint scales
# into them.
"""Part ``kv_scales`` of the ``nvfp4_kv`` component: calibrated per-layer global K/V scales for the NVFP4 KV cache.

The checkpoint ships no KV scales, so SGLang quantizes every layer's K and V with global scale 1.0: the E4M3 block
scale of a 16-value block is ``amax / 6``, which is subnormal below ``amax = 6 * 2**-6`` and rounds to zero below
``6 * 2**-10``, and the dequantized operand the attention kernels feed to their E4M3 MMA (E2M1 value x block scale) is
subnormal below ``2**-6``. With ``MUSE_VR_NVFP4_KV_SCALES`` set, this part replaces the 1.0 defaults with per-layer
scales from a data file (``1`` = the packaged file for this model, or a path), after SGLang's own
``load_scales_from_model`` ran. The values are used exactly where SGLang uses checkpoint KV scales: the KV write
(``NVFP4KVCacheMethod.quantize_and_store`` / the ``fused_store`` kernel) and the attention BMM scales
(``get_bmm_scales``: ``bmm1 = q_scale * k_scale * softmax_scale``, ``bmm2 = v_scale``).

The data file holds, per global layer id, ``k_scale`` / ``v_scale`` in the kernels' convention (stored value =
x / scale; the dequantized E4M3 operand must stay <= 448) and the layer's attention window, which must match the served
model's layers. Calibration source, rule and token counts are recorded in the file.

Hybrid sliding-window pools: the sub-pools are written with a sub-pool-local layer index, and the pool looks the scale
up by that index. ``nvfp4_kv_pool`` tags each sub-pool with its local-to-global layer map and the
``_set_quantized_kv_buffer`` hook installed here derives the global id from it, so every layer is written with its own
scale.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/model_loader/utils.py": ("resolve_language_model",),
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
}


logger = logging.getLogger("muse_vr_plugin")

FORMAT = "muse_vr_nvfp4_kv_scales/1"
DEFAULT_FILE = (
    Path(__file__).resolve().parent / "data" / "nvfp4_kv_scales_muse_spark_open.json"
)
E4M3_MAX = 448.0

_state: dict = {"data": None, "path": None, "sha": None, "applied": 0}


def requested() -> str | None:
    raw = os.environ.get("MUSE_VR_NVFP4_KV_SCALES", "").strip()
    if raw.lower() in ("", "0", "false", "no", "off"):
        return None
    return str(DEFAULT_FILE) if raw.lower() in ("1", "true", "yes", "on") else raw


def load(path: str) -> dict:
    """Read and check a scale file (format, lengths, finite positive values)."""
    blob = Path(path).read_bytes()
    data = json.loads(blob)
    if data.get("format") != FORMAT:
        raise ValueError(f"{path}: format {data.get('format')!r} is not {FORMAT!r}")
    ks, vs, win = data["k_scale"], data["v_scale"], data["window"]
    if not (len(ks) == len(vs) == len(win)) or not ks:
        raise ValueError(f"{path}: k_scale / v_scale / window lengths differ")
    for s in list(ks) + list(vs):
        if not (isinstance(s, (int, float)) and 0.0 < float(s) < float("inf")):
            raise ValueError(f"{path}: scale {s!r} is not a finite positive number")
    _state.update(data=data, path=str(path), sha=hashlib.sha256(blob).hexdigest())
    return data


def _attention_layers(model_runner):
    """The attention layers SGLang's ``load_scales_from_model`` visits (same traversal)."""
    from sglang.srt.model_loader.utils import resolve_language_model

    model = getattr(model_runner, "model", model_runner)
    lm = model if hasattr(model, "layers") else resolve_language_model(model)
    out = []
    for layer in lm.layers:
        if hasattr(layer, "self_attn"):
            if hasattr(layer.self_attn, "attn"):
                out.append(layer.self_attn.attn)
            elif hasattr(layer.self_attn, "attn_mqa"):
                out.append(layer.self_attn.attn_mqa)
        elif hasattr(layer, "attn"):
            out.append(layer.attn)
        elif hasattr(layer, "attention") and hasattr(layer.attention, "attn"):
            out.append(layer.attention.attn)
    return out


def apply(qm, model_runner) -> int:
    """Overwrite ``qm``'s per-layer scales with the file's; returns the number of layers set. Refuses a model whose
    attention layers (count and windows) differ from the file's."""
    import torch

    data = _state["data"]
    layers = _attention_layers(model_runner)
    if not layers:
        raise RuntimeError("nvfp4_kv kv_scales: no attention layers found")
    n = len(data["k_scale"])
    ids = sorted(int(a.layer_id) for a in layers)
    if ids != list(range(n)):
        raise RuntimeError(
            f"nvfp4_kv kv_scales: model attention layer ids {ids[:3]}..{ids[-1]} (count {len(ids)}) "
            f"do not match the file's {n} layers"
        )
    for a in layers:
        w = int(getattr(a, "sliding_window_size", -1) or -1)
        w = w if w > 0 else -1
        if w != int(data["window"][a.layer_id]):
            raise RuntimeError(
                f"nvfp4_kv kv_scales: layer {a.layer_id} window {w} != file {data['window'][a.layer_id]}"
            )
    if len(qm.k_scales_gpu) < n:
        raise RuntimeError(
            f"nvfp4_kv kv_scales: quant method holds {len(qm.k_scales_gpu)} scales < {n}"
        )
    for i in range(n):
        if qm.k_scales_float[i] != 1.0 or qm.v_scales_float[i] != 1.0:
            raise RuntimeError(
                "nvfp4_kv kv_scales: the checkpoint already provides KV scales; refusing to override"
            )
    k = torch.tensor([float(x) for x in data["k_scale"]], dtype=torch.float32)
    v = torch.tensor([float(x) for x in data["v_scale"]], dtype=torch.float32)
    kc, vc = qm.k_scales_gpu.cpu().clone(), qm.v_scales_gpu.cpu().clone()
    kc[:n], vc[:n] = k, v
    qm.k_scales_gpu.copy_(kc)
    qm.v_scales_gpu.copy_(vc)
    for i in range(n):  # the float32 values the kernels and the write path see
        qm.k_scales_float[i] = float(k[i])
        qm.v_scales_float[i] = float(v[i])
    _state["applied"] = n
    return n


def _load_scales_around(original, self, model_runner, *args, **kwargs):
    out = original(self, model_runner, *args, **kwargs)
    if getattr(self, "name", None) == "nvfp4" and _state["data"] is not None:
        n = apply(self, model_runner)
        ks, vs = self.k_scales_float[:n], self.v_scales_float[:n]
        bed.report(
            "nvfp4_kv_scales",
            True,
            file=os.path.basename(_state["path"]),
            sha256=_state["sha"][:16],
            layers=n,
            k_min=min(ks),
            k_max=max(ks),
            v_min=min(vs),
            v_max=max(vs),
        )
    return out


def _set_quantized_kv_buffer_around(
    original, self, layer_id, global_layer_id, *args, **kwargs
):
    """``MHATokenToKVPool._set_quantized_kv_buffer``: a hybrid sub-pool is written with ``layer=None`` and its local
    layer index, so SGLang passes the local index as ``global_layer_id`` too; derive the global id from the local one."""
    ids = getattr(self, "_muse_vr_global_layer_ids", None)
    if ids is not None:
        global_layer_id = ids[layer_id - self.start_layer]
    return original(self, layer_id, global_layer_id, *args, **kwargs)


def install() -> bool:
    """Load the requested file and install both hooks; False (and nothing installed) when not requested."""
    path = requested()
    if path is None:
        return False
    load(path)
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.layers.quantization.fp4_kv_cache_quant_method.NVFP4KVCacheMethod.load_scales_from_model",
        _load_scales_around,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.mem_cache.memory_pool.MHATokenToKVPool._set_quantized_kv_buffer",
        _set_quantized_kv_buffer_around,
        HookType.AROUND,
    )
    return True


def engaged() -> bool:
    return _state["data"] is not None


def tag_sub_pools(pool) -> None:
    """Give each sub-pool of a hybrid ``SWAKVPool`` its local -> global layer id map (read by the hook above)."""
    full = [None] * pool.full_layer_nums
    swa = [None] * pool.swa_layer_nums
    for gid, (idx, is_swa) in pool.layers_mapping.items():
        (swa if is_swa else full)[idx] = int(gid)
    if any(x is None for x in full + swa):
        raise RuntimeError("nvfp4_kv kv_scales: incomplete hybrid layer mapping")
    pool.full_kv_pool._muse_vr_global_layer_ids = full
    pool.swa_kv_pool._muse_vr_global_layer_ids = swa

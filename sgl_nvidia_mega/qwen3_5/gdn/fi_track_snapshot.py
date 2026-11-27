# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import weakref
from pathlib import Path
from typing import NamedTuple, Optional

import torch

SGLANG_READS = {
    "srt/layers/attention/linear/kernels/gdn_flashinfer.py": (
        "FlashInferGDNKernel",
        "_prefill_fn",
        "_run_prefill",
    ),
}
SGLANG_MIRRORS = {
    "srt/layers/attention/linear/kernels/gdn_flashinfer.py:FlashInferGDNKernel._run_prefill": "201bcf05dbed9433",
    "srt/layers/attention/linear/kernels/gdn_flashinfer.py:maybe_build_flashinfer_checkpoint_plan": "4bdf64c6dfbebd81",
    "srt/layers/attention/linear/kernels/gdn_flashinfer.py:_get_flashinfer_gdn_kernels": "20aa6e3367376d05",
}
FLASHINFER_SOURCES = {
    "gdn_prefill.py": (
        "962d2904136a9298f6c3408e059c61c363f98fb096f775c203f9b763322aa740",
    ),
    "gdn_kernels/blackwell/gdn_prefill.py": (
        "0068d71f55cc5028f91fe1537e9d7b42143f4ab1cb089ded2d1307ee73438ae6",
    ),
    "gdn_kernels/blackwell/gated_delta_net_chunked.py": (
        "906173728b8134af20808e64cefe4a200839831d21d03dc8ab04565d5af96016",
    ),
    "gdn_kernels/delta_rule_dsl/varlen_helper.py": (
        "b321564ed582784eac8c72596de8096425ec5eee959b2ec1297e7a039a1161dc",
    ),
}


_ENTRY = []


def flashinfer_entry():
    if not _ENTRY:
        try:
            from flashinfer.gdn_prefill import chunk_gated_delta_rule
        except (ImportError, RuntimeError):
            chunk_gated_delta_rule = None
        _ENTRY.append(chunk_gated_delta_rule)
    return _ENTRY[0]


def _flashinfer_root() -> Optional[Path]:
    found = importlib.util.find_spec("flashinfer")
    if found is None or found.origin is None:
        return None
    return Path(found.origin).parent


def flashinfer_reasons() -> list[str]:
    root = _flashinfer_root()
    if root is None:
        return ["needs FlashInfer"]
    try:
        version = importlib.metadata.version("flashinfer-python")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    changed = []
    for relative, digests in FLASHINFER_SOURCES.items():
        path = root / relative
        digest = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        )
        if digest not in digests:
            changed.append(relative)
    if changed:
        return [
            f"FlashInfer {version}'s {', '.join(changed)} differ from the GDN prefill "
            "the slot was checked against"
        ]
    return []


def routes_native(num_seqs: int, num_heads: int, device) -> bool:
    from flashinfer.gdn_kernels.delta_rule_dsl.varlen_helper import should_use_cp_host
    from flashinfer.utils import (
        get_compute_capability,
        get_device_name,
        get_device_sm_count,
    )

    capability = get_compute_capability(device)
    if capability[0] != 10:
        return False
    return not should_use_cp_host(
        num_seqs * num_heads,
        get_device_sm_count(device),
        get_device_name(device),
        device_capability=capability,
    )


class Plan(NamedTuple):
    rows: torch.Tensor
    slots: torch.Tensor


def checkpoint_rows(track_src: torch.Tensor, num_checkpoints: int) -> Plan:
    slots = torch.arange(
        track_src.numel(), dtype=track_src.dtype, device=track_src.device
    )
    rows = torch.full(
        (num_checkpoints,), -1, dtype=torch.int32, device=track_src.device
    )
    rows[track_src.long()] = slots.to(torch.int32)
    return Plan(rows, slots)


def run_prefill(
    kernel,
    *,
    q,
    k,
    v,
    alpha,
    beta,
    initial_state,
    cu_seqlens,
    output,
    state_checkpoints,
    checkpoint_cu_starts,
    checkpoint_every_n_tokens,
    checkpoint_rows,
):
    return kernel._prefill_fn(
        q=q,
        k=k,
        v=v,
        g=alpha,
        beta=beta,
        scale=None,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
        output=output,
        output_state=torch.empty_like(initial_state),
        state_checkpoints=state_checkpoints,
        checkpoint_cu_starts=checkpoint_cu_starts,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        use_cp=False,
        checkpoint_state_indices=checkpoint_rows,
    )


class TrackSnapshot:
    def __init__(self):
        self._src = None
        self._key = None
        self._plan = None
        self._native = {}

    def native(self, num_seqs: int, num_heads: int, device) -> bool:
        key = (num_seqs, num_heads, torch.device(device))
        if key not in self._native:
            self._native[key] = routes_native(num_seqs, num_heads, device)
        return self._native[key]

    def plan(self, view, kernel, num_heads: int) -> Optional[Plan]:
        src = view.track_ssm_h_src
        if (
            not view.has_track
            or src is None
            or not src.numel()
            or view.num_state_checkpoints <= 0
            or view.state_checkpoint_every_n_tokens <= 0
            or view.state_checkpoint_cu_starts is None
            or getattr(kernel, "_prefill_fn", None) is None
            or kernel._prefill_fn is not flashinfer_entry()
            or not self.native(view.query_start_loc.numel() - 1, num_heads, src.device)
        ):
            return None
        version = None if src.is_inference() else src._version
        key = (src.data_ptr(), version, src.numel(), view.num_state_checkpoints)
        if self._src is None or self._src() is not src or self._key != key:
            self._plan = checkpoint_rows(src, view.num_state_checkpoints)
            self._src, self._key = weakref.ref(src), key
        return self._plan

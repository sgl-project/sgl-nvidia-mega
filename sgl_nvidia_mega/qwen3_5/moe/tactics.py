# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The TEP8 routed body's default FP4 MoE tactics, which ``body.width`` takes before
asking the autotuner.

The routed op's online autotune measures each token bucket once at startup, and its
pick between near-tied tactics varies from boot to boot. This table fixes the
``flashinfer::trtllm_fp4_block_scale_moe`` tactic of each bucket for the one
configuration it was measured on: the FlashInfer version, the architecture and the
MoERunner signature (``get_cache_key_extras``) below. Any other version, architecture
or signature, and any bucket the table lacks, asks the autotuner as before. Only the
routed body reads the table; every other op keeps its own tuning.

``ENV`` switches the table; unset, it follows ``PARENT_ENV``, SGLang's switch of the
megakernel runtime optimizations.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Union

logger = logging.getLogger(__name__)

ENV = "SGLANG_NVIDIA_MEGAKERNEL_MOE_TACTICS"
PARENT_ENV = "SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION"

FLASHINFER_VERSION = "0.7.1"
ARCH = "sm103"
# Qwen3.5's routed experts under TEP8: top-10 of 512 experts, 128 a rank, NVFP4 (E2m1)
# activations and weights, hidden 4096, intermediate 512, SwiGLU, shuffled MajorK.
SIGNATURE = (
    "Module",
    10,
    128,
    512,
    0,
    17826818,
    17826818,
    0,
    4096,
    4096,
    512,
    3,
    True,
    0,
    False,
    False,
)
# The autotuner's token bucket -> [tile_n, config]; -1 is the launcher's default tile_n
# and config. Best of repeated cold-L2 CUDA-graph tunings on GB300.
TACTICS: dict[int, Union[int, tuple[int, int]]] = {
    8: -1,
    16: -1,
    32: -1,
    64: (8, 317),
    128: (16, 309),
    256: (16, 312),
    512: -1,
    768: (32, 371),
    1024: (64, 6),
}

_TRUE = ("1", "true", "yes", "on")


def enabled() -> bool:
    raw = os.environ.get(ENV)
    if raw is None:
        raw = os.environ.get(PARENT_ENV, "")
    return raw.strip().lower() in _TRUE


def mismatch(version: str, arch: str, signature: tuple) -> str:
    """Why the table does not apply to this configuration ("" when it does)."""
    if version != FLASHINFER_VERSION:
        return f"flashinfer {version}, the table's is {FLASHINFER_VERSION}"
    if arch != ARCH:
        return f"{arch}, the table's is {ARCH}"
    if tuple(signature) != SIGNATURE:
        return f"MoERunner signature {tuple(signature)} is not the table's"
    return ""


def lookup(
    m: int, bucket: int, version: str, arch: str, signature: tuple
) -> Optional[Union[int, list]]:
    """The table's tactic at ``bucket`` (-1 or [tile_n, config]), or None: the
    caller asks the autotuner. Logs the tactic's source either way."""
    if not enabled():
        reason = f"{ENV} / {PARENT_ENV} off"
    else:
        reason = mismatch(version, arch, signature)
        if not reason and bucket not in TACTICS:
            reason = f"no entry for bucket {bucket}"
    if reason:
        logger.info(
            "NVIDIA megakernel: TEP8 routed body at %d rows (bucket %d): tactic "
            "from the autotuner (%s)",
            m,
            bucket,
            reason,
        )
        return None
    tactic = TACTICS[bucket]
    logger.info(
        "NVIDIA megakernel: TEP8 routed body at %d rows (bucket %d): tactic %s "
        "from the plugin table (flashinfer %s, %s)",
        m,
        bucket,
        tactic,
        version,
        arch,
    )
    return tactic if isinstance(tactic, int) else list(tactic)


def pick(m: int, runner, config, inputs: list) -> Optional[Union[int, list]]:
    """``lookup`` for ``runner`` at the autotuner's bucket of ``inputs``."""
    import flashinfer
    import torch
    from flashinfer.autotuner import AutoTuner

    shapes = tuple(
        tuple(t.size()) if isinstance(t, torch.Tensor) else (0,) for t in inputs
    )
    spec = config.dynamic_tensor_specs[0]
    profile = AutoTuner._find_nearest_profile(shapes, config)
    bucket = profile[spec.input_idx[0]][spec.dim_idx[0]]
    major, minor = torch.cuda.get_device_capability()
    return lookup(
        m,
        bucket,
        flashinfer.__version__,
        f"sm{major}{minor}",
        runner.get_cache_key_extras(inputs),
    )

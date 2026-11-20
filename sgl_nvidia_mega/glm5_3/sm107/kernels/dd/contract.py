# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from dataclasses import dataclass


@dataclass(frozen=True)
class BuildConfig:

    static_flags: int = 2752
    include_indexer: bool = True
    slot_mode: str = "reference_logical"
    arch: str = "sm_107a"


@dataclass(frozen=True)
class LayerInputs:

    x: object
    residual: object
    topk: object
    positions: object
    sequence_lengths: object


@dataclass(frozen=True)
class PreparedLayer:

    weights: object
    cache: object
    workspace: object
    descriptors: tuple[object, ...]
    build: BuildConfig


@dataclass(frozen=True)
class LayerOutputs:

    residual: object
    normalized: object | None
    topk: object | None

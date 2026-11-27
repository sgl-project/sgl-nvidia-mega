# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from typing import Any, List

_POOL: List[Any] = []


def persistent(device):
    import torch

    if not _POOL:
        _POOL.append(torch.cuda.MemPool())
    return torch.cuda.use_mem_pool(_POOL[0], device=device)

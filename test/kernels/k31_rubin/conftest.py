# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""``K31_SHARD=i/n`` keeps every n-th collected test starting at i (one shard per GPU)."""

import os


def pytest_collection_modifyitems(config, items):
    shard = os.environ.get("K31_SHARD")
    if not shard:
        return
    index, count = map(int, shard.split("/"))
    keep = [item for j, item in enumerate(items) if j % count == index]
    drop = [item for j, item in enumerate(items) if j % count != index]
    items[:] = keep
    config.hook.pytest_deselected(items=drop)

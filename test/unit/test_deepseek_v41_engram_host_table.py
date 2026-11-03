# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Host-table compatibility keeps prefetch limited to rank-local shards."""

from types import SimpleNamespace

import pytest

from sgl_nvidia_mega.deepseek_v41.engram.prefetch import _per_rank_host_table


@pytest.mark.parametrize(
    "table, expected",
    [
        pytest.param(None, False, id="no-table"),
        pytest.param(object(), False, id="unknown-table"),
        pytest.param(SimpleNamespace(replication="none"), True, id="current-shard"),
        pytest.param(
            SimpleNamespace(replication="per_node"), False, id="current-shared"
        ),
        pytest.param(SimpleNamespace(layout="per_rank"), True, id="legacy-shard"),
        pytest.param(SimpleNamespace(layout="shared"), False, id="legacy-shared"),
        pytest.param(
            SimpleNamespace(replication="per_node", layout="per_rank"),
            False,
            id="current-reject-overrides-legacy",
        ),
        pytest.param(
            SimpleNamespace(replication="none", layout="shared"),
            True,
            id="current-accept-overrides-legacy",
        ),
        pytest.param(
            SimpleNamespace(replication=None, layout="per_rank"),
            False,
            id="unknown-replication-does-not-fall-back",
        ),
    ],
)
def test_per_rank_host_table(table, expected):
    assert _per_rank_host_table(table) is expected

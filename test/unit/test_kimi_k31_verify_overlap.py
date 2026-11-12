# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU check of fork/join ordering and dense-Q8 dispatch, without CUDA."""
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from sgl_nvidia_mega.kimi_k31 import verify_executor as module


@pytest.mark.parametrize("q,sparse,use_overlap", [(8, 0, True), (8, 1, False), (16, 0, False)])
def test_join_precedes_input_reuse_and_sparse_writer_is_direct(monkeypatch, q, sparse, use_overlap):
    calls = []
    def op(name):
        return Mock(side_effect=lambda *args, **kwargs: calls.append(name))
    helper = NS(run=op("fork"), join=op("join"))
    factory = Mock(return_value=helper)
    monkeypatch.setattr(module, "VerifyMlaIndexOverlap", factory)
    kinds = ("mla", "kda", "mla")
    resources = NS(
        layers=[NS(layer_idx=i, _is_moe_layer=1) for i in range(3)],
        scratch={k: torch.empty(q, 7168, dtype=torch.bfloat16) for k in
                 ("prefix", "blocks", "hidden.0", "hidden.1", "hidden.out",
                  "mla.projected", "kda.projected", "attention.local")},
        before={i: NS(run=op(f"before{i}")) for i in range(3)},
        final=NS(run=op("final")), fused_before={},
        attention={i: NS(kind=kind, projection=NS(run=op(f"projection{i}")),
                         core=NS(run=op(f"core{i}")), cache=None,
                         output=NS(run=op(f"output{i}"))) for i, kind in enumerate(kinds)},
        feed_forward={i: NS(run=Mock(side_effect=lambda *args, i=i, **kwargs:
                                    (calls.append(f"feed{i}"), args[1])[1])) for i in range(3)},
    )
    indexers = {i: NS(run=op(f"index{i}")) for i in (0, 2)}
    executor = module.K31VerifyExecutor(resources, indexers, (), query_width=q)
    executor.kda_cores[1] = NS(run=op("kda1"))
    executor.kda_pools[1] = None
    view = NS(validate=Mock(), row_count=q, query_width=q, batch_requests=1,
              aux_hidden=torch.empty(q, 0, dtype=torch.bfloat16), sparse=sparse,
              write_slots=torch.arange(q), block_table=None, sequence_lengths=None, kda=None)
    executor.run(torch.zeros(q, 7168, dtype=torch.bfloat16), view)
    expected = []
    for i, kind in enumerate(kinds):
        expected.append(f"before{i}")
        if kind == "mla": expected.append("fork" if use_overlap else f"index{i}")
        expected += [f"projection{i}", f"core{i}" if kind == "mla" else "kda1", f"output{i}"]
        if kind == "mla" and use_overlap: expected.append("join")
        expected.append(f"feed{i}")
    assert calls == expected + ["final"]
    if use_overlap:
        assert [call.args[0] for call in helper.run.call_args_list] == [indexers[0], indexers[2]]
        assert all(call.args[2] is view.write_slots for call in helper.run.call_args_list)
    else:
        helper.run.assert_not_called()
        helper.join.assert_not_called()
    if q == 8: factory.assert_called_once_with(resources.scratch["hidden.0"].device)
    else: factory.assert_not_called()
    assert not torch.cuda.is_initialized()

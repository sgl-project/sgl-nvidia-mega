# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU regression for the actual #4162 CP build layouts; applies to pure v3.

DEP8's actual EagleBinding exposes global TP8, not attention TP1. Before the
fix its unused CP state could initialize; TP2 instead raised. Topology filtering
must precede collective initialization in either case.
"""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize(
    "name,tp,dp,ep,batch,expects_cp",
    [
        ("agg-tp2", 2, 1, 1, 1, False),
        ("pd-d-tp2-b6", 2, 1, 1, 6, False),
        ("pd-d-tp2-b10", 2, 1, 1, 10, False),
        ("agg-tp4", 4, 1, 1, 1, True),
        ("agg-tp8", 8, 1, 1, 1, True),
        ("pd-dep8", 8, 8, 8, 32, False),
    ],
)
def test_actual_cp_layout_filters_before_target_build(
    monkeypatch, name, tp, dp, ep, batch, expects_cp
):
    from sglang.srt.kernel_plugins.nvidia import binding as plugin_binding

    from sgl_nvidia_mega.qwen3_5 import runtime
    from sgl_nvidia_mega.qwen3_5 import spec as target
    from sgl_nvidia_mega.qwen3_5.sm103.attention.q35_cp_lite import CpLite

    # Exact current B env, including CP names. No downstream topology import:
    # this file applies without changes to the upstream c9d3 source tree.
    requested_csv = (
        "route_topk,mkm_m32,mkm_m64,mkm_m128,mkm_m256,mkm_m512,gdn_verify,"
        "gdn_pdl,gdn_conv,hop,hop_fp8,multi_request,draft_loop,draft_extend,"
        "draft_multi,draft_embed,route,route_gemm,mkm,mkm_wide,gdn_core,"
        "hop_mid,gdn_push,gdn_front,gdn_trt_front,route_shared,route_lin,skv2,"
        "mka,mka_push,lowm_in,lowm_out,lowm_se,lowm_push,skv2_cp,mkm_wide_bal,"
        "draft_moe,draft_q1,draft_xchg,draft_hop,draft_fin,draft_front,"
        "draft_skv2,draft_q1_cp,draft_skv2_cp,sharded_gumbel,"
        "gdn_prefill_split_conv"
    )
    requested = frozenset(requested_csv.split(","))
    monkeypatch.setattr(runtime, "SLOTS", requested)
    parallel = SimpleNamespace(
        tp_size=tp,
        tp_rank=0,
        attn_dp_size=dp,
        attn_tp_size=tp // dp,
        attn_tp_rank=0,
        moe_ep_size=ep,
        tp_group=SimpleNamespace(device_group="group"),
    )
    monkeypatch.setattr("sglang.srt.runtime_context.get_parallel", lambda: parallel)
    monkeypatch.setattr(plugin_binding, "megakernel_agree", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(CpLite, "_instance", None)
    allocations = []
    monkeypatch.setattr(
        CpLite, "__init__", lambda self, *args: allocations.append(args)
    )
    # Actual EagleBinding.tp_size is global TP, including DEP8.
    binding = SimpleNamespace(
        page_size=64, tp_size=tp, tp_rank=0, tp_device_group="group"
    )
    original = runtime.build_cp_lite
    checked = []

    class SetupChecked(Exception):
        pass

    def setup(actual_binding):
        assert actual_binding is binding
        original(actual_binding)
        assert bool(runtime.SLOTS & runtime.CP_LITE_SLOTS) == expects_cp, name
        assert bool(allocations) == expects_cp, name
        checked.append(True)
        raise SetupChecked

    monkeypatch.setattr(runtime, "build_cp_lite", setup)
    ctx = SimpleNamespace(
        widths=(batch * 7,),
        tokens_per_request=7,
        deadline_s=60.0,
        binding=binding,
    )
    with pytest.raises(SetupChecked):
        target.SPEC.build(SimpleNamespace(), SimpleNamespace(), ctx)
    assert checked == [True]

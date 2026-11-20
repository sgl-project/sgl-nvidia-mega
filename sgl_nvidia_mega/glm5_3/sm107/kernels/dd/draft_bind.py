# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import cutlass
from cuda.bindings import driver
from cutlass.cute.runtime import from_dlpack

HEAD = (
    "x",
    "residual",
    "residual_out",
    "normalized",
    "wfa",
    "wqb",
    "wo",
    "w_router",
    "bias",
    "w3w1",
    "w2",
    "w_sgu",
    "w_sdn",
    "gamma_kv",
    "gamma_in",
    "gamma_post",
    "gamma_head",
    "gamma_hnorm",
    "rope",
    "length",
    "table",
    "cache",
    "topk",
    "arena",
    "partials",
    "stats",
    "counters",
    "uc",
    "mc",
    "lm_w",
    "token_table",
    "hidden_rows",
    "draft_tokens",
    "next_new_tokens",
    "accepted",
    "num_accepted",
)
SCALARS = (
    ("copy", "i32"),
    ("rank", "i32"),
    ("step", "i32"),
    ("last", "bool"),
    ("stride_rows", "i64"),
    ("layer_rows", "i64"),
    ("max_blocks", "i32"),
    ("clear_first", "bool"),
    ("kind", "i32"),
)
S0_TENSORS = (
    "wfa_leader",
    "widx",
    "idx_kn_w",
    "idx_kn_b",
    "idx_rope",
    "idx_pool",
    "idx_blocks",
)


def contract():
    rows = [{"role": n, "kind": "tensor", "owner": "caller"} for n in HEAD]
    rows += [{"role": n, "kind": "scalar", "owner": "caller"} for n, _k in SCALARS]
    rows += [
        {"role": n, "kind": "tensor", "owner": "prepared-layer"} for n in S0_TENSORS
    ]
    rows += [
        {"role": "idx_stride_bytes", "kind": "scalar", "owner": "caller"},
        {"role": "idx_pages", "kind": "scalar", "owner": "caller"},
        {"role": "idx_candidates", "kind": "tensor", "owner": "shared"},
        {"role": "embed", "kind": "tensor", "owner": "prepared-layer"},
        {"role": "embed_sharded", "kind": "scalar", "owner": "caller"},
        {"role": "eh_w", "kind": "tensor", "owner": "prepared-layer"},
        {"role": "gamma_enorm", "kind": "tensor", "owner": "prepared-layer"},
        {"role": "scale", "kind": "constexpr", "owner": "caller"},
        {"role": "stream", "kind": "stream", "owner": "caller"},
        {"role": "use_pdl", "kind": "constexpr", "owner": "caller"},
        {"role": "with_target_argmax", "kind": "constexpr", "owner": "caller"},
        {"role": "with_target_sample", "kind": "constexpr", "owner": "caller"},
    ]
    return rows


def _call(request, proxies, with_target_argmax, with_target_sample):
    tree = request["semantic_trees"]
    refs, values = tree["tensors"], tree["scalars"]

    def tensor(name):
        return from_dlpack(proxies[refs[name]["tensor_ref"]], enable_tvm_ffi=False)

    kinds = {"i32": cutlass.Int32, "i64": cutlass.Int64, "bool": cutlass.Boolean}
    args = [tensor(n) for n in HEAD]
    args += [kinds[k](values[n]) for n, k in SCALARS]
    args += [tensor(n) for n in S0_TENSORS]
    args += [
        cutlass.Int64(values["idx_stride_bytes"]),
        cutlass.Int64(values["idx_pages"]),
        tensor("idx_candidates"),
        tensor("embed"),
        cutlass.Boolean(values["embed_sharded"]),
        tensor("eh_w"),
        tensor("gamma_enorm"),
        float(values["scale"]),
        driver.CUstream(request["stream"]),
        bool(request["pdl"]),
        with_target_argmax,
        with_target_sample,
    ]
    return {"entry": draft_entry, "args": tuple(args), "argument_contract": contract()}


def draft_call(request, proxies):
    """Draft steps never reach the target argmax or sampling stage, so both are compiled out."""
    return _call(request, proxies, False, False)


def tail_call(request, proxies):
    """The verify tail picks the target token, so it keeps both target stages."""
    return _call(request, proxies, True, True)


from candidate.draft import draft_entry

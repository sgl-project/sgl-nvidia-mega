# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import cutlass
from cuda.bindings import driver
from cutlass.cute.runtime import from_dlpack


HEAD = ("x", "residual", "residual_out", "normalized", "wfa", "wqb", "wo", "wgu", "wdn",
        "gamma_kv", "gamma_post", "gamma_next", "rope", "length", "table", "cache", "topk",
        "arena", "partials", "stats", "counters", "uc", "mc")
SCALARS = (("copy", "i32"), ("rank", "i32"), ("early_x", "bool"), ("x_only", "bool"),
           ("stride_rows", "i64"), ("layer_rows", "i64"), ("max_blocks", "i32"),
           ("clear_first", "bool"))
INDEXER = ("widx", "idx_kn_w", "idx_kn_b", "idx_rope", "idx_pool", "idx_blocks")
TAIL = ("ids", "embed", "gamma_in")
OWNERS = {"x": "caller", "residual": "caller", "residual_out": "caller", "normalized": "caller",
          "length": "caller", "table": "caller", "cache": "caller", "topk": "caller",
          "idx_pool": "caller", "idx_blocks": "caller", "ids": "caller",
          "arena": "shared", "partials": "shared", "stats": "shared", "counters": "shared",
          "uc": "shared", "mc": "shared", "idx_candidates": "shared"}


def contract():
    rows = [{"role": n, "kind": "tensor", "owner": OWNERS.get(n, "prepared-layer")} for n in HEAD]
    rows += [{"role": n, "kind": "scalar", "owner": "caller"} for n, _k in SCALARS]
    rows += [{"role": n, "kind": "tensor", "owner": OWNERS.get(n, "prepared-layer")} for n in INDEXER]
    rows += [{"role": "idx_stride_bytes", "kind": "scalar", "owner": "caller"},
             {"role": "idx_pages", "kind": "scalar", "owner": "caller"},
             {"role": "idx_candidates", "kind": "tensor", "owner": "shared"},
             {"role": "layer0", "kind": "scalar", "owner": "caller"}]
    rows += [{"role": n, "kind": "tensor", "owner": OWNERS.get(n, "prepared-layer")} for n in TAIL]
    rows += [{"role": "stream", "kind": "stream", "owner": "caller"},
             {"role": "use_pdl", "kind": "constexpr", "owner": "caller"}]
    return rows


def d012_call(request, proxies):
    tree = request["semantic_trees"]
    refs, values = tree["tensors"], tree["scalars"]

    def tensor(name):
        return from_dlpack(proxies[refs[name]["tensor_ref"]], enable_tvm_ffi=False)

    kinds = {"i32": cutlass.Int32, "i64": cutlass.Int64, "bool": cutlass.Boolean}
    args = [tensor(n) for n in HEAD]
    args += [kinds[k](values[n]) for n, k in SCALARS]
    args += [tensor(n) for n in INDEXER]
    args += [cutlass.Int64(values["idx_stride_bytes"]), cutlass.Int64(values["idx_pages"]),
             tensor("idx_candidates"), cutlass.Boolean(values["layer0"])]
    args += [tensor(n) for n in TAIL]
    args += [driver.CUstream(request["stream"]), bool(request["pdl"])]
    return {"entry": d012_entry, "args": tuple(args), "argument_contract": contract()}


from candidate.d012 import d012_entry

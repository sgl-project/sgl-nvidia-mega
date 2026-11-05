# Vendored SGLang code

Files in this directory are copied from SGLang (https://github.com/sgl-project/sglang), licensed under the Apache License 2.0
(see ../include/LICENSE.sglang). They are SGLang pull-request versions that are not in the SGLang revision this plugin pins.

| Directory | Source | Commit | Files (path in the SGLang tree) |
|---|---|---|---|
| sglang_pr47174 | sgl-project/sglang pull request 47174 ("Fuse all-reduce + Norm") | e41c694aa247d29fac1d17caf4944ea3eebe8011 | python/sglang/kernels/jit/csrc/muse_spark/all_reduce_norm_residual.cuh, python/sglang/kernels/jit/csrc/muse_spark/norm_residual.cuh; and, in modified form only (see its header), python/sglang/kernels/ops/communication/muse_spark_all_reduce_norm_residual.py |
| sglang_pr46724 | sgl-project/sglang pull request 46724 ("Enable fused finalize+2shot push") | 553f3ed9862e32318efcf3b031c661a578985842 | python/sglang/kernels/ops/communication/muse_spark_moe_finalize_all_reduce_cutedsl.py, python/sglang/kernels/ops/communication/muse_spark_moe_finalize_all_reduce.py, python/sglang/srt/layers/communication/muse_spark_finalize_workspace.py |

The first commit that adds a file here carries it exactly as received from that commit, except the one file marked "in modified form only": it is added with its modifications and a "Modified by NVIDIA" header in the same commit. Later commits that change a file add a
"Modified by NVIDIA" line describing the change at the top of that file.

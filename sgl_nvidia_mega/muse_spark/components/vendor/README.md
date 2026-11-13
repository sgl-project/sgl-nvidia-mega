# Vendored SGLang code

Files in this directory are copied from SGLang (https://github.com/sgl-project/sglang), licensed under the Apache License 2.0
(see ../include/LICENSE.sglang). They are SGLang pull-request versions that are not in the SGLang revision this plugin pins.

| Directory | Source | Commit | Files (path in the SGLang tree) |
|---|---|---|---|
| sglang_pr47174 | sgl-project/sglang pull request 47174 ("Fuse all-reduce + Norm") | e41c694aa247d29fac1d17caf4944ea3eebe8011 | python/sglang/kernels/jit/csrc/muse_spark/all_reduce_norm_residual.cuh, python/sglang/kernels/jit/csrc/muse_spark/norm_residual.cuh; and, in modified form only (see its header), python/sglang/kernels/ops/communication/muse_spark_all_reduce_norm_residual.py |
| sglang_pr46724 | sgl-project/sglang pull request 46724 ("Enable fused finalize+2shot push") | 553f3ed9862e32318efcf3b031c661a578985842 | python/sglang/kernels/ops/communication/muse_spark_moe_finalize_all_reduce_cutedsl.py, python/sglang/kernels/ops/communication/muse_spark_moe_finalize_all_reduce.py, python/sglang/srt/layers/communication/muse_spark_finalize_workspace.py |

The first commit that adds a file here carries it exactly as received from that commit, except the one file marked "in modified form only": it is added with its modifications and a "Modified by NVIDIA" header in the same commit. Later commits that change a file add a
"Modified by NVIDIA" line describing the change at the top of that file.

# Vendored TensorRT-LLM code

trtllm_ab93b78/ holds the Rubin (sm_107) decode megakernel of TensorRT-LLM (https://github.com/NVIDIA/TensorRT-LLM), licensed under the Apache License 2.0 (see trtllm_ab93b78/LICENSE.tensorrt_llm), which the verify_mk component runs on SGLang (../verify_mk).

| Directory | Source | Commit | Files (path in the TensorRT-LLM tree) |
|---|---|---|---|
| trtllm_ab93b78/megakernel | NVIDIA/TensorRT-LLM | ab93b785a476f01d2f1e1907c930780fc6d7832e | tensorrt_llm/_torch/_experimental/modeling_v2/models/musespark_open/muse_spark_open_816b_a42b_nvfp4__sm_107__tep/kernels/megakernel/ (all 34 files); in modified form only (see their headers): library.py, msoMegakernel.cu, stages/attnHalf.cuh, stages/fullAttention.cuh |
| trtllm_ab93b78/replicated_experts | NVIDIA/TensorRT-LLM | ab93b785a476f01d2f1e1907c930780fc6d7832e | tensorrt_llm/_torch/_experimental/modeling_v2/models/musespark_open/muse_spark_open_816b_a42b_nvfp4__sm_107__tep/kernels/replicated_experts/ (all 5 files) |

Each file carries a header line naming its path and commit. The four modified files add SGLang's paged KV cache as a build option of the kernel (MSO_MK_SG_KV=1; the default build is TensorRT-LLM's, unchanged) and a pinned-memory fallback outside TensorRT-LLM; their "Modified by NVIDIA" header says what changed, and the changed blocks are marked verify_mk. The rest are unchanged apart from the header line. pre-commit's formatters and spell check skip this directory (.pre-commit-config.yaml), so the files stay comparable with the source.

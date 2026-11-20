# Muse Spark on SM107 (Rubin): verify_mk

`verify_mk` runs TensorRT-LLM's Muse Spark decode megakernel on SGLang's one-request verify step (target layers 1-60, TP8 = EP8), opt-in with `MUSE_VR_VERIFY_MK=1`. This directory holds its Rubin (sm_107) code: `verify_mk/`, the host runner (install, layer facades, workspaces on SGLang's TP group, KV-cache fields), and `trtllm_ab93b78/`, TensorRT-LLM's kernel and host code (below). The component loader imports `sm107.verify_mk` for the `verify_mk` component (`components.MODULES`).

## Vendored TensorRT-LLM code

trtllm_ab93b78/ holds the Rubin (sm_107) decode megakernel of TensorRT-LLM (https://github.com/NVIDIA/TensorRT-LLM), licensed under the Apache License 2.0 (see trtllm_ab93b78/LICENSE.tensorrt_llm), which the verify_mk component runs on SGLang (verify_mk/).

| Directory | Source | Commit | Files (path in the TensorRT-LLM tree) |
|---|---|---|---|
| trtllm_ab93b78/megakernel | NVIDIA/TensorRT-LLM | ab93b785a476f01d2f1e1907c930780fc6d7832e | tensorrt_llm/_torch/_experimental/modeling_v2/models/musespark_open/muse_spark_open_816b_a42b_nvfp4__sm_107__tep/kernels/megakernel/ (all 34 files); in modified form only (see their headers): library.py, msoMegakernel.cu, stages/attnHalf.cuh, stages/fullAttention.cuh |
| trtllm_ab93b78/replicated_experts | NVIDIA/TensorRT-LLM | ab93b785a476f01d2f1e1907c930780fc6d7832e | tensorrt_llm/_torch/_experimental/modeling_v2/models/musespark_open/muse_spark_open_816b_a42b_nvfp4__sm_107__tep/kernels/replicated_experts/ (all 5 files) |

Each file carries a header line naming its path and commit. The four modified files add SGLang's paged KV cache as a build option of the kernel (MSO_MK_SG_KV=1; the default build is TensorRT-LLM's, unchanged) and a pinned-memory fallback outside TensorRT-LLM; their "Modified by NVIDIA" header says what changed, and the changed blocks are marked verify_mk. The rest are unchanged apart from the header line. pre-commit's formatters and spell check skip trtllm_ab93b78/ (.pre-commit-config.yaml), so the files stay comparable with the source.

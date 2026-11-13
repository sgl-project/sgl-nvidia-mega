# Muse Spark on SM103 (GB300): verify_mk

`verify_mk` runs TensorRT-LLM's Muse Spark decode megakernel on SGLang's one-request verify step (target layers 1-60, TP8 = EP8). This directory is its GB300 (sm_103a) port. Rubin (sm_107) keeps `components/verify_mk` and `components/vendor/trtllm_ab93b78`, which this port does not change. On compute capability 10.3 the component loader imports `sm103.verify_mk` in place of `components.verify_mk` (`components.ARCH_MODULES`; the GPU is read through NVML); the switch `MUSE_VR_VERIFY_MK=1` and the other settings are the same.

| Directory | What | Copied from |
|---|---|---|
| `trtllm_ab93b78/` | TensorRT-LLM's megakernel and replicated-experts host code (NVIDIA/TensorRT-LLM ab93b785a476f01d2f1e1907c930780fc6d7832e, `tensorrt_llm/_torch/_experimental/modeling_v2/models/musespark_open/muse_spark_open_816b_a42b_nvfp4__sm_107__tep/kernels/{megakernel,replicated_experts}/`; Apache License 2.0, `trtllm_ab93b78/LICENSE.tensorrt_llm`) with the sm_103a port | `components/vendor/trtllm_ab93b78` |
| `verify_mk/` | the host runner: install, layer facades, workspaces on SGLang's TP group, KV-cache fields | `components/verify_mk` |

`diff -r components/vendor/trtllm_ab93b78 sm103/trtllm_ab93b78` and `diff -r components/verify_mk sm103/verify_mk` show the whole port. The vendored-code rule of `components/vendor/README.md` holds here too: TensorRT-LLM's files stay as received except for changed blocks, which are marked `verify_mk`, and each modified file's "Modified by NVIDIA" header says what changed. pre-commit's formatters and spell check skip `trtllm_ab93b78/`.

## The port

- Build target sm_103a (`library.arch_target`, `library.ARCH_DEFINES["103a"]`): one CTA per SM of GB300's 152 (`MSO_MK_CTAS=152`), the 212 QKV runs kept (`MSO_MK_QKV_RUNS=212`: a run's weights fit the QKV run region, so some CTAs run two), and every launch planned for TensorRT-LLM's 212 CTAs and folded onto 152 by `fold.py` (added; same units, same tiles). `serving.refusals` takes (10, 3).
- Modified TensorRT-LLM files: `library.py`, `serving.py`, `msoMegakernel.cu`, `stages/attnHalf.cuh`, `stages/experts.cuh`, `schedule.py`, `layer_mk.py`, `attn_half.py`, `full_half.py`, `qkv.py`.
- Dev-only builds, selected with `MUSE_VR_VERIFY_MK_BUILD`: single-lever variations of the served build (`SG_SWEEP_MODES`) and their combinations (`SG_COMBO_MODES`), an expert-weight L2 prefetch prototype (`MSO_MK_XPF`, `XPF_MODES`) and the served build's stamped twin (`SG_STAMPED_MODE`, with `MUSE_VR_VERIFY_MK_STAMPS`: per-unit timelines of every layer).
- Host runner: `MUSE_VR_VERIFY_MK_TAPS=1` has the kernel write the DFlash taps of layers 1-60 in place of the layer loop's copies.

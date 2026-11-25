# SGLang-NVIDIA-Mega

NVIDIA megakernels for [SGLang](https://github.com/sgl-project/sglang). For each
model it covers, this package runs the small-batch decode and verify steps as
chains of fused kernels that hand their outputs to each other through mailboxes
and programmatic dependent launch, instead of op by op.

The package is an SGLang
[general plugin](https://github.com/sgl-project/sglang/blob/main/docs/docs/hardware-platforms/plugin.mdx).
Start SGLang with `--enable-nvidia-megakernel` and it checks the setup at
startup, builds the megakernel after the model loads, and runs the covered steps
with it; every other step runs SGLang's own code, and every rank logs
`NVIDIA megakernel: active (…)`. Without the flag, SGLang runs untouched.

> **Experimental:** the kernels, the models and setups they cover, and the switch
> may change.

## Supported models

| Model | GPU | Tensor parallel | Covered steps | Details |
| --- | --- | --- | --- | --- |
| DeepSeek-V4.1-Flash | GB300; B300 | 4; 8 (MoE TP4 x EP2) | DSpark target verify and draft | [deepseek_v41](docs/models/deepseek_v41.md) |
| Qwen3.5-397B-A17B (NVFP4) | GB300; GB200 | 8 | MTP target verify and EAGLE draft | [qwen3_5](docs/models/qwen3_5.md) |
| GLM-5.3, GLM-5.2 (NVFP4) | B300, GB300, Rubin | 4; 8; 16 (GLM-5.2 on Rubin) | MTP target verify and NextN draft; at TP16 the whole one-request EAGLE step | [glm5_3](docs/models/glm5_3.md) |
| Kimi K3 (MXFP4) | GB300; B300, B200, GB200 | 8 | Decode and fixed-width verify | [kimi_k3](docs/models/kimi_k3.md) |
| Kimi K3.1 (MXFP4) | GB300; B300, B200, GB200 | 8; 16 | DFlash target verify and DFlash2 draft | [kimi_k31](docs/models/kimi_k31.md) |
| Muse Spark | GB300 | 4 | DFlash target verify | [muse_spark](sgl_nvidia_mega/muse_spark/PORT-NOTES-sm103.md) |
| MiniMax-M3.1-Flash (NVFP4) | GB300 | 4 | DSpark target verify at 1 to 16 requests and the draft's Markov proposal | [minimax_m31](docs/models/minimax_m31.md) |

B300, B200 and GB200 (148 SMs) are admitted with `sm148`; the model pages give
the exact row widths, batch sizes and what was validated on each.

Any EAGLE v2 model can also run its topk-1 decode step as one CUDA graph (off by
default, behind its own switches); see [docs/spec_step.md](docs/spec_step.md).

## Installation

Use SGLang's CUDA 13 image, which provides SGLang, FlashInfer, CuTe DSL and the
CUDA Python bindings. The package needs an SGLang with the megakernel plugin
interface (`sglang.kernels.plugin_kernels.nvidia`); some models need further
SGLang bindings, listed on their pages. Then:

```bash
git clone https://github.com/sgl-project/sgl-nvidia-mega.git
cd sgl-nvidia-mega
pip install -e .
```

## Quick start

DeepSeek-V4.1-Flash on 4 x GB300:

```bash
SGLANG_SPEC_TP_SYNC=rng,-dspark-accept-sample \
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V4.1-Flash \
  --tp 4 \
  --ep-size 1 \
  --kv-cache-dtype fp8_e4m3 \
  --dsv4-attn-backend flashmla \
  --engram-config '{"placement": "cpu"}' \
  --speculative-algorithm DSPARK \
  --speculative-dspark-block-size 5 \
  --enable-nvidia-megakernel
```

Each model page has its launch command. An unsupported setup stops the server
at startup and names every reason.

## Repository structure

```text
sgl-nvidia-mega/
├── sgl_nvidia_mega/
│   ├── plugin.py      # registers each model's spec with SGLang
│   ├── common/        # shared build, launch and sync helpers
│   └── <model>/       # one package per model: spec, plan, runtime, step code, sm103/ kernels
├── docs/models/       # one page per model
├── benchmark/         # measurement tools
└── test/
    ├── unit/          # CPU tests, run on this repository's CI
    ├── e2e/           # GPU tests, run on SGLang's GPU CI
    └── kernels/       # legacy standalone kernel tests, not run in CI
```

## Contributing

Read [AGENTS.md](AGENTS.md): the dev guide (what belongs in this repository and
how to write it), the contribution guide (the flow with SGLang and the PR) and
the CI test guide (unit and E2E tests and the comparison protocol).

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.

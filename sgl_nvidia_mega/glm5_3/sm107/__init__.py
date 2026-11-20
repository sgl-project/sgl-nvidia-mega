# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""GLM-5.2 at TP16 on Rubin (SM107), one request, EAGLE 7 steps / topk 1 / 8 draft
tokens: the 75-layer decoder chain, dense layers 0-2, the fused MTP draft and the
fused verify tail, launched from prebuilt cubins.

glm5_3's specs serve it. With ``--enable-nvidia-megakernel`` at TP16,
``glm5_3/spec.py`` and ``glm5_3/draft.py`` dispatch here: ``config_reasons`` at
argument resolution, ``model_reasons`` and ``build_target`` / ``build_draft`` at
load. There is no other switch. The cubin directory is configuration, not a switch:
``SGLANG_NVIDIA_MEGAKERNEL_GLM_SM107_CUBINS`` (``ARTIFACTS_ENV``), refused at
argument resolution when unset or incomplete.

The build packs the weights before SGLang sizes the KV cache; the chain and the
draft bind to their runner's KV pools just before that runner captures its CUDA
graphs (``ModelRunner.init_cuda_graphs``).

This package imports without SGLang (the kernel tests); SGLang is imported in the
functions that the specs call.
"""

from __future__ import annotations

import functools
import importlib.util
import os

TP_SIZE = 16
# The directory of the prebuilt cubins, {chain,d012,draft,tail}/{entry.cubin,CALL.json},
# captured from kernels/; config_reasons refuses the setup when it is unset or incomplete
ARTIFACTS_ENV = "SGLANG_NVIDIA_MEGAKERNEL_GLM_SM107_CUBINS"
ARTIFACT_PARTS = ("chain", "d012", "draft", "tail")
ARTIFACT_FILES = ("entry.cubin", "CALL.json")
ISA = "sm_107"
# EAGLE 7 steps, topk 1: one request's verify step is 8 rows
TOKENS_PER_REQUEST = 8
NUM_STEPS = TOKENS_PER_REQUEST - 1

# The server arguments this path was validated with (glm5_3/spec.py checks them the
# way SGLang checks a spec's requirements)
SERVER_ARGS = {
    "kv_cache_dtype": frozenset({"fp8_e4m3"}),
    "page_size": frozenset({None, 64}),
    "attention_backend": frozenset({None, "dsa"}),
    "decode_attention_backend": frozenset({None, "dsa"}),
    "dsa_decode_backend": frozenset({"trtllm"}),
    "moe_runner_backend": frozenset({"flashinfer_trtllm"}),
    "moe_a2a_backend": frozenset({None, "none"}),
    "dp_size": frozenset({None, 1}),
    "moe_dp_size": frozenset({None, 1}),
    "ep_num_redundant_experts": frozenset({None, 0}),
    "speculative_algorithm": frozenset({"EAGLE", "NEXTN"}),
    "speculative_eagle_topk": frozenset({1}),
    "speculative_num_steps": frozenset({NUM_STEPS}),
    "speculative_num_draft_tokens": frozenset({TOKENS_PER_REQUEST}),
    "speculative_adaptive": frozenset({None, False}),
    "enable_multi_layer_eagle": frozenset({None, False}),
    "speculative_use_rejection_sampling": frozenset({None, False}),
    # the kernels read req_to_token as physical KV slots; unified memory translates them
    "enable_unified_memory": frozenset({None, False}),
}
# The chain and layers 0-2 compute the DSA top-k themselves; validated with SGLang's
# unfused top-k on every other step
ENV = {"SGLANG_DSA_FUSE_TOPK": False}

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/model_executor/model_runner.py": (
        "ModelRunner",
        "init_cuda_graphs",
        "is_draft_worker",
    ),
    "srt/runtime_context.py": ("get_server_args",),
    "srt/arg_groups/fields/*.py": ("model_path", "speculative_draft_model_path"),
    "srt/kernel_plugins/nvidia/binding.py": ("device_isa",),
    "kernels/plugin_kernels/nvidia/spec.py|srt/megakernels/spec.py": ("Step",),
}


def artifact_reasons() -> list[str]:
    root = os.environ.get(ARTIFACTS_ENV)
    if not root:
        return [
            f"{ARTIFACTS_ENV} is not set; set it to the directory of the glm5_3/sm107 cubins"
        ]
    missing = [
        os.path.join(part, name)
        for part in ARTIFACT_PARTS
        for name in ARTIFACT_FILES
        if not os.path.isfile(os.path.join(root, part, name))
    ]
    if missing:
        return [f"{ARTIFACTS_ENV}={root} lacks {', '.join(missing)}"]
    return []


def config_reasons(cfg, hf_config) -> list[str]:
    """Every reason the TP16 setup cannot be served, at argument resolution."""
    from sglang.srt.kernel_plugins.nvidia.binding import device_isa

    from sgl_nvidia_mega.glm5_3.spec import pinned_reasons

    reasons = []
    isa = device_isa()
    if isa != ISA:
        reasons.append(f"TP{TP_SIZE} runs on {ISA} (Rubin) only, not {isa}")
    reasons += pinned_reasons(cfg, SERVER_ARGS, ENV)
    for module, what in (
        ("cutlass", "CuTe DSL (nvidia-cutlass-dsl)"),
        ("cuda.bindings", "the CUDA Python bindings"),
    ):
        if importlib.util.find_spec(module) is None:
            reasons.append(f"needs {what}")
    mode = getattr(cfg, "disaggregation_mode", None) or "null"
    if mode != "null":
        reasons.append(f"PD disaggregation ({mode} role) is not served at TP{TP_SIZE}")
    reasons += artifact_reasons()
    return [f"glm5_3/sm107: {r}" for r in reasons]


def model_reasons(model) -> list[str]:
    """This rank's reasons the loaded target or draft cannot run, at load."""
    if type(model).__name__ == "GlmMoeDsaForCausalLMNextN":
        return []
    inner = getattr(model, "model", None)
    if type(inner).__name__ != "DeepseekV2Model":
        return [
            f"glm5_3/sm107: the decoder is {type(inner).__name__}, not DeepseekV2Model"
        ]
    return []


@functools.lru_cache(maxsize=None)
def _step_class():
    try:
        from sglang.kernels.plugin_kernels.nvidia.spec import Step
    except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
        from sglang.srt.megakernels.spec import Step
    return Step


class Built:
    """What build_target and build_draft return (SGLang's BuiltMegakernel). The
    chain (target) and the fused draft run inside the model's own forward and
    SGLang's draft worker, so a covered step calls the original forward."""

    draft_sampler = draft_loop = draft_extend = begin_draft_round = graph_variants = (
        None
    )
    speculative_runtime_hooks = None

    def __init__(self, what: str, is_draft: bool):
        self.what, self.is_draft = what, is_draft
        self.widths = (TOKENS_PER_REQUEST,)

    def summary(self) -> str:
        return f"glm5_3/sm107 {self.what}, TP{TP_SIZE}, one request"

    def plan(self, input_ids, forward_batch, *args, **kwargs):
        if forward_batch.batch_size != 1:
            return None
        mode = forward_batch.forward_mode
        if self.is_draft:
            # one request's draft extend and draft decodes run the fused draft
            return (
                None
                if mode.is_idle()
                else _step_class()(mode="decode", rows=int(input_ids.shape[0]))
            )
        if mode.is_target_verify() and input_ids.shape[0] == TOKENS_PER_REQUEST:
            return _step_class()(mode="target_verify", rows=TOKENS_PER_REQUEST)
        return None

    def forward(self, step, original, *args, **kwargs):
        return original(*args, **kwargs)


def _bind_before_graphs() -> None:
    """Bind the engaged chain or draft to its runner's KV pools before that runner
    captures its CUDA graphs."""
    from sglang.srt.model_executor.model_runner import ModelRunner

    if getattr(ModelRunner, "_glm5_3_sm107", False):
        return
    init_cuda_graphs = ModelRunner.init_cuda_graphs

    @functools.wraps(init_cuda_graphs)
    def wrapped(self, *args, **kwargs):
        from . import draft, target

        if getattr(self, "is_draft_worker", False):
            draft.bind_runner(self)
        else:
            target.bind_runner(self)
        return init_cuda_graphs(self, *args, **kwargs)

    ModelRunner.init_cuda_graphs = wrapped
    ModelRunner._glm5_3_sm107 = True


def _check_context(ctx) -> None:
    if (
        ctx.tokens_per_request != TOKENS_PER_REQUEST
        or TOKENS_PER_REQUEST not in ctx.widths
    ):
        raise ValueError(
            f"glm5_3/sm107 builds {TOKENS_PER_REQUEST} tokens a request, not "
            f"{ctx.tokens_per_request} (widths {list(ctx.widths)})"
        )


def build_target(model, hf_config, ctx) -> Built:
    from . import target

    _check_context(ctx)
    _bind_before_graphs()
    target.engage(model)
    return Built("target: decoder chain + layers 0-2 + verify tail", is_draft=False)


def _model_path(hf_config) -> str:
    try:
        from sglang.srt.runtime_context import get_server_args

        args = get_server_args()
        path = args.speculative_draft_model_path or args.model_path
    except Exception:
        path = getattr(hf_config, "_name_or_path", None)
    if not path or not os.path.isfile(
        os.path.join(path, "model.safetensors.index.json")
    ):
        raise ValueError(
            f"glm5_3/sm107: the draft weights need a local checkpoint directory, not {path!r}"
        )
    return path


def build_draft(model, hf_config, ctx) -> Built:
    from . import draft

    _check_context(ctx)
    _bind_before_graphs()
    draft.engage(model, _model_path(hf_config))
    return Built("NextN draft", is_draft=True)

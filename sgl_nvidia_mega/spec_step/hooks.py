# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The speculative step graph and fusions, installed into SGLang as HookRegistry
hooks. ``register`` adds only the hooks of the switches that are on."""

from __future__ import annotations

import logging
import types

import torch

from sgl_nvidia_mega.spec_step import switches

logger = logging.getLogger(__name__)

_BACKEND = "sglang.srt.model_executor.runner_backend.full_cuda_graph_backend.FullCudaGraphBackend"
_WORKER = "sglang.srt.speculative.eagle_worker_v2.EAGLEWorkerV2"
_DRAFT_WORKER = "sglang.srt.speculative.eagle_worker_v2.EagleDraftWorker"
_FUTURE_MAP = "sglang.srt.managers.overlap_utils.FutureMap"
_SPEC_UTILS = "sglang.srt.speculative.spec_utils"

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/model_executor/runner_backend/full_cuda_graph_backend.py": (
        "FullCudaGraphBackend",
        "capture_one(shape_key)",
        "replay(shape_key)",
        "cleanup",
        "_graphs",
        "_outputs",
        "torch.cuda.CUDAGraph()",
    ),
    "srt/model_executor/runner_backend/utils.py": ("resolve_decode_backend",),
    "srt/speculative/eagle_worker_v2.py": (
        "EAGLEWorkerV2",
        "init_cuda_graphs",
        "forward_batch_generation",
        "activate_step_by_batch",
        "_hosts_draft",
        "server_args",
        "EagleDraftWorker",
        "_draft_extend_for_decode",
    ),
    "srt/speculative/eagle_utils.py": ("eagle_sample",),
    "srt/speculative/spec_utils.py": (
        "generate_simulated_accept_index",
        "GrammarTree.from_device",
        "build_grammar_vocab_mask",
    ),
    "kernels/ops/speculative/sampling.py": ("tree_speculative_sampling_target_only",),
    "srt/managers/overlap_utils.py": (
        "FutureMap",
        "_resolve_spec_extras",
        "stash",
        "_maybe_init_forward_bufs",
        "_maybe_init_dsa_topk_indices_buf",
        "_DEBUG_ASSERT",
    ),
    "srt/managers/scheduler_components/internal_state.py": (
        "SchedulerInternalStateController._add_speculative",
    ),
}


# ------------------------------------------------- per-stage graph backend


def after_backend_init(result, self, *args, **kwargs):
    self.keep_raw_graph = False
    # Bumped whenever this backend's graphs are captured or released, so a step
    # graph that spliced them as children can tell that it is stale.
    self.graph_epoch = 0


def after_resolve_decode_backend(result, *args, **kwargs):
    from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
        FullCudaGraphBackend,
    )

    # The step graph splices the decode-path graphs into its own.
    if isinstance(result, FullCudaGraphBackend):
        result.keep_raw_graph = True


def _keep_graph_class(base):
    class KeepGraph(base):
        def __new__(cls, keep_graph=False, *args, **kwargs):
            return super().__new__(cls, True, *args, **kwargs)

    return KeepGraph


def around_capture_one(original, self, shape_key, *args, **kwargs):
    if not getattr(self, "keep_raw_graph", False):
        original(self, shape_key, *args, **kwargs)
        self.graph_epoch = getattr(self, "graph_epoch", 0) + 1
        return
    base = torch.cuda.CUDAGraph
    torch.cuda.CUDAGraph = _keep_graph_class(base)
    try:
        original(self, shape_key, *args, **kwargs)
    finally:
        torch.cuda.CUDAGraph = base
    self._graphs[shape_key].instantiate()
    self.graph_epoch += 1


def around_replay(original, self, shape_key, *args, **kwargs):
    from sgl_nvidia_mega.spec_step.child_graph_capture import (
        current_child_graph_capture,
    )

    capture = current_child_graph_capture()
    if capture is not None:
        capture.add_graph(self._graphs[shape_key], owner=self)
        return self._outputs[shape_key]
    return original(self, shape_key, *args, **kwargs)


def after_cleanup(result, self, *args, **kwargs):
    self.graph_epoch = getattr(self, "graph_epoch", 0) + 1


# ------------------------------------------------------------- the worker


def after_init_cuda_graphs(result, self, *args, **kwargs):
    from sgl_nvidia_mega.spec_step.runner import (
        EagleStepCudaGraphRunner,
        step_cuda_graph_unsupported_reason,
    )

    self.step_cuda_graph_runner = None
    if not self._hosts_draft:
        return
    reason = step_cuda_graph_unsupported_reason(self.server_args)
    if reason is not None:
        logger.warning(
            "%s disabled: %s. Falling back to the per-stage CUDA graphs.",
            switches.STEP_GRAPH,
            reason,
        )
        return
    self.step_cuda_graph_runner = EagleStepCudaGraphRunner(self)


def around_forward_batch_generation(
    original, self, batch, on_publish=None, grammar_barrier=None, pp_proxy_tensors=None
):
    runner = getattr(self, "step_cuda_graph_runner", None)
    if runner is not None and not (
        batch.forward_mode.is_extend() or batch.is_extend_in_batch
    ):
        self.activate_step_by_batch(batch.seq_lens.shape[0])
        out = runner.try_replay(batch, on_publish, grammar_barrier, pp_proxy_tensors)
        if out is not None:
            return out
    return original(self, batch, on_publish, grammar_barrier, pp_proxy_tensors)


def after_add_speculative(result, self, state, *args, **kwargs):
    worker = getattr(self.scheduler, "draft_worker", None)
    runner = getattr(worker, "step_cuda_graph_runner", None)
    if runner is not None:
        state["spec_step_cuda_graph"] = runner.stats()


# -------------------------------------------------------------- the relay


_fused_relay_logged = set()


def _log_fused_relay_once(part: str) -> None:
    if part not in _fused_relay_logged:
        _fused_relay_logged.add(part)
        logger.info("%s on: %s", switches.FUSED_RELAY, part)


def _resolve_single_request_views(self, batch, indices) -> bool:
    """A single request whose future indices are the batch's own
    req_pool_indices resolves its draft inputs as row views of the relay
    buffers instead of gathering copies. The rows are written back only by this
    request's next stash, which follows its forward on the same stream."""
    from sglang.srt.managers import overlap_utils

    if (
        not indices.is_cuda
        or indices.shape[0] != 1
        or indices is not batch.req_pool_indices
        or len(batch.reqs) != 1
        or overlap_utils._DEBUG_ASSERT
    ):
        return False
    draft_input = batch.spec_info
    row = batch.reqs[0].kv.req_pool_idx
    draft_input.bonus_tokens = self.output_tokens_buf.narrow(0, row, 1)
    if self.need_topk:
        draft_input.topk_p = self.topk_p_buf.narrow(0, row, 1)
        draft_input.topk_index = self.topk_index_buf.narrow(0, row, 1)
        if self.need_hidden_states:
            draft_input.hidden_states = self.hidden_states_buf.narrow(0, row, 1)
        if self.draft_probs_buf is not None:
            draft_input.draft_probs = self.draft_probs_buf.narrow(0, row, 1)
    elif self.need_hidden_states:
        draft_input.hidden_states = self.hidden_states_buf.narrow(0, row, 1)
    if draft_input.future_dsa_topk_indices_available:
        assert self.dsa_topk_indices_buf is not None
        draft_input.dsa_topk_indices = self.dsa_topk_indices_buf.narrow(0, row, 1)
    else:
        draft_input.dsa_topk_indices = None
    _log_fused_relay_once("views")
    return True


def around_resolve_spec_extras(original, self, batch, *args, **kwargs):
    draft_input = batch.spec_info
    if not self.spec_algo.is_ngram() and draft_input is not None:
        indices = getattr(draft_input, "future_indices", None)
        if (
            indices is not None
            and indices.shape[0] != 0
            and _resolve_single_request_views(self, batch, indices)
        ):
            return None
    return original(self, batch, *args, **kwargs)


def around_stash(original, self, future_indices, payload, *args, **kwargs):
    indices = future_indices
    if indices.shape[0] == 0 or self.spec_algo.is_ngram() or not indices.is_cuda:
        return original(self, future_indices, payload, *args, **kwargs)
    from sgl_nvidia_mega.spec_step.ops import scatter_spec_extras

    self._maybe_init_forward_bufs(payload)
    self._maybe_init_dsa_topk_indices_buf(payload)
    # Every relay row in one scatter launch (same rows, same values; fields the
    # payload leaves out keep the rows they already hold).
    pairs = [(self.output_tokens_buf, payload.bonus_tokens)]
    if self.need_topk and payload.topk_p is not None:
        assert payload.topk_index is not None
        pairs.append((self.topk_p_buf, payload.topk_p))
        pairs.append((self.topk_index_buf, payload.topk_index))
    if self.need_hidden_states and payload.hidden_states is not None:
        pairs.append((self.hidden_states_buf, payload.hidden_states))
    if self.draft_probs_buf is not None and payload.draft_probs is not None:
        pairs.append((self.draft_probs_buf, payload.draft_probs))
    if self.dsa_topk_indices_buf is not None and payload.dsa_topk_indices is not None:
        pairs.append((self.dsa_topk_indices_buf, payload.dsa_topk_indices))
    scatter_spec_extras(indices, pairs)
    _log_fused_relay_once("stash")
    return None


# ------------------------------------------------- lifted SGLang functions


def _rebind(module, names):
    """The ``lifted`` functions ``names`` with ``module``'s globals, added to it;
    returns them by name."""
    from sgl_nvidia_mega.spec_step import lifted

    g = vars(module)
    out = {}
    for name in names:
        value = getattr(lifted, name)
        if isinstance(value, types.FunctionType):
            value = types.FunctionType(
                value.__code__, g, value.__name__, value.__defaults__, value.__closure__
            )
            value.__kwdefaults__ = getattr(lifted, name).__kwdefaults__
        elif isinstance(value, set):
            value = set()
        out[name] = value
        if name not in ("eagle_sample", "_draft_extend_for_decode"):
            g[name] = value
    g.setdefault("_spec_switches", switches)
    return out


def _eagle_sample():
    from sglang.srt.speculative import eagle_utils

    return _rebind(
        eagle_utils,
        (
            "_chunked_verify_sampling_logged",
            "_log_chunked_verify_sampling_once",
            "_alloc_verify_outputs",
            "_broadcast_verify_outputs",
            "eagle_sample",
        ),
    )["eagle_sample"]


def _draft_extend_for_decode():
    from sglang.srt.speculative import eagle_worker_v2

    from sgl_nvidia_mega.spec_step.ops import draft_extend_tail

    vars(eagle_worker_v2).setdefault("draft_extend_tail", draft_extend_tail)
    return _rebind(
        eagle_worker_v2,
        (
            "_draft_tail_logged",
            "_log_draft_tail_once",
            "_fused_draft_extend_tail_eligible",
            "_draft_extend_for_decode",
        ),
    )["_draft_extend_for_decode"]


def _lazy(factory):
    """A REPLACE hook that builds its function at the first call, so that
    registering it imports nothing."""
    built = []

    def hook(*args, **kwargs):
        if not built:
            built.append(factory())
        return built[0](*args, **kwargs)

    return hook


def _sampling(*args, **kwargs):
    from sgl_nvidia_mega.spec_step.ops import tree_speculative_sampling_target_only

    return tree_speculative_sampling_target_only(*args, **kwargs)


# ------------------------------------------------------------- registration


def hooks():
    """(target, kind, hook) for the switches that are on."""
    from sgl_nvidia_mega.spec_step.child_graph_capture import around_override

    out = []
    if switches.enabled(switches.STEP_GRAPH):
        out += [
            (f"{_BACKEND}.__init__", "AFTER", after_backend_init),
            (
                "sglang.srt.model_executor.runner_backend.utils.resolve_decode_backend",
                "AFTER",
                after_resolve_decode_backend,
            ),
            (f"{_BACKEND}.capture_one", "AROUND", around_capture_one),
            (f"{_BACKEND}.replay", "AROUND", around_replay),
            (f"{_BACKEND}.cleanup", "AFTER", after_cleanup),
            (f"{_WORKER}.init_cuda_graphs", "AFTER", after_init_cuda_graphs),
            (
                f"{_WORKER}.forward_batch_generation",
                "AROUND",
                around_forward_batch_generation,
            ),
            (
                "sglang.srt.managers.scheduler_components.internal_state."
                "SchedulerInternalStateController._add_speculative",
                "AFTER",
                after_add_speculative,
            ),
            (
                f"{_SPEC_UTILS}.generate_simulated_accept_index",
                "AROUND",
                around_override("spec_simulated_accept_index"),
            ),
            (
                f"{_SPEC_UTILS}.GrammarTree.from_device",
                "AROUND",
                around_override("spec_grammar_tree_from_device"),
            ),
            (
                f"{_SPEC_UTILS}.build_grammar_vocab_mask",
                "AROUND",
                around_override("spec_grammar_vocab_mask"),
            ),
        ]
    if switches.enabled(switches.FUSED_RELAY):
        out += [
            (f"{_FUTURE_MAP}._resolve_spec_extras", "AROUND", around_resolve_spec_extras),
            (f"{_FUTURE_MAP}.stash", "AROUND", around_stash),
        ]
    if any(
        switches.enabled(name)
        for name in (
            switches.PACKED_VERIFY_BROADCAST,
            switches.LEAN_SAMPLED_VERIFY,
            switches.CHUNKED_VERIFY_SAMPLING,
        )
    ):
        out += [
            ("sglang.srt.speculative.eagle_utils.eagle_sample", "REPLACE", _lazy(_eagle_sample)),
            (
                "sglang.kernels.ops.speculative.sampling.tree_speculative_sampling_target_only",
                "REPLACE",
                _sampling,
            ),
        ]
    if switches.enabled(switches.FUSED_DRAFT_EXTEND_TAIL):
        out.append(
            (
                f"{_DRAFT_WORKER}._draft_extend_for_decode",
                "REPLACE",
                _lazy(_draft_extend_for_decode),
            )
        )
    return out


def register() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    installed = hooks()
    for target, kind, hook in installed:
        HookRegistry.register(target, hook, HookType[kind])
    if installed:
        logger.info(
            "sgl-nvidia-mega: speculative step hooks on (%s)",
            ", ".join(name for name in switches.ALL if switches.enabled(name)),
        )

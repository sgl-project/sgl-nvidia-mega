# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""SGLang's real init order for a DSpark server, as a CPU fixture.

``scheduler.init_model_worker`` builds a DSpark server in this order (STAGES):

- ``install``: the target model loads and the plugin builds (``install_megakernel``);
- ``draft_worker``: the DSpark worker's ``__init__``: ``attach_shared_modules`` gives
  the draft the target's embedding and head, then the decode phases resolve;
- ``memory_pools``: ``init_memory_pools`` -> ``alloc_memory_pool`` ->
  ``_init_post_memory_pool_components`` (KV pools, req_to_token, ngram / Engram
  history);
- ``attention_backends``: ``init_all_attention_backends`` (each runner's
  ``attn_backend``, target first, then draft);
- ``target_graphs``: the target runner's decode graphs (``decode_cuda_graph_runner``);
- ``draft_graphs``: the draft runner's graphs, with the draft sampler factory;
- ``overlap``: ``Scheduler.init_overlap``, after the model workers (the FutureMap).

``SchedulerInit`` hands out a worker whose objects gain those attributes only at their
stage. Reading one earlier raises ``EarlyRead``, which is not an ``AttributeError``:
``getattr(obj, name, default)`` and ``hasattr`` do not swallow it, so a slot that would
silently go inert on an early read fails too. A slot that reads worker or runner state
adds a ``Scenario`` to test_deepseek_v41_init_order.py.
"""

from __future__ import annotations

import contextlib
import dataclasses
from types import SimpleNamespace
from typing import Callable, NamedTuple
from unittest import mock

import torch

STAGES = (
    "install",
    "draft_worker",
    "memory_pools",
    "attention_backends",
    "target_graphs",
    "draft_graphs",
    "overlap",
)
GAMMA = 5
CAPTURE_BS = (1, 2, 4, 8)
REQ_SLOTS = 16
HISTORY_WIDTH = 4
VOCAB, HIDDEN, TP = 128, 64, 4


class EarlyRead(AssertionError):
    """An attribute read before the init stage that sets it."""


class Staged:
    """An object whose ``later`` attributes appear at their stage; reading one before
    raises ``EarlyRead``, and unknown names raise ``AttributeError`` as usual."""

    def __init__(self, init: SchedulerInit, name: str, **now):
        object.__setattr__(self, "_init", init)
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_later", {})
        for key, value in now.items():
            object.__setattr__(self, key, value)
        init.objects.append(self)

    def later(self, stage: str, **attrs) -> Staged:
        assert stage in STAGES, stage
        for key, value in attrs.items():
            self._later[key] = (stage, value)
        return self

    def __getattr__(self, name: str):
        # reached only when the normal lookup fails
        pending = object.__getattribute__(self, "_later")
        if name in pending:
            raise EarlyRead(
                f"{self._name}.{name} is set at stage {pending[name][0]!r}, read during "
                f"{self._init.stage!r}"
            )
        raise AttributeError(name)

    def _arrive(self, stage: str) -> None:
        for key, (at, value) in list(self._later.items()):
            if at == stage:
                object.__setattr__(self, key, value)
                del self._later[key]


def _attn_backend(pool):
    """A DSV4 attention backend (isinstance holds) over ``pool``, without its init."""
    from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend

    backend = DeepseekV4AttnBackend.__new__(DeepseekV4AttnBackend)
    backend.__dict__.update(
        token_to_kv_pool=pool,
        trtllm_attn=False,
        online_c128_mtp=SimpleNamespace(enabled=lambda: False),
    )
    return backend


class Scenario(NamedTuple):
    """A slot's reads in init order: with ``slots`` requested (``init.slots``), each
    ``(stage, step)`` runs once that stage has completed; a step asserts what it needs,
    e.g. that the slot engaged."""

    name: str
    slots: frozenset[str]
    steps: tuple[tuple[str, Callable[[SchedulerInit], None]], ...]


class _Model(torch.nn.Module):
    """A loaded model: an nn.Module carrying the given attributes."""

    def __init__(self, **attrs):
        super().__init__()
        for k, v in attrs.items():
            setattr(self, k, v)


@dataclasses.dataclass
class _InputBuffers:
    """A graph runner's input buffers: a dataclass, as SGLang's DecodeInputBuffers."""

    input_ids: torch.Tensor
    positions: torch.Tensor
    out_cache_loc: torch.Tensor
    seq_lens: torch.Tensor

    @classmethod
    def new(cls, rows: int) -> _InputBuffers:
        return cls(
            input_ids=torch.full((rows,), -1, dtype=torch.int64),
            positions=torch.zeros(rows, dtype=torch.int64),
            out_cache_loc=torch.zeros(rows, dtype=torch.int64),
            seq_lens=torch.zeros(rows, dtype=torch.int32),
        )


class SchedulerInit:
    """The objects of one DSpark server, at ``stage``; ``advance`` moves forward."""

    def __init__(self, *, gamma: int = GAMMA, capture_bs=CAPTURE_BS):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.speculative.dspark_components.dspark_verify import (
            CommitInjectCtx,
            DsparkVerifyEpilogue,
        )

        self.stage = STAGES[0]
        self.slots: frozenset[str] = frozenset()
        self.objects: list[Staged] = []
        self.capture_bs = tuple(capture_bs)
        self.gamma = gamma
        width = gamma + 1
        rows = max(capture_bs) * width
        # what the steps record for later steps of the same scenario
        self.state = SimpleNamespace()
        self.tp_sync = SimpleNamespace(sync=lambda site, values: values)
        # this rank's LM-head shard: the target's, and the draft's once attached
        self.lm_head = SimpleNamespace(
            weight=torch.zeros(VOCAB // TP, HIDDEN, dtype=torch.bfloat16)
        )
        # loaded at install, with SGLang's head on its logits processor; an nn.Module
        # as SGLang's, so its ``buffers`` is a method, not an input-buffer dataclass
        self.target_model = _Model(
            lm_head=self.lm_head,
            logits_processor=SimpleNamespace(
                use_fp32_lm_head=False,
                _compute_lm_head=lambda x, lm_head, bias=None: x @ lm_head.weight.t(),
            ),
        )

        # the KV pools alloc_memory_pool gives each runner (and its attention backend)
        self.target_pool = SimpleNamespace(request_window=None)
        self.draft_pool = SimpleNamespace(
            request_window=None,
            swa_req_ring_size=None,
            full_to_swa_index_mapping=torch.arange(64, dtype=torch.int32),
            set_swa_key_buffer_radix_fused_norm_rope=None,
        )

        self.target_runner = (
            # ModelRunner.__init__ sets model_config, is_draft_worker and lora_manager
            Staged(
                self,
                "model_runner",
                is_draft_worker=False,
                lora_manager=None,
                model=self.target_model,
                model_config=SimpleNamespace(vocab_size=VOCAB, hidden_size=HIDDEN),
            )
            .later(
                "memory_pools",
                token_to_kv_pool=self.target_pool,
                req_to_token_pool=SimpleNamespace(
                    req_to_token=torch.zeros(REQ_SLOTS, 64, dtype=torch.int32)
                ),
                ngram_embedding_manager=SimpleNamespace(
                    engram_hasher=SimpleNamespace(
                        history=torch.zeros(
                            REQ_SLOTS + 1, HISTORY_WIDTH, dtype=torch.int32
                        )
                    )
                ),
            )
            .later("attention_backends", attn_backend=_attn_backend(self.target_pool))
            .later(
                "target_graphs",
                decode_cuda_graph_runner=SimpleNamespace(
                    buffers=_InputBuffers.new(rows)
                ),
            )
        )
        draft_backend = _attn_backend(self.draft_pool)
        self.draft_runner = Staged(
            self, "draft_model_runner", is_draft_worker=True, lora_manager=None
        )
        # the draft's verify graph runner, built at the draft graphs' capture with the
        # fields its __init__ sets
        self.draft_graph_runner = SimpleNamespace(
            capture_forward_mode=ForwardMode.TARGET_VERIFY,
            attn_backend=draft_backend,
            enable_two_batch_overlap=False,
            enable_pdmux=False,
            model_runner=self.draft_runner,
            buffers=_InputBuffers.new(rows),
        )
        self.draft_runner.later("memory_pools", token_to_kv_pool=self.draft_pool).later(
            "attention_backends", attn_backend=draft_backend
        ).later("draft_graphs", decode_cuda_graph_runner=self.draft_graph_runner)
        # the draft's built megakernel serves widths of gamma rows a request
        widths = tuple(gamma * bs for bs in range(1, max(capture_bs) + 1))
        # the draft model's own load-time attributes; the attached modules come later
        self.draft_model = Staged(
            self,
            "draft_model",
            _megakernel=SimpleNamespace(widths=widths),
            write_target_hidden_kv=lambda **kw: None,
            # stage 0's main projection: an MXFP8 linear stand-in (weight + swizzled scales)
            stages=[
                SimpleNamespace(
                    main_proj=SimpleNamespace(
                        weight=torch.zeros(HIDDEN, HIDDEN, dtype=torch.float8_e4m3fn),
                        weight_scale_inv_swizzled=torch.zeros(
                            HIDDEN * HIDDEN // 32, dtype=torch.uint8
                        ),
                    )
                ),
                object(),
            ],
            sample_from_anchor=True,
            uses_own_vocab_modules=False,
        ).later("draft_worker", embed_tokens=SimpleNamespace(), lm_head=self.lm_head)
        # SGLang's epilogue is built in the worker's __init__, before the phases
        epilogue = DsparkVerifyEpilogue(
            max_bs=max(capture_bs),
            verify_num_draft_tokens=width,
            device="cpu",
            tp_sync=self.tp_sync,
            commit_ctx=CommitInjectCtx(
                draft_model=self.draft_model,
                block_pos_offsets=torch.arange(width),
                resolve_pool=lambda: self.draft_runner.token_to_kv_pool,
                # as SGLang's: the target runner's req_to_token (memory_pools)
                resolve_req_to_token=lambda: (
                    self.target_runner.req_to_token_pool.req_to_token
                ),
            ),
        )
        self.worker = Staged(
            self,
            "worker",
            gamma=gamma,
            verify_num_draft_tokens=width,
            _tp_sync=self.tp_sync,
            _verify_epilogue=epilogue,
            draft_model=self.draft_model,
            model_runner=self.target_runner,
            draft_model_runner=self.draft_runner,
        )

    @contextlib.contextmanager
    def runtime_context(self):
        """SGLang's runtime context, published before any model loads: the parallel
        groups, the spec args and the graph config. Steps may patch over it."""
        group = SimpleNamespace(
            rank_in_group=0, world_size=TP, device_group=None, cpu_group=None
        )
        parallel = SimpleNamespace(tp_group=group, attn_cp_size=1, tp_size=TP)
        spec = SimpleNamespace(speculative_num_draft_tokens=self.gamma + 1)
        graph = SimpleNamespace(
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(bs=list(self.capture_bs))
            )
        )
        with (
            mock.patch(
                "sglang.srt.runtime_context.get_parallel", return_value=parallel
            ),
            mock.patch("sglang.srt.runtime_context.get_spec", return_value=spec),
            mock.patch(
                "sglang.srt.runtime_context.get_exec",
                return_value=SimpleNamespace(graph=graph),
            ),
        ):
            yield

    def advance(self, stage: str) -> None:
        """Complete every stage up to and including ``stage``."""
        target = STAGES.index(stage)
        assert target >= STAGES.index(self.stage), (self.stage, stage)
        for at in STAGES[STAGES.index(self.stage) + 1 : target + 1]:
            for obj in self.objects:
                obj._arrive(at)
        self.stage = stage

    def run(self, scenario: Scenario) -> None:
        """Walk the stages in order and run each step after its stage completes."""
        self.slots = scenario.slots
        with self.runtime_context():
            for stage in STAGES:
                self.advance(stage)
                for at, step in scenario.steps:
                    if at == stage:
                        step(self)

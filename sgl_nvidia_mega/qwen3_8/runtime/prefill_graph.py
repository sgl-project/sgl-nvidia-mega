# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from SGLang (Apache-2.0, SGLang Team): python/sglang/srt/models/qwen4_exp.py
# (_breakable_prepare_ple_batch, _breakable_ple_prefetch, _breakable_ple_forward,
# _breakable_commit_ple_batch), python/sglang/srt/layers/radix_attention.py
# (RadixAttention._eager_attention) and python/sglang/srt/layers/radix_linear_attention.py
# (_linear_attention_with_output_impl).
"""The graph breaks of the executor's prefill body in SGLang's breakable prefill CUDA graph.

With ``MegakernelSpec.prefill_graph`` SGLang hands the layer module's forward to
``Qwen3_8Executor.forward_body``, and a breakable prefill graph captures it once per
token bucket. A captured segment may depend on the bucket's shapes only, so each op that
reads the batch's host state (lengths, prefix lengths, request counts) is a graph break
(``eager_on_graph``): the capture records the call, holding its tensors weakly, and every
replay runs it eagerly between the segment replays on the live batch
(``get_tc_piecewise_forward_context().forward_batch``), writing buffers its caller
allocated in the segment before it and zeroing the bucket rows past the batch. The
capture's engagement record names each op ``break``; a break records what it ran.

* ``qsa``: a QSA layer's indexer and attention (``qsa_prefill.attend``) on the batch's
  real rows; the QKV / gate before and the output gate / o_proj after stay in segments.
* ``gdn_core``: a GDN layer's conv, gates, FlashInfer core and track copies
  (``gdn_prefill.core_rows``), for a bucket without static GDN tables; with them the
  core runs in the segment (``gdn_prefill._core_in_graph``).
* ``ple_prepare``, ``ple_prefetch``, ``ple_add`` and ``ple_commit``: the PLE batch, the
  next PLE layer's table prefetch, the decomposed layers' embedding add and the commit
  (``ple``), sharing SGLang's ``_BreakablePLEBatch``, which the prepare refills. The
  stage boundaries' own add (S0) is SGLang's break.
"""

from __future__ import annotations

import torch
from sglang.kernels.plugin_kernels.nvidia.spec import Step
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
)

from sgl_nvidia_mega.qwen3_8.runtime import gdn_prefill, ple, qsa_prefill
from sgl_nvidia_mega.qwen3_8.runtime.state import StepState

BREAK = "break"
_STEP = Step(mode="prefill", rows=0)

SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py": ("Step",),
    "srt/model_executor/runner_backend_utils/breakable_cuda_graph/__init__.py": (
        "eager_on_graph",
    ),
    "srt/model_executor/runner_backend_utils/tc_piecewise_cuda_graph/__init__.py": (
        "get_tc_piecewise_forward_context",
    ),
    "srt/model_executor/forward_batch_info.py": ("global_num_token_non_padded_cpu",),
    "srt/model_executor/forward_context.py": ("get_attn_backend",),
    "srt/layers/attention/graph_utils.py": ("attention_input_scope",),
    "srt/layers/attention/qsa/glue.py": ("resolve_qsa_sparse_backend",),
    "srt/layers/radix_attention.py": ("tp_q_head_num", "v_head_dim"),
    "srt/layers/radix_linear_attention.py": ("num_v_heads", "head_v_dim"),
    "srt/models/qwen4_exp.py": ("_BreakablePLEBatch", "batch", "forward_idle"),
}


def _live(executor) -> StepState:
    """The live batch's step state, for the step functions a break runs."""
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
        get_tc_piecewise_forward_context,
    )

    return StepState(
        lm=executor.lm,
        step=_STEP,
        forward_batch=get_tc_piecewise_forward_context().forward_batch,
        positions=None,
        input_ids=None,
        slots=executor.slots,
        binding=executor.binding,
        engagement=executor.engagement,
    )


def _real_rows(fb, rows: int) -> int:
    """The batch's tokens among the bucket's rows, as SGLang's eager breaks count them."""
    return min(int(fb.global_num_token_non_padded_cpu), rows)


# -- QSA ----------------------------------------------------------------------------


def qsa(executor, layer, hidden_states, q, k, v, st) -> torch.Tensor:
    """``qsa_prefill.attend`` as a break; returns the [rows, heads * head_dim]
    attention output (zero past the batch) the output gate takes."""
    attn = layer.attn
    output = v.new_empty((q.shape[0], attn.tp_q_head_num * attn.v_head_dim))
    st.ran("qsa", BREAK)
    _qsa(executor, layer, hidden_states, st.positions, q, k, v, output)
    return output


@eager_on_graph(True)
def _qsa(executor, layer, hidden_states, positions, q, k, v, output):
    from sglang.srt.layers.attention.graph_utils import attention_input_scope
    from sglang.srt.layers.attention.qsa.glue import resolve_qsa_sparse_backend
    from sglang.srt.model_executor.forward_context import get_attn_backend

    st = _live(executor)
    fb = st.fb
    n = _real_rows(fb, output.shape[0])
    if n:
        st.positions = positions[..., :n]
        backend = resolve_qsa_sparse_backend(get_attn_backend())
        # _eager_attention's scope: the real rows' cache slots and positions, and the
        # output rows the union kernel writes in place
        with attention_input_scope(fb, output, n, n, {}):
            out = qsa_prefill.attend(
                executor.qsa_prefill,
                layer,
                hidden_states[:n],
                q[:n],
                k[:n],
                v[:n],
                backend,
                st,
            )
        if out.data_ptr() != output.data_ptr():
            output[:n].copy_(out)
    output[n:].zero_()


# -- GDN ----------------------------------------------------------------------------


def gdn_core(executor, gdn, step, mixed_qkv, a, b) -> tuple:
    """``gdn_prefill.core_rows`` as a break; returns its [1, rows, HV, V] output (zero
    past the batch) and the impl."""
    attn = gdn.attn
    output = mixed_qkv.new_empty(
        (1, mixed_qkv.shape[0], attn.num_v_heads, attn.head_v_dim)
    )
    _gdn_core(executor, gdn, step, mixed_qkv, a, b, output)
    return output, BREAK


@eager_on_graph(True)
def _gdn_core(executor, gdn, step, mixed_qkv, a, b, output):
    st = _live(executor)
    n = _real_rows(st.fb, output.shape[1])
    if n:
        impl = gdn_prefill.core_rows(
            executor.gdn_prefill,
            step,
            gdn,
            st.fb,
            mixed_qkv[:n],
            a[:n],
            b[:n],
            output[:, :n],
        )
        st.ran("gdn.core", impl)
    output[:, n:].zero_()


# -- PLE ----------------------------------------------------------------------------


def ple_prepare(executor, input_ids, st):
    """``ple.prepare`` as a break; returns the holder every PLE break and SGLang's
    stage boundaries read the batch's ``_PLEBatch`` from."""
    from sglang.srt.models.qwen4_exp import _BreakablePLEBatch

    holder = _BreakablePLEBatch()
    st.ran("ple.prepare", BREAK)
    _ple_prepare(executor, input_ids, holder)
    return holder


@eager_on_graph(True)
def _ple_prepare(executor, input_ids, holder):
    holder.batch = ple.prepare(_live(executor), input_ids, options=executor.ple_options)


def ple_prefetch(executor, layer, st) -> None:
    """``ple.prefetch`` of ``layer`` (the next PLE layer) as a break."""
    st.ran("ple.prefetch", BREAK)
    _ple_prefetch(executor, layer, st.ple_batch)


@eager_on_graph(True)
def _ple_prefetch(executor, layer, holder):
    ple.prefetch(layer, holder.batch, _live(executor), options=executor.ple_options)


def ple_add(executor, layer, residual, st) -> torch.Tensor:
    """``ple.add`` as a break: ``residual`` plus the layer's PLE embedding."""
    output = torch.empty_like(residual)
    st.ran("ple.add", BREAK)
    _ple_add(executor, layer, residual, st.ple_batch, output)
    return output


@eager_on_graph(True)
def _ple_add(executor, layer, residual, holder, output):
    st = _live(executor)
    fb, batch, options = st.fb, holder.batch, executor.ple_options
    if batch is None:
        # an idle batch replayed as a dummy extend: join the collective, add nothing
        layer.ple.forward_idle(fb)
        output.copy_(residual)
    elif ple.served(options, layer, batch, fb, in_break=True):
        embedding = ple.forward(options, layer.ple, residual, fb, batch, st)
        torch.add(residual, embedding, out=output)
    else:
        st.ran("ple.add", ple.STOCK)
        embedding = layer.ple(hidden_states=residual, forward_batch=fb, batch=batch)
        torch.add(residual, embedding, out=output)


def ple_commit(executor, st) -> None:
    """``ple.commit`` as a break."""
    st.ran("ple.commit", BREAK)
    _ple_commit(executor, st.ple_batch)


@eager_on_graph(True)
def _ple_commit(executor, holder):
    ple.commit(holder.batch, _live(executor))

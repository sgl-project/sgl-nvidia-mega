# SPDX-License-Identifier: Apache-2.0
# From SGLang python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py
# .
"""FlashInfer's chunked GDN prefill as the stock ``FlashInferGDNKernel.extend`` runs
it, with the q / k l2norm on the prefill launches and the state rows gathered and
written back in one launch each.

Byte-identical to the source's ``FlashInferGDNKernel.extend`` body.
Changed: it is a function of the stock kernel object (``self`` is ``kernel``: its
``_prefill_fn``, ``use_state_pool`` and ``_prefill_needs_fp32_state``), the
environment reads are arguments (SGLANG_OPT_GDN_PREFILL_FUSED_STATE_COPY is
``fused_rows``, decided by the caller with ``gdn_state_rows_supported``; the l2norm
switches are ``l2norm_retile`` and ``qk_one_launch``), and the q / k / v preparation is
this package's copy. FlashInfer's prefill function is the stock kernel's, so
``gdn_prefill_tmem_dealloc.install()`` (FlashInfer #8854) reaches it too.

``extend_in_graph`` is the source's ``FlashInferGDNKernel.extend_in_graph`` (the core of
a GDN extend captured in a breakable prefill CUDA graph, on the bucket's static
``GDNFlashInferGraphTables``), with this package's q / k / v preparation and the
bucket rows past the batch neutralized before the non-CP call as well.
"""

from typing import Optional

import torch
from sglang.srt.layers.attention.linear.kernels.state_pool_io import (
    gather_pool_states,
    scatter_pool_states,
)

from sgl_nvidia_mega.qwen3_8.kernels.ops.attention.linear.gdn_prefill_l2norm import (
    gdn_prefill_qkv_prepare_fwd,
)
from sgl_nvidia_mega.qwen3_8.kernels.ops.mamba.gdn_prefill_state_glue import (
    gdn_gather_state_rows,
    gdn_scatter_state_rows,
)

SGLANG_READS = {
    "srt/layers/attention/linear/kernels/state_pool_io.py": (
        "gather_pool_states",
        "scatter_pool_states",
    ),
    "srt/layers/attention/linear/kernels/gdn_flashinfer.py": (
        "_prefill_fn",
        "use_state_pool",
        "_prefill_needs_fp32_state",
    ),
    "srt/layers/attention/linear/gdn_prefill_graph.py": (
        "cp_single",
        "cp_cu_seqlens",
        "cp_state_indices",
        "cp_checkpoint_cu",
        "ncp_cu_seqlens",
        "ncp_state_indices",
        "ncp_checkpoint_cu",
    ),
}

# Reads and mirrors that need an SGLang change newer than the unit job's pin: the
# contract test skips each, naming the change, only where the installed SGLang lacks it.
SGLANG_REQUIRES = {
    "srt/layers/attention/linear/gdn_prefill_graph.py:cp_checkpoint_cu": "sglang#50157",
    "srt/layers/attention/linear/gdn_prefill_graph.py:cp_cu_seqlens": "sglang#50157",
    "srt/layers/attention/linear/gdn_prefill_graph.py:cp_single": "sglang#50157",
    "srt/layers/attention/linear/gdn_prefill_graph.py:cp_state_indices": "sglang#50157",
    "srt/layers/attention/linear/gdn_prefill_graph.py:ncp_checkpoint_cu": "sglang#50157",
    "srt/layers/attention/linear/gdn_prefill_graph.py:ncp_cu_seqlens": "sglang#50157",
    "srt/layers/attention/linear/gdn_prefill_graph.py:ncp_state_indices": "sglang#50157",
}


def extend(
    kernel,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_checkpoint_cu_starts: Optional[torch.Tensor] = None,
    num_state_checkpoints: int = 0,
    state_checkpoint_every_n_tokens: int = 0,
    fused_rows: bool = False,
    l2norm_retile: bool = False,
    qk_one_launch: bool = False,
    **kwargs,
) -> tuple:
    """``FlashInferGDNKernel.extend`` of the stock ``kernel`` (its FlashInfer prefill
    function and state-pool flags). fused_rows: gather and write back the state rows
    in one launch each (the caller checked gdn_state_rows_supported); l2norm_retile and
    qk_one_launch: gdn_prefill_qkv_prepare_fwd's."""
    total_seq_len = q.shape[1]
    num_v_heads = v.shape[2]
    head_v_dim = v.shape[3]

    q_fi, k_fi, v_fi = gdn_prefill_qkv_prepare_fwd(
        q[0], k[0], v[0], retile=l2norm_retile, qk_one_launch=qk_one_launch
    )

    output = kwargs.get("output")
    output_fi = None
    if output is not None:
        expected_output_shape = (
            1,
            total_seq_len,
            num_v_heads,
            head_v_dim,
        )
        if tuple(output.shape) != expected_output_shape:
            raise ValueError(
                "FlashInfer GDN prefill output shape mismatch: "
                f"expected {expected_output_shape}, got {tuple(output.shape)}"
            )
        if output.dtype != v.dtype or output.device != v.device:
            raise ValueError(
                "FlashInfer GDN prefill output must match v dtype/device, "
                f"got output=({output.dtype}, {output.device}) and "
                f"v=({v.dtype}, {v.device})"
            )
        output_fi = output[0]
        if not output_fi.is_contiguous():
            raise ValueError("FlashInfer GDN prefill output must be contiguous")

    # g (alpha) and beta: [1, seq, HV] -> [seq, HV], float32 for FlashInfer
    alpha_fi = torch.exp(g[0].to(torch.float32))
    beta_fi = beta[0].to(torch.float32)

    if fused_rows:
        initial_state_fi = gdn_gather_state_rows(ssm_states, cache_indices)
    else:
        ssm_cache_indices, live_rows, initial_state_fi = gather_pool_states(
            ssm_states, cache_indices
        )
    if not kernel.use_state_pool or kernel._prefill_needs_fp32_state:
        initial_state_fi = initial_state_fi.to(torch.float32)
    cu_seqlens = query_start_loc.to(torch.int64)

    # Keep final state and checkpoints in the same kernel state dtype.
    output_state_fi = torch.empty_like(initial_state_fi)
    state_checkpoints = (
        initial_state_fi.new_empty((num_state_checkpoints, *initial_state_fi.shape[1:]))
        if num_state_checkpoints > 0
        else None
    )
    output_fi, output_state_fi = kernel._prefill_fn(
        q=q_fi,
        k=k_fi,
        v=v_fi,
        g=alpha_fi,
        beta=beta_fi,
        scale=None,
        initial_state=initial_state_fi,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
        output=output_fi,
        output_state=output_state_fi,
        state_checkpoints=state_checkpoints,
        checkpoint_cu_starts=state_checkpoint_cu_starts,
        checkpoint_every_n_tokens=state_checkpoint_every_n_tokens,
    )

    # Write back state to pool
    if fused_rows:
        gdn_scatter_state_rows(ssm_states, cache_indices, output_state_fi)
    else:
        scatter_pool_states(ssm_states, ssm_cache_indices, live_rows, output_state_fi)

    # Output: [seq, HV, V] -> [1, seq, HV, V]
    core_attn_out = output_fi.view(1, total_seq_len, num_v_heads, head_v_dim)

    # Match Triton's [1, checkpoints, H, V, K] intermediate-state layout.
    h = state_checkpoints.unsqueeze(0) if state_checkpoints is not None else None
    return core_attn_out, None, h


def extend_in_graph(
    kernel,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    ssm_states: torch.Tensor,
    tables,
    output: torch.Tensor,
    checkpoint_rows: int,
    checkpoint_every_n_tokens: int,
    l2norm_retile: bool = False,
    qk_one_launch: bool = False,
) -> Optional[torch.Tensor]:
    """The prefill core of a captured GDN extend on the bucket's static tables
    (``GDNFlashInferGraphTables``): the CP kernel takes a one-sequence batch (if
    ``tables.cp_single``), the non-CP kernel any other, and both update the pool in
    place by slot. ``output`` is [tokens, HV, V]; returns the [1, checkpoint_rows, HV,
    V, K] state checkpoints, if any."""
    from sgl_nvidia_mega.qwen3_8.kernels.ops.mamba.gdn_prefill_cp_tail import (
        gdn_cp_tail_neutralize,
    )

    q_fi, k_fi, v_fi = gdn_prefill_qkv_prepare_fwd(
        q[0], k[0], v[0], retile=l2norm_retile, qk_one_launch=qk_one_launch
    )
    alpha = torch.exp(g[0].to(torch.float32))
    beta_fi = beta[0].to(torch.float32)
    common = dict(
        q=q_fi,
        k=k_fi,
        v=v_fi,
        g=alpha,
        beta=beta_fi,
        scale=None,
        initial_state=ssm_states,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        output=output,
        output_state=ssm_states,
    )
    checkpoints = (
        ssm_states.new_empty((checkpoint_rows, *ssm_states.shape[1:]))
        if checkpoint_rows > 0
        else None
    )

    def checkpoint_args(cu: torch.Tensor) -> dict:
        if checkpoints is None:
            return {}
        return dict(
            state_checkpoints=checkpoints,
            checkpoint_cu_starts=cu,
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        )

    if tables.cp_single:
        # Rows past T are bucket padding (stale graph-pool memory, possibly not finite).
        gdn_cp_tail_neutralize(q_fi, k_fi, v_fi, alpha, beta_fi, tables.cp_cu_seqlens)
        # max_seqlen bounds the CP launch grid; the bucket bounds any sequence.
        kernel._prefill_fn(
            **common,
            cu_seqlens=tables.cp_cu_seqlens,
            state_indices=tables.cp_state_indices,
            use_cp=True,
            max_seqlen=q_fi.shape[0],
            **checkpoint_args(tables.cp_checkpoint_cu),
        )
    # The non-CP table's last entry is the batch's T (0 when the CP call took it).
    gdn_cp_tail_neutralize(q_fi, k_fi, v_fi, alpha, beta_fi, tables.ncp_cu_seqlens[-2:])
    kernel._prefill_fn(
        **common,
        cu_seqlens=tables.ncp_cu_seqlens,
        state_indices=tables.ncp_state_indices,
        use_cp=False,
        **checkpoint_args(tables.ncp_checkpoint_cu),
    )
    return checkpoints.unsqueeze(0) if checkpoints is not None else None

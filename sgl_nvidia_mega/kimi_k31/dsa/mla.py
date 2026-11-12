# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DCP1 sparse MLA over physical token slots in the original page64 pool."""

from functools import cache

import torch


@cache
def _workspace(device):
    from flashinfer.utils import get_trtllm_gen_multi_ctas_kv_counter_bytes

    from sgl_nvidia_mega.kimi_k31.k3.compat import TRTLLM_KV_COUNTER_ROWS

    sms = torch.cuda.get_device_properties(device).multi_processor_count
    size = get_trtllm_gen_multi_ctas_kv_counter_bytes(TRTLLM_KV_COUNTER_ROWS, 16, sms)
    return (
        torch.zeros(128 << 20, dtype=torch.int8, device=device),
        torch.zeros(size, dtype=torch.uint8, device=device),
    )


class SparseMla:
    """``heads`` local MLA heads (TP8 12, TP16 6), padded to 16 for trtllm-gen."""

    def __init__(self, device, max_context, rows=1, heads=12):
        if not 1 <= heads <= 16:
            raise ValueError(f"DCP1 sparse MLA pads 1-16 local heads to 16, got {heads}")
        self.max_context = max_context
        self.rows, self.heads = rows, heads
        self.workspace, self.counter = _workspace(device)
        self.query = torch.zeros(
            (rows, 1, 16, 576), dtype=torch.float8_e4m3fn, device=device
        )
        self.output = torch.empty((rows, 1, 16, 512), dtype=torch.bfloat16, device=device)

    def run(self, query, cache, selected, lengths, scales, out):
        self.prepare_query(query)
        self.run_core(cache, selected, lengths, scales)
        out.copy_(self.output[:, :, : self.heads].view(self.rows, self.heads, 512))
        return out

    def prepare_query(self, query):
        self.query[:, :, : self.heads].copy_(
            query.view(torch.float8_e4m3fn).view(self.rows, 1, self.heads, 576)
        )

    def run_core(self, cache, selected, lengths, scales):
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

        trtllm_batch_decode_with_kv_cache_mla(
            query=self.query,
            kv_cache=cache.view(torch.float8_e4m3fn).unsqueeze(1),
            workspace_buffer=self.workspace,
            qk_nope_head_dim=128,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            block_tables=selected.unsqueeze(1),
            seq_lens=lengths,
            max_seq_len=self.max_context,
            sparse_mla_top_k=2048,
            out=self.output,
            bmm1_scale=scales[0],
            bmm2_scale=scales[1],
            enable_pdl=True,
            backend="trtllm-gen",
            multi_ctas_kv_counter_buffer=self.counter,
        )
        return self.output

# SPDX-License-Identifier: Apache-2.0
"""DCP1 sparse MLA over physical token slots in the original page64 pool."""

from functools import cache

import torch

from ..verify_contract import Q8_REQUEST_BATCHES
from .scratch import row_buffer

# Names read from SGLang. A contract test checks these before GPU use.
SGLANG_READS = {
    "srt/layers/attention/dsa/dsa_dcp.py": (
        "trtllm_rows_per_call",
        "dcp_trtllm_sparse_attention",
    ),
}


@cache
def _workspace(device):
    from flashinfer.utils import get_trtllm_gen_multi_ctas_kv_counter_bytes

    sms = torch.cuda.get_device_properties(device).multi_processor_count
    size = get_trtllm_gen_multi_ctas_kv_counter_bytes(64, 16, sms)
    return (
        torch.zeros(128 << 20, dtype=torch.int8, device=device),
        torch.zeros(size, dtype=torch.uint8, device=device),
    )


class SparseMla:
    """``heads`` real query heads; the trtllm-gen sparse core runs 16 (the rest stay zero)."""

    def __init__(self, device, max_context, rows=1, *, heads, scratch=None):
        self.max_context = max_context
        self.rows, self.heads = rows, heads
        self.workspace, self.counter = _workspace(device)
        # zero: the core runs 16 heads and only the real ones are written
        self.query = row_buffer(
            scratch,
            "sparse_mla.query",
            rows,
            (1, 16, 576),
            torch.float8_e4m3fn,
            device,
            zero=True,
        )
        self.output = row_buffer(
            scratch, "sparse_mla.output", rows, (1, 16, 512), torch.bfloat16, device
        )

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


# dcp_trtllm_sparse_attention splits its rows by trtllm_rows_per_call: 128 MiB of scratch
# plus 2 KiB per (row, head); sized for verify's widest Q8 width at 128 heads so every width
# runs as one PDL launch (+24 MiB per device at 96 rows).
_CP_MAX_ROWS = 8 * max(Q8_REQUEST_BATCHES)
_CP_WORKSPACE_BYTES = (128 << 20) + _CP_MAX_ROWS * 128 * 2048


@cache
def _cp_workspace(device):
    from flashinfer.utils import get_trtllm_gen_multi_ctas_kv_counter_bytes

    sms = torch.cuda.get_device_properties(device).multi_processor_count
    size = get_trtllm_gen_multi_ctas_kv_counter_bytes(64, 128, sms)
    return (
        torch.zeros(_CP_WORKSPACE_BYTES, dtype=torch.int8, device=device),
        torch.zeros(size, dtype=torch.uint8, device=device),
    )


class CpSparseMla:
    """DCP sparse MLA: every rank runs all ``heads`` (TP x per-rank heads, padded to the
    trtllm-gen 128) over the selected keys it owns, rows compacted to the front of
    ``local_rows`` with ``counts`` of them, and returns BF16 rows with their base-2 LSE
    for the cross-rank merge (stock ``dcp_trtllm_sparse_attention``; ranks owning no
    selected key get the zero-row fixup)."""

    def __init__(self, device, max_context, rows, *, heads, scratch=None):
        from sglang.srt.layers.attention.dsa.dsa_dcp import trtllm_rows_per_call

        if heads > 128:
            raise ValueError(f"DCP sparse MLA pads to 128 heads, got {heads}")
        if rows > trtllm_rows_per_call(workspace_bytes=_CP_WORKSPACE_BYTES, heads=128):
            raise ValueError(
                f"DCP sparse MLA workspace runs at most {_CP_MAX_ROWS} rows per launch"
            )
        self.max_context, self.rows, self.heads = max_context, rows, heads
        self.workspace, self.counter = _cp_workspace(device)
        self.query = row_buffer(
            scratch,
            "cp_sparse_mla.query",
            rows,
            (1, 128, 576),
            torch.float8_e4m3fn,
            device,
            zero=True,
        )
        self.query_starts = torch.arange(rows + 1, dtype=torch.int32, device=device)

    def run_core(self, cache, local_rows, counts, scales):
        from sglang.srt.layers.attention.dsa.dsa_dcp import dcp_trtllm_sparse_attention

        return dcp_trtllm_sparse_attention(
            query=self.query,
            block_tables=local_rows.unsqueeze(1),
            seq_lens=counts,
            cu_seqlens_q=self.query_starts,
            kv_cache=cache.view(torch.float8_e4m3fn).unsqueeze(1),
            workspace_buffer=self.workspace,
            qk_nope_head_dim=128,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            max_seq_len=self.max_context,
            sparse_mla_top_k=2048,
            bmm1_scale=scales[0],
            bmm2_scale=scales[1],
            enable_pdl=True,
            backend="trtllm-gen",
            multi_ctas_kv_counter_buffer=self.counter,
        )

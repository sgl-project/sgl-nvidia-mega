# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The megakernel on SGLang's TP8 one-request verify step: TRT-LLM's ``ServedMegakernel`` with SGLang's buffers.

``SgMegakernel`` keeps the vendored ``prepare`` (configuration agreement, build, workspaces, weight packs) and
``_build`` (one parameter block per replaced layer for the copy it always runs on) and changes what is TRT-LLM's:

* workspaces: ``comm.HopWorkspace`` / ``comm.RlWorkspace`` on SGLang's TP group in place of ``McastGPUBuffer`` (swapped
  in for ``prepare`` only, as the vendored ``_packed_once`` swaps the packers);
* the KV cache: the stages are built on a placeholder pool, then each layer's KV fields are replaced by SGLang's
  buffers, page table and TMA maps (``kv.LayerKv``);
* the per-forward decision: ``begin_sg`` on SGLang's ``ForwardBatch`` (one target-verify request of 8 rows on the
  trtllm_mha verify graph's metadata), called from layer 0; ``run_layer`` launches one replaced layer from that layer's
  own forward and returns ``(stream, next attention input)``, the decoder layer's contract.

The layers are given as facades (``facades``) that expose the attributes the vendored host code reads.
"""

from __future__ import annotations

import contextlib
import os
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ..vendor.trtllm_ab93b78.megakernel import layer_mk, rl, serving
from . import comm as sg_comm
from . import kv, stages

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/layers/attention/trtllm_mha_backend.py": (
        "TRTLLMHAAttnBackend",
        "target_verify_metadata",
        "forward_metadata",
        "page_size",
        "is_ragged_verify",
        "max_seq_len_q",
        "page_table",
        "swa_page_table",
        "cache_seqlens_int32",
        "swa_out_cache_loc",
    ),
    "srt/model_executor/forward_context.py": (
        "get_attn_backend",
        "get_req_to_token_pool",
    ),
}

#: The served build of ``serving.DEFAULT_MODE`` reading SGLang's paged KV cache (the vendored library's ``_sg`` modes).
MODE = serving.DEFAULT_MODE[: -len("_s4")] + "_sg_s4"
#: Its check twin (rl 3, watchdog on) for the bring-up.
CHECK_MODE = MODE.replace("lean_", "check_", 1).replace("_rl1_", "_rl3_", 1)
ROWS = serving.ROWS
HIDDEN = serving.HIDDEN


@contextlib.contextmanager
def sg_workspaces():
    """The vendored code's workspace classes, swapped for ours while ``LayerKernels`` and the rl workspace are made."""
    saved = (layer_mk.HopWorkspace, rl.RlWorkspace)
    layer_mk.HopWorkspace, rl.RlWorkspace = sg_comm.HopWorkspace, sg_comm.RlWorkspace
    try:
        yield
    finally:
        layer_mk.HopWorkspace, rl.RlWorkspace = saved


def mapping_of(tp_rank: int, local_rank: int) -> SimpleNamespace:
    """The fields of TRT-LLM's ``Mapping`` the vendored host code reads, for SGLang's TP8 = EP8 group."""
    return SimpleNamespace(
        tp_rank=int(tp_rank),
        local_rank=int(local_rank),
        tp_size=8,
        moe_ep_size=8,
        pp_size=1,
    )


class SgMegakernel(serving.ServedMegakernel):
    """``serving.ServedMegakernel`` in the served configuration (``serving.install``'s arguments) on SGLang."""

    def __init__(
        self,
        layers: Sequence[object],
        tp_rank: int,
        local_rank: int,
        comm: sg_comm.TpComm,
        mode: str = MODE,
        compare: int = 0,
        compare_out: Optional[str] = None,
    ):
        super().__init__(
            layers,
            mapping_of(tp_rank, local_rank),
            comm,
            mode=mode,
            early_x=not compare,  # compare mode: every launch reads its x from its own input
            full_one=True,
            i2b_flags=serving.DEFAULT_I2B_FLAGS,
            final_norm=None,
            edges=False,
            cap_moe=int(serving.DEFAULT_CAP_MOE, 0),
            w1_plan=serving.DEFAULT_W1_PLAN,
            fa_flags=serving.FA_FLAGS,
            fa_entry_pages=0,
            fa_gates_early=False,
            fa_o_replicas=1,
            fa_late_first=serving.FA_LATE_FIRST,
            fh_plan=serving.FH_PLAN,
        )
        self.position_of = {index: pos for pos, index in enumerate(self.indices)}
        self.layer_kv: Dict[
            int, kv.LayerKv
        ] = {}  # by layer index; the pool does not move
        self.sg_bindings: Dict[tuple, Tuple[Dict[int, object], torch.Tensor]] = {}
        self.positions32: Optional[torch.Tensor] = None
        self.last_refusal = (
            "no forward yet"  # why the last forward ran stock ("" = it ran the kernel)
        )
        self._placeholder = None
        # compare mode (``begin_compare`` / ``compare_layer``)
        self.compare_forwards = int(compare)
        self.compare_out = compare_out
        self.compared = 0
        self.cmp: Optional[SimpleNamespace] = None
        self.cmp_stats: Dict[
            int, Dict[str, float]
        ] = {}  # per layer, the worst over the compared forwards
        self.cmp_rows: Dict[int, Dict[str, float]] = {}  # per layer, this forward's
        self.cmp_forwards: List[
            Dict[str, object]
        ] = []  # each compared forward's first position and rows
        self.cmp_warmups = 0  # verify forwards skipped as warm-ups (``_bind_compare``)
        self.stage_check = (
            bool(compare) and mode == CHECK_MODE
        )  # the check build's stage outputs (``stages``)

    # ------------------------------------------------------------------ install
    def prepare(self) -> None:
        with sg_workspaces():
            super().prepare()
        kv.check_build(self.kernels.lib)
        kv.check_build(self.kernels.full_lib)
        self._placeholder = kv.placeholder(self.kernels.ws.arena.device)

    # ------------------------------------------------------------------ bindings
    def _layer_pool(self, layer, md) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._placeholder

    def _layer_kv(self, index: int, md) -> kv.LayerKv:
        """The layer's buffers and its pool's page table of this binding (the maps are made once per layer)."""
        sliding = self.sliding[index]
        mapping = getattr(
            md.pool, "layers_mapping", None
        )  # SWAKVPool: layer -> (index in its pool, sliding)
        assert mapping is not None and bool(mapping[index][1]) == bool(sliding), (
            index,
            sliding,
        )
        table = md.swa_page_table if sliding else md.page_table
        lkv = self.layer_kv.get(index)
        k_buf, v_buf = md.pool.get_key_buffer(index), md.pool.get_value_buffer(index)
        if (
            lkv is None
            or lkv.k.data_ptr() != k_buf.data_ptr()
            or lkv.v.data_ptr() != v_buf.data_ptr()
        ):
            lkv = kv.LayerKv(self.kernels.lib, k_buf, v_buf, table, sliding)
            self.layer_kv[index] = lkv
        else:
            lkv.table = table
        return lkv

    def _build(self, md, positions: torch.Tensor, spec_metadata) -> Dict[int, object]:
        blocks = super()._build(
            md, positions, None
        )  # the DFlash taps stay SGLang's (the kernel's capture off)
        k = self.kernels
        for index in self.indices:
            entry = blocks[index]
            if isinstance(
                entry, dict
            ):  # a one-launch full layer, or a BF16-expert layer (as ``_rl_apply``)
                mk, layer_blocks = entry["mk"], entry.get("one", entry.get("swa"))
            else:
                mk, layer_blocks = k.swa_mk, entry
            rl.apply(mk, layer_blocks, self._layer_kv(index, md).fields())
        blocks["_md"] = (
            md  # the binding's tensors stay alive with its blocks (as ``begin``)
        )
        return blocks

    # ------------------------------------------------------------------ per forward
    def begin_sg(
        self,
        forward_batch,
        positions: torch.Tensor,
        stream: torch.Tensor,
        attn_input: torch.Tensor,
    ) -> bool:
        """Whether this forward runs layers 1-60 as the kernel (``last_refusal`` says why not); on True the RoPE
        positions are copied (int32) for the launches. Host-only and the same on every rank: it depends on the batch
        shape, the attention metadata object and the bindings."""
        self.current = None
        self.last_refusal = self._bind(forward_batch, positions, stream, attn_input)
        return not self.last_refusal

    def _checks(
        self,
        forward_batch,
        positions: torch.Tensor,
        stream: torch.Tensor,
        attn_input: torch.Tensor,
    ):
        """``(why not, backend, metadata)`` for a verify forward of one request on trtllm_mha ("" = it can run)."""
        from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend
        from sglang.srt.model_executor.forward_context import get_attn_backend

        if self.kernels is None:
            return "not built", None, None
        if self.kernels.ws.plan_pos != 0:
            return (
                f"the copy plan is at launch {self.kernels.ws.plan_pos}, not at a forward's start",
                None,
                None,
            )
        mode = forward_batch.forward_mode
        if not mode.is_target_verify() or forward_batch.batch_size != 1:
            return f"{mode.name} batch of {forward_batch.batch_size}", None, None
        if (
            positions.numel() != ROWS
            or tuple(stream.shape) != (ROWS, HIDDEN)
            or stream.dtype != torch.float32
        ):
            return (
                f"stream {tuple(stream.shape)} {stream.dtype} with {positions.numel()} positions",
                None,
                None,
            )
        if (
            attn_input is None
            or attn_input.dtype != torch.bfloat16
            or tuple(attn_input.shape) != (ROWS, HIDDEN)
        ):
            return "the attention input is not bf16 [8, 8192]", None, None
        backend = get_attn_backend()
        if not isinstance(backend, TRTLLMHAAttnBackend):
            return f"attention backend {type(backend).__name__}", None, None
        md = backend.forward_metadata
        if md is None:
            return "no attention metadata", None, None
        if md.is_ragged_verify or md.max_seq_len_q != ROWS:
            return (
                f"ragged {md.is_ragged_verify}, {md.max_seq_len_q} rows per request",
                None,
                None,
            )
        if md.swa_page_table is None or backend.page_size != kv.PAGE:
            return (
                f"page size {backend.page_size}, SWA page table {md.swa_page_table is not None}",
                None,
                None,
            )
        return "", backend, md

    def _bind(
        self,
        forward_batch,
        positions: torch.Tensor,
        stream: torch.Tensor,
        attn_input: torch.Tensor,
    ) -> str:
        reason, backend, md = self._checks(forward_batch, positions, stream, attn_input)
        if reason:
            return reason
        # Only the verify graph's static metadata (its warm-ups bind it before the capture): an eager verify forward
        # makes its own tensors every time and would use up the bindings.
        if md is not backend.target_verify_metadata.get(1):
            return "not the bs-1 verify graph's metadata"
        page_table, swa_page_table, seq = (
            md.page_table,
            md.swa_page_table,
            md.cache_seqlens_int32,
        )
        key = (
            id(md),
            page_table.data_ptr(),
            swa_page_table.data_ptr(),
            seq.data_ptr(),
            positions.data_ptr(),
        )
        binding = self.sg_bindings.get(key)
        if binding is None:
            if (
                torch.cuda.is_current_stream_capturing()
                or len(self.sg_bindings) >= serving.MAX_BINDINGS
            ):
                self.refused_bindings += 1
                return "a binding first seen inside a capture or past the binding limit"
            pos32 = torch.zeros(ROWS, dtype=torch.int32, device=positions.device)
            sg_md = SimpleNamespace(
                kv_lens_cuda_runtime=seq,
                page_table=page_table[:1],
                swa_page_table=swa_page_table[:1],
                pool=backend.token_to_kv_pool,
                metadata=md,
            )
            binding = (self._build(sg_md, pos32, None), pos32)
            self.sg_bindings[key] = binding
            serving._log(
                f"parameter blocks for verify binding {len(self.sg_bindings)}",
                self.rank,
            )
        self.current, self.positions32 = binding
        self.positions32.copy_(positions.view(-1))
        return ""

    def _launch(
        self,
        index: int,
        blocks,
        x: torch.Tensor,
        res: torch.Tensor,
        pdl: bool = True,
        qkv=None,
    ) -> None:
        k = self.kernels
        if (
            qkv is not None
        ):  # a sliding-window layer with the check build's QKV output (stages)
            from ..vendor.trtllm_ab93b78.megakernel import library

            layer = blocks[index]
            mk, blk = (
                (layer["mk"], layer["swa"])
                if isinstance(layer, dict)
                else (k.swa_mk, layer)
            )
            mk.launch(
                blk,
                library.KernelIo(x=x.contiguous(), res=res.contiguous(), qkv=qkv),
                pdl=pdl,
            )
        elif self.sliding[index]:
            k.run_swa(blocks[index], x.contiguous(), res.contiguous(), pdl=pdl)
        else:
            k.run_full(blocks[index], x.contiguous(), res.contiguous(), pdl=pdl)

    def run_layer(
        self, index: int, x: torch.Tensor, res: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Layer ``index``'s launch: ``(the stream after it, the next layer's attention input)``."""
        blocks = self.current
        assert blocks is not None, "begin_sg first"
        self._launch(index, blocks, x, res)
        pos = self.position_of[index]
        if index == self.indices[-1]:
            self.current = None
            self.forwards += 1
        return self.res_out[pos], self.x_out[pos]

    # ------------------------------------------------------------------ compare mode (bring-up)
    def begin_compare(
        self,
        forward_batch,
        positions: torch.Tensor,
        stream: torch.Tensor,
        attn_input: torch.Tensor,
    ):
        """Compare mode (eager forwards: ``--cuda-graph-backend-decode disabled``): whether this verify forward checks
        layers 1-60 against the stock layers (``compare_layer``). The forward's page tables, KV lengths and positions
        are copied into the static compare binding, built on the first compared forward."""
        self.current = None
        self.last_refusal = self._bind_compare(
            forward_batch, positions, stream, attn_input
        )
        return not self.last_refusal

    def _bind_compare(self, forward_batch, positions, stream, attn_input) -> str:
        from sglang.srt.model_executor.forward_context import get_req_to_token_pool

        if torch.cuda.is_current_stream_capturing():
            return "compare mode runs eager forwards only (--cuda-graph-backend-decode disabled)"
        reason, backend, md = self._checks(forward_batch, positions, stream, attn_input)
        if reason:
            return reason
        if self.compared >= self.compare_forwards:
            return f"compared {self.compared} forwards"
        # The boot's dummy verify forwards (autotune, warm-ups) put all 8 rows at position 0, and there the stock
        # attention returns zeros: only one request's 8 consecutive rows are compared (a host sync; eager anyway).
        rows = positions.view(-1).tolist()
        if rows != list(range(rows[0], rows[0] + ROWS)):
            self.cmp_warmups += 1
            return f"positions {rows}: a warm-up forward"
        if md.swa_out_cache_loc is None:
            return "no SWA write locations"
        if self.cmp is None:
            dev = positions.device
            pages = -(
                -get_req_to_token_pool().req_to_token.shape[1] // kv.PAGE
            )  # the longest request's pages
            c = SimpleNamespace(
                pool=backend.token_to_kv_pool,
                pt=torch.zeros(1, pages, dtype=torch.int32, device=dev),
                swa_pt=torch.zeros(1, pages, dtype=torch.int32, device=dev),
                seq=torch.zeros(1, dtype=torch.int32, device=dev),
                pos32=torch.zeros(ROWS, dtype=torch.int32, device=dev),
            )
            sg_md = SimpleNamespace(
                kv_lens_cuda_runtime=c.seq,
                page_table=c.pt,
                swa_page_table=c.swa_pt,
                pool=c.pool,
                metadata=None,
            )
            if self.stage_check:
                c.bufs = stages.Buffers(dev)
                c.dumps = stages.dump_layers(self.indices, self.sliding)
                with stages.check_outputs(self.kernels, c.bufs):
                    c.blocks = self._build(sg_md, c.pos32, None)
            else:
                c.blocks = self._build(sg_md, c.pos32, None)
            self.cmp = c
            serving._log(f"compare binding: {pages} pages per table", self.rank)
        c = self.cmp
        for dst, src in ((c.pt, md.page_table), (c.swa_pt, md.swa_page_table)):
            n = src.shape[1]
            assert n <= dst.shape[1], (n, dst.shape)
            dst[:, :n].copy_(src[:1, :n])
        c.seq.copy_(md.cache_seqlens_int32[:1])
        c.pos32.copy_(positions.view(-1))
        c.loc = {
            True: md.swa_out_cache_loc[:ROWS],
            False: forward_batch.out_cache_loc[:ROWS],
        }
        c.first = rows[0]
        self.current = c.blocks
        return ""

    def compare_layer(
        self,
        index: int,
        original,
        positions,
        stream: torch.Tensor,
        attn_input: torch.Tensor,
        batch,
    ):
        """Compare mode: the stock layer, then the layer's launch (no PDL) on the same inputs. Both outputs and the
        step's 8 KV rows of K and V are compared, and with the check build each stage (``stages``); the stock outputs
        go on, so the forward stays the stock model's."""
        c = self.cmp
        x_in, res_in = (
            attn_input.clone(),
            stream.clone(),
        )  # the stock layer may update its inputs in place
        taps = (
            stages.StockTaps(self.layers[index].sgl_layer)
            if self.stage_check
            else contextlib.nullcontext()
        )
        with taps:
            out = original(positions, stream, attn_input, batch)
        loc = c.loc[bool(self.sliding[index])]
        k_buf = c.pool.get_key_buffer(index).view(torch.uint8)
        v_buf = c.pool.get_value_buffer(index).view(torch.uint8)
        kv_ref = torch.stack([k_buf[loc], v_buf[loc]])
        dumping = (
            self.stage_check
            and self.compared == 0
            and index in c.dumps
            and bool(self.compare_out)
        )
        inputs = (
            dict(x=x_in.clone(), res=res_in.clone()) if dumping else None
        )  # the launch may update res in place
        if self.stage_check:
            c.bufs.poison()
        qkv = c.bufs.qkv if self.stage_check and self.sliding[index] else None
        self._launch(index, c.blocks, x_in, res_in, pdl=False, qkv=qkv)
        pos = self.position_of[index]
        kv_mk = torch.stack([k_buf[loc], v_buf[loc]])
        got = (self.res_out[pos], self.x_out[pos])
        extra = (
            self._stage_errors(index, taps, out, got, inputs)
            if self.stage_check
            else None
        )
        self._record(index, out, got, kv_ref, kv_mk, extra)
        if index == self.indices[-1]:
            self.current = None
            self.compared += 1
            self._report()
        return out

    def _stage_errors(self, index: int, taps, out, got, inputs) -> Dict[str, float]:
        """The stage check of layer ``index``'s launch (``stages.errors``); on the first compared forward, the dump
        layers' tensors go to ``compare_out`` (``stages_rank<r>_layer<i>.pt``)."""
        c, k = self.cmp, self.kernels
        layer = self.layers[index].sgl_layer
        sliding = bool(self.sliding[index])
        rows = stages.errors(layer, taps, c.bufs, k.res1, out[0], self.rank, sliding)
        if inputs is not None:
            seq_len = int(c.seq[0])
            kv_rows = stages.kv_rows(
                c.pool, index, c.swa_pt if sliding else c.pt, seq_len, sliding
            )
            stages.dump(
                os.path.join(
                    self.compare_out, f"stages_rank{self.rank}_layer{index}.pt"
                ),
                facade=self.layers[index],
                next_facade=self.layers[index + 1],
                layer=layer,
                taps=taps,
                bufs=c.bufs,
                res1=k.res1,
                inputs=inputs,
                stock_out=out,
                kernel_out=got,
                kv=dict(kv_rows, positions=c.pos32, seq_len=seq_len),
                rank=self.rank,
                index=index,
                sliding=sliding,
            )
        return rows

    def _record(
        self,
        index: int,
        ref,
        got,
        kv_ref: torch.Tensor,
        kv_got: torch.Tensor,
        extra=None,
    ) -> None:
        def err(a: torch.Tensor, b: torch.Tensor):
            a, b = a.float(), b.float()
            d = (a - b).abs()
            return (
                float(d.max()),
                float(d.norm() / a.norm().clamp_min(1e-30)),
                int((~torch.isfinite(b)).sum()),
            )

        s_max, s_rel, s_bad = err(ref[0], got[0])
        x_max, x_rel, x_bad = err(ref[1], got[1])
        kv_max, _, kv_bad = err(
            kv_ref.view(torch.float8_e4m3fn), kv_got.view(torch.float8_e4m3fn)
        )
        row = dict(
            stream_max=s_max,
            stream_rel=s_rel,
            x_max=x_max,
            x_rel=x_rel,
            kv_bytes=float((kv_ref != kv_got).sum()),
            kv_max=kv_max,
            nonfinite=float(s_bad + x_bad + kv_bad),
        )
        row.update(extra or {})
        row = {
            key: value if value == value else float("inf") for key, value in row.items()
        }  # a NaN error: infinite
        self.cmp_rows[index] = row
        st = self.cmp_stats.setdefault(index, {})
        for key, value in row.items():
            st[key] = max(st.get(key, float("-inf")), value)

    def _report(self) -> None:
        """Per compared forward: its worst layer of each measure (rank 0); after the last one, every layer's worst row
        over the forwards on every rank, and the rank's JSON files in ``compare_out``."""
        rows, self.cmp_rows = self.cmp_rows, {}
        self.cmp_forwards.append(
            dict(
                first_position=self.cmp.first,
                rows={str(i): r for i, r in sorted(rows.items())},
            )
        )
        if self.rank == 0:
            keys = dict.fromkeys(
                key for row in rows.values() for key in row
            )  # a layer's row has its own stages
            worst = {
                key: max(
                    (i for i in rows if key in rows[i]), key=lambda i: rows[i][key]
                )
                for key in keys
            }
            serving._log(
                f"compare forward {self.compared}/{self.compare_forwards} (rows at {self.cmp.first}-"
                f"{self.cmp.first + ROWS - 1}, {self.cmp_warmups} warm-up forwards skipped): "
                + ", ".join(
                    f"{key} {rows[i][key]:.3g} (layer {i})" for key, i in worst.items()
                ),
                self.rank,
            )
        if self.compared < self.compare_forwards:
            return
        stats = self.cmp_stats
        for index in sorted(stats):
            serving._log(
                f"compare layer {index}: "
                + ", ".join(f"{k} {v:.3g}" for k, v in stats[index].items()),
                self.rank,
                all_ranks=True,
            )
        if self.compare_out:
            import json
            import os

            os.makedirs(self.compare_out, exist_ok=True)
            with open(
                os.path.join(self.compare_out, f"compare_rank{self.rank}.json"), "w"
            ) as f:
                json.dump(
                    {str(i): row for i, row in sorted(stats.items())}, f, indent=1
                )
            with open(
                os.path.join(self.compare_out, f"forwards_rank{self.rank}.json"), "w"
            ) as f:
                json.dump(self.cmp_forwards, f, indent=1)

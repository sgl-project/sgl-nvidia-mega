# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import torch

from . import target as launch

SGLANG_READS = {
    "srt/distributed/parallel_state.py": ("get_tp_group",),
    "srt/distributed/communication_op.py": ("tensor_model_parallel_all_gather",),
    "srt/model_executor/forward_mode.py|srt/model_executor/forward_batch_info.py": (
        "ForwardMode",
        "DRAFT_EXTEND_V2",
    ),
    "srt/speculative/eagle_utils.py": ("TreeMaskMode",),
    "srt/speculative/eagle_worker_common.py": ("build_tree_kernel_efficient",),
    "srt/speculative/eagle_worker_v2.py": (
        "EagleDraftWorker",
        "_draft_extend_for_decode",
        "_draft_extend_for_prefill",
        "draft",
        "prepare_for_draft_extend",
    ),
}

COPIES = 3
ARENA_WORDS = 617440
COUNTER_WORDS = 800
COUNTED_CLEARS = 155
CAND_WORDS = COPIES * 8 * ((1 << 20) + 8192) * 2
DRAFT_TRANSPORT_WORDS = 3122112
VOCAB, SHARD, HIDDEN, INTER = 154880, 9680, 6144, 128
LG_W_WORDS = 16 * 38 * 512
W_ROUTER_ELEMS = 256 * HIDDEN + 2 * LG_W_WORDS
W_LM_ELEMS = SHARD * HIDDEN + (SHARD * 3456 + 4 * SHARD) // 2
PLAN = [(launch.KIND_S0, 0, False)] + [(launch.KIND_SK, k, k == 6) for k in range(1, 7)]

_DRAFT = {}


def _pack_bf16(w):
    from .target import pack_bf16

    return pack_bf16(w)


def _target_chain():
    from .target import _RUNNER

    assert len(_RUNNER) == 1, len(_RUNNER)
    return next(iter(_RUNNER.values()))


class DraftRunner:
    def __init__(self, draft_model, model_path):
        # the draft's ModelRunner (its KV pools) arrives before its CUDA graphs (bind_runner)
        self.mr, self.model, self.model_path = None, draft_model, model_path
        self.nextn = draft_model.model
        self.chain = _target_chain()
        self.rank = self.chain.rank
        self.bound = False
        self.pending = False
        self.dirty = False
        self.prepare()

    @torch.no_grad()
    def prepare(self):
        dev = torch.device("cuda", torch.cuda.current_device())
        n, L = self.nextn, self.nextn.decoder
        sa, mlp = L.self_attn, L.mlp
        assert type(mlp).__name__ == "DeepseekV2MoE", type(mlp).__name__
        heads, nope, rope_d, lora, vd = (
            sa.num_local_heads,
            sa.qk_nope_head_dim,
            sa.qk_rope_head_dim,
            sa.kv_lora_rank,
            sa.v_head_dim,
        )
        assert (heads, nope, rope_d, lora, vd) == (4, 192, 64, 512, 256)
        w = {}
        fa = sa.fused_qkv_a_proj_with_mqa.weight
        assert fa.dtype == torch.bfloat16 and tuple(fa.shape) == (2624, 6144), (
            fa.dtype,
            fa.shape,
        )
        gq = sa.q_a_layernorm.weight.double()
        ix = sa.indexer
        if hasattr(ix, "wk_weights_proj"):
            wkw = ix.wk_weights_proj.weight.to(torch.bfloat16)
            wk, wp = wkw[: ix.head_dim], wkw[ix.head_dim :]
        else:
            wk, wp = (
                ix.wk.weight.to(torch.bfloat16),
                ix.weights_proj.weight.to(torch.bfloat16),
            )
        w["wfa"] = _pack_bf16(fa.contiguous())
        w["wfa_leader"] = _pack_bf16(torch.cat([fa, wk, wp]).contiguous())
        assert tuple(w["wfa"].shape) == (164, 96, 2048) and tuple(
            w["wfa_leader"].shape
        ) == (174, 96, 2048)
        w["widx"] = _pack_bf16(
            (ix.wq_b.weight.double() * gq).to(torch.bfloat16).contiguous()
        )
        w["idx_kn_w"] = ix.k_norm.weight.float().contiguous()
        w["idx_kn_b"] = ix.k_norm.bias.float().contiguous()
        qb = sa.q_b_proj.weight.view(heads, nope + rope_d, -1)
        wkc = sa.w_kc
        kbt = wkc.transpose(1, 2) if tuple(wkc.shape) == (heads, nope, lora) else wkc
        a = (
            torch.cat(
                [torch.bmm(kbt.double(), qb[:, :nope].double()), qb[:, nope:].double()],
                dim=1,
            )
            * gq
        )
        w["wqb"] = _pack_bf16(
            a.to(torch.bfloat16).reshape(heads * (lora + rope_d), -1).contiguous()
        )
        del a
        wvc = sa.w_vc
        vb = wvc.transpose(1, 2) if tuple(wvc.shape) == (heads, lora, vd) else wvc
        wo = sa.o_proj.weight
        tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            wov = torch.cat(
                [
                    wo[:, 256 * h : 256 * h + 256].float() @ vb[h].float()
                    for h in range(4)
                ],
                dim=1,
            )
            w["wo"] = _pack_bf16(wov.to(torch.bfloat16))
            gate = mlp.gate
            r = self.rank
            wg = gate.weight.float() * L.post_attention_layernorm.weight.float().view(
                1, -1
            )
            m = torch.cat([wg @ wov, wg[:, 384 * r : 384 * r + 384]], dim=1)
            assert tuple(m.shape) == (256, 2432), m.shape
            packed = _pack_bf16(m.to(torch.bfloat16).contiguous()).reshape(-1)
            assert packed.numel() == 2 * (W_ROUTER_ELEMS - 256 * HIDDEN), packed.numel()
            buf = torch.empty((W_ROUTER_ELEMS,), dtype=torch.bfloat16, device=dev)
            buf[: 256 * HIDDEN].copy_(gate.weight.reshape(-1))
            buf[256 * HIDDEN :].view(torch.uint8).copy_(packed)
            w["w_router"] = buf
            del wg, m, packed, wov
            eh = n.eh_proj.weight
            assert eh.dtype == torch.bfloat16 and tuple(eh.shape) == (
                HIDDEN,
                2 * HIDDEN,
            ), (eh.dtype, eh.shape)
            r = self.rank
            w["eh_w"] = eh[:, 768 * r : 768 * r + 768].contiguous()
            w["hidden_rows"] = eh[384 * r : 384 * r + 384, HIDDEN:].contiguous()

            emb = self.chain.model.embed_tokens
            shard = emb.weight
            assert tuple(shard.shape) == (SHARD, HIDDEN), shard.shape
            start = getattr(
                getattr(emb, "shard_indices", None), "org_vocab_start_index", r * SHARD
            )
            assert start == r * SHARD, (start, r)
            en = n.enorm(shard.contiguous())
            we = eh[:, :HIDDEN].float()
            tt = torch.empty((SHARD, HIDDEN), dtype=torch.bfloat16, device=dev)
            for c0 in range(0, SHARD, 1210):
                tt[c0 : c0 + 1210] = (en[c0 : c0 + 1210].float() @ we.t()).to(
                    torch.bfloat16
                )
            w["token_table"] = tt
            del we, en
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32

        from sglang.srt.distributed import get_tp_group

        tp = get_tp_group()
        full = torch.empty((VOCAB, HIDDEN), dtype=torch.bfloat16, device=dev)
        torch.distributed.all_gather_into_tensor(
            full, shard.contiguous(), group=tp.device_group
        )
        w["embed"] = full

        b = gate.e_score_correction_bias.float()
        w["bias"] = (b - b.mean()).contiguous()
        w["w3w1"], w["w2"], w["w_sgu"], w["w_sdn"] = self._load_experts(dev)
        w["gamma_kv"] = sa.kv_a_layernorm.weight.contiguous()
        w["gamma_in"] = L.input_layernorm.weight.contiguous()
        w["gamma_post"] = L.post_attention_layernorm.weight.contiguous()
        w["gamma_head"] = n.shared_head.norm.weight.contiguous()
        w["gamma_hnorm"] = n.hnorm.weight.contiguous()
        w["gamma_enorm"] = n.enorm.weight.contiguous()

        lm = self.chain.lm_head.weight
        assert lm.dtype == torch.bfloat16 and tuple(lm.shape) == (SHARD, HIDDEN), (
            lm.dtype,
            lm.shape,
        )
        self.lm_bf16 = lm
        w["lm_w"] = pack_lm_w_fp4(lm.contiguous())
        for k, v in w.items():
            assert v.is_contiguous(), k
        sizes = {
            "w_router": W_ROUTER_ELEMS,
            "w3w1": 256 * 256 * HIDDEN,
            "w2": 256 * HIDDEN * INTER,
            "w_sgu": 256 * HIDDEN,
            "w_sdn": HIDDEN * INTER,
            "lm_w": W_LM_ELEMS,
            "token_table": SHARD * HIDDEN,
            "hidden_rows": 384 * HIDDEN,
            "eh_w": HIDDEN * 768,
            "embed": VOCAB * HIDDEN,
        }
        for k, v in sizes.items():
            assert w[k].numel() == v, (k, w[k].numel(), v)
        self.W = w
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def _load_experts(self, dev):
        import json

        from safetensors import safe_open

        r = self.rank
        idx = json.load(open(self.model_path + "/model.safetensors.index.json"))[
            "weight_map"
        ]
        pre = "model.layers.78.mlp."
        files = {}
        for k, f in idx.items():
            if k.startswith(pre):
                files.setdefault(f, []).append(k)
        w3w1 = torch.empty((256, 2 * INTER, HIDDEN), dtype=torch.bfloat16)
        w2 = torch.empty((256, HIDDEN, INTER), dtype=torch.bfloat16)
        sgu = torch.empty((2 * INTER, HIDDEN), dtype=torch.bfloat16)
        sdn = torch.empty((HIDDEN, INTER), dtype=torch.bfloat16)
        lo, hi = INTER * r, INTER * r + INTER
        seen = 0
        for f, keys in files.items():
            with safe_open(
                self.model_path + "/" + f, framework="pt", device="cpu"
            ) as h:
                for k in keys:
                    name = k[len(pre) :]
                    parts = name.split(".")
                    if parts[0] == "experts":
                        e, proj = int(parts[1]), parts[2]
                        sl = h.get_slice(k)
                        assert sl.get_dtype() in ("BF16", "bfloat16"), (
                            k,
                            sl.get_dtype(),
                        )
                        if proj == "up_proj":
                            w3w1[e, :INTER] = sl[lo:hi, :]
                        elif proj == "gate_proj":
                            w3w1[e, INTER:] = sl[lo:hi, :]
                        elif proj == "down_proj":
                            w2[e] = sl[:, lo:hi]
                        else:
                            continue
                        seen += 1
                    elif parts[0] == "shared_experts":
                        sl = h.get_slice(k)
                        if parts[1] == "gate_proj":
                            sgu[:INTER] = sl[lo:hi, :]
                        elif parts[1] == "up_proj":
                            sgu[INTER:] = sl[lo:hi, :]
                        elif parts[1] == "down_proj":
                            sdn[:] = sl[:, lo:hi]
                        else:
                            continue
                        seen += 1
        assert seen == 256 * 3 + 3, seen
        return (
            w3w1.to(dev).view(-1),
            w2.to(dev).view(-1),
            sgu.to(dev).view(-1),
            sdn.to(dev).view(-1),
        )

    @torch.no_grad()
    def bind(self):
        if self.bound:
            return
        from .target import TABLE_ROWS, make_mcast

        ch = self.chain
        if not ch.bound:
            ch.bind()
        dev = torch.device("cuda", torch.cuda.current_device())
        pool = self.mr.token_to_kv_pool
        attn = self.nextn.decoder.self_attn
        lid = getattr(getattr(attn, "attn_mqa", None), "layer_id", None)
        if lid is None:
            lid = getattr(attn, "layer_id", 0)
        kv = pool.get_key_buffer(lid)
        assert kv.is_contiguous() and kv.numel() * kv.element_size() % 576 == 0, (
            kv.shape,
            kv.dtype,
        )
        kb = pool.get_index_k_with_scale_buffer(lid)
        assert kb.dtype == torch.uint8 and kb.dim() == 2 and kb.shape[1] == 64 * 132, (
            kb.dtype,
            kb.shape,
        )
        stride = kb.stride(0) * kb.element_size()
        assert stride == 8448, stride
        self.pool = (kv, kb, lid)
        neg0 = lambda k: torch.full((k,), -0.0, dtype=torch.float32, device=dev)
        zero = lambda k: torch.zeros((k,), dtype=torch.int32, device=dev)
        self.ws = dict(
            arena=neg0(COPIES * ARENA_WORDS),
            part_o=neg0(262144),
            part_stats=neg0(1024),
            counters=zero(COPIES * COUNTER_WORDS),
            idx_candidates=zero(CAND_WORDS),
        )
        self.mcast, self.uc, self.mc = make_mcast(4 * DRAFT_TRANSPORT_WORDS)
        self.uc.fill_(-0.0)
        self.fence = torch.zeros((1,), dtype=torch.float32, device=dev)
        self.dirty = False
        torch.cuda.synchronize()
        from sglang.srt.distributed import get_tp_group

        torch.distributed.barrier(group=get_tp_group().cpu_group)
        bf = dict(dtype=torch.bfloat16, device=dev)
        i32 = dict(dtype=torch.int32, device=dev)
        self.p = dict(
            x=torch.zeros((8, HIDDEN), **bf),
            residual=torch.zeros((8, HIDDEN), **bf),
            residual_out=torch.zeros((8, HIDDEN), **bf),
            normalized=torch.zeros((8, HIDDEN), **bf),
            topk=torch.full((8, 2048), -1, **i32),
            draft_tokens=torch.zeros((8,), **i32),
            next_new_tokens=torch.zeros((8,), **i32),
            accepted=torch.zeros((8,), **i32),
            num_accepted=torch.ones((1,), **i32),
            length=torch.full((7,), 8, **i32),
        )
        self.dt64 = torch.zeros((1, 7), dtype=torch.int64, device=dev)
        self.sel_hidden = torch.zeros((1, HIDDEN), **bf)
        self.kernel = launch.Kernel(
            launch.artifact("draft", "entry.cubin"),
            launch.artifact("draft", "CALL.json"),
            launch.DRAFT_PARAMS,
        )
        w, p, ws = self.W, self.p, self.ws
        base = dict(
            x=p["x"].data_ptr(),
            residual=p["residual"].data_ptr(),
            residual_out=p["residual_out"].data_ptr(),
            normalized=p["normalized"].data_ptr(),
            w_fa=w["wfa"].data_ptr(),
            w_qb=w["wqb"].data_ptr(),
            w_o=w["wo"].data_ptr(),
            rope=ch.rope.data_ptr(),
            length=p["length"].data_ptr(),
            table=ch.bt.data_ptr(),
            cache=kv.data_ptr(),
            topk=p["topk"].data_ptr(),
            arena=ws["arena"].data_ptr(),
            part_o=ws["part_o"].data_ptr(),
            part_stats=ws["part_stats"].data_ptr(),
            counters=ws["counters"].data_ptr(),
            uc=self.uc.data_ptr(),
            mc=self.mc.data_ptr(),
            rank=self.rank,
            stride_rows=64,
            layer_rows=0,
            max_blocks=TABLE_ROWS,
            clear_first=0,
            idx_rope=ch.idx_rope.data_ptr(),
            idx_pool=kb.data_ptr(),
            idx_blocks=ch.bt.data_ptr(),
            idx_stride_bytes=stride,
            idx_pages=kb.shape[0],
            idx_candidates=ws["idx_candidates"].data_ptr(),
            embed_sharded=0,
            **{
                k: w[k].data_ptr()
                for k in (
                    "w_router",
                    "bias",
                    "w3w1",
                    "w2",
                    "w_sgu",
                    "w_sdn",
                    "gamma_kv",
                    "gamma_in",
                    "gamma_post",
                    "gamma_head",
                    "gamma_hnorm",
                    "lm_w",
                    "token_table",
                    "hidden_rows",
                    "wfa_leader",
                    "widx",
                    "idx_kn_w",
                    "idx_kn_b",
                    "embed",
                    "eh_w",
                    "gamma_enorm",
                )
            },
            **{
                k: p[k].data_ptr()
                for k in ("draft_tokens", "next_new_tokens", "accepted", "num_accepted")
            },
        )
        base["key_tma"] = launch.encode_key_tma(
            base["idx_pool"], launch.tma_rows(kb.shape[0], stride)
        )
        base["eh_tma"] = launch.encode_eh_tma(base["eh_w"])
        self.vals, self.calls = [], []
        for j, (kind, step, last) in enumerate(PLAN):
            v = dict(base, kind=kind, step=step, last=int(last), copy=j % COPIES)
            self.vals.append(v)
            self.calls.append(self.kernel.make_call(v))
        self.bound = True
        from . import verify_tail

        verify_tail.attach_draft(self)

    def reinit_transport(self):
        from sglang.srt.distributed import get_tp_group

        g = get_tp_group().device_group
        torch.distributed.all_reduce(self.fence, group=g)
        self.uc.fill_(-0.0)
        torch.distributed.all_reduce(self.fence, group=g)
        self.dirty = False

    def reset_ring(self):
        self.ws["arena"].fill_(-0.0)
        self.ws["counters"].zero_()
        self.ws["counters"][2 * COUNTER_WORDS : 2 * COUNTER_WORDS + 1].fill_(
            COUNTED_CLEARS
        )

    def run_s0(self, predict, accept_lens, hidden):
        p = self.p
        from . import verify_tail

        pv = verify_tail.pv_stash()
        pv = (
            pv
            if (pv is not None and pv[0] is accept_lens and pv[1] is predict)
            else None
        )
        if pv is None:
            p["accepted"].copy_(predict.view(-1)[:8].to(torch.int32))
            p["num_accepted"].copy_(accept_lens.view(-1)[:1].to(torch.int32))
        if (
            hidden.is_contiguous()
            and hidden.dtype == torch.bfloat16
            and hidden.numel() == 8 * HIDDEN
        ):
            if hidden.data_ptr() != getattr(self, "_x_bound", None):
                self.calls[0].set("x", hidden.data_ptr())
                self._x_bound = hidden.data_ptr()
        else:
            if getattr(self, "_x_bound", None) is not None:
                self.calls[0].set("x", self.p["x"].data_ptr())
                self._x_bound = None
            p["x"].copy_(hidden.view(8, HIDDEN))
        p["length"][:1].copy_(self.chain.kvl)
        if self.pending or self.dirty:
            self.reinit_transport()
        self.reset_ring()
        self.calls[0].launch(torch.cuda.current_stream().cuda_stream)
        sel = (
            pv[5] if pv is not None else torch.clamp(p["num_accepted"].long() - 1, 0, 7)
        )
        self.sel_hidden.copy_(p["normalized"].index_select(0, sel))
        self.pending = True
        return self.sel_hidden

    def run_sk(self, req_pool_indices, seq_lens):
        r2t = self.mr.req_to_token_pool.req_to_token
        if seq_lens.dtype == torch.int64 and req_pool_indices.dtype == torch.int64:
            from . import ops

            ops.draft_pages(
                r2t, req_pool_indices.view(-1)[:1], seq_lens.view(-1)[:1], self.chain.bt
            )
        else:
            pos = seq_lens.view(1).long() + torch.arange(8, device=seq_lens.device)
            pages = pos // 64
            locs = r2t[req_pool_indices.view(1).long(), pages * 64].view(-1)
            self.chain.bt.index_copy_(0, pages, (locs // 64).to(torch.int32))
        s = torch.cuda.current_stream().cuda_stream
        for c in self.calls[1:]:
            c.launch(s)
        self.dt64.copy_(self.p["draft_tokens"][:7].view(1, 7).to(torch.int64))
        self.pending = False
        return self.dt64


class _FakeLogits:
    def __init__(self, next_token_logits, hidden_states):
        self.next_token_logits = next_token_logits
        self.hidden_states = hidden_states


class _LeanFB:
    pass


def _lean_prepare_for_draft_extend(
    draft_extend_input,
    batch,
    predict,
    num_draft_tokens,
    draft_model_runner,
    cuda_graph_runner,
    *,
    return_hidden_states_before_norm,
    widened_out_cache_loc=None,
    widened_positions=None,
    iteration_ids=None,
):
    try:
        from sglang.srt.model_executor.forward_mode import ForwardMode
    except ModuleNotFoundError:  # SGLang before sgl-project/sglang#47731
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
    assert (
        getattr(draft_extend_input, "num_front_tokens", 0) == 0
        and widened_out_cache_loc is None
    )
    bs = len(batch.seq_lens)
    batch.spec_info = draft_extend_input
    batch.input_ids = predict
    if batch.seq_lens_cpu is None:
        batch.prefix_lens = batch.seq_lens.clamp(min=0).to(torch.int32)
        batch.extend_lens = torch.full(
            (bs,), num_draft_tokens, dtype=torch.int32, device=batch.seq_lens.device
        )
    else:
        batch.prefix_lens = [max(int(x), 0) for x in batch.seq_lens_cpu.tolist()]
        batch.extend_lens = [num_draft_tokens] * bs
    batch.extend_num_tokens = bs * num_draft_tokens
    batch.forward_mode = ForwardMode.DRAFT_EXTEND_V2
    return _LeanFB()


def _install_chain_tree():
    import sglang.srt.speculative.eagle_worker_common as ewc

    orig = ewc.build_tree_kernel_efficient
    if getattr(orig, "_glm5mk", False):
        return

    def bt_w(
        bonus_tokens,
        parent_list,
        top_scores_index,
        draft_tokens,
        seq_lens,
        seq_lens_sum,
        topk,
        spec_steps,
        num_verify_tokens,
        tree_mask_mode=None,
        tree_mask_buf=None,
        fill_prefix_mask=True,
    ):
        from sglang.srt.speculative.eagle_utils import TreeMaskMode

        if tree_mask_mode is None:
            tree_mask_mode = TreeMaskMode.FULL_MASK
        a = (
            bonus_tokens,
            parent_list,
            top_scores_index,
            draft_tokens,
            seq_lens,
            seq_lens_sum,
            topk,
            spec_steps,
            num_verify_tokens,
            tree_mask_mode,
            tree_mask_buf,
        )
        mode = int(tree_mask_mode)
        if not (
            topk == 1
            and spec_steps == 7
            and num_verify_tokens == 8
            and seq_lens.numel() == 1
            and mode in (0, 1)
            and draft_tokens.dtype == torch.int64
            and draft_tokens.numel() == 7
            and draft_tokens.is_contiguous()
            and bonus_tokens.numel() == 1
            and bonus_tokens.dtype in (torch.int32, torch.int64)
        ):
            return orig(*a, fill_prefix_mask=fill_prefix_mask)
        dev = seq_lens.device
        if tree_mask_buf is not None:
            tree_mask = tree_mask_buf
            if mode == 1 or fill_prefix_mask:
                tree_mask.fill_(True)
        elif mode == 1:
            tree_mask = torch.full((64,), True, dtype=torch.bool, device=dev)
        else:
            shp = (seq_lens_sum * 8 + 64,)
            tree_mask = (
                torch.full(shp, True, dtype=torch.bool, device=dev)
                if fill_prefix_mask
                else torch.empty(shp, dtype=torch.bool, device=dev)
            )
        rbuf = torch.empty((3, 1, 8), dtype=torch.long, device=dev)
        positions = torch.empty((8,), dtype=torch.long, device=dev)
        dout = torch.empty((8,), dtype=torch.long, device=dev)
        from . import ops

        ops.chain_tree(
            bonus_tokens.view(-1),
            draft_tokens,
            seq_lens.view(-1),
            tree_mask,
            mode,
            positions,
            rbuf,
            dout,
        )
        return (tree_mask, positions, rbuf[0], rbuf[1], rbuf[2], dout)

    bt_w._glm5mk = True
    ewc.build_tree_kernel_efficient = bt_w


def _host_len(batch):
    reqs = getattr(batch, "reqs", None)
    if reqs:
        r = reqs[0]
        return len(getattr(r, "origin_input_ids", None) or ()) + len(
            getattr(r, "output_ids", None) or ()
        )
    sc = getattr(batch, "seq_lens_cpu", None)
    if sc is not None and len(sc):
        return int(sc[0])
    return getattr(batch, "seq_lens_sum", None)


def _eligible_extend(worker, batch, batch_result):
    d = _DRAFT.get("runner")
    ok = (
        d is not None
        and d.chain.bound
        and not batch.forward_mode.is_idle()
        and len(batch.seq_lens) == 1
        and worker.topk == 1
        and worker.speculative_num_steps == 7
        and worker.speculative_num_draft_tokens == 8
        and not getattr(batch, "has_grammar", False)
        and batch_result.next_token_ids.numel() == 8
    )
    return ok and _host_len(batch) is not None


_ORIG_PFDE = {}


def _wrap_worker(module):
    _install_chain_tree()
    cls = module.EagleDraftWorker
    if getattr(cls, "_glm5mk", False):
        return
    orig_extend, orig_draft = cls._draft_extend_for_decode, cls.draft
    _ORIG_PFDE[id(module)] = module.prepare_for_draft_extend
    orig_prefill = cls._draft_extend_for_prefill

    def prefill_w(self, *a, **k):
        d = _DRAFT.get("runner")
        if d is not None:
            d.dirty = d.dirty or d.pending
            d.pending = False
        return orig_prefill(self, *a, **k)

    def extend_w(self, batch, batch_result, **kw):
        # kw: iteration_ids (SGLang after a4ff5634b): the plan of this iteration's
        # write window; the fused draft writes the same slots through req_to_token
        # (config_reasons refuses a pool whose ids need translating)
        d = _DRAFT.get("runner")
        if d is None or not _eligible_extend(self, batch, batch_result):
            if d is not None:
                d.dirty = d.dirty or d.pending
                d.pending = False
            return orig_extend(self, batch, batch_result, **kw)
        if not d.bound:
            d.bind()
        hidden = batch_result.logits_output.hidden_states
        sel_hidden = d.run_s0(
            batch_result.next_token_ids, batch_result.accept_lens, hidden
        )
        vocab = self.draft_runner.model_config.vocab_size
        if getattr(self, "_glm5mk_zero_logits", None) is None:
            self._glm5mk_zero_logits = {
                n: torch.zeros((n, vocab), dtype=torch.float32, device=hidden.device)
                for n in (1, 8)
            }
        zl = self._glm5mk_zero_logits
        fr, fe = self.cuda_graph_runner_for_draft_extend, self.draft_runner.forward

        class _Runner:
            def __getattr__(s, k):
                return getattr(fr, k)

            def can_run_graph(s, fb):
                return fr.can_run_graph(fb) if fr is not None else False

            def execute(s, fb, select_index):
                return _FakeLogits(zl[1], sel_hidden)

        class _Out:
            def __init__(s):
                s.logits_output = _FakeLogits(zl[8], d.p["normalized"])

        lean = (
            fr is not None
            and getattr(self.draft_runner, "canary_manager", None) is None
        )
        if lean:
            if getattr(self, "_glm5mk_lean_logits", None) is None:
                self._glm5mk_lean_logits = torch.zeros(
                    (1, 1), dtype=torch.float32, device=hidden.device
                )
            ll = self._glm5mk_lean_logits

            class _LeanRunner(_Runner):
                def can_run_graph(s, fb):
                    return True

                def execute(s, fb, select_index):
                    return _FakeLogits(ll, sel_hidden)

        seed = getattr(self, "seed_dsa_topk_from_draft_extend", None)
        try:
            self.cuda_graph_runner_for_draft_extend = (
                (_LeanRunner() if lean else _Runner()) if fr is not None else None
            )
            self.draft_runner.forward = lambda fb, *a, **k: _Out()
            if lean:
                module.prepare_for_draft_extend = _lean_prepare_for_draft_extend
                if seed is not None:
                    self.seed_dsa_topk_from_draft_extend = False
            out = orig_extend(self, batch, batch_result, **kw)
        finally:
            self.cuda_graph_runner_for_draft_extend = fr
            self.draft_runner.forward = fe
            if lean:
                module.prepare_for_draft_extend = _ORIG_PFDE[id(module)]
                if seed is not None:
                    self.seed_dsa_topk_from_draft_extend = seed
        nd = batch_result.next_draft_input
        nd._glm5mk_s0 = True
        if getattr(self, "seed_dsa_topk_from_draft_extend", False):
            nd.dsa_topk_indices = None
        return out

    def draft_w(self, batch, **kw):
        d = _DRAFT.get("runner")
        di = batch.spec_info
        ours = (
            d is not None
            and d.bound
            and d.pending
            and not batch.forward_mode.is_idle()
            and len(batch.seq_lens) == 1
            and di is not None
            and getattr(di, "hidden_states", None) is not None
            and not kw.get("with_topology", False)
        )
        if not ours:
            if (
                d is not None
                and d.pending
                and di is not None
                and getattr(di, "hidden_states", None) is not None
                and not batch.forward_mode.is_idle()
            ):
                _fallback_topk(self, d, di)
            elif d is not None:
                d.dirty = d.dirty or d.pending
                d.pending = False
            return orig_draft(self, batch, **kw)
        m = module
        args = (
            di,
            self.req_to_token_pool,
            batch,
            self.cuda_graph_runner,
            self.draft_runner,
            self.topk,
            self.speculative_num_steps,
        )
        iteration_ids = None
        if _PREPARE_PLANS_WINDOW.setdefault(id(m), _plans_window(m.prepare_for_draft)):
            # SGLang after a4ff5634b: plan the iteration's write window as its draft does
            _, _, iteration_ids = m.prepare_for_draft(
                *args,
                target_translator=self.target_worker.model_runner.runner_ids,
                num_draft_tokens=self.speculative_num_draft_tokens,
            )
            runtime = getattr(self, "draft_runtime", None)
            if runtime is not None:
                runtime.begin_draft_round(
                    runner=self.cuda_graph_runner,
                    batch=batch,
                    draft_input=di,
                    active=False,
                )
        else:
            m.prepare_for_draft(*args)
        draft_tokens = d.run_sk(batch.req_pool_indices, batch.seq_lens)
        di.topk_index = draft_tokens[:, :1]
        vi = m.build_eagle_verify_input(
            batch,
            di,
            self._topk1_parents_prealloc[:1],
            self._topk1_score_indices_prealloc[:1],
            draft_tokens,
            None,
            target_worker=self.target_worker,
            topk=self.topk,
            num_steps=self.speculative_num_steps,
            num_draft_tokens=self.speculative_num_draft_tokens,
            tree_mask_mode=self.tree_mask_mode,
            device=self.device,
        )
        vi._glm5mk_ours = True
        if iteration_ids is not None:
            # verify and draft extend write the window the draft planned (as SGLang's draft)
            vi.iteration_ids = iteration_ids
            vi.prepared_out_cache_loc = iteration_ids.write_virtual
        return vi

    (
        cls._draft_extend_for_decode,
        cls.draft,
        cls._draft_extend_for_prefill,
        cls._glm5mk,
    ) = extend_w, draft_w, prefill_w, True


_PREPARE_PLANS_WINDOW = {}


def _plans_window(prepare_for_draft):
    """Whether this SGLang's prepare_for_draft plans the write window (IterationIds)."""
    import inspect

    return "target_translator" in inspect.signature(prepare_for_draft).parameters


def _fallback_topk(worker, d, di):
    from sglang.srt.distributed import tensor_model_parallel_all_gather

    h = di.hidden_states
    logits = h.float() @ d.lm_bf16.float().t()
    full = tensor_model_parallel_all_gather(logits.contiguous(), dim=-1)
    di.topk_index = torch.argmax(full, dim=-1, keepdim=True)
    di._glm5mk_s0 = False
    d.dirty = True
    d.pending = False


def engage(draft_model, model_path):
    """At the draft's build: pack the NextN weights (the target is engaged already)
    and wrap SGLang's EAGLE v2 draft worker, which runs the fused draft at one request."""
    import sglang.srt.speculative.eagle_worker_v2 as worker

    assert type(draft_model).__name__ == "GlmMoeDsaForCausalLMNextN", type(
        draft_model
    ).__name__
    _DRAFT["runner"] = DraftRunner(draft_model, model_path)
    _wrap_worker(worker)
    return _DRAFT["runner"]


def bind_runner(model_runner):
    """Before the draft's CUDA graphs: bind the draft to the runner's KV pools. False
    when the runner's model is not the engaged draft."""
    d = _DRAFT.get("runner")
    if d is None or d.model is not getattr(model_runner, "model", None):
        return False
    d.mr = model_runner
    d.bind()
    return True


E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def pack_lm_w_fp4(w: torch.Tensor) -> torch.Tensor:
    assert w.dtype == torch.bfloat16 and w.dim() == 2
    v, h = w.shape
    assert h % 16 == 0
    dev = w.device
    lut = torch.tensor(E2M1, device=dev)
    wf = w.float()
    s_row = wf.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / (6.0 * 448.0)
    rows = torch.empty(v, h // 2 + h // 16, dtype=torch.uint8, device=dev)
    for i in range(0, v, 4096):
        b = wf[i : i + 4096].view(-1, h // 16, 16)
        sr = s_row[i : i + 4096].view(-1, 1, 1)
        bs8 = (b.abs().amax(dim=2, keepdim=True) / (6.0 * sr)).to(torch.float8_e4m3fn)
        q = b / (bs8.float().clamp_min(2**-9) * sr)
        code = (
            (q.abs().clamp(max=6.0).unsqueeze(-1) - lut)
            .abs()
            .argmin(dim=-1)
            .to(torch.uint8)
        )
        code = (code | ((q < 0).to(torch.uint8) << 3)).view(-1, h)
        rows[i : i + 4096, : h // 2] = code[:, 0::2] | (code[:, 1::2] << 4)
        rows[i : i + 4096, h // 2 :] = bs8.view(-1, h // 16).view(torch.uint8)
    nbytes = 2 * v * h + v * (h // 2 + h // 16) + 4 * v
    buf = torch.empty(nbytes, dtype=torch.uint8, device=dev)
    buf[: 2 * v * h] = w.contiguous().view(torch.uint8).flatten()
    o = 2 * v * h
    buf[o : o + rows.numel()] = rows.flatten()
    o += rows.numel()
    buf[o : o + 4 * v] = s_row.view(v).contiguous().view(torch.uint8)
    return buf.view(torch.bfloat16)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import torch

from . import target as launch

SGLANG_READS = {
    "srt/speculative/spec_utils.py": (
        "SIMULATE_ACC_LEN",
        "SIMULATE_ACC_METHOD",
        "SIMULATE_ACC_TOKEN_MODE",
        "sample_simulated_acc_len",
    ),
    "kernels/ops/speculative/eagle.py": ("fill_bonus_tokens_func",),
    "srt/speculative/eagle_worker_common.py": ("eagle_sample", "run_eagle_verify"),
    "srt/speculative/eagle_worker_v2.py": ("run_eagle_verify",),
}

SEED = 20261116
KIND_T, KIND_TS = 3, 4
SHARD = 9680
_STATE = {}


class VerifyTail:
    def __init__(self, dev):
        self.d = None
        self.bound = False
        self.shard = torch.zeros((8, SHARD), dtype=torch.bfloat16, device=dev)
        self.full_placeholder = None
        self.in_graph = 0
        self.active = False
        self._pv = None

    def bind(self, draft_runner):
        if self.bound:
            return
        self.d = d = draft_runner
        dev = d.p["accepted"].device
        self.x_ts = torch.zeros((64,), dtype=torch.float32, device=dev)
        xi = self.x_ts.view(torch.int32)
        xi[11] = 1
        xi[12] = SEED & 0x7FFFFFFF
        self.x_ts[9] = 1.0
        self.x_ts[10] = 1.0
        self.argmax = torch.zeros((8,), dtype=torch.int32, device=dev)
        self.kernel = launch.Kernel(launch.ARTIFACTS + "/tail/entry.cubin", launch.ARTIFACTS + "/tail/CALL.json",
                                    launch.DRAFT_PARAMS)
        common = dict(d.vals[0], step=0, last=0, copy=0, normalized=self.shard.data_ptr(), x=self.x_ts.data_ptr(),
                      next_new_tokens=self.argmax.data_ptr())
        self.call_t = self.kernel.make_call(dict(common, kind=KIND_T))
        self.call_ts = self.kernel.make_call(dict(common, kind=KIND_TS))
        self.bound = True

    def gather(self, orig, logits):
        """Keeps the rank's [8, 9680] shard: the kernel exchanges it, so the all-gather is not needed."""
        if not self.active or logits.dim() != 2 or tuple(logits.shape) != (8, SHARD):
            return orig(logits)
        self.shard.copy_(logits)
        self.in_graph += 1
        if self.full_placeholder is None or self.full_placeholder.dtype != logits.dtype:
            self.full_placeholder = torch.zeros((8, SHARD * 16), dtype=logits.dtype, device=logits.device)
        return self.full_placeholder

    @staticmethod
    def uncovered(verify_input, batch):
        si = batch.sampling_info
        reasons = []
        if len(batch.seq_lens) != 1 or verify_input.draft_token_num != 8 or getattr(verify_input, "tree_topk", 1) != 1:
            reasons.append("shape")
        if batch.has_grammar:
            reasons.append("grammar")
        if getattr(batch, "return_logprob", False):
            reasons.append("logprob")
        if si.acc_additive_penalties is not None or si.acc_scaling_penalties is not None:
            reasons.append("penalty")
        if si.logit_bias is not None:
            reasons.append("logit_bias")
        if getattr(si, "sampling_seed", None) is not None:
            reasons.append("seed")
        if not si.is_all_greedy and (getattr(si, "need_top_k_sampling", False) or getattr(si, "need_min_p_sampling", False)):
            reasons.append("top_k/min_p")
        return reasons

    def sample(self, orig, verify_input, batch, logits_output, grammar_mask=None, **kw):
        if batch.forward_mode.is_idle():
            return orig(verify_input, batch, logits_output, grammar_mask, **kw)
        self._pv = None
        reasons = self.uncovered(verify_input, batch)
        if not self.active and self.in_graph == 0:
            reasons.append("not-our-verify")
        if reasons or not self.bound:
            raise RuntimeError(f"glm52.sm107: verify mode {reasons or ['unbound']} not covered (bs=1 only, no fallback)")
        d = self.d
        si = batch.sampling_info
        if not getattr(verify_input, "_glm5mk_ours", False):
            d.p["draft_tokens"][:7].copy_(verify_input.draft_token.view(-1)[1:8].to(torch.int32))
        if d.dirty or d.pending:
            d.reinit_transport()
            d.pending = False
        d.reset_ring()
        s = torch.cuda.current_stream().cuda_stream
        if si.is_all_greedy:
            self.call_t.launch(s)
        else:
            self.x_ts[9:10].copy_(si.temperatures.view(-1)[:1])
            self.x_ts[10:11].copy_(si.top_ps.view(-1)[:1] if getattr(si, "need_top_p_sampling", False)
                                   else torch.ones_like(self.x_ts[10:11]))
            self.call_ts.launch(s)
        return self._post_verify(verify_input, batch, d.p["accepted"], d.p["num_accepted"])

    def _post_verify(self, verify_input, batch, acc, na):
        import sglang.srt.speculative.spec_utils as su
        dev = acc.device
        L = 0
        if su.SIMULATE_ACC_LEN > 0:
            if su.SIMULATE_ACC_TOKEN_MODE != "real-draft-token" or verify_input.max_tree_depth != 8:
                raise RuntimeError(f"glm52.sm107: simulation mode {su.SIMULATE_ACC_TOKEN_MODE} depth "
                                   f"{verify_input.max_tree_depth} not covered")
            L = su.sample_simulated_acc_len(su.SIMULATE_ACC_LEN, su.SIMULATE_ACC_METHOD, verify_input.max_tree_depth)
        cand = verify_input.draft_token.reshape(-1)
        if not cand.is_contiguous():
            cand = cand.contiguous()
        if not hasattr(self, "pv_sel"):
            self.pv_sel = torch.zeros((1,), dtype=torch.int64, device=dev)
        i32 = dict(dtype=torch.int32, device=dev)
        predict, ai, al = torch.empty((8,), **i32), torch.empty((1, 8), **i32), torch.empty((1,), **i32)
        nsl, at, bonus = torch.empty((1,), dtype=torch.int64, device=dev), torch.empty((1, 8), **i32), torch.empty((1,), **i32)
        from . import ops
        ops.post_verify(acc, na, self.argmax, cand, batch.seq_lens, L, predict, ai, al, nsl, at, bonus, acc, na,
                        self.pv_sel)
        self._pv = (al, predict, nsl, at, bonus, self.pv_sel)
        return predict, al, ai


def pv_stash():
    vt = _STATE.get("vt")
    return getattr(vt, "_pv", None) if vt is not None else None


def _pv_nsl(batch, accept_lens):
    st = pv_stash()
    if st is not None and st[0] is accept_lens:
        return st[2]
    return batch.seq_lens + accept_lens


def _pv_tok(predict, accept_index, accept_lens, bs):
    st = pv_stash()
    if st is not None and st[0] is accept_lens and st[1] is predict:
        return st[3], st[4]
    from sglang.kernels.ops.speculative.eagle import fill_bonus_tokens_func
    accept_tokens = predict[accept_index]
    bonus_tokens = torch.empty_like(accept_lens, dtype=torch.int32)
    fill_bonus_tokens_func(accept_tokens, accept_lens, bonus_tokens, accept_index.shape[1], bs)
    return accept_tokens, bonus_tokens


def _patch_run_eagle_verify(ewc):
    import inspect
    import re
    src = inspect.getsource(ewc.run_eagle_verify)
    a = "    new_seq_lens = batch.seq_lens + accept_lens\n"
    assert src.count(a) == 1, "run_eagle_verify: new_seq_lens line not found"
    src = src.replace(a, "    new_seq_lens = _glm5mk_pv_nsl(batch, accept_lens)\n")
    pat = re.compile(r"        accept_tokens = predict\[accept_index\]\n.*?            bs,\n        \)\n", re.S)
    assert len(pat.findall(src)) == 1, "run_eagle_verify: accept_tokens / fill_bonus block not found"
    src = pat.sub("        accept_tokens, bonus_tokens = _glm5mk_pv_tok(predict, accept_index, accept_lens, bs)\n", src)
    g = ewc.__dict__
    g["_glm5mk_pv_nsl"], g["_glm5mk_pv_tok"] = _pv_nsl, _pv_tok
    exec(compile("from __future__ import annotations\n" + src, ewc.__file__, "exec"), g)
    import sglang.srt.speculative.eagle_worker_v2 as ew2
    ew2.run_eagle_verify = g["run_eagle_verify"]


def attach_draft(draft_runner):
    vt = _STATE.get("vt")
    if vt is not None:
        vt.bind(draft_runner)


def set_active(flag):
    vt = _STATE.get("vt")
    if vt is not None:
        vt.active = flag


def install(target_model_runner):
    if "vt" in _STATE:
        return _STATE["vt"]
    vt = VerifyTail(torch.device("cuda", torch.cuda.current_device()))
    _STATE["vt"] = vt
    lp = target_model_runner.model.logits_processor
    orig_g = lp._logits_gatherer

    class _Gatherer:
        def __getattr__(s, k):
            return getattr(orig_g, k)

        def __call__(s, logits, *a, **k):
            return vt.gather(lambda x: orig_g(x, *a, **k), logits)

    lp._logits_gatherer = _Gatherer()
    orig_c = lp._copy_logits_to_buffer

    def copy_w(logits, logits_metadata, use_buffer=True, _o=orig_c):
        # The kernel holds the rank's shard, so the placeholder is never copied into the logits buffer.
        if vt.full_placeholder is not None and logits is vt.full_placeholder:
            buf = logits_metadata.next_token_logits_buffer if use_buffer else None
            return buf if buf is not None and tuple(buf.shape) == tuple(logits.shape) else logits
        return _o(logits, logits_metadata, use_buffer)

    lp._copy_logits_to_buffer = copy_w
    import sglang.srt.speculative.eagle_worker_common as ewc
    _patch_run_eagle_verify(ewc)
    orig_s = ewc.eagle_sample
    ewc.eagle_sample = lambda vi, b, lo, gm=None, _o=orig_s, **k: vt.sample(_o, vi, b, lo, gm, **k)
    return vt

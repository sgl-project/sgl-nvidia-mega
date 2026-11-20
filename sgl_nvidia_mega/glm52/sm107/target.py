# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import ctypes
import json
import os
import struct
import types

import torch

from .kernels.chain_n148sq.schedule import fc2_lists_payload


SGLANG_READS = {
    "srt/distributed/parallel_state.py": ("get_tp_group",),
    "srt/layers/moe/token_dispatcher/flashinfer_utils.py": ("TorchDistributedCommBackend",),
    "srt/layers/attention/dsa_backend.py": (
        "DeepseekSparseAttnBackend",
        "fused_dsa_target_verify_metadata",
        "_apply_cuda_graph_metadata",
        "_refresh_paged_mqa_schedule_metadata",
        "_refresh_topk_v2_plan",
    ),
}

ARTIFACTS = os.environ.get("GLM5MK_ARTIFACTS", "/glm5mk")

PARAMS = (
    ("x", 8, "p"), ("residual", 8, "p"), ("residual_out", 8, "p"), ("normalized", 8, "p"),
    ("w_fa", 8, "p"), ("w_qb", 8, "p"), ("w_o", 8, "p"), ("w_lg", 8, "p"), ("w_gu", 8, "p"), ("w_d", 8, "p"),
    ("w13", 8, "p"), ("w2", 8, "p"), ("gamma_kv", 8, "p"), ("gamma_post", 8, "p"), ("gamma_next", 8, "p"),
    ("router_bias", 8, "p"), ("alpha1", 8, "p"), ("alpha2", 8, "p"), ("rope", 8, "p"), ("length", 8, "p"),
    ("table", 8, "p"), ("cache", 8, "p"), ("topk", 8, "p"), ("arena", 8, "p"), ("part_o", 8, "p"),
    ("part_stats", 8, "p"), ("fc2out", 8, "p"), ("counters", 8, "p"), ("fc2_lists", 8, "p"), ("uc", 8, "p"),
    ("mc", 8, "p"), ("copy", 4, "i"), ("rank", 4, "i"), ("early_x", 1, "b"), ("x_only", 1, "b"),
    ("stride_rows", 8, "q"), ("layer_rows", 8, "q"), ("max_blocks", 4, "i"), ("clear_first", 1, "b"),
    ("widx", 8, "p"), ("idx_kn_w", 8, "p"), ("idx_kn_b", 8, "p"), ("idx_rope", 8, "p"),
    ("idx_pool", 8, "p"), ("idx_blocks", 8, "p"), ("idx_stride_bytes", 8, "q"), ("idx_pages", 8, "q"),
    ("idx_candidates", 8, "p"), ("key_tma", 128, "t"),
)
FMT = {"p": "<Q", "q": "<q", "i": "<i", "b": "<B"}


D012_PARAMS = (
    ("x", 8, "p"), ("residual", 8, "p"), ("residual_out", 8, "p"), ("normalized", 8, "p"),
    ("w_fa", 8, "p"), ("w_qb", 8, "p"), ("w_o", 8, "p"), ("w_gu", 8, "p"), ("w_dn", 8, "p"),
    ("gamma_kv", 8, "p"), ("gamma_post", 8, "p"), ("gamma_next", 8, "p"), ("rope", 8, "p"), ("length", 8, "p"),
    ("table", 8, "p"), ("cache", 8, "p"), ("topk", 8, "p"), ("arena", 8, "p"), ("part_o", 8, "p"),
    ("part_stats", 8, "p"), ("counters", 8, "p"), ("uc", 8, "p"), ("mc", 8, "p"), ("copy", 4, "i"), ("rank", 4, "i"),
    ("early_x", 1, "b"), ("x_only", 1, "b"), ("stride_rows", 8, "q"), ("layer_rows", 8, "q"), ("max_blocks", 4, "i"),
    ("clear_first", 1, "b"), ("widx", 8, "p"), ("idx_kn_w", 8, "p"), ("idx_kn_b", 8, "p"),
    ("idx_rope", 8, "p"), ("idx_pool", 8, "p"), ("idx_blocks", 8, "p"), ("idx_stride_bytes", 8, "q"),
    ("idx_pages", 8, "q"), ("idx_candidates", 8, "p"), ("key_tma", 128, "t"), ("layer0", 1, "b"), ("ids", 8, "p"),
    ("embed", 8, "p"), ("gamma_in", 8, "p"),
)


CU_LAUNCH_ATTRIBUTE_COOPERATIVE = 2
CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION = 6
FUNC_ATTR_IDS = {"CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES": 8, "CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED": 14,
                 "CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE": 17}


DRAFT_PARAMS = (
    ("x", 8, "p"), ("residual", 8, "p"), ("residual_out", 8, "p"), ("normalized", 8, "p"), ("w_fa", 8, "p"),
    ("w_qb", 8, "p"), ("w_o", 8, "p"), ("w_router", 8, "p"), ("bias", 8, "p"), ("w3w1", 8, "p"), ("w2", 8, "p"),
    ("w_sgu", 8, "p"), ("w_sdn", 8, "p"), ("gamma_kv", 8, "p"), ("gamma_in", 8, "p"), ("gamma_post", 8, "p"),
    ("gamma_head", 8, "p"), ("gamma_hnorm", 8, "p"), ("rope", 8, "p"), ("length", 8, "p"), ("table", 8, "p"),
    ("cache", 8, "p"), ("topk", 8, "p"), ("arena", 8, "p"), ("part_o", 8, "p"), ("part_stats", 8, "p"),
    ("counters", 8, "p"), ("uc", 8, "p"), ("mc", 8, "p"), ("lm_w", 8, "p"), ("token_table", 8, "p"),
    ("hidden_rows", 8, "p"), ("draft_tokens", 8, "p"), ("next_new_tokens", 8, "p"), ("accepted", 8, "p"),
    ("num_accepted", 8, "p"), ("copy", 4, "i"), ("rank", 4, "i"), ("step", 4, "i"), ("last", 1, "b"),
    ("stride_rows", 8, "q"), ("layer_rows", 8, "q"), ("max_blocks", 4, "i"), ("clear_first", 1, "b"),
    ("kind", 4, "i"), ("wfa_leader", 8, "p"), ("widx", 8, "p"), ("idx_kn_w", 8, "p"),
    ("idx_kn_b", 8, "p"), ("idx_rope", 8, "p"), ("idx_pool", 8, "p"), ("idx_blocks", 8, "p"),
    ("idx_stride_bytes", 8, "q"), ("idx_pages", 8, "q"), ("idx_candidates", 8, "p"), ("key_tma", 128, "t"),
    ("embed", 8, "p"), ("embed_sharded", 1, "b"), ("gamma_enorm", 8, "p"),
    ("eh_tma", 128, "t"),
)
KIND_SK, KIND_S0 = 0, 2


def layout_from_call(call, params=PARAMS):
    rows = [(p["index"], p["offset"], p["size"]) for p in call["parameters"]]
    assert len(rows) == len(params), (len(rows), len(params))
    for (i, off, size), (name, want, _) in zip(rows, params):
        assert size == want, (i, name, size, want)
    return rows


def pack(values, layout, params=PARAMS):
    out = []
    for (i, off, size), (name, _, kind) in zip(layout, params):
        v = values[name]
        if kind == "t":
            assert isinstance(v, (bytes, bytearray)) and len(v) == 128, name
            out.append(bytes(v))
        else:
            out.append(struct.pack(FMT[kind], int(v)))
    return out


class _CUlaunchAttribute(ctypes.Structure):
    _fields_ = [("id", ctypes.c_int), ("pad", ctypes.c_char * 4), ("value", ctypes.c_uint8 * 64)]


class _CUlaunchConfig(ctypes.Structure):
    _fields_ = [("gridDimX", ctypes.c_uint), ("gridDimY", ctypes.c_uint), ("gridDimZ", ctypes.c_uint),
                ("blockDimX", ctypes.c_uint), ("blockDimY", ctypes.c_uint), ("blockDimZ", ctypes.c_uint),
                ("sharedMemBytes", ctypes.c_uint), ("hStream", ctypes.c_void_p),
                ("attrs", ctypes.POINTER(_CUlaunchAttribute)), ("numAttrs", ctypes.c_uint)]


_cu = None


def driver():
    global _cu
    if _cu is None:
        _cu = ctypes.CDLL("libcuda.so.1")
    return _cu


def check(rc, what):
    if rc != 0:
        name = ctypes.c_char_p()
        driver().cuGetErrorName(rc, ctypes.byref(name))
        raise RuntimeError(f"{what}: CUDA driver error {rc} {name.value}")


def encode_tma_u8(addr, width_bytes, rows):
    cu = driver()
    tm = (ctypes.c_uint8 * 128)()
    gdim = (ctypes.c_uint64 * 2)(width_bytes, rows)
    gstr = (ctypes.c_uint64 * 1)(width_bytes)
    box = (ctypes.c_uint32 * 2)(128, 64)
    es = (ctypes.c_uint32 * 2)(1, 1)
    check(cu.cuTensorMapEncodeTiled(tm, ctypes.c_int(0), ctypes.c_uint32(2), ctypes.c_void_p(addr), gdim, gstr, box,
                                    es, ctypes.c_int(0), ctypes.c_int(3), ctypes.c_int(0), ctypes.c_int(0)),
          "cuTensorMapEncodeTiled")
    b = bytearray(bytes(tm))
    b[8] |= 0x02
    b[10] |= 0x04
    return bytes(b)


def encode_eh_tma(addr):
    return encode_tma_u8(addr, 1536, 6144)


def encode_key_tma(addr, rows):
    return encode_tma_u8(addr, 128, rows)


def tma_rows(idx_pages, idx_stride_bytes):
    return ((idx_pages - 1) * idx_stride_bytes + 8448) // 128


class Kernel:

    def __init__(self, cubin_path, call_json_path, params=PARAMS):
        cu = driver()
        call = json.load(open(call_json_path))
        self.call, self.params = call, params
        self.layout = layout_from_call(call, params)
        self.code = ctypes.create_string_buffer(open(cubin_path, "rb").read())
        ctx = ctypes.c_void_p()
        check(cu.cuCtxGetCurrent(ctypes.byref(ctx)), "cuCtxGetCurrent")
        assert ctx.value, "no current CUDA context (torch must initialize the device first)"
        self.module = ctypes.c_void_p()
        check(cu.cuModuleLoadData(ctypes.byref(self.module), self.code), "cuModuleLoadData")
        self.function = ctypes.c_void_p()
        check(cu.cuModuleGetFunction(ctypes.byref(self.function), self.module, call["symbol"].encode()),
              "cuModuleGetFunction")
        seen = set()
        for a in call.get("function_attributes", []):
            key = (a["name"], a["value"])
            if key in seen:
                continue
            seen.add(key)
            check(cu.cuFuncSetAttribute(self.function, FUNC_ATTR_IDS[a["name"]], a["value"]), "cuFuncSetAttribute")
        for i, off, size in self.layout:
            o, s = ctypes.c_size_t(), ctypes.c_size_t()
            check(cu.cuFuncGetParamInfo(self.function, ctypes.c_size_t(i), ctypes.byref(o), ctypes.byref(s)),
                  "cuFuncGetParamInfo")
            assert (o.value, s.value) == (off, size), (i, o.value, s.value, off, size)
        self.grid, self.block, self.smem = call["grid"], call["block"], call["dynamic_shared_bytes"]
        attrs = {a["id"]: a["value"] for a in call["launch_attributes"]}
        self.pdl = bool(attrs.get("CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION", 0))
        assert not attrs.get("CU_LAUNCH_ATTRIBUTE_COOPERATIVE", 0)

    def make_call(self, values, pdl=None):
        return LayerCall(self, pack(values, self.layout, self.params), self.pdl if pdl is None else pdl)


class LayerCall:
    def __init__(self, kernel, words, pdl):
        self.kernel = kernel
        self.bufs = [ctypes.create_string_buffer(w, len(w)) for w in words]
        self.argv = (ctypes.c_void_p * len(self.bufs))(*[ctypes.addressof(b) for b in self.bufs])
        self.attrs = (_CUlaunchAttribute * 2)()
        self.attrs[0].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
        self.attrs[0].value[0] = 1 if pdl else 0
        self.attrs[1].id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE
        self.attrs[1].value[0] = 0
        self.cfg = _CUlaunchConfig()
        self.cfg.gridDimX, self.cfg.gridDimY, self.cfg.gridDimZ = kernel.grid
        self.cfg.blockDimX, self.cfg.blockDimY, self.cfg.blockDimZ = kernel.block
        self.cfg.sharedMemBytes = kernel.smem
        self.cfg.attrs = self.attrs
        self.cfg.numAttrs = 2

    def set(self, name, value):
        params = self.kernel.params
        idx = [p[0] for p in params].index(name)
        kind = params[idx][2]
        struct.pack_into(FMT[kind], self.bufs[idx], 0, int(value))

    def launch(self, stream_handle):
        self.cfg.hStream = ctypes.c_void_p(stream_handle)
        check(driver().cuLaunchKernelEx(ctypes.byref(self.cfg), self.kernel.function, self.argv, None),
              "cuLaunchKernelEx")


LAYERS = tuple(range(3, 78))
COPIES = 3
ARENA_WORDS = 617440
HOP_WORDS = 393216
TK_WORDS = 16384
LG_WORDS = 32768
Y_WORDS = 8 * 1312
COUNTER_WORDS = 800
CAND_WORDS = COPIES * 8 * ((1 << 20) + 8192) * 2
TRANSPORT_WORDS = COPIES * (2 * HOP_WORDS + TK_WORDS + LG_WORDS + Y_WORDS)
TABLE_ROWS = 16388


def _rank():
    from sglang.srt.distributed import get_tp_group
    return get_tp_group().rank_in_group


def pack_bf16(w):
    m, k = w.shape
    assert w.dtype == torch.bfloat16 and m % 16 == 0 and k % 64 == 0, (w.dtype, w.shape)
    v = w.contiguous().view(m // 16, 2, 8, k // 64, 4, 2, 8).permute(0, 3, 1, 5, 2, 4, 6)
    return v.contiguous().view(torch.uint8).reshape(m // 16, k // 64, 2048)


def swizzle_64b(t):
    *lead, rows, width = t.shape
    assert width == 64
    v = t.reshape(-1, rows, 4, 16)
    r = torch.arange(rows, device=t.device).view(rows, 1)
    dst = torch.arange(4, device=t.device).view(1, 4) ^ ((r >> 1) & 3)
    out = torch.empty_like(v)
    out.scatter_(2, dst.view(1, rows, 4, 1).expand(v.shape[0], rows, 4, 16), v)
    return out.view(*lead, rows, 64)


def _u8(t):
    return t.contiguous().view(torch.uint8)


def _scalar(t):
    v = t.detach().float().reshape(-1)
    assert bool(torch.isfinite(v).all()) and bool((v > 0).all()), v
    assert torch.allclose(v, v[:1].expand_as(v)), "per-expert activation scales differ"
    return v[:1].contiguous()


def make_mcast(total_bytes):
    from flashinfer.comm.mnnvl import McastGPUBuffer
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.layers.moe.token_dispatcher.flashinfer_utils import TorchDistributedCommBackend

    tp = get_tp_group()
    dev = torch.device("cuda", torch.cuda.current_device())
    owner = McastGPUBuffer(total_bytes, tp.world_size, tp.rank_in_group, dev, TorchDistributedCommBackend(tp.cpu_group))
    n = total_bytes // 4

    class _Raw:
        def __init__(self, ptr):
            self.__cuda_array_interface__ = {"shape": (n,), "typestr": "<f4", "data": (int(ptr), False), "version": 3}

    uc = torch.as_tensor(_Raw(owner.get_unicast_ptr(tp.rank_in_group)), device=dev)
    mc = torch.as_tensor(_Raw(owner.get_multicast_ptr()), device=dev)
    return owner, uc, mc


class ChainRunner:
    def __init__(self, model, model_runner):
        self.model, self.mr, self.cfg = model, model_runner, model.config
        cfg = self.cfg
        assert cfg.num_hidden_layers == 78 and cfg.first_k_dense_replace == 3 and cfg.hidden_size == 6144
        self.leaders = {i for i in LAYERS if cfg.indexer_types[i] == "full"}
        assert self.leaders == set(range(6, 75, 4)), sorted(self.leaders)
        self.leader_of = {}
        last = 2
        for i in LAYERS:
            if i in self.leaders:
                last = i
            else:
                self.leader_of[i] = last
        self.rank = _rank()
        self.bound = False
        self.prepare()
        self.d012 = D012Runner(self)

    @torch.no_grad()
    def prepare(self):
        layers, dev = self.model.layers, torch.device("cuda", torch.cuda.current_device())
        a3 = layers[3].self_attn
        P = 1048576
        cs = a3.rotary_emb.cos_sin_cache
        assert cs.shape[1] == 64 and cs.shape[0] >= P, cs.shape
        cs = cs[:P].float()
        pairs = torch.stack([cs[:, :32], cs[:, 32:]], dim=-1).reshape(P, 64)
        self.rope = torch.cat([pairs, pairs], dim=1).contiguous().view(-1)
        ix0 = layers[min(self.leaders)].self_attn.indexer
        ics = ix0.rotary_emb.cos_sin_cache
        assert ics.shape[1] == 64 and ics.shape[0] >= P, ics.shape
        self.idx_rope = ics[:P].float().contiguous().view(-1)
        assert self.idx_rope.numel() == P * 64
        self.W = {}
        for i in LAYERS:
            self.W[i] = self._pack_layer(i, dev)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def _pack_layer(self, i, dev):
        L = self.model.layers[i]
        sa, mlp = L.self_attn, L.mlp
        heads, nope, rope_d, lora, vd = sa.num_local_heads, sa.qk_nope_head_dim, sa.qk_rope_head_dim, sa.kv_lora_rank, \
            sa.v_head_dim
        assert (heads, nope, rope_d, lora, vd) == (4, 192, 64, 512, 256), (heads, nope, rope_d, lora, vd)
        w = {}
        fa = sa.fused_qkv_a_proj_with_mqa.weight
        assert fa.dtype == torch.bfloat16 and tuple(fa.shape) == (2624, 6144), (fa.dtype, fa.shape)
        gq = sa.q_a_layernorm.weight.double()
        if i in self.leaders:
            ix = sa.indexer
            if hasattr(ix, "wk_weights_proj"):
                wkw = ix.wk_weights_proj.weight.to(torch.bfloat16)
                wk, wp = wkw[: ix.head_dim], wkw[ix.head_dim:]
            else:
                wk, wp = ix.wk.weight.to(torch.bfloat16), ix.weights_proj.weight.to(torch.bfloat16)
            assert tuple(wk.shape) == (128, 6144) and tuple(wp.shape) == (32, 6144), (wk.shape, wp.shape)
            fa = torch.cat([fa, wk, wp])
            wq = ix.wq_b.weight
            assert tuple(wq.shape) == (4096, 2048), wq.shape
            w["w_idx"] = pack_bf16((wq.double() * gq).to(torch.bfloat16).contiguous())
            w["idx_kn_w"] = ix.k_norm.weight.float().contiguous()
            w["idx_kn_b"] = ix.k_norm.bias.float().contiguous()
            eps = getattr(ix.k_norm, "eps", getattr(ix.k_norm, "variance_epsilon", None))
            assert eps is not None and abs(float(eps) - 1e-6) < 1e-12, eps
            assert ix.n_heads == 32 and ix.head_dim == 128 and ix.index_topk == 2048
        w["w_fa"] = pack_bf16(fa.contiguous())

        qb = sa.q_b_proj.weight
        assert qb.dtype == torch.bfloat16 and tuple(qb.shape) == (heads * (nope + rope_d), 2048), (qb.dtype, qb.shape)
        qb = qb.view(heads, nope + rope_d, -1)
        wkc = sa.w_kc
        kbt = wkc.transpose(1, 2) if tuple(wkc.shape) == (heads, nope, lora) else wkc
        assert tuple(kbt.shape) == (heads, lora, nope) and kbt.dtype == torch.bfloat16, (kbt.shape, kbt.dtype)
        a = torch.cat([torch.bmm(kbt.double(), qb[:, :nope].double()), qb[:, nope:].double()], dim=1) * gq
        w["w_qb"] = pack_bf16(a.to(torch.bfloat16).reshape(heads * (lora + rope_d), -1).contiguous())
        del a

        wvc = sa.w_vc
        vb = wvc.transpose(1, 2) if tuple(wvc.shape) == (heads, lora, vd) else wvc
        assert tuple(vb.shape) == (heads, vd, lora), vb.shape
        wo = sa.o_proj.weight
        assert wo.dtype == torch.bfloat16 and tuple(wo.shape) == (6144, 1024), (wo.dtype, wo.shape)
        tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            wov = torch.cat([wo[:, 256 * h:256 * h + 256].float() @ vb[h].float() for h in range(4)], dim=1)
            w["w_o"] = pack_bf16(wov.to(torch.bfloat16))
            gamma_post = L.post_attention_layernorm.weight
            wg = mlp.gate.weight.float() * gamma_post.float().view(1, -1)
            r = self.rank
            m = torch.cat([wg @ wov, wg[:, 384 * r:384 * r + 384]], dim=1)
            assert tuple(m.shape) == (256, 64 * 38), m.shape
            w["w_lg"] = pack_bf16(m.to(torch.bfloat16).contiguous())
            del wov, wg, m
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32
        se = mlp.shared_experts
        gu, dn = se.gate_up_proj.weight, se.down_proj.weight
        assert gu.dtype == torch.bfloat16 and tuple(gu.shape) == (256, 6144) and tuple(dn.shape) == (6144, 128)
        w["w_gu"], w["w_d"] = pack_bf16(gu), pack_bf16(dn)
        ex = mlp.experts
        assert ex.w13_weight.shape[0] == 256 and not getattr(mlp, "num_fused_shared_experts", 0)
        w13 = _u8(ex.w13_weight).reshape(256, 2, 128, 48, 64).permute(0, 1, 3, 2, 4)
        w13 = swizzle_64b(w13.contiguous()).reshape(256, 2, 8, 6 * 8192)
        s13 = _u8(ex.w13_weight_scale).reshape(256, 2, 8, 12 * 512)
        w["w13"] = torch.cat([w13, s13], dim=3).contiguous()
        del w13, s13
        w2 = swizzle_64b(_u8(ex.w2_weight).reshape(256, 48, 128, 64)).reshape(256, 8, 6 * 8192)
        s2 = _u8(ex.w2_weight_scale).reshape(256, 8, 6 * 2 * 512)
        w["w2"] = torch.cat([w2, s2], dim=2).contiguous()
        del w2, s2
        g1 = ex.g1_alphas.float().reshape(-1)
        g1u = getattr(ex, "g1_alphas_up", None)
        if g1u is not None:
            assert torch.equal(g1u.float().reshape(-1), g1), "w1/w3 weight_scale_2 differ"
        w["alpha1"] = torch.cat([g1.reshape(256), _scalar(ex.w13_input_scale_quant)]).contiguous()
        w["alpha2"] = torch.cat([ex.g2_alphas.float().reshape(256), _scalar(ex.w2_input_scale_quant)]).contiguous()
        w["gamma_kv"] = sa.kv_a_layernorm.weight.contiguous()
        w["gamma_post"] = L.post_attention_layernorm.weight.contiguous()
        nxt = self.model.layers[i + 1].input_layernorm if i < 77 else self.model.norm
        w["gamma_next"] = nxt.weight.contiguous()
        b = mlp.gate.e_score_correction_bias.float()
        b = b - b.mean()
        w["router_bias"] = b.to(torch.bfloat16).contiguous()
        for name in ("gamma_kv", "gamma_post", "gamma_next"):
            assert w[name].dtype == torch.bfloat16, (name, w[name].dtype)
        return w

    @torch.no_grad()
    def bind(self):
        if self.bound:
            return
        dev = torch.device("cuda", torch.cuda.current_device())
        pool = self.mr.token_to_kv_pool
        assert pool.page_size == 64, pool.page_size
        r2t = self.mr.req_to_token_pool.req_to_token
        assert (r2t.shape[1] + 63) // 64 <= TABLE_ROWS, (r2t.shape, TABLE_ROWS)
        neg0 = lambda n: torch.full((n,), -0.0, dtype=torch.float32, device=dev)
        zero = lambda n: torch.zeros((n,), dtype=torch.int32, device=dev)
        self.ws = dict(arena=neg0(COPIES * ARENA_WORDS), part_o=neg0(262144), part_stats=neg0(1024),
                       fc2out=neg0(64 * 8 * 3072), counters=zero(COPIES * COUNTER_WORDS), fc2_lists=zero(12480),
                       idx_candidates=zero(CAND_WORDS))
        payload = bytearray(fc2_lists_payload())
        self.ws["fc2_lists"].copy_(torch.frombuffer(payload, dtype=torch.uint8).view(torch.int32))
        self.mcast, self.uc, self.mc = make_mcast(4 * TRANSPORT_WORDS)
        self.uc.fill_(-0.0)
        torch.cuda.synchronize()
        from sglang.srt.distributed import get_tp_group
        torch.distributed.barrier(group=get_tp_group().cpu_group)
        self.bt = torch.zeros((TABLE_ROWS,), dtype=torch.int32, device=dev)
        self.kvl = torch.zeros((1,), dtype=torch.int32, device=dev)
        self.topk_in = torch.full((8, 2048), -1, dtype=torch.int32, device=dev)
        self.topk = {i: torch.full((8, 2048), -1, dtype=torch.int32, device=dev) for i in sorted(self.leaders)}
        self.res = {i: torch.empty((8, 6144), dtype=torch.bfloat16, device=dev) for i in LAYERS}
        self.norm_out = torch.empty((8, 6144), dtype=torch.bfloat16, device=dev)
        self.x_ph = torch.zeros((8, 6144), dtype=torch.bfloat16, device=dev)
        self.r_ph = torch.zeros((8, 6144), dtype=torch.bfloat16, device=dev)
        self.kernel = Kernel(ARTIFACTS + "/chain/entry.cubin", ARTIFACTS + "/chain/CALL.json")
        self.layer_calls = []
        ws = self.ws
        for i in LAYERS:
            w = self.W[i]
            kv = pool.get_key_buffer(i)
            assert kv.is_contiguous() and kv.numel() * kv.element_size() % 576 == 0, (kv.shape, kv.dtype)
            leader = i in self.leaders
            v = dict(
                x=self.x_ph.data_ptr() if i == 3 else ws["arena"].data_ptr(),
                residual=self.r_ph.data_ptr() if i == 3 else self.res[i - 1].data_ptr(),
                residual_out=self.res[i].data_ptr(),
                normalized=self.norm_out.data_ptr() if i == 77 else ws["arena"].data_ptr(),
                rope=self.rope.data_ptr(), length=self.kvl.data_ptr(), table=self.bt.data_ptr(), cache=kv.data_ptr(),
                topk=self.topk[i].data_ptr() if leader else
                (self.topk_in.data_ptr() if self.leader_of[i] == 2 else self.topk[self.leader_of[i]].data_ptr()),
                uc=self.uc.data_ptr(), mc=self.mc.data_ptr(), copy=(i - 3) % COPIES, rank=self.rank,
                early_x=int(i > 3), x_only=int(i < 77), stride_rows=64, layer_rows=0, max_blocks=TABLE_ROWS,
                clear_first=0,
                **{k: w[k].data_ptr() for k in ("w_fa", "w_qb", "w_o", "w_lg", "w_gu", "w_d", "w13", "w2", "gamma_kv",
                                                "gamma_post", "gamma_next", "router_bias", "alpha1", "alpha2")},
                **{k: ws[k].data_ptr() for k in ("arena", "part_o", "part_stats", "fc2out", "counters", "fc2_lists",
                                                 "idx_candidates")},
            )
            if leader:
                kb = pool.get_index_k_with_scale_buffer(i)
                assert kb.dtype == torch.uint8 and kb.dim() == 2 and kb.shape[1] == 64 * 132, (kb.dtype, kb.shape)
                stride = kb.stride(0) * kb.element_size()
                assert stride % 128 == 0 and stride >= 8448, stride
                pages = kb.shape[0]
                v.update(widx=w["w_idx"].data_ptr(), idx_kn_w=w["idx_kn_w"].data_ptr(),
                         idx_kn_b=w["idx_kn_b"].data_ptr(), idx_rope=self.idx_rope.data_ptr(),
                         idx_pool=kb.data_ptr(), idx_blocks=self.bt.data_ptr(), idx_stride_bytes=stride,
                         idx_pages=pages)
            else:
                v.update(widx=w["w_qb"].data_ptr(), idx_kn_w=w["alpha1"].data_ptr(), idx_kn_b=w["alpha2"].data_ptr(),
                         idx_rope=self.rope.data_ptr(), idx_pool=kv.data_ptr(), idx_blocks=self.bt.data_ptr(),
                         idx_stride_bytes=-8448, idx_pages=-1)
            v["key_tma"] = encode_key_tma(v["idx_pool"], tma_rows(v["idx_pages"], v["idx_stride_bytes"]))
            self.layer_calls.append(self.kernel.make_call(v))
        self.first_call = self.layer_calls[0]
        from . import ops
        ops.warm_up(dev)
        self.d012.bind()
        x3, r3 = self.d012.nrm[2], self.d012.rout[2]
        self.first_call.set("x", x3.data_ptr())
        self.first_call.set("residual", r3.data_ptr())
        self.cur_in = (x3.data_ptr(), r3.data_ptr())
        self.bound = True

    def eligible(self, forward_batch, n):
        return (self.bound and forward_batch.forward_mode.is_target_verify() and forward_batch.batch_size == 1
                and n == 8)

    def fill_step(self, forward_batch, positions):
        from . import ops
        ops.fill_step_all(self.mr.req_to_token_pool.req_to_token, forward_batch.req_pool_indices[:1].to(torch.int64),
                          positions.to(torch.int64), self.bt, self.kvl)

    def launch_chain(self, x, r):
        key = (x.data_ptr(), r.data_ptr())
        if key != self.cur_in:
            self.first_call.set("x", key[0])
            self.first_call.set("residual", key[1])
            self.cur_in = key
        s = torch.cuda.current_stream().cuda_stream
        for c in self.layer_calls:
            c.launch(s)
        return self.norm_out


_RUNNER = {}


def patched_forward(self, input_ids, positions, forward_batch, input_embeds=None, pp_proxy_tensors=None):
    runner = _RUNNER.get(id(self))
    n = input_ids.shape[0] if input_embeds is None else input_embeds.shape[0]
    if runner is not None and not runner.bound and not torch.cuda.is_current_stream_capturing():
        runner.bind()
    if runner is not None:
        from . import verify_tail
        verify_tail.set_active(runner.bound and runner.eligible(forward_batch, n))
    if runner is None or not runner.eligible(forward_batch, n):
        return self._glm5mk_orig_forward(input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors)
    assert self.pp_group.is_first_rank and self.pp_group.is_last_rank and len(self.layers_to_capture) == 0
    if not torch.cuda.is_current_stream_capturing() and int(positions[-1]) + 1 > (1 << 20):
        return self._glm5mk_orig_forward(input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors)
    hidden = self.embed_tokens(input_ids) if input_embeds is None else input_embeds
    assert not getattr(hidden, "_sglang_needs_allreduce_fusion", False), "embedding output not reduced"
    assert hidden.shape == (8, 6144) and hidden.dtype == torch.bfloat16, (hidden.shape, hidden.dtype)
    runner.fill_step(forward_batch, positions)
    x, r = runner.d012.launch_dense(hidden)
    return runner.launch_chain(x, r).clone()


def on_model_loaded(model_runner):
    if getattr(model_runner, "is_draft_worker", False):
        return
    inner = getattr(getattr(model_runner, "model", None), "model", None)
    if inner is None or type(inner).__name__ != "DeepseekV2Model":
        return
    from sglang.srt.distributed import get_tp_group
    assert get_tp_group().world_size == 16
    if not hasattr(type(inner), "_glm5mk_orig_forward"):
        type(inner)._glm5mk_orig_forward = type(inner).forward
    inner.forward = types.MethodType(patched_forward, inner)
    _RUNNER[id(inner)] = ChainRunner(inner, model_runner)
    _skip_verify_meta(_RUNNER[id(inner)])
    _skip_verify_meta2(_RUNNER[id(inner)])


def _skip_verify_meta(runner):
    import sglang.srt.layers.attention.dsa_backend as db
    orig = db.fused_dsa_target_verify_metadata
    if getattr(orig, "_glm5mk", False):
        return

    def wrapped(*a, **k):
        if k.get("bs") == 1 and k.get("next_n") == 8 and runner.bound:
            return None
        return orig(*a, **k)

    wrapped._glm5mk = True
    db.fused_dsa_target_verify_metadata = wrapped


def _skip_verify_meta2(runner):
    import sglang.srt.layers.attention.dsa_backend as db
    cls = db.DeepseekSparseAttnBackend
    if getattr(cls._apply_cuda_graph_metadata, "_glm5mk", False):
        return
    oa, opm, otp = cls._apply_cuda_graph_metadata, cls._refresh_paged_mqa_schedule_metadata, cls._refresh_topk_v2_plan
    st = {"skip": False}

    def apply_w(self, bs, req_pool_indices, seq_lens, seq_lens_cpu, forward_mode, *a, **k):
        prev = st["skip"]
        st["skip"] = (bs == 1 and forward_mode.is_target_verify() and runner.bound
                      and not torch.cuda.is_current_stream_capturing())
        try:
            return oa(self, bs, req_pool_indices, seq_lens, seq_lens_cpu, forward_mode, *a, **k)
        finally:
            st["skip"] = prev

    def pm_w(self, metadata, seqlens_32_2d):
        if st["skip"] and metadata.paged_mqa_schedule_metadata is not None:
            return
        return opm(self, metadata, seqlens_32_2d)

    def tp_w(self, metadata):
        if st["skip"] and metadata.topk_v2_plan is not None:
            return
        return otp(self, metadata)

    apply_w._glm5mk = True
    cls._apply_cuda_graph_metadata, cls._refresh_paged_mqa_schedule_metadata, cls._refresh_topk_v2_plan = apply_w, pm_w, tp_w


def before_graphs(model_runner):
    if getattr(model_runner, "is_draft_worker", False):
        return
    inner = getattr(getattr(model_runner, "model", None), "model", None)
    r = _RUNNER.get(id(inner))
    if r is not None:
        r.bind()
        from . import verify_tail
        verify_tail.install(model_runner)


D_LAYERS = (0, 1, 2)
COUNTED_CLEARS = 155
MCAST_BYTES = 10152960


class D012Runner:
    def __init__(self, chain):
        self.chain, self.model, self.mr = chain, chain.model, chain.mr
        cfg = self.model.config
        assert all(cfg.indexer_types[i] == "full" for i in D_LAYERS), cfg.indexer_types[:3]
        assert cfg.first_k_dense_replace == 3 and cfg.intermediate_size == 12288
        self.rank = chain.rank
        self.bound = False
        self.prepare()

    @torch.no_grad()
    def prepare(self):
        self.W = {i: self._pack_layer(i) for i in D_LAYERS}
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def _pack_layer(self, i):
        L = self.model.layers[i]
        sa, mlp = L.self_attn, L.mlp
        heads, nope, rope_d, lora, vd = sa.num_local_heads, sa.qk_nope_head_dim, sa.qk_rope_head_dim, sa.kv_lora_rank, \
            sa.v_head_dim
        assert (heads, nope, rope_d, lora, vd) == (4, 192, 64, 512, 256), (heads, nope, rope_d, lora, vd)
        assert type(mlp).__name__ == "DeepseekV2MLP", type(mlp).__name__
        w = {}
        fa = sa.fused_qkv_a_proj_with_mqa.weight
        assert fa.dtype == torch.bfloat16 and tuple(fa.shape) == (2624, 6144), (fa.dtype, fa.shape)
        gq = sa.q_a_layernorm.weight.double()
        ix = sa.indexer
        if hasattr(ix, "wk_weights_proj"):
            wkw = ix.wk_weights_proj.weight.to(torch.bfloat16)
            wk, wp = wkw[: ix.head_dim], wkw[ix.head_dim:]
        else:
            wk, wp = ix.wk.weight.to(torch.bfloat16), ix.weights_proj.weight.to(torch.bfloat16)
        assert tuple(wk.shape) == (128, 6144) and tuple(wp.shape) == (32, 6144), (wk.shape, wp.shape)
        w["w_fa"] = pack_bf16(torch.cat([fa, wk, wp]).contiguous())
        assert tuple(w["w_fa"].shape) == (174, 96, 2048), w["w_fa"].shape
        wq = ix.wq_b.weight
        assert tuple(wq.shape) == (4096, 2048), wq.shape
        w["w_idx"] = pack_bf16((wq.double() * gq).to(torch.bfloat16).contiguous())
        w["idx_kn_w"] = ix.k_norm.weight.float().contiguous()
        w["idx_kn_b"] = ix.k_norm.bias.float().contiguous()
        eps = getattr(ix.k_norm, "eps", getattr(ix.k_norm, "variance_epsilon", None))
        assert eps is not None and abs(float(eps) - 1e-6) < 1e-12, eps
        assert ix.n_heads == 32 and ix.head_dim == 128 and ix.index_topk == 2048
        qb = sa.q_b_proj.weight
        assert qb.dtype == torch.bfloat16 and tuple(qb.shape) == (heads * (nope + rope_d), 2048), (qb.dtype, qb.shape)
        qb = qb.view(heads, nope + rope_d, -1)
        wkc = sa.w_kc
        kbt = wkc.transpose(1, 2) if tuple(wkc.shape) == (heads, nope, lora) else wkc
        assert tuple(kbt.shape) == (heads, lora, nope) and kbt.dtype == torch.bfloat16, (kbt.shape, kbt.dtype)
        a = torch.cat([torch.bmm(kbt.double(), qb[:, :nope].double()), qb[:, nope:].double()], dim=1) * gq
        w["w_qb"] = pack_bf16(a.to(torch.bfloat16).reshape(heads * (lora + rope_d), -1).contiguous())
        assert tuple(w["w_qb"].shape) == (144, 32, 2048), w["w_qb"].shape
        del a
        wvc = sa.w_vc
        vb = wvc.transpose(1, 2) if tuple(wvc.shape) == (heads, lora, vd) else wvc
        assert tuple(vb.shape) == (heads, vd, lora), vb.shape
        wo = sa.o_proj.weight
        assert wo.dtype == torch.bfloat16 and tuple(wo.shape) == (6144, 1024), (wo.dtype, wo.shape)
        tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            wov = torch.cat([wo[:, 256 * h:256 * h + 256].float() @ vb[h].float() for h in range(4)], dim=1)
            w["w_o"] = pack_bf16(wov.to(torch.bfloat16))
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32
        assert tuple(w["w_o"].shape) == (384, 32, 2048), w["w_o"].shape
        del wov

        gu, dn = mlp.gate_up_proj.weight, mlp.down_proj.weight
        assert gu.dtype == torch.bfloat16 and tuple(gu.shape) == (1536, 6144) and gu.is_contiguous(), (gu.dtype, gu.shape)
        assert dn.dtype == torch.bfloat16 and tuple(dn.shape) == (6144, 768) and dn.is_contiguous(), (dn.dtype, dn.shape)
        w["w_gu"], w["w_dn"] = gu, dn
        w["gamma_kv"] = sa.kv_a_layernorm.weight.contiguous()
        w["gamma_post"] = L.post_attention_layernorm.weight.contiguous()
        w["gamma_next"] = self.model.layers[i + 1].input_layernorm.weight.contiguous()
        for k in ("gamma_kv", "gamma_post", "gamma_next"):
            assert w[k].dtype == torch.bfloat16, (k, w[k].dtype)
        e = getattr(L.input_layernorm, "variance_epsilon", getattr(L.input_layernorm, "eps", None))
        assert e is not None and abs(float(e) - 1e-5) < 1e-12, e
        return w

    @torch.no_grad()
    def bind(self):
        if self.bound:
            return
        ch = self.chain
        dev = torch.device("cuda", torch.cuda.current_device())
        pool = self.mr.token_to_kv_pool
        neg0 = lambda n: torch.full((n,), -0.0, dtype=torch.float32, device=dev)
        zero = lambda n: torch.zeros((n,), dtype=torch.int32, device=dev)
        self.ws = dict(arena=neg0(COPIES * ARENA_WORDS), part_o=neg0(262144), part_stats=neg0(1024),
                       counters=zero(COPIES * COUNTER_WORDS), idx_candidates=zero(CAND_WORDS))
        self.ws["counters"][2 * COUNTER_WORDS] = COUNTED_CLEARS
        self.mcast, self.uc, self.mc = make_mcast(MCAST_BYTES)
        self.uc.fill_(-0.0)
        torch.cuda.synchronize()
        from sglang.srt.distributed import get_tp_group
        torch.distributed.barrier(group=get_tp_group().cpu_group)
        bf = dict(dtype=torch.bfloat16, device=dev)
        self.e_in = torch.zeros((8, 6144), **bf)
        self.ids = torch.arange(8, dtype=torch.int32, device=dev)
        self.x0 = torch.zeros((8, 6144), **bf)
        self.res0 = torch.zeros((8, 6144), **bf)
        self.nrm = {i: torch.zeros((8, 6144), **bf) for i in D_LAYERS}
        self.rout = {i: torch.zeros((8, 6144), **bf) for i in D_LAYERS}
        self.topk = {0: torch.full((8, 2048), -1, dtype=torch.int32, device=dev),
                     1: torch.full((8, 2048), -1, dtype=torch.int32, device=dev),
                     2: ch.topk_in}
        self.kernel = Kernel(ARTIFACTS + "/d012/entry.cubin", ARTIFACTS + "/d012/CALL.json",
                                    D012_PARAMS)
        self.calls_l = []
        gamma_in = self.model.layers[0].input_layernorm.weight.contiguous()
        assert gamma_in.dtype == torch.bfloat16
        self.gamma_in = gamma_in
        for i in D_LAYERS:
            w = self.W[i]
            kv = pool.get_key_buffer(i)
            assert kv.is_contiguous() and kv.numel() * kv.element_size() % 576 == 0, (kv.shape, kv.dtype)
            kb = pool.get_index_k_with_scale_buffer(i)
            assert kb.dtype == torch.uint8 and kb.dim() == 2 and kb.shape[1] == 64 * 132, (kb.dtype, kb.shape)
            stride = kb.stride(0) * kb.element_size()
            assert stride == 8448, stride
            pages = kb.shape[0]
            v = dict(
                x=self.x0.data_ptr() if i == 0 else self.nrm[i - 1].data_ptr(),
                residual=self.res0.data_ptr() if i == 0 else self.rout[i - 1].data_ptr(),
                residual_out=self.rout[i].data_ptr(), normalized=self.nrm[i].data_ptr(),
                w_fa=w["w_fa"].data_ptr(), w_qb=w["w_qb"].data_ptr(), w_o=w["w_o"].data_ptr(),
                w_gu=w["w_gu"].data_ptr(), w_dn=w["w_dn"].data_ptr(), gamma_kv=w["gamma_kv"].data_ptr(),
                gamma_post=w["gamma_post"].data_ptr(), gamma_next=w["gamma_next"].data_ptr(),
                rope=ch.rope.data_ptr(), length=ch.kvl.data_ptr(), table=ch.bt.data_ptr(), cache=kv.data_ptr(),
                topk=self.topk[i].data_ptr(), arena=self.ws["arena"].data_ptr(), part_o=self.ws["part_o"].data_ptr(),
                part_stats=self.ws["part_stats"].data_ptr(), counters=self.ws["counters"].data_ptr(),
                uc=self.uc.data_ptr(), mc=self.mc.data_ptr(), copy=i, rank=self.rank, early_x=int(i == 0), x_only=0,
                stride_rows=64, layer_rows=0, max_blocks=TABLE_ROWS, clear_first=0,
                widx=w["w_idx"].data_ptr(), idx_kn_w=w["idx_kn_w"].data_ptr(), idx_kn_b=w["idx_kn_b"].data_ptr(),
                idx_rope=ch.idx_rope.data_ptr(), idx_pool=kb.data_ptr(), idx_blocks=ch.bt.data_ptr(),
                idx_stride_bytes=stride, idx_pages=pages, idx_candidates=self.ws["idx_candidates"].data_ptr(),
                layer0=int(i == 0), ids=self.ids.data_ptr(), embed=self.e_in.data_ptr(),
                gamma_in=self.gamma_in.data_ptr(),
            )
            v["key_tma"] = encode_key_tma(v["idx_pool"], tma_rows(pages, stride))
            self.calls_l.append(self.kernel.make_call(v))
        self.bound = True

    def launch_dense(self, hidden):
        self.e_in.copy_(hidden)
        s = torch.cuda.current_stream().cuda_stream
        for c in self.calls_l:
            c.launch(s)
        return self.nrm[2], self.rout[2]

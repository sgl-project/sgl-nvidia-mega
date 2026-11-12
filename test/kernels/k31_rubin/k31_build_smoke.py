# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Whole-plugin build smoke for the K3.1 megakernel on one node (VR200 SM107 or GB300).

Runs the exact build path the server takes at --context-length > 2048 (No-MTP):
``kimi_k31.dsa_spec.build_sparse_decode`` -> ``providers.build_resources`` over a 93-layer
fake K3.1 model with TP8 rank-local shapes (24 MLA/DSA + 69 KDA layers, dense layer 0,
92 MoE layers; random BF16 weights; one MXFP4 expert set shared by all MoE layers to save
memory), then ``kimi_k31.spec._prepare`` (index-K bind + ``ResourceSet.bind`` over fake
typed pool views: compiles the MLA/KDA cores and runs FlashInfer's warm trtllm call).

W processes (one per GPU, W = 4 in-tray) each build TP rank ``g`` of TP8: the
``get_parallel`` the plugin reads is patched to (tp_size 8, tp_rank g, the real W-rank
group), so every mailbox is a real symm_mem multicast allocation with TP8 geometry.
Nothing is launched across ranks (ranks 4-7 do not exist here): this checks that every
kernel compiles for this GPU's ISA, every residency/cluster/SMEM check passes, and every
binding validates. Kernel correctness is the per-family tests' job.

Records: per phase wall time, the plugin's compilation inventory, the CUDA memory delta,
and the binary version / registers / SMEM / spills of every CuTe module compiled in the
process (CUDA driver attributes; 107 = sm_107a SASS).

Usage (inside the image, plugin + MK host tree on PYTHONPATH):
  python3 -m torch.distributed.run --nproc-per-node 4 k31_build_smoke.py --out DIR [--widths 1,2,...]
Exit 0 only when every rank built and bound.
"""

import argparse
import json
import math
import os
import sys
import time
import traceback
from types import SimpleNamespace as NS

import torch
import torch.distributed as dist

BF16, F32, U8 = torch.bfloat16, torch.float32, torch.uint8
FULL_1IDX = [4 * i for i in range(1, 24)] + [93]  # dsa_config.dsa_layers (1-indexed)
FULL_IDS = [i - 1 for i in FULL_1IDX]
LINEAR_IDS = [i for i in range(93) if i not in FULL_IDS]


class Mod(torch.nn.Module):
    """Attribute container that registers Parameters and submodules like SGLang's."""


def param(t):
    return torch.nn.Parameter(t, requires_grad=False)


def rand(shape, dev, scale=0.02, dtype=BF16):
    return (torch.randn(shape, device=dev, dtype=F32) * scale).to(dtype)


def linear(rows, cols, dev, scale=0.02):
    m = Mod()
    m.weight = param(rand((rows, cols), dev, scale))
    return m


def rmsnorm(n, dev, eps=1e-5):
    m = Mod()
    m.weight = param((1 + rand((n,), dev, 0.05, F32)).to(BF16))
    m.variance_epsilon = eps
    return m


def mla_attention(dev):
    a = Mod()
    a.fused_qkv_a_proj_with_mqa = linear(2112, 7168, dev)
    a.g_proj = linear(1536, 7168, dev)
    a.q_b_proj = linear(12 * 192, 1536, dev)
    a.o_proj = linear(7168, 1536, dev)
    a.q_a_layernorm = rmsnorm(1536, dev)
    a.kv_a_layernorm = rmsnorm(512, dev)
    # Stock keeps W_UK_T as a transposed view of [12, 512, 128] and W_UV of [12, 128, 512].
    a.w_kc = rand((12, 512, 128), dev, 0.05).transpose(1, 2)
    a.w_vc = rand((12, 128, 512), dev, 0.05).transpose(1, 2)
    a.attn_mqa = NS()
    a.scaling = 192.0**-0.5
    a.w_scale = 1.0
    ix = Mod()
    ix.wk = linear(128, 7168, dev)
    ix._wk_gate_weight = None
    ix.wq_b = linear(4096, 1536, dev)
    ix.weights_proj = linear(32, 7168, dev)
    ix.num_init_tokens, ix.num_local_tokens = 0, 0
    ix.skip_rope, ix.apply_hadamard = True, False
    kn = Mod()
    kn.weight = param(torch.ones(128, device=dev, dtype=F32))
    kn.bias = param(torch.zeros(128, device=dev, dtype=F32))
    kn.variance_epsilon = 1e-6
    ix.k_norm = kn
    a.indexer = ix
    return a


def kda_attention(dev):
    a = Mod()
    a.fused_qkvg_proj = linear(6144, 7168, dev)
    bfa = rand((144, 7168), dev)
    bfa[140:].zero_()
    a.f_a_proj, a.b_proj = Mod(), Mod()
    a.f_a_proj.weight = param(bfa[:128])
    a.b_proj.weight = param(bfa[128:140])
    a.__dict__["_bfa_w"] = bfa  # stock's unregistered [f_a | b | pad] alias
    a.attn = NS(lower_bound=-5.0)
    a.qkv_conv1d = Mod()
    a.qkv_conv1d.weight = param(rand((4608, 1, 4), dev, 0.3, F32))  # stock keeps the conv FP32
    a.qkv_conv1d.bias = None
    a.dt_bias = param(rand((1536,), dev, 0.5, F32))
    a.A_log = param(rand((12,), dev, 0.5, F32))
    a.f_b_proj = linear(1536, 128, dev)
    a.o_norm = Mod()
    a.o_norm.weight = param((1 + rand((128,), dev, 0.05, F32)).to(BF16))
    a.o_norm.eps = 1e-5
    a.o_proj = linear(7168, 1536, dev)
    return a


def experts(dev):
    """Packed MXFP4 routed experts, TP8 rank shard (intermediate 3072 / 8 = 384)."""
    e = Mod()
    g = torch.Generator(device=dev)
    g.manual_seed(7)
    r = lambda shape, lo, hi: torch.randint(lo, hi, shape, generator=g, device=dev, dtype=torch.int32).to(U8)
    e.w13_weight = param(r((896, 768, 1792), 0, 256))
    e.w13_weight_scale = param(r((896, 768, 112), 118, 126))
    e.w2_weight = param(r((896, 3584, 192), 0, 256))
    e.w2_weight_scale = param(r((896, 3584, 12), 118, 126))
    return e


def moe(dev, routed):
    m = Mod()
    m.experts = routed
    m.shared_experts = Mod()
    m.shared_experts.gate_up_proj = linear(2 * 768, 7168, dev)
    m.shared_experts.down_proj = linear(7168, 768, dev)
    m.gate = Mod()
    m.gate.weight = param(rand((896, 7168), dev))
    m.gate.e_score_correction_bias = param(rand((896,), dev, 0.1, F32))
    m.routed_expert_down_proj = linear(3584, 7168, dev)
    m.routed_expert_norm = rmsnorm(3584, dev)
    m.routed_expert_up_proj = linear(7168, 3584, dev)
    m.routed_scaling_factor = 1.0
    return m


def dense(dev):
    m = Mod()
    m.gate_up_proj = linear(2 * 4224, 7168, dev)
    m.down_proj = linear(7168, 4224, dev)
    m.act_fn = NS(beta=4.0, linear_beta=25.0)
    return m


def build_model(dev):
    routed = experts(dev)
    layers = torch.nn.ModuleList()
    for i in range(93):
        layer = Mod()
        layer.layer_idx = i
        layer._is_moe_layer = i >= 1
        layer.attn_res_block_size = 12
        layer.is_block_write_layer = i % 12 == 0
        layer.prev_valid_blocks = math.ceil(i / 12)
        for name in ("self_attention_res_norm", "mlp_res_norm", "input_layernorm", "post_attention_layernorm"):
            setattr(layer, name, rmsnorm(7168, dev))
        layer.self_attention_res_proj = linear(1, 7168, dev)
        layer.mlp_res_proj = linear(1, 7168, dev)
        layer.self_attn = mla_attention(dev) if i in FULL_IDS else kda_attention(dev)
        layer.mlp = moe(dev, routed) if i >= 1 else dense(dev)
        layers.append(layer)
    backbone = Mod()
    backbone.layers = layers
    backbone.start_layer, backbone.end_layer = 0, 93
    backbone.config = NS(
        num_hidden_layers=93, attn_res_block_size=12, activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0, rms_norm_eps=1e-5,
    )
    backbone.output_attn_res_norm = rmsnorm(7168, dev)
    backbone.output_attn_res_proj = linear(1, 7168, dev)
    backbone.norm = rmsnorm(7168, dev)
    lm = Mod()
    lm.model = backbone
    lm.config = NS(
        full_attention_layer_ids=FULL_IDS, linear_layer_ids=LINEAR_IDS,
        moe_renormalize=True, rms_norm_eps=1e-5,
    )
    top = Mod()
    top.language_model = lm
    return top


def verify_weights(model, dev, query_width=8, request_batches=(1, 2, 4), context=10240):
    """The K31 verify tensor ABI (``kimi_k31.verify_binding.VerifyWeights``) of the fake
    model, as the host adapter builds it (tuple orders: ``verify_weights.WeightModel`` and
    ``dsa.verify.BuiltSparseVerify``)."""
    from sgl_nvidia_mega.kimi_k31.projection_storage import coalesce_index_projection
    from sgl_nvidia_mega.kimi_k31.k3.projection_storage import coalesce_kda_projection_storage
    from sgl_nvidia_mega.kimi_k31.verify_binding import VerifyLayerWeights, VerifyWeights

    backbone = model.language_model.model
    layers = []
    for layer in backbone.layers:
        a = layer.self_attn
        if layer.layer_idx in FULL_IDS:
            ix = a.indexer
            # The host verify adapter packs [qkv_a | g_proj] (3648 rows; index-K is written
            # separately): the first 3648 rows of the decode build's K3.1 storage.
            packed = coalesce_index_projection(a)[:3648]
            attention = (packed, a.q_a_layernorm.weight, a.kv_a_layernorm.weight, a.q_b_proj.weight,
                         a.w_kc, a.w_vc, a.o_proj.weight, ix.wk.weight, ix.k_norm.weight,
                         ix.k_norm.bias, ix.wq_b.weight, ix.weights_proj.weight)
            scalars = (1e-5, 1e-5, a.scaling, 1.0, 1.0, 1.0, ix.k_norm.variance_epsilon)
            kind = 0
        else:
            packed = coalesce_kda_projection_storage(a)
            attention = (packed, a.qkv_conv1d.weight, a.f_b_proj.weight, a.A_log, a.dt_bias,
                         a.o_norm.weight, a.o_proj.weight)
            scalars, kind = (-5.0, a.o_norm.eps), 1
        m = layer.mlp
        if layer._is_moe_layer:
            e = m.experts
            ff = (e.w13_weight, e.w13_weight_scale, e.w2_weight, e.w2_weight_scale,
                  m.shared_experts.gate_up_proj.weight, m.shared_experts.down_proj.weight, m.gate.weight,
                  m.gate.e_score_correction_bias, m.routed_expert_down_proj.weight,
                  m.routed_expert_up_proj.weight, m.routed_expert_norm.weight)
            ffs = (1.0,)
        else:
            ff, ffs = (m.gate_up_proj.weight, m.down_proj.weight), ()
        residual = tuple(getattr(layer, n).weight for n in (
            "self_attention_res_norm", "self_attention_res_proj", "input_layernorm",
            "mlp_res_norm", "mlp_res_proj", "post_attention_layernorm"))
        layers.append(VerifyLayerWeights(
            layer_id=layer.layer_idx, attention_kind=kind, is_moe=int(layer._is_moe_layer),
            attention=attention, attention_scalars=scalars, residual=residual,
            residual_eps=(1e-5,) * 4, feed_forward=ff, feed_forward_scalars=ffs,
            previous_blocks=layer.prev_valid_blocks, writes_block=int(layer.is_block_write_layer)))
    final = (backbone.output_attn_res_norm.weight, backbone.output_attn_res_proj.weight, backbone.norm.weight)
    return VerifyWeights(
        layers=tuple(layers), embedding=rand((1024, 7168), dev), final_weights=final,
        final_eps=(1e-5, 1e-5), full_attention_ids=tuple(FULL_IDS), linear_attention_ids=tuple(LINEAR_IDS),
        attn_res_block_size=12, rms_eps=1e-5, situ_beta=4.0, situ_linear_beta=25.0, moe_renormalize=1,
        context_length=context, query_width=query_width, request_batches=tuple(request_batches))


def verify_pools(dev, pages=168, kda_cap=64, ring_len=16, scratch_rows=16, q=8):
    """``kimi_k31.verify_binding.VerifyPoolView`` over fresh pools (shapes as the host's
    ReplaySSM rings and the KDA verify test use)."""
    from sgl_nvidia_mega.kimi_k31.verify_binding import KdaVerifyPools, VerifyLayerPools, VerifyPoolView

    layers = []
    for l in range(93):
        if l in FULL_IDS:
            mla = (torch.zeros((pages + 1, 64, 576), dtype=U8, device=dev),
                   torch.zeros((pages + 1, 8448), dtype=U8, device=dev))
            layers.append(VerifyLayerPools(layer_id=l, kda=(), mla=mla))
        else:
            conv = torch.zeros((kda_cap + 1, 3, 4608), dtype=BF16, device=dev)
            pools = KdaVerifyPools(
                temporal=torch.zeros((kda_cap + 1, 12, 128, 128), dtype=F32, device=dev),
                conv=conv.transpose(-1, -2),
                rawv=torch.zeros((kda_cap + 1, 12, ring_len, 128), dtype=BF16, device=dev),
                rawk=torch.zeros((kda_cap + 1, 12, ring_len, 128), dtype=BF16, device=dev),
                log_g=torch.zeros((kda_cap + 1, 12, ring_len, 128), dtype=F32, device=dev),
                beta=torch.zeros((kda_cap + 1, 12, ring_len), dtype=F32, device=dev),
                conv_windows=torch.zeros((scratch_rows, q, 3, 4608), dtype=BF16, device=dev).transpose(-1, -2),
            )
            layers.append(VerifyLayerPools(layer_id=l, kda=(pools,), mla=()))
    return VerifyPoolView(layers=tuple(layers), page_size=64, context_capacity=pages * 64)


def pool_views(dev, pages=168, kda_cap=64):
    """Typed pool views as the MK host's HybridPoolView exposes them (providers.bind)."""
    mla = {l: torch.zeros((pages + 1, 64, 576), dtype=U8, device=dev) for l in FULL_IDS}
    conv = {l: torch.zeros((kda_cap + 1, 3, 4608), dtype=BF16, device=dev) for l in LINEAR_IDS}
    temporal = {l: torch.zeros((kda_cap + 1, 12, 128, 128), dtype=F32, device=dev) for l in LINEAR_IDS}
    index = {l: torch.zeros((pages + 1, 8448), dtype=U8, device=dev) for l in FULL_IDS}
    mla_view = NS(
        layer_mapping={l: i for i, l in enumerate(FULL_IDS)}, page_size=64,
        logical_dtype=torch.float8_e4m3fn, storage_dtype=U8,
        reserved_capacity=pages * 64, serving_capacity=pages * 64, post_capture_active=False,
        cache=lambda l: mla[l],
    )
    kda_view = NS(
        layer_mapping={l: i for i, l in enumerate(LINEAR_IDS)}, capacity=kda_cap,
        state=lambda l: NS(conv=[conv[l]], temporal=temporal[l]),
    )
    hybrid = NS(mla=mla_view, kda=kda_view, indexer=NS(caches=index, page_size=64))
    return NS(hybrid=hybrid)


def compiled_resources():
    """Driver attributes of every CuTe JIT module compiled in this process."""
    import cuda.bindings.driver as cuda
    import gc

    rows = []
    attr = cuda.CUfunction_attribute
    seen = set()
    for obj in gc.get_objects():
        try:
            lib = getattr(obj, "library", None) if "Compiled" in type(obj).__name__ else None
        except Exception:
            lib = None
        if lib is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        try:
            status, count = cuda.cuLibraryGetKernelCount(lib)
            if status != cuda.CUresult.CUDA_SUCCESS:
                continue
            status, kernels = cuda.cuLibraryEnumerateKernels(count, lib)
            for k in kernels:
                status, fn = cuda.cuKernelGetFunction(k)
                vals = {}
                for key, a in (("regs", attr.CU_FUNC_ATTRIBUTE_NUM_REGS), ("static_smem", attr.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES),
                               ("max_dyn_smem", attr.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES), ("local", attr.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES),
                               ("bin", attr.CU_FUNC_ATTRIBUTE_BINARY_VERSION)):
                    s, v = cuda.cuFuncGetAttribute(a, fn)
                    vals[key] = int(v) if s == cuda.CUresult.CUDA_SUCCESS else None
                s, name = cuda.cuFuncGetName(fn) if hasattr(cuda, "cuFuncGetName") else (None, b"?")
                vals["name"] = (name.decode() if isinstance(name, bytes) else str(name))[:120]
                rows.append(vals)
        except Exception as e:  # attribute probing must not fail the smoke
            rows.append({"error": repr(e)[:200]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--widths", default="1,2,3,4,5,6,7,8,16,32,64")
    ap.add_argument("--context", type=int, default=10240)
    ap.add_argument("--dense", action="store_true", help="context <= 2048 path (_BuiltK31) instead of sparse")
    ap.add_argument("--verify", default="", help="also build the MTP verify MK (kind 31) for these request batches, e.g. 1,2,4 (build only)")
    args = ap.parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank)))
    torch.cuda.set_device(dev)
    out = {"rank": rank, "world": world, "phases": {}, "ok": False}
    t0 = time.time()

    def phase(name, fn):
        t = time.time()
        try:
            value = fn()
            out["phases"][name] = {"ok": True, "s": round(time.time() - t, 1)}
            print(f"[rank {rank}] PHASE {name} OK {time.time() - t:.1f}s", flush=True)
            return value
        except BaseException as e:
            out["phases"][name] = {"ok": False, "s": round(time.time() - t, 1), "error": repr(e)[:2000], "trace": traceback.format_exc()[-6000:]}
            print(f"[rank {rank}] PHASE {name} FAILED: {e!r}\n{traceback.format_exc()}", flush=True)
            raise

    try:
        widths = tuple(int(w) for w in args.widths.split(","))
        p = torch.cuda.get_device_properties(dev)
        out["device"] = {"name": p.name, "cc": f"{p.major}.{p.minor}", "sms": p.multi_processor_count}
        import sgl_nvidia_mega
        import sglang

        out["paths"] = {"plugin": sgl_nvidia_mega.__file__, "sglang": sglang.__file__}
        from sgl_nvidia_mega.kimi_k31.k3 import providers, spec as k3spec
        from sgl_nvidia_mega.kimi_k31 import dsa_spec, spec as k31spec

        out["spec_isas"] = sorted(k31spec.SPEC.requires.isas)
        out["spec_resident_blocks"] = k31spec.SPEC.requires.resident_blocks
        isa = f"sm_{p.major}{p.minor}"
        if isa not in k31spec.SPEC.requires.isas:
            raise RuntimeError(f"spec refuses {isa}")
        if p.multi_processor_count < k31spec.SPEC.requires.resident_blocks:
            raise RuntimeError("spec resident_blocks exceeds SMs")
        parallel = NS(tp_size=8, tp_rank=rank, tp_group=NS(device_group=dist.group.WORLD))
        model_ctx = NS(context_length=args.context)
        # The plugin reads sglang.srt.runtime_context.get_parallel / get_model, bound at import in some modules
        # and imported inside functions in others: patch the source and every module-level binding.
        import sglang.srt.runtime_context as runtime_context

        for module in (runtime_context, providers, k31spec, dsa_spec):
            if hasattr(module, "get_parallel"):
                module.get_parallel = lambda: parallel
            if hasattr(module, "get_model"):
                module.get_model = lambda: model_ctx
        model = phase("fake_model", lambda: build_model(dev))
        torch.cuda.synchronize()
        out["model_bytes"] = torch.cuda.memory_allocated(dev)
        context = NS(widths=widths, tokens_per_request=1, binding=object(), deadline_s=30.0)
        mem0 = torch.cuda.memory_allocated(dev)
        if args.dense:
            built = phase("build_dense", lambda: k31spec._build(model, NS(text_config=NS(dsa_config=dict(
                index_topk=2048, index_num_heads=32, index_head_dim=128, index_rope_dim=0,
                index_k_norm_type="layernorm", index_topk_freq=1, indexer_use_fp8=False, dsa_layers=FULL_1IDX),
                linear_attn_config={"full_attn_layers": FULL_1IDX})), context))
        else:
            built = phase("build_sparse", lambda: dsa_spec.build_sparse_decode(model, None, context))
        out["build_bytes"] = torch.cuda.memory_allocated(dev) - mem0
        out["compilation_inventory"] = len(built.resources.compilation_inventory)
        view = pool_views(dev)
        phase("prepare_bind", lambda: k31spec._prepare(built, view))
        torch.cuda.synchronize()
        out["bind_bytes"] = torch.cuda.memory_allocated(dev) - mem0 - out["build_bytes"]
        if args.verify:
            from sgl_nvidia_mega.kimi_k31.dsa.verify import build_from_tensors

            batches = tuple(int(b) for b in args.verify.split(","))
            weights = phase("verify_weights", lambda: verify_weights(model, dev, 8, batches, args.context))
            vbuilt = phase("build_verify", lambda: build_from_tensors(weights, 30.0))
            out["verify_widths"] = list(vbuilt.widths)
            vpools = verify_pools(dev)
            phase("prepare_verify", lambda: vbuilt.prepare(vpools))
            out["verify_kda_cores"] = len(vbuilt.executor.kda_cores)
        res = phase("resources", compiled_resources)
        out["kernels"] = len(res)
        bins = sorted({r.get("bin") for r in res if "bin" in r}, key=str)
        out["binary_versions"] = bins
        out["max_regs"] = max((r.get("regs") or 0) for r in res) if res else None
        out["spills"] = [r for r in res if (r.get("local") or 0) > 0]
        with open(os.path.join(args.out, f"kernels_rank{rank}.json"), "w") as f:
            json.dump(res, f, indent=1)
        out["ok"] = True
    except BaseException:
        pass
    out["total_s"] = round(time.time() - t0, 1)
    try:  # K31MK_KERNEL_CACHE counters (hit / miss / stored / rejected / ...), when the plugin has the cache
        from sgl_nvidia_mega.common import kernel_cache

        out["kcache"] = kernel_cache.summary()
    except ImportError:
        pass
    with open(os.path.join(args.out, f"build_rank{rank}.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    oks = [None] * world
    dist.all_gather_object(oks, out["ok"])
    if rank == 0:
        print("BUILD_RESULT", json.dumps({"ok": all(oks), "per_rank": oks, "phases": out["phases"], "kernels": out.get("kernels"),
                                         "bins": out.get("binary_versions"), "spills": len(out.get("spills", [])), "total_s": out["total_s"]}), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(0 if all(oks) else 1)


if __name__ == "__main__":
    main()

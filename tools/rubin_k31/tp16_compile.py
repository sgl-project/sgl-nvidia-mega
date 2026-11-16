# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU-only sm_107a compile of the K3.1 MK attention kernels at TP16 per-rank shapes (6 heads).

No GPU: meta tensors + the plugin's ``fake`` layouts + ``cute.compile`` (CUTE_DSL_ARCH=sm_107a).
Each arm prints OK/FAIL; with CUTE_DSL_KEEP=cubin the cubins land in CUTE_DSL_DUMP_DIR for
``cuobjdump --dump-resource-usage``.

Usage (desktop venv with nvidia-cutlass-dsl[cu13]==4.8.0, cuda-python 13.4, CPU torch):
  CUTE_DSL_ARCH=sm_107a CUTE_DSL_KEEP=cubin CUTE_DSL_DUMP_DIR=OUT PYTHONPATH=<worktree> \
      python tools/rubin_k31/tp16_compile.py [heads=6] [arms ...]
Arms: mla_query1 mla_query8 index_query1 index_query8 verify_query8 value1 value8 value_q8 kda1 kda_verify8
      front16_b1 front16_b2 experts16_b1 experts16_b2 (K3's TP16 small-batch MoE kernels; heads=6 only)
      publish16_b8 route16_b8 route16_b16 situ16_b8 persistent16_b1 persistent16_b8 persistent16_b16 plan16_b16
      merged16_b33 (TP16 generic MoE path on the FP32 latent, A4; heads=6 only)
      oproj_fused1 oproj_fused8 oproj_split_mla1 tail16_b1 tail16_b2 tail16_q8 (TP16 o_proj / MoE tail; heads=6)
      oproj_fused8_xs oproj_fused8_dr oproj_fused8_xd oproj_fused5_xs (fused x_shared / drain options; heads=6)
"""

import contextlib
import os
import sys
import time
import traceback

import torch

import cuda.bindings.driver as _drv

_drv.cuInit(0)  # the DSL queries device attributes when min_blocks_per_mp > 1
torch.accelerator.device_index = lambda *_: contextlib.nullcontext()  # CPU torch: no accelerator context

import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass.cute.runtime import make_fake_stream  # noqa: E402

from sgl_nvidia_mega.kimi_k31.k3.cutedsl.common.host import fake, layout  # noqa: E402
from sgl_nvidia_mega.kimi_k31.k3.cutedsl.mla.query import MlaQuery  # noqa: E402
from sgl_nvidia_mega.kimi_k31.k3.cutedsl.mla.value import MlaValue  # noqa: E402
from sgl_nvidia_mega.kimi_k31.k3.shapes import k31_pitch, mla_pitch, tp_of_heads  # noqa: E402

META = dict(device="meta")
BF16, F32, U8 = torch.bfloat16, torch.float32, torch.uint8


def mla_weight_specs(heads):
    w_uk = torch.empty((heads, 128, 512), dtype=BF16, **META)
    w_uv = torch.empty((heads, 128, 512), dtype=BF16, **META).transpose(1, 2)  # (h, 512, 128), stride (65536, 1, 512)
    assert w_uv.stride() == (65536, 1, 512)
    tensors = (
        torch.empty(1536, dtype=BF16, **META),
        torch.empty(512, dtype=BF16, **META),
        torch.empty((heads * 192, 1536), dtype=BF16, **META),
        w_uk,
        w_uv,
        torch.empty(1, dtype=F32, **META),
        torch.empty(1, dtype=F32, **META),
        torch.empty((65, 64, 576), dtype=U8, **META),
    )
    return tuple(layout(t) for t in tensors)


def compile_query(heads, batch, q, query_type, pitch, index=False):
    specs = mla_weight_specs(heads)
    qnorm, kvnorm, qb, uk, uv, qinv, kinv, kv = (
        fake(shape, stride, dtype, 4 if dtype == F32 else 16) for shape, stride, dtype in specs
    )
    query = fake((batch, heads, 576), (heads * 576, 576, 1), U8)
    projected = fake((batch, 2112), (pitch, 1))
    slots = fake((batch,), (1,), torch.int64, 8)
    kernel = query_type(heads, batch, q_b_cp_async=heads == 12 and batch == 1, dcp=0)
    args = (projected, qnorm, kvnorm, qb, uk, kv, slots, qinv, kinv, cutlass.Float32(1e-6), query)
    if index:  # folded index-K side output: cache [pages, 8448], LayerNorm weight/bias, epsilon
        args += tuple(
            fake(shape, None, dtype, 4)
            for shape, dtype in (((65, 8448), U8), ((128,), F32), ((128,), F32), ((1,), F32))
        )
    return cute.compile(kernel, *args, make_fake_stream())


def compile_value(heads, rows, q, pitch):
    latent = fake((rows, heads, 512), (16 * 512, 512, 1))
    w_uv = fake((heads, 512, 128), (65536, 1, 512))
    gate = fake((rows, heads * 128), (pitch, 1))
    lengths = fake((rows // q,), (1,), torch.int32, 4)
    output = fake((rows, heads * 128))
    return cute.compile(MlaValue(heads, rows, q), latent, w_uv, gate, lengths, output, make_fake_stream())


def compile_kda_decode(heads, batch):
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.kda import prepared as kp
    from sgl_nvidia_mega.kimi_k31.k3.shapes import kda_pitch

    tp, dim = tp_of_heads(heads), heads * 128
    attention = type("A", (), {})()
    attention.conv1d = type("C", (), {"bias": None, "weight": torch.empty((3 * dim, 1, 4), dtype=F32, **META)})()
    attention.dt_bias = torch.empty(dim, dtype=F32, **META)
    attention.A_log = torch.empty(heads, dtype=F32, **META)
    attention.f_b_proj = type("L", (), {"weight": torch.empty((dim, 128), dtype=BF16, **META)})()
    attention.o_norm = type("N", (), {"weight": torch.empty(128, dtype=F32, **META), "eps": 1e-6})()
    attention.gate_lower_bound = -5.0
    mode = 1 if batch == 1 else 2  # GB300 decode modes (decode_mode needs a GPU to pick the SM107 retune)
    core = kp.PreparedKda(attention, {batch: (kda_pitch(tp), mode)}, False)
    caches = (
        torch.empty((5, 3, 3 * dim), dtype=BF16, **META),
        torch.empty((5, heads, 128, 128), dtype=F32, **META),
    )
    core.prepare_caches(caches)
    return core.kernels[batch]


def compile_kda_verify(heads, requests=1, queries=8):
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.kda import verify as kv
    from sgl_nvidia_mega.kimi_k31.k3.shapes import kda_pitch

    tp, dim = tp_of_heads(heads), heads * 128
    core = kv.PreparedKdaVerify(
        conv_weight=torch.empty((3 * dim, 1, 4), dtype=F32, **META),
        f_b_weight=torch.empty((dim, 128), dtype=BF16, **META),
        a_log=torch.empty(heads, **META),
        dt_bias=torch.empty(dim, **META),
        norm_weight=torch.empty(128, **META),
        lower_bound=-5.0,
        eps=1e-6,
        policies={(requests, queries): (kda_pitch(tp), 0)},
        conv_dim_first=True,
    )
    states, rows_s, ring = 2 * requests + 3, 2 * requests + 1, 16
    conv = torch.empty((states, 3, 3 * dim), dtype=BF16, **META).transpose(-1, -2)
    state = torch.empty((states, heads, 128, 128), dtype=F32, **META)
    windows = torch.empty((rows_s, queries, 3, 3 * dim), dtype=BF16, **META)
    rings = (
        torch.empty((states, heads, ring, 128), dtype=BF16, **META),
        torch.empty((states, heads, ring, 128), dtype=BF16, **META),
        torch.empty((states, heads, ring, 128), dtype=F32, **META),
        torch.empty((states, heads, ring), dtype=F32, **META),
    )
    proj = torch.empty((requests * queries, kda_pitch(tp)), dtype=BF16, **META)
    out = torch.empty((requests * queries, dim), dtype=BF16, **META)
    idx = torch.empty(requests, dtype=torch.int32, **META)
    ops = core._verify_operands((conv, state), proj, idx, out, idx, windows, rings)
    specs = tuple(map(layout, ops))
    return kv.executable(specs, heads, requests, queries, 0, False, 0, (0, 8, 2, 8))


def _target_sms(sms):
    """The target's SM count: ``sms``, else ``K31_TARGET_SMS`` (no GPU here to ask)."""
    if sms is None:
        sms = int(os.environ["K31_TARGET_SMS"])
    return sms


def compile_small_batch(kind, batch, sms=None):
    """K3's TP16 small-batch MoE kernels with the tensor layouts SmallBatchBox / SmallBatchChain pass at TP16."""
    sms = _target_sms(sms)
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe.small_batch import chain
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe.tail_runtime import POLL_CAP

    u32, i32 = torch.uint32, torch.int32
    t = lambda shape, dtype=BF16: torch.empty(shape, dtype=dtype, **META)  # noqa: E731
    if kind == "front16":
        tensors = (
            t((896, 7168)), t((896,), F32), t((2, 1, 4, 7168)), t((256,), u32),  # router, bias, x_moe, epoch
            t((2 * batch * 896 * 2,), u32), t((batch * 16,), i32), t((3, POLL_CAP, 16)),  # scores, ids, w
            # down rows, h2 (FP32 latent rows in BF16 slots, A4), shared gate/up
            t((3584 // 16, 7168)), t((2, POLL_CAP, 7168)), t((768, 7168)),
            t((2 * batch * 768,), u32), t((7168, 384)), t((3, POLL_CAP, 7168)), t((sms + 16,), i32),
        )
        grid = sms - 7 * batch if batch == 1 else min(sms - 7 * batch, 160)
        return chain._compile("front16", tuple(map(layout, tensors)), 0, batch, grid=grid, renorm=True)
    tensors = (
        t((batch * 3584,), F32), t((batch * 16,), i32), t((2, 128), U8),  # FP32 latent rows (A4), ids, TMA maps
        t((896 * 57344,), U8), t((896 * 28672,), U8), t((batch * 16, 3584)),  # w13 / w2 scales, output
    )
    return chain._compile("experts_tp16", tuple(map(layout, tensors)), 0, batch)


def compile_generic(kind, batch, sms=None):
    """TP16 generic MoE path with the FP32 latent (A4): the front GEMM's FP32 publish (288 packed rows: 56 router
    + 8 pad + 224 latent), the polled routing kernel's FP32 copy role, the persistent experts on FP32 x, the plan
    writer from FP32, and the merged shared gate/up + front kernel (B33+)."""
    sms = _target_sms(sms)
    import importlib.util

    if importlib.util.find_spec("flashinfer") is None:  # CPU venv: gemm.splitk imports FlashInfer names unused here
        import types

        name = "flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk"
        stub = types.ModuleType(name)
        stub.OWNER_RANK, stub.SplitKTactic, stub._store_shared_remote_v4 = 0, object, None
        sys.modules[name] = stub
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import experts_runtime, front, shared

    n, valid, shard, pitch = 288, 56, 224, 7168 + 2 * 896
    if kind == "publish":
        return front.compile_publish(n, n - shard, valid, batch, 0, front.front_ks(batch, n))
    if kind == "route":
        bounded = batch in (8, 16)
        return front.compile_route(batch, 384, True, (True, True, False), 0, True, batch > 8, 10**9, bounded, True)
    if kind == "situ":
        return front.compile_route(batch, 384, True, (False, False, True), 0, deadline_ns=10**9, fp32=True)
    if kind == "experts":
        knobs = (experts_runtime.streamed(batch), experts_runtime.pregrouped(16, batch),
                 experts_runtime.stealing(batch))
        return experts_runtime.compile_experts(16, 896, batch, sms, 0, False, *knobs, activation_floor=False)
    if kind == "plan":
        return experts_runtime.compile_plan(batch, 0, True)
    return shared.compile_shared(16, batch, 0, True, (n, n - shard, valid, pitch))


def compile_tp16_comm(kind, batch, **options):
    """TP16 o_proj (OprojFused for KDA B<=8, split producer/consumer for MLA B<5) and the clustered MoE tail.
    ``options``: the fused kernel's x_shared / drain."""
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.moe import tail_runtime
    from sgl_nvidia_mega.kimi_k31.k3.cutedsl.oproj.prepared import compile_oproj

    deadline = 10**9
    if kind == "fused":  # (kind, k, batch, tp, nb, has_delta, mla, front, cap, twoshot, deadline)
        return compile_oproj("fused", 768, batch, 16, 1, True, False, False, 64, False, deadline, **options)
    if kind == "split_mla":
        compile_oproj("producer", 768, batch, 16, 1, True, True, False, 64, False, deadline)
        return compile_oproj("consumer", 768, batch, 16, 1, True, True, False, 64, False, deadline)
    tail_runtime.prepare_shared_carveout = lambda code, *_: code  # a driver attribute call (needs a GPU)
    mode = "front_poll" if kind == "tail_poll" else "front"
    return tail_runtime.compile_tail(16, batch, 0, mode, 2, deadline_ns=deadline)


def main():
    heads = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    tp = tp_of_heads(heads)
    from sgl_nvidia_mega.kimi_k31.folded_index import IndexMlaQuery

    arms = {
        "mla_query1": lambda: compile_query(heads, 1, 1, MlaQuery, mla_pitch(tp)),
        "mla_query8": lambda: compile_query(heads, 8, 1, MlaQuery, mla_pitch(tp)),
        "index_query1": lambda: compile_query(heads, 1, 1, IndexMlaQuery, k31_pitch(tp), index=True),
        "index_query8": lambda: compile_query(heads, 8, 1, IndexMlaQuery, k31_pitch(tp), index=True),
        "verify_query8": lambda: compile_query(heads, 8, 8, MlaQuery, mla_pitch(tp)),
        "value1": lambda: compile_value(heads, 1, 1, k31_pitch(tp)),
        "value8": lambda: compile_value(heads, 8, 1, k31_pitch(tp)),
        "value_q8": lambda: compile_value(heads, 8, 8, mla_pitch(tp)),
        "kda1": lambda: compile_kda_decode(heads, 1),
        "kda_verify8": lambda: compile_kda_verify(heads),
        "front16_b1": lambda: compile_small_batch("front16", 1),
        "front16_b2": lambda: compile_small_batch("front16", 2),
        "experts16_b1": lambda: compile_small_batch("experts16", 1),
        "experts16_b2": lambda: compile_small_batch("experts16", 2),
        "publish16_b8": lambda: compile_generic("publish", 8),
        "route16_b8": lambda: compile_generic("route", 8),
        "route16_b16": lambda: compile_generic("route", 16),
        "situ16_b8": lambda: compile_generic("situ", 8),
        "persistent16_b1": lambda: compile_generic("experts", 1),
        "persistent16_b8": lambda: compile_generic("experts", 8),
        "persistent16_b16": lambda: compile_generic("experts", 16),
        "plan16_b16": lambda: compile_generic("plan", 16),
        "merged16_b33": lambda: compile_generic("merged", 33),
        "oproj_fused1": lambda: compile_tp16_comm("fused", 1),
        "oproj_fused8": lambda: compile_tp16_comm("fused", 8),
        "oproj_fused8_xs": lambda: compile_tp16_comm("fused", 8, x_shared=True),
        "oproj_fused8_dr": lambda: compile_tp16_comm("fused", 8, drain=True),
        "oproj_fused8_xd": lambda: compile_tp16_comm("fused", 8, x_shared=True, drain=True),
        "oproj_fused5_xs": lambda: compile_tp16_comm("fused", 5, x_shared=True),
        "oproj_split_mla1": lambda: compile_tp16_comm("split_mla", 1),
        "tail16_b1": lambda: compile_tp16_comm("tail_poll", 1),
        "tail16_b2": lambda: compile_tp16_comm("tail_poll", 2),
        "tail16_q8": lambda: compile_tp16_comm("tail", 8),
    }
    selected = sys.argv[2:] or list(arms)
    failures = 0
    for name in selected:
        t0 = time.time()
        try:
            arms[name]()
            print(f"OK   {name} heads={heads} {time.time() - t0:.1f}s", flush=True)
        except Exception as error:  # noqa: BLE001 - report every arm
            failures += 1
            print(f"FAIL {name} heads={heads} {type(error).__name__}: {str(error)[:600]}", flush=True)
            traceback.print_exc(limit=4)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Regenerate the Qwen3.8 executor's two tuning-pin tables for the stack this runs on.

Writes, in the format ``runtime/flashinfer_tactic_pins.py`` and ``runtime/bf16_gemm_pins.py``
read (``_metadata`` taken from this process the way the readers take it, so a server on the
same image, FlashInfer, CUDA, cuBLAS, GPU and SM accepts the files and any other refuses them):

  <out>/moe_tactic_pins.json.py  SGLANG_FLASHINFER_TACTIC_PIN_FILE: per FlashInfer MoE token
      bucket, the fastest tactic of the TRT-LLM-gen FP4 MoE (trtllm_fp4_block_scale_moe,
      SGLang's call: RenormalizeNaive routing, NVFP4 activations, deferred finalize in the
      decode buckets and fused finalize in the prefill buckets) that is not split-K, from
      every tactic FlashInfer lists valid for the bucket and its default (-1). Each tactic is
      served to the stock call through the autotuner's winner cache, as the pins are, and
      timed under CUDA graph replay. A tactic is split-K when a kernel it launches is named
      split-K with more than one split (torch profiler); those are listed under
      split_k_tactics.
  <out>/bf16_gemm_pins.json.py  SGLANG_BF16_GEMM_PIN_FILE: per (M, N, K), the fastest kernel
      among the cuBLASLt heuristic algorithms (32 MiB workspace) and SGLang's TGV tactics that
      is bit-equal to SGLang's BF16 dispatch (``_bf16_gemm_dispatch_impl`` with the server's
      BF16 backend, ``--server-args``) on two seeds, kept when at least --min-gain-pct and
      --min-gain-us faster than that dispatch under CUDA graph replay, with the weight
      rotated over copies spanning twice the L2 (the source's rule).

and prints each file's SHA-256 (``*_PIN_FILE_SHA256``). Seeds, candidate order and the
reduction (median of replays; ties keep the earlier candidate) are fixed; near-tied
tactics can still trade places between runs, so each entry records both times.

Defaults are the TP1 serving points: verify rows 4..512 in steps of 4 (the GEMM M values
and, through FlashInfer's bucket map, the MoE decode buckets) and the MoE prefill buckets
above 512 up to --moe-prefill-max. --gemm-ops picks the projections (N, K at TP1):
gdn_in_proj 16480x2560, attn_qkv 13312x2560, o_proj 2560x6144 (GDN out_proj and attention
o_proj), router 512x2560, attn_idx_qk 640x2560 are the executor's own calls (the default);
lm_head 248320x2560, hc_mix_down 320x10240 and hc_mix_up 10240x320 are launched by SGLang's
modules and take a pin only through an SGLang hook. One GPU, about 10 minutes for the
defaults (about 6 more with lm_head).

  python benchmark/tune_qwen3_8_pins.py --out DIR [--rows 4-512:4] [--moe-prefill-max 16384]
      [--gemm-ops gdn_in_proj,attn_qkv,...] [--server-args '{"bf16_gemm_backend": "auto"}']
      [--skip-moe] [--skip-gemm]

Off SM107, SGLang's dispatch takes its autotuned low-M kernel at M <= 16 under graph capture
only; the bit check here (as the reader's) runs eagerly, so such rows are not pinned there.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

import torch

GEMM_OPS = {
    "gdn_in_proj": (16480, 2560),
    "attn_qkv": (13312, 2560),
    "o_proj": (2560, 6144),
    "router": (512, 2560),
    "attn_idx_qk": (640, 2560),
    "lm_head": (248320, 2560),
    "hc_mix_down": (320, 10240),
    "hc_mix_up": (10240, 320),
}
DEFAULT_GEMM_OPS = "gdn_in_proj,attn_qkv,o_proj,router,attn_idx_qk"
WORKSPACE = 32 << 20
ALGO_BYTES = 64
SEEDS = (0x4442, 0x4443)
MAX_DECODE_BUCKET = 512
# A kernel name's split-K marker; a count of 1 is not a split
SPLIT_K = re.compile(r"split_?k(\d*)", re.IGNORECASE)


def parse_rows(spec: str) -> list[int]:
    """ "4-512:4" -> [4, 8, ..., 512]; "36,40" -> [36, 40]."""
    rows = []
    for part in spec.split(","):
        if "-" in part:
            span, _, step = part.partition(":")
            first, last = (int(v) for v in span.split("-"))
            rows += range(first, last + 1, int(step or 1))
        else:
            rows.append(int(part))
    return sorted(set(rows))


def graph_us(fn, inner: int, reps: int) -> float:
    """Median microseconds per call of ``fn(i)`` (i = 0..inner-1) under CUDA graph replay."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for i in range(inner):  # warm-up outside the capture
            fn(i)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for i in range(inner):
            fn(i)
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    times = []
    for _ in range(reps):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1e3 / inner)
    del graph
    return statistics.median(times)


def adaptive_inner(fn, target_us: float = 400.0, cap: int = 64) -> int:
    """Calls per graph so one replay lasts about ``target_us``."""
    fn(0)
    torch.cuda.synchronize()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    fn(0)
    end.record()
    end.synchronize()
    once = max(start.elapsed_time(end) * 1e3, 1.0)
    return max(2, min(cap, int(target_us // once)))


def publish_sglang(server_args: dict) -> None:
    """The server's process config, so SGLang's BF16 dispatch and MoE glue pick what a
    server with these arguments picks."""
    from sglang.srt.layers.quantization.unquant import initialize_bf16_gemm_config
    from sglang.srt.runtime_context import SpawnRanks, publish
    from sglang.srt.server_args import ServerArgs

    fields = {"model_path": "dummy", **server_args}
    publish(
        ServerArgs(**fields), role="scheduler", ranks=SpawnRanks(world_rank=0, gpu_id=0)
    )
    initialize_bf16_gemm_config()


def metadata() -> dict:
    from sgl_nvidia_mega.qwen3_8.runtime.flashinfer_tactic_pins import current_metadata

    return current_metadata()


def write(path: Path, document: dict) -> str:
    text = json.dumps(document, indent=1) + "\n"
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


# -- BF16 GEMMs ----------------------------------------------------------------------


class Gemm:
    """One (N, K): weight copies spanning twice the L2, the stock dispatch and the
    candidate kernels."""

    def __init__(self, n: int, k: int, device):
        from flashinfer.jit.gemm import gen_mm_bf16_cublaslt_module
        from sglang.kernels.ops.gemm import cutedsl_bf16_gemm
        from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl

        self.n, self.k, self.device = n, k, device
        l2 = torch.cuda.get_device_properties(device).L2_cache_size
        copies = max(2, math.ceil(2 * l2 / (n * k * 2)))
        gen = torch.Generator(device=device).manual_seed(n * 131 + k)
        self.weights = [
            (torch.randn(n, k, generator=gen, device=device) * 0.02).to(torch.bfloat16)
            for _ in range(copies)
        ]
        self.stock = lambda x, w: _bf16_gemm_dispatch_impl(x, w, None)
        self.lt = gen_mm_bf16_cublaslt_module().build_and_load()
        self.tgv = cutedsl_bf16_gemm
        self.workspace = torch.empty(WORKSPACE, dtype=torch.uint8, device=device)
        with torch.cuda.device(device):
            self.handle = torch.cuda.current_blas_handle()

    def algos(self, x, out) -> list[bytes]:
        """The cuBLASLt heuristic's algorithms for this exact problem, serialized."""
        from flashinfer.gemm.gemm_base import _CUBLASLT_MAX_ALGOS

        buf = torch.empty(_CUBLASLT_MAX_ALGOS * ALGO_BYTES, dtype=torch.uint8)
        count = self.lt.mm_bf16_cublaslt_get_algos(
            x, self.weights[0], None, out, self.workspace, self.handle, buf
        )
        raw = buf.numpy().tobytes()
        return [raw[i * ALGO_BYTES : (i + 1) * ALGO_BYTES] for i in range(count)]

    def cublaslt(self, algo: bytes):
        algo_buf = torch.frombuffer(bytearray(algo), dtype=torch.uint8).clone()

        def run(x, w):
            out = torch.empty(
                (x.shape[0], w.shape[0]), dtype=torch.bfloat16, device=x.device
            )
            self.lt.mm_bf16_cublaslt_run_with_algo(
                x, w, None, out, self.workspace, self.handle, algo_buf, 0
            )
            return out

        return run

    def tgv_tactic(self, tactic: int):
        def run(x, w):
            out = torch.empty(
                (x.shape[0], w.shape[0]), dtype=torch.bfloat16, device=x.device
            )
            self.tgv._run_tgv(x, w.t(), None, out, True, tactic)
            return out

        return run

    def candidates(self, m: int):
        x = torch.empty(m, self.k, dtype=torch.bfloat16, device=self.device)
        out = torch.empty(m, self.n, dtype=torch.bfloat16, device=self.device)
        found = [
            (
                {"kind": "cublaslt", "algo": a.hex(), "workspace": WORKSPACE},
                self.cublaslt(a),
            )
            for a in self.algos(x, out)
        ]
        tactics = len(self.tgv._TGV_CUTE_EXT_TACTIC_CONFIGS)
        found += [
            ({"kind": "tgv", "tactic": t}, self.tgv_tactic(t)) for t in range(tactics)
        ]
        return found

    def bit_equal(self, run, m: int) -> bool:
        for seed in SEEDS:
            gen = torch.Generator(device=self.device).manual_seed(seed + m)
            x = torch.randn((m, self.k), device=self.device, generator=gen).to(
                torch.bfloat16
            )
            try:
                got = run(x, self.weights[0])
            except Exception:  # noqa: BLE001 - a tactic that does not take this shape
                return False
            ref = self.stock(x, self.weights[0])
            if not torch.equal(got.view(torch.int16), ref.view(torch.int16)):
                return False
        return True

    def time(self, run, m: int, reps: int) -> float:
        x = torch.randn(m, self.k, device=self.device).to(torch.bfloat16)
        copies = len(self.weights)
        call = lambda i: run(x, self.weights[i % copies])
        return graph_us(call, max(copies, adaptive_inner(call)), reps)


def tune_gemms(args, device) -> list[dict]:
    pins = []
    for op in args.gemm_ops.split(","):
        n, k = GEMM_OPS[op]
        gemm = Gemm(n, k, device)
        for m in args.rows:
            stock_us = gemm.time(gemm.stock, m, args.reps)
            best = None
            for entry, run in gemm.candidates(m):
                if not gemm.bit_equal(run, m):
                    continue
                us = gemm.time(run, m, args.reps)
                if best is None or us < best[0]:
                    best = (us, entry)
            gain = None if best is None else stock_us - best[0]
            if (
                best is not None
                and gain >= args.min_gain_us
                and (gain >= stock_us * args.min_gain_pct / 100)
            ):
                pins.append(
                    {
                        "op": op,
                        "m": m,
                        "n": n,
                        "k": k,
                        **best[1],
                        "us": round(best[0], 3),
                        "stock_us": round(stock_us, 3),
                    }
                )
            found = "none" if best is None else f"{best[0]:.2f} us {best[1]['kind']}"
            print(
                f"bf16 {op} M={m}: stock {stock_us:.2f} us, best bit-equal {found}",
                flush=True,
            )
        del gemm
        torch.cuda.empty_cache()
    return pins


# -- MoE tactics ---------------------------------------------------------------------


class Moe:
    """SGLang's trtllm_fp4_block_scale_moe call on random NVFP4 experts of one shape."""

    def __init__(self, shape, device):
        hidden, inter, experts, top_k = shape
        self.shape, self.device = tuple(shape), device
        gen = torch.Generator(device=device).manual_seed(7)
        u8 = lambda *s: torch.randint(
            0, 256, s, dtype=torch.uint8, device=device, generator=gen
        )
        ones = lambda *s: torch.ones(s, device=device).to(torch.float8_e4m3fn)
        self.weights = dict(
            gemm1_weights=u8(experts, 2 * inter, hidden // 2),
            gemm1_weights_scale=ones(experts, 2 * inter, hidden // 16),
            gemm2_weights=u8(experts, hidden, inter // 2),
            gemm2_weights_scale=ones(experts, hidden, inter // 16),
            output1_scale_scalar=torch.ones(experts, device=device),
            output1_scale_gate_scalar=torch.ones(experts, device=device),
            output2_scale_scalar=torch.ones(experts, device=device),
        )
        self.gen = gen

    def inputs(self, tokens: int):
        hidden, _, experts, _ = self.shape
        return dict(
            routing_logits=torch.randn(
                tokens, experts, device=self.device, generator=self.gen
            ).to(torch.bfloat16),
            hidden_states=torch.randint(
                0,
                256,
                (tokens, hidden // 2),
                dtype=torch.uint8,
                device=self.device,
                generator=self.gen,
            ),
            hidden_states_scale=torch.ones(
                tokens * hidden // 16, device=self.device
            ).to(torch.float8_e4m3fn),
        )

    def call(self, inputs, *, finalize: bool, tune_max: int):
        from flashinfer.fused_moe import RoutingMethodType, trtllm_fp4_block_scale_moe
        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
            trtllm_moe_enable_pdl,
        )

        hidden, inter, experts, top_k = self.shape
        tokens = inputs["hidden_states"].shape[0]
        return trtllm_fp4_block_scale_moe(
            routing_bias=None,
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_bias=None,
            num_experts=experts,
            top_k=top_k,
            n_group=None,
            topk_group=None,
            intermediate_size=inter,
            local_expert_offset=0,
            local_num_experts=experts,
            routed_scaling_factor=None,
            routing_method_type=RoutingMethodType.RenormalizeNaive,
            do_finalize=finalize,
            enable_pdl=trtllm_moe_enable_pdl(tokens),
            tune_max_num_tokens=tune_max,
            **self.weights,
            **inputs,
        )


def moe_entries(tuner, op, shape, finalize: bool):
    from sgl_nvidia_mega.qwen3_8.runtime.flashinfer_tactic_pins import (
        cache_key_bucket,
        cache_key_shape,
    )

    hidden = shape[0]
    found = {}
    for key in tuner._winner_cache():
        if key.custom_op != op or cache_key_shape(key) != shape:
            continue
        # the output placeholder: (rows, hidden) when the op finalizes, (rows, 0) deferred
        width = key.nearest_profile[0][-1] if key.nearest_profile[0] else 0
        if (width == hidden) == finalize:
            found[cache_key_bucket(key)] = key
    return found


def kernel_names(fn) -> list[str]:
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sorted({e.name for e in prof.events() if e.device_type.name == "CUDA"})


def is_split_k(names) -> bool:
    for name in names:
        for count in SPLIT_K.findall(name):
            if not count or int(count) > 1:
                return True
    return False


def tune_moe(args, device) -> dict:
    from flashinfer.autotuner import AutoTuner, autotune
    from flashinfer.fused_moe.backends.trtllm.sm100_runner import MoERunner
    from flashinfer.fused_moe.utils import (
        get_hybrid_num_tokens_buckets,
        map_to_hybrid_bucket_uncapped,
    )

    from sgl_nvidia_mega.qwen3_8.runtime.flashinfer_tactic_pins import (
        DEFAULT_TACTIC,
        MOE_OP,
        normalize_tactic,
        valid_tactics_by_bucket,
    )

    shape = tuple(int(v) for v in args.moe_shape.split(","))
    moe = Moe(shape, device)
    tuner = AutoTuner.get()
    decode = sorted({map_to_hybrid_bucket_uncapped(r) for r in args.rows})
    decode = [b for b in decode if b <= MAX_DECODE_BUCKET]
    prefill = [
        b
        for b in get_hybrid_num_tokens_buckets(args.moe_prefill_max)
        if b > MAX_DECODE_BUCKET
    ]
    split_k, tables = set(), {}
    for phase, buckets, finalize in (
        ("decode", decode, False),
        ("prefill", prefill, True),
    ):
        if not buckets:
            continue
        top = buckets[-1]
        with autotune(True):  # FlashInfer's own pass creates the entries the pins edit
            moe.call(moe.inputs(top), finalize=finalize, tune_max=top)
        torch.cuda.synchronize()
        entries = moe_entries(tuner, MOE_OP, shape, finalize)
        valid = valid_tactics_by_bucket(MoERunner.valid_tactics_dict, shape)
        table, seen = {}, {}
        for bucket in buckets:
            key = entries.get(bucket)
            if key is None:
                print(
                    f"moe {phase} bucket {bucket}: no tuner entry, skipped", flush=True
                )
                continue
            inputs = moe.inputs(bucket)
            run = lambda i: moe.call(inputs, finalize=finalize, tune_max=top)
            tuned, profile = tuner._winner_cache()[key]
            timed = []
            for tactic in [DEFAULT_TACTIC, *sorted(valid.get(bucket, ()))]:
                with tuner._lock:
                    tuner._winner_cache()[key] = (
                        DEFAULT_TACTIC if tactic == DEFAULT_TACTIC else list(tactic),
                        profile,
                    )
                if tactic != DEFAULT_TACTIC and tactic not in seen:
                    seen[tactic] = is_split_k(kernel_names(lambda: run(0)))
                if seen.get(tactic):
                    split_k.add(tactic)
                    continue
                timed.append((graph_us(run, args.moe_inner, args.reps), tactic))
            with tuner._lock:
                tuner._winner_cache()[key] = (tuned, profile)
            us, best = min(timed, key=lambda t: t[0])
            table[str(bucket)] = best if best == DEFAULT_TACTIC else list(best)
            print(
                f"moe {phase} bucket {bucket}: {best} {us:.1f} us "
                f"(tuner {normalize_tactic(tuned)}, {len(timed)} timed)",
                flush=True,
            )
        tables[phase] = table
    hidden, inter, experts, top_k = shape
    return {
        "hidden_size": hidden,
        "intermediate_size": inter,
        "num_experts": experts,
        "top_k": top_k,
        **tables,
        "split_k_tactics": sorted(list(t) for t in split_k),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", type=parse_rows, default=parse_rows("4-512:4"))
    parser.add_argument("--gemm-ops", default=DEFAULT_GEMM_OPS)
    parser.add_argument(
        "--moe-shape",
        default="2560,640,512,10",
        help="hidden,intermediate,experts,top_k at TP1",
    )
    parser.add_argument("--moe-prefill-max", type=int, default=16384)
    parser.add_argument("--moe-inner", type=int, default=4)
    parser.add_argument("--reps", type=int, default=15)
    parser.add_argument("--min-gain-pct", type=float, default=3.0)
    parser.add_argument("--min-gain-us", type=float, default=0.15)
    parser.add_argument(
        "--server-args",
        type=json.loads,
        default={},
        help="ServerArgs fields as JSON, e.g. the BF16 backend",
    )
    parser.add_argument("--skip-moe", action="store_true")
    parser.add_argument("--skip-gemm", action="store_true")
    args = parser.parse_args()
    unknown = set(args.gemm_ops.split(",")) - set(GEMM_OPS)
    if unknown:
        parser.error(f"unknown --gemm-ops {sorted(unknown)}; known: {sorted(GEMM_OPS)}")

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    publish_sglang(args.server_args)
    args.out.mkdir(parents=True, exist_ok=True)
    meta = metadata()
    print(f"_metadata: {meta}", flush=True)
    if not args.skip_moe:
        document = {
            "_metadata": meta,
            "trtllm_fp4_block_scale_moe": tune_moe(args, device),
        }
        path = args.out / "moe_tactic_pins.json.py"
        print(
            f"{path}  SGLANG_FLASHINFER_TACTIC_PIN_FILE_SHA256={write(path, document)}"
        )
    if not args.skip_gemm:
        document = {"_metadata": meta, "bf16_gemm_pins": tune_gemms(args, device)}
        path = args.out / "bf16_gemm_pins.json.py"
        print(f"{path}  SGLANG_BF16_GEMM_PIN_FILE_SHA256={write(path, document)}")


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""TRT-LLM-shaped views of SGLang's Muse Spark decoder layers for the vendored host code.

``text_layers`` makes one facade per decoder layer with the attributes ``serving.ServedMegakernel``,
``layer_mk.moe_inputs_from_layer`` and ``rl.LayerSlabs`` read, from SGLang's loaded tensors (local, no communication):

* attention: the QKV weight in TRT-LLM's checkpoint row order (SGLang's is already heads first and the vendored packers
  reorder it heads first, so the facade undoes SGLang's order and the packed bytes come out the same), o_proj, the FP32
  RoPE table as ``[P, 2, 32]``, the window, the q scale and eps; no KV scale (SGLang's FP8 pool is unscaled);
* norms: the raw ``.weight`` (the kernel adds the Gemma 1), except ``post_ffn_norm.gain()`` (SGLang's effective weight);
* MoE: router, pool / unpool, expert bias, the residual gates' coefficients (SGLang's own, same formula), and
  ``_replicated_experts.stacks`` filled by ``fill_expert_stacks`` (collective).

``fill_expert_stacks`` makes each replaced layer's three-window stack ``[own | rank r-1 | rank r-2]`` of 96 experts.
SGLang's own 32 experts (already in the kernel's layout: the same w3/w1 reorder, shuffle and scale interleave) are
copied in and SGLang's parameters rebound onto the stack's first window, so one copy serves both paths; the two
predecessor windows come over NCCL from the ranks that own them. The FP32 expert scalars and the activations' global
scale come from the input scales' maxima over all 256 experts (SGLang's own use the local maxima and per-token
activation scales).

Layers 0 and 61 stay SGLang's; their facades carry what the replaced layers read from them (``is_sliding``, layer 61's
pre-QKV norm for layer 60's boundary).
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Dict, List

import torch
import torch.distributed as dist

from ..vendor.trtllm_ab93b78.megakernel import experts as mk_experts
from ..vendor.trtllm_ab93b78.megakernel import layer_mk, moe_hops, qkv

LAYERS = 62
RANKS = 8
LOCAL = mk_experts.EXPERTS // RANKS  # experts per rank (EP8)
QKV_ROWS = 2304
#: stack key -> SGLang's parameter on ``MuseSpark1xFp4Experts``
STACK_PARAMS = (
    ("w3_w1_weight", "gate_up_proj"),
    ("w3_w1_weight_scale", "gate_up_proj_weight_scale"),
    ("w2_weight", "down_proj"),
    ("w2_weight_scale", "down_proj_weight_scale"),
)


def _checkpoint_rows(device: torch.device) -> torch.Tensor:
    """Row indices that take a heads-first QKV weight back to the checkpoint order (``layer_mk.heads_first_qkv``'s
    inverse)."""
    blocks = torch.argsort(torch.tensor(layer_mk.HEADS_FIRST_BLOCKS))
    return (
        (blocks.view(-1, 1) * 64 + torch.arange(64).view(1, -1)).reshape(-1).to(device)
    )


class _QkvCheckpointOrder:
    """``qkv_proj``: ``.weight`` is SGLang's heads-first weight in the checkpoint row order, made on each read (the
    vendored code packs it once in ``prepare``; ``_build`` only checks its shape, the packs hold the bytes)."""

    def __init__(self, w: torch.Tensor):
        assert w.dtype == torch.bfloat16 and tuple(w.shape) == (
            QKV_ROWS,
            moe_hops.HIDDEN,
        ), (w.dtype, tuple(w.shape))
        self._w = w

    @property
    def weight(self) -> torch.Tensor:
        return self._w.index_select(0, _checkpoint_rows(self._w.device))


class _Gate:
    """A residual gate's ``coefficients()``: SGLang's precomputed FP32 alpha and beta."""

    def __init__(self, gate):
        a, b = gate._alpha, gate._beta
        assert a is not None and b is not None, (
            "the residual gate's coefficients are not precomputed"
        )
        assert a.numel() == moe_hops.HIDDEN and b.numel() == moe_hops.HIDDEN, (
            a.shape,
            b.shape,
        )
        self._a, self._b = a.float().reshape(-1), b.float().reshape(-1)

    def coefficients(self):
        return self._a, self._b


class _Gain:
    """``post_ffn_norm``: ``gain()`` = SGLang's effective weight (weight + 1)."""

    def __init__(self, norm):
        self._norm = norm

    def gain(self) -> torch.Tensor:
        return self._norm._get_effective_weight()


def _eps(module) -> float:
    return float(getattr(module, "eps", float("nan")))


def _close(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=0.0)


def layer_reasons(index: int, sl) -> List[str]:
    """Why SGLang's layer ``index`` does not match what the kernel compiles in (empty: it matches)."""
    reasons = []
    attn = sl.self_attn
    sliding = int(getattr(attn.attn, "sliding_window_size", -1) or -1) > 0
    checks = [
        ("input_layernorm eps", _eps(sl.input_layernorm), qkv.PRE_QKV_EPS),
        ("input_layernorm eps", _eps(sl.input_layernorm), moe_hops.PRE_EPS),
        ("post_attn_norm eps", _eps(sl.post_attn_norm), moe_hops.POST_EPS),
        (
            "pre_feedforward_layernorm eps",
            _eps(sl.pre_feedforward_layernorm),
            moe_hops.PRE_EPS,
        ),
        (
            "post_feedforward_layernorm eps",
            _eps(sl.post_feedforward_layernorm),
            moe_hops.POST_EPS,
        ),
    ]
    for norm in (
        "input_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
    ):
        center = getattr(getattr(sl, norm), "gain_center", None)
        if center != 1.0:
            reasons.append(
                f"layer {index}: {norm} gain center {center} (the kernel adds 1)"
            )
    if 1 <= index <= LAYERS - 2:
        mlp = sl.mlp
        checks += [
            (
                "pre_expert_norm eps",
                _eps(mlp.pre_expert_norm),
                mk_experts.PRE_EXPERT_EPS,
            ),
            (
                "post_expert_norm eps",
                _eps(mlp.experts.post_expert_norm),
                mk_experts.POST_EXPERT_EPS,
            ),
            ("route eps", mlp.route_eps, mk_experts.RENORM_EPS),
            ("router multiplier", mlp.router_multiplier, moe_hops.ROUTER_SCALE),
            ("route scale", mlp.route_scale, 1.0),
            ("top k", mlp.top_k, mk_experts.TOP_K),
            ("experts", mlp.num_experts, mk_experts.EXPERTS),
            ("local experts", mlp.num_local_experts, LOCAL),
        ]
        for norm in (mlp.pre_expert_norm, mlp.experts.post_expert_norm):
            if getattr(norm, "gain_center", None) != 1.0:
                reasons.append(
                    f"layer {index}: expert norm gain center {norm.gain_center} (the kernel adds 1)"
                )
        ex = mlp.experts
        if type(ex).__name__ != "MuseSpark1xFp4Experts":
            reasons.append(
                f"layer {index}: experts {type(ex).__name__} (the kernel runs the NVFP4 experts)"
            )
        else:
            h, i = (
                mk_experts.HIDDEN,
                mk_experts.HIDDEN,
            )  # the experts' hidden and intermediate sizes
            want = dict(
                gate_up_proj=((LOCAL, 2 * i, h // 2), torch.uint8),
                gate_up_proj_weight_scale=(
                    (LOCAL, 2 * i, h // 16),
                    torch.float8_e4m3fn,
                ),
                down_proj=((LOCAL, h, i // 2), torch.uint8),
                down_proj_weight_scale=((LOCAL, h, i // 16), torch.float8_e4m3fn),
                gate_up_proj_weight_scale_2=((LOCAL, 2), torch.float32),
                down_proj_weight_scale_2=((LOCAL,), torch.float32),
                gate_up_proj_input_scale=((LOCAL, 2), torch.float32),
                down_proj_input_scale=((LOCAL,), torch.float32),
            )
            for name, (shape, dtype) in want.items():
                t = getattr(ex, name, None)
                if t is None or tuple(t.shape) != shape or t.dtype != dtype:
                    got = None if t is None else (tuple(t.shape), t.dtype)
                    reasons.append(
                        f"layer {index}: experts.{name} {got} (want {shape} {dtype}: the trtllm-gen layout)"
                    )
        if mlp.gate.weight.dtype != torch.float32:
            reasons.append(
                f"layer {index}: router {mlp.gate.weight.dtype} (the kernel reads the FP32 router)"
            )
        if sliding:
            rope = attn.rotary_emb
            if (
                rope is None
                or rope.cos_sin_cache.dtype != torch.float32
                or rope.cos_sin_cache.shape[-1] != 64
            ):
                reasons.append(
                    f"layer {index}: RoPE table {None if rope is None else rope.cos_sin_cache.dtype}"
                )
            if int(attn.attn.sliding_window_size) + 1 != 2048:
                reasons.append(
                    f"layer {index}: window {int(attn.attn.sliding_window_size) + 1}"
                )
    for what, got, want in checks:
        if not _close(got, want):
            reasons.append(f"layer {index}: {what} {got} (the kernel compiles {want})")
    return reasons


def _attention(sl) -> SimpleNamespace:
    attn = sl.self_attn
    window = int(getattr(attn.attn, "sliding_window_size", -1) or -1)
    sliding = window > 0
    eps = SimpleNamespace(variance_epsilon=float(attn.norm_eps))
    rope = None
    if sliding and attn.rotary_emb is not None:
        # a reference to SGLang's FP32 table: it stays valid if SGLang later swaps in a BF16 copy
        rope = SimpleNamespace(
            rotary_cos_sin=attn.rotary_emb.cos_sin_cache.view(-1, 2, 32)
        )
    return SimpleNamespace(
        is_sliding=sliding,
        layer_idx=int(attn.layer_idx),
        qkv_proj=_QkvCheckpointOrder(attn.qkv_proj.weight),
        o_proj=attn.o_proj,
        rotary_emb=rope,
        scale_query_by=float(attn.scale_query_by),
        q_norm=eps,
        o_head_norm=eps,
        attention_window_size=window + 1 if sliding else 0,
        attn=SimpleNamespace(kv_scale_orig_quant=None),
    )


def _feed_forward(sl) -> SimpleNamespace:
    mlp = sl.mlp
    return SimpleNamespace(
        router=mlp.gate,
        expert_bias=mlp.gate.e_score_correction_bias,
        pooler=mlp.pre_expert_proj,
        unpooler=mlp.post_expert_proj,
        pre_expert_norm=mlp.pre_expert_norm,
        post_expert_norm_weight=mlp.experts.post_expert_norm.weight,
        experts=SimpleNamespace(fc31_input_scale=None),  # fill_expert_stacks
        _replicated_experts=None,  # fill_expert_stacks
        sgl_experts=mlp.experts,
    )


def text_layers(lm) -> List[SimpleNamespace]:
    """The facades of ``lm.layers`` (``MuseSpark1xModel``); raises on a mismatch with the kernel's constants."""
    layers = list(lm.layers)
    assert len(layers) == LAYERS, len(layers)
    reasons = [r for i, sl in enumerate(layers) for r in layer_reasons(i, sl)]
    if reasons:
        raise ValueError(
            "; ".join(reasons[:8])
            + (f" (+{len(reasons) - 8} more)" if len(reasons) > 8 else "")
        )
    out = []
    for i, sl in enumerate(layers):
        f = SimpleNamespace(
            self_attn=_attention(sl),
            pre_qkv_proj_norm=sl.input_layernorm,
            sgl_layer=sl,
        )
        if 1 <= i <= LAYERS - 2:
            f.feed_forward = _feed_forward(sl)
            f.pre_ffn_norm = sl.pre_feedforward_layernorm
            f.post_ffn_norm = _Gain(sl.post_feedforward_layernorm)
            f.add_attn = _Gate(sl.post_attention_residual_gate)
            f.add_ffn = _Gate(sl.post_feedforward_residual_gate)
        out.append(f)
    return out


def fill_expert_stacks(
    layers: List[SimpleNamespace], group: dist.ProcessGroup, rank: int
) -> Dict[str, float]:
    """The three-window stacks and scalars of layers 1-60 (collective over ``group``, SGLang's TP device group, where
    rank r holds experts ``[32 r, 32 r + 32)``). Returns the memory record (GiB)."""
    world = dist.get_world_size(group)
    assert world == RANKS and dist.get_rank(group) == rank
    peers = {d: dist.get_global_rank(group, (rank + d) % RANKS) for d in (-2, -1, 1, 2)}
    blocks = [rank, (rank - 1) % RANKS, (rank - 2) % RANKS]  # the windows' owners
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    for f in layers[1:-1]:
        ff, ex = f.feed_forward, f.feed_forward.sgl_experts
        stacks: Dict[str, torch.Tensor] = {}
        for key, name in STACK_PARAMS:
            p = getattr(ex, name)
            assert p.shape[0] == LOCAL and p.is_contiguous(), (name, tuple(p.shape))
            stack = torch.empty(
                (3 * LOCAL,) + tuple(p.shape[1:]), dtype=p.dtype, device=p.device
            )
            stack[:LOCAL].copy_(p.data)
            p.data = stack[
                :LOCAL
            ]  # SGLang's experts now live in the stack's own window
            stacks[key] = stack
        ops = []
        for key, _ in STACK_PARAMS:
            s = stacks[key].view(
                torch.uint8
            )  # bytes: NCCL moves the FP8 scales as they are
            ops += [
                dist.P2POp(dist.isend, s[:LOCAL], peers[1], group),
                dist.P2POp(dist.isend, s[:LOCAL], peers[2], group),
                dist.P2POp(dist.irecv, s[LOCAL : 2 * LOCAL], peers[-1], group),
                dist.P2POp(dist.irecv, s[2 * LOCAL :], peers[-2], group),
            ]
        for req in dist.batch_isend_irecv(ops):
            req.wait()
        # the scalars: every rank's weight scales, and the input scales' maxima over all 256 experts
        ws13 = ex.gate_up_proj_weight_scale_2.float()  # [32, 2]: gate (w1), up (w3)
        ws2 = ex.down_proj_weight_scale_2.float().reshape(LOCAL)
        gathered = torch.empty(RANKS, LOCAL, 3, dtype=torch.float32, device=ws13.device)
        dist.all_gather_into_tensor(
            gathered,
            torch.cat([ws13, ws2.view(-1, 1)], dim=1).contiguous(),
            group=group,
        )
        in13 = ex.gate_up_proj_input_scale.float()
        maxima = torch.stack([in13.max(), ex.down_proj_input_scale.float().max()])
        dist.all_reduce(maxima, op=dist.ReduceOp.MAX, group=group)
        if not bool(torch.isfinite(maxima).all()) or float(maxima.min()) <= 0:
            raise ValueError(
                f"layer {f.self_attn.layer_idx}: input-scale maxima {maxima.tolist()}"
            )
        mg, md = maxima[0], maxima[1]
        w = gathered[blocks].reshape(3 * LOCAL, 3)  # the window order
        stacks["fc31_alpha"] = (w[:, 0] * mg).contiguous()
        stacks["fc31_scale_c"] = (w[:, 1] * mg / md).contiguous()
        stacks["fc2_alpha"] = (w[:, 2] * md).contiguous()
        ff.experts.fc31_input_scale = (1.0 / mg).reshape(1).contiguous()
        ff._replicated_experts = SimpleNamespace(stacks=stacks)
        mk_experts.Window(
            w13=stacks["w3_w1_weight"].view(torch.uint8),
            sf13=stacks["w3_w1_weight_scale"],
            w2=stacks["w2_weight"].view(torch.uint8),
            sf2=stacks["w2_weight_scale"],
            alpha1=stacks["fc31_alpha"],
            scale_c=stacks["fc31_scale_c"],
            alpha2=stacks["fc2_alpha"],
        ).check()
    torch.cuda.synchronize()
    grown = (torch.cuda.memory_allocated() - before) / 2**30
    # the replicas only (two windows of 864 MiB per layer) when SGLang's old expert tensors were freed
    return dict(grown_gib=grown, expected_gib=2 * 0.84375 * (LAYERS - 2))

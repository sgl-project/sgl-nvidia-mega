# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Compare mode's stage check (the check build, ``MUSE_VR_VERIFY_MK_MODE=check``): where inside a replaced layer the
kernel and the stock layer part.

The kernel side is the check build's own outputs (``moe_hops.OUTPUTS`` and ``AttnHalf.make_layer``'s ``attn_out``),
added to every layer of the compare binding (``check_outputs``); the stock side is forward hooks on the SGLang decoder
layer's modules for one forward (``StockTaps``). ``errors`` gives each stage's relative error; ``dump`` saves a layer's
tensors of both sides, its boundary parameters and its KV rows for an offline look (``torch.load``).

Stages, in a layer's order (``st_<stage>``):

* ``q`` / ``k`` / ``v`` / ``g``: the QKV projection's rows (the launch's ``io.qkv``, sliding-window layers), against
  SGLang's ``qkv_proj`` output in the checkpoint column order;
* ``attn``: the ungated attention output, this rank's 16 query heads (sliding-window layers: the one-launch full layer writes
  its attention into the arena only);
* ``res1``: the residual after the attention boundary (``MhOutRes1``, the kernel's ``res1``);
* ``xffn``: the pre-FFN norm's output;
* ``pool``: the gathered pool output, against SGLang's ``pre_expert_proj`` on the stock ``xffn``;
* ``logits``: the gathered router logits, against SGLang's router on the stock router input (``logits_fit`` after the
  least-squares scale ``logits_scale``, in case one side carries the router multiplier);
* ``m``: the experts' all-reduced sum (when SGLang's experts reduced it themselves);
* ``b``: the gathered unpool output (this rank's columns when SGLang defers the unpool gather to the FFN boundary);
* ``res2``: the residual after the FFN boundary.
"""

from __future__ import annotations

import contextlib
import os
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F

from ..trtllm_ab93b78.megakernel import layer_mk, moe_hops

ROWS = 8
ATTN_COLS = 1024  # this rank's 16 query heads of 64
QKV_COLS = 2304
#: column blocks of the checkpoint-order QKV output (q_h at 128 h, g_h at 128 h + 64, then k, v)
QKV_BLOCKS = dict(
    q=[c for h in range(16) for c in range(128 * h, 128 * h + 64)],
    g=[c for h in range(16) for c in range(128 * h + 64, 128 * h + 128)],
    k=list(range(2048, 2176)),
    v=list(range(2176, 2304)),
)
#: the check outputs the binding adds (``MhOutRes1`` and ``MhOutXattn`` are the kernel's own res1 and x_out)
MOE_OUTPUTS = ("MhOutXffn", "MhOutPool", "MhOutLogits", "MhOutM", "MhOutB", "MhOutRes2")
BUILDERS = ("make_swa_layer", "make_full_layer_one", "make_full_layer")
WINDOW = 2048
PAGE = 128


class Buffers:
    """One set of check outputs for every layer of the compare binding: the layers launch one at a time and the outputs
    are read after each launch. ``poison`` (NaN) before a launch, so an output the launch did not write shows."""

    def __init__(self, device: torch.device):
        self.moe = {
            name: torch.empty(
                moe_hops.OUTPUTS[name][0],
                dtype=moe_hops.OUTPUTS[name][1],
                device=device,
            )
            for name in MOE_OUTPUTS
        }
        self.attn = torch.empty(ROWS, ATTN_COLS, dtype=torch.bfloat16, device=device)
        self.qkv = torch.empty(
            ROWS, QKV_COLS, dtype=torch.bfloat16, device=device
        )  # the launch's io.qkv

    def poison(self) -> None:
        for t in (*self.moe.values(), self.attn, self.qkv):
            t.fill_(float("nan"))


@contextlib.contextmanager
def check_outputs(kernels, bufs: Buffers):
    """``kernels``' layer builders (``LayerKernels.make_*``, as the vendored ``_build`` calls them) with ``bufs`` as
    the check outputs of every layer they build."""
    saved = {name: kernels.__dict__.get(name) for name in BUILDERS}

    def with_outputs(name, build):
        def wrapped(*args, **kwargs):
            kwargs["check_out"] = dict(bufs.moe)
            if (
                name == "make_swa_layer"
            ):  # (attn_args, attn_kwargs, moe, ...): the attention half's attn_out
                args = (args[0], dict(args[1], attn_out=bufs.attn)) + tuple(args[2:])
            return build(*args, **kwargs)

        return wrapped

    for name in BUILDERS:
        setattr(kernels, name, with_outputs(name, getattr(kernels, name)))
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                delattr(kernels, name)  # the class's method again
            else:
                setattr(kernels, name, value)


def _clone(t):
    return t.detach().clone() if isinstance(t, torch.Tensor) else None


class StockTaps:
    """Forward hooks on one SGLang decoder layer (``MuseSpark1xDecoderLayer``) for one forward; ``out`` holds each
    stage's tensors, cloned (the stock layer updates some of them in place)."""

    def __init__(self, layer):
        self.out: Dict[str, object] = {}
        mlp = layer.mlp
        self._hooks = [
            (layer.self_attn.qkv_proj, self._qkv, False),
            (layer.self_attn.attn, self._attn, False),
            (layer.self_attn, self._branch1, False),
            (layer._fused_attn_boundary, self._boundary1, False),
            (mlp.gate, self._logits, False),
            (mlp.experts, self._experts, True),
            (mlp, self._branch2, False),
        ]
        self._handles = []

    def __enter__(self) -> StockTaps:
        for module, hook, with_kwargs in self._hooks:
            self._handles.append(
                module.register_forward_hook(hook, with_kwargs=with_kwargs)
            )
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def _qkv(self, module, args, output):  # (output, bias)
        self.out["qkv"] = _clone(output[0] if isinstance(output, tuple) else output)

    def _attn(
        self, module, args, output
    ):  # RadixAttention(q, k, v, forward_batch, ...)
        for name, t in zip(("q", "k", "v"), args[:3]):
            self.out[f"attn_{name}"] = _clone(t)
        self.out["attn"] = _clone(output.reshape(output.shape[0], -1))

    def _branch1(self, module, args, output):
        self.out["branch1"] = _clone(output)

    def _boundary1(
        self, module, args, output
    ):  # (ffn input, stream, router input or None)
        self.out["xffn"], self.out["res1"], self.out["router_in"] = (
            _clone(t) for t in output
        )

    def _logits(self, module, args, output):
        self.out["logits"] = _clone(output)

    def _experts(self, module, args, kwargs, output):
        self.out["m"] = _clone(output)
        self.out["m_reduced"] = (
            kwargs.get("ar_workspace") is not None
        )  # a fused finalize all-reduce

    def _branch2(self, module, args, output):
        self.out["branch2"] = _clone(output)


def rel(ref: torch.Tensor, got: torch.Tensor) -> float:
    a, b = ref.float(), got.float()
    return float((a - b).norm() / a.norm().clamp_min(1e-30))


@torch.no_grad()
def errors(
    layer,
    taps: StockTaps,
    bufs: Buffers,
    res1: torch.Tensor,
    stream_out: torch.Tensor,
    rank: int,
    sliding: bool,
) -> Dict[str, float]:
    """``st_<stage>``: ||kernel - stock|| / ||stock|| of each stage the stock forward showed (module docstring)."""
    t, moe, rows = taps.out, bufs.moe, {}

    def put(stage: str, ref: Optional[torch.Tensor], got: torch.Tensor) -> None:
        if ref is not None:
            rows[f"st_{stage}"] = rel(ref, got)

    if sliding:
        qkv = t.get("qkv")
        if qkv is not None:
            qkv = qkv.index_select(1, checkpoint_cols(qkv.device))
            for block, cols in QKV_BLOCKS.items():
                put(block, qkv[:, cols], bufs.qkv[:, cols])
        put("attn", t.get("attn"), bufs.attn)
    put("res1", t.get("res1"), res1)
    xffn = t.get("xffn")
    put("xffn", xffn, moe["MhOutXffn"])
    if xffn is not None:
        put("pool", F.linear(xffn, layer.mlp.pre_expert_proj.weight), moe["MhOutPool"])
        router_in = t.get("router_in")
        ref = F.linear(
            (xffn if router_in is None else router_in).float(),
            layer.mlp.gate.weight.float(),
        )
        got = moe["MhOutLogits"].t().float()
        put("logits", ref, got)
        scale = float((ref * got).sum() / (ref * ref).sum().clamp_min(1e-30))
        rows["st_logits_fit"], rows["logits_scale"] = rel(ref * scale, got), scale
    if t.get("m_reduced"):
        put("m", t.get("m"), moe["MhOutM"])
    b = t.get("branch2")
    if b is not None:
        got = moe["MhOutB"]
        if (
            b.shape[-1] != got.shape[-1]
        ):  # SGLang's TP-sliced branch (the gather deferred to the boundary)
            w = b.shape[-1]
            got = got[:, rank * w : (rank + 1) * w]
        put("b", b, got)
    put("res2", stream_out, moe["MhOutRes2"])
    return rows


def checkpoint_cols(device: torch.device) -> torch.Tensor:
    """Column indices that take SGLang's heads-first QKV output to the checkpoint order (``facades._checkpoint_rows``)."""
    blocks = torch.argsort(torch.tensor(layer_mk.HEADS_FIRST_BLOCKS))
    return (
        (blocks.view(-1, 1) * 64 + torch.arange(64).view(1, -1)).reshape(-1).to(device)
    )


def dump_layers(indices: Sequence[int], sliding: Sequence[bool]) -> set:
    """The layers dumped on the first compared forward: the first two replaced layers and the first full-attention one,
    or ``MUSE_VR_VERIFY_MK_COMPARE_DUMP`` (comma-separated layer indices)."""
    env = os.environ.get("MUSE_VR_VERIFY_MK_COMPARE_DUMP", "").strip()
    if env:
        return {int(i) for i in env.split(",") if i.strip()}
    full = [i for i in indices if not sliding[i]]
    return set(indices[:2]) | set(full[:1])


def kv_rows(
    pool, index: int, table: torch.Tensor, seq_len: int, sliding: bool
) -> Dict[str, torch.Tensor]:
    """The layer's K and V rows (FP8 bytes) of positions ``[start, seq_len)`` (the window for a sliding layer)."""
    start = max(0, seq_len - WINDOW) if sliding else 0
    pos = torch.arange(start, seq_len, device=table.device)
    loc = table[0, pos // PAGE].long() * PAGE + pos % PAGE
    k = pool.get_key_buffer(index).view(torch.uint8)
    v = pool.get_value_buffer(index).view(torch.uint8)
    return dict(
        kv_positions=pos.cpu(),
        kv_loc=loc.cpu(),
        k_rows=k[loc].cpu(),
        v_rows=v[loc].cpu(),
    )


def dump(
    path: str,
    *,
    facade,
    next_facade,
    layer,
    taps: StockTaps,
    bufs: Buffers,
    res1: torch.Tensor,
    inputs: Dict[str, torch.Tensor],
    stock_out,
    kernel_out,
    kv: Dict[str, object],
    rank: int,
    index: int,
    sliding: bool,
) -> None:
    """One layer's tensors of both sides (CPU), its boundary parameters and its KV rows, ``torch.save``d to ``path``."""

    def cpu(d):
        return {
            k: v.detach().cpu() for k, v in d.items() if isinstance(v, torch.Tensor)
        }

    ff = facade.feed_forward
    a1, b1 = facade.add_attn.coefficients()
    a2, b2 = facade.add_ffn.coefficients()
    params = dict(
        alpha1=a1,
        beta1=b1,
        alpha2=a2,
        beta2=b2,
        w_preffn=facade.pre_ffn_norm.weight,
        gain_pf=facade.post_ffn_norm.gain(),
        w_next=next_facade.pre_qkv_proj_norm.weight,
        expert_bias=ff.expert_bias,
        w_pre_expert=ff.pre_expert_norm.weight,
        w_post_expert=ff.post_expert_norm_weight,
        fc31_input_scale=ff.experts.fc31_input_scale,
    )
    attn = layer.self_attn
    scalars = dict(
        softmax_scale=float(attn.attn.scaling),
        scale_query_by=float(attn.scale_query_by),
        post_attn_eps=float(layer.post_attn_norm.eps),
        pre_ffn_eps=float(layer.pre_feedforward_layernorm.eps),
        post_ffn_eps=float(layer.post_feedforward_layernorm.eps),
        router_multiplier=float(layer.mlp.router_multiplier),
        route_scale=float(layer.mlp.route_scale),
        top_k=int(layer.mlp.top_k),
        m_reduced=bool(taps.out.get("m_reduced", False)),
    )
    kernel = dict(
        res1=res1,
        attn=bufs.attn,
        qkv=bufs.qkv,
        **bufs.moe,
        res=kernel_out[0],
        x=kernel_out[1],
    )
    stock = dict(taps.out, out_res=stock_out[0], out_x=stock_out[1])
    blob = dict(
        layer=index,
        rank=rank,
        sliding=sliding,
        scalars=scalars,
        inputs=cpu(inputs),
        stock=cpu(stock),
        kernel=cpu(kernel),
        params=cpu(params),
        kv=cpu(kv) | {k: v for k, v in kv.items() if not isinstance(v, torch.Tensor)},
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(blob, path)

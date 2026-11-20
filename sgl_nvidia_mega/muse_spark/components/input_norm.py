# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``input_norm`` (env ``MUSE_VR_INPUT_NORM=1``, default off).

Before the first Muse Spark decoder layer, SGLang applies the embedding RMSNorm (no weight), casts the result to the
fp32 residual stream, and applies the first layer's input RMSNorm to the stream (PyTorch's composite form, because the
stream is fp32 and the weight bf16): ten kernels per forward. This component runs them as three
(``csrc/input_norm.cuh``): one kernel for the embedding norm, the stream and the squares; PyTorch's own row mean of the
squares; one kernel for the rest of the input norm. Same operations, roundings and summation orders, bit-identical
outputs (tests/test_input_norm.py).

When the embeddings arrive already normalized (``input_embeds`` given, as the multimodal model wrapper passes them),
SGLang casts them to the fp32 stream and applies the first layer's input RMSNorm (eight kernels); this component then
runs one kernel for the stream and the squares (``stream_sq``), PyTorch's row mean and the same finishing kernel, and
hands the model the fp32 stream as ``input_embeds`` (so its cast is a no-op). Bit-identical as well.

Mechanism (as ``final_norm``): an AROUND hook on ``MuseSpark1xModel._forward`` that, for decode and target-verify
batches without sequence parallelism, temporarily replaces the model's embedding norm and the first layer's input norm by
stand-ins: the first returns the fp32 stream (so the model's cast is a no-op) and keeps the squares, the second
finishes the input norm. Everything else is SGLang's forward.

Sub-component ``input_norm_l0_table`` (env ``MUSE_VR_INPUT_NORM_L0_TABLE=1``, with ``input_norm``; default off), after
TRT-LLM's ``layer0_qkv_table`` fast path (``kernels/tt_table/target_l0.py``): layer 0's fused Q / K / V / gate
projection as a per-token table.
    In a target-verify step of one request (8 ids) through the multimodal wrapper, the input of layer 0's ``qkv_proj``
    is a function of each row's token id alone: the replicated embedding row, the weightless embedding norm, this
    component's stream / squares, row mean and finishing kernels, and the projection's 8-row GEMM (a fixed warp split-K
    tactic) are each per row. After the target's weights load (``ModelRunner.load_model``, before the KV cache is
    sized), the table is built by running that chain, the same calls the step makes, on every vocabulary id in calls of
    8 ids; one more entry holds the chain of an id outside the vocabulary. TP8: [202049, 2304] bf16, 0.87 GiB per rank.
    Per step (AROUND hook on ``MuseSparkForConditionalGeneration.forward``), all on the current stream: a Triton
    gather writes the step's [8, 2304] rows (the step's first kernel), then the embedding with its norm (the wrapper's
    own call) and this component's stream / squares kernel make the residual stream as served. The ``_Input``
    stand-in then leaves layer 0's attention input unwritten (no row mean, no finishing kernel): in the pinned
    sources its one reader is layer 0's ``qkv_proj``, whose GEMM the gathered rows replace (AROUND hooks on
    ``UnquantizedLinearMethod.apply`` / ``apply_into``; the rows stand in only for that exact operand). So an armed
    step runs the gather in place of the row mean, the finishing kernel and the QKV(G) GEMM. (This version runs on
    one stream. The first form forked the embedding and the norms onto a side stream joined after layer 0's
    ``o_proj``; in the verify graph the two branches did not overlap and the fork cost more than it hid.)
    Any other GEMM that gets the unwritten input first writes it (the served finish, counted "stock projection");
    a forward hook on layer 0 raises if an armed step ends layer 0 with the input still unwritten and the rows
    unused (a reader the hooks do not see).
    Every value is a bit copy of what the chain computes for that id (value-identical by construction). Install-time
    checks, all bitwise: the 8-row GEMM's row independence, the gather against the chain on 513 random 8-id batches
    (plus edge ids) in two orders, and 300 CUDA-graph replays of chain and gather on fresh ids. A failed check leaves
    the sub-component off (``bed.report``).
    ``MUSE_VR_INPUT_NORM_L0_TABLE_SHADOW=1`` (debug, eager steps only) also runs the chain on the step's ids after the
    forward and tallies bitwise equality with the gathered rows (``bed.shadow_result``).

    ``MUSE_VR_INPUT_NORM_L0_TABLE_STREAM=1`` (with the table; default off; after
    TRT-LLM's ``pfPreHoistKernel``, kernels/pf_pre): the residual stream as a table too. The wrapper's normalized
    embedding (``_NormalizedEmbedding``: the replicated lookup and ``F.rms_norm`` without weight, bf16, per row) is a
    function of each row's id alone, and this component's stream is its fp32 cast (``stream_sq``: stream = float(x)).
    The build keeps each id's normalized embedding row as well ([202049, H] bf16, 3.1 GiB per rank at H = 8192), and
    the step's one gather writes both: the QKV(G) rows and the fp32 stream rows (the bf16 -> fp32 cast is exact). The
    wrapper then gets the stream as ``input_embeds`` (its embedding call and this component's stream / squares kernel
    do not run; the model's cast to the fp32 residual dtype is a no-op on it). The squares are made only if a finish
    runs (another reader, or the stand-in stepping aside): by this component's own ``stream_sq`` on the stream's bf16
    values (an exact round trip), so the served finish gets the served squares. Install-time checks compare the
    gathered stream with the served stream (the wrapper's embedding call, then ``stream_sq``) bit for bit, with the
    QKV(G) rows, on the same ids and replays; any difference leaves the table off.
"""

from __future__ import annotations

import functools
import logging
import time
from pathlib import Path

import torch

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/layers/layernorm_sp.py": ("runs_sp",),
    "kernels/jit/utils/__init__.py": (
        "is_arch_support_pdl",
        "load_jit",
        "make_cpp_args",
    ),
    "srt/layers/muse_spark/muse_spark_layers.py|srt/layers/muse_spark_layers.py": (
        "MuseSpark1xRMSNorm",
    ),
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
    "srt/layers/quantization/unquant.py": ("UnquantizedLinearMethod",),
    "srt/models/muse_spark_omni.py": ("MuseSparkForConditionalGeneration",),
    "srt/environ.py": ("envs",),
}


logger = logging.getLogger(__name__)

NAME = "input_norm"  # env MUSE_VR_INPUT_NORM
TARGET = "sglang.srt.models.muse_spark.MuseSpark1xModel._forward"
_CUH = Path(__file__).resolve().parent / "csrc" / "input_norm.cuh"

PINS = {
    "sglang/srt/models/muse_spark.py": (
        "7e5018b3cb2ef41ee6cf09407e91e28dde069e5a33719ea58beb0d7137087355",
        "8bafbb8802fb0d9f81c190e87b9012fca546182be08bef6c8b3db814d27bfe19",
        "c0db42467a960c509b658c3c40570ee889f73d41b52cdbd80b9c291c1792a33f",
    ),
    "sglang/srt/layers/muse_spark_layers.py": "930e66da715bae576ae681f2dfd8353a36a9ca469a229d88fa3f77bc1e74cef0",
    "sglang/srt/layers/layernorm_sp.py": (
        "ffdde97e6282103614d7cec20cc29fb3cf0157d051778e315b03a99bb8e81090",
        "bc814d983b945e5bdd41db436966eb4b3fee4c410bb1dcccfc98857d4b026bda",
    ),
}
VALIDATED_TORCH = (
    "e0474491d782a73b73cf132f732a8513b3c9f49d",
    "7bf52b3a9782d3440a17eb3f00983795f3b12af0",
)
_loaded: set = set()


def enabled() -> bool:
    return bed.enabled(NAME)


def pins_match() -> tuple[bool, str]:
    for rel, want in PINS.items():
        if not bed.pinned(rel, want):
            return False, f"{rel} differs from the pinned revision"
    if torch.version.git_version not in VALIDATED_TORCH:
        return (
            False,
            f"PyTorch build {torch.version.git_version[:12]} not in the validated set",
        )
    return True, "ok"


@functools.lru_cache(maxsize=None)
def _use_pdl() -> bool:
    from sglang.kernels.jit.utils import is_arch_support_pdl

    return bool(is_arch_support_pdl())


@functools.lru_cache(maxsize=None)
def module(hidden: int, pdl: bool):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    args = make_cpp_args(hidden, pdl)
    return load_jit(
        "muse_vr_input_norm",
        *args,
        cuda_files=[str(_CUH)],
        cuda_wrappers=[
            ("embed_stream", f"muse_vr::input_norm::InputNorm<{args}>::embed_stream"),
            ("stream_sq", f"muse_vr::input_norm::InputNorm<{args}>::stream_sq"),
            ("finish", f"muse_vr::input_norm::InputNorm<{args}>::finish"),
        ],
    )


def embed_stream(x, eps, pdl=None):
    """(stream fp32, squares fp32) for the embedding output ``x`` (bf16)."""
    x = x.contiguous()
    stream = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    sq = torch.empty_like(stream)
    module(x.shape[-1], _use_pdl() if pdl is None else pdl).embed_stream(
        x, stream, sq, float(eps)
    )
    return stream, sq


def stream_sq(x, pdl=None):
    """(stream fp32, squares fp32) for already normalized embeddings ``x`` (bf16)."""
    x = x.contiguous()
    stream = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    sq = torch.empty_like(stream)
    module(x.shape[-1], _use_pdl() if pdl is None else pdl).stream_sq(x, stream, sq)
    return stream, sq


def finish(stream, sq, weight, eps, pdl=None):
    mean = sq.mean(-1, keepdim=True)
    out = torch.empty(stream.shape, dtype=torch.bfloat16, device=stream.device)
    module(stream.shape[-1], _use_pdl() if pdl is None else pdl).finish(
        stream, mean, weight, out, float(eps)
    )
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Sub-component input_norm_l0_table: layer 0's QKV(G) projection as a per-token table (module docstring).

TABLE = (
    "input_norm_l0_table"  # env MUSE_VR_INPUT_NORM_L0_TABLE (with MUSE_VR_INPUT_NORM)
)
T_LOAD = "sglang.srt.model_executor.model_runner.ModelRunner.load_model"
T_OMNI = "sglang.srt.models.muse_spark_omni.MuseSparkForConditionalGeneration.forward"
T_APPLY = "sglang.srt.layers.quantization.unquant.UnquantizedLinearMethod.apply"
T_APPLY_INTO = (
    "sglang.srt.layers.quantization.unquant.UnquantizedLinearMethod.apply_into"
)
# sha256 of the SGLang sources whose behaviour the table path mirrors: the wrapper's forward (embeddings made iff not
# given), the bf16 GEMM dispatch (8-row tactic) and the linear layers' call of it, the replicated embedding lookup
# (embed_gather's pins) and the load order (the table is built before the KV cache is sized)
TABLE_PINS = {
    "sglang/srt/models/muse_spark_omni.py": "f186e56bbfa76ecb06d92330a5597513f5179efd7c1b9bc567955d4e800ef777",
    "sglang/srt/layers/quantization/unquant.py": (
        "f428022195466613e7e988265a1090862a5aa0557bb7a063c0c172f9bd3988c5",
        "7285d3e9e9fa881c5ecd97d093a8f59e11e69f26d5e3e7006925149f624a2f9c",
    ),
    "sglang/srt/layers/linear.py": "3ce31df0a8aab09ed8afa0ada8c0ce30ca207a748c02c3073d7f53648b9b7c3c",
    "sglang/srt/layers/vocab_parallel_embedding.py": (
        "5c4a96a8ad291f7601876368cec75769adc152b6af6fffe317ae5fc44d10d4d1",
        "c36f5682e528fafe922b88ad0e9dd418b7f9558762058858707a6a75c25ea5cf",
    ),
    "sglang/kernels/ops/embeddings/vocab_parallel_embedding.py": "7766e50514e621317fec277fd5975adc469d70ab2951722a99d56afbb6dc16d4",
    "sglang/srt/model_executor/model_runner.py": "c3e8e4cab00b8d6a4303c8332ff5e2fea4618c57000aa2c9b4b78b42eeaab6a7",
}
ROWS = 8  # the verify block of one request (K + 1 = 8): the only call the table takes
CHECK_BATCHES = 512  # random 8-id batches of the install-time eager check (plus one batch of edge ids)
CHECK_REPLAYS = 300  # CUDA-graph replays of the install-time replay check
GATHER_BLOCK = 256  # columns per gather program

STREAM = (
    TABLE + "_stream"
)  # env MUSE_VR_INPUT_NORM_L0_TABLE_STREAM: the residual stream from the table

_tq: dict = {
    "table": None,
    "omni": None,
    "lm": None,
    "qkv": None,
    "o": None,
    "shadow": False,
    "guard": None,
    "etab": None,
    "stream_on": False,
}
_arm: list = [
    None
]  # the armed step (one per process at a time: the forward is single-threaded)

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU test environments; the table needs CUDA anyway
    triton = None

if triton is not None:

    @triton.jit
    def _l0_table_gather_kernel(
        ids_ptr, table_ptr, out_ptr, vocab, N: tl.constexpr, BLOCK: tl.constexpr
    ):
        # program (row, chunk): BLOCK columns of the row's projection, moved as their 16 bits (TRT-LLM's
        # kernels/tt_table/target_l0.py ``_tq_gather_kernel`` without the PDL wait)
        row = tl.program_id(0)
        chunk = tl.program_id(1)
        token = tl.load(ids_ptr + row).to(tl.int64)
        entry = tl.where(
            (token >= 0) & (token < vocab), token, vocab
        )  # entry ``vocab``: an id outside the table
        cols = chunk * BLOCK + tl.arange(0, BLOCK)
        in_row = cols < N
        bits = tl.load(table_ptr + entry * N + cols, mask=in_row, other=0)
        tl.store(out_ptr + row.to(tl.int64) * N + cols, bits, mask=in_row)

    @triton.jit
    def _l0_table_gather2_kernel(
        ids_ptr,
        table_ptr,
        etab_ptr,
        out_ptr,
        stream_ptr,
        vocab,
        N: tl.constexpr,
        NE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # program (row, chunk) over [QKV(G) row (N) | stream row (NE)]: the projection's columns moved as
        # their 16 bits, the stream's as the bf16 embedding value cast to fp32 (exact)
        row = tl.program_id(0)
        chunk = tl.program_id(1)
        token = tl.load(ids_ptr + row).to(tl.int64)
        entry = tl.where(
            (token >= 0) & (token < vocab), token, vocab
        )  # entry ``vocab``: an id outside the table
        cols = chunk * BLOCK + tl.arange(0, BLOCK)
        in_q = cols < N
        bits = tl.load(table_ptr + entry * N + cols, mask=in_q, other=0)
        tl.store(out_ptr + row.to(tl.int64) * N + cols, bits, mask=in_q)
        ecol = cols - N
        in_e = (cols >= N) & (ecol < NE)
        x = tl.load(etab_ptr + entry * NE + ecol, mask=in_e, other=0.0)
        tl.store(stream_ptr + row.to(tl.int64) * NE + ecol, x.to(tl.float32), mask=in_e)


def gather2(table, etab, ids):
    """``(rows [r, N] bf16, stream [r, NE] fp32)``: ``gather``'s rows and the fp32 cast of ``etab``'s row
    of each id (``etab`` ``[V + 1, NE]`` bf16, contiguous; the same entry for an id outside ``[0, V)``); one launch."""
    rows, n, ne = ids.numel(), int(table.shape[1]), int(etab.shape[1])
    out = torch.empty((rows, n), dtype=table.dtype, device=table.device)
    stream = torch.empty((rows, ne), dtype=torch.float32, device=table.device)
    _l0_table_gather2_kernel[(rows, triton.cdiv(n + ne, GATHER_BLOCK))](
        ids,
        table.view(torch.int16),
        etab,
        out.view(torch.int16),
        stream,
        int(table.shape[0]) - 1,
        N=n,
        NE=ne,
        BLOCK=GATHER_BLOCK,
        num_warps=4,
    )
    return out, stream


def gather2_torch(table, etab, ids):
    """``gather2`` by torch indexing (CPU tests)."""
    v = int(table.shape[0]) - 1
    entry = torch.where((ids >= 0) & (ids < v), ids, torch.full_like(ids, v))
    return table[entry].clone(), etab[entry].to(torch.float32)


def gather(table, ids):
    """``[rows, N]``: row ``i`` is ``table[ids[i]]``, or ``table[V]`` for an id outside ``[0, V)``; ``table`` is
    ``[V + 1, N]`` bf16 (contiguous), ``ids`` 1-D int32 / int64 (contiguous)."""
    rows, n = ids.numel(), int(table.shape[1])
    out = torch.empty((rows, n), dtype=table.dtype, device=table.device)
    _l0_table_gather_kernel[(rows, triton.cdiv(n, GATHER_BLOCK))](
        ids,
        table.view(torch.int16),
        out.view(torch.int16),
        int(table.shape[0]) - 1,
        N=n,
        BLOCK=GATHER_BLOCK,
        num_warps=4,
    )
    return out


class _Arm:
    """One armed step: the gathered rows (until layer 0's qkv_proj takes them), the attention input the stand-in left
    unwritten in their place (the operand the rows stand in for), the served finish's inputs while it is unwritten
    (``pending``), and whether the step left the armed order."""

    __slots__ = ("rows", "x", "pending", "unexpected", "ids", "ref", "stream", "sref")

    def __init__(self, rows, ids, stream=None):
        self.rows, self.x, self.pending = rows, None, None
        self.unexpected = False
        self.ids, self.ref = ids, None
        self.stream, self.sref = (
            stream,
            None,
        )  # the step's stream rows from the table (or None)


def _unwritten(shape, device):
    """Layer 0's attention input on an armed step: allocated, never written unless ``_materialize`` runs."""
    return torch.empty(shape, dtype=torch.bfloat16, device=device)


def _squares(stream):
    """the served squares of a stream from the table, by this component's ``stream_sq`` on the stream's bf16
    values (the stream is the fp32 cast of bf16 values, so the round trip is exact and ``stream_sq`` returns the same
    stream and the served squares)."""
    return stream_sq(stream.to(torch.bfloat16))[1]


def _materialize(a) -> None:
    """Write the attention input the stand-in left unwritten (the served row mean and finish, copied into it), for a
    reader other than layer 0's qkv_proj (not the armed order: that GEMM then runs stock as well)."""
    stream, sq, weight, eps = a.pending
    a.pending = None
    a.unexpected = True
    if sq is None:  # the stream came from the table; make its squares as served
        sq = _squares(stream)
    a.x.copy_(finish(stream, sq, weight, eps))


def table_chain(model):
    """The served chain from 8 ids to layer 0's QKV(G) output, by the step's own calls: the wrapper's embedding (with
    its norm), this component's stream / squares and finish (the ``input_embeds`` path of ``model_forward``), and
    layer 0's GEMM as ``ColumnParallelLinear.forward`` calls it (no bias, no sequence parallelism)."""
    lm = model.model.language_model
    lin, ln = lm.layers[0].self_attn.qkv_proj, lm.layers[0].input_layernorm
    embed = model.get_input_embeddings()

    def chain(ids):
        x = embed(ids)
        stream, sq = stream_sq(x)
        attn = finish(stream, sq, ln._get_effective_weight(), ln.eps)
        return lin.quant_method.apply(lin, attn, None)

    return chain


def table_chain2(model):
    """``table_chain``'s served calls returning ``(QKV(G) rows, stream, normalized embedding)``: the build keeps
    the first and the last, the checks compare the first two with the gather's."""
    lm = model.model.language_model
    lin, ln = lm.layers[0].self_attn.qkv_proj, lm.layers[0].input_layernorm
    embed = model.get_input_embeddings()

    def chain(ids):
        x = embed(ids)
        stream, sq = stream_sq(x)
        attn = finish(stream, sq, ln._get_effective_weight(), ln.eps)
        return lin.quant_method.apply(lin, attn, None), stream, x

    return chain


def row_independent(product, x) -> list:
    """What breaks row independence of ``product`` (an ``[8, K] -> [8, N]`` map) on ``x``, or ``[]``: bit for bit,
    a permutation of the rows permutes the output rows, and a row's output does not change when every other row of the
    call is replaced (TRT-LLM's ``target_l0.row_independent``)."""
    problems = []
    base = product(x)
    generator = torch.Generator(device="cpu").manual_seed(20261101)
    perm = torch.randperm(x.shape[0], generator=generator).to(x.device)
    if not torch.equal(
        product(x[perm].contiguous()).view(torch.int16), base[perm].view(torch.int16)
    ):
        problems.append("a permutation of the rows does not permute the output rows")
    for row in range(x.shape[0]):
        other = torch.randn(x.shape, generator=generator).to(
            device=x.device, dtype=x.dtype
        )
        other[row] = x[row]
        if not torch.equal(
            product(other)[row].view(torch.int16), base[row].view(torch.int16)
        ):
            problems.append(f"row {row} changes with the other rows of its call")
    return problems


def build(chain, vocab: int, n: int, device):
    """``[vocab + 1, n]``: ``chain`` on the ids ``0 .. vocab - 1`` in calls of 8 consecutive ids, and on ``-1``
    (entry ``vocab``)."""
    table = torch.empty((vocab + 1, n), dtype=torch.bfloat16, device=device)
    ids = torch.arange(vocab + ROWS, dtype=torch.int64, device=device)
    ids[vocab:] = -1
    with torch.inference_mode():
        for begin in range(0, vocab + 1, ROWS):
            rows = chain(ids[begin : begin + ROWS])
            end = min(begin + ROWS, vocab + 1)
            table[begin:end] = rows[: end - begin]
    return table


def build2(chain2, vocab: int, n: int, ne: int, device):
    """``([vocab + 1, n], [vocab + 1, ne])`` bf16: ``build``'s table and each id's normalized embedding row, from
    the same calls of ``chain2`` (8 consecutive ids; entry ``vocab`` from id -1)."""
    table = torch.empty((vocab + 1, n), dtype=torch.bfloat16, device=device)
    etab = torch.empty((vocab + 1, ne), dtype=torch.bfloat16, device=device)
    ids = torch.arange(vocab + ROWS, dtype=torch.int64, device=device)
    ids[vocab:] = -1
    with torch.inference_mode():
        for begin in range(0, vocab + 1, ROWS):
            rows, _, x = chain2(ids[begin : begin + ROWS])
            end = min(begin + ROWS, vocab + 1)
            table[begin:end] = rows[: end - begin]
            etab[begin:end] = x[: end - begin]
    return table, etab


def _differs2(a, b):
    """Per row: whether two ``(rows bf16, stream fp32)`` pairs differ in any bit."""
    return (a[0].view(torch.int16) != b[0].view(torch.int16)).any(dim=1) | (
        a[1].view(torch.int32) != b[1].view(torch.int32)
    ).any(dim=1)


def check2(chain2, gather2_fn, ids) -> tuple:
    """``check`` for both outputs: ``gather2_fn``'s ``(rows, stream)`` against ``chain2``'s QKV(G) rows and served
    stream, in calls of 8, in the given and the reversed order, bit for bit (at most 16 differing ids listed)."""
    bad: list = []
    with torch.inference_mode():
        for begin in range(0, ids.numel(), ROWS):
            batch = ids[begin : begin + ROWS].contiguous()
            for order in (batch, batch.flip(0).contiguous()):
                rows = _differs2(chain2(order)[:2], gather2_fn(order))
                if bool(rows.any()):
                    bad.extend(
                        int(order[i])
                        for i in torch.nonzero(rows).flatten().tolist()[: 16 - len(bad)]
                    )
    return int(ids.numel()), bad


def check_replays2(chain2, gather2_fn, replays: int, vocab: int, device) -> int:
    """``check_replays`` for both outputs (differing values of the rows and of the stream)."""
    ids = torch.zeros(ROWS, dtype=torch.int64, device=device)
    generator = torch.Generator(device="cpu").manual_seed(20261103)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.inference_mode(), torch.cuda.stream(stream):
        chain2(ids)
        gather2_fn(ids)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            graph, stream=stream, capture_error_mode="thread_local"
        ):  # other init threads may run
            c, g = chain2(ids), gather2_fn(ids)
            diff = (c[0].view(torch.int16) != g[0].view(torch.int16)).sum() + (
                c[1].view(torch.int32) != g[1].view(torch.int32)
            ).sum()
        total = torch.zeros((), dtype=torch.int64, device=device)
        for _ in range(replays):
            ids.copy_(
                torch.randint(0, vocab, (ROWS,), generator=generator, dtype=torch.int64)
            )
            graph.replay()
            total += diff
    torch.cuda.current_stream(device).wait_stream(stream)
    return int(total.item())


def check_ids(vocab: int, device):
    """The ids the eager check covers: one batch of edge ids and ``CHECK_BATCHES`` random 8-id batches."""
    generator = torch.Generator(device="cpu").manual_seed(20261102)
    ids = torch.cat(
        [
            torch.tensor(
                [0, 1, vocab - 1, vocab, -1, vocab + 7, 2, 3], dtype=torch.int64
            ),
            torch.randint(
                0,
                vocab,
                (CHECK_BATCHES * ROWS,),
                generator=generator,
                dtype=torch.int64,
            ),
        ]
    )
    return ids.to(device)


def check(chain, gather_fn, ids) -> tuple:
    """``(ids checked, ids whose row differs)``: ``gather_fn`` against ``chain`` on ``ids`` in calls of 8, in the given
    and in the reversed order, bit for bit (at most 16 differing ids listed)."""
    bad: list = []
    with torch.inference_mode():
        for begin in range(0, ids.numel(), ROWS):
            batch = ids[begin : begin + ROWS].contiguous()
            for order in (batch, batch.flip(0).contiguous()):
                rows = (
                    chain(order).view(torch.int16) != gather_fn(order).view(torch.int16)
                ).any(dim=1)
                if bool(rows.any()):
                    bad.extend(
                        int(order[i])
                        for i in torch.nonzero(rows).flatten().tolist()[: 16 - len(bad)]
                    )
    return int(ids.numel()), bad


def check_replays(chain, gather_fn, replays: int, vocab: int, device) -> int:
    """Differing values over ``replays`` CUDA-graph replays of ``chain`` and ``gather_fn`` (one graph, both on one
    static ids buffer refilled with a fresh random batch before each replay)."""
    ids = torch.zeros(ROWS, dtype=torch.int64, device=device)
    generator = torch.Generator(device="cpu").manual_seed(20261103)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.inference_mode(), torch.cuda.stream(stream):
        chain(ids)
        gather_fn(ids)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            graph, stream=stream, capture_error_mode="thread_local"
        ):  # other init threads may run
            diff = (
                chain(ids).view(torch.int16) != gather_fn(ids).view(torch.int16)
            ).sum()
        total = torch.zeros((), dtype=torch.int64, device=device)
        for _ in range(replays):
            ids.copy_(
                torch.randint(0, vocab, (ROWS,), generator=generator, dtype=torch.int64)
            )
            graph.replay()
            total += diff
    torch.cuda.current_stream(device).wait_stream(stream)
    return int(total.item())


def _static_reason(model) -> str | None:
    """Why the table cannot stand in for this model's layer-0 projection (construction-time state only; rank-uniform),
    or None."""
    from sglang.srt.layers.muse_spark_layers import MuseSpark1xRMSNorm
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
    from sglang.srt.models.muse_spark_omni import MuseSparkForConditionalGeneration

    if type(model) is not MuseSparkForConditionalGeneration:
        return "not the multimodal target wrapper"
    if (
        getattr(model, "_normalized_embedding", None) is None
        or model.get_input_embeddings() is not model._normalized_embedding
    ):
        return "no normalized embedding"
    from sglang.srt.environ import envs

    if envs.SGLANG_ENABLE_ASYNC_ASSERT.get():
        return "async id asserts on (the table's outside entry is built from id -1)"
    lm = model.model.language_model
    layer0 = lm.layers[0]
    lin, o = layer0.self_attn.qkv_proj, layer0.self_attn.o_proj
    if (
        type(lin.quant_method) is not UnquantizedLinearMethod
        or type(o.quant_method) is not UnquantizedLinearMethod
    ):
        return "projection methods"
    w = lin.weight
    if (
        w.dtype != torch.bfloat16
        or w.dim() != 2
        or not w.is_contiguous()
        or not w.is_cuda
    ):
        return "qkv weight layout"
    if getattr(lin, "bias", None) is not None or getattr(lin, "gather_output", False):
        return "qkv bias or output gather"
    embed = lm.embed_tokens
    if getattr(embed, "tp_size", None) != 1 or embed.weight.shape[1] != w.shape[1]:
        return "embedding not replicated"
    if not bed.installs().get("embed_gather", {}).get("engaged"):
        return "embed_gather not engaged (the outside entry needs its masked lookup)"
    en, ln = lm.embed_norm, layer0.input_layernorm
    if (
        type(en) is not MuseSpark1xRMSNorm
        or en.weight is not None
        or type(ln) is not MuseSpark1xRMSNorm
    ):
        return "norm module types"
    eff = ln._get_effective_weight()
    if (
        eff is None
        or eff.dtype != torch.bfloat16
        or eff.numel() % 512
        or lm.residual_dtype != torch.float32
        or (lm.operator_dtype != torch.bfloat16)
    ):
        return "dtypes"
    if ln.zero_centered_gamma and ln._frozen_weight is None:
        return "input norm weight not frozen"
    return None


def install(model) -> str | None:
    """Build the table for ``model`` (the target wrapper) and arm the hooks; the reason it is not installed, or None."""
    reason = _static_reason(model)
    if reason is not None:
        return reason
    lm = model.model.language_model
    lin = lm.layers[0].self_attn.qkv_proj
    w = lin.weight
    device = w.device
    vocab, n = int(lm.embed_tokens.num_embeddings), int(w.shape[0])
    module(int(w.shape[1]), _use_pdl())
    _loaded.add((int(w.shape[1]), _use_pdl()))
    chain = table_chain(model)
    torch.cuda.synchronize(device)
    free0 = torch.cuda.mem_get_info(device)[0]
    t0 = time.monotonic()
    with torch.inference_mode():
        x = torch.randn(
            (ROWS, int(w.shape[1])),
            generator=torch.Generator(device="cpu").manual_seed(20261100),
        )
        problems = row_independent(
            lambda a: lin.quant_method.apply(lin, a, None),
            x.to(device=device, dtype=torch.bfloat16),
        )
    if problems:
        return "the 8-row QKV GEMM is not row-independent: " + "; ".join(problems)
    etab = None
    if _tq["stream_on"]:  # the normalized embedding rows too; both outputs checked
        ne = int(lm.embed_tokens.weight.shape[1])
        chain2 = table_chain2(model)
        table, etab = build2(chain2, vocab, n, ne, device)
        torch.cuda.synchronize(device)
        built = time.monotonic() - t0
        checked, bad = check2(
            chain2, lambda ids: gather2(table, etab, ids), check_ids(vocab, device)
        )
        replay_diff = check_replays2(
            chain2, lambda ids: gather2(table, etab, ids), CHECK_REPLAYS, vocab, device
        )
    else:
        table = build(chain, vocab, n, device)
        torch.cuda.synchronize(device)
        built = time.monotonic() - t0
        checked, bad = check(
            chain, lambda ids: gather(table, ids), check_ids(vocab, device)
        )
        replay_diff = check_replays(
            chain, lambda ids: gather(table, ids), CHECK_REPLAYS, vocab, device
        )
    if bad or replay_diff:
        return f"the table differs from the chain (ids {bad}; {replay_diff} values over {CHECK_REPLAYS} replays)"
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    free1 = torch.cuda.mem_get_info(device)[0]
    _tq.update(
        table=table,
        etab=etab,
        omni=model,
        lm=lm,
        qkv=lin,
        o=lm.layers[0].self_attn.o_proj,
    )
    if _tq["guard"] is None:
        _tq["guard"] = lm.layers[0].register_forward_hook(_layer0_guard)
    gib = float(1 << 30)
    extra = (
        {}
        if etab is None
        else dict(
            stream=True,
            stream_cols=int(etab.shape[1]),
            stream_gib=round(etab.numel() * 2 / gib, 3),
        )
    )
    bed.report(
        TABLE,
        True,
        rows=vocab + 1,
        cols=n,
        gib=round(table.numel() * 2 / gib, 3),
        built_s=round(built, 1),
        checked=f"{checked} ids x 2 orders",
        replays=CHECK_REPLAYS,
        free_gib=f"{free0 / gib:.2f}->{free1 / gib:.2f}",
        shadow=_tq["shadow"],
        **extra,
    )
    return None


def _after_load(result, runner, *args, **kwargs):
    """``ModelRunner.load_model`` (AFTER): build the table for the target runner (not the draft's)."""
    if getattr(runner, "is_draft_worker", False):
        return None
    try:
        reason = install(getattr(runner, "model", None))
    except (
        Exception
    ) as e:  # a failed build leaves the sub-component off; the served chain runs
        _tq.update(table=None, etab=None)
        reason = f"{type(e).__name__}: {e}"[:300]
    if reason is not None:
        _tq.update(table=None, etab=None)
        bed.report(TABLE, False, reason=reason)
    return None


def _layer0_guard(module_, args, output):
    """Forward hook on layer 0: an armed step must not end the layer with its attention input still unwritten and the
    table's rows unused (the input then had a reader the hooks did not see)."""
    a = _arm[0]
    if a is not None and a.pending is not None:
        a.pending = None
        raise RuntimeError(
            "input_norm_l0_table: layer 0 ended with its attention input unwritten and the table's rows "
            "unused"
        )
    return None


def _arm_reason(model, input_ids, forward_batch, input_embeds) -> str | None:
    """Why this wrapper call does not take the table (host-side, rank-uniform), or None."""
    if _tq["table"] is None:
        return "off"
    if model is not _tq["omni"]:
        return "model"
    if input_embeds is not None or input_ids is None:
        return "embeddings given"
    if not forward_batch.forward_mode.is_target_verify():
        return "not target verify"
    if not (
        isinstance(input_ids, torch.Tensor)
        and input_ids.is_cuda
        and input_ids.dim() == 1
        and input_ids.numel() == ROWS
        and input_ids.dtype in (torch.int32, torch.int64)
        and input_ids.is_contiguous()
    ):
        return "ids"
    if torch.is_grad_enabled():
        return "grad"
    if forward_batch.contains_mm_inputs():
        return "multimodal inputs"
    if _arm[0] is not None:
        return "nested"
    lm = _tq["lm"]
    reason = _eligible(
        lm,
        torch.empty(
            (ROWS, lm.embed_tokens.weight.shape[1]), dtype=torch.bfloat16, device="meta"
        ),
        forward_batch,
    )
    if reason is not None:
        return "input_norm: " + reason
    return None


def omni_forward(
    original, model, input_ids, positions, forward_batch, input_embeds=None
):
    reason = _arm_reason(model, input_ids, forward_batch, input_embeds)
    if reason is not None:
        if reason != "off":
            bed.count(TABLE, "stock:" + reason)
        return original(model, input_ids, positions, forward_batch, input_embeds)
    if (
        _tq["etab"] is not None
    ):  # one gather for the QKV(G) rows and the step's fp32 stream rows
        rows, stream = gather2(_tq["table"], _tq["etab"], input_ids)
        a = _Arm(rows, input_ids, stream)
    else:
        a = _Arm(
            gather(_tq["table"], input_ids), input_ids
        )  # the step's first kernel: layer 0's QKV(G) rows of the step
    shadow = _tq["shadow"] and not torch.cuda.is_current_stream_capturing()
    if shadow:
        a.ref = a.rows.clone()  # the attention may rewrite its QKV operand in place
        if a.stream is not None:
            a.sref = a.stream.clone()  # the layers update the stream in place
    _arm[0] = a
    try:
        if a.stream is None:
            input_embeds = model.get_input_embeddings()(
                input_ids
            )  # the wrapper's own call
        else:
            input_embeds = a.stream  # the wrapper's embedding call, as its fp32 cast (the model's cast: a no-op)
        out = original(model, input_ids, positions, forward_batch, input_embeds)
    finally:
        _arm[0] = None
    if a.rows is not None or a.unexpected:
        bed.count(TABLE, "stock projection")
    else:
        bed.count(TABLE, "table")
    if shadow:
        with torch.inference_mode():
            if a.stream is None:
                live = table_chain(model)(input_ids)
                ok = torch.equal(live.view(torch.int16), a.ref.view(torch.int16))
            else:
                live, live_stream, _ = table_chain2(model)(input_ids)
                ok = torch.equal(
                    live.view(torch.int16), a.ref.view(torch.int16)
                ) and torch.equal(
                    live_stream.view(torch.int32), a.sref.view(torch.int32)
                )
        bed.shadow_result(TABLE, bool(ok))
    return out


def _apply_hook(original, method, layer, x, bias=None):
    a = _arm[0]
    if a is None or a.pending is None or x is not a.x:
        return original(method, layer, x, bias)
    if layer is _tq["qkv"] and bias is None:
        rows, a.rows, a.pending = a.rows, None, None
        return rows  # the table's rows of this step in place of the GEMM (the stand-in's operand stays unwritten)
    _materialize(
        a
    )  # another GEMM reads the unwritten input (not the armed order): write it first
    return original(method, layer, x, bias)


def _apply_into_hook(original, method, layer, x, output, bias=None):
    a = _arm[0]
    if a is not None and a.pending is not None and x is a.x:
        _materialize(
            a
        )  # (not the armed order) this GEMM reads the unwritten input: write it first
    return original(method, layer, x, output, bias)


def _register_table(HookRegistry, HookType) -> None:
    if not bed.enabled(TABLE):
        return
    for rel, want in TABLE_PINS.items():
        if not bed.pinned(rel, want):
            bed.report(TABLE, False, reason=f"{rel} differs from the pinned revision")
            return
    if triton is None:
        bed.report(TABLE, False, reason="triton unavailable")
        return
    _tq["shadow"] = bed.enabled(TABLE + "_shadow")
    _tq["stream_on"] = bed.enabled(STREAM)

    def omni_hook(original, *args, **kwargs):
        names = ("input_ids", "positions", "forward_batch", "input_embeds")
        if len(args) < 4 or len(args) > 5 or set(kwargs) - set(names[len(args) - 1 :]):
            return original(*args, **kwargs)
        kw = dict(zip(names, args[1:]))
        kw.update(kwargs)
        return omni_forward(
            original,
            args[0],
            kw["input_ids"],
            kw["positions"],
            kw["forward_batch"],
            kw.get("input_embeds"),
        )

    def apply_hook(original, method, layer, x, bias=None):
        return _apply_hook(original, method, layer, x, bias)

    def apply_into_hook(original, method, layer, x, output, bias=None):
        return _apply_into_hook(original, method, layer, x, output, bias)

    HookRegistry.register(T_LOAD, _after_load, HookType.AFTER)
    HookRegistry.register(T_OMNI, omni_hook, HookType.AROUND)
    HookRegistry.register(T_APPLY, apply_hook, HookType.AROUND)
    HookRegistry.register(T_APPLY_INTO, apply_into_hook, HookType.AROUND)
    bed.report(
        TABLE, False, reason="registered: the table is built after the target loads"
    )


class _Embed(torch.nn.Module):
    def __init__(self, ctx, norm):
        super().__init__()
        self._ctx, self._norm = ctx, norm

    def forward(self, x):
        stream, sq = embed_stream(x, self._norm.eps)
        self._ctx["stream"], self._ctx["sq"] = stream, sq
        return stream  # fp32: the model's cast to the residual dtype is then a no-op


class _Input(torch.nn.Module):
    def __init__(self, ctx, norm):
        super().__init__()
        self._ctx, self._norm = ctx, norm

    def forward(self, stream):
        if (
            stream.data_ptr() != self._ctx.get("stream", stream).data_ptr()
            or "sq" not in self._ctx
        ):
            raise RuntimeError("input_norm: unexpected call order")
        sq, weight, eps = (
            self._ctx.pop("sq"),
            self._norm._get_effective_weight(),
            self._norm.eps,
        )
        arm = _arm[0]
        if (
            arm is not None
            and arm.rows is not None
            and arm.x is None
            and not arm.unexpected
        ):
            # an armed step: the one reader of this output is layer 0's qkv_proj, whose GEMM the table's rows replace;
            # no row mean, no finish (``_materialize`` writes it if another reader turns up)
            out = _unwritten(stream.shape, stream.device)
            arm.x, arm.pending = out, (stream, sq, weight, eps)
        else:
            if sq is None:  # the stream came from the table; make its squares as served
                sq = _squares(stream)
            out = finish(stream, sq, weight, eps)
        self._ctx["fused"] = True
        return out  # bf16: the layer's cast to the operator dtype is then a no-op


def _eligible(model, input_embeds, forward_batch) -> str | None:
    import sglang.srt.layers.layernorm_sp as layernorm_sp
    from sglang.srt.layers.muse_spark_layers import MuseSpark1xRMSNorm

    mode = forward_batch.forward_mode
    if not (mode.is_decode() or mode.is_target_verify()):
        return "not decode/verify"
    if input_embeds is not None and (
        input_embeds.dtype != torch.bfloat16 or input_embeds.dim() != 2
    ):
        return "input embeddings layout"
    if input_embeds is None and model.embed_norm is None:
        return "no embedding norm"
    if layernorm_sp.runs_sp(mode):
        return "sequence parallel"
    en, ln = model.embed_norm, model.layers[0].input_layernorm
    if type(ln) is not MuseSpark1xRMSNorm or (
        input_embeds is None
        and (type(en) is not MuseSpark1xRMSNorm or en.weight is not None)
    ):
        return "module types"
    w = ln._get_effective_weight()
    if (
        w is None
        or w.dtype != torch.bfloat16
        or w.numel() % 512
        or model.residual_dtype != torch.float32
        or (model.operator_dtype != torch.bfloat16)
    ):
        return "dtypes"
    if torch.cuda.is_current_stream_capturing():
        if (w.numel(), _use_pdl()) not in _loaded:
            return "kernel not loaded before graph capture"
    else:
        module(w.numel(), _use_pdl())
        _loaded.add((w.numel(), _use_pdl()))
    return None


def model_forward(
    original, model, input_ids, positions, forward_batch, input_embeds=None
):
    arm = _arm[0]
    # the table's fp32 stream rows in place of the wrapper's bf16 embeddings (eligibility as for those)
    table_stream = (
        arm is not None and arm.stream is not None and input_embeds is arm.stream
    )
    probe = (
        torch.empty(input_embeds.shape, dtype=torch.bfloat16, device="meta")
        if table_stream
        else input_embeds
    )
    reason = _eligible(
        model,
        probe,
        forward_batch,
    )
    if arm is not None and (reason is not None or input_embeds is None):
        arm.unexpected = True  # (not the armed order) the input norm below runs as served; the rows stay unused
    if reason is not None:
        bed.count(NAME, "stock:" + reason)
        # (the model casts the given fp32 stream to the residual dtype as a no-op: the served stream)
        return original(model, input_ids, positions, forward_batch, input_embeds)
    first = model.layers[0]
    ctx: dict = {}
    saved = (model.embed_norm, first.input_layernorm)
    if input_embeds is None:
        model.embed_norm = _Embed(ctx, saved[0])
        tag = "fused"
    elif table_stream:
        # the stream is ready (the model's cast to the fp32 residual dtype is a no-op); the squares are made only if a
        # finish runs (``_squares``)
        ctx["stream"], ctx["sq"] = input_embeds, None
        tag = "fused:table_stream"
    else:
        # the model casts input_embeds to the fp32 stream: hand it the stream (the cast is then a no-op)
        input_embeds, ctx["sq"] = stream_sq(input_embeds)
        ctx["stream"] = input_embeds
        tag = "fused:input_embeds"
    first.input_layernorm = _Input(ctx, saved[1])
    try:
        out = original(model, input_ids, positions, forward_batch, input_embeds)
    finally:
        model.embed_norm, first.input_layernorm = saved
    if not ctx.get("fused"):
        raise RuntimeError("input_norm: the input boundary was not reached")
    bed.count(NAME, tag)
    return out


def register() -> bool:
    if not enabled():
        return False
    ok, why = pins_match()
    if not ok:
        bed.report(NAME, False, reason=why)
        return False
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    def hook(original, *args, **kwargs):
        names = ("input_ids", "positions", "forward_batch", "input_embeds")
        if len(args) < 4 or len(args) > 5 or set(kwargs) - set(names[len(args) - 1 :]):
            return original(*args, **kwargs)
        model, rest = args[0], list(args[1:])
        kw = dict(zip(names, rest))
        kw.update(kwargs)
        return model_forward(
            original,
            model,
            kw["input_ids"],
            kw["positions"],
            kw["forward_batch"],
            kw.get("input_embeds"),
        )

    HookRegistry.register(TARGET, hook, HookType.AROUND)
    bed.report(NAME, True)
    _register_table(HookRegistry, HookType)
    return True

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``ctx_proj_overlap`` (env ``MUSE_VR_CTX_PROJ_OVERLAP=1``, default off).

After each target-verify forward, SGLang's DFLASH worker first verifies and samples (gather of the vocab-parallel
logits, accept/bonus selection) and only then projects the verify hidden states into the draft's context features
(``DFlashDraftModel.project_target_hidden``: the context ``fc`` GEMM and ``hidden_norm``) to write the draft KV cache.
The projection reads only the verify hidden states, so this component issues the same call on a side CUDA stream as
soon as the verify forward returns, and hands its result to the worker's own ``project_target_hidden`` call on the same
tensor, after making the worker's stream wait for it. The sampling path and the projection then run concurrently. The
projection is computed in full, by the same code on the same input, once per step (bit-identical result:
``tests/test_ctx_proj_overlap.py``).

Matching is strict: the next ``project_target_hidden`` call on that draft model must pass the very tensor the verify
returned (same storage, shape, strides, dtype); any other call computes as stock, after the worker's stream has waited
for the side work so the verify buffers are never rewritten while it still reads them. Nothing runs inside CUDA-graph
capture.

``MUSE_VR_CTX_PROJ_OVERLAP_SHARD=1`` (default off) splits the work across the tensor-parallel ranks for verify blocks of
at most ``MUSE_VR_CTX_PROJ_OVERLAP_SHARD_MAX_ROWS`` rows (default 32): the side stream computes only this rank's
slice of output columns (the same ``F.linear`` call on a row slice of the replicated weight), and the worker's call
gathers the slices with SGLang's NVLink pull all-gather and applies ``hidden_norm`` to the full row. A column slice of
this GEMM measured bit-identical to the same columns of the full GEMM up to 32 rows (tests/test_ctx_proj_overlap.py
checks it on the served shapes); above that the library may choose another split of the reduction, so larger blocks
stay unsharded unless the cap is raised. Because that equality depends on the GEMM library's choice, the first
shard-eligible call runs a self-check on the served weight: for every row count 8, 16, ... up to the cap, this rank's
slice computed on the side stream must equal the same columns of the full GEMM bit for bit; the ranks agree on the
verdict (minimum over the tensor-parallel group) and shard mode stays off for good if any rank saw a difference. Every
rank takes the same path for a block, so the gather (a collective) is issued on all of them or none.

``MUSE_VR_CTX_PROJ_OVERLAP_SIDE_GATHER=1`` (default off; shard mode only) also moves the gather and ``hidden_norm`` to the
side stream, right after the slice GEMM, so the worker's call only waits for finished work. The gather cannot share the
group's communicator there: its barrier windows live on that communicator's semaphores, which kernels on the worker's
stream use at the same time (the verify records exchange, the all-reduces), and two streams reserving windows on one
semaphore can interleave differently on different ranks. So the side gather runs on a communicator of its own (a second
custom all-reduce instance over the same group: own symmetric workspace and semaphores, used by this call site only)
with its own pull output. Same kernels on the same values: the result is bit-identical (tests/dist_ctx_proj_side_gather.py).
The dedicated plane is created collectively at the first eligible call (every rank reaches it at the same call) and the
ranks agree on its availability (minimum over the group); without it the call keeps the worker-stream gather.

``MUSE_VR_CTX_PROJ_OVERLAP_SIDE_KV=1`` (default off; with the side gather) also runs the next consumer's batched K/V
projection of those rows on the side stream: the worker writes the context K/V into the draft cache through
``FusedKVMaterializeHelper.materialize`` (one GEMM of the normalized rows with every draft layer's stacked K/V weight,
then one norm + RoPE kernel), and that GEMM depends only on the rows. Its product is handed to ``materialize`` (hooked:
only for the very tensor this component returned), which then runs the rest of its work unchanged. Same GEMM call (same
operands, shapes and output layout) and the same remaining kernels: bit-identical cache rows
(tests/dist_ctx_proj_side_gather.py).

Lifetime invariant: ``hidden`` can be the verify graph's static output buffer, which the next verify replay overwrites.
The side stream's read of it is safe because ``take`` joins the side branch before that replay; the paths that skip the
projection (``_drop``) carry no rows today. Any future path that skips the projection with rows must join (or copy the
rows) before the next verify replay.

``MUSE_VR_CTX_PROJ_OVERLAP_SHARD_PIN=1`` (default off; with shard mode) computes the slice with the full GEMM's split
of the reduction instead of ``F.linear`` on the slice (whose configuration the library chooses for the slice shape and
which can split the reduction differently): ``ctx_proj_lt`` takes the cuBLASLt heuristic's configuration for the full
shape and lists slice configurations that keep its split count and reduction scheme (the heuristic's picks for the slice
shape with those two values put in, then the full configuration itself); the slice runs on the side stream with a
workspace of its own. The self-check then gives a verdict per row count instead of one for all: for every row count 8,
16, ... up to the cap (default 384 in this mode) that ``MUSE_VR_CTX_PROJ_OVERLAP_SHARD_PIN_ROWS`` admits (``lo-hi``
ranges separated by ``,``; default all), every listed configuration's slice on the side stream is compared with the same
columns of PyTorch's full GEMM on two input sets and the equal ones are timed on the side stream; the ranks agree per
count (bitwise: minimum over the group; time: maximum over the group) and keep, all with the same index, the bitwise
configuration whose slowest rank is fastest. Only the row counts with such a configuration run sharded. Every other row
count (no bitwise configuration on some rank, rejected by the library, not admitted, or not a multiple of 8) runs the
unsharded projection.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import weakref

import torch

from . import bed, pull_bounded

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/distributed/*.py": ("get_tp_group",),
    "srt/distributed/device_communicators/vocab_gather.py": (
        "NVLinkVocabGather",
        "make_vocab_gather",
        "NcclVocabGather",
    ),
    "srt/distributed/device_communicators/custom_all_reduce_v2.py": (
        "CustomAllReduceV2",
    ),
    "srt/speculative/spec_utils.py": ("draft_tp_context",),
    "kernels/ops/speculative/fused_kv_materialize.py": ("_fused_norm_rope_stacked",),
    "srt/models/dflash.py": ("DFlashDraftModel",),
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
}


logger = logging.getLogger(__name__)

NAME = "ctx_proj_overlap"  # env MUSE_VR_CTX_PROJ_OVERLAP
TARGET_INIT = "sglang.srt.speculative.dflash_worker_v2.DFlashWorkerV2.__init__"
TARGET_FWD = "sglang.srt.managers.tp_worker.TpModelWorker.forward_batch_generation"
TARGET_PROJ = "sglang.srt.models.dflash.DFlashDraftModel.project_target_hidden"

# sha256 of the SGLang sources (relative to the directory holding the ``sglang`` package) whose call order this
# component relies on: the worker (verify forward, then sampling, then the projection of the same hidden tensor), the
# target worker's forward signature, and the draft model's projection.
PINS = {
    "sglang/srt/speculative/dflash_worker_v2.py": (
        "7d462f5f175828bcc3b4d47fcdfa33cb6af4b14669ef016eeb79b1a5191eec8c",
        "8f82de8d666f81f3c1aa43dc651006f4182ff58593fdfe4eea28c5ee2a14bfc0",
    ),
    "sglang/srt/managers/tp_worker.py": (
        "73b69f16193de495b4536b8e80a540de4dbd5d2c7f2f1f1696b8ebdf8f70d44d",
        "3668a19cd3f6d5e923cd4620dd32e880dc9006411402b3c95c648566a8a377e3",
    ),
    "sglang/srt/models/dflash.py": (
        "59ae9eab87fd3416828604018c36ad6a57be07c4ff6d17f699bb9c0be8c719ff",
        "b7b541368cbbb072a679d08ba5bf0f952f4a43f02b6c078562b2ea7a9ddfd01b",
    ),
}

ENV_SHARD_MAX_ROWS = "MUSE_VR_CTX_PROJ_OVERLAP_SHARD_MAX_ROWS"
DEFAULT_SHARD_MAX_ROWS = 32
DEFAULT_SHARD_MAX_ROWS_PIN = (
    384  # with MUSE_VR_CTX_PROJ_OVERLAP_SHARD_PIN=1: every count is checked on its own
)
ENV_PIN_ROWS = "MUSE_VR_CTX_PROJ_OVERLAP_SHARD_PIN_ROWS"

_workers: weakref.WeakValueDictionary[int, object] = weakref.WeakValueDictionary()
_pending: dict = {}  # id(draft model) -> (key, result, event); result = projection, ("shard", local slice, group) or
# ("side", normalized projection)
_gathers: dict = {}
_side_planes: dict = {}  # (id(group), width, dtype) -> (dedicated communicator, its gather plane) or None
SIDE_PLANE_MAX_BYTES = (
    1 << 20
)  # workspace cap of the dedicated communicator (its push / pull buffers stay unused)
TARGET_KV = "sglang.kernels.ops.speculative.fused_kv_materialize.FusedKVMaterializeHelper.materialize"
PINS_KV = {
    "sglang/kernels/ops/speculative/fused_kv_materialize.py": "b8dd985b7764d3212e9e3045b2daa5be0c481215346ab28f2513f8201e9cbfeb"
}
_kv_pre: dict = {}  # id(helper) -> (normalized rows handed to the worker, their K/V projection)
_streams: dict = {}
_shard_checked: dict = {}  # id(draft model) -> verdict of the shard self-check, agreed by every rank
_pin_tables: dict = {}  # id(draft model) -> {row count: pinned configuration}, agreed by every rank (pin mode)
_tls = threading.local()


def enabled() -> bool:
    return bed.enabled(NAME)


def pins_match() -> tuple[bool, str]:
    for rel, want in PINS.items():
        if not bed.pinned(rel, want):
            return False, f"{rel} differs from the pinned revision"
    return True, "ok"


def _key(t: torch.Tensor):
    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, t.device)


def _side_stream(device: torch.device) -> torch.cuda.Stream:
    s = _streams.get(device)
    if s is None:
        s = _streams[device] = torch.cuda.Stream(device=device)
    return s


def _drop(dm_id: int) -> None:
    """Discard an unconsumed projection; the current stream waits for it so its inputs stay intact meanwhile."""
    p = _pending.pop(dm_id, None)
    if p is not None:
        torch.cuda.current_stream().wait_event(p[2])
        bed.count(NAME, "dropped")


def _pin_mode() -> bool:
    return bed.flag(NAME + "_shard_pin")


def _shard_max_rows() -> int:
    return bed.env_int(
        ENV_SHARD_MAX_ROWS,
        DEFAULT_SHARD_MAX_ROWS_PIN if _pin_mode() else DEFAULT_SHARD_MAX_ROWS,
    )


def parse_pin_rows(text: str | None, cap: int) -> list:
    """Row counts the pin self-check covers: multiples of 8 up to ``cap`` inside the ``lo-hi[,lo-hi...]`` ranges of
    ``text`` (None or empty: all of them). A malformed entry admits nothing."""
    rows = list(range(8, cap + 1, 8))
    if not text or not text.strip():
        return rows
    spans = []
    for item in filter(None, (t.strip() for t in text.split(","))):
        lo, _, hi = item.partition("-")
        try:
            lo_i = int(lo)
            hi_i = int(hi) if hi else lo_i
        except ValueError:
            return []
        spans.append((lo_i, hi_i))
    return [m for m in rows if any(lo <= m <= hi for lo, hi in spans)]


def _shard_group(draft_model, hidden: torch.Tensor):
    """(group, pinned configuration) to shard over, or (None, None) for the unsharded path; the configuration is None
    outside pin mode (the slice is then ``F.linear`` on the slice). Depends only on facts every rank shares
    (configuration, shapes, verdicts agreed over the group), so all ranks decide alike."""
    group = _shard_group_any(draft_model, hidden)
    if group is None or not _pin_mode():
        return group, None
    weight = draft_model.fc.weight
    if not (
        hidden.dtype == weight.dtype == torch.bfloat16
        and hidden.is_contiguous()
        and weight.is_contiguous()
    ):
        return None, None  # the pinned slice takes contiguous bf16 rows only
    algo = _pin_table(draft_model, group, hidden.device).get(hidden.shape[0])
    return (group, algo) if algo is not None else (None, None)


def _shard_group_any(draft_model, hidden: torch.Tensor):
    if not bed.flag(NAME + "_shard") or hidden.shape[0] > _shard_max_rows():
        return None
    fc = getattr(draft_model, "fc", None)
    if (
        type(fc) is not torch.nn.Linear
        or fc.bias is not None
        or getattr(draft_model, "quantizable", True)
    ):
        return None
    from sglang.srt.distributed import get_tp_group

    group = get_tp_group()
    w = group.world_size
    if (
        w < 2
        or fc.out_features % w
        or (fc.out_features // w * fc.weight.element_size()) % 16
    ):
        return None
    if _pin_mode():
        return group  # the per-count pin table decides (``_shard_group``)
    if not _shard_self_check(draft_model, group, hidden.device):
        return None
    return group


def _shard_self_check(draft_model, group, device: torch.device) -> bool:
    """Whether this rank's column slice of the draft ``fc`` GEMM, computed on the side stream, equals the same columns
    of the full GEMM bit for bit at every row count shard mode serves (8, 16, ... up to the cap), on every rank. Runs
    once per draft model, at the first shard-eligible call (eager; every rank reaches it at the same call)."""
    key = id(draft_model)
    verdict = _shard_checked.get(key)
    if verdict is not None:
        return verdict
    weight = draft_model.fc.weight
    w, r = group.world_size, group.rank_in_group
    n = weight.shape[0] // w
    gen = torch.Generator(device=device)
    gen.manual_seed(0x5EED)
    main = torch.cuda.current_stream(device)
    side = _side_stream(device)
    rows = list(range(8, _shard_max_rows() + 1, 8))
    ok = True
    with torch.inference_mode():
        for m in rows:
            x = torch.randn(m, weight.shape[1], device=device, generator=gen).to(
                weight.dtype
            )
            full = torch.nn.functional.linear(x, weight)[
                :, r * n : (r + 1) * n
            ].contiguous()
            side.wait_stream(main)
            with torch.cuda.stream(side):
                part = torch.nn.functional.linear(x, weight[r * n : (r + 1) * n])
            main.wait_stream(side)
            ok = ok and bool(
                torch.equal(part.view(torch.int16), full.view(torch.int16))
            )
    flag = torch.tensor([1 if ok else 0], dtype=torch.int32)
    torch.distributed.all_reduce(
        flag, op=torch.distributed.ReduceOp.MIN, group=group.cpu_group
    )
    verdict = bool(flag.item())
    _shard_checked[key] = verdict
    bed.report(NAME + ":shard_check", verdict, rows=rows, world=w, this_rank_bitwise=ok)
    return verdict


MAX_PIN_CANDIDATES = 4  # slice configurations tried per row count (ctx_proj_lt.slice_candidates: 3 + the full one)
PIN_TIMING_ITERS = 5


def _pin_table(draft_model, group, device: torch.device) -> dict:
    """{row count: configuration} for the counts at which some slice configuration with the full GEMM's split of the
    reduction, computed on the side stream, equals the same columns of PyTorch's full GEMM bit for bit on two input sets
    on EVERY rank. Among those, the one whose slowest rank is fastest (side-stream GPU time, median of a few runs) is
    kept, the same index on every rank (bitwise flags agreed by a minimum, times by a maximum over the group). Runs once
    per draft model, at the first shard-eligible call (eager; every rank reaches it at the same call)."""
    key = id(draft_model)
    table = _pin_tables.get(key)
    if table is not None:
        return table
    from . import ctx_proj_lt

    weight = draft_model.fc.weight
    w, r = group.world_size, group.rank_in_group
    n_full, k = weight.shape
    n = n_full // w
    w_slice = weight[r * n : (r + 1) * n]
    rows = parse_pin_rows(os.environ.get(ENV_PIN_ROWS), _shard_max_rows())
    gen = torch.Generator(device=device)
    gen.manual_seed(0x5EED)
    main = torch.cuda.current_stream(device)
    side = _side_stream(device)
    c = MAX_PIN_CANDIDATES
    flags = torch.zeros((max(len(rows), 1), c), dtype=torch.int32)
    times = torch.full((max(len(rows), 1), c), float("inf"), dtype=torch.float64)
    lists = torch.zeros(
        (max(len(rows), 1), 2), dtype=torch.int64
    )  # per count: hash of the candidate list, twice
    plans, why, lib = {}, {}, None
    with torch.inference_mode():
        try:
            lib = ctx_proj_lt.library_version()
            for m in rows:
                algo = ctx_proj_lt.full_shape_algo(device, m, n_full, k)
                cands = (
                    []
                    if algo is None
                    else ctx_proj_lt.slice_candidates(device, algo, m, n, k)[:c]
                )
                if cands:
                    plans[m] = cands
            if plans:
                with torch.cuda.stream(side):
                    ctx_proj_lt.reserve_workspace(
                        device, max(ws for cs in plans.values() for _, ws in cs)
                    )
                have = ctx_proj_lt.workspace_bytes(device)
                for m in list(plans):  # never more workspace than reserved
                    plans[m] = [(a, ws) for a, ws in plans[m] if ws <= have]
                    if not plans[m]:
                        del plans[m]
                        why[f"workspace_m{m}"] = (
                            "configuration needs more workspace than reserved"
                        )
        except (OSError, RuntimeError, AttributeError) as e:
            plans = {}
            why["setup"] = f"{type(e).__name__}: {e}"
        for j, m in enumerate(rows):
            if m not in plans:
                continue
            digest = hashlib.sha256(
                repr([ctx_proj_lt.config(a) for a, _ in plans[m]]).encode()
            ).digest()
            lists[j, 0] = lists[j, 1] = int.from_bytes(digest[:7], "little")
            xs = [
                (torch.randn(m, k, device=device, generator=gen) * scale).to(
                    weight.dtype
                )
                for scale in (1.0, 3.0)
            ]
            fulls = [
                torch.nn.functional.linear(x, weight)[
                    :, r * n : (r + 1) * n
                ].contiguous()
                for x in xs
            ]
            for i, (algo, _) in enumerate(plans[m]):
                try:
                    same = True
                    for x, full in zip(xs, fulls):
                        side.wait_stream(main)
                        with torch.cuda.stream(side):
                            part = ctx_proj_lt.run_slice(algo, x, w_slice)
                        main.wait_stream(side)
                        same = same and bool(
                            torch.equal(part.view(torch.int16), full.view(torch.int16))
                        )
                    if same:
                        side.wait_stream(main)
                        ev = [
                            torch.cuda.Event(enable_timing=True)
                            for _ in range(PIN_TIMING_ITERS + 1)
                        ]
                        with torch.cuda.stream(side):
                            ctx_proj_lt.run_slice(algo, xs[0], w_slice)  # warm
                            ev[0].record(side)
                            for t in range(PIN_TIMING_ITERS):
                                ctx_proj_lt.run_slice(algo, xs[0], w_slice)
                                ev[t + 1].record(side)
                        main.wait_stream(side)
                        ev[-1].synchronize()
                        dts = sorted(
                            ev[t].elapsed_time(ev[t + 1])
                            for t in range(PIN_TIMING_ITERS)
                        )
                        flags[j, i] = 1
                        times[j, i] = dts[len(dts) // 2]
                except (
                    RuntimeError
                ) as e:  # a library error: this candidate is out at this count
                    main.wait_stream(side)
                    why[f"error_m{m}_c{i}"] = str(e)
    this_rank = [m for j, m in enumerate(rows) if int(flags[j].max()) == 1]
    lists_min = lists.clone()
    torch.distributed.all_reduce(
        flags, op=torch.distributed.ReduceOp.MIN, group=group.cpu_group
    )
    torch.distributed.all_reduce(
        times, op=torch.distributed.ReduceOp.MAX, group=group.cpu_group
    )
    torch.distributed.all_reduce(
        lists_min, op=torch.distributed.ReduceOp.MIN, group=group.cpu_group
    )
    torch.distributed.all_reduce(
        lists, op=torch.distributed.ReduceOp.MAX, group=group.cpu_group
    )
    table, chosen = {}, {}
    for j, m in enumerate(rows):
        if m not in plans or int(lists_min[j, 0]) != int(lists[j, 0]):
            continue  # unavailable here, or another rank listed other candidates: unsharded
        ok = [i for i in range(len(plans[m])) if int(flags[j, i]) == 1]
        if ok:
            best = min(ok, key=lambda i: float(times[j, i]))
            table[m] = plans[m][best][0]
            chosen[m] = best
    agreed = sorted(table)
    _pin_tables[key] = table
    bed.report(
        NAME + ":shard_check",
        bool(table),
        mode="pin",
        world=w,
        lib_version=lib,
        rows_pinned=_spans(agreed),
        rows_pinned_sha16=hashlib.sha256(_spans(agreed).encode()).hexdigest()[:16],
        rows_unpinned=_spans([m for m in rows if m not in table]),
        this_rank_bitwise=_spans(this_rank),
        candidate=",".join(f"{m}:{i}" for m, i in sorted(chosen.items())) or "-",
        slowest_rank_us=",".join(
            f"{m}:{float(times[rows.index(m), i]) * 1e3:.1f}"
            for m, i in sorted(chosen.items())
        )
        or "-",
        library_errors=sum(1 for k_ in why if k_.startswith("error_")),
        **why,
    )
    return table


def _spans(rows: list) -> str:
    """``[8, 16, 24, 40]`` -> ``"8-24,40"`` (step 8)."""
    out, start, prev = [], None, None
    for m in rows:
        if start is None:
            start = prev = m
        elif m == prev + 8:
            prev = m
        else:
            out.append(f"{start}-{prev}" if prev != start else f"{start}")
            start = prev = m
    if start is not None:
        out.append(f"{start}-{prev}" if prev != start else f"{start}")
    return ",".join(out) or "-"


def _gather_columns(group, local: torch.Tensor):
    """``[rows, world * n]`` from every rank's ``[rows, n]`` (rank r's columns at r * n), or None. SGLang's NVLink pull
    all-gather into a symmetric stack (bit copy; bounded barrier waits with ``MUSE_VR_PULL_BOUNDED=1``), then a
    transposing copy out of it."""
    from sglang.srt.distributed.device_communicators.vocab_gather import (
        NVLinkVocabGather,
        make_vocab_gather,
    )

    rows, n = local.shape
    w = group.world_size
    key = (id(group), n, local.dtype)
    if key not in _gathers:
        pull_bounded.prepare(
            local.dtype
        )  # before the plane's collective set-up (MUSE_VR_PULL_BOUNDED)
        vg = make_vocab_gather(
            group, local_width=n, dtype=local.dtype, symm_rows=_shard_max_rows()
        )
        _gathers[key] = (
            vg
            if isinstance(vg, NVLinkVocabGather) and vg.pull_out is not None
            else None
        )
        bed.report(NAME + ":shard_plane", _gathers[key] is not None, world=w, width=n)
    vg = _gathers[key]
    if vg is None or w * rows > vg.pull_out.shape[0]:
        return None
    return _gather_columns_on(vg, group, local, NAME)


def _gather_columns_on(vg, group, local: torch.Tensor, site: str) -> torch.Tensor:
    """``[rows, world * n]`` from every rank's ``[rows, n]`` through gather plane ``vg`` (capacity checked by the caller)."""
    rows, n = local.shape
    w = group.world_size
    assert w * rows <= vg.pull_out.shape[0], "gather rows beyond the symmetric stack"
    stack = vg.pull_out[: w * rows]
    # SGLang's pull kernel, or its copy with bounded barrier waits (MUSE_VR_PULL_BOUNDED=1): same bytes
    pull_bounded.all_gather_pull(
        vg.comm, local, stack, out_mc_ptr=vg.pull_mc_ptr, site=site
    )
    return stack.view(w, rows, n).permute(1, 0, 2).reshape(rows, w * n)


def _side_plane(group, n: int, dtype: torch.dtype, device: torch.device):
    """The gather plane on a communicator of its own (MUSE_VR_CTX_PROJ_OVERLAP_SIDE_GATHER), or None. Collective at the
    first call for a (group, width, dtype); the ranks agree on the verdict (minimum over the group)."""
    key = (id(group), n, dtype)
    if key in _side_planes:
        p = _side_planes[key]
        return None if p is None else p[1]
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )
    from sglang.srt.distributed.device_communicators.vocab_gather import (
        NcclVocabGather,
        NVLinkVocabGather,
    )

    pull_bounded.prepare(
        dtype
    )  # before the plane's collective set-up (MUSE_VR_PULL_BOUNDED)
    ca = CustomAllReduceV2(
        group=group.cpu_group, device=device, max_size=SIDE_PLANE_MAX_BYTES
    )
    usable = (
        not ca.disabled
        and bool(getattr(ca, "has_multicast", False))
        and ca.obj.pull is not None
        and ca.obj.push is not None
        and ca.world_size == group.world_size
        and n % (128 // torch.finfo(dtype).bits) == 0
    )
    vg = None
    # the pull output is a collective allocation: every rank makes it whenever the communicator was built
    if not ca.disabled:
        vg = NVLinkVocabGather(
            ca_comm=ca,
            group=group,
            local_width=n,
            dtype=dtype,
            symm_rows=_shard_max_rows(),
            fallback=NcclVocabGather(group),
        )
        usable = usable and vg.pull_out is not None
    flag = torch.tensor([1 if usable else 0], dtype=torch.int32)
    torch.distributed.all_reduce(
        flag, op=torch.distributed.ReduceOp.MIN, group=group.cpu_group
    )
    usable = bool(flag.item())
    _side_planes[key] = (ca, vg) if usable else None
    bed.report(NAME + ":side_plane", usable, world=group.world_size, width=n)
    return vg if usable else None


def launch(
    draft_model, hidden: torch.Tensor, owns_attention: bool, kv_helper=None
) -> bool:
    """Issue ``draft_model.project_target_hidden(hidden)`` (or, sharded, this rank's column slice of its GEMM) on the
    side stream; returns whether it was issued."""
    from sglang.srt.speculative.spec_utils import draft_tp_context

    _drop(id(draft_model))
    main = torch.cuda.current_stream(hidden.device)
    side = _side_stream(hidden.device)
    group, algo = _shard_group(draft_model, hidden)
    vg = None
    if group is not None and bed.flag(NAME + "_side_gather"):
        # collective at the first call (every rank decides alike), so before any side-stream work
        vg = _side_plane(
            group,
            draft_model.fc.out_features // group.world_size,
            draft_model.fc.weight.dtype,
            hidden.device,
        )
        if vg is not None and group.world_size * hidden.shape[0] > vg.pull_out.shape[0]:
            vg = None
    side.wait_stream(main)
    hidden.record_stream(
        side
    )  # read on the side stream: its memory is not reused before that work completes
    _tls.bypass = True
    try:
        with (
            torch.cuda.stream(side),
            torch.inference_mode(),
            draft_tp_context(owns_attention),
        ):
            if group is None:
                result = draft_model.project_target_hidden(hidden)
            else:
                n = draft_model.fc.out_features // group.world_size
                r = group.rank_in_group
                w_slice = draft_model.fc.weight[r * n : (r + 1) * n]
                if algo is None:
                    local = torch.nn.functional.linear(hidden, w_slice)
                else:  # pin mode: the full GEMM's own configuration (checked bitwise at this row count, every rank)
                    from . import ctx_proj_lt

                    local = ctx_proj_lt.run_slice(algo, hidden, w_slice)
                if vg is None:
                    result = ("shard", local, group)
                else:
                    # gather + hidden_norm here, on the dedicated plane (the worker's call only waits for them)
                    full = _gather_columns_on(vg, group, local, NAME + ":side")
                    normed = draft_model.hidden_norm(full)
                    result = ("side", normed, _side_kv_projection(kv_helper, normed))
    finally:
        _tls.bypass = False
    ev = torch.cuda.Event()
    ev.record(side)
    _pending[id(draft_model)] = (_key(hidden), result, ev)
    return True


def take(draft_model, target_hidden: torch.Tensor):
    """The side-stream projection of exactly ``target_hidden`` (current stream made to wait for it), or None."""
    p = _pending.pop(id(draft_model), None)
    if p is None:
        return None
    key, result, ev = p
    main = torch.cuda.current_stream(target_hidden.device)
    main.wait_event(ev)
    stale = key != _key(target_hidden)
    if isinstance(result, tuple) and result[0] == "side":
        # gathered and normalized on the side stream (every rank issued its gather in launch)
        if stale:
            bed.count(NAME, "stale")
            return None
        bed.count(NAME, f"side:rows={target_hidden.shape[0]}")
        result[1].record_stream(main)
        if result[2] is not None:
            helper, proj = result[2]
            proj.record_stream(main)
            _kv_pre[id(helper)] = (result[1], proj)
        return result[1]
    if isinstance(result, tuple):
        _, local, group = result
        local.record_stream(main)
        # gathered even when stale: every rank issued this slice, so every rank joins the collective
        full = _gather_columns(group, local)
        if stale or full is None:
            bed.count(NAME, "stale" if stale else "stock:no pull plane")
            return None
        bed.count(NAME, f"shard:rows={target_hidden.shape[0]}")
        return draft_model.hidden_norm(full)
    if stale:
        bed.count(NAME, "stale")
        return None
    bed.count(NAME, f"overlap:rows={target_hidden.shape[0]}")
    result.record_stream(main)
    return result


def _side_kv_projection(helper, rows: torch.Tensor):
    """(helper, the batched K/V projection of ``rows``) issued on the current (side) stream, or None. The same call as
    ``FusedKVMaterializeHelper.materialize`` makes: ``torch.mm(rows, flat_kv_weight_t, out=...)`` into a contiguous
    ``[rows, layers * 2 * kv_size]`` buffer."""
    if helper is None or not _state_kv["on"]:
        return None
    w = getattr(helper, "flat_kv_weight_t", None)
    if (
        w is None
        or rows.dtype != w.dtype
        or rows.device != w.device
        or not getattr(helper, "_mm_out_supported", False)
    ):
        bed.count(NAME, "side_kv:not eligible")
        return None
    proj = torch.empty(
        (rows.shape[0], helper.n_layers * helper.layer_out_dim),
        dtype=rows.dtype,
        device=rows.device,
    )
    torch.mm(rows, w, out=proj)
    return helper, proj


# _materialize_pre is derived from SGLang (https://github.com/sgl-project/sglang), Apache License 2.0; see
# include/LICENSE.sglang. Source at revision a55ab7d: python/sglang/kernels/ops/speculative/fused_kv_materialize.py
# (FusedKVMaterializeHelper.materialize), Copyright 2023-2024 SGLang Team.
# Modified by NVIDIA: the batched K/V projection is taken from a product computed earlier (the same torch.mm call on the
# same rows) instead of being computed here; the input checks, RoPE cache, workspaces, norm + RoPE kernel and per-layer
# writes are unchanged.
def _materialize_pre(
    helper,
    ctx_hidden: torch.Tensor,
    proj: torch.Tensor,
    positions: torch.Tensor,
    write_layer_kv,
):
    from sglang.kernels.ops.speculative.fused_kv_materialize import (
        _fused_norm_rope_stacked,
    )

    total_ctx = ctx_hidden.shape[0]
    if positions.ndim != 1:
        positions = positions.reshape(-1)
    if positions.numel() != total_ctx:
        raise ValueError(
            "positions must match ctx_hidden token count for fused KV materialization: "
            f"positions={positions.numel()}, total_ctx={total_ctx}."
        )
    if positions.device != helper.device:
        positions = positions.to(
            device=helper.device, dtype=torch.int64, non_blocking=True
        )
    elif positions.dtype != torch.int64:
        positions = positions.to(torch.int64)
    max_position = (
        helper.max_position_hint
        if helper.max_position_hint is not None
        else int(positions.max().item())
    )
    cos_sin_cache = helper._ensure_rope_cache(max_position)
    helper._ensure_workspace(total_ctx, ctx_hidden.dtype)
    proj_out = proj.view(total_ctx, helper.n_layers, helper.layer_out_dim)
    tmp_k = helper._k_workspace[:, :total_ctx]
    tmp_v = helper._v_workspace[:, :total_ctx]
    cache_k, cache_v = _fused_norm_rope_stacked(
        proj_out,
        helper.k_norm_weights,
        helper.eps_values,
        cos_sin_cache,
        positions,
        helper.num_kv_heads,
        helper.head_dim,
        helper.rotary_dim,
        k_out=tmp_k,
        v_out=tmp_v,
    )
    for layer_idx in range(helper.n_layers):
        write_layer_kv(layer_idx, cache_k[layer_idx], cache_v[layer_idx])


def _around_materialize(original, helper, *args, **kwargs):
    pre = _kv_pre.pop(id(helper), None)
    ctx_hidden = kwargs.get("ctx_hidden", args[0] if args else None)
    # only the very tensor this component handed over (its product was computed from it, with the weight's dtype)
    if pre is None or ctx_hidden is not pre[0]:
        if pre is not None:
            bed.count(NAME, "side_kv:not this tensor")
        return original(helper, *args, **kwargs)
    positions = kwargs.get("positions", args[1] if len(args) > 1 else None)
    write_layer_kv = kwargs.get("write_layer_kv", args[2] if len(args) > 2 else None)
    if positions is None or write_layer_kv is None or ctx_hidden.shape[0] == 0:
        return original(helper, *args, **kwargs)
    bed.count(NAME, f"side_kv:rows={ctx_hidden.shape[0]}")
    return _materialize_pre(helper, ctx_hidden, pre[1], positions, write_layer_kv)


_state_kv = {"on": False}


def _around_proj(original, draft_model, target_hidden):
    if getattr(_tls, "bypass", False):
        return original(draft_model, target_hidden)
    proj = take(draft_model, target_hidden)
    if proj is not None:
        return proj
    return original(draft_model, target_hidden)


def _after_init(result, worker, *args, **kwargs):
    tw = getattr(worker, "_target_worker", None)
    if tw is not None:
        _workers[id(tw)] = worker
    return result


def _eligible_draft(dm) -> bool:
    from sglang.srt.models.dflash import DFlashDraftModel

    # only models that use DFlashDraftModel's own projection (whose calls the hook above sees)
    return (
        isinstance(dm, DFlashDraftModel)
        and type(dm).project_target_hidden is DFlashDraftModel.project_target_hidden
    )


def _around_fwd(original, target_worker, *args, **kwargs):
    out = original(target_worker, *args, **kwargs)
    if not kwargs.get("is_verify", False):
        return out
    worker = _workers.get(id(target_worker))
    if worker is None:
        return out
    hidden = getattr(getattr(out, "logits_output", None), "hidden_states", None)
    dm = getattr(worker, "draft_model", None)
    if (
        hidden is None
        or not isinstance(hidden, torch.Tensor)
        or hidden.ndim != 2
        or not hidden.is_cuda
        or torch.cuda.is_current_stream_capturing()
        or dm is None
        or not _eligible_draft(dm)
    ):
        bed.count(NAME, "stock:verify output not eligible")
        return out
    launch(
        dm,
        hidden,
        bool(getattr(worker, "draft_owns_attention", False)),
        kv_helper=getattr(worker, "_fused_kv_helper", None),
    )
    return out


def register() -> bool:
    if not enabled():
        return False
    ok, why = pins_match()
    if not ok:
        bed.report(NAME, False, reason=why)
        return False
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    def proj_hook(original, *args, **kwargs):
        if kwargs or len(args) != 2:
            return original(*args, **kwargs)
        return _around_proj(original, *args)

    HookRegistry.register(TARGET_INIT, _after_init, HookType.AFTER)
    HookRegistry.register(TARGET_FWD, _around_fwd, HookType.AROUND)
    HookRegistry.register(TARGET_PROJ, proj_hook, HookType.AROUND)
    side_kv = enable_side_kv(HookRegistry, HookType)
    bed.report(NAME, True, side_gather=bed.flag(NAME + "_side_gather"), side_kv=side_kv)
    return True


def enable_side_kv(registry=None, hook_type=None) -> bool:
    """Hook ``FusedKVMaterializeHelper.materialize`` for MUSE_VR_CTX_PROJ_OVERLAP_SIDE_KV (with the side gather and the
    pinned source); returns whether the side K/V projection is on."""
    if not (bed.flag(NAME + "_side_gather") and bed.flag(NAME + "_side_kv")):
        return False
    for rel, want in PINS_KV.items():
        if not bed.pinned(rel, want):
            bed.report(
                NAME + ":side_kv",
                False,
                reason=f"{rel} differs from the pinned revision",
            )
            return False
    if registry is None:
        import sglang.srt.plugins.hook_registry as _sgl_srt_plugins_hook_registry

        registry = _sgl_srt_plugins_hook_registry.HookRegistry
        import sglang.srt.plugins.hook_registry as _sgl_srt_plugins_hook_registry

        hook_type = _sgl_srt_plugins_hook_registry.HookType
    registry.register(
        TARGET_KV,
        lambda original, *a, **k: _around_materialize(original, *a, **k),
        hook_type.AROUND,
    )
    _state_kv["on"] = True
    return True

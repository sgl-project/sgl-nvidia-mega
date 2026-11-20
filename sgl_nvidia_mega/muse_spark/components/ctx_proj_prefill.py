# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``ctx_proj_prefill_shard`` (env ``MUSE_VR_CTX_PROJ_PREFILL_SHARD=1``, default off).

On a sequence-parallel prefill pass (``--enable-layernorm-sp``; ``layernorm_sp.runs_sp``), every tensor-parallel rank
holds the target's DFLASH taps for its own rows only. SGLang's Muse Spark causal-LM wrapper packs them
(``[rows / tp, 5 x hidden]``) and all-gathers them to every rank (``layernorm_sp.sp_exit_gather``), so that the DFLASH
worker can run the draft's context projection (``DFlashDraftModel.project_target_hidden``: the ``fc`` GEMM, then
``hidden_norm``) on all rows on every rank.

With this component, that exit runs the ``fc`` GEMM on the rank's own rows and all-gathers its result (``[rows / tp,
hidden]``, five times fewer bytes than the taps) on the same group with the same call; the worker's
``project_target_hidden`` call on that tensor then applies only ``hidden_norm``, the stock call on the same values.

Equality with the stock projection. Each row of the stock ``fc`` output is the reduction over the input features in
the order the library's configuration for the full shape fixes: its split of the reduction dimension and the reduction
of the split parts. The rank's GEMM runs through ``ctx_proj_lt`` with a configuration for its own shape that keeps the
full shape's split count and reduction scheme (the heuristic's first pick for the local shape with those two values put
in), on a workspace of its own. That this gives the same bits is not assumed: a class of passes (the full shape's
complete configuration at that row count, the local configuration, the group width) is engaged only after a check on
the first pass of that class, on every rank: that pass runs the stock gather, the stock ``fc`` on the gathered taps and the sharded
form, compares them bit for bit, agrees the verdict over the group (minimum), and RETURNS the stock gather (the worker
then projects as stock). Later passes of a verified class run the sharded form; a class that failed stays stock.

Every rank must take the same form on a pass (the two forms issue different collectives), so the decision for a row
count is agreed over the group once (host collective on the group's CPU communicator) and cached: the class key must
be equal on every rank and verified on every rank.

Engaged only for the call it is meant for: inside a DFLASH target forward with full hidden capture on an extend batch
(marked by the target worker hook), the first ``sp_exit_gather`` call whose input is 2-D, ``bf16``, contiguous, CUDA,
as wide as the draft ``fc`` input, with the pass's token count at least the batch's extend token count (rows past it
are the trailing padding of the forward's token dimension: the attention tensor-parallel alignment and the token
buckets of the prefill pass; the consumer drops them before it projects, so both forms are compared and served on the
leading extend rows), outside compilation, for an eligible draft (``DFlashDraftModel``'s own projection, an unquantized
``torch.nn.Linear`` without bias). A pass with fewer rows than the extend count runs stock. Everything else calls
the stock function. The marked forward is an eager extend: the
exit hook raises if it finds itself inside CUDA-graph capture. The projection hook takes only the tensor the exit
returned on that pass (or a leading-row view of it); any other projected-width tensor is refused loudly. Replicated
(non-SP) passes never reach the exit and run stock. A library error in the first-pass check keeps the check's
collective sequence (that rank votes 0, the class stays stock).

``MUSE_VR_CTX_PROJ_PREFILL_SHARD_MIN_ROWS`` (default 1): passes with fewer extend tokens stay stock.

Extended form (either knob; both default off, and with both off every path above is unchanged):

``MUSE_VR_CTX_PROJ_PREFILL_SHARD_PICK=1``: the local GEMM of a class uses the fastest of up to four configurations
(the heuristic's first picks for the local shape and the full configuration itself, each with the full shape's split
count and reduction scheme put in) instead of the first. The class's first-pass check runs every candidate on the
rank's rows, compares each with the same rows of the stock ``fc`` bit for bit, and times the bitwise ones on the
current stream (median of a few runs); one host collective agrees the verdicts (minimum) and the times (maximum over
the ranks): the choice is the candidate bitwise on every rank whose slowest rank is fastest, and it replaces the first
candidate only when it is faster by at least ``PICK_MARGIN`` (near-ties keep the first). The class key holds every
candidate's descriptor, so all ranks choose among the same list.

``MUSE_VR_CTX_PROJ_PREFILL_SHARD_CHUNKS=N`` (2..4): the rank's rows run as up to N row chunks (at least 256 rows each),
each chunk's GEMM on the current stream and its gather (the group's ``all_gather_into_tensor``, the stock collective)
on a side stream of this component behind an event recorded after that GEMM, so the gather of chunk j overlaps the GEMM
of chunk j + 1; the side stream copies each gathered chunk into its place of the rank-major output, and the current
stream waits for the side stream's last work before the output is handed out. Each chunk row count has its own
configuration (same rule as above), and the first-pass check runs the whole chunked form end to end against the stock
``fc``. Between the first chunk's event and the final wait the current stream runs only this component's GEMMs, so the
group's communicator is never used from two streams at once.

Stream use: the base form uses the current stream only (no new streams, events or cross-stream edges); the gather is
the stock group's ``all_gather_into_tensor``. The extended form with chunks adds the edges described above.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import threading
import weakref

import torch

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/models/dflash.py": ("DFlashDraftModel",),
    "srt/runtime_context.py": ("get_parallel", "get_exec"),
    "srt/model_executor/forward_batch_info.py": ("CaptureHiddenMode",),
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
}


logger = logging.getLogger(__name__)

NAME = "ctx_proj_prefill_shard"  # env MUSE_VR_CTX_PROJ_PREFILL_SHARD
TARGET_INIT = "sglang.srt.speculative.dflash_worker_v2.DFlashWorkerV2.__init__"
TARGET_FWD = "sglang.srt.managers.tp_worker.TpModelWorker.forward_batch_generation"
TARGET_GATHER = "sglang.srt.layers.layernorm_sp.sp_exit_gather"
TARGET_PROJ = "sglang.srt.models.dflash.DFlashDraftModel.project_target_hidden"

# sha256 of the SGLang sources whose call order this component relies on: the worker's extend branch (target forward
# with full capture, then the projection of logits_output.hidden_states), the target worker's forward signature, the
# Muse Spark wrapper (pack, then sp_exit_gather of the taps on an SP pass, then the logits processor), the SP exit
# gather, the logits processor's full-capture store, the taps packer and the draft's projection.
PINS = {
    "sglang/srt/speculative/dflash_worker_v2.py": "8f82de8d666f81f3c1aa43dc651006f4182ff58593fdfe4eea28c5ee2a14bfc0",
    "sglang/srt/managers/tp_worker.py": "3668a19cd3f6d5e923cd4620dd32e880dc9006411402b3c95c648566a8a377e3",
    "sglang/srt/models/muse_spark.py": "8bafbb8802fb0d9f81c190e87b9012fca546182be08bef6c8b3db814d27bfe19",
    "sglang/srt/layers/layernorm_sp.py": "bc814d983b945e5bdd41db436966eb4b3fee4c410bb1dcccfc98857d4b026bda",
    "sglang/srt/layers/logits_processor.py": "3558330bccf3ea0e46aabdaa0dee6ae7daf294c8e15b9a4e9744dab6883c4398",
    "sglang/srt/layers/aux_hidden_states.py": "1c0b2055a5bfef2b06096ffac1fdea073cf471f0c48ccd3b4ab577118facae92",
    "sglang/srt/models/dflash.py": "b7b541368cbbb072a679d08ba5bf0f952f4a43f02b6c078562b2ea7a9ddfd01b",
}

ENV_MIN_ROWS = "MUSE_VR_CTX_PROJ_PREFILL_SHARD_MIN_ROWS"
ENV_PICK = "MUSE_VR_CTX_PROJ_PREFILL_SHARD_PICK"  # 1: per class, the fastest of up to 4 bitwise local configurations
ENV_CHUNKS = "MUSE_VR_CTX_PROJ_PREFILL_SHARD_CHUNKS"  # 2..4: local GEMM in row chunks, each chunk's gather on a side stream
CHUNK_MIN_ROWS = (
    256  # smallest local row chunk (fewer chunks when the rank's rows are fewer)
)
PICK_CANDIDATES = 3  # heuristic picks asked for the local shape (plus the full configuration itself: at most 4)
PICK_REPS = 7  # timed runs per candidate in the first-pass check (median; round-robin after two warm runs each)
PICK_MARGIN = 0.03  # another candidate replaces the first only if its agreed time is at least this much lower
CACHE_MAX = 8192  # row counts whose agreed decision is cached (oldest dropped first)

_tls = (
    threading.local()
)  # .ctx = (draft model, expected token count) while a marked target forward runs
_workers: weakref.WeakValueDictionary[int, object] = weakref.WeakValueDictionary()
_pending: dict = {}  # id(draft model) -> the projected rows handed out on this pass (pre-norm, [tokens, hidden])
_plans: dict = {}  # (device, rows, width) -> plan dict or None (library rejected / no configuration)
_decisions: dict = {}  # (device, projected rows, local rows, width) -> agreed engage decision (bool)
_verdicts: dict = {}  # class key -> agreed verdict of the first-pass check (bool)
_ws: dict = {}  # device index -> workspace of this component (current stream only)
_choices: dict = {}  # class key (extended form) -> {local GEMM rows: agreed candidate index}
_side: dict = {}  # device index -> this component's side stream (extended form with chunks)
_lock = threading.Lock()


def enabled() -> bool:
    return bed.enabled(NAME)


def pins_match() -> tuple[bool, str]:
    for rel, want in PINS.items():
        if not bed.pinned(rel, want):
            return False, f"{rel} differs from the pinned revision"
    return True, "ok"


def _eligible_draft(dm) -> bool:
    from sglang.srt.models.dflash import DFlashDraftModel

    if (
        not isinstance(dm, DFlashDraftModel)
        or type(dm).project_target_hidden is not DFlashDraftModel.project_target_hidden
    ):
        return False
    return _eligible_fc(dm)


def _eligible_fc(dm) -> bool:
    fc = getattr(dm, "fc", None)
    return (
        type(fc) is torch.nn.Linear
        and fc.bias is None
        and not getattr(dm, "quantizable", True)
        and fc.weight.dtype == torch.bfloat16
        and fc.weight.is_contiguous()
    )


def _cache_put(d: dict, key, value) -> None:
    if len(d) >= CACHE_MAX:
        d.pop(next(iter(d)))
    d[key] = value


def _workspace(device: torch.device, nbytes: int) -> torch.Tensor:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    buf = _ws.get(idx)
    if buf is None or buf.numel() < nbytes:
        buf = torch.empty(max(nbytes, 256), dtype=torch.uint8, device=device)
        _ws[idx] = buf
    return buf


def plan(device: torch.device, rows: int, local_rows: int, n: int, k: int):
    """The local GEMM for a pass of ``rows`` tokens, ``local_rows`` of them on this rank (padded share): {algo, ws,
    key} with ``key`` = (full split count, full reduction scheme, local configuration), or None (no configuration)."""
    ck = (device.index, rows, local_rows)
    if ck in _plans:
        return _plans[ck]
    from . import ctx_proj_lt as lt

    out = None
    full = lt.full_shape_algo(device, rows, n, k)
    if full is not None:
        fcfg = lt.config(full)
        if (
            _force_split_for_test() is not None
        ):  # negative control only (tests): a different split of the reduction
            fcfg = list(fcfg)
            fcfg[2], fcfg[3] = _force_split_for_test()
            full = lt._build(lt._device_state(device)["handle"], tuple(fcfg))
            fcfg = tuple(fcfg) if full is not None else None
        if full is not None:
            cands = lt.slice_candidates(device, full, local_rows, n, k, count=1)
            if cands:
                algo, ws = cands[0]
                # class key: the full pick's complete descriptor and the local configuration's (all nine config values)
                out = {
                    "algo": algo,
                    "ws": ws,
                    "key": (tuple(int(v) for v in fcfg), tuple(lt.config(algo))),
                }
    _cache_put(_plans, ck, out)
    return out


def _force_split_for_test():
    v = os.environ.get("MUSE_VR_CTX_PROJ_PREFILL_SHARD_TEST_FORCE_SPLIT")
    if not v:
        return None
    a, b = (int(x) for x in v.split(","))
    return a, b


def _key_hash(key) -> int:
    return int.from_bytes(
        hashlib.sha256(repr(key).encode()).digest()[:7], "little"
    )  # fits int64 (MIN of +/-)


def _agree(group, values: list) -> list:
    """Elementwise minimum over the group (host collective on its CPU communicator)."""
    t = torch.tensor(values, dtype=torch.int64)
    torch.distributed.all_reduce(
        t, op=torch.distributed.ReduceOp.MIN, group=group.cpu_group
    )
    return [int(v) for v in t.tolist()]


def _local_fc(p: dict, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    from . import ctx_proj_lt as lt

    return lt.run_slice(p["algo"], x, weight, workspace=_workspace(x.device, p["ws"]))


def _row_form(
    group, p: dict, taps: torch.Tensor, weight: torch.Tensor, rows: int
) -> torch.Tensor:
    local = _local_fc(p, taps, weight)
    out = local.new_empty((local.shape[0] * group.world_size, local.shape[1]))
    group.all_gather_into_tensor(out, local)
    return out[:rows]


def _ext_settings() -> tuple:
    """(pick, chunks) of the extended form; (False, 1) keeps the base form (``plan`` / ``_row_form`` / ``_check_pass``)."""
    return bed.env_int(ENV_PICK, 0) == 1, min(4, max(1, bed.env_int(ENV_CHUNKS, 1)))


def _segments(local_rows: int, chunks: int) -> list:
    """Row ranges ``[a, b)`` of the rank's local rows: up to ``chunks`` chunks of a multiple of 8 rows (the last one takes
    the rest), at least CHUNK_MIN_ROWS rows per chunk (fewer chunks otherwise)."""
    c = max(1, min(chunks, local_rows // CHUNK_MIN_ROWS))
    while c > 1:
        step = ((local_rows + c - 1) // c + 7) // 8 * 8
        if local_rows - step * (c - 1) >= CHUNK_MIN_ROWS:
            break
        c -= 1
    if c == 1:
        return [(0, local_rows)]
    segs, a = [], 0
    while a < local_rows:
        b = min(local_rows, a + step)
        segs.append((a, b))
        a = b
    return segs


def plan_ext(
    device: torch.device,
    rows: int,
    local_rows: int,
    n: int,
    k: int,
    pick: bool,
    chunks: int,
):
    """The extended form's local GEMMs for a pass of ``rows`` tokens, ``local_rows`` of them on this rank: {segs, shapes,
    key}. ``segs``: the rank's row chunks (``_segments``); ``shapes``: per distinct chunk row count, the configurations
    to try, in order (``ctx_proj_lt.slice_candidates``: the full pick's split count and reduction scheme forced; the first
    one only without ``pick``), each (algo, workspace bytes, nine config values); ``key``: the class key (full
    descriptor, every candidate's descriptor per chunk row count, chunk count, pick). None: no configuration."""
    ck = ("ext", device.index, rows, local_rows, bool(pick), int(chunks))
    if ck in _plans:
        return _plans[ck]
    from . import ctx_proj_lt as lt

    out = None
    full = lt.full_shape_algo(device, rows, n, k)
    if full is not None:
        fcfg = lt.config(full)
        if (
            _force_split_for_test() is not None
        ):  # negative control only (tests): a different split of the reduction
            fcfg = list(fcfg)
            fcfg[2], fcfg[3] = _force_split_for_test()
            full = lt._build(lt._device_state(device)["handle"], tuple(fcfg))
            fcfg = tuple(fcfg) if full is not None else None
        if full is not None:
            segs = _segments(local_rows, chunks)
            shapes = {}
            for m in sorted({b - a for a, b in segs}):
                cands = lt.slice_candidates(
                    device, full, m, n, k, count=PICK_CANDIDATES if pick else 1
                )
                cands = cands if pick else cands[:1]
                if not cands:
                    shapes = None
                    break
                shapes[m] = [
                    (algo, ws, tuple(int(v) for v in lt.config(algo)))
                    for algo, ws in cands
                ]
                plant = os.environ.get("MUSE_VR_CTX_PROJ_PREFILL_SHARD_TEST_PLANT_CAND")
                if plant is not None and int(plant) < len(
                    shapes[m]
                ):  # tests only: that candidate gets another split
                    i = int(plant)
                    cfg = list(shapes[m][i][2])
                    cfg[2], cfg[3] = 4, 2
                    bad = lt._build(lt._device_state(device)["handle"], tuple(cfg))
                    bws = (
                        None
                        if bad is None
                        else lt.slice_workspace(device, bad, m, n, k)
                    )
                    if bws is not None:
                        shapes[m][i] = (bad, bws, tuple(cfg))
            if shapes:
                key = (
                    "ext",
                    tuple(int(v) for v in fcfg),
                    tuple((m, tuple(c[2] for c in shapes[m])) for m in sorted(shapes)),
                    len(segs),
                    int(bool(pick)),
                )
                out = {"segs": segs, "shapes": shapes, "key": key}
    _cache_put(_plans, ck, out)
    return out


def _side_stream(device: torch.device) -> torch.cuda.Stream:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    s = _side.get(idx)
    if s is None:
        s = _side[idx] = torch.cuda.Stream(device=device)
    return s


def _seg_gemm(
    p: dict,
    choice: dict,
    taps: torch.Tensor,
    weight: torch.Tensor,
    a: int,
    b: int,
    safe: bool,
):
    """The rank's rows ``[a, b)`` through the chosen configuration for that row count, on the current stream: (output,
    library ok). ``safe``: a library error gives zeros and ok False (the caller keeps its collective sequence)."""
    from . import ctx_proj_lt as lt

    algo, ws, _ = p["shapes"][b - a][choice[b - a]]
    if not safe:
        return lt.run_slice(
            algo, taps[a:b], weight, workspace=_workspace(taps.device, ws)
        ), True
    try:
        if (
            os.environ.get("MUSE_VR_CTX_PROJ_PREFILL_SHARD_TEST_RAISE") == "1"
        ):  # tests only: a library error here
            raise RuntimeError("test-only library error")
        return lt.run_slice(
            algo, taps[a:b], weight, workspace=_workspace(taps.device, ws)
        ), True
    except Exception as e:  # noqa: BLE001
        bed.count(NAME, f"plan error: check {type(e).__name__}")
        return taps.new_zeros((b - a, weight.shape[0])), False


def _ext_form(
    group,
    p: dict,
    choice: dict,
    taps: torch.Tensor,
    weight: torch.Tensor,
    rows: int,
    safe: bool = False,
):
    """The extended sharded form: the rank's local rows through the chosen configurations, gathered rank-major into
    ``[w x local rows, n]`` (the first ``rows`` returned) with the group's gather (the stock collective). One chunk: GEMM
    and gather on the current stream (the base form). Several chunks: each chunk's GEMM on the current stream; behind it
    (an event recorded after that GEMM, waited for by this component's side stream) the side stream gathers the chunk
    and copies the gathered chunk into its rank-major place; the current stream waits for the side stream's last work
    before this returns. The chunk tensors and the output are marked as used by the side stream (record_stream).
    Returns (output, library ok)."""
    w = group.world_size
    L, n = taps.shape[0], weight.shape[0]
    segs = p["segs"]
    out = taps.new_empty((w * L, n))
    if len(segs) == 1:
        local, ok = _seg_gemm(p, choice, taps, weight, 0, L, safe)
        group.all_gather_into_tensor(out, local)
        return out[:rows], ok
    main = torch.cuda.current_stream(taps.device)
    side = _side_stream(taps.device)
    out3 = out.view(w, L, n)
    lib_ok = True
    for a, b in segs:
        local, ok = _seg_gemm(p, choice, taps, weight, a, b, safe)
        lib_ok = lib_ok and ok
        ready = torch.cuda.Event()
        ready.record(main)
        side.wait_event(ready)
        local.record_stream(side)
        with torch.cuda.stream(side):
            part = torch.empty((w * (b - a), n), dtype=local.dtype, device=local.device)
            group.all_gather_into_tensor(part, local)
            out3[:, a:b].copy_(part.view(w, b - a, n))
    done = torch.cuda.Event()
    done.record(side)
    main.wait_event(done)
    out.record_stream(side)
    return out[:rows], lib_ok


def _time_candidates(cands: list, x: torch.Tensor, weight: torch.Tensor) -> list:
    """GPU time (ns, median of PICK_REPS runs, round-robin over the candidates after two warm runs each) of each
    ``(algo, ws, cfg)`` on ``x``, on the current stream; None entries are skipped (returned as 0)."""
    from . import ctx_proj_lt as lt

    live = [i for i, c in enumerate(cands) if c is not None]
    for _ in range(2):
        for i in live:
            lt.run_slice(
                cands[i][0], x, weight, workspace=_workspace(x.device, cands[i][1])
            )
    ev = {i: [] for i in live}
    for _ in range(PICK_REPS):
        for i in live:
            e0, e1 = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            e0.record()
            lt.run_slice(
                cands[i][0], x, weight, workspace=_workspace(x.device, cands[i][1])
            )
            e1.record()
            ev[i].append((e0, e1))
    torch.cuda.synchronize(x.device)
    out = [0] * len(cands)
    for i in live:
        ts = sorted(int(e0.elapsed_time(e1) * 1e6) for e0, e1 in ev[i])
        out[i] = max(1, ts[len(ts) // 2])
    return out


def _check_pass_ext(
    original, group, dm, p, key, hidden_states, expected, rows, w, *args, **kwargs
):
    """First pass of an extended-form class on every rank. Collective sequence, the same on every rank: the stock
    gather; one host MIN over (every candidate's bitwise verdict on this rank's rows, minus its GPU time) - the agreed
    choice per chunk row count is the candidate bitwise on every rank whose slowest rank is fastest (ties: the earlier
    one); then, if every chunk row count has a choice (every rank knows it from the same agreed vector), the form's
    gathers (``_ext_form`` with the choice, compared end to end with the stock fc on the leading ``expected`` rows);
    then one host MIN of the verdict. A library error keeps the sequence (zeros, verdict 0). Returns the stock gather."""
    from . import ctx_proj_lt as lt

    gathered = original(hidden_states, *args, **kwargs)
    weight = dm.fc.weight
    ref = torch.nn.functional.linear(gathered[:expected], weight)
    pick = bool(p["key"][4])
    L = hidden_states.shape[0]
    base = group.rank_in_group * L  # this rank's first row in the gathered order
    shapes = sorted(p["shapes"])
    oks, times = [], []
    for m in shapes:
        segs_m = [(a, b) for a, b in p["segs"] if b - a == m]
        live = []
        for algo, ws, _cfg in p["shapes"][m]:
            good = True
            try:
                if (
                    os.environ.get("MUSE_VR_CTX_PROJ_PREFILL_SHARD_TEST_RAISE") == "1"
                ):  # tests only
                    raise RuntimeError("test-only library error")
                if os.environ.get(
                    "MUSE_VR_CTX_PROJ_PREFILL_SHARD_TEST_RAISE_CAND"
                ) == str(len(live)):  # tests only
                    raise RuntimeError("test-only library error (one candidate)")
                for a, b in segs_m:
                    o = lt.run_slice(
                        algo,
                        hidden_states[a:b],
                        weight,
                        workspace=_workspace(hidden_states.device, ws),
                    )
                    v = max(
                        0, min(b - a, expected - (base + a))
                    )  # rows of this chunk inside the projected rows
                    if v and not torch.equal(
                        o[:v].view(torch.int16),
                        ref[base + a : base + a + v].view(torch.int16),
                    ):
                        good = False
            except Exception as e:  # noqa: BLE001 - this candidate votes 0 on this rank; the sequence stays the same
                bed.count(NAME, f"plan error: check {type(e).__name__}")
                good = False
            live.append((algo, ws, _cfg) if good else None)
            oks.append(1 if good else 0)
        a0, b0 = segs_m[0]
        ts = (
            _time_candidates(live, hidden_states[a0:b0], weight)
            if pick and any(live)
            else [0] * len(live)
        )
        times.extend(ts)
    agreed = _agree(group, oks + [-t for t in times])
    a_ok, a_t = agreed[: len(oks)], [-v for v in agreed[len(oks) :]]
    choice, i = {}, 0
    for m in shapes:
        best = None
        for ci in range(len(p["shapes"][m])):
            if a_ok[i + ci] == 1 and (best is None or a_t[i + ci] < a_t[i + best]):
                best = ci
        if (
            best is not None
            and best != 0
            and a_ok[i] == 1
            and a_t[i + best] > (1.0 - PICK_MARGIN) * a_t[i]
        ):
            best = 0  # near-tie with the first candidate: keep it (timing noise must not pick a slower one)
        if best is not None:
            choice[m] = best
        i += len(p["shapes"][m])
    ok = False
    if len(choice) == len(shapes):
        got, lib_ok = _ext_form(
            group, p, choice, hidden_states, weight, rows, safe=True
        )
        got = got[:expected]
        ok = bool(
            lib_ok
            and ref.shape == got.shape
            and torch.equal(ref.view(torch.int16), got.view(torch.int16))
        )
    (final,) = _agree(group, [1 if ok else 0])
    _verdicts[key] = bool(final)
    if final:
        _choices[key] = dict(choice)
    bed.count(NAME, "check:" + ("verified" if final else "FAILED"))
    if pick and final:
        bed.count(
            NAME, "pick:" + ",".join(f"{m}:{c}" for m, c in sorted(choice.items()))
        )
    if rows != expected:
        bed.count(NAME, f"check:pad:{rows - expected}")
    if not final:
        logger.warning(
            "muse_vr_plugin: %s class %s FAILED its first-pass check (this rank: candidates bitwise %s, "
            "end to end %s); that class stays stock",
            NAME,
            key,
            oks,
            ok,
        )
    _report_classes()
    return gathered


def _step_aside(original, why: str, *args, **kwargs):
    bed.count(NAME, "stock:" + why)
    return original(*args, **kwargs)


def _around_gather(original, hidden_states, *args, **kwargs):
    ctx = getattr(_tls, "ctx", None)
    if ctx is None:
        return original(hidden_states, *args, **kwargs)
    dm, expected = ctx
    fc = dm.fc
    if not (
        isinstance(hidden_states, torch.Tensor)
        and hidden_states.ndim == 2
        and hidden_states.shape[1] == fc.in_features
    ):
        return original(
            hidden_states, *args, **kwargs
        )  # another exit (e.g. the routed-expert ids): not ours
    _tls.ctx = None  # one taps exit per marked forward
    num_tokens = kwargs.get("num_tokens", args[0] if len(args) == 1 else None)
    if len(args) + len(kwargs) != 1 or num_tokens is None:
        return _step_aside(original, "call form", hidden_states, *args, **kwargs)
    if torch.cuda.is_current_stream_capturing():
        # the marked forward is an eager extend: a capture here breaks this component's contract
        raise RuntimeError(
            f"{NAME}: the SP exit of a marked DFLASH extend runs inside CUDA-graph capture"
        )
    if torch.compiler.is_compiling():
        return _step_aside(original, "compiling", hidden_states, *args, **kwargs)
    if not (
        hidden_states.is_cuda
        and hidden_states.dtype == torch.bfloat16
        and hidden_states.is_contiguous()
    ):
        return _step_aside(original, "taps layout", hidden_states, *args, **kwargs)
    rows = int(num_tokens)
    if (
        rows < expected
    ):  # fewer rows than the batch's extend tokens: not a padded pass of this batch
        return _step_aside(original, "padded rows", hidden_states, *args, **kwargs)
    if expected < bed.env_int(ENV_MIN_ROWS, 1):
        return _step_aside(original, "below min rows", hidden_states, *args, **kwargs)
    from sglang.srt.runtime_context import get_parallel

    group = get_parallel().tp_group
    w = group.world_size
    local_rows = int(hidden_states.shape[0])
    if w < 2 or local_rows * w < rows or local_rows != math.ceil(rows / w):
        return _step_aside(original, "shape", hidden_states, *args, **kwargs)
    # Rows past ``expected`` are trailing padding of the forward's token dimension (attention-TP alignment and token
    # buckets); the stock consumer drops them before it projects (the DFLASH worker trims the captured rows to its cache
    # locations), so the stock GEMM it runs has ``expected`` rows: the full shape of the plan, the reference of the
    # first-pass check and the rows compared are the leading ``expected`` rows; the local GEMM covers this rank's
    # ceil(rows / w) rows as for any SP pass (pad rows are computed, never read).
    device = hidden_states.device
    dk = (device.index, expected, local_rows, w)
    engage = _decisions.get(dk)
    pick, chunks = _ext_settings()
    ext = pick or chunks > 1
    try:
        if ext:
            p = plan_ext(
                device,
                expected,
                local_rows,
                fc.out_features,
                fc.in_features,
                pick,
                chunks,
            )
        else:
            p = plan(device, expected, local_rows, fc.out_features, fc.in_features)
    except Exception as e:  # noqa: BLE001 - no configuration on this rank: the agreement below keeps every rank stock
        bed.count(NAME, f"plan error: {type(e).__name__}")
        p = None
    if engage is None:
        key = None if p is None else (p["key"], w)
        h = 0 if key is None else _key_hash(key)
        verified = 1 if (key is not None and _verdicts.get(key) is True) else 0
        failed = 1 if (key is not None and _verdicts.get(key) is False) else 0
        lo, neg_hi, all_verified, none_failed, all_planned = _agree(
            group, [h, -h, verified, 1 - failed, 1 if p is not None else 0]
        )
        same = lo == -neg_hi and all_planned == 1
        if same and all_verified == 1:
            engage = True
        elif same and none_failed == 1 and key is not None and key not in _verdicts:
            check = _check_pass_ext if ext else _check_pass
            return check(
                original,
                group,
                dm,
                p,
                key,
                hidden_states,
                expected,
                rows,
                w,
                *args,
                **kwargs,
            )
        else:
            engage = False
            bed.count(
                NAME,
                "decide:stock " + ("classes differ" if not same else "class failed"),
            )
        _cache_put(_decisions, dk, engage)
    if not engage:
        return _step_aside(
            original, "class not verified", hidden_states, *args, **kwargs
        )
    if ext:
        out, _ = _ext_form(
            group, p, _choices[(p["key"], w)], hidden_states, fc.weight, rows
        )
        bed.count(NAME, f"ext:chunks{len(p['segs'])}")
    else:
        out = _row_form(group, p, hidden_states, fc.weight, rows)
    _pending[id(dm)] = out
    bed.count(NAME, f"row:w{w}")
    if rows != expected:
        bed.count(
            NAME, f"pad:{rows - expected}"
        )  # INFO: an alignment-padded pass took the sharded form
    return out


def _check_pass(
    original, group, dm, p, key, hidden_states, expected, rows, w, *args, **kwargs
):
    """First pass of a class on every rank: stock gather + stock fc on the ``expected`` leading rows (the rows the stock
    consumer projects) vs the sharded form on the same rows, bit for bit; verdict agreed by minimum; returns the stock
    gather (the worker projects as stock on this pass)."""
    # Collective sequence, the same on every rank: the stock gather, the sharded form's gather, the host MIN. A library
    # error in this rank's GEMM keeps that sequence (zeros take the GEMM's place, the rank votes 0).
    gathered = original(hidden_states, *args, **kwargs)
    ref = torch.nn.functional.linear(gathered[:expected], dm.fc.weight)
    weight = dm.fc.weight
    try:
        if (
            os.environ.get("MUSE_VR_CTX_PROJ_PREFILL_SHARD_TEST_RAISE") == "1"
        ):  # tests only: a library error here
            raise RuntimeError("test-only library error")
        local = _local_fc(p, hidden_states, weight)
        lib_ok = True
    except Exception as e:  # noqa: BLE001 - this rank votes 0; the collective sequence stays the same
        bed.count(NAME, f"plan error: check {type(e).__name__}")
        local = hidden_states.new_zeros((hidden_states.shape[0], weight.shape[0]))
        lib_ok = False
    got = local.new_empty((local.shape[0] * w, local.shape[1]))
    group.all_gather_into_tensor(got, local)
    got = got[:expected]
    ok = bool(
        lib_ok
        and ref.shape == got.shape
        and torch.equal(ref.view(torch.int16), got.view(torch.int16))
    )
    (agreed,) = _agree(group, [1 if ok else 0])
    _verdicts[key] = bool(agreed)
    bed.count(NAME, "check:" + ("verified" if agreed else "FAILED"))
    if rows != expected:
        bed.count(
            NAME, f"check:pad:{rows - expected}"
        )  # INFO: the class's first pass was an alignment-padded one
    if not agreed:
        logger.warning(
            "muse_vr_plugin: %s class %s FAILED its first-pass check (this rank bitwise=%s); that class "
            "stays stock",
            NAME,
            key,
            ok,
        )
    _report_classes()
    return gathered


def class_list_hash() -> str:
    """sha256 (16 hex) of the sorted (class key, verdict) list; equal on every rank by construction (agreed verdicts)."""
    items = sorted((repr(k), v) for k, v in _verdicts.items())
    if _choices:  # extended form: the agreed choices are part of the class list (equal on every rank by construction)
        items += sorted((repr(k), repr(sorted(v.items()))) for k, v in _choices.items())
    return hashlib.sha256(repr(items).encode()).hexdigest()[:16]


def _report_classes() -> None:
    bed.report(
        NAME + ":classes",
        True,
        seen=len(_verdicts),
        verified=sum(1 for v in _verdicts.values() if v),
        failed=sum(1 for v in _verdicts.values() if not v),
        class_list_sha16=class_list_hash(),
        failed_keys=[repr(k) for k, v in _verdicts.items() if not v][:8],
        choices=[f"{k!r}:{sorted(v.items())}" for k, v in _choices.items()][:8],
    )


def _around_proj(original, draft_model, target_hidden):
    pend = _pending.pop(id(draft_model), None)
    if pend is None:
        return original(draft_model, target_hidden)
    if (
        isinstance(target_hidden, torch.Tensor)
        and target_hidden.dtype == pend.dtype
        and target_hidden.device == pend.device
        and target_hidden.ndim == 2
        and target_hidden.shape[1] == pend.shape[1]
        and target_hidden.stride() == pend.stride()
        and target_hidden.data_ptr() == pend.data_ptr()
        and target_hidden.shape[0] <= pend.shape[0]
    ):
        bed.count(NAME, "take")
        return draft_model.hidden_norm(target_hidden)
    raise RuntimeError(
        f"{NAME}: the projection was called on another tensor than the one this pass handed out "
        f"({tuple(getattr(target_hidden, 'shape', ()))} vs {tuple(pend.shape)})"
    )


def _after_init(result, worker, *args, **kwargs):
    tw = getattr(worker, "_target_worker", None)
    if tw is not None:
        _workers[id(tw)] = worker
    return result


def _encoder_swa_replay():
    """The server's encoder SWA bounded-replay setting (True / False), or None when it cannot be read."""
    try:
        from sglang.srt.runtime_context import get_exec

        return bool(
            getattr(get_exec().features, "enable_encoder_swa_bounded_replay", False)
        )
    except Exception:  # noqa: BLE001
        return None


def _expected_tokens(batch):
    lens = getattr(batch, "extend_lens", None)
    if not isinstance(lens, (list, tuple)) or not lens:
        return None
    return int(sum(lens))


def _around_fwd(original, target_worker, *args, **kwargs):
    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

    worker = _workers.get(id(target_worker))
    batch = args[0] if args else kwargs.get("batch")
    if (
        worker is None
        or kwargs.get("is_verify", False)
        or kwargs.get("capture_hidden_mode") != CaptureHiddenMode.FULL
        or batch is None
        or not batch.forward_mode.is_extend()
        or batch.forward_mode.is_idle()
    ):
        return original(target_worker, *args, **kwargs)
    dm = getattr(worker, "draft_model", None)
    if _pending.pop(id(dm), None) is not None:
        bed.count(NAME, "drop:stale handout")
    expected = _expected_tokens(batch)
    if dm is None or not _eligible_draft(dm) or expected is None:
        bed.count(NAME, "stock:not eligible")
        return original(target_worker, *args, **kwargs)
    replay = _encoder_swa_replay()
    if replay is not False:
        # the replay fold drops rows of the captured taps; an unreadable setting counts as on
        bed.count(
            NAME,
            "stock:encoder swa replay" if replay else "stock:exec config unreadable",
        )
        return original(target_worker, *args, **kwargs)
    _tls.ctx = (dm, expected)
    try:
        return original(target_worker, *args, **kwargs)
    finally:
        if getattr(_tls, "ctx", None) is not None:
            bed.count(
                NAME, "stock:no sp exit"
            )  # a replicated pass (not sequence-parallel): stock projection
        _tls.ctx = None


def register() -> bool:
    if not enabled():
        return False
    ok, why = pins_match()
    if not ok:
        bed.report(NAME, False, reason=why)
        return False
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    def gather_hook(original, *args, **kwargs):
        if not args:
            return original(*args, **kwargs)
        return _around_gather(original, *args, **kwargs)

    def proj_hook(original, *args, **kwargs):
        if kwargs or len(args) != 2:
            return original(*args, **kwargs)
        return _around_proj(original, *args)

    HookRegistry.register(TARGET_INIT, _after_init, HookType.AFTER)
    HookRegistry.register(TARGET_FWD, _around_fwd, HookType.AROUND)
    HookRegistry.register(TARGET_GATHER, gather_hook, HookType.AROUND)
    HookRegistry.register(TARGET_PROJ, proj_hook, HookType.AROUND)
    pick, chunks = _ext_settings()
    bed.report(
        NAME, True, min_rows=bed.env_int(ENV_MIN_ROWS, 1), pick=pick, chunks=chunks
    )
    return True

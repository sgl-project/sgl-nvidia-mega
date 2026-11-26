# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``step_graphs`` (env ``MUSE_VR_STEP_GRAPHS=1``, default off): the DFLASH decode step's eager region after the
target verify runs from captured CUDA graphs instead of op-by-op launches.

After the verify forward, SGLang's DFLASH worker (``DFlashWorkerV2.forward_batch_generation``, decode path) runs a few
dozen small operations from Python: the verify-logits gather, the sampling and acceptance, the commit, the publication
of the new sequence lengths, the draft-KV materialization and the next draft input. Their GPU time is small; their host
cost (Python, operator dispatch, launches) is most of the scheduler loop's host work, which bounds the step time once
the GPU step is short.

SGLang's method stays in charge of every step: it runs as stock and makes every decision. Two runs of consecutive calls
it makes are *segments*:

- segment A: from the first of {``ctx_proj_overlap.launch``, ``LogitsProcessor.finish_deferred_gather``,
  ``LogitsProcessor.deferred_argmax``, ``apply_dflash_verify_logits_adjustments``, ``_accept_block``} to the return of
  ``_accept_block``;
- segment B: from ``FutureMap.publish`` (the method's ``on_publish`` callback) to the return of
  ``_make_next_draft_input_decode``, with ``_draft_write_ids`` and ``_append_target_hidden_to_draft_kv_by_loc`` between;
- segment P: the draft sampler's ``stage_sampling_params`` (before the draft forward);
- segment E (opt-in): the draft block's preparation (``_prepare_dflash_draft_block_unchecked``: block ids, positions,
  cache locations) and its embedding (the target's input-embedding module), when the step embeds through that module.

Whatever the method does after ``_accept_block`` and before ``on_publish`` always runs eagerly and unchanged. (On SGLang
main the synthetic-acceptance forcing is the last statement of ``_accept_block``; it is kept out of segment A and runs
eagerly right after it, see ``_FORCING``.) The
wrappers are installed outermost (other components' hooks on these calls run inside the segment, as they do eagerly)
once the worker is built.

Capture. When a segment has run eagerly ``MUSE_VR_STEP_GRAPHS_WARMUP`` times (default 3) for the same batch shape, its
next occurrence is captured: the tensors the segment may read from the method's locals are classified by the warm-up
steps: every small tensor the segment only reads is copied into a private buffer that the segment's calls receive
instead (as an argument, or set on the locals' object for the step), and only large persistent buffers, the pools the
method passes whole (``PERSISTENT_PATHS``, e.g. the token table) and the worker's own attributes are read in place
(and guarded). So whether a step replays never depends on where a rank's allocator
placed a per-batch tensor: every tensor-parallel rank takes the same path in every step. Inputs the segment WRITES are never copied (a replay would write the copy and leave the method's
tensor as it was): the result buffers the method passes in (``OUTPUT_PATHS``) and every input whose contents the
segment's own calls changed during the warm-up are read in place at their current address and guarded; when one moves
(SGLang grows its result buffers for a larger batch) the shape is captured again at the new address. Each graph
also keeps every CUDA tensor held by the worker's attributes and by the module state of the other plugin components and
of SGLang's decode-path modules at its capture (``_pins``): code that later replaces such a buffer (a pool or cache
re-made for a larger batch) would otherwise free memory the graph still reads and writes. And when a decode batch
larger than any before arrives (the step in which such buffers are re-made), every graph is dropped and captured
again (all ranks alike: it depends only on the batch size). The first call starts a CUDA-graph capture on the current stream, the calls run as usual and are
recorded, the last call ends the capture; the generator state is restored and the graph replayed once, which does the
step's work. Checks during capture: no operation may be recorded between two calls of a segment (the capture graph's
node count must not change there); streams that joined the capture are joined back before it ends; a wait on an event
recorded by an earlier segment's capture of the same step is left out (that segment joined it into this stream); any
other wait on an event or stream outside the capture fails the capture, and so does a recorded copy from or to host
memory (a per-step host value would replay frozen, or from / to a freed buffer). A failed capture is recovered by running the
recorded calls again eagerly into the tensors they returned, and that shape stays eager.

Replay. The first call of a segment checks the guards (an input read in place that moved: this step runs eagerly and
the shape is captured again; a moved result buffer at its next occurrence, without a new warm-up), copies the per-step
tensors into the private buffers (grouped multi-tensor copies; a large tensor one plain copy) and replays the graph; every call of the segment then returns what it returned
at capture (objects as shallow copies). Integer attributes of the worker that the captured calls changed are set as
the eager calls would set them. Two variants per shape alternate between steps, so outputs are double-buffered as the
stock path double-buffers them. Generator: the draws come from the default CUDA generator through PyTorch's graph-safe
RNG, so a replay draws what the eager calls would draw from the same generator state.

Steps aside (stock, counted): extend / idle / mixed batches, grammar, logprobs, hidden-state outputs, more than
``MUSE_VR_STEP_GRAPHS_MAX_BS`` (default 8) requests, another capture in progress, Python other than 3.10-3.12, a large
per-step input. ``MUSE_VR_STEP_GRAPHS_VERIFY=1`` (debug; host syncs): segments replay as usual, except one segment per
step (the graphed segments in rotation), which is replayed on the side first; then every tensor it reads in place is
put back, the generator state restored, and the method's calls for it run eagerly (that step uses their results); their returns, the in-place tensors and (segment B) the
draft KV rows written are compared bit for bit with the replay's (``bed.shadow_result``), and every staged input's own
tensor must come out of the eager calls unchanged (``staged_untouched``: a staged input the segment writes).
With VERIFY, ``MUSE_VR_STEP_GRAPHS_VERIFY_LOG=<n>`` (debug) logs every capture (key, worker counters, result
addresses) and the values and addresses behind the first ``n`` mismatches, and ``MUSE_VR_STEP_GRAPHS_VERIFY_RANGE=<vocab>``
adds a token-id range check after each replay of A, B and E (logged within the same budget); a failing log call is
counted and never changes the step.
``MUSE_VR_STEP_GRAPHS_SHARED=1`` (opt-in): fewer staging copies per step (a source copied once per step even when
several segments read it; views of the worker's own buffers and earlier segments' graph results read in place; see
``ENV_SHARED``).
``MUSE_VR_STEP_GRAPHS_LAUNCH_LATE=1`` (opt-in): segment A's capture records the side-stream projection launch after the
logits gather, with the launch's dependencies unchanged, so a replay submits the gather first as the eager step does
(see ``ENV_LAUNCH_LATE``).
``MUSE_VR_STEP_GRAPHS_OBSERVE=1`` only records, per segment and shape, how many addresses each input took.
``MUSE_VR_STEP_GRAPHS_GLUE=1`` (separate, default off) enables SGLang's metadata glue graph for the draft runner when
its attention backend is FlashAttention with device-only metadata (see ``_maybe_glue``).
``MUSE_VR_STEP_GRAPHS_SIM_FOLD=1`` (opt-in): the synthetic-acceptance forcing statement
(eager, in ``_accept_block`` or right after it) takes two launches instead of about ten, with the same forced length
from the same host draw and bit-identical results, and hands back the new sequence lengths (see ``_sim_fold``;
``MUSE_VR_STEP_GRAPHS_SIM_FOLD_SHADOW=1``, debug, compares every forcing with the statement's own operations). With a
given target_predict (the greedy verify, which passes next_token_logits=None) it is one launch and reads no logits.
``MUSE_VR_STEP_GRAPHS_ACCEPT_TRITON=1`` (opt-in): the worker's selector-path accept takes SGLang's
``accept_sampling_triton`` on CUDA inputs, as SGLang's own dispatcher of that function does (one gather launch for the
bonus instead of six; bit-identical; see ``ENV_ACCEPT_TRITON``; ``_SHADOW=1``, debug, compares it with the stock one).

Engagement requires the sha256 of the SGLang sources whose call order and local names the component relies on.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import gc
import inspect
import logging
import operator
import os
import sys
import threading
import time
import weakref

import torch

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/jit/utils/__init__.py": ("load_jit",),
    "srt/layers/logits_processor.py": ("LogitsProcessor",),
    "srt/managers/overlap_utils.py": ("FutureMap",),
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
    "srt/model_executor/runner/metadata_glue_graph.py": ("MetadataGlueGraph",),
    "srt/model_executor/runner/decode_cuda_graph_runner.py": ("DecodeCudaGraphRunner",),
    "kernels/ops/speculative/dspark/dispatch.py": ("inputs_on_cuda",),
}


try:  # the sim fold's kernels (module globals: Triton resolves a kernel's names in its module's globals)
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001
    triton = tl = None

logger = logging.getLogger(__name__)

NAME = "step_graphs"
# the package holding the other components (muse_vr_plugin in the plugin): the module-name lookups below use it
_PKG = __package__ or "muse_vr_plugin"
PINS = {
    "sglang/srt/speculative/dflash_worker_v2.py": (
        "7d462f5f175828bcc3b4d47fcdfa33cb6af4b14669ef016eeb79b1a5191eec8c",
        "8f82de8d666f81f3c1aa43dc651006f4182ff58593fdfe4eea28c5ee2a14bfc0",
    ),
    "sglang/srt/managers/scheduler.py": (
        "c1edcfe6f85e49bf85faf1f79129f4452445585de17cf1cdcc821de5008b1491",
        "fc96ed4b680fdf55621a5397f844cfd4cbd30213f44eec8a888b7cce75cafe60",
    ),
}
TARGET_INIT = "sglang.srt.speculative.dflash_worker_v2.DFlashWorkerV2.__init__"
ENV_WARMUP = "MUSE_VR_STEP_GRAPHS_WARMUP"
ENV_MAX_BS = "MUSE_VR_STEP_GRAPHS_MAX_BS"
ENV_SEGMENTS = "MUSE_VR_STEP_GRAPHS_SEGMENTS"  # debug: comma list of the segments to graph (default all)
SMALL_BYTES = (
    16 << 20
)  # per-step inputs up to this size are copied into private buffers; larger ones must not move

# segment -> (members, the member whose return ends it, the method's locals (or local.attribute) whose CUDA tensors
# its calls may read)
SEGMENTS = {
    "A": (
        ("launch", "gather", "argmax", "adjust", "accept"),
        "accept",
        (
            "draft_tokens",
            "candidates",
            "sampling_info",
            "draft_input",
            "prefix_lens",
            "logits_output",
            "positions",
            "verify_out_cache_loc_2d",
            "kv_loc_plan",
        ),
    ),
    "B": (
        ("publish", "write_ids", "append", "next"),
        "next",
        (
            "new_seq_lens",
            "commit_lens",
            "bonus",
            "out_tokens",
            "accept_len",
            "logits_output.hidden_states",
            "kv_loc_plan",
            "positions",
            "verify_out_cache_loc_2d",
            "on_publish",
        ),
    ),
    # before the draft forward: the draft sampler's refresh of its static sampling parameters (a few small launches)
    "P": (("stage",), "stage", ("@sampling_info",)),
    # the per-replay input staging of the two decode graph runners (``load_batch``: copies into the runner's static
    # buffers + attention metadata), for the draft forward (D) and for the target verify, which SGLang plans ahead
    # inside ``prepare_for_verify`` (V); the graph replays themselves stay with the runners
    "D": (("load_d",), "load_d", ("@1", "@1.kv_loc_plan", "@1.spec_info")),
    "V": (("load_v",), "load_v", ("@1", "@1.kv_loc_plan", "@1.spec_info")),
    # the draft block (one fused preparation kernel) and its embedding, first in the step
    "E": (
        ("prep", "embed"),
        "embed",
        (
            "@bonus_tokens",
            "@prefix_lens",
            "@req_pool_indices",
            "@req_to_token",
            "@block_ids_out",
            "@positions_out",
            "@cache_loc_out",
        ),
    ),
}
# segments whose captured work depends on how far the batch's KV reaches (page-table widths are sized on the host from
# the batch's CPU lengths): their graphs are keyed on that page count as well
WIDTH_SEGMENTS = ("D", "V")
WIDTH_KEEP = 4  # page-span variants kept per shape
MEMBER_SEGMENT = {m: s for s, (ms, _, _) in SEGMENTS.items() for m in ms}
# segments some of whose inputs are the opening call's arguments (``@...`` paths)
ARG_SEGMENTS = frozenset(
    s for s, (_, _, ins) in SEGMENTS.items() if any(p.startswith("@") for p in ins)
)
# arguments a segment's calls WRITE (buffers the method passes in for the results): never copied into private buffers
# (a replay would write the copy and leave the method's buffer as it was); they are read in place at their current
# address and guarded, and when one moves (a buffer that grows with the batch) the shape is captured again there.
# Inputs the warm-up steps see written by the segment's own calls are treated the same way (``_note_written``).
OUTPUT_PATHS = {"E": (("@block_ids_out",), ("@positions_out",), ("@cache_loc_out",))}
# arguments that are persistent pools the method passes whole (allocated once at startup, written in place by the
# scheduler between steps): read in place and guarded like the worker's attributes, never copied (a per-step copy of
# a whole token table costs more than the segment). The guard can only fire if the pool itself is re-made, which
# every rank does in the same step.
PERSISTENT_PATHS = {
    "E": (("@req_to_token",),),
    # the target verify graph's output rows (a static runner buffer per batch size): read in place, so
    # the context projection launched in segment A (ctx_proj_overlap, keyed by this tensor) is found
    # by its consumer in segment B
    "B": (("logits_output", "hidden_states"),),
    "A": (("logits_output", "hidden_states"),),
}
# module-level hand-offs a segment's calls leave for a later segment of the same step: ctx_proj_overlap.launch (in A)
# stores the side-stream projection in the module's ``_pending`` for take() (in B). A replay does not run the calls, so
# the entries captured with the graph are put back after each replay (their tensors are the graph's own; their events
# are recorded again on the step stream).
HANDOFFS = {"A": ((_PKG + ".ctx_proj_overlap", "_pending"),)}
# staged tensors above this size are copied one by one (a plain device-to-device copy) instead of in the grouped
# multi-tensor copy, which gives each large chunk to one thread block and is slow on them (measured; tunable)
COPY_SPLIT_BYTES = 32 << 10
ENV_COPY_SPLIT = (
    "MUSE_VR_STEP_GRAPHS_COPY_SPLIT_BYTES"  # tunable (default COPY_SPLIT_BYTES)
)
# opt-in: a segment's staged copies in ONE batched byte-copy launch (csrc/batch_copy.cuh) instead of one launch per
# word-width / dtype group plus one per large tensor; pairs it cannot take (non-contiguous) keep the grouped path
ENV_BATCH_COPY = "MUSE_VR_STEP_GRAPHS_BATCH_COPY"
_BC_CUH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "csrc", "batch_copy.cuh"
)
BC_MAX = 64
# opt-in: fewer staging copies per step. (1) A staged input whose source tensor (same address, layout and version) was
# already copied into a private buffer earlier in the same step shares that buffer: the copy runs once per step, not
# once per segment. (2) Inputs that are views of the worker's own buffers (re-made only when a larger batch arrives,
# the same step on every rank), or results of an earlier segment's graph in this step, are read in place (guarded);
# a segment reading such results runs from its graph only after that segment ran from its graph in the same step.
# (3) D and V keep one graph per step parity (their staging then reads E's per-variant result in place). (4) Segment
# A returns the new sequence lengths its caller computes right after it when A returns none (the same expression, now
# inside A's graph); never while acceptance is simulated (the caller then recomputes them after the forcing).
ENV_SHARED = "MUSE_VR_STEP_GRAPHS_SHARED"
# opt-in: a PURE segment (its graph writes only outputs that are a function of its staged inputs; here P, the draft
# sampler's parameter refresh) whose replay would read the very tensors its graph last read (same objects, same
# versions, same shape key, and no eager / capture / debug run of it since) is not replayed: its outputs already hold
# those values. The sampling parameters live as long as the batch, so most decode steps skip P.
ENV_PURE = "MUSE_VR_STEP_GRAPHS_PURE"
PURE_SEGMENTS = ("P",)
WORKER_ATTRS = ("_selector_sample",)

_tls = threading.local()
_state: dict = {
    "installed": False,
    "driver_code": None,
    "entries": {},
    "warm": {},
    "observe": {},
    "failed": set(),
    "written": {},
    "graveyard": [],
}

# debug (MUSE_VR_STEP_GRAPHS_PROF=1): host time of this component's own replay path, per phase and segment, logged as
# mean microseconds per eligible step every PROF_EVERY steps (development instrument; off by default)
PROF_EVERY = 2048
_PROF: dict = {"on": None, "ns": {}, "steps": 0}


def _prof_on() -> bool:
    on = _PROF["on"]
    if on is None:
        on = _PROF["on"] = bed.flag(NAME + "_prof")
    return on


_PROF_ON = False  # set once per process at install (hot paths read this flag, not the function)


def _padd(key: str, t0: int) -> int:
    t = time.perf_counter_ns()
    ns = _PROF["ns"]
    ns[key] = ns.get(key, 0) + t - t0
    return t


def _prof_step() -> None:
    _PROF["steps"] += 1
    if _PROF["steps"] % PROF_EVERY == 0:
        n = _PROF["steps"]
        txt = " ".join(f"{k}={v / n / 1e3:.1f}" for k, v in sorted(_PROF["ns"].items()))
        logger.info("muse_vr_plugin: %s prof us/step over %d steps: %s", NAME, n, txt)


def enabled() -> bool:
    return bed.enabled(NAME)


def pins_match() -> tuple[bool, str]:
    for rel, want in PINS.items():
        if not bed.pinned(rel, want):
            return False, f"{rel} differs from the pinned revision"
    return True, "ok"


def capturing() -> bool:
    """True while this component captures a segment (other components then run under capture as they do eagerly)."""
    return getattr(_tls, "cap", None) is not None


# ------------------------------------------------------------------------------------------------- tensors of locals
def _leaves(obj, path, out, depth=0):
    """(path, tensor) for every tensor in obj: tuples / lists / dicts / partial arguments, and one object level."""
    if isinstance(obj, torch.Tensor):
        if obj.is_cuda:  # host tensors are host values: the segments' device work must not depend on them
            out.append((path, obj))
    elif obj is None or isinstance(
        obj, (int, float, bool, str, bytes, torch.dtype, torch.device)
    ):
        return
    elif isinstance(obj, (tuple, list)):
        for i, v in enumerate(obj):
            _leaves(v, path + (i,), out, depth)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str):
                _leaves(v, path + (k,), out, depth)
    elif isinstance(obj, functools.partial):
        _leaves(list(obj.args), path + ("args",), out, depth)
    elif depth < 1 and not isinstance(obj, torch.nn.Module):
        d = getattr(obj, "__dict__", None)
        if d is None and dataclasses.is_dataclass(obj):
            d = {f.name: getattr(obj, f.name, None) for f in dataclasses.fields(obj)}
        if isinstance(d, dict):
            for k, v in list(d.items()):
                if isinstance(k, str) and not k.startswith("__"):
                    _leaves(v, path + (k,), out, depth + 1)


def _sig(t: torch.Tensor):
    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, t.device)


_ARGNAMES = tuple(f"@{i}" for i in range(16))


class _Loc:
    """Read-only view: the opening call's arguments (``@<index>`` / ``@<keyword>``) over the method's locals, without
    copying the locals; the frame's locals dict (rebuilt on every access) is read only if a non-argument name is asked."""

    __slots__ = ("fr", "fl", "a")

    def __init__(self, fr, a):
        self.fr, self.fl, self.a = fr, None, a

    @property
    def f(self):
        if self.fl is None:
            self.fl = self.fr.f_locals
        return self.fl

    def get(self, k, d=None):
        return self.a[k] if k in self.a else self.f.get(k, d)

    def __contains__(self, k):
        return k in self.a or k in self.f

    def __getitem__(self, k):
        return self.a[k] if k in self.a else self.f[k]


def _loc(step):
    """The method's locals, plus the opening call's arguments as ``@<index>`` / ``@<keyword>`` (inputs that reach a
    segment only as arguments of its first call, e.g. the sampling parameters handed to the draft sampler)."""
    return _Loc(step.frame, step.argloc) if step.argloc else step.frame.f_locals


def _externals(seg: str, loc, worker) -> list:
    out = []
    for spec in SEGMENTS[seg][2]:
        name, _, attr = spec.partition(".")
        if name not in loc:
            continue
        if attr:
            _leaves(getattr(loc[name], attr, None), (name, attr), out)
        else:
            _leaves(loc[name], (name,), out)
    for name in WORKER_ATTRS:
        _leaves(getattr(worker, name, None), ("self", name), out)
    seen, uniq = set(), []
    for (
        path,
        t,
    ) in out:  # one entry per tensor object (locals alias each other, e.g. candidates / draft_tokens)
        if id(t) not in seen:
            seen.add(id(t))
            uniq.append((path, t))
    return uniq


def _driver_frame():
    code = _state["driver_code"]
    f = sys._getframe(1)
    for _ in range(32):
        if f is None:
            return None
        if f.f_code is code:
            return f
        f = f.f_back
    return None


# ------------------------------------------------------------------------------------------------- capture plumbing
def _capture_node_count(stream) -> int | None:
    try:
        from cuda.bindings import runtime as rt

        res = rt.cudaStreamGetCaptureInfo(stream.cuda_stream)
        if res[0] != rt.cudaError_t.cudaSuccess:
            return None
        graph = next((x for x in res[1:] if type(x).__name__ == "cudaGraph_t"), None)
        if graph is None:
            return None
        r2 = rt.cudaGraphGetNodes(graph, 0)
        return int(r2[-1]) if r2[0] == rt.cudaError_t.cudaSuccess else None
    except Exception:  # noqa: BLE001
        return None


def _capture_host_copies(stream) -> int | None:
    """Copy nodes of the capture graph that read or write host memory. A captured copy replays to / from the host address
    it had at capture (often a temporary pinned buffer): per-step host values would be frozen, read from freed memory or
    written to a buffer nobody reads any more, so such a capture is refused. None when the graph cannot be inspected."""
    try:
        from cuda.bindings import runtime as rt

        res = rt.cudaStreamGetCaptureInfo(stream.cuda_stream)
        if res[0] != rt.cudaError_t.cudaSuccess:
            return None
        graph = next((x for x in res[1:] if type(x).__name__ == "cudaGraph_t"), None)
        if graph is None:
            return None
        r = rt.cudaGraphGetNodes(graph, 0)
        if r[0] != rt.cudaError_t.cudaSuccess:
            return None
        r = rt.cudaGraphGetNodes(graph, int(r[-1]))
        if r[0] != rt.cudaError_t.cudaSuccess:
            return None
        n = 0
        for node in r[1]:
            t = rt.cudaGraphNodeGetType(node)
            if (
                t[0] != rt.cudaError_t.cudaSuccess
                or t[1] != rt.cudaGraphNodeType.cudaGraphNodeTypeMemcpy
            ):
                continue
            m = rt.cudaGraphMemcpyNodeGetParams(node)
            if m[0] != rt.cudaError_t.cudaSuccess:
                return None
            kind, src, dst = m[1].kind, int(m[1].srcPtr.ptr), int(m[1].dstPtr.ptr)
            if kind in (
                rt.cudaMemcpyKind.cudaMemcpyHostToDevice,
                rt.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                rt.cudaMemcpyKind.cudaMemcpyHostToHost,
            ):
                n += 1
            elif kind == rt.cudaMemcpyKind.cudaMemcpyDefault:
                for ptr in (src, dst):
                    a = rt.cudaPointerGetAttributes(ptr)
                    if (
                        a[0] != rt.cudaError_t.cudaSuccess
                        or a[1].type != rt.cudaMemoryType.cudaMemoryTypeDevice
                    ):
                        n += 1
                        break
        return n
    except Exception:  # noqa: BLE001
        return None


class _Capture:
    def __init__(self, stream, step):
        self.stream, self.step = stream, step
        self.joined = []
        self.events = set()
        self.event_objs = []
        self.nodes_after = None
        self.error = None

    def member_start(self, member):
        if self.nodes_after is not None:
            n = _capture_node_count(self.stream)
            if n is not None and n != self.nodes_after:
                self.error = (
                    self.error or f"operations recorded between calls (before {member})"
                )

    def member_end(self):
        self.nodes_after = _capture_node_count(self.stream)

    def join_all(self):
        for s in self.joined:
            ev = torch.cuda.Event()
            _ORIG["Event.record"](ev, s)
            _ORIG["Event.wait"](ev, self.stream)


_ORIG: dict = {}


_WRAP: dict = {}


def _remove_capture_patches() -> None:
    """Put the originals back once a capturing step ends (replaying steps never go through the wrappers); left in
    place if anything else wrapped the same attributes after them."""
    if not _WRAP.get("on"):
        return
    E = torch.cuda.Event
    if (
        E.record is not _WRAP["record"]
        or E.wait is not _WRAP["wait"]
        or torch.cuda.is_current_stream_capturing is not _WRAP["is_capturing"]
    ):
        return
    E.record, E.wait = _ORIG["Event.record"], _ORIG["Event.wait"]
    torch.cuda.is_current_stream_capturing = _ORIG["is_capturing"]
    _WRAP["on"] = False


def _install_capture_patches():
    """torch.cuda.Event record / wait (every stream-level wait and record goes through them) and the capture query,
    tracking this component's captures (pass-through otherwise). Installed when a capture begins, removed when that
    step ends."""
    if _WRAP.get("on"):
        return
    E = torch.cuda.Event
    if _WRAP:
        E.record, E.wait = _WRAP["record"], _WRAP["wait"]
        torch.cuda.is_current_stream_capturing = _WRAP["is_capturing"]
        _WRAP["on"] = True
        return
    _ORIG.update(
        {
            "Event.record": E.record,
            "Event.wait": E.wait,
            "is_capturing": torch.cuda.is_current_stream_capturing,
        }
    )

    def _in_capture(cap, s):
        return s.cuda_stream == cap.stream.cuda_stream or any(
            s.cuda_stream == j.cuda_stream for j in cap.joined
        )

    def record(self, stream=None):
        cap = getattr(_tls, "cap", None)
        if cap is not None:
            st = stream if stream is not None else torch.cuda.current_stream()
            if _in_capture(cap, st):
                if id(self) not in cap.events:
                    cap.event_objs.append(self)
                cap.events.add(id(self))
                cap.step.captured_events.add(id(self))
        return _ORIG["Event.record"](self, stream)

    def wait(self, stream=None):
        cap = getattr(_tls, "cap", None)
        step = getattr(_tls, "step", None)
        if cap is None and step is not None and id(self) in step.captured_events:
            # an event recorded by a segment captured earlier in this step: the segment's replay (on the step's stream,
            # side streams joined back) already ran, so waiting on the step's stream orders the same way
            st = stream if stream is not None else torch.cuda.current_stream()
            if step.stream is not None and st.cuda_stream != step.stream.cuda_stream:
                ev = torch.cuda.Event()
                _ORIG["Event.record"](ev, step.stream)
                return _ORIG["Event.wait"](ev, st)
            return None
        if cap is not None:
            st = stream if stream is not None else torch.cuda.current_stream()
            if id(self) not in cap.events:
                if id(self) not in cap.step.captured_events:
                    if cap.error is None:
                        import traceback

                        where = " < ".join(
                            f"{os.path.basename(f.filename)}:{f.name}:{f.lineno}"
                            for f in reversed(traceback.extract_stack(limit=7)[:-1])
                        )
                        cap.error = (
                            f"wait on an event recorded outside the capture ({where})"
                        )
                return None  # recorded by an earlier segment's capture this step: already ordered by this stream
            if not _in_capture(cap, st):
                cap.joined.append(st)
        return _ORIG["Event.wait"](self, stream)

    def is_capturing():
        if getattr(_tls, "cap", None) is not None:
            mod = sys._getframe(1).f_globals.get("__name__", "")
            if mod.startswith(__package__ or "muse_vr_plugin"):
                return False  # this component's capture: plugin components behave as they do eagerly
        return _ORIG["is_capturing"]()

    E.record, E.wait = record, wait
    torch.cuda.is_current_stream_capturing = is_capturing
    _WRAP.update(
        {"record": record, "wait": wait, "is_capturing": is_capturing, "on": True}
    )


# ------------------------------------------------------------------------------------------------- per step
class _Step:
    def __init__(self, worker, parity):
        self.worker, self.parity = worker, parity
        self.stream = torch.cuda.current_stream()
        self.frame = None
        self.key = None
        self.runs = {}
        self.captured_events = set()
        self.handoffs = []  # hand-off entries put back after replays this step (dropped at the step's end if unconsumed)
        self.restore = []  # (object, attribute, value) set back when the step ends
        self.argloc = {}
        self.vstep = 0
        self.e_open = (
            False  # segment E: its prep ran this step, its embedding call is next
        )
        self.batch = None
        self.base_key = None
        # MUSE_VR_STEP_GRAPHS_SHARED: id(private buffer) -> (source tensor, its version) copied into it this step (the
        # reference keeps the source's memory from being reused within the step); (address, layout) of a source ->
        # (buffer, version) for captures later in the step; addresses of results of segments that ran from graphs
        self.fresh = {}
        self.shared = {}
        self.result_ptrs = {}  # address -> segment
        self.graphed = (
            set()
        )  # segments that ran from their graph (replay or capture) this step
        self.hoist_extra = None  # (buffer, source) pairs of later segments staged with segment E's copy
        self.pure_srcs = (
            None  # a pure segment's staged sources, as read for this step's replay
        )
        self.spec0 = None  # the batch's draft input when the step began (shared staging's hoist list)
        self.reentry_pairs = []  # debug: (field, runner rows, the batch's own tensor) re-pointed this step


class _Run:
    def __init__(self, step, seg, mode, entry=None):
        self.step, self.seg, self.mode, self.entry = step, seg, mode, entry
        self.cursor = 0
        self.obj_delta = []  # (weak owner, attribute, value) set by the captured calls
        self.calls = []
        self.redo = []
        self.subst = {}  # id(original tensor) -> private buffer (capture)


def _widthed(seg) -> bool:
    return seg in WIDTH_SEGMENTS and _state.get("translating", {}).get(seg, True)


def _wkey(step, seg):
    """Warm-up / failure / promotion state: per shape, shared by the page-span variants of D and V (their inputs are
    classified once; a new span is captured at its first occurrence)."""
    return (id(step.worker), seg, step.key[:-1] if _widthed(seg) else step.key)


# segments whose outputs are not double-buffered by the method (they fill the runners' static buffers): one graph per
# shape, not one per step parity
SINGLE_VARIANT = ("D", "V")


def _single(seg) -> bool:
    """One graph per shape (not per step parity); with shared staging D and V get one per parity too (their staging
    reads segment E's result, which alternates between E's two variants)."""
    return seg in SINGLE_VARIANT and not _shared_on()


ENV_SHARED_CHECK = (
    "MUSE_VR_STEP_GRAPHS_SHARED_CHECK"  # debug (host syncs): see _check_skip
)


def _check_on() -> bool:
    on = _state.get("check_on")
    if on is None:
        on = _state["check_on"] = bed.env_int(ENV_SHARED_CHECK, 0) == 1
    return on


def _after_unreplayed(step) -> str | None:
    """Debug tag: an earlier segment of this step did not run from its graph (VERIFY's eager check, or eager)."""
    if step is None:
        return None
    modes = {r.mode for r in step.runs.values()}
    return (
        "after-verify"
        if "verify" in modes
        else ("after-eager" if "eager" in modes else None)
    )


def _check_skip(entry, path, t, buf, why, step=None) -> None:
    """Debug (MUSE_VR_STEP_GRAPHS_SHARED_CHECK=1): a staged copy is being skipped because ``buf`` already holds this
    source's values (copied earlier in the step, an unchanged sampling parameter, or a per-shape constant): compare the
    live source with the buffer bit for bit and count the outcome (one host sync per skip)."""
    try:
        ok = _same(t, buf)
    except Exception as e:  # noqa: BLE001 - debug only
        bed.count(NAME, f"shared check error:{type(e).__name__}")
        return
    seg = entry.get("tally", "?:?").split(":")[1]
    if ok:
        bed.count(NAME, f"shared check:{why}:ok")
    else:
        bed.count(NAME, f"shared check MISMATCH:{seg}:{why}:{'/'.join(map(str, path))}")
    tag = _after_unreplayed(step)
    if tag:  # the same outcome, counted again for steps in which an earlier segment did not replay
        bed.count(NAME, f"shared check:{tag}:{'ok' if ok else 'MISMATCH'}")


def _check_pure(step, seg, entry) -> None:
    """Debug: a pure segment's replay is due to be skipped; replay it anyway and compare the outputs it rewrites
    (EXTRA_OUTPUTS) with what they held: equal means the skip would have left exactly these values."""
    loc = _loc(step)
    outs = [(p, _fetch(p, loc, step.worker)) for p in EXTRA_OUTPUTS.get(seg, ())]
    outs = [(p, t, t.clone()) for p, t in outs if isinstance(t, torch.Tensor)]
    entry["graph"].replay()
    bad = [p for p, t, old in outs if not _same(t, old)]
    if bad:
        bed.count(NAME, f"pure check MISMATCH:{seg}:{'/'.join(map(str, bad[0]))}")
    else:
        bed.count(NAME, f"pure check:{seg}:ok")
    tag = _after_unreplayed(step)
    if tag:
        bed.count(NAME, f"pure check:{tag}:{'MISMATCH' if bad else 'ok'}")


# opt-in: while segment A is captured, its first call (the side-stream context projection's launch) is recorded after the
# NEXT call of the segment (the verify logits gather) instead of before it. The launch's work still depends only on what
# the step had issued at its own position (it is issued from a fork stream that waits on an event recorded there), so
# the graph has the same dependencies; only the order in which the replay submits the two independent branches changes.
# Eagerly the gather is queued first on the step's stream and starts first; a replay submits nodes in capture order, so
# without this the projection's GEMM is submitted first and the gather starts later than it does eagerly (measured in a
# profiled A/B; it is on the segment's critical path). Calls, results and the hand-off are
# unchanged (the launch returns True as eagerly; its result is taken in segment B).
ENV_LAUNCH_LATE = "MUSE_VR_STEP_GRAPHS_LAUNCH_LATE"
LATE_MEMBERS = {"A": "launch"}


def _late_on() -> bool:
    on = _state.get("late_on")
    if on is None:
        on = _state["late_on"] = bed.env_int(ENV_LAUNCH_LATE, 0) == 1
    return on


def _late_stream(device):
    key = ("late_stream", torch.device(device).index)
    s = _state.get(key)
    if s is None:
        s = _state[key] = torch.cuda.Stream(device=device)
    return s


def _run_late(run) -> None:
    """Issue the deferred launch now (inside the current call's window), from a fork stream that waits on the event
    recorded at the launch's own position."""
    original, args, kwargs, ev0, dev = run.late
    run.late = None
    fork = _late_stream(dev)
    ev0.wait(fork)  # the fork stream joins the capture at the launch's position
    with torch.cuda.stream(fork):
        res = original(*args, **kwargs)
    if res is not True:
        raise RuntimeError(f"{NAME}: deferred launch returned {res!r}")
    bed.count(NAME, f"late:{run.seg}")


# opt-in: after the target runner's planned staging (segment V), the verify batch's input ids and positions are pointed at
# the runner's static rows they were just copied into. SGLang's pre-planned re-entry (inside the forward) copies those
# two fields into the same rows again; with the batch pointing at the rows themselves both copies are the same tensor,
# which PyTorch's copy skips (no launch). Values are unchanged; nothing else in the decode step reads these two fields.
ENV_REENTRY = "MUSE_VR_STEP_GRAPHS_REENTRY"
REENTRY_FIELDS = ("input_ids", "positions")


def _reentry_on() -> bool:
    on = _state.get("reentry_on")
    if on is None:
        on = _state["reentry_on"] = bed.env_int(ENV_REENTRY, 0) == 1
    return on


def _alias_planned(args, kwargs) -> None:
    try:
        runner = args[0]
        fb = args[1] if len(args) > 1 else kwargs.get("forward_batch")
        bufs, n = (
            getattr(runner, "buffers", None),
            getattr(runner, "raw_num_token", None),
        )
        if fb is None or bufs is None or not isinstance(n, int):
            return
        for name in REENTRY_FIELDS:
            src, buf = getattr(fb, name, None), getattr(bufs, name, None)
            if (
                not isinstance(src, torch.Tensor)
                or not isinstance(buf, torch.Tensor)
                or buf.dim() != 1
                or src.dim() != 1
                or src.shape[0] != n
                or buf.shape[0] < n
                or src.dtype != buf.dtype
            ):
                continue
            setattr(fb, name, buf[:n])
            bed.count(NAME, f"reentry:{name}")
            step = getattr(_tls, "step", None)
            if step is not None and _check_on():
                step.reentry_pairs.append((name, buf[:n], src))
    except Exception as e:  # noqa: BLE001 - the batch keeps its own tensors (the re-entry then copies as stock)
        bed.count(NAME, f"reentry error:{type(e).__name__}")


def _check_reentry(step) -> None:
    """Debug (SHARED_CHECK with REENTRY): after the verify forward, the runner rows the verify batch was pointed at must
    still equal the batch's own tensors (the forward reads them; it must not have written them)."""
    pairs, step.reentry_pairs = step.reentry_pairs, []
    for name, rows, src in pairs:
        try:
            ok = _same(rows, src)
        except Exception as e:  # noqa: BLE001 - debug only
            bed.count(NAME, f"reentry check error:{type(e).__name__}")
            continue
        bed.count(NAME, f"reentry check:{name}:{'ok' if ok else 'MISMATCH'}")


def _pure_on() -> bool:
    on = _state.get("pure_on")
    if on is None:
        on = _state["pure_on"] = bed.env_int(ENV_PURE, 0) == 1
    return on


def _shared_on() -> bool:
    on = _state.get("shared_on")
    if on is None:
        on = _state["shared_on"] = bed.env_int(ENV_SHARED, 0) == 1
    return on


def _ekey(step, seg):
    return (id(step.worker), seg, step.key, 0 if _single(seg) else step.parity)


def _rptrs(entry) -> frozenset:
    """Addresses of the tensors a graph returns (its own memory; the same on every replay of the graph)."""
    r = entry.get("rptrs")
    if r is None:
        a = []
        _leaves(entry["results"], (), a)
        r = entry["rptrs"] = frozenset(t.data_ptr() for _, t in a if t.numel())
    return r


def _skey(t):
    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype)


# per-shape constants among the staged inputs (SGLang's token counts of the batch: the batch size times the block
# size): with shared staging, an input of this name whose value was the same through the warm-up and at the capture
# is copied at the capture only (VERIFY compares every check against the eager calls, so a changed value would show)
CONST_NAMES = ("num_token_non_padded", "global_num_token_non_padded")
CONST = 2  # input mode: staged once (truthy, like a staged input, for the debug paths)
HOIST_ROOTS = (
    "spec_info",
    "sampling_info",
)  # objects of the batch whose tensors are walked for the hoist list
HOIST_KEEP_ROOTS = (
    "sampling_info",
)  # hoisted roots skipped while unchanged (as SKIP_UNCHANGED_ROOTS)


def _batch_leaves(batch, spec0=None) -> dict:
    """skey -> path under the batch, for the batch's own tensors and those of its draft-input and sampling objects.
    ``spec0`` is the batch's draft input as it was when the step began (the method later points the batch at the verify
    input), walked under the same path, since that is where segment E's copy finds it."""
    out = []
    _leaves(batch, ("batch",), out)
    for name in HOIST_ROOTS:
        obj = (
            spec0
            if name == "spec_info" and spec0 is not None
            else getattr(batch, name, None)
        )
        _leaves(obj, ("batch", name), out)
    return {_skey(t): path for path, t in out if t.numel()}


def _hoist_list(bl, staged, bufs) -> tuple:
    """A later segment's staged inputs that are tensors of the step's batch (same address and layout at capture; ``bl``
    from _batch_leaves before any private buffer replaced them): (path under the batch, private buffer). Segment E's
    copy fills them at the start of the step (one launch for the step's staging); the segment's own staging then finds
    them fresh and skips them."""
    out = []
    for path, t in staged.items():
        bp = bl.get(_skey(t)) if t.numel() else None
        if bp is not None:
            out.append((bp[1:], bufs[path]))
    return tuple(out)


def _hoist_pairs(step) -> list:
    """(buffer, source) for the hoist lists of the entries registered for this step's batch key and parity."""
    keys = _state.get("hoist_idx", {}).get(
        (id(step.worker), step.base_key, step.parity)
    )
    if not keys or step.batch is None:
        return []
    out = []
    for ek in keys:
        e2 = _state["entries"].get(ek)
        if e2 is None:
            continue
        for bpath, buf in e2.get("hoist", ()):
            t = (
                _walk(step.spec0, bpath[1:])
                if bpath[0] == "spec_info" and step.spec0 is not None
                else _walk(step.batch, bpath)
            )
            if (
                isinstance(t, torch.Tensor)
                and t.shape == buf.shape
                and t.dtype == buf.dtype
                and t.stride() == buf.stride()
            ):
                out.append((buf, t, bpath[0]))
    return out


def _note_fresh(step, buf, t) -> None:
    """``buf`` now holds ``t``'s values (copied in this step)."""
    v = t._version
    step.fresh[id(buf)] = (t, v)
    step.shared[_skey(t)] = (buf, v)
    buf._muse_vr_keep = (
        t,
        v,
    )  # cross-step skip of unchanged sources (sampling parameters), per buffer


def _worker_storages(worker) -> set:
    """Storage addresses of the worker's own CUDA buffers (attributes, and lists / tuples of them)."""
    out = set()
    for v in vars(worker).values():
        for t in v if isinstance(v, (list, tuple)) else (v,):
            if isinstance(t, torch.Tensor) and t.is_cuda and t.numel():
                out.add(t.untyped_storage().data_ptr())
    return out


def _open(step, seg):
    if seg not in _segments_on() or _wkey(step, seg) in _state["failed"]:
        return _Run(step, seg, "eager")
    if seg == "A" and _forcing_in_call(step.worker) and not _FORCING["suppressed"]:
        # SGLang main applies the synthetic-acceptance forcing inside _accept_block, the end of this segment; it is
        # never captured: on this path the segment runs eagerly (see _forcing_after)
        bed.count(NAME, "eager:A:forcing in call")
        return _Run(step, seg, "eager")
    if seg == "B":
        a = step.runs.get("A")
        if a is None or a.mode not in ("replay", "done"):
            # B's graph may read A's captured tensors (through calls of other components, not only the method's
            # locals): it runs only after A ran from its graph of the same variant in this step
            return _Run(step, seg, "eager")
    if step.reentry_pairs and seg == "A":
        _check_reentry(
            step
        )  # debug: the verify forward has run; its rows must still hold the batch's values
    entry = _state["entries"].get(_ekey(step, seg))
    if entry is not None:  # fast path: fetch only the captured inputs
        needs = entry.get("needs")
        if needs and not needs <= step.graphed:
            # the graph reads results of these segments in place: they did not run from their graphs this step
            bed.count(NAME, f"eager:{seg}:producer not graphed")
            return _Run(step, seg, "eager")
        prof = _PROF_ON
        if seg == "E" and _shared_on():
            step.hoist_extra = _hoist_pairs(step)
        pure = seg in PURE_SEGMENTS and _pure_on()
        step.pure_srcs = [] if pure else None
        why = _stage(entry, _loc(step), step.worker, seg if prof else None, step)
        if why is None:
            if bed.flag(NAME + "_verify") and _verify_pick(step, seg):
                return _verify_open(step, seg, entry)
            t0 = time.perf_counter_ns() if prof else 0
            skip = False
            if pure:
                # a pure segment whose graph last ran from these very input tensors, unwritten since: its outputs hold
                # what this replay would write (strong references keep the identities from being reused)
                # (either variant: both write the same outputs, so the key leaves the step parity out)
                sig = ((id(step.worker), seg, step.key),) + tuple(
                    (id(t), t._version) for t in step.pure_srcs
                )
                last = _state.setdefault("pure_last", {}).get(seg)
                skip = last is not None and last[0] == sig
                if not skip:
                    _state["pure_last"][seg] = (sig, list(step.pure_srcs))
            if skip and _check_on():
                _check_pure(
                    step, seg, entry
                )  # debug: replays anyway and compares the outputs it would rewrite
            elif skip:
                _tally(f"skip:{seg}:bs={step.key[0]}")
            else:
                entry["graph"].replay()
            if prof:
                t0 = _padd("replay:" + seg, t0)
            if _shared_on():
                step.graphed.add(seg)
                step.result_ptrs.update(dict.fromkeys(_rptrs(entry), seg))
            if entry["events"]:
                _rerecord(entry["events"], step.stream)
                # like events of a segment captured in this step: ordered by the step stream (side streams joined)
                step.captured_events.update(
                    id(e) for e in (r() for r in entry["events"]) if e is not None
                )
            for d, snap in entry.get("handoff", ()):
                d.update(snap)
                step.handoffs.append((d, snap))
            for k, v in entry["state_delta"].items():
                setattr(step.worker, k, v)
            for ref, k, v in entry["obj_delta"]:
                o = ref()
                if o is not None:
                    setattr(o, k, v)
            _tally(entry["tally"])
            if prof:
                _padd("post:" + seg, t0)
            if _vlog_left() > 0 and bed.env_int(ENV_VERIFY_RANGE, 0) > 0:
                try:
                    _range_check(step, seg, entry)
                except Exception as e:  # noqa: BLE001 - debug only: never change the step
                    bed.count(NAME, f"debug log error:{type(e).__name__}")
            return _Run(step, seg, "replay", entry)
        why, moved = why
        bed.count(NAME, f"guard:{seg}:{why}")
        for s2 in (
            ("A", "B") if seg == "A" else (seg,)
        ):  # B's graphs were captured against A's: they go with A's
            for p in (0, 1):
                _bury(_state["entries"].pop((id(step.worker), s2, step.key, p), None))
            w2 = _state["warm"].get(_wkey(step, s2))
            if moved is not None and w2 is not None:
                # a small result buffer of the method moved (grown for a larger batch): capture again at the next
                # occurrence, at its new address (the warm-up classification is kept), instead of a full warm-up
                w2["n"] = bed.env_int(ENV_WARMUP, 3)
            else:
                _state["warm"].pop(_wkey(step, s2), None)
        return _Run(step, seg, "eager")
    ext = _externals(seg, _loc(step), step.worker)
    if bed.flag(NAME + "_observe"):
        ob = _state["observe"].setdefault(_wkey(step, seg), {})
        for path, t in ext:
            ob.setdefault(path, set()).add(_sig(t))
        bed.count(NAME, f"observe:{seg}")
        return _Run(step, seg, "observe")
    w = _state["warm"].setdefault(_wkey(step, seg), {"n": 0, "sigs": ({}, {})})
    # per variant: inputs that alternate between two buffers are stable per variant (one variant: D, V)
    sigs = w["sigs"][0 if _single(seg) else step.parity]
    for path, t in ext:
        sigs.setdefault(path, set()).add(_sig(t))
    w["n"] += 1
    if (
        w["n"] > bed.env_int(ENV_WARMUP, 3)
        and not _ORIG.get("is_capturing", torch.cuda.is_current_stream_capturing)()
    ):
        return _begin_capture(step, seg, ext, w)
    run = _Run(step, seg, "eager")
    # warm-up only: snapshots of the inputs, compared when the segment's last call returns (``_note_written``)
    run.watch = [
        (path, t, t.clone())
        for path, t in ext
        if t.is_cuda and t.numel() * t.element_size() <= SMALL_BYTES
    ]
    return run


_INT_OF = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _differs(t, old):
    """A 0-dim device bool: the bits of ``t`` differ from ``old`` (floats compared as bits, so NaN equals NaN)."""
    if t.shape != old.shape or t.dtype != old.dtype:
        return torch.ones((), dtype=torch.bool, device=t.device)
    if (
        t.is_floating_point()
        and t.is_contiguous()
        and old.is_contiguous()
        and t.element_size() in _INT_OF
    ):
        it = _INT_OF[t.element_size()]
        return torch.ne(t.view(it), old.view(it)).any()
    return torch.ne(t, old).any()


def _note_written(run) -> None:
    """Warm-up: the inputs whose contents the segment's own calls changed are outputs (never staged, see
    OUTPUT_PATHS). One host sync per warm-up occurrence."""
    watch, run.watch = run.watch, None
    if not watch:
        return
    try:
        changed = torch.stack([_differs(t, old) for _, t, old in watch]).tolist()
    except Exception:  # noqa: BLE001 - e.g. an input freed or resized by the calls: no record
        return
    paths = {path for (path, _, _), c in zip(watch, changed) if c}
    if _shared_on():
        try:
            vals = _state.setdefault("const_vals", {}).setdefault(
                _wkey(run.step, run.seg), {}
            )
            for path, t, _ in watch:
                if path[-1] in CONST_NAMES and t.numel() <= 8:
                    vals.setdefault(path, set()).add(tuple(t.reshape(-1).tolist()))
        except Exception:  # noqa: BLE001 - no record: the input stays staged every step
            pass
    if paths:
        rec = _state["written"].setdefault(_wkey(run.step, run.seg), set())
        for path in paths - rec:
            bed.count(NAME, f"written:{run.seg}:{'/'.join(map(str, path))}")
        rec |= paths


def _step1(obj, part):
    if isinstance(part, int):
        return obj[part] if isinstance(obj, (tuple, list)) and part < len(obj) else None
    if isinstance(obj, dict):
        return obj.get(part)
    if part == "args" and isinstance(obj, functools.partial):
        return obj.args
    return getattr(obj, part, None)


def _fetch(path, loc, worker):
    """The object at ``path`` (local name or 'self', then attributes / keys / indices; 'args' of a partial)."""
    obj = worker if path[0] == "self" else loc.get(path[0])
    for part in path[1:]:
        if obj is None:
            return None
        obj = _step1(obj, part)
    return obj


DEFAULT_SEGMENTS = (
    "P",
    "A",
    "B",
)  # D and V are opt-in (MUSE_VR_STEP_GRAPHS_SEGMENTS=P,D,V,A,B)


@functools.lru_cache(maxsize=None)
def _segments_on() -> frozenset:
    v = os.environ.get(ENV_SEGMENTS, "").strip()
    return (
        frozenset(x.strip() for x in v.split(",") if x.strip())
        if v
        else frozenset(DEFAULT_SEGMENTS)
    )


def _bury(entry) -> None:
    """Graphs are destroyed only between steps, never while a capture may run (destroying a graph during a capture
    invalidates it)."""
    if entry is not None:
        _state["graveyard"].append(entry)


def _weak(events) -> list:
    """Weak references to the events a capture recorded: transient ones (e.g. made by Stream.wait_stream) die with
    the capture and are skipped; long-lived ones (the scheduler's publish event) stay."""
    out = []
    for ev in events:
        try:
            out.append(weakref.ref(ev))
        except TypeError:
            out.append(lambda ev=ev: ev)
    return out


def _rerecord(events, stream) -> None:
    """Events recorded inside a capture are capture-internal (no record node in the graph); code that waits on them
    after the step (e.g. the scheduler's publish event) needs them recorded eagerly. Recording them at the segment's
    end on the step stream orders those waits after everything the segment did (its side streams are joined)."""
    for ref in events:
        ev = ref()
        if ev is not None:
            _ORIG["Event.record"](ev, stream)


def _tally(key: str) -> None:
    """bed.count, plus a ``key>=2^k`` record each time the tally reaches a power of two (the engagement file is written
    on new keys only)."""
    bed.count(NAME, key)
    n = bed._counts.get(NAME, {}).get(key, 0)
    if n >= 64 and n & (n - 1) == 0:
        bed.count(NAME, f"{key}>={n}")
    elif n % 32 == 0 and bed.flag(NAME + "_verify"):
        bed._dump()  # debug mode only: keep the engagement record's tallies (and comparisons) current


_MISS = object()


# staged inputs under these roots (the batch's sampling parameters: made with the batch, never written afterwards) are
# copied only when the source is another tensor object or its version counter moved, i.e. when the batch changed; the
# entry keeps a reference to the last source copied, so its identity cannot be reused (MUSE_VR_STEP_GRAPHS_RESTAGE=1:
# copy them every step)
SKIP_UNCHANGED_ROOTS = ("sampling_info", "@sampling_info")
ENV_RESTAGE = "MUSE_VR_STEP_GRAPHS_RESTAGE"


def _walk(obj, parts):
    for part in parts:
        if obj is None:
            return None
        obj = _step1(obj, part)
    return obj


def _getter(parts):
    """A fast reader of ``parts`` below a root object: one C-level attribute chain when every part is a name, used only
    once its first result was the very object the generic walk (keys, indices, a partial's args) returns; else the walk."""
    if parts and all(isinstance(x, str) and x != "args" for x in parts):
        g = operator.attrgetter(".".join(parts))
        state = [None]

        def get(obj):
            ok = state[0]
            if ok is None:
                w = _walk(obj, parts)
                try:
                    a = g(obj)
                except AttributeError:
                    a = _MISS
                state[0] = a is w
                return w
            if ok:
                try:
                    return g(obj)
                except AttributeError:
                    return _walk(obj, parts)
            return _walk(obj, parts)

        return get
    return lambda obj: _walk(obj, parts)


def _compile_inputs(inputs):
    """(parent readers, per input (parent index or -1, part)): each distinct parent path is read once per step."""
    parents, idx, items = [], {}, []
    for path, _, _ in inputs:
        if len(path) > 1:
            pp = path[:-1]
            j = idx.get(pp)
            if j is None:
                j = idx[pp] = len(parents)
                parents.append((pp[0], _getter(pp[1:])))
            items.append((j, path[-1]))
        else:
            items.append((-1, path[0]))
    return parents, items


def _stage(entry, loc, worker, prof_seg=None, step=None):
    """Check the guards and copy the per-step inputs into the private buffers. None when the graph may replay, else
    (reason, path of a small guarded input that moved or None)."""
    t0 = time.perf_counter_ns() if prof_seg else 0
    dst, src = [], []
    sh = step is not None and _shared_on()
    chk = sh and _check_on()
    fresh = step.fresh if sh else None
    pend = {} if sh else None  # id(buffer) -> source key, this call
    keep = entry.get("keep")
    if keep is None:
        keep = entry["keep"] = {}
        entry["skip_ok"] = (
            frozenset()
            if os.environ.get(ENV_RESTAGE) == "1"
            else frozenset(
                i
                for i, (path, mode, _) in enumerate(entry["inputs"])
                if mode and path[0] in SKIP_UNCHANGED_ROOTS
            )
        )
    skip_ok = entry["skip_ok"]
    commit = []
    plan = entry.get("plan")
    if plan is None:
        plan = entry["plan"] = _compile_inputs(entry["inputs"])
    readers, items = plan
    # inputs share parent objects (a batch, a plan): each parent path is read once
    pars = [g(worker if root == "self" else loc.get(root)) for root, g in readers]
    for i, (path, mode, ref) in enumerate(entry["inputs"]):
        j, part = items[i]
        if j >= 0:
            par = pars[j]
            t = (
                None
                if par is None
                else (
                    getattr(par, part, None)
                    if type(part) is str
                    and part != "args"
                    and not isinstance(par, dict)
                    else _step1(par, part)
                )
            )
        else:
            t = worker if part == "self" else loc.get(part)
        if not isinstance(t, torch.Tensor):
            return "input missing " + "/".join(map(str, path)), None
        if mode:  # staged: ref is the private buffer
            if sh and t.numel() == 0 and ref.numel() == 0 and t.dtype == ref.dtype:
                continue  # empty both now and at the capture: nothing to copy, and no captured kernel reads it
            if t.shape != ref.shape or t.dtype != ref.dtype:
                return "input shape " + "/".join(map(str, path)), None
            if step is not None and step.pure_srcs is not None:
                step.pure_srcs.append(t)
            if mode == CONST:
                if chk:
                    _check_skip(entry, path, t, ref, "const", step)
                continue  # one value for this shape through the warm-up: the buffer holds it from the capture
            if sh:
                f = fresh.get(id(ref))
                v = t._version
                if (
                    f is not None
                    and f[1] == v
                    and (f[0] is t or _skey(f[0]) == _skey(t))
                ):
                    if chk:
                        _check_skip(entry, path, t, ref, "fresh", step)
                    continue  # this buffer received this very source earlier in the step
                k = _skey(t)
                p0 = pend.get(id(ref))
                if p0 is not None:
                    if (
                        p0 != k
                    ):  # two inputs of this graph share a buffer but now come from different sources
                        return "shared input conflict " + "/".join(map(str, path)), None
                    continue
                if i in skip_ok:
                    kk = getattr(ref, "_muse_vr_keep", None)
                    if kk is not None and kk[0] is t and kk[1] == v:
                        if chk:
                            _check_skip(entry, path, t, ref, "keep", step)
                        _note_fresh(step, ref, t)
                        continue  # the same tensor, unwritten since its copy into this buffer
                pend[id(ref)] = k
                dst.append(ref)
                src.append(t)
                continue
            if i in skip_ok:
                k, v = keep.get(i), t._version
                if k is not None and k[0] is t and k[1] == v:
                    continue  # the same tensor, unwritten since its copy: the buffer holds its values
                commit.append((i, t, v))
            dst.append(ref)
            src.append(t)
        elif t.data_ptr() != ref:  # fixed: ref is the address the capture read
            small = path[0] != "self" and t.numel() * t.element_size() <= SMALL_BYTES
            return "fixed input moved " + "/".join(map(str, path)), (
                path if small else None
            )
    if sh and step.hoist_extra:
        for (
            buf,
            t,
            root,
        ) in (
            step.hoist_extra
        ):  # later segments' staged batch inputs, in this copy (see _hoist_pairs)
            f = fresh.get(id(buf))
            if (
                f is not None
                and f[1] == t._version
                and (f[0] is t or _skey(f[0]) == _skey(t))
            ) or id(buf) in pend:
                continue
            if root in HOIST_KEEP_ROOTS:
                kk = getattr(buf, "_muse_vr_keep", None)
                if kk is not None and kk[0] is t and kk[1] == t._version:
                    _note_fresh(step, buf, t)
                    continue  # the batch's sampling parameters, unchanged since their copy into this buffer
            pend[id(buf)] = _skey(t)
            dst.append(buf)
            src.append(t)
        step.hoist_extra = None
    if prof_seg:
        t0 = _padd("walk:" + prof_seg, t0)
    if dst:
        _copy_grouped(dst, src, entry)
        if prof_seg:
            _padd("copy:" + prof_seg, t0)
            _PROF["ns"]["ncopy:" + prof_seg] = _PROF["ns"].get(
                "ncopy:" + prof_seg, 0
            ) + 1000 * len(dst)
    for i, t, v in commit:
        keep[i] = (t, v)
    if sh:
        for d, s2 in zip(dst, src):
            _note_fresh(step, d, s2)
    return None


def _word_view(t):
    """(width, view): a contiguous tensor seen as 4-byte words when its rows allow, else as bytes (a 0-dim tensor as its
    one-element 1-dim view first); (0, None) if not."""
    if not t.is_contiguous():
        return 0, None
    try:
        if t.dim() == 0:
            t = t.view(1)
        if (t.shape[-1] * t.element_size()) % 4 == 0:
            return 4, t.view(torch.int32)
        return 1, t.view(torch.uint8)
    except RuntimeError:
        return 0, None


def _copy_grouped(dst, src, entry=None) -> None:
    """Bit copies with as few launches as possible: contiguous pairs go through one multi-tensor copy per word width
    (4-byte words or bytes) whatever their dtypes (the private buffers' views are made once per entry); the rest one
    ``torch._foreach_copy_`` per dtype (a list of one dtype takes the fused multi-tensor path, one launch); tensors above
    ``COPY_SPLIT_BYTES`` one plain copy each."""
    if _bc_on():
        rest = _batch_copy(dst, src)
        if not rest:
            return
        dst, src = [d for d, _ in rest], [s2 for _, s2 in rest]
    cache = entry.setdefault("wview", {}) if entry is not None else None
    w4d, w4s, w1d, w1s = [], [], [], []
    groups = {}
    big = []
    split = bed.env_int(ENV_COPY_SPLIT, COPY_SPLIT_BYTES)
    for d, s2 in zip(dst, src):
        if d.numel() * d.element_size() > split:
            big.append((d, s2))
            continue
        if cache is not None and s2.is_contiguous():
            # keyed by the buffer's address and layout, not its Python id: with shared staging an entry also copies into
            # buffers of other entries, and an id can be reused once such a buffer object is gone
            ck = (d.data_ptr(), tuple(d.shape), d.dtype)
            wv = cache.get(ck)
            if wv is None:
                wv = cache[ck] = _word_view(d)
            if wv[0]:
                s1 = s2.view(1) if s2.dim() == 0 else s2
                if wv[0] == 4:
                    w4d.append(wv[1])
                    w4s.append(s1.view(torch.int32))
                else:
                    w1d.append(wv[1])
                    w1s.append(s1.view(torch.uint8))
                continue
        g = groups.get(d.dtype)
        if g is None:
            groups[d.dtype] = ([d], [s2])
        else:
            g[0].append(d)
            g[1].append(s2)
    if w4d:
        groups[0] = (w4d, w4s)
    if w1d:
        groups[1] = (w1d, w1s)
    for ds, ss in groups.values():
        try:
            torch._foreach_copy_(ds, ss)
        except Exception:  # noqa: BLE001
            for d, s2 in zip(ds, ss):
                d.copy_(s2)
    if big:
        _copy_each(big)


@functools.lru_cache(maxsize=None)
def _bc_module():
    from sglang.kernels.jit.utils import load_jit

    return load_jit(
        "muse_vr_batch_copy",
        cuda_files=[_BC_CUH],
        cuda_wrappers=[("run", "MuseVrBatchCopy::run")],
    )


def _bc_on() -> bool:
    on = _state.get("bc_on")
    if on is None:
        on = False
        if bed.env_int(ENV_BATCH_COPY, 0) == 1:
            try:
                _bc_module()
                on = True
            except Exception as e:  # noqa: BLE001 - the grouped copies stay in use (loud)
                bed.count(NAME, f"batch copy unavailable:{type(e).__name__}")
                logger.warning(
                    "muse_vr_plugin: %s batch copy unavailable (%s); grouped copies stay",
                    NAME,
                    e,
                )
        _state["bc_on"] = on
    return on


_BC_TABLES: dict = {}  # n -> (CPU int64 tensor [3, n], its numpy view): reused, the launch reads the table on the host


def _bc_table(n: int):
    t = _BC_TABLES.get(n)
    if t is None:
        ten = torch.empty((3, n), dtype=torch.int64)
        t = _BC_TABLES[n] = (ten, ten.numpy())
    return t


def _batch_copy(dst, src) -> list:
    """Every contiguous pair in one batched byte-copy launch (at most BC_MAX per launch); returns the pairs left."""
    rest, dp, sp, nb = [], [], [], []
    for d, s2 in zip(dst, src):
        if len(nb) < BC_MAX and d.is_contiguous() and s2.is_contiguous():
            n = d.numel() * d.element_size()
            if n:
                dp.append(d.data_ptr())
                sp.append(s2.data_ptr())
                nb.append(n)
            continue
        rest.append((d, s2))
    if nb:
        ten, arr = _bc_table(len(nb))
        arr[0] = dp
        arr[1] = sp
        arr[2] = nb
        _bc_module().run(ten, dst[0])  # dst[0]: device + current stream
    return rest


def _copy_each(pairs) -> None:
    """Large staged tensors: one copy each (a contiguous same-dtype pair is a plain device-to-device memory copy)."""
    for d, s2 in pairs:
        d.copy_(s2)


def _set_path(step, path, value):
    """Make the method's object local at ``path`` hold ``value`` for this step (restored when the step ends)."""
    obj = _loc(step).get(path[0])
    for i, part in enumerate(path[1:-1], start=1):
        obj = obj[part] if isinstance(part, int) else getattr(obj, part)
    last = path[-1]
    if isinstance(last, int):
        return (
            False  # elements of local tuples / lists are substituted as call arguments
        )
    if isinstance(obj, functools.partial):
        return False
    try:
        old = getattr(obj, last)
        setattr(obj, last, value)
    except Exception:  # noqa: BLE001
        return False
    step.restore.append((obj, last, old))
    return True


def _written(step, seg) -> set:
    return set(OUTPUT_PATHS.get(seg, ())) | _state["written"].get(
        _wkey(step, seg), set()
    )


def _begin_capture(step, seg, ext, warm):
    fixed, staged = {}, {}
    written = _written(step, seg)
    sh = _shared_on()
    wst = _worker_storages(step.worker) if sh else ()
    needs = set()
    for path, t in ext:
        sigs = warm["sigs"][0 if _single(seg) else step.parity].get(path, set())
        if path in written:
            # an output: read in place at its current address (a buffer that moved since the warm-up, e.g. grown for
            # a larger batch, is the one the method writes now); if it took several addresses within the warm-up
            # of this variant, it is a per-step tensor the segment writes: no graph for this shape
            if len(sigs) > 1 and path not in OUTPUT_PATHS.get(seg, ()):
                bed.count(
                    NAME,
                    f"stock:{seg}:written per-step input {'/'.join(map(str, path))}",
                )
                _state["failed"].add(_wkey(step, seg))
                return _Run(step, seg, "eager")
            fixed[path] = _sig(t)
            continue
        # every small input the segment only reads is copied into a private buffer: no address guard, so whether a
        # step replays never depends on where this rank's allocator put a tensor (all tensor-parallel ranks must take
        # the same path: a check, an eager fallback or a capture on one rank only would pair different collectives)
        persistent = path in PERSISTENT_PATHS.get(seg, ())
        if (
            path[0] != "self"
            and not persistent
            and t.numel() * t.element_size() <= SMALL_BYTES
        ):
            if sh and t.numel() and t.data_ptr() in step.result_ptrs:
                # a result of a segment that ran from its graph this step: that graph's own memory, the same on every
                # replay of it (this graph then replays only after that one in the same step)
                fixed[path] = _sig(t)
                needs.add(step.result_ptrs[t.data_ptr()])
            elif (
                sh
                and t.numel()
                and t.untyped_storage().data_ptr() in wst
                and len(sigs) == 1
                and _sig(t) in sigs
            ):
                fixed[path] = _sig(
                    t
                )  # a view of the worker's own buffer, at one address through the warm-up
            else:
                staged[path] = t
        elif path[0] == "self" or persistent or (len(sigs) == 1 and _sig(t) in sigs):
            fixed[path] = _sig(
                t
            )  # worker attributes and large persistent buffers: read in place, guarded
        else:
            bed.count(
                NAME, f"stock:{seg}:large per-step input {'/'.join(map(str, path))}"
            )
            _state["failed"].add(_wkey(step, seg))
            return _Run(step, seg, "eager")
    run = _Run(step, seg, "capture")
    run.needs = frozenset(needs)
    # the batch's tensors as they are now: the private buffers below replace some of them on their objects for the step
    bl = (
        _batch_leaves(step.batch, step.spec0)
        if sh and seg != "E" and step.batch is not None
        else None
    )
    const = set()
    if sh:
        cv = _state.get("const_vals", {}).get(_wkey(step, seg), {})
        for path, t in staged.items():
            seen = cv.get(path)
            if (
                seen is not None
                and len(seen) == 1
                and t.numel() <= 8
                and tuple(t.reshape(-1).tolist()) in seen
            ):
                const.add(path)
    bufs = {}
    for path, t in staged.items():
        buf = None
        if sh:  # the same source copied earlier in this step (by a replay or a capture): share its private buffer
            hit = step.shared.get(_skey(t))
            if hit is not None and hit[1] == t._version:
                buf = hit[0]
        if buf is None:
            buf = torch.empty_strided(
                t.size(), t.stride(), dtype=t.dtype, device=t.device
            )
            buf.copy_(t)
            if sh:
                _note_fresh(step, buf, t)
        bufs[path] = buf
        run.subst[id(t)] = buf
        if len(path) > 1:
            _set_path(step, path, buf)
    run.inputs = [
        (path, CONST if path in const else True, buf) for path, buf in bufs.items()
    ] + [(path, False, sig[0]) for path, sig in fixed.items()]
    if bl is not None:
        run.hoist = _hoist_list(bl, staged, bufs)
    run.state_before = {k: v for k, v in vars(step.worker).items() if type(v) is int}
    run.rng = torch.cuda.get_rng_state()
    run.graph = torch.cuda.CUDAGraph()
    run.pool = torch.cuda.graph_pool_handle()
    _install_capture_patches()
    run.cap = _Capture(torch.cuda.current_stream(), step)
    run.gc_was_enabled = gc.isenabled()
    gc.disable()  # no collection (and no destruction of other graphs) inside the capture window
    run.graph.capture_begin(pool=run.pool, capture_error_mode="thread_local")
    _tls.cap = run.cap
    bed.count(NAME, f"capture:{seg}")
    return run


def _end_capture(run):
    cap = run.cap
    try:
        cap.join_all()
        h2d = _capture_host_copies(cap.stream)
        if h2d:
            cap.error = (
                cap.error
                or f"{h2d} copies to / from host memory recorded (per-step host values)"
            )
    finally:
        _tls.cap = None
        try:
            run.graph.capture_end()
        finally:
            if run.gc_was_enabled:
                gc.enable()
    torch.cuda.set_rng_state(run.rng)
    if cap.error is not None:
        raise RuntimeError(cap.error)
    run.graph.replay()
    step = run.step
    _rerecord(_weak(cap.event_objs), step.stream)
    delta = {
        k: v
        for k, v in vars(step.worker).items()
        if type(v) is int and run.state_before.get(k, v) != v
    }
    if (
        run.seg == "A"
    ):  # B's variant of this parity was captured against A's previous graph
        _bury(
            _state["entries"].pop((id(step.worker), "B", step.key, step.parity), None)
        )
    _bury(_state["entries"].get(_ekey(step, run.seg)))
    handoff = []
    for mod_name, attr in HANDOFFS.get(run.seg, ()):
        d = getattr(sys.modules.get(mod_name), attr, None)
        if isinstance(d, dict) and d:
            handoff.append((d, dict(d)))
    _state["entries"][_ekey(step, run.seg)] = {
        "graph": run.graph,
        "pool": run.pool,
        "inputs": run.inputs,
        "handoff": handoff,
        "results": [c[1] for c in run.calls],
        "members": [c[0] for c in run.calls],
        "state_delta": delta,
        "events": _weak(cap.event_objs),
        "obj_delta": _last_assignments(run.obj_delta),
        "tally": f"replay:{run.seg}:bs={step.key[0]}",
        "pins": _pins(step.worker),
        "needs": getattr(run, "needs", frozenset()),
        "hoist": getattr(run, "hoist", ()),
    }
    if getattr(run, "hoist", ()):
        _state.setdefault("hoist_idx", {}).setdefault(
            (id(step.worker), step.base_key, step.parity), set()
        ).add(_ekey(step, run.seg))
    if _shared_on():
        step.graphed.add(run.seg)
        step.result_ptrs.update(
            dict.fromkeys(_rptrs(_state["entries"][_ekey(step, run.seg)]), run.seg)
        )
    if (
        _vlog_left() > 0
    ):  # debug: where and with which worker counters this graph was captured
        try:
            if (
                run.seg == "A"
            ):  # the sampling component's scratch this graph was captured with
                _state["entries"][_ekey(step, run.seg)]["dbg_vts"] = _vts_scratch()
            _capture_log(run, step, delta)
        except Exception as e:  # noqa: BLE001 - debug only
            bed.count(NAME, f"debug log error:{type(e).__name__}")
    if _widthed(
        run.seg
    ):  # spans only grow within a request: keep the few newest per shape
        base = (id(step.worker), run.seg, step.key[:-1])
        same = [
            k for k in _state["entries"] if k[:2] == base[:2] and k[2][:-1] == base[2]
        ]
        for k in same[:-WIDTH_KEEP]:
            _bury(_state["entries"].pop(k, None))
    if not _state.setdefault("reported", set()) & {run.seg}:
        _state["reported"].add(run.seg)
        bed.report(
            f"{NAME}:inputs:{run.seg}",
            True,
            staged=["/".join(map(str, p)) for p, m, _ in run.inputs if m],
            const=["/".join(map(str, p)) for p, m, _ in run.inputs if m == CONST],
            hoisted=len(getattr(run, "hoist", ())),
            needs=sorted(getattr(run, "needs", ())),
            fixed=len([1 for _, m, _ in run.inputs if not m]),
            events=len(_state["entries"][_ekey(step, run.seg)]["events"]),
        )
    bed.count(NAME, f"captured:{run.seg}:bs={step.key[0]}")


PIN_DEPTH = 4
# tensors kept per graph at most (host references only); module state is walked first, so the replaceable
# component and SGLang buffers are always within it; hitting the cap is counted ("pins capped")
PIN_MAX = 1 << 16
# module state walked for pins: the other plugin components, and SGLang's own module-level caches of the decode path
# (e.g. the speculative chain-verify buffers, re-made when a larger batch arrives)
PIN_MODULES = (
    _PKG + ".",
    "sglang.srt.speculative.",
    "sglang.srt.layers.",
    "sglang.srt.sampling.",
    "sglang.kernels.",
)
_PIN_OBJ_MODULES = (
    _PKG,
    "sglang",
    "types",
)  # classes whose instances' attributes are walked


def _pins(worker) -> list:
    """The CUDA tensors a captured graph may use without the entry holding them: the worker's attributes and the
    module state of the other plugin components (pools, scratch rows, caches). A component that replaces such a buffer
    (grown for a larger batch) frees the old one, and a graph captured before would then read and write freed memory;
    the entry keeps them alive, so each graph keeps the buffers it was captured with."""
    out, seen, depth_of = (
        [],
        set(),
        {},
    )  # tensors by id; containers by id -> the shallowest depth they were walked at
    capped = []

    def walk(obj, depth):
        if len(out) >= PIN_MAX:
            if not capped:
                capped.append(True)
            return
        if depth > PIN_DEPTH:
            return
        if isinstance(obj, torch.Tensor):
            if obj.is_cuda and id(obj) not in seen:
                seen.add(id(obj))
                out.append(obj)
            return
        if (
            isinstance(obj, (torch.nn.Module, type, str, bytes, int, float, bool))
            or obj is None
        ):
            return
        d0 = depth_of.get(id(obj))
        if d0 is not None and d0 <= depth:
            return  # walked already from at least as shallow (a deeper first visit is walked again: it reaches further)
        depth_of[id(obj)] = depth
        if isinstance(obj, dict):
            vals = list(obj.values())
        elif isinstance(obj, (list, tuple)):
            if obj and isinstance(obj[0], (int, float, str, bool)):
                return  # a list of plain values
            vals = list(obj)
        elif type(obj).__module__.startswith(
            _PIN_OBJ_MODULES
        ):  # plugin / SGLang objects (planes, caches)
            d = getattr(obj, "__dict__", None)
            vals = list(d.values()) if isinstance(d, dict) else []
        else:
            return
        for v in vals:
            walk(v, depth + 1)

    roots = _state.get("pin_roots")
    if roots is None or roots[0] != len(
        sys.modules
    ):  # the module scan is cached until new modules load
        found = []
        for name, mod in list(sys.modules.items()):
            if mod is None or name == __name__ or not name.startswith(PIN_MODULES):
                continue
            for k, v in list(vars(mod).items()):
                if (
                    k.startswith("_")
                    and not k.startswith("__")
                    and not callable(v)
                    and not inspect.ismodule(v)
                    and not isinstance(v, (str, bytes, int, float, bool, type(None)))
                ):
                    found.append((mod, k))
        roots = _state["pin_roots"] = (len(sys.modules), found)
    # module state first (the buffers that components and SGLang re-make when a larger batch arrives), then the
    # worker's attributes: with the cap reached during a large worker walk, module state would otherwise be skipped
    for mod, k in roots[1]:
        walk(
            getattr(mod, k, None), 1
        )  # read again each time: a module may rebind its cache
    walk(
        {k: v for k, v in vars(worker).items() if not isinstance(v, torch.nn.Module)}, 0
    )
    if capped:
        bed.count(NAME, "pins capped")
    return out


def _last_assignments(rec) -> list:
    """The recorded attribute assignments, keeping only the last one per (object, attribute), in that order."""
    last = {}
    for i, (ref, k, v) in enumerate(rec):
        last[(id(ref()), k)] = i
    return [rec[i] for i in sorted(last.values())]


def _consumable_state() -> list:
    """State a segment's calls may consume, as it is now: the other plugin components' thread-local hand-offs (a module
    ``_tls``, e.g. the top-k sampler's per-step pending marker) and the HANDOFFS dicts. Saved before each captured call,
    so that a failed capture's eager re-run of each call sees what that call saw."""
    mods = _state.get("tls_mods")
    if mods is None or mods[0] != len(sys.modules):
        found = [
            m
            for n, m in list(sys.modules.items())
            if m is not None
            and n.startswith(_PKG + ".")
            and n != __name__
            and isinstance(getattr(m, "_tls", None), threading.local)
        ]
        mods = _state["tls_mods"] = (len(sys.modules), found)
    out = [(m._tls, dict(vars(m._tls))) for m in mods[1]]
    for hs in HANDOFFS.values():
        for mod_name, attr in hs:
            d = getattr(sys.modules.get(mod_name), attr, None)
            if isinstance(d, dict):
                out.append((d, dict(d)))
    return out


def _restore_state(saved) -> None:
    for obj, snap in saved:
        d = obj if isinstance(obj, dict) else vars(obj)
        d.clear()
        d.update(snap)


def _abort_capture(run, exc):
    """Capture failed: end it, restore the generator, rerun the recorded calls eagerly into the tensors already
    returned to the method. The shape stays eager."""
    logger.warning(
        "muse_vr_plugin: %s capture of segment %s failed (%s: %s); this shape stays eager",
        NAME,
        run.seg,
        type(exc).__name__,
        exc,
    )
    if getattr(_tls, "cap", None) is not None:
        _tls.cap = None
        try:
            run.graph.capture_end()
        except Exception:  # noqa: BLE001
            pass
    if getattr(run, "gc_was_enabled", False):
        gc.enable()
    torch.cuda.set_rng_state(run.rng)
    _state["failed"].add(_wkey(run.step, run.seg))
    bed.count(NAME, f"capture failed:{run.seg}")
    for original, args, kwargs, result, saved in run.redo:
        _restore_state(
            saved
        )  # each call re-runs from the state it saw when it was captured
        _copy_into(result, original(*args, **kwargs))


def _copy_into(dst, src):
    a, b = [], []
    _leaves(dst, (), a)
    _leaves(src, (), b)
    if [p for p, _ in a] != [p for p, _ in b]:
        raise RuntimeError(
            f"{NAME}: cannot recover a failed capture (result structure differs)"
        )
    for (_, d), (_, s) in zip(a, b):
        if d is not s:
            d.copy_(s)


def _rewrite(obj, subst):
    if isinstance(obj, torch.Tensor):
        return subst.get(id(obj), obj)
    if isinstance(obj, tuple) and not hasattr(obj, "_fields"):
        return tuple(_rewrite(v, subst) for v in obj)
    if isinstance(obj, list):
        return [_rewrite(v, subst) for v in obj]
    return obj


_PLAIN_COPY: dict = {}  # class -> True when copy.copy of its instances is a new object with the same __dict__


def _plain_copyable(cls) -> bool:
    ok = _PLAIN_COPY.get(cls)
    if ok is None:
        import copyreg

        ok = (
            getattr(cls, "__copy__", None) is None
            and cls.__reduce_ex__ is object.__reduce_ex__
            and cls.__reduce__ is object.__reduce__
            and cls not in copyreg.dispatch_table
            and getattr(cls, "__getstate__", object.__getstate__) is object.__getstate__
            and not hasattr(cls, "__slots__")
            and "__dict__" in dir(cls)
            and not issubclass(cls, type)
        )
        _PLAIN_COPY[cls] = ok
    return ok


def _shallow(obj):
    if obj is None or isinstance(obj, (torch.Tensor, tuple, int, float, bool, str)):
        return obj
    cls = type(obj)
    if _plain_copyable(
        cls
    ):  # what copy.copy does for such a class, without its reduce protocol round trip
        new = cls.__new__(cls)
        new.__dict__.update(obj.__dict__)
        return new
    try:
        return copy.copy(obj)
    except Exception:  # noqa: BLE001
        return obj


# ------------------------------------------------------------------------------------------------- calls
SIM_MODULE = "sglang.srt.speculative.dflash_worker_v2"

# SGLang main applies the synthetic-acceptance forcing as the last statement of _accept_block (the branch layout's
# method applied it after the call). Segment A ends at _accept_block's return, so on main the forcing is kept out of
# every capture: the outermost wrapper of _accept_block (_forcing_after) runs the call with that statement switched off
# and then runs the same statement itself, eagerly, once per call, with the same arguments. Only on the selector path,
# where that statement's target_predict is the call's own argument; on any other path segment A runs eagerly while
# acceptance is simulated (_open). With acceptance not simulated the wrapper only reads the setting and calls through.
_FORCING: dict = {"suppressed": False, "true": None}


def _sim_len(mod):
    """The method's SIMULATE_ACC_LEN, also while _forcing_after has it switched off for the inner call."""
    return (
        _FORCING["true"]
        if _FORCING["suppressed"]
        else getattr(mod, "SIMULATE_ACC_LEN", None)
    )


def _simulated(mod) -> bool:
    val = _sim_len(mod)
    return isinstance(val, (int, float)) and val > 0


def _forcing_in_call(worker) -> bool:
    """SGLang main, acceptance simulated, and a path where the forcing cannot be moved after the call."""
    from . import sgl_compat

    mod = sys.modules.get(SIM_MODULE)
    return (
        sgl_compat.layout() == "main"
        and mod is not None
        and _simulated(mod)
        and getattr(worker, "_selector_sample", None) is None
    )


def _forcing_after(fn):
    """Outermost wrapper of ``_accept_block`` on SGLang main (see _FORCING)."""

    @functools.wraps(fn)
    def accept_block(self, **kw):
        mod = sys.modules[SIM_MODULE]
        sim = mod.SIMULATE_ACC_LEN
        if (
            not (isinstance(sim, (int, float)) and sim > 0)
            or _FORCING["suppressed"]
            or getattr(self, "_selector_sample", None) is None
        ):
            if (
                isinstance(sim, (int, float))
                and sim > 0
                and not _FORCING["suppressed"]
                and _sim_fold_on()
            ):
                return _with_sim_nsl(
                    fn, self, kw
                )  # the forcing stays in the call; its fold returns the lengths
            return fn(self, **kw)
        # The switch is a process-wide module setting, so it must only ever be flipped from one thread: SGLang runs
        # _accept_block on the scheduler's forward (event-loop) thread; the plugin's own helper threads (e.g.
        # host_ingest's request broadcast) never read the setting. A call from another thread is refused loudly.
        tid = threading.get_ident()
        if _FORCING.setdefault("thread", tid) != tid:
            raise RuntimeError(
                f"{NAME}: _accept_block called from a second thread under simulated acceptance"
            )
        _FORCING.update(suppressed=True, true=sim)
        mod.SIMULATE_ACC_LEN = 0
        try:
            accept_len, commit_lens, bonus, out_tokens, _ = fn(self, **kw)
        finally:
            mod.SIMULATE_ACC_LEN = sim
            _FORCING.update(suppressed=False, true=None)
        _SIMCTX.prefix_lens, _SIMCTX.nsl = (
            kw.get("prefix_lens"),
            None,
        )  # for the statement's fold
        try:
            mod.apply_dflash_simulated_acceptance(
                candidates=kw["candidates"],
                target_predict=kw.get("target_predict"),
                accept_len=accept_len,
                commit_lens=commit_lens,
                bonus=bonus,
                out_tokens=out_tokens,
                simulate_acc_len=sim,
                simulate_acc_method=mod.SIMULATE_ACC_METHOD,
                simulate_acc_token_mode=mod.SIMULATE_ACC_TOKEN_MODE,
                next_token_logits=kw["next_token_logits"],
            )
        finally:
            _SIMCTX.prefix_lens = None
        bed.count(NAME, "forcing after A")
        return accept_len, commit_lens, bonus, out_tokens, _take_sim_nsl(commit_lens)

    accept_block._muse_vr_step_graphs = True
    accept_block._muse_vr_forcing_after = True
    return accept_block


# ------------------------------------------------------------------------------------------- the sim fold
# MUSE_VR_STEP_GRAPHS_SIM_FOLD=1 (opt-in). The synthetic-acceptance forcing statement,
# dflash_utils.apply_dflash_simulated_acceptance (as dflash_worker_v2 imports it; token mode "real-draft-token"), runs
# eagerly at the end of _accept_block (the in-call path, segment A eager) or right after it (_forcing_after, selector
# path). For the forced commit length k + 1 it draws on the host it launches: the target argmax of every verify row
# (bs * block rows of the gathered logits) when no target_predict is given, two fills, a zero fill and three device
# copies; the caller then adds the commit lengths to the prefix lengths (a cast and an add). About ten launches, each
# with an eager launch's gap (~29 us per step at c1, 393K t1, host ~5 ms ahead). The fold wraps the statement
# (installed with the other wrappers): it draws k + 1 with the same function and arguments (the statement's one host
# draw) and runs two Triton launches: a partial argmax of the ONE row per request the statement reads (row b * block + k)
# over vocabulary chunks; then one program per request reduces the partials by torch.argmax's rule (the largest value,
# the lowest index among equal values, the first NaN if there is one) and writes accept_len (k), commit_lens (k + 1),
# bonus, out_tokens (candidates[b, 1:k+1], then the bonus, then zeros) and, when _forcing_after passed the call's
# prefix lengths (_SIMCTX), the new sequence lengths (prefix_lens + k + 1 in prefix_lens' dtype), which
# _forcing_after returns in place of None so the caller skips its recomputation. A given target_predict (greedy verify)
# is read at [b, k]. Calls outside the statement's plain form (other token modes, shapes, dtypes, strides) run the
# statement (before any draw); a launch that fails after the draw writes the outputs with the statement's own
# operations for that k (no second draw) and turns the fold off. SIM_PINS: the statement and the draw it mirrors.
ENV_SIM_FOLD = "MUSE_VR_STEP_GRAPHS_SIM_FOLD"
SIM_UTILS = "sglang.srt.speculative.dflash_utils"
SIM_PINS = {
    "sglang/srt/speculative/dflash_utils.py": (
        "35e885f930e95ed82a1f47a0dfa2dab583225d170c6cabe54f6c7bb4e3abbf79",
    ),
    "sglang/srt/speculative/spec_utils.py": (
        "bfa9d4cd9565b808684f6a51cf5cf27804fc2914c0053beb04ad2ed01eadac64",
    ),
}
SIM_CHUNK = 4096  # vocabulary columns per partial-argmax program
SIM_MAX_BS, SIM_MAX_BLOCK, SIM_MAX_PARTS = 64, 64, 4096
_SIM_KEYS = frozenset(
    (
        "candidates",
        "target_predict",
        "accept_len",
        "commit_lens",
        "bonus",
        "out_tokens",
        "simulate_acc_len",
        "simulate_acc_method",
        "simulate_acc_token_mode",
        "next_token_logits",
    )
)
_SIM_FLOATS = (torch.float32, torch.bfloat16, torch.float16)
_SIM_INTS = (torch.int32, torch.int64)
_SIMCTX = threading.local()  # _forcing_after -> the fold: the call's prefix lengths; back: (commit_lens, new lengths)


def _sim_fold_on() -> bool:
    on = _state.get("sim_fold_on")
    if on is None:
        on = bed.flag(NAME + "_sim_fold")
        if on:
            bad = [rel for rel, want in SIM_PINS.items() if not bed.pinned(rel, want)]
            if bad:
                bed.report(NAME + ":sim_fold", False, reason="pins: " + ", ".join(bad))
            else:
                bed.report(NAME + ":sim_fold", True, chunk=SIM_CHUNK)
            on = not bad
        _state["sim_fold_on"] = on
    return on


def _with_sim_nsl(fn, self, kw):
    """``fn(self, **kw)`` (the in-call path) with the call's prefix lengths offered to the statement's fold; its
    return with the fold's new sequence lengths in place of None."""
    _SIMCTX.prefix_lens, _SIMCTX.nsl = kw.get("prefix_lens"), None
    try:
        out = fn(self, **kw)
    finally:
        _SIMCTX.prefix_lens = None
    if not (isinstance(out, tuple) and len(out) == 5 and out[4] is None):
        _SIMCTX.nsl = None
        return out
    nsl = _take_sim_nsl(out[1])
    return out if nsl is None else out[:4] + (nsl,)


def _take_sim_nsl(commit_lens):
    """The fold's new sequence lengths for the forcing that wrote ``commit_lens`` (else None), taken once."""
    got, _SIMCTX.nsl = getattr(_SIMCTX, "nsl", None), None
    return got[1] if got is not None and got[0] is commit_lens else None


def _sim_statement_hook(original):
    """Wrapper of dflash_worker_v2.apply_dflash_simulated_acceptance (see the section comment)."""

    @functools.wraps(original)
    def apply_dflash_simulated_acceptance(*args, **kw):
        if args or not _sim_fold_on() or not _sim_fold(kw):
            return original(*args, **kw)
        return None

    apply_dflash_simulated_acceptance._muse_vr_sim_fold = True
    return apply_dflash_simulated_acceptance


@functools.lru_cache(maxsize=None)
def _sim_fold_kernels():
    if triton is None:
        raise RuntimeError("triton is not importable")

    @triton.jit(do_not_specialize=["row_in_block"])
    def sim_partial_argmax(
        logits_ptr,
        row_stride,
        vocab,
        row_in_block,
        block,
        part_val_ptr,
        part_idx_ptr,
        n_parts,
        CHUNK: tl.constexpr,
    ):
        # one chunk of row b * block + row_in_block: its largest value and the lowest column holding it, or
        # (first NaN column - vocab), a negative index, when the chunk has a NaN
        b = tl.program_id(0)
        c = tl.program_id(1)
        row = logits_ptr + (b * block + row_in_block).to(tl.int64) * row_stride
        offs = c * CHUNK + tl.arange(0, CHUNK)
        inr = offs < vocab
        x = tl.load(row + offs, mask=inr, other=float("-inf")).to(tl.float32)
        nan = (x != x) & inr
        first_nan = tl.min(tl.where(nan, offs, vocab), axis=0)
        xv = tl.where(nan, float("-inf"), x)
        m = tl.max(xv, axis=0)
        first_max = tl.min(tl.where((xv == m) & inr, offs, vocab), axis=0)
        tl.store(part_val_ptr + b * n_parts + c, m)
        tl.store(
            part_idx_ptr + b * n_parts + c,
            tl.where(first_nan < vocab, first_nan - vocab, first_max),
        )

    @triton.jit(do_not_specialize=["k"])
    def sim_finish(
        part_val_ptr,
        part_idx_ptr,
        n_parts,
        vocab,
        cand_ptr,
        cand_stride,
        tp_ptr,
        tp_stride,
        prefix_ptr,
        accept_ptr,
        commit_ptr,
        bonus_ptr,
        out_ptr,
        out_stride,
        nsl_ptr,
        k,
        block,
        HAS_TP: tl.constexpr,
        HAS_NSL: tl.constexpr,
        PARTS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        b = tl.program_id(0)
        if HAS_TP:
            best = tl.load(tp_ptr + b * tp_stride + k).to(tl.int64)
        else:
            p = tl.arange(0, PARTS)
            pm = p < n_parts
            v = tl.load(part_val_ptr + b * n_parts + p, mask=pm, other=float("-inf"))
            i = tl.load(part_idx_ptr + b * n_parts + p, mask=pm, other=0)
            nanp = (i < 0) & pm
            any_nan = tl.max(nanp.to(tl.int32), axis=0) > 0
            first_nan = tl.min(tl.where(nanp, i, 0), axis=0) + vocab
            m = tl.max(v, axis=0)
            first_max = tl.min(tl.where((v == m) & pm, i, 2147483647), axis=0)
            best = tl.where(any_nan, first_nan, first_max).to(tl.int64)
        j = tl.arange(0, BLOCK)
        jm = j < block
        cand = tl.load(
            cand_ptr + b * cand_stride + j + 1, mask=jm & (j < k), other=0
        ).to(tl.int64)
        out = tl.where(j < k, cand, tl.where(j == k, best, 0))
        tl.store(
            out_ptr + b * out_stride + j, out.to(out_ptr.dtype.element_ty), mask=jm
        )
        tl.store(bonus_ptr + b, best.to(bonus_ptr.dtype.element_ty))
        tl.store(accept_ptr + b, k.to(accept_ptr.dtype.element_ty))
        tl.store(commit_ptr + b, (k + 1).to(commit_ptr.dtype.element_ty))
        if HAS_NSL:
            pre = tl.load(prefix_ptr + b)
            tl.store(nsl_ptr + b, (pre + (k + 1)).to(nsl_ptr.dtype.element_ty))

    return sim_partial_argmax, sim_finish


def _sim_launch(kw, pl, nsl, k: int) -> None:
    """The fold's two launches (see the section comment); ``pl`` / ``nsl``: prefix and new lengths, or None."""
    part, finish = _sim_fold_kernels()
    logits, cand, tp = (
        kw["next_token_logits"],
        kw["candidates"],
        kw.get("target_predict"),
    )
    accept_len, commit_lens, bonus, out_tokens = (
        kw[n] for n in ("accept_len", "commit_lens", "bonus", "out_tokens")
    )
    bs, block = cand.shape
    if tp is None:
        vocab = logits.shape[1]
        n_parts = triton.cdiv(vocab, SIM_CHUNK)
        pv = torch.empty((bs, n_parts), dtype=torch.float32, device=logits.device)
        pi = torch.empty((bs, n_parts), dtype=torch.int32, device=logits.device)
        part[(bs, n_parts)](
            logits,
            logits.stride(0),
            vocab,
            k,
            block,
            pv,
            pi,
            n_parts,
            CHUNK=SIM_CHUNK,
            num_warps=4,
        )
    else:  # target_predict given (logits may be None): one launch; the partials are not read (HAS_TP)
        vocab, n_parts = 1, 1
        pv = pi = accept_len
    finish[(bs,)](
        pv,
        pi,
        n_parts,
        vocab,
        cand,
        cand.stride(0),
        cand if tp is None else tp,
        0 if tp is None else tp.stride(0),
        accept_len if pl is None else pl,
        accept_len,
        commit_lens,
        bonus,
        out_tokens,
        out_tokens.stride(0),
        accept_len if nsl is None else nsl,
        k,
        block,
        HAS_TP=tp is not None,
        HAS_NSL=nsl is not None,
        PARTS=triton.next_power_of_2(n_parts),
        BLOCK=triton.next_power_of_2(block),
        num_warps=4,
    )


def _sim_apply_torch(kw, accept_len, commit_lens, bonus, out_tokens, k: int):
    """The statement's operations (token mode "real-draft-token") for the forced commit length k + 1, in place on the
    given outputs."""
    cand, tp = kw["candidates"], kw.get("target_predict")
    bs, block = cand.shape
    if tp is None:
        tp = torch.argmax(kw["next_token_logits"], dim=-1).view(bs, block)
    accept_len.fill_(k)
    commit_lens.fill_(k + 1)
    out_tokens.zero_()
    if k > 0:
        out_tokens[:, :k].copy_(cand[:, 1 : k + 1])
    bonus.copy_(tp[:, k].to(dtype=bonus.dtype))
    out_tokens[:, k].copy_(bonus.to(dtype=out_tokens.dtype))
    return accept_len, commit_lens, bonus, out_tokens


def _sim_fold_reason(kw) -> str | None:
    """None when the fold takes this call of the statement (its plain form), else why not."""
    if not set(kw) - {"fixed_token_id"} == _SIM_KEYS:
        return "arguments"
    if kw["simulate_acc_token_mode"] != "real-draft-token":
        return "token mode"
    cand, logits, tp = kw["candidates"], kw["next_token_logits"], kw["target_predict"]
    outs = tuple(kw[n] for n in ("accept_len", "commit_lens", "bonus", "out_tokens"))
    # the logits are read only when no target_predict is given (the statement's argmax); the greedy verify passes
    # target_predict with next_token_logits=None (the method drops the logits after deferred_argmax)
    ts = (cand,) + outs + ((logits,) if tp is None else (tp,))
    if (
        not all(isinstance(t, torch.Tensor) and t.is_cuda for t in ts)
        or len({t.device for t in ts}) != 1
    ):
        return "inputs"
    if cand.dim() != 2 or cand.stride(1) != 1:
        return "candidates"
    bs, block = cand.shape
    if not (0 < bs <= SIM_MAX_BS and 1 < block <= SIM_MAX_BLOCK):
        return "batch or block size"
    if tp is None and (
        logits.dim() != 2
        or logits.shape[0] != bs * block
        or logits.stride(1) != 1
        or logits.dtype not in _SIM_FLOATS
        or not 0 < logits.shape[1] <= SIM_MAX_PARTS * SIM_CHUNK
    ):
        return "logits"
    for t in (outs[3],) + (() if tp is None else (tp,)):
        if t.dim() != 2 or tuple(t.shape) != (bs, block) or t.stride(1) != 1:
            return "token tensors"
    for t in outs[:3]:
        if t.dim() != 1 or t.numel() != bs:
            return "per-request tensors"
    if any(t.dtype not in _SIM_INTS for t in ts if t is not logits):
        return "dtypes"
    if torch.cuda.is_current_stream_capturing():
        return "graph capture"
    return None


def _sim_prefix(bs: int, device):
    """The prefix lengths _forcing_after offered for this call, when they fit the fold (else None: no new lengths)."""
    pl = getattr(_SIMCTX, "prefix_lens", None)
    if (
        isinstance(pl, torch.Tensor)
        and pl.is_cuda
        and pl.device == device
        and pl.dim() == 1
        and pl.numel() == bs
        and pl.dtype in _SIM_INTS
    ):
        return pl
    return None


def _sim_fold(kw) -> bool:
    """The statement for this call by the fold (True), or False (a call it does not take: nothing drawn)."""
    why = _sim_fold_reason(kw)
    if why is not None:
        bed.count(NAME, "sim fold: stock: " + why)
        return False
    cand = kw["candidates"]
    bs, block = cand.shape
    k = (
        int(
            sys.modules[SIM_UTILS].sample_simulated_acc_len(
                kw["simulate_acc_len"], kw["simulate_acc_method"], block
            )
        )
        - 1
    )
    pl = _sim_prefix(bs, cand.device)
    nsl = torch.empty_like(pl) if pl is not None else None
    outs = tuple(kw[n] for n in ("accept_len", "commit_lens", "bonus", "out_tokens"))
    try:
        if not 0 <= k < block:
            raise ValueError(f"forced length {k + 1} outside [1, {block}]")
        _sim_launch(kw, pl, nsl, k)
    except Exception as e:  # noqa: BLE001
        _state["sim_fold_on"] = False
        bed.report(
            NAME + ":sim_fold",
            False,
            reason=f"launch failed: {type(e).__name__}: {e}"[:300],
        )
        logger.warning("%s: sim fold off after a failed launch: %s", NAME, e)
        bed.count(NAME, "sim fold: the statement's operations after a failed launch")
        _sim_apply_torch(kw, *outs, k)
        return True
    bed.count(
        NAME, f"sim fold: rows={bs}" + ("" if nsl is not None else " (no lengths)")
    )
    if nsl is not None:
        _SIMCTX.nsl = (kw["commit_lens"], nsl)
    if bed.flag(
        NAME + "_sim_fold_shadow"
    ):  # debug: the statement's operations on copies, compared bit for bit
        ref = _sim_apply_torch(kw, *(t.clone() for t in outs), k)
        ok = all(a.dtype == r.dtype and torch.equal(a, r) for a, r in zip(outs, ref))
        if nsl is not None:
            want = pl + ref[1].to(pl.dtype)
            ok = ok and nsl.dtype == want.dtype and torch.equal(nsl, want)
        bed.shadow_result(NAME + ":sim_fold", ok)
    return True


# ------------------------------------------------------------------------------------- accept_triton
# MUSE_VR_STEP_GRAPHS_ACCEPT_TRITON=1 (opt-in). The selector-path accept
# (dflash_worker_v2._selector_sampling_accept, every sampled step) calls dspark_accept.accept_sampling, whose tail
# gathers the bonus with torch ops: an arange, a cast of correct_len, accept_index[rows, correct_len], a cast,
# predicts[accept_pos], a cast (six launches inside segment A, ~10 us with their gaps at c1). SGLang's own dispatcher
# of that function (dspark_accept.AcceptSampling.execute) takes accept_sampling_triton on CUDA inputs: the same core
# (_accept_sampling_core) and one Triton gather of the bonus (_gather_two_level_bonus_kernel). Integer gathers of the
# same tensors, so its results are bit-identical. The wrapper routes the worker's call as the dispatcher would (the
# first tensor argument on CUDA, SGLang's inputs_on_cuda); anything else runs the stock function.
# MUSE_VR_STEP_GRAPHS_ACCEPT_TRITON_SHADOW=1 (debug; eager calls only, host syncs): the stock function runs first, its
# results are copied, the default CUDA generator is put back to the state it started from, the Triton function runs,
# and (correct_len, bonus, cap_trim_lens) are compared bit for bit (bed.shadow_result).
ENV_ACCEPT_TRITON = "MUSE_VR_STEP_GRAPHS_ACCEPT_TRITON"
ACCEPT_MODULE = "sglang.kernels.ops.speculative.dspark.dspark_accept"
ACCEPT_PINS = {
    "sglang/kernels/ops/speculative/dspark/dspark_accept.py": (
        "4d87d25d0510a309e7f3484c0d2c6d806da39bd5557096c49b9ac2a92e5c3ddf",
    ),
    "sglang/kernels/ops/speculative/dspark/dispatch.py": (
        "8baa09b85ad97aae209b55f40271bdd4168a2b889247dd5aaa6a7dd1e0d1772d",
    ),
}


def _accept_triton_on() -> bool:
    on = _state.get("accept_triton_on")
    if on is None:
        on = bed.flag(NAME + "_accept_triton")
        if on:
            bad = [
                rel for rel, want in ACCEPT_PINS.items() if not bed.pinned(rel, want)
            ]
            bed.report(
                NAME + ":accept_triton",
                not bad,
                reason="pins: " + ", ".join(bad) if bad else "ok",
            )
            on = not bad
        _state["accept_triton_on"] = on
    return on


def _accept_hook(original, fast, on_cuda):
    """Wrapper of dflash_worker_v2.accept_sampling: ``fast`` (accept_sampling_triton) for CUDA inputs."""

    @functools.wraps(original)
    def accept_sampling(*args, **kw):
        if args or not _accept_triton_on() or not on_cuda(**kw):
            bed.count(NAME, "accept triton: stock")
            return original(*args, **kw)
        if (
            bed.flag(NAME + "_accept_triton_shadow")
            and not torch.cuda.is_current_stream_capturing()
        ):
            dev = kw["candidates"].device
            gen = torch.cuda.get_rng_state(dev)
            ref = tuple(t.clone() for t in original(**kw))
            torch.cuda.set_rng_state(gen, dev)
            out = fast(**kw)
            ok = len(out) == len(ref) and all(
                a.dtype == r.dtype and a.shape == r.shape and torch.equal(a, r)
                for a, r in zip(out, ref)
            )
            bed.shadow_result(NAME + ":accept_triton", ok)
        else:
            out = fast(**kw)
        bed.count(NAME, "accept triton")
        return out

    accept_sampling._muse_vr_accept_triton = True
    return accept_sampling


def _fold_on() -> bool:
    """Shared staging on and acceptance not simulated (the method's SIMULATE_ACC_LEN <= 0): segment A's graph also
    computes the new sequence lengths (the method recomputes them itself after a simulated forcing)."""
    on = _state.get("fold_on")
    if on is None:
        mod = sys.modules.get(SIM_MODULE)
        val = _sim_len(mod) if mod is not None else _state.get("sim_len")
        on = _state["fold_on"] = (
            _shared_on() and isinstance(val, (int, float)) and val <= 0
        )
    return on


def _fold_nsl(out, kwargs):
    """``_accept_block``'s return with its new sequence lengths filled in when it returned none: the expression the
    method evaluates right after the call (``prefix_lens + commit_lens.to(prefix_lens.dtype)``), so the values and the
    dtype are the method's own."""
    if not (isinstance(out, tuple) and len(out) >= 5 and out[4] is None):
        return out
    pl, cl = kwargs.get("prefix_lens"), out[1]
    if not isinstance(pl, torch.Tensor) or not isinstance(cl, torch.Tensor):
        return out
    return out[:4] + (pl + cl.to(pl.dtype),) + out[5:]


def _call_fold(original, *args, **kwargs):
    return _fold_nsl(original(*args, **kwargs), kwargs)


def _member_call(member, original, args, kwargs):
    step = getattr(_tls, "step", None)
    if step is None:
        return original(*args, **kwargs)
    seg = MEMBER_SEGMENT[member]
    run = step.runs.get(seg)
    if run is None:
        try:
            prof = _PROF_ON
            t0 = time.perf_counter_ns() if prof else 0
            if seg in ARG_SEGMENTS:
                step.argloc = {
                    _ARGNAMES[i] if i < len(_ARGNAMES) else f"@{i}": a
                    for i, a in enumerate(args)
                }
                for k, v in kwargs.items():
                    step.argloc["@" + k] = v
            else:
                step.argloc = {}
            loc = step.frame.f_locals if _widthed(seg) else None
            step.key = _step_key(step.worker, step.batch, seg, loc, step.base_key)
            if prof:
                t0 = _padd("key:" + seg, t0)
            run = _open(step, seg)
            if prof:
                _padd("open:" + seg, t0)
        except Exception as e:  # noqa: BLE001 - a decision error leaves this segment on the stock path
            if getattr(_tls, "cap", None) is not None:
                _tls.cap = None
            bed.count(NAME, f"open error:{seg}:{type(e).__name__}")
            logger.warning(
                "muse_vr_plugin: %s open error (%s: %s)", NAME, type(e).__name__, e
            )
            run = _Run(step, seg, "eager")
        step.runs[seg] = run
    last = member == SEGMENTS[seg][1]
    if run.mode != "replay" and seg in PURE_SEGMENTS:
        _state.get("pure_last", {}).pop(
            seg, None
        )  # its outputs are written by these calls, not from the graph
    if run.mode in ("eager", "observe"):
        if not last or getattr(run, "watch", None) is None:
            return original(*args, **kwargs)
        out = original(*args, **kwargs)
        _note_written(run)
        return out
    if run.mode == "verify":
        out = original(*args, **kwargs)
        if member == "accept" and _fold_on():
            out = _fold_nsl(out, kwargs)
        return _verify_call(run, member, out, last)
    if run.mode == "replay":
        members = run.entry["members"]
        if run.cursor >= len(members) or members[run.cursor] != member:
            raise RuntimeError(
                f"{NAME}: segment {seg} was replayed but the method called {member} (expected "
                f"{members[run.cursor] if run.cursor < len(members) else 'no further call'})"
            )
        out = run.entry["results"][run.cursor]
        run.cursor += 1
        if _PROF_ON:
            t0 = time.perf_counter_ns()
            out = _shallow(out)
            _padd("shallow:" + seg, t0)
            return out
        return _shallow(out)
    # capture
    run.cap.member_start(member)
    args = _rewrite(tuple(args), run.subst)
    kwargs = {k: _rewrite(v, run.subst) for k, v in kwargs.items()}
    if (
        not last
        and LATE_MEMBERS.get(seg) == member
        and not run.calls
        and _late_on()
        and getattr(run, "late", None) is None
    ):
        # recorded after the segment's next call (see ENV_LAUNCH_LATE); its own position is marked by an event
        ev0 = torch.cuda.Event()
        ev0.record()
        dev = torch.cuda.current_stream().device
        run.late = (original, args, kwargs, ev0, dev)
        run.calls.append((member, True))
        run.redo.append((original, args, kwargs, True, _consumable_state()))
        run.cap.member_end()
        return True
    recorder = _SetattrRecorder(
        [o for o in _owner_objects(member, args) if hasattr(o, "__dict__")]
    )
    saved = (
        _consumable_state()
    )  # what this call sees (code between the calls may have set some of it)
    fold = member == "accept" and _fold_on()
    call = functools.partial(_call_fold, original) if fold else original
    try:
        with recorder:
            out = call(*args, **kwargs)
            if getattr(run, "late", None) is not None:
                _run_late(run)
    except Exception as e:  # noqa: BLE001
        run.late = None
        _abort_capture(run, e)
        run.mode = "eager"
        _restore_state(saved)
        return original(*args, **kwargs)
    run.obj_delta += recorder.rec
    run.calls.append((member, _shallow(out)))
    run.redo.append((call, args, kwargs, out, saved))
    run.cap.member_end()
    if last:
        try:
            _end_capture(run)
            run.mode = "done"
        except Exception as e:  # noqa: BLE001
            _abort_capture(run, e)
            run.mode = "eager"
    return out


VERIFY_ORDER = ("E", "P", "D", "V", "A", "B")
EXTRA_OUTPUTS = {  # tensors a segment writes in place and does not return (compared by VERIFY)
    "P": (
        ("self", "_draft_sampler", "temperatures"),
        ("self", "_draft_sampler", "greedy_mask"),
    ),
}


def _verify_pick(step, seg) -> bool:
    """VERIFY checks one segment per step, in rotation (the step counter advances identically on every rank)."""
    order = [x for x in VERIFY_ORDER if x in _segments_on() and x in SEGMENTS]
    # decided from the step counter only (the same on every rank): a check doubles the segment's collectives, so the
    # decision must never depend on rank-local state such as which graphs exist
    return bool(order) and seg == order[step.vstep % len(order)]


def _same(a, b) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)


def _draft_kv_buffers(worker) -> list:
    """(name, tensor) of the draft KV pool's per-layer buffers (lists of CUDA tensors on the pool or a nested pool)."""
    pool = getattr(
        getattr(worker, "draft_model_runner", None), "token_to_kv_pool", None
    )
    out, todo, seen = [], [("pool", pool, 0)], set()
    while todo:
        name, obj, depth = todo.pop()
        if obj is None or id(obj) in seen or depth > 2:
            continue
        seen.add(id(obj))
        for k, v in list(getattr(obj, "__dict__", {}).items()):
            if (
                isinstance(v, (list, tuple))
                and v
                and all(
                    isinstance(x, torch.Tensor) and x.is_cuda and x.dim() >= 2
                    for x in v
                )
            ):
                out += [(f"{name}.{k}[{i}]", x) for i, x in enumerate(v)]
            elif (
                hasattr(v, "__dict__")
                and not isinstance(v, (torch.Tensor, torch.nn.Module))
                and "pool" in k
            ):
                todo.append((f"{name}.{k}", v, depth + 1))
    return out


def _kv_rows(worker, loc) -> dict | None:
    try:
        idx = loc.reshape(-1).long()
        return {n: b.index_select(0, idx) for n, b in _draft_kv_buffers(worker)}
    except Exception:  # noqa: BLE001
        return None


def _verify_open(step, seg, entry):
    """Debug (VERIFY): replay the graph and keep what it produced, put back every tensor it reads in place, then let the
    method's calls run eagerly from that state and the same generator state (the step is the stock step) and compare
    their returns, the in-place tensors and (segment B) the draft KV rows written."""
    torch.cuda.synchronize()
    loc = _loc(step)
    keep = []
    staged_now = []
    for path, staged, _ in entry["inputs"]:
        t = _fetch(path, loc, step.worker)
        if not isinstance(t, torch.Tensor):
            continue
        if not staged:
            keep.append(("/".join(map(str, path)), t))
        else:  # the method's own tensor behind a staged input: the segment must not write it (checked at the end)
            staged_now.append(("/".join(map(str, path)), t, t.clone()))
    for path in EXTRA_OUTPUTS.get(seg, ()):
        t = _fetch(path, loc, step.worker)
        if isinstance(t, torch.Tensor):
            keep.append(("/".join(map(str, path)), t))
    if (
        seg in ("D", "V")
    ):  # the runner's static buffers and the backends' metadata tensors the staging writes
        for o in _owner_objects("load_" + seg.lower(), (loc.get("@0"),)):
            a = []
            _leaves(
                getattr(o, "forward_metadata", None)
                if o is not loc.get("@0")
                else getattr(o, "buffers", None),
                (type(o).__name__,),
                a,
            )
            keep += [("/".join(map(str, p)), t) for p, t in a]
    seen, uniq = set(), []
    for n, t in keep:
        if id(t) not in seen:
            seen.add(id(t))
            uniq.append((n, t))
    keep = uniq
    pre = [(n, t, t.clone()) for n, t in keep]
    rng = torch.cuda.get_rng_state()
    entry["graph"].replay()
    torch.cuda.synchronize()
    ref = []
    for r in entry["results"]:
        a = []
        _leaves(r, (), a)
        ref.append({p: t.clone() for p, t in a})
    post = {n: t.clone() for n, t, _ in pre}
    rows = None
    if "write_ids" in entry["members"]:
        rows = _kv_rows(
            step.worker, entry["results"][entry["members"].index("write_ids")]
        )
    for _, t, old in pre:
        t.copy_(old)
    torch.cuda.set_rng_state(rng)
    run = _Run(step, seg, "verify", entry)
    run.vref, run.vpre, run.vpost, run.vrows, run.vloc = ref, pre, post, rows, None
    run.vstaged = staged_now
    if (
        _vlog_left() > 0
    ):  # debug log of mismatches: the replay's result addresses and every member's eager output
        try:
            run.vptr = []
            for r in entry["results"]:
                a = []
                _leaves(r, (), a)
                run.vptr.append({p: t.data_ptr() for p, t in a})
            run.vseen = {}
        except Exception as e:  # noqa: BLE001 - debug only
            bed.count(NAME, f"debug log error:{type(e).__name__}")
    _tally(f"verify:{seg}:bs={step.key[0]}")
    return run


ENV_VERIFY_LOG = "MUSE_VR_STEP_GRAPHS_VERIFY_LOG"  # debug (with VERIFY): log the first N mismatches per process


def _vlog_left() -> int:
    n = _state.get("vlog")
    if n is None:
        n = _state["vlog"] = (
            bed.env_int(ENV_VERIFY_LOG, 0) if bed.flag(NAME + "_verify") else 0
        )
    return n


def _small(t):
    return (
        t.reshape(-1).tolist()
        if isinstance(t, torch.Tensor) and t.numel() <= 64
        else None
    )


def _vts_scratch() -> list:
    """Debug: the per-stream scratch dicts of the top-k sampling component (if loaded), as they are now."""
    vts = sys.modules.get(_PKG + ".verify_topk_sample")
    st = getattr(vts, "_state", None) if vts is not None else None
    return list(st.get("scratch", {}).values()) if isinstance(st, dict) else []


def _vts_summary(scs) -> list:
    """Debug (syncs): per scratch dict, its capacity, the address of its zero rows, how many zero-row entries and arrival
    counters are not zero (both must be zero between calls) and the zero rows' minimum."""
    out = []
    for sc in scs:
        try:
            z, a = sc["zero"], sc["arrived"]
            out.append(
                {
                    "cap": sc.get("cap"),
                    "zero_ptr": z.data_ptr(),
                    "zero_nz": int(z.count_nonzero()),
                    "zero_min": float(z.min()),
                    "arrived_nz": int(a.count_nonzero()),
                    "arrived": a.tolist()[:8],
                }
            )
        except Exception as e:  # noqa: BLE001 - debug only
            out.append({"error": type(e).__name__})
    return out


def _verify_log(run, member, want, got, bad, cursor) -> None:
    _state["vlog"] = _vlog_left() - 1
    step = run.step
    ints = {
        k: v
        for k, v in vars(step.worker).items()
        if type(v) is int and ("slot" in k or "cap" in k)
    }
    rec = {
        "seg": run.seg,
        "member": member,
        "key": list(step.key),
        "parity": step.parity,
        "vstep": step.vstep,
        "ints": ints,
        "bad": ["/".join(map(str, p)) for p in bad],
        "want": {
            "/".join(map(str, p)): [
                str(t.dtype),
                list(t.shape),
                run.vptr[cursor].get(p),
                _small(t),
            ]
            for p, t in want.items()
        },
        "got": {
            "/".join(map(str, p)): [
                str(t.dtype),
                list(t.shape),
                t.data_ptr(),
                _small(t),
            ]
            for p, t in got.items()
            if isinstance(t, torch.Tensor)
        },
        "seen": run.vseen,
        "ref_all": [
            {"/".join(map(str, p)): _small(t) for p, t in r.items()} for r in run.vref
        ],
        "members": list(run.entry["members"]),
    }
    if run.seg == "A":
        rec["vts_graph"] = _vts_summary(run.entry.get("dbg_vts", ()))
        rec["vts_now"] = _vts_summary(_vts_scratch())
    import json

    logger.warning(
        "muse_vr_plugin: %s VERIFY-LOG %s", NAME, json.dumps(rec, default=str)
    )


ENV_VERIFY_RANGE = "MUSE_VR_STEP_GRAPHS_VERIFY_RANGE"  # debug (with VERIFY_LOG): vocabulary size for the token check


def _range_check(step, seg, entry) -> None:
    """Debug: after a replay of A, B or E, the token ids it produced (bonus, out_tokens, the next draft input's bonus
    tokens, the draft block's ids) must lie in [0, vocab); log the first violations with the entry's capture record."""
    vocab = bed.env_int(ENV_VERIFY_RANGE, 0)
    torch.cuda.synchronize()
    bad = {}
    for member, r in zip(entry["members"], entry["results"]):
        a = []
        _leaves(r, (), a)
        for p, t in a:
            name = "/".join(map(str, p))
            tok = (member == "accept" and p[:1] in ((2,), (3,))) or (
                member == "next" and "bonus" in name
            )
            if tok and t.dtype in (torch.int32, torch.int64) and t.numel():
                lo, hi = int(t.min()), int(t.max())
                if lo < 0 or hi >= vocab:
                    bad[f"{member}:{name}"] = [lo, hi, t.data_ptr(), _small(t)]
    if (
        seg == "E"
    ):  # the draft block's token ids (prep writes them into its output argument)
        for name in ("@bonus_tokens", "@block_ids_out"):
            t = step.argloc.get(name)
            if (
                isinstance(t, torch.Tensor)
                and t.dtype in (torch.int32, torch.int64)
                and t.numel()
            ):
                lo, hi = int(t.min()), int(t.max())
                if lo < 0 or hi >= vocab:
                    bad[name] = [lo, hi, t.data_ptr(), _small(t)]
    if seg in (
        "D",
        "V",
    ):  # the runner's static token ids for this batch (the draft block / verify tokens)
        bufs = getattr(step.argloc.get("@0"), "buffers", None)
        t = getattr(bufs, "input_ids", None)
        n = step.key[0] * int(getattr(step.worker, "block_size", 1) or 1)
        if isinstance(t, torch.Tensor) and t.numel() >= n:
            t = t[:n]
            lo, hi = int(t.min()), int(t.max())
            if lo < 0 or hi >= vocab:
                bad["runner:input_ids"] = [lo, hi, t.data_ptr(), _small(t)]
    for member, r in zip(
        entry["members"], entry["results"]
    ):  # non-finite floats in any returned tensor
        a = []
        _leaves(r, (), a)
        for p, t in a:
            if (
                t.is_floating_point()
                and t.numel()
                and not bool(torch.isfinite(t).all())
            ):
                bad[f"{member}:{'/'.join(map(str, p))}:nonfinite"] = [
                    str(t.dtype),
                    list(t.shape),
                    t.data_ptr(),
                ]
    if "write_ids" in entry["members"]:  # the draft KV rows this replay wrote
        rows = _kv_rows(
            step.worker, entry["results"][entry["members"].index("write_ids")]
        )
        for n, t in (rows or {}).items():
            if t.is_floating_point() and not bool(torch.isfinite(t.float()).all()):
                bad[f"kv_rows:{n}:nonfinite"] = [str(t.dtype), list(t.shape)]
    if bad:
        _state["vlog"] = _vlog_left() - 1
        import json

        logger.warning(
            "muse_vr_plugin: %s RANGE-LOG %s",
            NAME,
            json.dumps(
                {
                    "seg": seg,
                    "key": list(step.key),
                    "parity": step.parity,
                    "vstep": step.vstep,
                    "bad": bad,
                    "cap": entry.get("caprec"),
                },
                default=str,
            ),
        )


def _capture_log(run, step, delta) -> None:
    rp = []
    for c in run.calls:
        a = []
        _leaves(c[1], (), a)
        rp.append({"/".join(map(str, p)): t.data_ptr() for p, t in a})
    rec = {
        "seg": run.seg,
        "key": list(step.key),
        "parity": step.parity,
        "vstep": step.vstep,
        "ints": {
            k: v
            for k, v in vars(step.worker).items()
            if type(v) is int and ("slot" in k or "cap" in k)
        },
        "delta": delta,
        "results": rp,
        "pins": len(_state["entries"][_ekey(step, run.seg)].get("pins", ())),
        "vts": [
            (sc.get("cap"), sc["zero"].data_ptr())
            for sc in _state["entries"][_ekey(step, run.seg)].get("dbg_vts", ())
            if "zero" in sc
        ],
    }
    _state["entries"][_ekey(step, run.seg)]["caprec"] = rec
    import json

    logger.warning(
        "muse_vr_plugin: %s CAPTURE-LOG %s", NAME, json.dumps(rec, default=str)
    )


def _verify_call(run, member, out, last):
    members = run.entry["members"]
    if run.cursor >= len(members) or members[run.cursor] != member:
        bed.count(NAME, f"verify order:{run.seg}:{member}")
        run.mode = "eager"
        return out
    torch.cuda.synchronize()
    a = []
    _leaves(out, (), a)
    got = dict(a)
    want = run.vref[run.cursor]
    run.cursor += 1
    bad = [p for p, t in want.items() if p not in got or not _same(got[p], t)]
    bad += [p for p in got if p not in want]
    bed.shadow_result(f"{NAME}:{member}", not bad)
    if getattr(run, "vseen", None) is not None:
        try:
            if bad and _vlog_left() > 0:
                _verify_log(run, member, want, got, bad, run.cursor - 1)
            run.vseen[member] = {
                "/".join(map(str, p)): _small(t)
                for p, t in got.items()
                if isinstance(t, torch.Tensor)
            }
        except Exception as e:  # noqa: BLE001 - debug only
            bed.count(NAME, f"debug log error:{type(e).__name__}")
    for p in bad[:4]:
        bed.count(NAME, f"verify mismatch:{member}:{'/'.join(map(str, p))}")
    if member == "write_ids" and isinstance(out, torch.Tensor):
        run.vloc = out
    if member == "append" and run.vrows is not None and run.vloc is not None:
        now = _kv_rows(run.step.worker, run.vloc)
        badk = [
            n
            for n, t in run.vrows.items()
            if now is None or n not in now or not _same(now[n], t)
        ]
        bed.shadow_result(f"{NAME}:append:kv_rows", not badk)
        for n in badk[:4]:
            bed.count(NAME, f"verify mismatch:append:kv_rows:{n}")
    if last:
        badp = [n for n, t, _ in run.vpre if not _same(t, run.vpost[n])]
        bed.shadow_result(f"{NAME}:{run.seg}:in_place", not badp)
        for n in badp[:4]:
            bed.count(NAME, f"verify mismatch:{run.seg}:in_place:{n}")
        # a staged input the eager calls changed: a replay would have written only the private copy
        bads = [
            n
            for n, t, old in (getattr(run, "vstaged", None) or [])
            if not _same(t, old)
        ]
        bed.shadow_result(f"{NAME}:{run.seg}:staged_untouched", not bads)
        for n in bads[:4]:
            bed.count(NAME, f"verify mismatch:{run.seg}:staged written:{n}")
        run.vref = run.vpre = run.vpost = run.vrows = run.vloc = run.vstaged = None
    return out


# ------------------------------------------------------------------------------------------------- driver
def _eligible(batch):
    if _ORIG.get("is_capturing", torch.cuda.is_current_stream_capturing)():
        return "graph capture"
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        return "python version"
    if torch.cuda.current_stream() == torch.cuda.default_stream():
        return "default stream"
    fm = getattr(batch, "forward_mode", None)
    if fm is None or not fm.is_decode() or getattr(batch, "is_extend_in_batch", False):
        return "not decode"
    if getattr(batch, "has_grammar", False):
        return "grammar"
    if getattr(batch, "return_logprob", False) or getattr(
        batch, "return_hidden_states", False
    ):
        return "logprob / hidden states"
    bs = len(getattr(batch, "reqs", ()) or ())
    if bs == 0 or bs > bed.env_int(ENV_MAX_BS, 8):
        return "batch size"
    return None


def _batch_key(batch):
    si = batch.sampling_info
    return (
        len(batch.reqs),
        bool(getattr(si, "is_all_greedy", False)),
        bool(getattr(si, "need_top_k_sampling", False)),
        bool(getattr(si, "need_top_p_sampling", False)),
        bool(getattr(si, "need_min_p_sampling", False)),
    )


def _step_key(worker, batch, seg=None, loc=None, base=None):
    """The segment's shape key: the batch's part (computed once per step when given), whether the worker holds a
    selector sample at this point of the step, and for the width segments the page span."""
    key = (base if base is not None else _batch_key(batch)) + (
        getattr(worker, "_selector_sample", None) is not None,
    )
    if _widthed(seg):
        key += (_page_span(worker, batch, loc),)
    return key


def _page_span(worker, batch, loc):
    """Pages the iteration's reads reach (the host-side quantity the KV read tables are sized by): from the batch's CPU
    lengths, else the draft input's reserved lengths, plus one verify block."""
    lens = getattr(batch, "seq_lens_cpu", None)
    if lens is None and loc is not None:
        lens = getattr(loc.get("draft_input"), "nxt_kv_lens_cpu", None)
    if lens is None or not isinstance(lens, torch.Tensor) or lens.numel() == 0:
        return None
    pb = _state.get("page_block")
    if pb is None:
        pb = _state["page_block"] = (
            int(getattr(getattr(worker, "model_runner", None), "page_size", 0) or 1),
            int(getattr(worker, "block_size", 0) or 0),
        )
    page, block = pb
    # a few host values: a Python max is far cheaper than a tensor reduction (which may wake a thread pool)
    top = max(lens.tolist()) if lens.numel() <= 64 else int(lens.max())
    return (int(top) + block + page - 1) // page


def _maybe_glue(worker) -> None:
    """``MUSE_VR_STEP_GRAPHS_GLUE=1``: SGLang's own per-replay attention-metadata glue graph for the draft runner.
    SGLang turns it off for the whole DFlash family because FlashInfer backends recompute plan inputs on the host every
    replay; the draft's backend here is FlashAttention (device-only metadata when the CPU lengths are off), so the
    glue is enabled for that runner only, and only for that backend class."""
    if _state.get("glue_checked"):
        return
    _state["glue_checked"] = True
    try:
        runner = getattr(
            getattr(worker, "draft_model_runner", None),
            "decode_cuda_graph_runner",
            None,
        )
        if runner is None or getattr(runner, "_metadata_glue", "absent") is not None:
            bed.report(
                NAME + ":glue",
                False,
                reason="no draft graph runner or glue already set",
            )
            return
        backend = runner._replay_attn_backend()
        cls = type(backend).__name__
        if (
            "FlashInfer" in cls
            or "FlashAttention" not in cls
            or getattr(backend, "needs_cpu_seq_lens", True)
        ):
            bed.report(NAME + ":glue", False, reason=f"backend {cls}")
            return
        from sglang.srt.model_executor.runner.metadata_glue_graph import (
            MetadataGlueGraph,
        )

        runner._metadata_glue = MetadataGlueGraph(runner.device)
        bed.report(NAME + ":glue", True, backend=cls)
    except Exception as e:  # noqa: BLE001
        bed.report(NAME + ":glue", False, reason=f"{type(e).__name__}: {e}")


def _ensure_runners(worker) -> None:
    """Segments D / V: find the two decode graph runners (they are built after the worker's constructor, so this runs
    at the first decode steps until both exist) and wrap ``load_batch`` once."""
    if _state.get("runners_ok") or not ({"D", "V"} & _segments_on()):
        return
    try:
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
            DecodeCudaGraphRunner,
        )

        d = getattr(
            getattr(worker, "draft_model_runner", None),
            "decode_cuda_graph_runner",
            None,
        )
        t = getattr(
            getattr(getattr(worker, "target_worker", None), "model_runner", None),
            "decode_cuda_graph_runner",
            None,
        )
        runners = {}
        if d is not None and "D" in _segments_on():
            runners[id(d)] = "load_d"
        if t is not None and "V" in _segments_on():
            runners[id(t)] = "load_v"
        n = _state["runner_checks"] = _state.get("runner_checks", 0) + 1
        if len(runners) < len({"D", "V"} & _segments_on()) and n < 64:
            return  # not built yet: look again at the next step
        _state["runners"] = runners
        _state["runners_ok"] = True
        fn = DecodeCudaGraphRunner.load_batch
        if runners and not getattr(fn, "_muse_vr_step_graphs", False):
            DecodeCudaGraphRunner.load_batch = _wrap_member(_load_member, fn)
        # the page-span key matters only where the runner's reads go through a translated (built-per-iteration) table
        tr_d = getattr(
            getattr(
                getattr(worker, "draft_model_runner", None), "kv_index_translator", None
            ),
            "is_translating",
            True,
        )
        tr_t = getattr(
            getattr(
                getattr(getattr(worker, "target_worker", None), "model_runner", None),
                "kv_index_translator",
                None,
            ),
            "is_translating",
            True,
        )
        _state["translating"] = {"D": bool(tr_d) or bool(tr_t), "V": bool(tr_t)}
        bed.report(
            NAME + ":runners",
            bool(runners),
            draft=d is not None,
            target=t is not None,
            checks=n,
            translating_draft=bool(tr_d),
            translating_target=bool(tr_t),
        )
    except Exception as e:  # noqa: BLE001
        _state["runners_ok"] = True
        bed.report(NAME + ":runners", False, reason=f"{type(e).__name__}: {e}")


def _driver(original, worker, batch, *args, **kwargs):
    prof = _PROF_ON
    t0 = t_start = time.perf_counter_ns() if prof else 0
    if bed.flag(NAME + "_glue") and not _state.get("glue_checked"):
        _maybe_glue(worker)
    why = _eligible(batch)
    if why is not None:
        bed.count(NAME, "stock:" + why)
        return original(worker, batch, *args, **kwargs)
    _ensure_runners(worker)
    if _state["graveyard"]:
        _state["graveyard"].clear()  # between steps: no capture is running
    nb = len(batch.reqs)
    if nb > _state.get("max_bs", 0):
        # a decode batch larger than any before: SGLang and the components grow (re-make) their per-batch buffers in
        # this step, so every graph captured so far may hold buffers that are about to be replaced; drop them all
        # (and their warm-up records) and capture again. The same on every rank (host-side batch size only).
        if _state["entries"]:
            bed.count(NAME, f"epoch:bs={nb}")
        for k in list(_state["entries"]):
            _bury(_state["entries"].pop(k))
        _state["warm"].clear()
        _state["max_bs"] = nb
    par = getattr(worker, "_muse_vr_step_parity", 0)
    worker._muse_vr_step_parity = 1 - par
    step = _Step(worker, par)
    step.batch = batch
    step.spec0 = getattr(batch, "spec_info", None)
    step.vstep = getattr(worker, "_muse_vr_vstep", 0)
    worker._muse_vr_vstep = step.vstep + 1
    _tls.step = step
    if prof:
        _padd("driver_pre", t0)
    try:
        return original(worker, batch, *args, **kwargs)
    finally:
        t0 = time.perf_counter_ns() if prof else 0
        _tls.step = None
        # a hand-off put back after a replay and not taken this step (its consumer replayed too) is dropped: left in
        # place, a later capture of the producer would find it and wait on an event recorded in an earlier step
        for d, snap in step.handoffs:
            for k, v in snap.items():
                if d.get(k) is v:
                    d.pop(k, None)
        step.handoffs = []
        if getattr(_tls, "cap", None) is not None:
            _tls.cap = None
            bed.count(NAME, "segment left open")
        for obj, attr, old in reversed(step.restore):
            try:
                setattr(obj, attr, old)
            except Exception:  # noqa: BLE001
                pass
        # break the step <-> run reference cycle and drop the method's frame now: left to the cyclic garbage
        # collector, every step's objects (and the frame's locals) would pile up and trigger collections mid-step
        step.runs.clear()
        step.frame = None
        step.argloc = {}
        step.restore = []
        step.fresh, step.shared, step.result_ptrs = {}, {}, {}
        if _WRAP.get("on") and getattr(_tls, "cap", None) is None:
            _remove_capture_patches()
        if prof:
            _padd("driver_post", t0)
            _padd("driver_total", t_start)
            _prof_step()


def _load_member(args, kwargs):
    """``DecodeCudaGraphRunner.load_batch``: the draft runner's full staging is segment D's call, the target runner's
    (planned ahead in ``prepare_for_verify``) segment V's; the runner's own short re-entry for a pre-planned batch is
    not a member."""
    if not args:
        return None
    fb = args[1] if len(args) > 1 else kwargs.get("forward_batch")
    need = getattr(fb, "needs_forward_metadata_init", None)
    if fb is None or need is None or not need():
        return None
    return _state.get("runners", {}).get(id(args[0]))


def _prep_member(args, kwargs):
    """``_prepare_dflash_draft_block_unchecked`` opens segment E when the step embeds the block through the embedding
    module (the replicated-table lookup used under DP attention is a plain function call: no E then)."""
    step = getattr(_tls, "step", None)
    if step is None or "E" not in _segments_on() or not _state.get("embed_ok"):
        return None
    if getattr(step.worker, "_full_embed_gpu", None) is not None:
        return None
    return "prep"


def _embed_member(args, kwargs):
    """The input-embedding module's forward is a member only right after segment E's ``prep`` in the same step."""
    step = getattr(_tls, "step", None)
    return "embed" if step is not None and step.e_open else None


def _owner_objects(member, args) -> list:
    """Objects whose Python attributes a member call sets for later code (restored on replay): the graph runner and
    its attention backends for the runners' staging calls."""
    if member not in ("load_d", "load_v") or not args:
        return []
    runner = args[0]
    out = [runner]
    try:
        backend = runner._replay_attn_backend()
        out.append(backend)
        out += list(getattr(backend, "attn_backend_list", None) or [])
    except Exception:  # noqa: BLE001
        pass
    reg = getattr(runner, "buffer_registry", None)
    if reg is not None:
        out.append(reg)
    return out


class _SetattrRecorder:
    """Records every attribute assignment made on the given objects while active (their classes' ``__setattr__`` is
    wrapped for the duration), so a replay can make the same assignments - including ones that store a value equal to
    the one already there, which a before / after comparison cannot see."""

    def __init__(self, objs):
        self.ids = {id(o) for o in objs}
        self.classes = {type(o) for o in objs}
        self.rec = []
        self.patched = []

    def __enter__(self):
        ids, rec = self.ids, self.rec
        for cls in self.classes:
            orig = cls.__setattr__

            def sa(obj, name, value, _orig=orig):
                if id(obj) in ids:
                    rec.append((weakref.ref(obj), name, value))
                _orig(obj, name, value)

            had = "__setattr__" in cls.__dict__
            try:
                cls.__setattr__ = sa
            except TypeError:  # a built-in / immutable type: its instances' assignments are not recorded
                continue
            self.patched.append((cls, had, orig))
        return self

    def __exit__(self, *exc):
        for cls, had, orig in reversed(self.patched):
            if had:
                cls.__setattr__ = orig
            else:
                try:
                    del cls.__setattr__
                except AttributeError:
                    pass
        return False


def _wrap_member(member, fn):
    @functools.wraps(fn)
    def w(*args, **kwargs):
        name = member(args, kwargs) if callable(member) else member
        if name is None:
            return fn(*args, **kwargs)
        step = getattr(_tls, "step", None)
        prof = step is not None and _PROF_ON
        t0 = time.perf_counter_ns() if prof else 0
        if step is not None and step.frame is None:
            step.frame = _driver_frame()
            if step.frame is None:
                bed.count(NAME, "stock:no driver frame")
                _tls.step = None
                return fn(*args, **kwargs)
            step.base_key = _batch_key(step.batch)
            step.key = _step_key(step.worker, step.batch, base=step.base_key)
            if prof:
                t0 = _padd("frame_key", t0)
        if name == "embed":
            step.e_open = False
        out = _member_call(name, fn, args, kwargs)
        if name == "load_v" and _reentry_on():
            _alias_planned(args, kwargs)
        if name == "prep":
            step.e_open = True
        if prof:
            seg = MEMBER_SEGMENT[name]
            run = step.runs.get(seg)
            if run is not None and run.mode == "replay":
                _padd("w:" + seg, t0)
        return out

    w._muse_vr_step_graphs = True
    return w


def _install(worker) -> None:
    if _state["installed"]:
        return
    _state["installed"] = True
    _bc_on()  # builds the batched-copy kernel now (opt-in), not inside a step
    import sglang.srt.speculative.dflash_worker_v2 as dw
    from sglang.srt.layers.logits_processor import LogitsProcessor
    from sglang.srt.managers.overlap_utils import FutureMap

    W = type(worker)
    _state["driver_code"] = inspect.unwrap(
        dw.DFlashWorkerV2.forward_batch_generation
    ).__code__
    current = W.forward_batch_generation

    @functools.wraps(current)
    def driver(self, batch, *args, **kwargs):
        return _driver(current, self, batch, *args, **kwargs)

    driver._muse_vr_step_graphs = True
    W.forward_batch_generation = driver
    for owner, attr, member in (
        (W, "_accept_block", "accept"),
        (W, "_draft_write_ids", "write_ids"),
        (W, "_append_target_hidden_to_draft_kv_by_loc", "append"),
        (W, "_make_next_draft_input_decode", "next"),
        (LogitsProcessor, "finish_deferred_gather", "gather"),
        (LogitsProcessor, "deferred_argmax", "argmax"),
        (FutureMap, "publish", "publish"),
        (dw, "apply_dflash_verify_logits_adjustments", "adjust"),
        (getattr(dw, "_SelectorDraftSampler", None), "stage_sampling_params", "stage"),
    ):
        fn = getattr(owner, attr, None) if owner is not None else None
        if fn is not None and not getattr(fn, "_muse_vr_step_graphs", False):
            setattr(owner, attr, _wrap_member(member, fn))
    from . import sgl_compat

    if sgl_compat.layout() == "main" and not getattr(
        W._accept_block, "_muse_vr_forcing_after", False
    ):
        W._accept_block = _forcing_after(
            W._accept_block
        )  # outermost: the forcing runs after segment A, eagerly
    stmt = getattr(dw, "apply_dflash_simulated_acceptance", None)
    if (
        stmt is not None
        and _sim_fold_on()
        and not getattr(stmt, "_muse_vr_sim_fold", False)
    ):
        dw.apply_dflash_simulated_acceptance = _sim_statement_hook(
            stmt
        )  # the sim fold (both call sites)
    acc = getattr(dw, "accept_sampling", None)
    if (
        acc is not None
        and _accept_triton_on()
        and not getattr(acc, "_muse_vr_accept_triton", False)
    ):
        try:  # the selector-path accept on SGLang's Triton variant (see ENV_ACCEPT_TRITON)
            import sglang.kernels.ops.speculative.dspark.dspark_accept as da
            from sglang.kernels.ops.speculative.dspark.dispatch import inputs_on_cuda

            dw.accept_sampling = _accept_hook(
                acc, da.accept_sampling_triton, inputs_on_cuda
            )
        except Exception as e:  # noqa: BLE001 - stays stock
            _state["accept_triton_on"] = False
            bed.report(
                NAME + ":accept_triton", False, reason=f"{type(e).__name__}: {e}"[:300]
            )
    if "E" in _segments_on():
        try:
            fn = getattr(dw, "_prepare_dflash_draft_block_unchecked", None)
            em = dw.unwrap_lora_layer(
                dw._resolve_dflash_embedding_module(
                    worker.draft_model, worker.target_worker.model_runner.model
                )
            )
            if fn is None or "forward" in vars(em):
                raise RuntimeError(
                    "no block preparation function or an instance-level embedding forward"
                )
            cls = type(em)
            if not getattr(fn, "_muse_vr_step_graphs", False):
                dw._prepare_dflash_draft_block_unchecked = _wrap_member(
                    _prep_member, fn
                )
            if not getattr(cls.forward, "_muse_vr_step_graphs", False):
                cls.forward = _wrap_member(_embed_member, cls.forward)
            _state["embed_ok"] = True
            bed.report(NAME + ":E", True, embed=cls.__name__)
        except Exception as e:  # noqa: BLE001 - segment E stays off
            bed.report(NAME + ":E", False, reason=f"{type(e).__name__}: {e}")
    try:
        from . import ctx_proj_overlap

        if not getattr(ctx_proj_overlap.launch, "_muse_vr_step_graphs", False):
            ctx_proj_overlap.launch = _wrap_member("launch", ctx_proj_overlap.launch)
    except Exception:  # noqa: BLE001
        pass
    _maybe_merge_ab()
    global _PROF_ON
    _PROF_ON = _prof_on()
    bed.report(
        NAME + ":installed",
        True,
        worker=W.__name__,
        warmup=bed.env_int(ENV_WARMUP, 3),
        max_bs=bed.env_int(ENV_MAX_BS, 8),
        observe=bed.flag(NAME + "_observe"),
        verify=bed.flag(NAME + "_verify"),
    )


# opt-in (with MUSE_VR_STEP_GRAPHS_SHARED=1, acceptance not simulated): segment A runs on through B's calls as ONE graph.
# Once A returns the new sequence lengths, the method launches nothing between A's last call and B's first, so the two
# graphs become one (one launch boundary less per step). While acceptance is simulated the forcing block runs between
# them and A / B stay apart. A capture still refuses anything recorded between two calls of the segment.
ENV_MERGE_AB = "MUSE_VR_STEP_GRAPHS_MERGE_AB"
_SAVED_AB: dict = {}


def _maybe_merge_ab() -> bool:
    if (
        bed.env_int(ENV_MERGE_AB, 0) != 1
        or not _fold_on()
        or "B" not in SEGMENTS
        or not {"A", "B"} <= _segments_on()
    ):
        return False
    _SAVED_AB.update(
        segments=dict(SEGMENTS),
        persistent=dict(PERSISTENT_PATHS),
        members=dict(MEMBER_SEGMENT),
    )
    a, b = SEGMENTS["A"], SEGMENTS.pop("B")
    SEGMENTS["A"] = (a[0] + b[0], b[1], a[2] + tuple(x for x in b[2] if x not in a[2]))
    PERSISTENT_PATHS["A"] = tuple(
        dict.fromkeys(PERSISTENT_PATHS.get("A", ()) + PERSISTENT_PATHS.pop("B", ()))
    )
    MEMBER_SEGMENT.clear()
    MEMBER_SEGMENT.update({m: s2 for s2, (ms, _, _) in SEGMENTS.items() for m in ms})
    bed.report(NAME + ":merged_ab", True, members=list(SEGMENTS["A"][0]))
    return True


def _unmerge_ab() -> None:
    """Back to separate A / B (tests)."""
    if not _SAVED_AB:
        return
    SEGMENTS.clear()
    SEGMENTS.update(_SAVED_AB["segments"])
    PERSISTENT_PATHS.clear()
    PERSISTENT_PATHS.update(_SAVED_AB["persistent"])
    MEMBER_SEGMENT.clear()
    MEMBER_SEGMENT.update(_SAVED_AB["members"])
    _SAVED_AB.clear()


def _after_init(result, worker, *args, **kwargs):
    try:
        _install(worker)
    except Exception as e:  # noqa: BLE001 - leave SGLang stock
        bed.report(NAME + ":installed", False, reason=f"{type(e).__name__}: {e}")
    return result


def observed() -> dict:
    """Observe mode: per (segment, shape), how many distinct (address, shape, stride) each input took."""
    return {
        f"{k[1]}:{k[2]}": {"/".join(map(str, p)): len(v) for p, v in ob.items()}
        for k, ob in _state["observe"].items()
    }


def register() -> bool:
    if not enabled():
        return False
    ok, why = pins_match()
    if not ok:
        bed.report(NAME, False, reason=why)
        return False
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(TARGET_INIT, _after_init, HookType.AFTER)
    bed.report(NAME, True)
    return True

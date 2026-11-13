# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""verify_mk: TensorRT-LLM's Rubin (sm_107) decode megakernel on layers 1-60 of the TP8 one-request target-verify step.

The kernel also builds for GB300 (sm_103a, ``ARCHS``): one CTA per SM there as on Rubin, laid out for 152 SMs.

Opt-in with ``MUSE_VR_VERIFY_MK=1`` (default off). The kernel and its host code are TensorRT-LLM's
(``sm103/trtllm_ab93b78``, sm103/README.md); this package runs them on SGLang's model, KV cache and TP group:

* ``facades``: TensorRT-LLM-shaped views of SGLang's decoder layers, and the replicated expert stacks;
* ``comm``: the cross-GPU workspaces on SGLang's TP group (torch symmetric memory with multicast) in place of
  TensorRT-LLM's ``McastGPUBuffer`` and MPI communicator;
* ``kv``: SGLang's paged FP8 KV buffers and page tables as the kernel's TMA maps and parameters;
* ``runtime``: ``SgMegakernel``, TensorRT-LLM's ``ServedMegakernel`` with SGLang's buffers and per-forward decision;
* ``stages``: the compare mode's per-stage check (bring-up).

``register`` adds an AROUND hook on SGLang's ``install_megakernel``, which the model runner calls after loading the
model and its post-load transforms and before sizing the KV cache, so the weight packs come out of the KV budget. On the
target model the hook:

1. checks the server (``refusals``: device, parallelism, KV cache, draft width, conflicting components, the model) and
   builds the layer facades, then agrees with every rank: any refusal or error fails the boot on every rank (an opted-in
   server never runs these layers stock);
2. builds the kernels, the workspaces and the weight packs (``runtime.SgMegakernel.prepare``, collective);
3. wraps the decoder layers' instances (an instance ``forward``, outside every class-level hook): layer 0 decides
   whether the forward runs the kernel (``begin_sg``) and then runs as before; layers 1-60 launch their kernel on such a
   forward and run as before on any other (prefill, other batch shapes);
4. adds a capture tail hook that fails the boot when the one-request verify graph is captured without the kernel.

Settings read at install:

* ``MUSE_VR_VERIFY_MK_MODE``: ``lean`` (default, the served build) or ``check`` (rl 3 and the watchdog);
* ``MUSE_VR_VERIFY_MK_BUILD=<build>`` (dev only): installs that library build instead, one of the vendored library's
  ``SG_SWEEP_MODES`` (the lever sweep: value-identical one-lever variations of the served build) or ``XPF_MODES`` (the
  expert-weight L2 prefetch prototype, ``MSO_MK_XPF``: prefetches only) or ``SG_COMBO_MODES`` (combinations of those)
  or ``SG_STAMPED_MODE`` (the served build's per-unit stamped twin); overrides ``MUSE_VR_VERIFY_MK_MODE``; the install
  log line names the build;
* ``MUSE_VR_VERIFY_MK_STAMPS=<dir>`` (dev only; needs ``MUSE_VR_VERIFY_MK_BUILD`` = the stamped twin): every layer's
  parameter blocks get a stamp buffer, and an eager forward that does not run the kernel (a prefill) writes the
  buffers - each layer's last launch, i.e. the previous verify forward - with the layers' plans to
  ``<dir>/stamps_r<rank>_<n>.pt``, at most ``MUSE_VR_VERIFY_MK_STAMPS_MAX`` (default 6) times per rank. Attribution
  only: a stamped build is never timed;
* ``MUSE_VR_VERIFY_MK_COMPARE=<n>``: checks the first n verify forwards layer by layer against the stock layers and
  keeps the stock outputs (eager forwards: ``--cuda-graph-backend-decode disabled``); ``MUSE_VR_VERIFY_MK_COMPARE_OUT``
  (a directory) gets each rank's worst errors as JSON. With the check build the compare also checks each stage of a
  layer and dumps a few layers' tensors there (``stages``; ``MUSE_VR_VERIFY_MK_COMPARE_DUMP=<i,j,...>`` picks them).
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from typing import Callable, List

import torch

from ...components import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType", "register", "AROUND"),
    "srt/kernel_plugins/nvidia/binding.py|srt/megakernels/runtime.py": (
        "install_megakernel",
        "install_megakernel(model)",
        "install_megakernel(capture_tail_hooks)",
        "install_megakernel(is_draft)",
    ),
    "srt/runtime_context.py": ("get_parallel", "get_server_args"),
    "srt/arg_groups/fields/*.py|srt/server_args.py": (
        "page_size",
        "kv_cache_dtype",
        "speculative_num_draft_tokens",
    ),
    "srt/distributed/parallel_state.py": ("get_tp_group",),
}

logger = logging.getLogger(__name__)

NAME = "verify_mk"  # env MUSE_VR_VERIFY_MK
MODE = "MUSE_VR_VERIFY_MK_MODE"  # lean (default) | check
BUILD = "MUSE_VR_VERIFY_MK_BUILD"  # dev only: one of the library's SG_SWEEP_MODES / XPF_MODES; overrides MODE
COMPARE = "MUSE_VR_VERIFY_MK_COMPARE"  # verify forwards to compare with the stock layers (0 = serve)
COMPARE_OUT = "MUSE_VR_VERIFY_MK_COMPARE_OUT"
STAMPS = "MUSE_VR_VERIFY_MK_STAMPS"  # dev only: the stamp dumps' directory (needs the stamped twin build)
STAMPS_MAX = "MUSE_VR_VERIFY_MK_STAMPS_MAX"  # dumps per rank [6]
TAPS = "MUSE_VR_VERIFY_MK_TAPS"  # dev only: the kernel writes the DFlash taps of layers 1-60 (``_wrap_taps``): 0 / 1
# SGLang's install_megakernel at its original location; sgl_compat resolves where the installed SGLang has it
TARGET = "sglang.srt.megakernels.runtime.install_megakernel"
LAYERS = 62
DRAFT_TOKENS = 8
#: Compute capability -> (build target, SMs): the kernel launches one CTA per SM, and each target's build is laid out
#: for its SM count (the vendored ``library.ARCH_CTAS``).
ARCHS = {(10, 7): ("sm_107a", 212), (10, 3): ("sm_103a", 152)}

#: Components that would spend memory on layers 1-60 of the verify step, which the kernel replaces, or change what those
#: layers write (mid_rs keeps the residual stream split by rows between layers), and moe_ld, whose expert placements the
#: kernel's expert stacks rebind; the kernel refuses a server that opts in to one of them.
CONFLICTS = ("full_attn_wide", "mid_rs", "moe_ld", "moe_tail")

# The SGLang sources whose modules and fields the facades and the per-forward checks read, at the revisions the kernel was
# measured on: SGLang 306ccf6, and 306ccf6 with sgl-project/sglang #48947, #48948, #48949 and #48950 (muse_spark.py only).
PINS = {
    "sglang/srt/models/muse_spark.py": (
        "8bafbb8802fb0d9f81c190e87b9012fca546182be08bef6c8b3db814d27bfe19",
        "59b83f90d6140cdc789785c4c5b092b1c40777d49e2920bf5bd6bc65f2f24617",
    ),
    "sglang/srt/layers/muse_spark_layers.py": "930e66da715bae576ae681f2dfd8353a36a9ca469a229d88fa3f77bc1e74cef0",
    "sglang/srt/layers/muse_spark_fp4_moe.py": "1d7676d6d086e6aa45139671e345ef32d28a81b7de1b50810d0e512ee95cc738",
    "sglang/srt/layers/attention/trtllm_mha_backend.py": (
        "07223cbf82ab5ac14fcbaa9a47a8351132991016bae66c122bf09762ed8ed646"
    ),
}


def enabled() -> bool:
    return bed.enabled(NAME)


def pins_match() -> tuple[bool, str]:
    for rel, want in PINS.items():
        if not bed.pinned(rel, want):
            return False, f"{rel} differs from the pinned revisions"
    return True, "ok"


def register() -> bool:
    if not enabled():
        return False
    ok, why = pins_match()
    if not ok:
        bed.report(NAME, False, reason=why)
        return False
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    from ...components import sgl_compat

    target = sgl_compat.target(TARGET)
    HookRegistry.register(target, _around_install, HookType.AROUND)
    bed.report(
        NAME,
        True,
        target=target,
        mode=os.environ.get(MODE, "lean"),
        build=os.environ.get(BUILD, ""),
        compare=os.environ.get(COMPARE, "0"),
    )
    return True


def build_mode(library, runtime) -> str:
    """The library build to install: ``MUSE_VR_VERIFY_MK_BUILD`` when set (one of ``library.SG_SWEEP_MODES``,
    ``library.XPF_MODES``, ``library.SG_COMBO_MODES`` or ``library.SG_STAMPED_MODE``), else the served lean build or,
    with ``MUSE_VR_VERIFY_MK_MODE=check``, its check twin. ``MUSE_VR_VERIFY_MK_STAMPS`` needs the stamped twin."""
    build = os.environ.get(BUILD, "").strip()
    stamps = bool(os.environ.get(STAMPS, "").strip())
    if stamps and build != library.SG_STAMPED_MODE:
        raise ValueError(
            f"{STAMPS} needs {BUILD}={library.SG_STAMPED_MODE} (the stamped twin), not {build!r}"
        )
    if build:
        if build not in (
            library.SG_SWEEP_MODES
            + library.XPF_MODES
            + library.SG_COMBO_MODES
            + [library.SG_STAMPED_MODE]
        ):
            raise ValueError(
                f"{BUILD}={build!r}: not one of the library's SG_SWEEP_MODES, XPF_MODES, SG_COMBO_MODES or "
                "SG_STAMPED_MODE"
            )
        return build
    return (
        runtime.CHECK_MODE if os.environ.get(MODE, "lean") == "check" else runtime.MODE
    )


def _around_install(original: Callable, *args, **kwargs):
    out = original(
        *args, **kwargs
    )  # SGLang's own megakernel API (off unless --enable-nvidia-megakernel)
    if not kwargs.get("is_draft", True):
        install(kwargs["model"], kwargs["capture_tail_hooks"])
    return out


def refusals(lm) -> List[str]:
    """Why this server cannot take the kernel (empty: it can). Host-only and the same on every rank."""
    from sglang.srt.runtime_context import get_parallel, get_server_args

    reasons = []
    cap = tuple(torch.cuda.get_device_capability())
    if cap not in ARCHS:
        reasons.append(
            f"compute capability {cap} (the kernel is built for "
            + " and ".join(target for target, _ in ARCHS.values())
            + ")"
        )
    else:
        target, sms = ARCHS[cap]
        got = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        if got != sms:  # one CTA per SM: the build's plan holds exactly this many
            reasons.append(f"{got} SMs (the {target} build is laid out for {sms})")
    par = get_parallel()
    sizes = dict(
        tp=par.tp_size, ep=par.moe_ep_size, attn_tp=par.attn_tp_size, pp=par.pp_size
    )
    if sizes != dict(tp=8, ep=8, attn_tp=8, pp=1):
        reasons.append(
            f"parallelism {sizes} (the kernel runs TP8 = EP8, no attention DP, no PP)"
        )
    elif (
        par.moe_ep_rank != par.tp_rank
    ):  # the own expert window is the TP rank's (facades.fill_expert_stacks)
        reasons.append(f"EP rank {par.moe_ep_rank} on TP rank {par.tp_rank}")
    args = get_server_args()
    page, kv_dtype = (
        getattr(args, "page_size", None),
        str(getattr(args, "kv_cache_dtype", None)),
    )
    if page != 128 or kv_dtype != "fp8_e4m3":
        reasons.append(
            f"KV cache {kv_dtype} with pages of {page} (the kernel reads fp8_e4m3 pages of 128)"
        )
    draft = getattr(args, "speculative_num_draft_tokens", None)
    if draft != DRAFT_TOKENS:
        reasons.append(
            f"{draft} draft tokens (the kernel verifies {DRAFT_TOKENS} rows)"
        )
    conflicts = [f"MUSE_VR_{name.upper()}" for name in CONFLICTS if bed.enabled(name)]
    if conflicts:
        reasons.append(
            "opted in to a component this kernel replaces: " + " ".join(conflicts)
        )
    if lm is None:
        reasons.append("no MuseSpark1xModel in the model")
    elif len(lm.layers) != LAYERS:
        reasons.append(f"{len(lm.layers)} decoder layers")
    return reasons


def _text_model(model: torch.nn.Module):
    for module in model.modules():
        if type(module).__name__ == "MuseSpark1xModel":
            return module
    return None


def install(model: torch.nn.Module, capture_tail_hooks: list) -> None:
    """Build the kernel for ``model``'s text decoder on every TP rank and route its verify forward through it, or
    raise on every rank. Collective over the TP group."""
    from sglang.srt.distributed.parallel_state import get_tp_group

    from ..trtllm_ab93b78.megakernel import library, serving
    from . import comm as sg_comm
    from . import facades, runtime

    lm = _text_model(model)
    if getattr(lm, "_verify_mk", None) is not None:
        return
    group = get_tp_group()
    comm = sg_comm.TpComm(group.cpu_group)
    rank = comm.Get_rank()
    mode = _phase(comm, rank, "build mode", lambda: build_mode(library, runtime))

    # Each phase ends in an agreement, so a rank that fails alone raises with all the others, and the collective phases
    # (the expert stacks, the build) start only when every rank is ready for them.
    layers = _phase(comm, rank, "checks", lambda: _checked_facades(lm, facades))
    free_before = torch.cuda.mem_get_info()[0]
    stacks = _phase(
        comm,
        rank,
        "expert stacks",
        lambda: facades.fill_expert_stacks(layers, group.device_group, rank),
    )
    capability = tuple(torch.cuda.get_device_capability())
    mapping = runtime.mapping_of(rank, group.local_rank)
    _phase(
        comm,
        rank,
        "kernel checks",
        lambda: _raise_on(serving.refusals(capability, mapping, layers)),
    )
    # Every rank loads the library before the vendored prepare (rank 0 builds there, then a barrier): one rank builds
    # under torch's file lock in TORCH_EXTENSIONS_DIR while the others wait outside any collective (gloo times out).
    _phase(comm, rank, "compile", lambda: library.library(mode))
    compare = int(os.environ.get(COMPARE, "") or 0)
    mk = runtime.SgMegakernel(
        layers,
        rank,
        group.local_rank,
        comm,
        mode=mode,
        compare=compare,
        compare_out=os.environ.get(COMPARE_OUT),
        stamps_dir=os.environ.get(STAMPS, "").strip() or None,
        stamps_max=int(os.environ.get(STAMPS_MAX, "") or 6),
    )
    mk.prepare()  # collective; it raises alike on every rank or hangs in the vendored barrier
    used = (free_before - torch.cuda.mem_get_info()[0]) / 2**30
    _phase(
        comm,
        rank,
        "build",
        lambda: _raise_on([] if mk.kernels is not None else ["not built"]),
    )
    taps = _wrap_taps(lm, mk) if taps_on() and not compare else None
    _wrap_layers(lm, mk, taps)
    capture_tail_hooks.append(_capture_check(mk))
    lm.__dict__["_verify_mk"] = (
        mk  # not an nn.Module attribute: nothing joins the state dict
    )
    bed.report(
        NAME + ":install",
        True,
        layers=f"{mk.indices[0]}-{mk.indices[-1]}",
        mode=mode,
        compare=compare,
        taps=sorted(mk.taps) if taps is not None else "off",
    )
    logger.info(
        "[verify_mk] rank %d: installed on layers %d-%d (%s, compare %d, kernel taps %s): %.2f GiB device memory, "
        "expert replicas %.2f GiB (%.2f expected)",
        rank,
        mk.indices[0],
        mk.indices[-1],
        mode,
        compare,
        sorted(mk.taps) if taps is not None else "off",
        used,
        stacks["grown_gib"],
        stacks["expected_gib"],
    )


def _raise_on(reasons: List[str]) -> None:
    if reasons:
        raise ValueError("; ".join(reasons))


def _checked_facades(lm, facades):
    _raise_on(refusals(lm))
    return facades.text_layers(lm)


def _phase(comm, rank: int, what: str, fn: Callable):
    """``fn()`` on every rank, then an agreement: any rank's exception raises on every rank."""
    from . import comm as sg_comm

    out, error = None, ""
    try:
        out = fn()
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        logger.exception("[verify_mk] rank %d: %s", rank, what)
    sg_comm._agree(comm, error, f"[verify_mk] {what}")
    return out


def _wrap_layers(lm, mk, taps=None) -> None:
    """Instance ``forward`` of layers 0..60 (``MuseSpark1xDecoderLayer.forward``'s arguments and result); ``taps``: the
    state of ``_wrap_taps``, which layer 0 tells whether this forward runs the kernel."""
    first = lm.layers[0]
    original0 = first.forward
    compare = mk.compare_forwards > 0
    begin = mk.begin_compare if compare else mk.begin_sg

    def layer0_forward(positions, stream, attn_input, forward_batch):
        ran = begin(forward_batch, positions, stream, attn_input)
        if taps is not None:
            taps.on = bool(ran)
        return original0(positions, stream, attn_input, forward_batch)

    first.forward = layer0_forward
    for index in mk.indices:
        layer = lm.layers[index]

        def forward(
            positions,
            stream,
            attn_input,
            forward_batch,
            _index=index,
            _original=layer.forward,
        ):
            if mk.current is None:
                return _original(positions, stream, attn_input, forward_batch)
            if compare:
                return mk.compare_layer(
                    _index, _original, positions, stream, attn_input, forward_batch
                )
            return mk.run_layer(_index, attn_input, stream)

        layer.forward = forward


def taps_on() -> bool:
    v = os.environ.get(TAPS, "0").strip()
    if v not in ("0", "1"):
        raise ValueError(f"{TAPS}={v!r}: 0 or 1")
    return v == "1"


class _Taps(list):
    """The text model's ``layers_to_capture`` with the kernel's taps armed: on a forward that runs the kernel
    (``state.on``), a layer the kernel taps is not in it, so the layer loop makes no copy of that layer's stream."""

    def __init__(self, layers, state):
        super().__init__(layers)
        self.state = state

    def __contains__(self, index) -> bool:
        if self.state.on and index in self.state.mk.taps:
            return False
        return super().__contains__(index)


def _wrap_taps(lm, mk) -> SimpleNamespace:
    """Dev only (``MUSE_VR_VERIFY_MK_TAPS=1``; ported from the sm_107 (Rubin) tuning work): the kernel writes the
    DFlash taps of layers 1-60 (``SgMegakernel.enable_taps``) and the text model's instance ``forward`` returns them in
    place of the layer loop's copies, each a stream-ordered kernel between two launches that ends their PDL overlap.
    Same tensors (``bf16(stream)``, rounded to nearest even in both) and the same order: the requested layers,
    ascending."""
    state = SimpleNamespace(on=False, mk=mk)
    mk.enable_taps()
    original = lm.forward

    def forward(*args, **kwargs):
        if not isinstance(
            lm.layers_to_capture, _Taps
        ):  # set_dflash_layers_to_capture assigns a plain list
            lm.layers_to_capture = _Taps(lm.layers_to_capture, state)
        state.on = False  # layer 0 sets it when this forward runs the kernel
        out = original(*args, **kwargs)
        on, state.on = state.on, False
        requested = sorted(set(lm.layers_to_capture))
        if not on or not any(index in mk.taps for index in requested):
            return out
        hidden, aux = out if isinstance(out, tuple) else (out, [])
        rest = iter(aux)  # the loop's own taps, ascending
        return hidden, [
            mk.tap(index) if index in mk.taps else next(rest) for index in requested
        ]

    lm.forward = forward
    return state


def _capture_check(mk) -> Callable:
    """A capture tail hook (``capture_hook(runner, out, forward_batch, num_tokens)``, after every warm-up and capture
    forward): the one-request verify graph must run the kernel. Every rank decides alike, so every rank raises."""

    def check(runner, out, forward_batch, num_tokens) -> None:
        mode = forward_batch.forward_mode
        if (
            mode.is_target_verify()
            and forward_batch.batch_size == 1
            and num_tokens == DRAFT_TOKENS
        ):
            if mk.last_refusal:
                raise RuntimeError(
                    f"[verify_mk] the one-request verify graph would run layers 1-60 stock: {mk.last_refusal}"
                )

    return check

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``nvfp4_kv``: NVFP4 target KV cache with a DFLASH draft, hybrid sliding-window pools and HiCache.

Serve with ``--kv-cache-dtype nvfp4`` (target attention ``trtllm_mha``, native NVFP4 prefill and decode), keep the
DFLASH draft on ``--speculative-draft-attention-backend fa4`` and set ``SGLANG_ENABLE_POST_CAPTURE_KV_SIZING=0``.
The DFLASH draft keeps its own KV pool in the model dtype (BF16): SGLang already gives an fa4 draft a compute-dtype
pool whenever the target KV dtype differs (``configure_kv_cache_dtype``); nothing here changes the draft pool.

Opt in with ``MUSE_VR_NVFP4_KV=1``. Parts, each reported in the engagement record:

``fi_gate``
    FlashInfer 0.7.1 refuses packed-FP4 KV caches on some devices with a Python device check (an ``if`` on the
    compute capability that raises ``ValueError``) at the top of the packed-FP4 KV branch of
    ``trtllm_batch_decode_with_kv_cache`` and ``trtllm_batch_context_with_kv_cache``, although the trtllm-gen
    E2M1-KV kernels for those devices ship with it. This part rebuilds both functions in-process from the installed
    FlashInfer source with exactly that check removed (nothing else changes: same module globals, same decorators,
    same kernels) and registers them in place of the originals, so SGLang's trtllm_mha backend calls the rebuilt
    functions for every KV dtype. The installed files are not edited; each rebuild first checks the module file against
    its sha256 pin, and the check is matched structurally: it must be the only such check in the function, the first
    statement of the packed-FP4 KV branch and on lines of its own, or the rebuild raises instead. FlashInfer 0.7.2
    no longer has that check in either function: with a pinned 0.7.2 file that holds no such check the functions are
    used as installed (no rebuild, no replacement; ``prefill_bf16_out`` still wraps the context function). Calls with FP8 or
    BF16 KV never reach the removed lines and behave as with the originals. This is a deviation next to the engine:
    two FlashInfer functions run without their device check. It is installed whenever the component is engaged,
    whatever the other switches, so every configuration of this component runs it: with the defaults
    (``verify_kernel`` and ``prefill_bf16_out`` on, with or without ``swa_fp8``) the rebuilt context function serves
    every prefill over NVFP4 KV with a BF16 output, and the rebuilt decode function serves only the NVFP4 attention
    calls the plugin kernel does not take; with the stock verify kernel and FP8 prefill output
    (``MUSE_VR_NVFP4_KV_VERIFY_KERNEL=0`` and ``MUSE_VR_NVFP4_KV_PREFILL_BF16_OUT=0``) the rebuilt decode function
    serves every target verify over NVFP4 KV and the rebuilt context function writes the stock FP8 output. Under
    ``swa_fp8`` only the full-attention layers hold NVFP4 KV; the sliding-window layers' FP8 calls pass through the
    rebuilt functions on the FP8 branch, which is unchanged.
``resolution``
    Overrides the ``handle_kv4_compatibility`` resolution step (SGLang's plugin API for named steps) with a copy
    that differs from the stock step in two places: (1) DFLASH with an explicit fa4 draft backend is accepted. The
    stock rejection exists because ``set_kv_buffer_prefix_valid`` does not write the native scale layout; DFLASH
    only calls it on the draft pool, which is BF16 for an fa4 draft, so the target never reaches it. The stock
    step would also force the draft onto ``trtllm_mha``; with fa4 the draft keeps its framework-default pool.
    (2) HiCache is accepted when the ``host_pool`` part below is installed (``page_first`` layout, ``kernel`` I/O,
    no storage backend). Every other stock check is kept.
``swa_pool``
    Hybrid sliding-window models build their ``SWAKVPool`` without the FP4 quant method, so the sub-pools fall back
    to an unquantized pool of ``float4_e2m1fn_x2`` elements and the first write fails. This part builds both
    sub-pools with the NVFP4 quant method (one instance, global-layer indexed scales), sizes them with the NVFP4
    bytes per token (packed data + native block scales instead of one byte per element), and lets the trtllm_mha
    backend find the native scale buffers of the sub-pool that owns a layer.
``host_pool``
    HiCache host pool for the native NVFP4 device pool (``nvfp4_kv_host``): packed K/V rows ride the stock MHA host
    pool at ``head_dim / 2`` bytes per head, and the per-page native block-scale tiles move as whole-page rows with
    the same staged JIT kernels (the stock MXFP8 host pool's scheme). SGLang main builds host pools from the device
    pool's buffer description: there the part makes a native NVFP4 pool describe packed rows
    (``MHATokenToKVPool._mha_buffer_info``) and hands the host class its native scale tiles from the host-pool
    assembler (``hybrid_pool_assembler._mha_host_inputs``); the branch layout keeps the device-pool hooks.
``verify_kernel`` (``MUSE_VR_NVFP4_KV_VERIFY_KERNEL``, default 1; module ``nvfp4_kv_verify``)
    DFLASH target verify over the native NVFP4 pool runs the plugin's tensor-core kernel (``ops/nvfp4_verify_attn``):
    the stock E2M1-KV generation kernels serve one verify token per CTA and stream the KV 8 times per step.
``fused_store`` (``MUSE_VR_NVFP4_KV_FUSED_STORE``, default 1; kernel ``ops/nvfp4_kv_store``)
    The NVFP4 KV write (``NVFP4KVCacheMethod.quantize_and_store``: quantize K, copy V to contiguous rows, quantize V,
    scatter rows + native scales = four launches per layer) runs as one kernel that reads the strided projection
    rows directly and writes the same bytes (bit-identical; tests/test_nvfp4_kv_store.py). Writes the kernel does not
    cover (linear scale buffers, other dtypes or layouts) take the stock path. ``MUSE_VR_NVFP4_KV_STORE_PDL`` (default
    1) launches it with programmatic dependent launch (it reads nothing before griddepcontrol.wait).
``kv_scales`` (``MUSE_VR_NVFP4_KV_SCALES``: unset/0 = off, 1 = the packaged file, or a path; module ``nvfp4_kv_scales``)
    Calibrated per-layer global K/V scales replace the 1.0 defaults (the checkpoint ships none); hybrid sub-pools write
    each layer with its own scale. No scale file is packaged in this tree, so ``1`` fails to load and keeps nvfp4
    rejected; pass a path.
``swa_fp8`` (``MUSE_VR_NVFP4_KV_SWA_FP8=1``; module ``nvfp4_kv_mixed``)
    NVFP4 only on the full-attention layers: the sliding-window sub-pool is FP8 and those layers run SGLang's FP8-KV
    path (pool, KV store, kernels), selected per layer.
``prefill_dq`` (``MUSE_VR_NVFP4_PREFILL_DQ=1``, default off; module ``nvfp4_prefill_dq``)
    Prefill attention over the native NVFP4 pool runs the trtllm-gen FP8-KV context kernel over a batch-sized FP8
    workspace that one dequant kernel per layer fills (``ops/nvfp4_prefill_dq``); the E2M1-KV context kernel converts
    each K/V tile once per query tile that reads it. Batches that do not fit the workspace keep the stock call. With
    ``xattn_b`` enabled, its stand-in for the context function sits innermost, under the workspace wrapper
    (``xattn_b.compose_context``), and serves the workspace calls inside its window.
``prefill_bf16_out`` (sub-switch of ``fi_gate``, ``MUSE_VR_NVFP4_KV_PREFILL_BF16_OUT``, default 1)
    Context (prefill) attention over NVFP4 KV writes a BF16 output (trtllm-gen ships those kernels) instead of the FP8
    output the stock native path requests. The stock decode kernels keep their FP8 output (trtllm-gen has no
    BF16-output decode kernels for E2M1 KV); target verify writes BF16 through ``verify_kernel``.

Numerics, every rounding that differs from the FP8-KV recipe (FP8 E4M3 K/V at scale 1.0, FP8 E4M3 Q, BF16 output):
  * storage: K and V are E2M1 with one E4M3 block scale per 16 values, global scale 1.0 (the checkpoint ships no KV
    scales; per-layer calibrated scales with ``kv_scales``); ``fused_store`` writes the same bytes as the stock write
    and the ``host_pool`` round trip is bitwise;
  * Q is rounded to E4M3, as in the FP8-KV recipe;
  * target verify with ``verify_kernel`` (default): K and V are dequantized exactly (E2M1 x E4M3 is exact in f16) and
    rounded once to E4M3 for the FP8 MMA, the probabilities are rounded to E4M3 (as in the FP8-KV verify kernel), and
    the output is BF16. The kernel's key-split count follows the batch size and the device's SM count; different
    split counts merge the partial results in a different summation order (with one split the output is identical to
    the merged path). With ``MUSE_VR_NVFP4_KV_VERIFY_KERNEL=0`` the stock E2M1-KV kernels run and write an FP8 E4M3
    output that is then copied to BF16;
  * prefill: BF16 output with ``prefill_bf16_out`` (default); FP8 E4M3 output, copied to BF16, with
    ``MUSE_VR_NVFP4_KV_PREFILL_BF16_OUT=0``; with ``prefill_dq`` K and V are dequantized exactly and rounded once to
    E4M3 (as for ``verify_kernel``) and the FP8-KV context kernel runs (its output differs from the E2M1-KV
    kernel's in a small fraction of elements, by one bf16 ulp);
  * any other decode over NVFP4 KV (calls the plugin kernel does not take): the stock kernels, FP8 E4M3 output copied
    to BF16;
  * the DFLASH draft KV stays BF16 (unchanged).
With ``swa_fp8`` the sliding-window layers store and attend exactly as the FP8-KV recipe; the items above apply to
the full-attention layers only. These are target precision changes the accuracy ladder judges.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import logging
import os
import threading
from pathlib import Path

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
    "srt/arg_groups/resolution_hooks.py": ("register_resolution_hook",),
    "srt/utils/common.py": ("is_float4_e2m1fn_x2",),
    "srt/mem_cache/swa_memory_pool.py": ("SWAKVPool",),
}


logger = logging.getLogger("muse_vr_plugin")

NAME = "nvfp4_kv"

# sha256 of files relative to the directory that holds the ``sglang`` package.
SGLANG_PINS = {
    "sglang/srt/arg_groups/kv_cache_hook.py": (
        "cca1d08b1e06d3ada9c765c6ed93cf880024194e3a194bc90509b790d621fa97",
        "ff60f663c4b5dc147cee95cb01892bca34c870139d6d384d823087bb5567f318",
    ),
    "sglang/srt/arg_groups/resolution_hooks.py": (
        "09a255a23f107fb37792aa14dc76052ebf7a347894a39a56a2c519fdc7be5be4",
        "223cad17d8ae0be65cf4058330148fd76f7566a5e949c288006dbf228763407a",
    ),
    "sglang/srt/arg_groups/overrides.py": (
        "e7887fe00392f2eb7de91655692bf2da0a58edb9caf397fd06b137800dbaa544",
        "f3455f5465c678de9ab578c238424bb92fde4e652159056ae955cca2d8711c4a",
    ),
    "sglang/srt/mem_cache/kv_cache_configurator.py": (
        "570b22f6aca4d7bb2330fc15a5912ee2f61e500626cce21d69c27e71ec132fb2",
        "5f7f4132923880cb2aa58ef55a5b18233df20bf1ffd47eb0d27b9781a8226c67",
    ),
    "sglang/srt/mem_cache/kv_cache_dtype.py": "2ecb9e1444588f5a8d6f8ecc1b99b96b692fcfedd655c2d5533955ddd84d69a8",
    "sglang/srt/model_executor/pool_configurator.py": (
        "d6e49edf1d7f6cd23f435257b0230b97fb23d401051c36e1c3b386b24b018c2d",
        "abcadf80aa7dceb6eb1c54735132ec300f1e150d52cd9d95a6bac23620b90ac5",
    ),
    "sglang/srt/layers/attention/trtllm_mha_backend.py": (
        "566d1031d644f71a8adb84d1e2a1581d3a6d1c3cac1e507451ac4cf034c39da4",
        "07223cbf82ab5ac14fcbaa9a47a8351132991016bae66c122bf09762ed8ed646",
    ),
    "sglang/srt/mem_cache/swa_memory_pool.py": (
        "5d518102fad0bb444176815e1e829a63197b6939ff656bf282253ffa09c90fab",
        "14f2b1daebdc9d872b5786cfc0beed26a413bb41717b70a737a4cb4c31951dfc",
    ),
    "sglang/srt/mem_cache/memory_pool.py": (
        "b439441448d1112952876cd5f814ea47fcd561ece1a9aa0e7d2080d5f4874106",
        "f3e12b293f3bdce21604ca5e5b473c07dcb6c8e4a34b582c4498d0bd45a10900",
    ),
    "sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py": "9800db3ab0968c921ca66ce806f1e842b9c61d4db577358712d6fdd0466de2d2",
    "sglang/srt/layers/quantization/nvfp4_kv_cache.py": "a0511ad263e22b0d0343cb5befbbfe73fde6a72d1efb8c50b166e01078460ae8",
    "sglang/srt/speculative/dflash_worker_v2.py": (
        "7d462f5f175828bcc3b4d47fcdfa33cb6af4b14669ef016eeb79b1a5191eec8c",
        "8f82de8d666f81f3c1aa43dc651006f4182ff58593fdfe4eea28c5ee2a14bfc0",
    ),
}
HOST_POOL_PINS = {
    "sglang/srt/mem_cache/pool_host/mha.py": (
        "58962740251a936c1a940cc382fcc22eccd34d68120700ec790d3f9a6b150aa4",
        "bdc20a7643e0b98eb774a46c781cb048f579396f696169689b3616b7f0dc91b6",
    ),
    "sglang/srt/mem_cache/pool_host/mha_mxfp8.py": (
        "9a167bd9adfc1e1945ccb5eb292757eea308175a6f485cb7f6cb82b38286f394",
        "e0c0842585e4f94e95b7825ebcc80f38e7b02fd0b0684ca834275257f11bb830",
    ),
    # SGLang main only (host pools built from buffer descriptions); absent on the branch layout
    "sglang/srt/mem_cache/device_pool_info.py": (
        "absent",
        "728ac7605216fc149a47278654cd70388d5237ca3462a7b3d952303a01cb2837",
    ),
    "sglang/srt/mem_cache/pool_host/base.py": (
        "982db6b57fe90ced7d9b052f8ce965c48ccfec7f4d71ec9cdab03cbce253ad4c",
        "e57a060e2fd54905a4f16c4e440bafb8356f1967713c8a50f2667b2057d65bf6",
    ),
    "sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py": (
        "35365ec237d193ac77b5adbba9fe239cde205e3747b0da635d3491126b594ac7",
        "a5e1156a9db13593641ea9d649769761ce745fe82cf17c087a59b300d67e17ae",
    ),
    "sglang/srt/mem_cache/kv_cache_builder.py": (
        "79808d4f054dc9edbae0c5c1e2d92818442d74b8a66381f6835eaa64f135853c",
        "578ce61e11904331f6a0bf7d85bf7f5679249b4d1a6718860a9c71b44ad45147",
    ),
    "sglang/kernels/ops/kvcache/hicache.py": (
        "78ecd67d91a8a371a069db722e8f9cc2159e09f616487f659ef971acf7f73071",
        "f26aa62750f30ae9ce7a49ced671407daa97e3334288f8d335d51d666c5cc97d",
    ),
}
# sha256 of files relative to the directory that holds the ``flashinfer`` package.
FLASHINFER_PINS = {
    "flashinfer/decode.py": (
        "8917d7f1a80707d823455d69d105d90a57e8bb8a71b248348f02d62c547a6f91",  # FlashInfer 0.7.1 (device check)
        "de1f242bf2f598bc973a9d2f9625f8db1e4ccfc24085f686b2d0304c9076b20d",  # FlashInfer 0.7.2 (no device check)
    ),
    "flashinfer/prefill.py": (
        "0f1f9e03a7aa59ab9b95f270b85b240041717f3e08ee20c30d1332d5136d19a0",  # FlashInfer 0.7.1 (device check)
        "531e220624ebc543fe865b1f6f078c5646233abc287b8407ac955e26d44f73b8",  # FlashInfer 0.7.2 (no device check)
    ),
}

# (module, function) pairs SGLang's trtllm_mha backend calls. In each, the device check is the first statement of the
# packed-FP4 (uint8) KV branch: an ``if`` comparing ``get_compute_capability(...)`` with a (major, minor) tuple whose
# body raises ``ValueError``. It is found structurally (``_fp4kv_device_checks``) and its lines are dropped.
FI_GUARDED = (
    ("flashinfer.decode", "trtllm_batch_decode_with_kv_cache"),
    ("flashinfer.prefill", "trtllm_batch_context_with_kv_cache"),
)

_lock = threading.Lock()
_state: dict = {"engaged": None, "parts": {}}


def _pkg_root(pkg: str) -> Path | None:
    spec = importlib.util.find_spec(pkg)
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(list(spec.submodule_search_locations)[0]).resolve().parent


def _flashinfer_pins_ok() -> bool:
    root = _pkg_root("flashinfer")
    if root is None:
        return False
    for rel, sha in FLASHINFER_PINS.items():
        try:
            got = hashlib.sha256((root / rel).read_bytes()).hexdigest()
        except OSError:
            got = None
        if not bed.sha_ok(got, sha):
            logger.warning(
                "muse_vr_plugin: %s sha256 %s does not match the pin %s",
                rel,
                (got or "<missing>")[:16],
                bed.sha_label(sha),
            )
            return False
    return True


def _verify_kernel_enabled() -> bool:
    return os.environ.get(
        "MUSE_VR_NVFP4_KV_VERIFY_KERNEL", "1"
    ).strip().lower() not in ("0", "false", "no", "off")


def part_engaged(part: str) -> bool:
    return bool(_state["parts"].get(part))


# ----------------------------------------------------------------------------------------------- fi_gate


def _is_device_check(node: ast.AST) -> bool:
    """``if get_compute_capability(<expr>) == (<int>, <int>): raise ValueError(...)`` without an ``else``."""
    if not (isinstance(node, ast.If) and not node.orelse and len(node.body) == 1):
        return False
    test, body = node.test, node.body[0]
    if not (
        isinstance(test, ast.Compare)
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
    ):
        return False
    call, cap = test.left, test.comparators[0]
    if not (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "get_compute_capability"
    ):
        return False
    if not (
        isinstance(cap, ast.Tuple)
        and len(cap.elts) == 2
        and all(isinstance(e, ast.Constant) and type(e.value) is int for e in cap.elts)
    ):
        return False
    return (
        isinstance(body, ast.Raise)
        and isinstance(body.exc, ast.Call)
        and isinstance(body.exc.func, ast.Name)
        and body.exc.func.id == "ValueError"
    )


def _fp4kv_device_checks(fn: ast.AST) -> list:
    """Device checks (``_is_device_check``) that are the first statement of an ``if`` on a ``uint8`` KV dtype."""
    found = []
    for branch in ast.walk(fn):
        if (
            isinstance(branch, ast.If)
            and branch.body
            and _is_device_check(branch.body[0])
            and any(
                isinstance(n, ast.Attribute) and n.attr == "uint8"
                for n in ast.walk(branch.test)
            )
        ):
            found.append(branch.body[0])
    return found


def derive_without_fp4kv_device_check(modname: str, fname: str):
    """Return ``modname.fname`` re-created from its installed source with the FP4-KV device check removed.

    The module file must match its sha256 pin (``FLASHINFER_PINS``). A pinned file whose function holds no device
    check at all (FlashInfer 0.7.2 dropped it) returns the installed function itself, unchanged. The function is compiled from that file, in the
    module's own globals (so it resolves every name exactly like the original), with its decorators. The function must
    hold exactly one device check (``_is_device_check``) anywhere, as the first statement of the packed-FP4 KV branch,
    on lines of its own; exactly those lines are dropped. Anything else raises ``RuntimeError``. Line numbers before
    the removed lines are unchanged.
    """
    mod = importlib.import_module(modname)
    path = mod.__file__
    pin = FLASHINFER_PINS.get(modname.replace(".", "/") + ".py")
    got = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if pin is None or not bed.sha_ok(got, pin):
        raise RuntimeError(
            f"{modname}: {path} sha256 {got[:16]} does not match the pin "
            f"{bed.sha_label(pin) if pin else '<none>'}"
        )
    text = Path(path).read_text()
    tree = ast.parse(text)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fname]
    if len(nodes) != 1:
        raise RuntimeError(
            f"{modname}.{fname}: expected one top-level definition, found {len(nodes)}"
        )
    node = nodes[0]
    start = min([d.lineno for d in node.decorator_list] + [node.lineno])
    lines = text.splitlines(keepends=True)
    checks = _fp4kv_device_checks(node)
    every = [n for n in ast.walk(node) if _is_device_check(n)]
    if not every:
        return getattr(mod, fname)
    if len(checks) != 1 or len(every) != 1:
        raise RuntimeError(
            f"{modname}.{fname}: device check found {len(every)} times, {len(checks)} of them first in "
            "the packed-FP4 KV branch (expected 1 and 1)"
        )
    chk = checks[0]
    first, last = chk.lineno, chk.end_lineno
    # AST column offsets count UTF-8 bytes.
    if (
        lines[first - 1].encode("utf-8")[: chk.col_offset].strip()
        or lines[last - 1].encode("utf-8")[chk.end_col_offset :].strip()
    ):
        raise RuntimeError(
            f"{modname}.{fname}: device check does not occupy whole lines"
        )
    patched = "".join(lines[start - 1 : first - 1] + lines[last : node.end_lineno])
    code = compile("\n" * (start - 1) + patched, path, "exec")
    namespace: dict = {}
    exec(code, mod.__dict__, namespace)  # noqa: S102 - compiles the pinned installed source minus the check
    fn = namespace[fname]
    fn.__muse_vr_derived_from__ = getattr(mod, fname)
    return fn


def prefill_bf16_out_enabled() -> bool:
    return os.environ.get(
        "MUSE_VR_NVFP4_KV_PREFILL_BF16_OUT", "1"
    ).strip().lower() not in ("0", "false", "no", "off")


def context_with_bf16_output(context_fn):
    """Wrap the context (prefill) kernel entry: NVFP4-KV calls that ask for an FP8 output write BF16 instead.

    trtllm-gen ships context kernels with a BF16 output for E2M1 KV (decode kernels exist only with FP8 output).
    SGLang's trtllm_mha backend passes its FP8 staging buffer as ``out`` and copies it to BF16 afterwards; returning a
    BF16 tensor skips that copy (``_finalize_nvfp4_output`` returns BF16 outputs unchanged) and removes the FP8
    rounding of the prefill attention output. Only keyword calls with packed-FP4 KV and an FP8 ``out`` are changed.
    """
    import torch

    def ctx(*args, **kwargs):
        out = kwargs.get("out")
        if (
            not args
            and kwargs.get("kv_cache_sf") is not None
            and isinstance(out, torch.Tensor)
            and out.dtype == torch.float8_e4m3fn
            and kwargs.get("o_sf_scale") is None
            and not kwargs.get("return_lse", False)
        ):
            query = kwargs["query"]
            kwargs["out"] = torch.empty(
                query.shape, dtype=torch.bfloat16, device=query.device
            )
            kwargs["out_dtype"] = torch.bfloat16
        return context_fn(*args, **kwargs)

    ctx.__muse_vr_derived_from__ = getattr(
        context_fn, "__muse_vr_derived_from__", context_fn
    )
    return ctx


def _install_fi_gate() -> bool:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    derived = {}
    for modname, fname in FI_GUARDED:
        derived[(modname, fname)] = derive_without_fp4kv_device_check(modname, fname)
    upstream = {
        k
        for k, fn in derived.items()
        if fn is getattr(importlib.import_module(k[0]), k[1])
    }
    _state["fi_gate_upstream"] = len(upstream)
    key = ("flashinfer.prefill", "trtllm_batch_context_with_kv_cache")
    if _prefill_dq_requested():
        from . import nvfp4_prefill_dq

        fn = derived[key]
        if bed.enabled("xattn_b"):
            from . import xattn_b

            # innermost, around FlashInfer's function: the own extend-attention op serves the workspace calls (E4M3
            # cache, BF16 output); it returns fn unchanged when it refuses (reported)
            fn = xattn_b.compose_context(fn, bf16_out=prefill_bf16_out_enabled())
        # inside the BF16-output wrapper: the workspace call sees the BF16 output that wrapper substitutes
        derived[key] = nvfp4_prefill_dq.wrap_context(fn)
        upstream.discard(key)
    if prefill_bf16_out_enabled():
        derived[key] = context_with_bf16_output(derived[key])
        upstream.discard(key)
    for (modname, fname), fn in derived.items():
        if (modname, fname) in upstream:
            continue  # the installed function already serves packed-FP4 KV: nothing to replace
        HookRegistry.register(f"{modname}.{fname}", fn, HookType.REPLACE)
    return True


# --------------------------------------------------------------------------------------------- resolution


def _prefill_dq_requested() -> bool:
    from . import nvfp4_prefill_dq

    return nvfp4_prefill_dq.requested()


def _install_prefill_dq() -> bool:
    from . import nvfp4_prefill_dq

    return nvfp4_prefill_dq.install()


def _install_resolution() -> bool:
    from sglang.srt.arg_groups.resolution_hooks import register_resolution_hook

    from . import nvfp4_kv_resolution

    @register_resolution_hook("handle_kv4_compatibility")
    def _kv4(server_args, previous):
        return nvfp4_kv_resolution.handle_kv4_compatibility(
            server_args, previous, allow_hicache=part_engaged("host_pool")
        )

    return True


# ----------------------------------------------------------------------------------------------- swa_pool


def _is_nvfp4(dtype_str, dtype) -> bool:
    from sglang.srt.utils.common import is_float4_e2m1fn_x2

    return dtype_str == "nvfp4" and is_float4_e2m1fn_x2(dtype)


def _build_hybrid_swa_kv_pool_around(original, self, *args, **kwargs):
    """``KVCacheConfigurator._build_hybrid_swa_kv_pool`` for NVFP4: both sub-pools carry the FP4 quant method."""
    if not _is_nvfp4(self.kv_cache_dtype_str, self.kv_cache_dtype):
        return original(self, *args, **kwargs)
    from . import nvfp4_kv_pool

    return nvfp4_kv_pool.build_hybrid_swa_kv_pool(self, *args, **kwargs)


def _hybrid_sizer_init_around(original, self, kvc, *args, **kwargs):
    """``HybridSWAPoolConfigurator.__init__``: NVFP4 bytes per token instead of one byte per element."""
    original(self, kvc, *args, **kwargs)
    if _is_nvfp4(kvc.kv_cache_dtype_str, kvc.kv_cache_dtype):
        from . import nvfp4_kv_pool

        nvfp4_kv_pool.fix_hybrid_sizer(self, kvc)


def _native_kv_buffer_around(original, token_to_kv_pool, layer_id):
    """``trtllm_mha_backend._trtllm_native_nvfp4_kv_buffer`` for an ``SWAKVPool``: use the owning sub-pool."""
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

    if isinstance(token_to_kv_pool, SWAKVPool):
        token_to_kv_pool._wait_for_layer(layer_id)
        index, is_swa = token_to_kv_pool.layers_mapping[layer_id]
        sub = token_to_kv_pool.swa_kv_pool if is_swa else token_to_kv_pool.full_kv_pool
        return original(sub, index + sub.start_layer)
    return original(token_to_kv_pool, layer_id)


def _install_swa_pool() -> bool:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator._build_hybrid_swa_kv_pool",
        _build_hybrid_swa_kv_pool_around,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.model_executor.pool_configurator.HybridSWAPoolConfigurator.__init__",
        _hybrid_sizer_init_around,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.layers.attention.trtllm_mha_backend._trtllm_native_nvfp4_kv_buffer",
        _native_kv_buffer_around,
        HookType.AROUND,
    )
    return True


# ---------------------------------------------------------------------------------------------- host_pool


def _host_pool_cls_around(original, device_pool):
    from . import nvfp4_kv_host

    if nvfp4_kv_host.is_native_nvfp4_pool(device_pool):
        return nvfp4_kv_host.MHATokenToKVPoolNVFP4Host
    return original(device_pool)


def _write_back_staging_around(original, device_pool, *args, **kwargs):
    from . import nvfp4_kv_host

    if nvfp4_kv_host.is_native_nvfp4_pool(device_pool):
        return original(nvfp4_kv_host.PackedRowView(device_pool), *args, **kwargs)
    return original(device_pool, *args, **kwargs)


def _install_verify_kernel() -> bool:
    from . import nvfp4_kv_verify

    return nvfp4_kv_verify.install()


def _fused_store_enabled() -> bool:
    return os.environ.get("MUSE_VR_NVFP4_KV_FUSED_STORE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _fused_store_pdl() -> bool:
    return os.environ.get("MUSE_VR_NVFP4_KV_STORE_PDL", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _quantize_and_store_around(
    original,
    self,
    k_buffer,
    v_buffer,
    k_scale_buffer,
    v_scale_buffer,
    loc,
    cache_k,
    cache_v,
    k_scale=None,
    v_scale=None,
    native_k_scale_buffer=None,
    native_v_scale_buffer=None,
):
    from .ops import nvfp4_kv_store as op

    if op.supported(
        cache_k,
        cache_v,
        loc,
        k_scale,
        v_scale,
        k_buffer,
        v_buffer,
        k_scale_buffer,
        v_scale_buffer,
        native_k_scale_buffer,
        native_v_scale_buffer,
        self.page_size,
    ):
        op.nvfp4_kv_store(
            cache_k,
            cache_v,
            loc,
            k_scale,
            v_scale,
            k_buffer,
            v_buffer,
            native_k_scale_buffer,
            native_v_scale_buffer,
            self.page_size,
            use_pdl=_state.get("store_pdl", False),
        )
        return None
    bed.count("nvfp4_kv_fused_store", "stock")
    return original(
        self,
        k_buffer,
        v_buffer,
        k_scale_buffer,
        v_scale_buffer,
        loc,
        cache_k,
        cache_v,
        k_scale,
        v_scale,
        native_k_scale_buffer=native_k_scale_buffer,
        native_v_scale_buffer=native_v_scale_buffer,
    )


def _create_buffers_around(original, self, *args, **kwargs):
    out = original(self, *args, **kwargs)
    try:  # build/load the write kernel on this device before any CUDA-graph capture
        import torch

        from .ops import nvfp4_kv_store as op

        dev = torch.device("cuda", torch.cuda.current_device())
        key = ("store", dev.index, _state.get("store_pdl", False))
        if key not in _state.setdefault("preloaded", set()):
            op.preload(dev, use_pdl=_state.get("store_pdl", False))
            _state["preloaded"].add(key)
    except Exception as e:  # noqa: BLE001 - a failed build must not break pool creation; writes then fail loudly
        logger.warning(
            "muse_vr_plugin: nvfp4_kv fused_store preload failed: %s: %s",
            type(e).__name__,
            e,
        )
    return out


def _install_fused_store() -> bool:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    _state["store_pdl"] = _fused_store_pdl()
    cls = "sglang.srt.layers.quantization.fp4_kv_cache_quant_method.NVFP4KVCacheMethod"
    HookRegistry.register(
        f"{cls}.quantize_and_store", _quantize_and_store_around, HookType.AROUND
    )
    HookRegistry.register(
        f"{cls}.create_buffers", _create_buffers_around, HookType.AROUND
    )
    return True


def _kv_scales_requested() -> bool:
    from . import nvfp4_kv_scales

    return nvfp4_kv_scales.requested() is not None


def _install_kv_scales() -> bool:
    from . import nvfp4_kv_scales

    return nvfp4_kv_scales.install()


def _swa_fp8_requested() -> bool:
    from . import nvfp4_kv_mixed

    return nvfp4_kv_mixed.requested()


def _install_swa_fp8() -> bool:
    from . import nvfp4_kv_mixed

    return nvfp4_kv_mixed.install()


def _buffer_info_around(original, pool, *args, **kwargs):
    """SGLang main: a native NVFP4 pool describes packed rows (head_dim / 2 bytes per head, uint8)."""
    from . import nvfp4_kv_host

    info = original(pool, *args, **kwargs)
    if nvfp4_kv_host.is_native_nvfp4_pool(pool):
        return nvfp4_kv_host.packed_buffer_info(pool, info)
    return info


def _host_inputs_around(original, pool, *args, **kwargs):
    """SGLang main: a native NVFP4 pool gets the NVFP4 host class plus its native scale tiles."""
    from . import nvfp4_kv_host

    cls, info, mha_kwargs = original(pool, *args, **kwargs)
    if not nvfp4_kv_host.is_native_nvfp4_pool(pool):
        return cls, info, mha_kwargs
    return (
        nvfp4_kv_host.MHATokenToKVPoolNVFP4HostMain,
        info,
        dict(
            mha_kwargs,
            native_k_scales=tuple(pool.native_k_scale_buffer),
            native_v_scales=tuple(pool.native_v_scale_buffer),
        ),
    )


def _install_host_pool() -> bool:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    from . import sgl_compat

    if sgl_compat.layout() == "main":
        HookRegistry.register(
            "sglang.srt.mem_cache.memory_pool.MHATokenToKVPool._mha_buffer_info",
            _buffer_info_around,
            HookType.AROUND,
        )
        HookRegistry.register(
            "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler._mha_host_inputs",
            _host_inputs_around,
            HookType.AROUND,
        )
        return True
    HookRegistry.register(
        "sglang.srt.mem_cache.pool_host.mha.get_mha_host_pool_cls",
        _host_pool_cls_around,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.mem_cache.pool_host.mha.prepare_mha_write_back_staging",
        _write_back_staging_around,
        HookType.AROUND,
    )
    return True


# ----------------------------------------------------------------------------------------------- register


def register() -> bool:
    """Install every part whose pins match; the component counts as engaged only when all parts are installed."""
    with _lock:
        if _state["engaged"] is not None:
            return _state["engaged"]
        if not bed.enabled(NAME):
            _state["engaged"] = False
            return False
        parts: dict[str, bool] = {}
        reasons: dict[str, str] = {}
        core_pins = all(bed.pinned(rel, sha) for rel, sha in SGLANG_PINS.items())
        host_pins = all(bed.pinned(rel, sha) for rel, sha in HOST_POOL_PINS.items())
        fi_pins = _flashinfer_pins_ok()
        plan = (
            ("fi_gate", fi_pins and core_pins, _install_fi_gate),
            ("swa_pool", core_pins, _install_swa_pool),
            (
                "host_pool",
                core_pins
                and host_pins
                and os.environ.get("MUSE_VR_NVFP4_KV_HOST_POOL", "1").strip()
                not in ("0", "false", "no", "off"),
                _install_host_pool,
            ),
            (
                "verify_kernel",
                core_pins and _verify_kernel_enabled(),
                _install_verify_kernel,
            ),
            ("fused_store", core_pins and _fused_store_enabled(), _install_fused_store),
            ("kv_scales", core_pins and _kv_scales_requested(), _install_kv_scales),
            ("swa_fp8", core_pins and _swa_fp8_requested(), _install_swa_fp8),
            ("prefill_dq", core_pins and _prefill_dq_requested(), _install_prefill_dq),
            # last: the resolution step accepts nvfp4 only when the runtime parts are in place
            ("resolution", core_pins, _install_resolution),
        )
        for part, ok, install in plan:
            if part == "resolution":
                # a requested part that failed (e.g. calibrated scales that do not load) keeps nvfp4 rejected
                ok = (
                    ok
                    and parts.get("fi_gate", False)
                    and parts.get("swa_pool", False)
                    and (parts.get("kv_scales", False) or not _kv_scales_requested())
                    and (parts.get("swa_fp8", False) or not _swa_fp8_requested())
                )
            if part == "prefill_dq":
                # the context-function wrapper rides on fi_gate's derived function
                ok = ok and parts.get("fi_gate", False)
            if not ok:
                parts[part] = False
                reasons[part] = "pins or switch"
                continue
            try:
                parts[part] = bool(install())
            except Exception as e:  # noqa: BLE001 - a failed part leaves SGLang stock (and nvfp4 rejected)
                parts[part] = False
                reasons[part] = f"{type(e).__name__}: {e}"
            _state["parts"][part] = parts[part]
        engaged = all(parts[p] for p in ("fi_gate", "swa_pool", "resolution"))
        _state["engaged"] = engaged
        bed.report(
            NAME,
            engaged,
            **{f"part_{k}": v for k, v in parts.items()},
            prefill_bf16_out=prefill_bf16_out_enabled() and parts.get("fi_gate", False),
            store_pdl=_state.get("store_pdl", False)
            and parts.get("fused_store", False),
            fi_gate_upstream=_state.get("fi_gate_upstream", 0)
            if parts.get("fi_gate", False)
            else 0,
            **({"reasons": reasons} if reasons else {}),
        )
        return engaged

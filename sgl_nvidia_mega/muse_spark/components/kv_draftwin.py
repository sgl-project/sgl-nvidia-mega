# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Component ``kv_draftwin`` (env ``MUSE_VR_KV_DRAFTWIN=1``, default off): DFLASH draft KV in the target's
sliding-window slot space.

Every layer of the DFlash draft is a sliding-window layer, yet SGLang gives a DFLASH draft a plain MHA pool indexed by
the target's full-attention token ids: one draft slot per target token, charged per full token in the pool budget,
and a full-indexed HiCache host sidecar. A draft query at position ``p`` reads keys at ``>= p - window_left`` only,
and the target already keeps those tokens' sliding-window slots: it frees a token's slot once the token is more than
``max(window, page)`` behind the request (page-floored; SGLang main rounds that distance up to whole pages, which keeps
at least as many slots), and the draft block's queries sit at the same positions
as the target's verify block in the same iteration. This component stores the draft KV in that slot space:

* The draft pool is an ``SWAKVPool`` with no full-attention layers and every draft layer on its sliding-window side,
  sized to the target's sliding-window token count and registered on the target allocator's full->SWA mapping. This
  is the plumbing SGLang already uses for sliding-window EAGLE/MTP drafts: the FlashAttention backends translate the
  page table and the write locations, ``KVIndexTranslator`` gets a SLIDING_WINDOW id space, and HiCache builds an
  SWA-indexed draft sidecar whose pages follow the target's sliding-window pages (``build_swa_draft_pools``).
* DFLASH's own draft KV writes (prompt materialization by location, the post-verify prefix-valid commit) arrive with
  full-attention ids; the pool translates them through the same mapping, once per materialization call.
* The pool budget charges the draft's bytes per sliding-window token (``x swa_full_tokens_ratio``) instead of per full
  token, so the memory that held never-read draft KV becomes target tokens.

The draft computes exactly what it computed before: the same projections, norm, RoPE and attention kernels over the
same in-window keys, for every token. Only the storage of tokens that left the window is dropped (those tokens have no
sliding-window slot). Draft attention runs over the same sequence lengths and window as stock; keys outside the window
are skipped or masked by the kernel and contribute exact zeros, so the draft's outputs are bit-identical to stock
(tests/test_kv_draftwin.py).

Engaged only when: DFLASH (not DSPARK) with the fa4 draft attention backend, the target is hybrid sliding-window with
a non-unified, paged, non-ring ``SWATokenToKVPoolAllocator``, every draft layer is a sliding-window layer with
window_left <= the target's, no DCP / PP / disaggregation / unified memory / page-major layout, and the SGLang sources
match their sha256 pins. The decision is taken once, when the target pool is sized; the draft pool build re-checks it
against the loaded draft model and raises rather than fall back to a full-size pool the budget did not charge. At the
first draft KV write, every attention backend of the draft runner is checked to read the pool through its
sliding-window page table (``kv_draftwin:attn`` engagement record; raises otherwise).
"""

from __future__ import annotations

import functools
import logging
import threading

from . import bed

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "srt/runtime_context.py": (
        "get_disagg",
        "get_memory",
        "get_parallel",
        "get_spec",
        "get_exec",
    ),
    "srt/speculative/dflash_utils.py": (
        "get_dflash_attention_sliding_window_size",
        "get_dflash_layer_types",
    ),
    "srt/mem_cache/memory_pool.py": (
        "KVWriteLoc",
        "unwrap_write_loc",
        "write_loc_is_physical",
        "MHATokenToKVPool",
    ),
    "srt/mem_cache/swa_memory_pool.py": ("SWAKVPool",),
    "srt/layers/radix_attention.py": ("RadixAttention",),
    "srt/mem_cache/allocator/swa.py": ("SWATokenToKVPoolAllocator",),
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType"),
}


logger = logging.getLogger(__name__)

NAME = "kv_draftwin"  # env MUSE_VR_KV_DRAFTWIN

T_SIZING = (
    "sglang.srt.model_executor.pool_configurator.HybridSWAPoolConfigurator.__init__"
)
T_SWA_CELL = "sglang.srt.model_executor.pool_configurator.HybridSWAPoolConfigurator._swa_cell_bytes"
T_BUILD = "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator._build_token_to_kv_pool"
T_APPEND = "sglang.srt.speculative.dflash_worker_v2.DFlashWorkerV2._append_target_hidden_to_draft_kv_by_loc"

# sha256 of the SGLang sources (relative to the directory holding the ``sglang`` package) whose behaviour this relies
# on: the DFLASH worker's draft KV writes, the pool / allocator / configurator code that builds and sizes the pools,
# the sliding-window pool and its full->SWA mapping, the out-of-window freeing rule, the FlashAttention backend's
# sliding-window translation, the index translator, the HiCache sidecar builder and the draft's window convention.
PINS = {
    "sglang/srt/speculative/dflash_worker_v2.py": (
        "7d462f5f175828bcc3b4d47fcdfa33cb6af4b14669ef016eeb79b1a5191eec8c",
        "8f82de8d666f81f3c1aa43dc651006f4182ff58593fdfe4eea28c5ee2a14bfc0",
    ),
    "sglang/srt/speculative/dflash_utils.py": (
        "6e0423578bade4a8c58b921430df678840c6fa2de6ea411fe58eb77c98d5a820",
        "35e885f930e95ed82a1f47a0dfa2dab583225d170c6cabe54f6c7bb4e3abbf79",
    ),
    "sglang/srt/models/dflash.py": (
        "59ae9eab87fd3416828604018c36ad6a57be07c4ff6d17f699bb9c0be8c719ff",
        "b7b541368cbbb072a679d08ba5bf0f952f4a43f02b6c078562b2ea7a9ddfd01b",
    ),
    "sglang/srt/mem_cache/memory_pool.py": (
        "b439441448d1112952876cd5f814ea47fcd561ece1a9aa0e7d2080d5f4874106",
        "f3e12b293f3bdce21604ca5e5b473c07dcb6c8e4a34b582c4498d0bd45a10900",
    ),
    "sglang/srt/mem_cache/swa_memory_pool.py": (
        "5d518102fad0bb444176815e1e829a63197b6939ff656bf282253ffa09c90fab",
        "14f2b1daebdc9d872b5786cfc0beed26a413bb41717b70a737a4cb4c31951dfc",
    ),
    "sglang/srt/mem_cache/allocator/swa.py": (
        "5e87b8dd8552d4b0f95012c8bd829b469d2cd06cdc57f2dee2d9ec635fc0f1f7",
        "57e6e797556d8e6d696a9343de3fe2f447ab3ae473b7588b025e59d555a63665",
    ),
    "sglang/srt/mem_cache/common.py": (
        "32112d8a228fa2d2c70062408f63a4e2e4bca1d1c99a3e9a584719de4b5407d3",
        "31855ecda6410c78966952c3d19b1b6556f13a23b7378cc05df51320ba2e43f1",
    ),
    "sglang/srt/mem_cache/kv_cache_configurator.py": (
        "570b22f6aca4d7bb2330fc15a5912ee2f61e500626cce21d69c27e71ec132fb2",
        "5f7f4132923880cb2aa58ef55a5b18233df20bf1ffd47eb0d27b9781a8226c67",
    ),
    "sglang/srt/model_executor/pool_configurator.py": (
        "d6e49edf1d7f6cd23f435257b0230b97fb23d401051c36e1c3b386b24b018c2d",
        "abcadf80aa7dceb6eb1c54735132ec300f1e150d52cd9d95a6bac23620b90ac5",
    ),
    "sglang/srt/model_executor/model_runner_components/kv_pool_runtime.py": (
        "9622b2dfbcc9594a6639dcb78fa276eef3337731f5920c158f85ca3c7cb585ac",
        "5e20b0e7fe9850f8545f8609b546c114842f314542b0fb167664ca90694c8082",
    ),
    "sglang/srt/layers/attention/flashattention_backend.py": (
        "53489175aace7b97498efb10ae8b01c489d35e91058842eee02b19aec18f1a3c",
        "a7b8d205742765e12bf0bad48b151007258736e31f6c97a03725cac6ca2c1d17",
    ),
    "sglang/srt/mem_cache/kv_index_translator.py": (
        "a88d376a1289ebbb3ede7b3dae461c631bbd45a0735610cd63d53fcb4d5e1659",
        "1d2a73c303a3e1dfe4e82eeb73a9a17daa7d4c802259e389c55f51f97ffb0396",
    ),
    "sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py": (
        "35365ec237d193ac77b5adbba9fe239cde205e3747b0da635d3491126b594ac7",
        "a5e1156a9db13593641ea9d649769761ce745fe82cf17c087a59b300d67e17ae",
    ),
    "sglang/srt/mem_cache/kv_cache_builder.py": (
        "79808d4f054dc9edbae0c5c1e2d92818442d74b8a66381f6835eaa64f135853c",
        "578ce61e11904331f6a0bf7d85bf7f5679249b4d1a6718860a9c71b44ad45147",
    ),
    "sglang/srt/speculative/base_spec_worker.py": (
        "5c3ad14c99df94699e210a766d25bc69a14d89895708bcab50cc9da169849ea1",
        "badea4156c24c06848cdf7b572ccb08bf4a74e8e580c38795ff9de560546d168",
    ),
}

# plan: the decision taken when the target pool is sized (None until then, or when not engaged)
_state: dict = {"plan": None, "pool": None}
_tls = threading.local()


def enabled() -> bool:
    return bed.enabled(NAME)


def pins_match() -> tuple[bool, str]:
    for rel, want in PINS.items():
        if not bed.pinned(rel, want):
            return False, f"{rel} differs from the pinned revision"
    return True, "ok"


def plan() -> dict | None:
    """The engagement plan of this process (None when not engaged)."""
    return None if _state["plan"] is None else dict(_state["plan"])


# --------------------------------------------------------------------------------------------------------------------
# decision (target pool sizing)
# --------------------------------------------------------------------------------------------------------------------


def decide(kvc) -> tuple[dict | None, str]:
    """(plan, reason) for a target ``KVCacheConfigurator``; plan is None when the component must stay off."""
    from sglang.srt.runtime_context import (
        get_disagg,
        get_memory,
        get_parallel,
        get_spec,
    )

    if kvc.is_draft_worker:
        return None, "draft configurator"
    if not kvc.spec_algorithm.is_dflash():
        return None, "speculative algorithm is not DFLASH"
    backend = get_spec().speculative_draft_attention_backend
    if backend != "fa4":
        return None, f"draft attention backend {backend!r} (checked: fa4)"
    memory = get_memory()
    if memory.enable_unified_memory:
        return None, "unified memory pool"
    if getattr(memory, "enable_page_major_kv_layout", False):
        return None, "page-major KV layout"
    parallel = get_parallel()
    if parallel.attn_dcp_size > 1 or parallel.pp_size > 1:
        return None, "DCP or pipeline parallelism"
    if get_disagg().disaggregation_mode != "null":
        return None, "disaggregation"
    if kvc.page_size <= 1:
        return None, "page size 1"
    info = kvc.layer_info
    if (
        not kvc.is_hybrid_swa
        or not info.swa_attention_layer_ids
        or not info.full_attention_layer_ids
    ):
        return None, "target is not a hybrid sliding-window model"
    target_window = kvc.sliding_window_size
    if target_window is None or target_window <= 0:
        return None, "target sliding window unknown"
    aux = kvc.spec_aux_config
    draft_config = getattr(aux, "draft_model_config", None)
    n_layers = getattr(aux, "dflash_draft_num_layers", None)
    cell = getattr(aux, "dflash_draft_cell_size_per_token", None)
    if draft_config is None or not n_layers or not cell:
        return None, "draft geometry unresolved"
    from sglang.srt.speculative.dflash_utils import (
        get_dflash_attention_sliding_window_size,
        get_dflash_layer_types,
    )

    hf = draft_config.hf_config
    types = get_dflash_layer_types(hf)
    if (
        not types
        or len(types) < int(n_layers)
        or any(t != "sliding_attention" for t in types[: int(n_layers)])
    ):
        return None, "draft has non-sliding-window layers"
    draft_window = get_dflash_attention_sliding_window_size(hf)
    if draft_window is None or draft_window <= 0 or draft_window > target_window:
        return (
            None,
            f"draft window_left {draft_window} exceeds the target's {target_window}",
        )
    return {
        "draft_window_left": int(draft_window),
        "target_window_left": int(target_window),
        "draft_layers": int(n_layers),
        "draft_bytes_per_token": int(cell),
    }, "ok"


def _sizing_init(original, self, kvc, *args, **kwargs):
    original(self, kvc, *args, **kwargs)
    if kvc.is_draft_worker:
        return
    p, why = decide(kvc)
    if p is None:
        bed.report(NAME, False, reason=why)
        return
    if (
        int(self._draft_cell_size) != p["draft_bytes_per_token"]
        or getattr(self, "_fused_full_entry", None) is not None
    ):
        bed.report(
            NAME,
            False,
            reason=f"unexpected draft charge {self._draft_cell_size} / fused entry",
        )
        return
    # Charge the draft per sliding-window token instead of per full token.
    self._draft_cell_size = 0
    self._kvdw_draft_swa_bytes = p["draft_bytes_per_token"]
    self._recompute_cell_size()
    p["cell_bytes_per_full_token"] = float(self._cell_size)
    p["swa_full_tokens_ratio"] = float(self._swa_full_tokens_ratio)
    first = _state["plan"] is None
    _state["plan"] = p
    if first:
        bed.report(NAME, True, **p)
        logger.warning(
            "muse_vr_plugin: kv_draftwin: draft KV in the target's sliding-window slots: %d B/token charged per "
            "sliding-window token (ratio %.3f); pool cell %.1f B per full token",
            p["draft_bytes_per_token"],
            p["swa_full_tokens_ratio"],
            p["cell_bytes_per_full_token"],
        )


def _swa_cell_bytes(original, self, *args, **kwargs):
    return original(self, *args, **kwargs) + int(
        getattr(self, "_kvdw_draft_swa_bytes", 0)
    )


# --------------------------------------------------------------------------------------------------------------------
# the draft pool
# --------------------------------------------------------------------------------------------------------------------


class _Scope:
    """Translation cache for one materialization call: every draft layer writes the same locations, so the full->SWA
    gather runs once per call. Keyed by tensor identity and valid only inside the call (no kernel rewrites the
    location tensor between the layers' writes)."""

    def __enter__(self):
        self.prev = getattr(_tls, "scope", None)
        _tls.scope = {}
        return self

    def __exit__(self, *exc):
        _tls.scope = self.prev
        return False


@functools.lru_cache(maxsize=None)
def pool_class():
    import torch
    from sglang.srt.mem_cache.memory_pool import (
        KVWriteLoc,
        unwrap_write_loc,
        write_loc_is_physical,
    )
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

    class DraftWindowSWAKVPool(SWAKVPool):
        """DFLASH draft pool: every layer on the sliding-window side, indexed through the target's full->SWA mapping.
        Writes that arrive with full-attention ids (DFLASH's own materialization) are translated here."""

        kvdw_translations = 0

        def kvdw_to_swa(self, loc):
            scope = getattr(_tls, "scope", None)
            if scope is not None:
                hit = scope.get(id(loc))
                if hit is not None and hit[0] is loc:
                    return hit[1]
            idx = loc if loc.dtype == torch.int64 else loc.to(torch.int64)
            out = self.full_to_swa_index_mapping[idx]
            self.kvdw_translations += 1
            if scope is not None:
                scope[id(loc)] = (
                    loc,
                    out,
                )  # keyed by the caller's tensor; holding it keeps the id unique
            return out

        def set_kv_buffer(
            self, layer, loc_info, cache_k, cache_v, k_scale=1.0, v_scale=1.0
        ):
            loc, swa_loc, full_loc = unwrap_write_loc(loc_info)
            if swa_loc is None:
                loc_info = KVWriteLoc(
                    loc,
                    swa_loc=self.kvdw_to_swa(loc),
                    full_loc=full_loc,
                    physical=write_loc_is_physical(loc_info),
                )
            return super().set_kv_buffer(
                layer, loc_info, cache_k, cache_v, k_scale, v_scale
            )

        def set_kv_buffer_prefix_valid(
            self,
            layer,
            loc_2d,
            commit_lens,
            cache_k,
            cache_v,
            k_scale=None,
            v_scale=None,
            layer_id_override=None,
        ):
            layer_id = (
                layer.layer_id if layer_id_override is None else layer_id_override
            )
            index, is_swa = self.layers_mapping[layer_id]
            if not is_swa:
                raise RuntimeError(
                    f"kv_draftwin: draft layer {layer_id} is not on the sliding-window side"
                )
            return self.swa_kv_pool.set_kv_buffer_prefix_valid(
                layer,
                self.kvdw_to_swa(loc_2d),
                commit_lens,
                cache_k,
                cache_v,
                k_scale,
                v_scale,
                layer_id_override=index,
            )

    return DraftWindowSWAKVPool


def _draft_attention_layers(model):
    from sglang.srt.layers.radix_attention import RadixAttention

    return [m for m in model.modules() if isinstance(m, RadixAttention)]


def build_draft_pool(kvc, p: dict):
    """The draft's ``DraftWindowSWAKVPool`` for a draft ``KVCacheConfigurator`` (raises when the plan does not hold)."""
    import torch
    from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.runtime_context import get_disagg, get_exec, get_parallel

    alloc = kvc.token_to_kv_pool_allocator
    if type(alloc) is not SWATokenToKVPoolAllocator or alloc.swa_req_ring:
        raise RuntimeError(
            f"kv_draftwin: target allocator {type(alloc).__name__} is not a paged SWA allocator"
        )
    if kvc.loc_space_scale != 1:
        raise RuntimeError("kv_draftwin: widened location space")
    cfg = kvc.memory_pool_config
    full_tokens, swa_tokens = (
        cfg.full_max_total_num_tokens,
        cfg.swa_max_total_num_tokens,
    )
    if not full_tokens or not swa_tokens:
        raise RuntimeError(
            f"kv_draftwin: pool config has no full/SWA split ({full_tokens}, {swa_tokens})"
        )
    attns = _draft_attention_layers(kvc.model)
    ids = [a.layer_id for a in attns]
    if len(attns) != p["draft_layers"] or sorted(ids) != list(range(len(attns))):
        raise RuntimeError(
            f"kv_draftwin: draft attention layers {ids} do not match the plan ({p['draft_layers']})"
        )
    for a in attns:
        w = a.sliding_window_size
        if (
            w is None
            or w < 0
            or w != p["draft_window_left"]
            or w > p["target_window_left"]
        ):
            raise RuntimeError(
                f"kv_draftwin: draft layer {a.layer_id} window_left {w} does not match the plan"
            )
    mha_cls = kvc._resolve_kv_pool_class(kind="mha", default=MHATokenToKVPool)
    if mha_cls is not MHATokenToKVPool:
        raise RuntimeError(f"kv_draftwin: platform KV pool class {mha_cls.__name__}")
    parallel = get_parallel()
    head_num = kvc.model_config.get_num_kv_heads(
        parallel.attn_tp_size, parallel.attn_dcp_size
    )
    elem = torch._utils._element_size(kvc.kv_cache_dtype)
    per_token = (
        head_num
        * (kvc.model_config.head_dim + kvc.model_config.v_head_dim)
        * len(attns)
        * elem
    )
    if per_token != p["draft_bytes_per_token"]:
        raise RuntimeError(
            f"kv_draftwin: draft pool bytes/token {per_token} differ from the budget's {p['draft_bytes_per_token']}"
        )
    pool = pool_class()(
        size=full_tokens,
        size_swa=swa_tokens,
        page_size=kvc.pool_page_size,
        dtype=kvc.kv_cache_dtype,
        head_num=head_num,
        head_dim=kvc.model_config.head_dim,
        v_head_dim=kvc.model_config.v_head_dim,
        swa_attention_layer_ids=sorted(ids),
        full_attention_layer_ids=[],
        device=kvc.device,
        token_to_kv_pool_class=MHATokenToKVPool,
        enable_memory_saver=get_exec().features.enable_memory_saver,
        post_capture_active=kvc.post_capture_kv_active,
        enable_alt_stream=not get_disagg().enable_pdmux,
        enable_kv_cache_copy=True,
    )
    pool.register_mapping(alloc.full_to_swa_index_mapping)
    return pool


def _build_pool(original, self, *args, **kwargs):
    p = _state["plan"]
    if not self.is_draft_worker or p is None or not self.spec_algorithm.is_dflash():
        return original(self, *args, **kwargs)
    pool = build_draft_pool(
        self, p
    )  # raises: the budget already left no room for a full-size draft pool
    _state["pool"] = pool
    info = dict(
        swa_tokens=int(pool.size_swa),
        full_tokens=int(pool.size),
        layers=int(pool.swa_layer_nums),
        bytes_per_swa_token=p["draft_bytes_per_token"],
    )
    bed.report(NAME + ":pool", True, **info)
    return pool


def check_draft_backends(runner) -> list[str]:
    """Runtime self-check: every attention backend of the draft runner reads the draft pool through its sliding-window
    page table (the FlashAttention backend decides this once, from the pool type, when it is built). Raises otherwise:
    a backend that indexed this pool with full-attention ids would read the wrong slots."""
    names = []
    for backend in {
        id(b): b
        for b in (
            getattr(runner, "attn_backend", None),
            getattr(runner, "decode_attn_backend", None),
        )
        if b is not None
    }.values():
        if getattr(backend, "use_sliding_window_kv_pool", None) is not True:
            raise RuntimeError(
                f"kv_draftwin: draft attention backend {type(backend).__name__} does not read the "
                "draft pool through the sliding-window page table"
            )
        names.append(type(backend).__name__)
    if not names:
        raise RuntimeError(
            "kv_draftwin: the draft runner has no attention backend to check"
        )
    return names


def _append(original, self, *args, **kwargs):
    if (
        _state["pool"] is None
        or self.draft_model_runner.token_to_kv_pool is not _state["pool"]
    ):
        return original(self, *args, **kwargs)
    if not _state.get("backends"):
        _state["backends"] = check_draft_backends(self.draft_model_runner)
        bed.report(NAME + ":attn", True, backends=_state["backends"])
    with _Scope():
        return original(self, *args, **kwargs)


def register() -> bool:
    if not enabled():
        return False
    ok, why = pins_match()
    if not ok:
        bed.report(NAME, False, reason=why)
        return False
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(T_SIZING, _sizing_init, HookType.AROUND)
    HookRegistry.register(T_SWA_CELL, _swa_cell_bytes, HookType.AROUND)
    HookRegistry.register(T_BUILD, _build_pool, HookType.AROUND)
    HookRegistry.register(T_APPEND, _append, HookType.AROUND)
    bed.report(NAME + ":hooks", True)
    return True

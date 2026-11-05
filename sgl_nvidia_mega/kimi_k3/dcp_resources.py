# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Private q1/fixed-q6 DCP resources, deliberately without model admission.

The native groups exist before pool sizing. Cache preparation is a separate,
explicit operation after allocation; it does not fabricate a framework binding.
Borrowed group owners are opaque to the plugin's recursive storage inventory.
Use build_resources with both mla_core="tokenspeed" and DcpBuildOptions, never
the default TRTLLM core. Bounded q1/q6 evidence is at 8K with additional native
framework bindings; these resources do not advertise registered DCP support or
qualify long-context DCP. Full execution requires PreparedDcpHybrid or
PreparedDcpVerify, not just cache preparation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

SGLANG_READS = {
    "srt/kernel_plugins/nvidia/binding.py": ("megakernel_agree",),
    "srt/runtime_context.py": ("get_parallel",),
}


@dataclass(frozen=True, slots=True)
class DcpBuildOptions:
    size: int = 8
    heads: int = 12
    cap: int = 64
    page_size: int = 64
    interleave: int = 1

    def __post_init__(self):
        values = (self.size, self.heads, self.cap, self.page_size, self.interleave)
        if any(type(value) is not int for value in values) or values != (
            8,
            12,
            64,
            64,
            1,
        ):
            raise ValueError(
                "private DCP requires TP8/DCP8, heads12, cap64, page64/cyclic1"
            )


def _native_context():
    from sglang.srt.kernel_plugins.nvidia.binding import megakernel_agree
    from sglang.srt.runtime_context import get_parallel

    return get_parallel, megakernel_agree


def _group_snapshot(parallel):
    fields = (
        parallel.dcp_enabled,
        parallel.tp_size,
        parallel.tp_rank,
        parallel.dcp_size,
        parallel.dcp_rank,
        parallel.attn_dcp_size,
        parallel.attn_dcp_rank,
    )
    if fields[0] is not True or any(type(value) is not int for value in fields[1:]):
        raise ValueError("private DCP requires explicit native parallel geometry")
    _, tp, rank, size, dcp_rank, attn_size, attn_rank = fields
    if (
        (tp, size, attn_size) != (8, 8, 8)
        or not 0 <= rank < 8
        or (rank, dcp_rank, attn_rank) != (rank, rank, rank)
    ):
        raise ValueError("private DCP requires matching TP8/DCP8 rank placement")
    groups = (parallel.tp_group, parallel.dcp_group)
    snapshots = []
    ranks = None
    for group in groups:
        current = group.ranks
        if not isinstance(current, (tuple, list)) or (
            len(current) != 8
            or any(type(value) is not int or value < 0 for value in current)
            or len(set(current)) != 8
            or any(
                type(value) is not int
                for value in (group.world_size, group.rank_in_group, group.rank)
            )
            or group.world_size != 8
            or group.rank_in_group != rank
            or group.rank != current[rank]
            or group.device_group is None
            or group.cpu_group is None
        ):
            raise ValueError("private DCP requires intact native group geometry")
        if ranks is not None and tuple(current) != ranks:
            raise ValueError("private DCP and TP ordered global ranks differ")
        ranks = tuple(current)
        snapshots.append(
            (
                id(group),
                id(group.device_group),
                id(group.cpu_group),
                id(current),
                ranks,
                group.world_size,
                group.rank_in_group,
                group.rank,
            )
        )
    return fields, tuple(snapshots)


class NativeDcpOwner:
    """Strong borrowed owners; no __dict__ for resource-inventory traversal."""

    __slots__ = (
        "_lookup",
        "_owners",
        "_owner_ids",
        "_snapshot",
        "_placement",
        "_contract",
        "device",
        "rank",
        "ranks",
    )

    @classmethod
    def capture(
        cls,
        options,
        *,
        device,
        tokens_per_request,
        deadline_ns,
        widths,
        max_context,
        mla_core,
    ):
        lookup, agree = _native_context()
        parallel = lookup()
        # Use the existing native TP CPU group/timeout, never a new communicator.
        # A rank-local validation failure participates in agreement so peers do
        # not enter symmetric allocation while this rank exits early.
        error = None
        snapshot = None
        try:
            if not isinstance(options, DcpBuildOptions):
                raise ValueError("private DCP requires typed DcpBuildOptions")
            if type(tokens_per_request) is not int or tokens_per_request not in (1, 6):
                raise ValueError("private DCP resource preparation requires q1 or q6")
            if (
                not isinstance(widths, tuple)
                or not widths
                or any(
                    type(width) is not int or not 1 <= width <= 64 for width in widths
                )
                or tuple(sorted(set(widths))) != widths
                or type(max_context) is not int
                or not 0 < max_context <= 262144 + (6 if tokens_per_request == 6 else 0)
                or (
                    tokens_per_request == 6
                    and any(width % 6 or width > 60 for width in widths)
                )
            ):
                raise ValueError("private DCP requires bounded prepared widths/context")
            if mla_core != "tokenspeed":
                raise ValueError("private DCP requires forced TokenSpeed MLA")
            if type(deadline_ns) is not int or not 0 < deadline_ns < (1 << 63):
                raise ValueError("private DCP requires a positive int64 deadline")
            if device.type != "cuda" or device.index is None:
                raise ValueError("private DCP requires an explicit CUDA device")
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "private DCP preparation during capture is forbidden"
                )
            if torch.cuda.get_device_capability(device) != (10, 3):
                raise ValueError("private DCP resource preparation requires SM103")
            snapshot = _group_snapshot(parallel)
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        ranks = None if snapshot is None else snapshot[1][0][4]
        # Never coerce or pickle a rejected input (e.g. widths=None or a tensor).
        # All ranks still reach the existing agreement before any allocation.
        contract = (
            None
            if error is not None
            else (options, widths, max_context, deadline_ns, ranks, tokens_per_request)
        )
        agree(
            (error, contract),
            what="private DCP resource contract",
            group=parallel.tp_group,
        )
        if error is not None:
            raise ValueError(error)
        owner = cls()
        owner._lookup, owner._snapshot = lookup, snapshot
        tp, dcp = parallel.tp_group, parallel.dcp_group
        owner._owners = (
            tp,
            dcp,
            tp.device_group,
            tp.cpu_group,
            dcp.device_group,
            dcp.cpu_group,
        )
        owner._owner_ids = tuple(id(value) for value in owner._owners)
        owner._contract = options, widths, max_context, deadline_ns, tokens_per_request
        owner.device, owner.rank, owner.ranks = device, parallel.dcp_rank, ranks
        owner._placement = device, parallel.dcp_rank, ranks
        return owner

    @property
    def device_group(self):
        return self._owners[4]

    def agreement_group(self):
        """Original native CPU route, even if the current parallel view is stale.

        Never select a replacement group. If the retained coordinator's CPU
        route itself changed, safe agreement is impossible: fail and tear down
        the worker group (peers retain the native CPU-group timeout).
        """
        group = self._owners[0]
        if (
            tuple(id(value) for value in self._owners) != self._owner_ids
            or group.cpu_group is not self._owners[3]
            or type(group.world_size) is not int
            or group.world_size != self._snapshot[1][0][5]
        ):
            raise RuntimeError(
                "private DCP retained native TP CPU agreement route changed"
            )
        return group

    def assert_live(self):
        try:
            current = _group_snapshot(self._lookup())
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError("private DCP native group changed") from exc
        if (
            current != self._snapshot
            or (self.device, self.rank, self.ranks) != self._placement
            or tuple(id(value) for value in self._owners) != self._owner_ids
        ):
            raise RuntimeError("private DCP native group changed")

    def check_plan(
        self, options, widths, max_context, deadline_ns, tokens_per_request=1
    ):
        self.assert_live()
        if (
            type(tokens_per_request) is not int
            or (
                options,
                widths,
                max_context,
                deadline_ns,
                tokens_per_request,
            )
            != self._contract
        ):
            raise ValueError(
                "private DCP resources differ from all-rank agreed preparation"
            )


def _identity(tensor):
    storage = tensor.untyped_storage()
    return (
        id(tensor),
        tensor.data_ptr(),
        storage.data_ptr(),
        storage.nbytes(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


class DcpResources:
    """One model's box, finite prewarm inputs and final 96-head workspace."""

    def __init__(
        self,
        owner,
        options,
        module,
        *,
        widths,
        max_context,
        deadline_ns,
        tokens_per_request=1,
    ):
        if not isinstance(owner, NativeDcpOwner) or not isinstance(
            options, DcpBuildOptions
        ):
            raise ValueError(
                "private DCP resources require captured native owners/options"
            )
        owner.check_plan(options, widths, max_context, deadline_ns, tokens_per_request)
        self.owner, self.options = owner, options
        self._agreement_owner = owner
        self.widths, self.deadline_ns = tuple(widths), deadline_ns
        self.tokens_per_request = tokens_per_request
        self.device = owner.device
        self.attention_heads = options.size * options.heads
        self.max_seq = module.local_max(max_context, options.size)
        # Module globals and borrowed group internals are not owned storage.
        self._module = lambda: module
        self.box = module.DcpBox(
            owner.device_group,
            self.device,
            options.size,
            owner.rank,
            options.heads,
            cap=options.cap,
            interleave=options.interleave,
        )
        factory = dict(device=self.device)
        self.warm_query = torch.zeros(
            (options.cap, 1, self.attention_heads, 576), dtype=torch.uint8, **factory
        )
        self.warm_cache = torch.zeros(
            (1, options.page_size, 576), dtype=torch.uint8, **factory
        )
        self.warm_table = torch.zeros((options.cap, 1), dtype=torch.int32, **factory)
        self.warm_lengths = torch.ones((options.cap,), dtype=torch.int32, **factory)
        if tokens_per_request == 6:
            # Prefix eight plus six verify rows is finite and leaves every rank
            # nonempty even for the first causal query. Never prewarm from the
            # mailbox sentinels or from borrowed native KV/length storage.
            self.warm_totals = torch.full(
                (options.cap,), 14, dtype=torch.int32, **factory
            )
            self.warm_lengths.fill_(2 if owner.rank < 6 else 1)
        self.workspace = None
        self._workspace_identity = None
        self.failed = False
        self._prepared = False
        self._box_geometry = self._geometry()
        self._seals = {name: _identity(tensor) for name, tensor in self._tensors()}
        self._handle_ids = tuple(id(handle) for handle in self.box.handles)
        self.owner.assert_live()

    def agreement_group(self):
        """The originally captured native CPU route, not a replaced owner field."""
        if id(self._agreement_owner) != self._box_geometry[0]:
            raise RuntimeError("private DCP retained agreement owner changed")
        return self._agreement_owner.agreement_group()

    def _tensors(self):
        for name in ("q", "partial", "lse", "peers", "lse_peers", "rows", "rows_lse"):
            yield f"box.{name}", getattr(self.box, name)
        for name in ("warm_query", "warm_cache", "warm_table", "warm_lengths"):
            yield name, getattr(self, name)
        if self.tokens_per_request == 6:
            yield "warm_totals", self.warm_totals

    def _geometry(self):
        box = self.box
        return (
            id(self.owner),
            id(self.options),
            id(box),
            box.size,
            box.rank,
            box.heads,
            box.cap,
            box.interleave,
            box.q_mc,
            self.widths,
            self.deadline_ns,
            self.max_seq,
            self.attention_heads,
            self.device,
            self.tokens_per_request,
        )

    def assert_live(self):
        if self.failed:
            raise RuntimeError(
                "private DCP resources failed preparation; recreate them"
            )
        self.owner.assert_live()
        if (
            self._geometry() != self._box_geometry
            or tuple(id(handle) for handle in self.box.handles) != self._handle_ids
            or self.box.device_group is not self.owner.device_group
        ):
            raise RuntimeError("private DCP box geometry or handles changed")
        for name, tensor in self._tensors():
            if _identity(tensor) != self._seals[name]:
                raise RuntimeError(f"private DCP sealed storage changed: {name}")
        if self._prepared and (
            self._module()._WORKSPACE.get((self.device, self.attention_heads))
            is not self.workspace
            or _identity(self.workspace) != self._workspace_identity
        ):
            raise RuntimeError("private DCP finalized workspace changed")

    def prewarm(self, core, batch, q):
        """Warm the real native core without reading/writing any mailbox or KV pool."""
        self.assert_live()
        if self._prepared or q != self.tokens_per_request or batch not in self.widths:
            raise RuntimeError("private DCP invalid or already finalized prewarm")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("private DCP prewarm cannot run during capture")
        module = self._module()
        try:
            requests = batch // q
            query = self.warm_query[:batch].view(torch.float8_e4m3fn)
            out, lse = self.box.rows[:batch], self.box.rows_lse[:batch]
            causal = {}
            if q == 6:
                query = query.view(requests, q, self.attention_heads, 576)
                out = out.view(requests, q, self.attention_heads, 512)
                lse = lse.view(requests, q, self.attention_heads)
                causal = dict(
                    cp=(self.box.size, self.box.rank, self.box.interleave),
                    totals=self.warm_totals[:requests],
                    pack=True,
                )
            module.stock_decode(
                query,
                self.warm_cache,
                self.warm_table[:requests],
                self.warm_lengths[:requests],
                self.max_seq,
                core.stock_scales,
                out,
                lse,
                module.fill_splits(batch, self.device),
                **causal,
            )
            # This is preparation, outside capture: finish each use of shared
            # scratch before another PDL-capable kernel can overwrite it.
            torch.cuda.synchronize(self.device)
            self.assert_live()
        except BaseException:
            self.failed = True
            raise

    def prepare_mla_caches(self, providers, caches):
        """Explicit MLA-only preparation; never marks a full-model binding ready."""
        self.assert_live()
        if self._prepared:
            raise RuntimeError("private DCP MLA caches are already prepared")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "private DCP cache preparation during capture is forbidden"
            )
        mla = {key: value for key, value in providers.items() if value.kind == "mla"}
        if not mla or set(mla) != set(caches):
            raise ValueError("private DCP MLA cache mapping differs from model")
        for key, cache in caches.items():
            if not isinstance(cache, torch.Tensor) or (
                cache.device != self.device
                or cache.dtype != torch.uint8
                or cache.ndim != 3
                or cache.shape[0] <= 0
                or tuple(cache.shape[1:]) != (64, 576)
                or not cache.is_contiguous()
                or not isinstance(mla[key].core, PreparedDcpMla)
                or mla[key].core.resources is not self
            ):
                raise ValueError(
                    "private DCP requires native contiguous page64 MLA caches"
                )
        try:
            for key, provider in mla.items():
                # Retain borrowed native storage even when compilation fails.
                provider.cache = caches[key]
                provider.core.prepare_cache(caches[key])
            workspaces = [
                provider.core.finalize_workspace() for provider in mla.values()
            ]
            if any(workspace is not workspaces[0] for workspace in workspaces):
                raise RuntimeError(
                    "private DCP MLA layers must share one 96-head workspace"
                )
            self.workspace = workspaces[0]
            self._workspace_identity = _identity(self.workspace)
            self._prepared = True
            self.assert_live()
        except BaseException:
            self.failed = True
            raise


class PreparedDcpMla:
    """DCP-only wrapper; the existing DCP-off TokenSpeed adapter stays unchanged."""

    def __init__(self, core, resources):
        if (
            core.dcp is not resources.box
            or core.heads != 12
            or core.device != resources.device
            or core.deadline_ns != resources.deadline_ns
            or core.parity not in (0, 1)
            or set(core.cores)
            != {(batch, resources.tokens_per_request) for batch in resources.widths}
            or set(core.cores.values()) != {"tokenspeed"}
            or core.max_seq != resources.max_seq
        ):
            raise ValueError(
                "private DCP requires its prepared q1/q6 12-head native core"
            )
        self.core, self.resources = core, resources
        self.workspace = None
        self._prepared_geometry = None
        self._workspace_identity = None

    @property
    def heads(self):
        return self.core.heads

    @property
    def weights(self):
        return self.core.weights

    def _geometry(self):
        core = self.core
        return (
            core.device,
            core.heads,
            core.max_seq,
            id(core.dcp),
            core.parity,
            core.deadline_ns,
            tuple(core.stock_scales),
            tuple(sorted(core.cores.items())),
            tuple(sorted(core.pitches.items())),
            tuple(sorted(core.kernels)),
        )

    def prepare_cache(self, cache):
        self.resources.assert_live()
        if self._prepared_geometry is not None or self.workspace is not None:
            raise RuntimeError("private DCP core cannot be prepared twice")
        try:
            self.core.prepare_cache(cache, prewarm=self.resources.prewarm)
            self._prepared_geometry = self._geometry()
        except BaseException:
            self.resources.failed = True
            raise
        return self

    def finalize_workspace(self):
        self.resources.assert_live()
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("private DCP workspace cannot finalize during capture")
        if self.workspace is not None:
            self.check_workspace()
            return self.workspace
        if (
            self._prepared_geometry is None
            or self._geometry() != self._prepared_geometry
        ):
            raise RuntimeError(
                "private DCP core must prewarm before workspace finalization"
            )
        key = self.core.device, self.resources.attention_heads
        workspace = self.resources._module()._WORKSPACE.get(key)
        if not isinstance(workspace, torch.Tensor) or (
            workspace.device != self.core.device
            or workspace.device.type != "cuda"
            or workspace.dtype != torch.int8
            or workspace.ndim != 1
            or not workspace.is_contiguous()
            or workspace.numel() == 0
        ):
            raise RuntimeError(
                "private DCP prewarm did not produce the 96-head workspace"
            )
        self.workspace, self._workspace_identity = workspace, _identity(workspace)
        return workspace

    def check_workspace(self):
        self.resources.assert_live()
        key = self.core.device, self.resources.attention_heads
        if self.workspace is None or (
            self.resources._module()._WORKSPACE.get(key) is not self.workspace
            or _identity(self.workspace) != self._workspace_identity
            or self._geometry() != self._prepared_geometry
        ):
            raise RuntimeError("private DCP workspace or geometry changed")

    def run(self, *args: Any, **kwargs: Any):
        # No serving hook calls this resource-only checkpoint. Explicit native
        # component callers still need a separate genuine step/slot contract.
        self.check_workspace()
        if not self.resources._prepared:
            raise RuntimeError("private DCP resource preparation is not finalized")
        if (
            kwargs.get("q", 1) != self.resources.tokens_per_request
            or kwargs.get("totals") is None
        ):
            raise ValueError(
                "private DCP execution requires its prepared q and explicit global totals"
            )
        result = self.core.run(*args, **kwargs)
        self.check_workspace()
        return result

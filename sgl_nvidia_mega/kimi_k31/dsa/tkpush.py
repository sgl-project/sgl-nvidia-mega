# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K31MK_DSA_REQ_SHARD_TKPUSH (default off; dsa/req_shard.py): the request-sharded selection's owner top-k (x47v2,
dsa/x47960) stores its output rows as tagged words straight into every rank's mailbox, so the tag / fused routes skip
dsa/tag_gather.py's _push_kernel launch on the critical path after the top-k.

The kernel is dsa/x47960/topk_v2_tkpush.cuh: a fork of the vendored #47960 topk_v2.cuh (its own inline namespace and JIT
name, the same topk_impl.cuh) whose small-batch cluster kernel takes a kTagPush flag; device side dsa/tag_push.cuh. The word
format, mailbox layout, release fence and consumers are tag_gather.py's; the tag is read from the consumer's own use counter
of the copy (TagGather.tag_epoch / the fused core's counters), so this file keeps no epoch of its own. When the launch takes a
path without the in-kernel push (score rows not above the small-batch cluster floor, or more rows than the small-batch
kernels take), the module returns 0 and the caller pushes with TagGather.push_ext (same epoch source).

TkPushSelect stands in for X47960Select in VerifySelection.x47 (same ``run`` signature); ``arm`` sets the next call's
target (copy, epoch counter, block offset) and ``take`` returns whether that call pushed (None: the selection did not reach
the top-k, e.g. a bounded "short" step, so the caller must push). Built at prepare time (host nvcc), never inside a capture."""

import logging
from functools import cache
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
X47 = HERE / "x47960"
MODE_EXACT = 1


@cache
def load(mode: int = MODE_EXACT, nofence: bool = False):
    """The JIT module: x47960.load's build recipe (mode macro, split-compile off, cluster occupancy macros) on the fork file."""
    from sglang.kernels.jit.utils import is_arch_support_pdl, load_jit, make_cpp_args
    from sglang.kernels.jit.utils.occupancy import NoSchedulableClustersError, get_max_active_clusters

    flags = [f"-DSGL_TOPK_MODE={mode}", "-Xptxas=--split-compile=0"]
    flags += ["-DK31_TKP_NOFENCE=1"] if nofence else []  # TAGNF (K31MK_DSA_REQ_SHARD_TAG_NOFENCE): no release fence in the top-k
    if is_arch_support_pdl():
        for cluster_size, occupancy in ((8, 2), (16, 1)):
            try:
                n = get_max_active_clusters(cluster_size=cluster_size, occupancy=occupancy)
            except NoSchedulableClustersError:
                n = 0
            flags.append(f"-DSGL_TOPK_V2_MAX_C{cluster_size}_OCC{occupancy}={n}")
    args = make_cpp_args(is_arch_support_pdl(), mode)
    kernel = f"TopKKernel<{args}>"
    return load_jit(
        f"k31_x47960_tkpush_m{mode}" + ("_nf" if nofence else ""),
        *args,
        extra_cuda_cflags=flags,
        cuda_files=[str(X47 / "topk_v2_tkpush.cuh")],
        extra_include_paths=[str(X47), str(HERE)],  # topk_impl.cuh (vendored x47960 copy) and tag_push.cuh
        cuda_wrappers=[
            ("topk_transform_paged", f"{kernel}::transform_paged"),
            ("topk_transform_paged_push", f"{kernel}::transform_paged_push"),
            ("topk_plan", f"{kernel}::plan"),
        ],
    )


class TkPushSelect:
    """X47960Select's exact top-2048 (mode 1, per-row page table, paged output) whose call also pushes the output rows as
    tagged words into copy ``c`` of every rank's mailbox (``peers``: the TagGather's symmetric-memory buffer addresses)."""

    def __init__(self, device, rows, peers, world, rank, n_own, mode: int = MODE_EXACT, nofence: bool = False):
        self.nofence = bool(nofence)
        self.mod = load(mode, self.nofence)
        self.plan = torch.full((rows + 1, 2), -1, dtype=torch.int32, device=device)
        self.peers = torch.tensor([int(p) for p in peers], dtype=torch.int64)  # CPU: copied into the kernel's parameters
        if self.peers.numel() != int(world):
            raise ValueError(f"k31 TKPUSH: {self.peers.numel()} mailbox addresses for world {world}")
        self.world, self.rank, self.n_own = int(world), int(rank), int(n_own)
        self.target = None
        self.pushed = None
        self.calls = self.fallbacks = 0
        self.logged = False

    def arm(self, c, epoch):
        """Next run's push target: copy ``c``, ``epoch`` = a 1-element device int32 view of the consumer's use counter."""
        self.target = (int(c), epoch, (int(c) * self.world + self.rank) * self.n_own)
        self.pushed = None

    def take(self):
        """Whether the last armed call pushed in-kernel (True / False), or None when no top-k call consumed the target."""
        pushed, self.pushed, self.target = self.pushed, None, None
        return pushed

    def run(self, scores, causal_lengths, row_table, selected, page_size=64):
        self.mod.topk_plan(causal_lengths, self.plan, -1)
        if self.target is None:  # not armed (a caller outside ReqShardState.select): the plain top-k, no push
            self.mod.topk_transform_paged(scores, causal_lengths, row_table, selected, page_size, self.plan, None)
            return selected
        _, epoch, base = self.target
        self.target = None
        self.pushed = bool(self.mod.topk_transform_paged_push(scores, causal_lengths, row_table, selected, page_size, self.plan,
                                                              self.peers, epoch, base))
        self.calls += 1
        self.fallbacks += 0 if self.pushed else 1
        if not self.logged:
            self.logged = True
            logger.info("K31MK_DSA_REQ_SHARD_TKPUSH %s: rows %d, score width %d, world %d, nofence %d",
                        "ENGAGED (in-kernel push)" if self.pushed else "FALLBACK (push_ext)", selected.shape[0],
                        scores.shape[1], self.world, int(self.nofence))
        return selected

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K3.1 TP16 sparse MLA core (sparse_mla6.cuh): 6 heads per rank without padding, split-KV partials (unnormalized
FP32 U + (max, sum) per head and split) that MlaValueFold merges inside the W_UV / gate kernel. Used by
dsa/verify.py under K31MK_SPARSE_MLA6=1 instead of trtllm-gen's 16-head sparse MLA + MlaValue."""

import logging
import math
import os
from functools import cache
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {"kernels/jit/utils/compile/loader.py": ("load_jit",)}

LOG2E = 1.0 / math.log(2.0)
# Tokens per split (CTA): ceil(2048 / TOKENS) splits per row. ``K31MK_SPARSE_MLA6_TOKENS`` (64, 80, 96, 128 or 256;
# default 128). 80 / 96 use 5 / 6 warps (26 / 22 splits: more CTAs per row, ~all SMs at 8 rows).
TOKENS = int(os.environ.get("K31MK_SPARSE_MLA6_TOKENS", "128"))
TOKEN_CHOICES = (64, 80, 96, 128, 256)


def _parse_tokens_rows(spec):
    """``K31MK_SPARSE_MLA6_TOKENS_ROWS`` = "<T>:<rows>[,<rows>...]", each <rows> a verify row count N or a range lo-hi
    (e.g. "80:32,56-64"). Default "" = off. Returns (T, frozenset of row counts); invalid text raises (fail closed)."""
    if not spec:
        return None, frozenset()
    head, sep, body = spec.partition(":")
    if not sep or not head.strip().isdigit() or int(head) not in TOKEN_CHOICES:
        raise ValueError(f"K31MK_SPARSE_MLA6_TOKENS_ROWS={spec!r}: expected '<T>:<rows>[,...]' with T in {TOKEN_CHOICES}")
    rows = set()
    for part in body.split(","):
        lo, dash, hi = part.strip().partition("-")
        if not lo.isdigit() or (dash and not hi.isdigit()):
            raise ValueError(f"K31MK_SPARSE_MLA6_TOKENS_ROWS={spec!r}: bad row item {part!r} (N or lo-hi)")
        lo, hi = int(lo), int(hi) if dash else int(lo)
        if lo < 1 or hi < lo or hi > 4096:
            raise ValueError(f"K31MK_SPARSE_MLA6_TOKENS_ROWS={spec!r}: row item {part!r} must satisfy 1 <= lo <= hi <= 4096")
        rows.update(range(lo, hi + 1))
    return int(head), frozenset(rows)


# ``K31MK_SPARSE_MLA6_TOKENS_ROWS`` (default "" = off = the served core at every row count):
# tokens per split T for the listed verify row counts only; every other row count keeps
# K31MK_SPARSE_MLA6_TOKENS. Reason (measured): the served T=128 core (2 CTAs / SM) runs 32 rows in more than one wave;
# T=80 (5 warps, 4 CTAs / SM) fits one wave there (a faster core) but is slower at 8 / 16 rows. Same kernel family,
# same PDL edges (no new synchronization); the fold reads ``splits`` from this object. NOT bitwise: split boundaries move
# the softmax max / sum and the fold's summation order (FP64 / KL class). Not supported with K31MK_DSA_REQ_SHARD_AG=fused
# (req_shard sizes the core's per-split epoch counters from the module-wide TOKENS): refused at import.
# K31MK_SPARSE_MLA6_FEED=1 (default 0; TMA-FEED): at T 128 the core's KV rows and query arrive by cp.async.bulk
# with mbarrier completion instead of cp.async (same SMEM bytes, same math: bitwise to the default). Other T keep the default kernels.
FEED = os.environ.get("K31MK_SPARSE_MLA6_FEED", "0") == "1"
_FEED_LOGGED = set()
# K31MK_SPARSE_MLA6_FEED_W8=1 (default 0; FEED_W8): with the feed at T 128, the untagged core runs 8 warps
# (one 16-token m-tile per warp, P.V on all 8 warps) - bitwise to the 4-warp core; it needs K31MK_SPARSE_MLA6_FEED=1 (alone, 8 warps are not
# faster). The tagged route (K31MK_DSA_REQ_SHARD_AG=fused) keeps the 4-warp kernels.
FEED_W8 = os.environ.get("K31MK_SPARSE_MLA6_FEED_W8", "0") == "1"
_W8_LOGGED = set()

TOKENS_ROWS_T, TOKENS_ROWS = _parse_tokens_rows(os.environ.get("K31MK_SPARSE_MLA6_TOKENS_ROWS", "").strip())
if TOKENS_ROWS and os.environ.get("K31MK_DSA_REQ_SHARD", "0") == "1" and \
        os.environ.get("K31MK_DSA_REQ_SHARD_AG", "nccl") == "fused":
    raise ValueError("K31MK_SPARSE_MLA6_TOKENS_ROWS is not supported with K31MK_DSA_REQ_SHARD_AG=fused "
                     "(its core epoch counters assume K31MK_SPARSE_MLA6_TOKENS at every row count)")
_ENGAGED = set()


def tokens_for(rows):
    """Tokens per split the core uses at ``rows`` verify rows (K31MK_SPARSE_MLA6_TOKENS_ROWS, else TOKENS)."""
    return TOKENS_ROWS_T if rows in TOKENS_ROWS else TOKENS

# KNOWN K-13: (device index, graph capture id or "eager") keys whose first launch passed the stream check.
_STREAM_CHECKED = set()


@cache
def _load(feed=False, w8=False):
    import sglang
    from sglang.kernels.jit.utils import load_jit

    flags = (["-DK31_SMLA6_FEED=1"] if feed else []) + (["-DK31_SMLA6_W8=1"] if w8 else [])
    return load_jit(
        "k31_sparse_mla6" + ("_feed" if feed else "") + ("_w8" if w8 else ""),
        cuda_files=[str(Path(__file__).with_suffix(".cuh"))],
        cuda_wrappers=[("run", "k31_sparse_mla6::Binding::run"),
                       ("run_checked", "k31_sparse_mla6::Binding::run_checked"),
                       ("run_tagged", "k31_sparse_mla6::Binding::run_tagged")],
        extra_cuda_cflags=flags or None,
        extra_include_paths=[str(Path(sglang.__file__).parent / "kernels/jit/csrc")],
    )


def _stream_check_key(device_index):
    """This launch's K-13 key, or None when this device (eager) or this graph capture already passed the check.
    Cost per launch: one cudaStreamIsCapturing and a set lookup (a graph replay runs no Python)."""
    capture = "eager"
    if torch.cuda.is_current_stream_capturing():
        import cuda.bindings.driver as drv

        current = torch.cuda.current_stream(device_index).cuda_stream
        info = drv.cuStreamGetCaptureInfo(drv.CUstream(current))
        if info[0] != drv.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"K-13 stream check: cuStreamGetCaptureInfo failed ({info[0]})")
        capture = int(info[2])
    key = (device_index, capture)
    return None if key in _STREAM_CHECKED else key


class SparseMla6:
    """Split partials of ``rows`` query rows x 6 heads over each row's 2048 selected physical slots."""

    def __init__(self, device, rows, heads=6, tokens=None, feed=None, feed_w8=None):
        engaged = tokens is None and rows in TOKENS_ROWS
        tokens = tokens_for(rows) if tokens is None else tokens
        if heads != 6 or tokens not in TOKEN_CHOICES:
            raise ValueError(f"sparse MLA6: 6 heads and 64/80/96/128/256 tokens per split, got {heads}, {tokens}")
        self.rows, self.heads, self.tokens = rows, heads, tokens
        self.splits = -(-2048 // tokens)
        if engaged and (rows, tokens) not in _ENGAGED:
            _ENGAGED.add((rows, tokens))
            logger.info("K31MK_SPARSE_MLA6_TOKENS_ROWS ENGAGE rows=%d tokens=%d splits=%d (default tokens %d)",
                        rows, tokens, self.splits, TOKENS)
        self.partial = torch.empty((rows, self.splits, heads, 512), dtype=torch.float32, device=device)
        self.stats = torch.empty((rows, self.splits, heads, 2), dtype=torch.float32, device=device)
        feed = FEED if feed is None else bool(feed)
        if feed and (rows, tokens) not in _FEED_LOGGED:
            _FEED_LOGGED.add((rows, tokens))
            if tokens == 128:
                logger.info("K31MK_SPARSE_MLA6_FEED ENGAGED: TMA row feed (cp.async.bulk + mbarrier) rows=%d tokens=128", rows)
            else:
                logger.info("K31MK_SPARSE_MLA6_FEED DECLINED: rows=%d tokens=%d (the feed form is built for T 128 only)", rows, tokens)
        self.feed = feed and tokens == 128
        feed_w8 = FEED_W8 if feed_w8 is None else bool(feed_w8)
        self.feed_w8 = feed_w8 and self.feed
        if feed_w8 and (rows, tokens) not in _W8_LOGGED:
            _W8_LOGGED.add((rows, tokens))
            if self.feed_w8:
                logger.info("K31MK_SPARSE_MLA6_FEED_W8 ENGAGED: 8-warp TMA-fed core rows=%d tokens=128 (tagged route keeps 4 warps)", rows)
            else:
                logger.info("K31MK_SPARSE_MLA6_FEED_W8 DECLINED: rows=%d tokens=%d feed=%d (needs K31MK_SPARSE_MLA6_FEED=1 and T 128)",
                            rows, tokens, int(self.feed))
        self.module = _load(self.feed, self.feed_w8)

    def run(self, query, cache, selected, lengths, scales, stream=None):
        """``query`` [rows, 6, 576] FP8 codes, ``cache`` the page-64 KV pool (FP8 codes, 576 B per slot),
        ``selected`` [rows, 2048] int32 slots (-1 = none), ``lengths`` [rows] causal lengths, ``scales`` the stock
        (bmm1, bmm2) pair; bmm2 is applied by the fold. ``stream``: raw handle of the stream the dependent fold
        launches on (default: torch's current stream). The first launch per device and per graph capture raises,
        launching nothing, unless tvm-ffi launches this core on that stream (KNOWN K-13)."""
        rows = self.rows
        q = query.view(torch.uint8).reshape(query.shape[0], -1)[:rows]
        kv = cache.view(torch.uint8).view(-1)
        args = (q, kv, selected[:rows], lengths[:rows], self.partial, self.stats, float(scales[0]) * LOG2E, self.tokens)
        key = _stream_check_key(kv.device.index)
        if key is None:
            self.module.run(*args)
            return self.partial, self.stats
        if stream is None:
            stream = torch.cuda.current_stream(kv.device).cuda_stream
        self.module.run_checked(*args, int(stream))
        _STREAM_CHECKED.add(key)
        return self.partial, self.stats

    def run_tagged(self, query, cache, tagged, lengths, scales, stream=None):
        """As ``run``, with the slots read by the core itself from the request-sharded selection's mailbox
        (K31MK_DSA_REQ_SHARD_AG=fused): ``tagged`` is a req_shard.TaggedRows (this layer's mailbox copy as [blocks x
        rows per block, 2048] int32 tagged words, the core's per-(row, split) epoch counters of that copy, the spin
        limit, ``fence``). The core polls each slot word until it carries this use's tag (trap after the limit), fences
        (unless ``tagged.fence`` is False: TAGNF, dsa/req_shard.py), then computes as ``run``. Same K-13 stream check on the
        first launch per device and per graph capture."""
        rows = self.rows
        q = query.view(torch.uint8).reshape(query.shape[0], -1)[:rows]
        kv = cache.view(torch.uint8).view(-1)
        if tuple(tagged.epochs.shape) != (rows, self.splits):
            raise ValueError(f"sparse MLA6 tagged: epochs {tuple(tagged.epochs.shape)} != {(rows, self.splits)}")
        key = _stream_check_key(kv.device.index)
        expected = 0
        if key is not None:
            expected = int(torch.cuda.current_stream(kv.device).cuda_stream if stream is None else stream)
        self.module.run_tagged(q, kv, tagged.words, tagged.epochs, lengths[:rows], self.partial, self.stats,
                               float(scales[0]) * LOG2E, self.tokens, int(tagged.deadline_ns), int(tagged.block_rows),
                               int(tagged.blocks), int(getattr(tagged, "fence", True)), expected)
        if key is not None:
            _STREAM_CHECKED.add(key)
        return self.partial, self.stats

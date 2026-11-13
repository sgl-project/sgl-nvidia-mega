# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Spread router role (csrc/wfc_k1s.cuh), probe form: dev, not wired into the server.

One launch computes router logits with the package's fixed-split TF32 router contract (16 chunks of 512 columns, one
mma.sync m16n8k8 TF32 chain per chunk in ascending k, chunk-order fp32 sum), then SGLang's routing arithmetic per row, and
releases a per-row ready word. ``shard=True`` is the N-sharded variant (this rank's 64 experts + a merge of per-shard
candidate lists). The launch is cooperative (all CTAs resident) because CTAs wait on a shared go word.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/jit/utils/__init__.py": ("load_jit",),
    "kernels/jit/utils/compile/paths.py": ("KERNEL_PATH",),
}


_CSRC = Path(__file__).resolve().parents[1] / "csrc"
_CUH = _CSRC / "wfc_k1s.cuh"
K, N, CHUNKS, SUBS, PAD, G = 8192, 256, 16, 8, 68, 192


def _stamp_dir():
    """Timing builds only: a directory holding the (private) stamp header; unset = production build, no stamps."""
    import os

    d = os.environ.get("MUSE_VR_WFC_K1S_STAMPS_DIR", "").strip()
    return d or None


@functools.lru_cache(maxsize=None)
def module(shard: int, pool: int):
    from sglang.kernels.jit.utils import load_jit
    from sglang.kernels.jit.utils.compile.paths import KERNEL_PATH

    from sgl_nvidia_mega.muse_spark.components.sgl_compat import csrc_include_paths

    sd = _stamp_dir()
    extra = {"extra_cuda_cflags": ["-DMUSE_VR_WFC_K1S_STAMPS"]} if sd else {}
    return load_jit(
        "muse_vr_wfc_k1s" + ("_t" if sd else ""),
        str(shard),
        str(pool),
        cuda_files=[str(_CUH)],
        cuda_wrappers=[
            ("run", f"WfcK1S<{shard}, {pool}>::run"),
            ("ximg", f"WfcK1S<{shard}, {pool}>::ximg"),
            ("ref", f"WfcK1S<{shard}, {pool}>::ref"),
        ]
        + (
            [
                ("run_prod", f"WfcK1S<{shard}, 0>::run_prod"),
                ("merge", f"WfcK1S<{shard}, 0>::merge"),
            ]
            if pool == 0
            else []
        ),
        extra_include_paths=csrc_include_paths(KERNEL_PATH / "csrc")
        + [str(_CSRC)]
        + ([sd] if sd else []),
        **extra,
    )


def tiling(m: int, shard: bool, grid: int = G):
    """(tg, rslab, jobs, mpad): row tiles per CTA group, row tiles per chain warp, route jobs per CTA, padded rows. Shard grids
    are 16 x (row groups) CTAs (grid < 192 = a reserved slice of the SMs)."""
    tiles = (m + 15) // 16
    groups = grid // 16 if shard else 3
    tg = max(1, (tiles + groups - 1) // groups)
    rslab = tg if tg <= 3 else (tg + 1) // 2
    jobs = (m + grid - 1) // grid
    return tg, rslab, jobs, tiles * 16


class State:
    """Device buffers for one (M, shard) configuration; the control words persist across launches (epochs)."""

    def __init__(
        self,
        m: int,
        shard: bool,
        device,
        pool_bytes: int = 4096 * 8192 * 2,
        grid: int = G,
    ):
        self.m, self.shard, self.grid = m, shard, grid
        self.tg, self.rslab, self.jobs, self.mpad = tiling(m, shard, grid)
        ne = 64 if shard else 256
        self.ne = ne
        d = device
        self.ximg = torch.zeros(
            CHUNKS * SUBS * self.mpad * PAD, dtype=torch.float32, device=d
        )
        self.board = torch.zeros(CHUNKS * self.mpad * ne, dtype=torch.int64, device=d)
        self.topk_w = torch.zeros(m, 8, dtype=torch.float32, device=d)
        self.topk_ids = torch.zeros(m, 8, dtype=torch.int32, device=d)
        self.packed = torch.zeros(m, 8, dtype=torch.int32, device=d)
        self.ready = torch.zeros(m * 8, dtype=torch.int64, device=d)
        self.ctl = torch.zeros(8, dtype=torch.int64, device=d)
        self.stamps = torch.zeros(
            8 + 3 * G + 5 * m + 32, dtype=torch.int64, device=d
        )  # sized for the 192-CTA grid
        self.pool_cnt = torch.zeros(G, dtype=torch.int64, device=d)
        self.dbg = torch.zeros(m, ne, dtype=torch.float32, device=d)
        self.cand = torch.zeros(m * 4 * 8, dtype=torch.int64, device=d)
        self.cand_score = torch.zeros(m * 4 * 8, dtype=torch.float32, device=d)
        self.cand_out = torch.zeros(m * 8, dtype=torch.int64, device=d)
        self.cand_out_score = torch.zeros(m * 8, dtype=torch.float32, device=d)
        self.pool_src = (
            torch.zeros(3 * pool_bytes, dtype=torch.uint8, device=d)
            if pool_bytes
            else torch.zeros(48, dtype=torch.uint8, device=d)
        )

    def set_x(self, x: torch.Tensor):
        module(int(self.shard), 0).ximg(x.contiguous(), self.m, self.mpad, self.ximg)

    def launch(
        self,
        w,
        bias,
        pool: int = 1,
        q0: int = 0,
        pmma_iters: int = 0,
        dbg: bool = False,
        mult: float = 1.0,
        route_scale: float = 1.0,
        eps: float = 1e-15,
        pace_ns: int = 0,
    ):
        import os

        tmo = int(
            os.environ.get("MUSE_VR_WFC_K1S_TMO_SCALE", "1")
        )  # bounded-wait scale (sanitizer runs only)
        module(int(self.shard), pool).run(
            self.ximg,
            w,
            bias,
            self.board,
            self.topk_w,
            self.topk_ids,
            self.packed,
            self.ready,
            self.ctl,
            self.stamps,
            self.pool_src,
            self.pool_cnt,
            self.dbg,
            self.cand,
            self.cand_score,
            self.cand_out,
            self.cand_out_score,
            self.m,
            self.mpad,
            G,
            self.tg,
            self.rslab,
            self.jobs,
            q0,
            pmma_iters,
            int(dbg),
            float(mult),
            float(route_scale),
            float(eps),
            int(pace_ns),
            tmo,
        )

    def launch_prod(
        self,
        x,
        w,
        bias,
        seq_word,
        layer: int,
        dbg: bool = False,
        mult: float = 1.0,
        route_scale: float = 1.0,
        eps: float = 1e-15,
        q0: int = 0,
        cand_dst=None,
        cand_stride: int = 0,
        cand_per: int = 0,
        cand_line: int = 0,
    ):
        """Production form: x = the router input [M, 8192] fp32 (row-major, as the boundary writes it); PDL launch; tag =
        seq_word[0] * 64 + layer (seq_word: int64 device word, advanced once per forward by its owner).
        Shard candidates-only mode (cand_dst: f32 tensor whose first element is row 0's list): row m's [8 keys | 8 ids |
        8 scores] at m * cand_stride, or with cand_per > 0 at (m // cand_per) * cand_line + (m % cand_per) * cand_stride."""
        import os

        tmo = int(os.environ.get("MUSE_VR_WFC_K1S_TMO_SCALE", "1"))
        module(int(self.shard), 0).run_prod(
            x.contiguous(),
            w,
            bias,
            self.board,
            self.topk_w,
            self.topk_ids,
            self.packed,
            self.ready,
            self.ctl,
            self.stamps,
            self.dbg,
            seq_word,
            int(layer),
            self.m,
            self.mpad,
            self.grid,
            self.tg,
            self.rslab,
            self.jobs,
            int(q0),
            int(dbg),
            float(mult),
            float(route_scale),
            float(eps),
            tmo,
            self.cand_out,
            self.cand_out_score,
            cand_dst if cand_dst is not None else self.cand_out_score,
            int(cand_stride if cand_dst is not None else 0),
            int(cand_per),
            int(cand_line),
        )

    def router_to_ids_us(self) -> float:
        """Timing builds only (stamps compiled in)."""
        s = self.stamps.cpu()
        t0 = int(s[0])
        last = int(s[8 + 2 * self.grid : 8 + 2 * self.grid + self.m].max())
        return (last - t0) / 1000.0


def ref_logits(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """The contract's two-kernel reference form (partials, then the chunk-order sum)."""
    m = x.shape[0]
    part = torch.empty(CHUNKS * m * N, dtype=torch.float32, device=x.device)
    out = torch.empty(m, N, dtype=torch.float32, device=x.device)
    module(0, 0).ref(x.contiguous(), w.contiguous(), m, part, out)
    return out


def merge(
    cand,
    world: int,
    m: int,
    stride: int,
    off: int,
    route_scale: float,
    eps: float,
    topk_w,
    topk_ids,
    packed,
    per: int = 0,
    line: int = 0,
    offs=None,
    err_ctl=None,
    err_dst=None,
    err_code: int = 0,
):
    """Top-8 merge of gathered per-shard candidate lists (f32). Default layout: rows [world][m][stride], lists at column
    `off`. Packed layout (per > 0): list s, row m at offs[s] + (m // per) * line + (m % per) * stride elements of cand.
    Error relay (optional): err_ctl = int64 [n, 8] producer control blocks (word 3 = sticky error word); any nonzero word
    poisons every row and ORs err_code into err_dst (int32, the consumer's sticky error word)."""
    if offs is None:
        offs = [s * m * stride + off for s in range(world)]
    offs = [int(o) for o in offs] + [0] * (4 - len(offs))
    n_err = 0 if err_ctl is None else err_ctl.numel() // 8
    ctl = err_ctl if err_ctl is not None else topk_ids
    dst = err_dst if (err_dst is not None and err_code) else topk_ids
    module(1, 0).merge(
        cand,
        world,
        m,
        stride,
        int(per),
        int(line),
        *offs,
        float(route_scale),
        float(eps),
        topk_w,
        topk_ids,
        packed,
        ctl,
        int(n_err),
        dst,
        int(err_code) if err_dst is not None else 0,
    )

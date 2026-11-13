# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MoE front roles of one GPU's owned rows in one launch (csrc/wfc_c4f.cuh): development form, not wired into the server.

One cooperative launch (one CTA per SM) runs, per CTA: the pooler as fp32 chunk partials (bf16 warp-MMA chains per
512-column K chunk), the TF32 router contract for the CTA's expert tiles, the chunk-order logit sum + routing of one owned
row, the norm + NVFP4 quant of one owned row from the 16 chunk partials, and (optionally, one CTA) the per-step expert
assignment over a replica pool (``muse_vr_plugin.bal``). Roles are selected per launch with a mask (ROLE_*).
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import torch

# The SGLang names this module reads (test/unit/test_contract.py pins them against the supported SGLang)
SGLANG_READS = {
    "kernels/jit/utils/__init__.py": ("load_jit",),
    "kernels/jit/utils/compile/paths.py": ("KERNEL_PATH",),
}


_CSRC = Path(__file__).resolve().parents[1] / "csrc"
_CUH = _CSRC / "wfc_c4f.cuh"
K, NE, HID, CHUNKS, XP_PAD = 8192, 256, 4096, 16, 520
ROLE_POOL, ROLE_ROUTER, ROLE_BAL, ROLE_NO_PRESTAGE = 1, 2, 4, 8
NUM_STAMPS = 8 + 64 + 64 + 4 * 256
S_GO, S_ASG, S_BAL0, S_IDS, S_TOK, S_PEND = 0, 1, 2, 8, 72, 136
S_GOSEEN, S_RTR_END, S_SUM = S_PEND + 256, S_PEND + 512, S_PEND + 768


def _stamp_dir():
    """Timing builds only: a directory holding the (private) stamp header; unset = production build, no stamps."""
    d = os.environ.get("MUSE_VR_WFC_C4F_STAMPS_DIR", "").strip()
    return d or None


@functools.lru_cache(maxsize=None)
def module():
    from sglang.kernels.jit.utils import load_jit
    from sglang.kernels.jit.utils.compile.paths import KERNEL_PATH

    from sgl_nvidia_mega.muse_spark.components.sgl_compat import csrc_include_paths

    sd = _stamp_dir()
    plant = (
        os.environ.get("MUSE_VR_WFC_C4F_TEST_PLANT", "") == "1"
    )  # tests only: the planted-race control build
    flags = (["-DMUSE_VR_WFC_C4F_STAMPS"] if sd else []) + (
        ["-DMUSE_VR_C4F_TEST_PLANT_RACE"] if plant else []
    )
    extra = {"extra_cuda_cflags": flags} if flags else {}
    return load_jit(
        "muse_vr_wfc_c4f" + ("_t" if sd else "") + ("_plant" if plant else ""),
        cuda_files=[str(_CUH)],
        cuda_wrappers=[
            ("run", "WfcC4F::run"),
            ("prep", "WfcC4F::prep"),
            ("ref_pool", "WfcC4F::ref_pool"),
            ("ref_quant", "WfcC4F::ref_quant"),
        ],
        extra_include_paths=csrc_include_paths(KERNEL_PATH / "csrc")
        + [str(_CSRC)]
        + ([sd] if sd else []),
        **extra,
    )


def pooler_frag(w: torch.Tensor) -> torch.Tensor:
    """Pooler weight [4096, 8192] bf16 -> B-fragment order [16 chunks][512 tiles][32 k16 steps][32 lanes][2 regs][2] bf16:
    lane = 4 * (column % 8) + t, register h holds k = 16 * step + 8 h + 2 t + (0, 1) of chunk c."""
    assert w.shape == (HID, K) and w.dtype == torch.bfloat16
    v = w.view(HID // 8, 8, CHUNKS, 32, 2, 4, 2)  # tile, g, c, step, h, t, e
    return v.permute(2, 0, 3, 1, 5, 4, 6).contiguous()


def pooler_input_image(x: torch.Tensor, rpad: int) -> torch.Tensor:
    """Pooler input [R, 8192] bf16 -> [16][rpad][520] bf16 (chunk-major, rows >= R and columns >= 512 zero)."""
    r = x.shape[0]
    img = torch.zeros(CHUNKS, rpad, XP_PAD, dtype=torch.bfloat16, device=x.device)
    img[:, :r, :512] = x.view(r, CHUNKS, 512).permute(1, 0, 2)
    return img


class State:
    """Buffers of one owned-row count R (and one step row count for the balance role); the control words persist."""

    def __init__(self, R: int, device, Mtot: int = 0):
        d = device
        self.R = R
        self.rpad = (R + 15) // 16 * 16
        self.mt = self.rpad // 16
        self.Mtot = max(Mtot, R)
        self.xr_img = torch.zeros(
            CHUNKS, self.mt, 64, 32, 4, dtype=torch.float32, device=d
        )
        self.wr_frag = torch.zeros(
            CHUNKS, NE // 8, 64, 32, 2, dtype=torch.int32, device=d
        )
        self.board = torch.zeros(CHUNKS, R, NE, dtype=torch.int64, device=d)
        self.topk_w = torch.zeros(R, 8, dtype=torch.float32, device=d)
        self.topk_ids = torch.zeros(R, 8, dtype=torch.int32, device=d)
        self.packed = torch.zeros(R, 8, dtype=torch.int32, device=d)
        self.rwords = torch.zeros(R * 8, dtype=torch.int64, device=d)
        self.part = torch.zeros(CHUNKS, R, HID, dtype=torch.float32, device=d)
        self.pflag = torch.zeros(256, dtype=torch.int64, device=d)
        self.sflag = torch.zeros(256, dtype=torch.int64, device=d)
        self.fp4 = torch.zeros(R, HID // 2, dtype=torch.uint8, device=d)
        self.sf = torch.zeros(R, HID // 16, dtype=torch.uint8, device=d)
        self.pts = torch.zeros(R, dtype=torch.float32, device=d)
        self.qwords = torch.zeros(R, dtype=torch.int64, device=d)
        self.pooled = torch.zeros(R, HID, dtype=torch.bfloat16, device=d)
        self.bal_ids = torch.zeros(self.Mtot, 8, dtype=torch.int32, device=d)
        self.bal_table = torch.zeros(NE, dtype=torch.int32, device=d)
        self.bal_vids = torch.zeros(self.Mtot, 8, dtype=torch.int32, device=d)
        self.bal_asg = torch.zeros(NE, dtype=torch.uint8, device=d)
        self.bal_word = torch.zeros(1, dtype=torch.int64, device=d)
        self.ctl = torch.zeros(8, dtype=torch.int64, device=d)
        self.stamps = torch.zeros(NUM_STAMPS, dtype=torch.int64, device=d)
        self.nwords = torch.zeros(max(R, 1), dtype=torch.int64, device=d)
        self.nnew_stream = torch.zeros(R, K, dtype=torch.float32, device=d)
        self.boundary = None
        self.xp_img = None
        self.wp_frag = None
        self.bias = None
        self.qweight = None

    def load(
        self,
        xr: torch.Tensor,
        wr: torch.Tensor,
        bias: torch.Tensor,
        xp: torch.Tensor,
        wp_frag: torch.Tensor,
        qweight=None,
    ):
        """xr [R, 8192] fp32 (router input), wr [256, 8192] fp32, bias [256] fp32, xp [R, 8192] bf16 (pooler input),
        wp_frag = pooler_frag(pooler weight), qweight [4096] bf16 or None."""
        module().prep(
            xr.contiguous(), self.R, self.xr_img, wr.contiguous(), self.wr_frag
        )
        self.xp_img = pooler_input_image(xp, self.rpad)
        self.wp_frag = wp_frag
        self.bias = bias.contiguous()
        self.qweight = qweight

    def set_boundary(
        self,
        branch,
        stream,
        *,
        vt: int,
        pre_weight=None,
        alpha,
        beta,
        post_eps: float,
        pre_eps: float,
    ):
        """Front-of-MoE mode: the launch first runs the owner-row boundary role (ops/wfc_c4_norm.py arithmetic, the served
        attention -> MoE variant) on branch [R, 8192] bf16 / stream [R, 8192] fp32 and writes the input images itself; vt =
        the boundary thread count of the served launch (ops.wfc_c4_norm.served_threads(step rows)). None turns it off."""
        if branch is None:
            self.boundary = None
            return
        self.boundary = dict(
            branch=branch.contiguous(),
            stream=stream.contiguous(),
            vt=int(vt),
            pre_weight=pre_weight,
            alpha=alpha.contiguous(),
            beta=beta.contiguous(),
            post_eps=float(post_eps),
            pre_eps=float(pre_eps),
        )
        self.xr_img.zero_()
        self.xp_img = torch.zeros(
            CHUNKS, self.rpad, XP_PAD, dtype=torch.bfloat16, device=branch.device
        )

    def run(
        self,
        *,
        grid: int,
        roles: int,
        ns: int = 3,
        stile: int = 4,
        oversize: bool = False,
        prego_ns: int = 0,
        tmo_scale: int = 1,
        mult: float = 1.0,
        route_scale: float = 1.0,
        div_eps: float = 0.0,
        qeps: float = 1e-6,
        gsi: float = 1.0 / (448.0 * 6.0),
        want_pooled: bool = True,
        bal_pool: int = 64,
        bal_cta: int = -1,
        bal_pass_rows: int = 48,
    ):
        """want_pooled: kept for callers; the pooled bf16 rows are always written (the quant role reads them)."""
        q = self.qweight
        bd = self.boundary
        nb = bd["branch"] if bd else self.pooled
        ns_ = bd["stream"] if bd else self.nnew_stream
        npw = (
            bd["pre_weight"]
            if (bd and bd["pre_weight"] is not None)
            else self.pooled[0]
        )
        na = bd["alpha"] if bd else self.pts[:1]
        nbt = bd["beta"] if bd else self.pts[:1]
        module().run(
            self.xr_img,
            self.wr_frag,
            self.bias,
            self.wp_frag,
            self.xp_img,
            q if q is not None else self.pooled[0],
            int(q is not None),
            self.board,
            self.topk_w,
            self.topk_ids,
            self.packed,
            self.rwords,
            self.part,
            self.pflag,
            self.sflag,
            self.fp4,
            self.sf,
            self.pts,
            self.qwords,
            self.pooled,
            self.bal_ids,
            self.bal_table,
            self.bal_vids,
            self.bal_asg,
            self.bal_word,
            self.ctl,
            self.stamps,
            self.R,
            int(grid),
            int(ns),
            int(stile),
            int(bool(oversize)),
            int(roles),
            int(prego_ns),
            int(tmo_scale),
            self.Mtot,
            int(bal_pool),
            int(bal_cta if bal_cta >= 0 else grid - 1),
            int(bal_pass_rows),
            float(mult),
            float(route_scale),
            float(div_eps),
            float(qeps),
            float(gsi),
            int(bd["vt"]) if bd else 0,
            nb,
            ns_,
            self.nnew_stream,
            npw,
            int(bool(bd) and bd["pre_weight"] is not None),
            na,
            nbt,
            float(bd["post_eps"]) if bd else 0.0,
            float(bd["pre_eps"]) if bd else 0.0,
            self.nwords,
        )


def ref_pool(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Test-only reference chunk partials [16, R, 4096] fp32 of x [R, 8192] bf16 against w [4096, 8192] bf16."""
    r = x.shape[0]
    part = torch.zeros(CHUNKS, r, HID, dtype=torch.float32, device=x.device)
    module().ref_pool(x.contiguous(), r, w.contiguous(), part)
    return part


def ref_quant(
    part: torch.Tensor,
    qweight=None,
    qeps: float = 1e-6,
    gsi: float = 1.0 / (448.0 * 6.0),
):
    """Test-only reference norm + quant of every row from the 16 partial slices: (fp4, sf, pts, pooled bf16)."""
    r = part.shape[1]
    d = part.device
    fp4 = torch.zeros(r, HID // 2, dtype=torch.uint8, device=d)
    sf = torch.zeros(r, HID // 16, dtype=torch.uint8, device=d)
    pts = torch.zeros(r, dtype=torch.float32, device=d)
    pooled = torch.zeros(r, HID, dtype=torch.bfloat16, device=d)
    module().ref_quant(
        part,
        r,
        qweight if qweight is not None else pooled[0],
        int(qweight is not None),
        fp4,
        sf,
        pts,
        pooled,
        float(qeps),
        float(gsi),
    )
    return fp4, sf, pts, pooled

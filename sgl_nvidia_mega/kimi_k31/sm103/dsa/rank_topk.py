# SPDX-License-Identifier: Apache-2.0
"""Exact FP32 top-k with adaptive binary scaling of coarse histogram keys."""

from functools import cache
from pathlib import Path

# Names read from SGLang. A contract test checks these before GPU use.
SGLANG_READS = {
    "kernels/jit/utils/__init__.py": ("load_jit",),
}


@cache
def _load():
    import sglang
    from sglang.kernels.jit.utils import load_jit

    return load_jit(
        "k31_rank_coarse_v2",
        cuda_files=[str(Path(__file__).with_suffix(".cuh"))],
        cuda_wrappers=[("run", "k31_rank_topk::Binding::run")],
        extra_include_paths=[str(Path(sglang.__file__).parent / "kernels/jit/csrc")],
        extra_cuda_cflags=["-DSGL_TOPK_STABLE_TIES=0"],
    )


def select_rank_topk(scores, lengths, table, selected, plan, *, queries_per_request=1):
    _load().run(scores, lengths, table, selected, plan, int(queries_per_request))
    return selected

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Vendored SGLang top-k v2 (#47960 exactness modes; exact mode 1), for K31MK_VERIFY_TOPK=x47960.

topk_v2.cuh / topk_impl.cuh are the files of public sgl-project/sglang commit b2833196d8 (main, #48501, which carries #47960's
merge aaca6d0791 and the exact cluster path): python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh and
python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/topk_impl.cuh, with six marked local edits ("k31 edit N"):
1. topk_v2.cuh includes the topk_impl.cuh next to it, not the installed sgl_kernel one;
2. topk_small_batch_cluster_kernel's enable_smem_spilling() pragma is removed (ptxas refuses it together with dynamic smem);
3. that kernel's staging buffer s_topk_indices[kMaxTopK] lives in dynamic smem (kMaxTopK * 4 bytes, opt-in attribute set at both
   launch sites), so it builds on the cu134 image (CUDA 13.4, sm_100f), where the static form fails ptxas
   (static smem 0xcfe0 / 0xdca0 > 0xc000);
4. topk_impl.cuh uses __fadd2_rn where public uses math::add_f32x2, which the served SGLang headers do not have;
5. topk_impl.cuh includes k31_barrier.cuh (next to it), which takes sgl_kernel/ptx/barrier.cuh when the SGLang tree has it, else
   sgl_kernel/mbarrier.cuh (both define the mbar_init / to_shared it takes, with the same bodies).
6. topk_v2.cuh's TopKKernel host struct takes the mode as a second template argument (as SGLang's #47960 port had it), so its
   host symbols differ from SGLang's own topk_v2 module of the same mode: the two modules' once-per-device launch setup
   (runtime::init_per_device, keyed on the setup lambda's type) stays apart.
topk_v2_tkpush.cuh (built by ../tkpush.py) is a second copy of the same topk_v2.cuh with edits 1-3 and the tag-push additions
marked "k31 TKPUSH": its own inline namespace, a tagged mailbox-store epilogue in the small-batch cluster kernel (PAGE_TABLE
mode only) and the transform_paged_push host entry.
Each SGL_TOPK_MODE builds its symbols in its own inline namespace, so this module and the served topk_v2 module coexist in one
process. Built at prepare time (host nvcc); never inside a graph capture.
"""
from functools import cache
from pathlib import Path

# SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/jit/utils/compile/loader.py": ("load_jit",),
    "kernels/jit/utils/__init__.py": ("is_arch_support_pdl", "make_cpp_args"),
    "kernels/jit/utils/occupancy.py": ("NoSchedulableClustersError", "get_max_active_clusters"),
}

MODE_EXACT = 1
HERE = Path(__file__).resolve().parent


@cache
def load(mode: int = MODE_EXACT):
    """The JIT module (same build recipe as SGLang's _jit_topk_v2_module: mode macro, split-compile off, cluster occupancy macros)."""
    from sglang.kernels.jit.utils import is_arch_support_pdl, load_jit, make_cpp_args
    from sglang.kernels.jit.utils.occupancy import NoSchedulableClustersError, get_max_active_clusters

    flags = [f"-DSGL_TOPK_MODE={mode}", "-Xptxas=--split-compile=0"]
    if is_arch_support_pdl():
        for cluster_size, occupancy in ((8, 2), (16, 1)):
            try:
                n = get_max_active_clusters(cluster_size=cluster_size, occupancy=occupancy)
            except NoSchedulableClustersError:
                n = 0
            flags.append(f"-DSGL_TOPK_V2_MAX_C{cluster_size}_OCC{occupancy}={n}")
    args = make_cpp_args(is_arch_support_pdl(), mode)  # the module's identity (name); one module per mode
    kernel = f"TopKKernel<{args}>"  # k31 edit 6: the mode is a template argument of the host struct
    return load_jit(
        f"k31_x47960_topk_v2_m{mode}",
        *args,
        extra_cuda_cflags=flags,
        cuda_files=[str(HERE / "topk_v2.cuh")],
        extra_include_paths=[str(HERE)],  # "topk_impl.cuh" resolves to the vendored copy even if the source is staged elsewhere
        cuda_wrappers=[("topk_transform_paged", f"{kernel}::transform_paged"), ("topk_plan", f"{kernel}::plan")],
    )


class X47960Select:
    """Exact top-2048 (mode 1) over per-row causal lengths with the per-row page table, paged output (the call stock
    topk_transform_paged_v2 makes). The plan comes from this module (its own routing); at rows <= its persistent-cluster
    count the plan call returns on the host without a launch."""

    def __init__(self, device, rows, mode: int = MODE_EXACT):
        import torch

        self.mod = load(mode)
        self.plan = torch.full((rows + 1, 2), -1, dtype=torch.int32, device=device)

    def run(self, scores, causal_lengths, row_table, selected, page_size=64):
        self.mod.topk_plan(causal_lengths, self.plan, -1)
        self.mod.topk_transform_paged(scores, causal_lengths, row_table, selected, page_size, self.plan, None)
        return selected

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""K3.1 DSA query-front index query, IQ-PIPE form (index_qp.cuh; K31MK_DSA_IQ_PIPE): index_q.cuh's run_qf contract and
outputs bit for bit, with w_qa read as 16-byte vectors, the BF16 normalization packed in pairs and the wq_b fragments in registers before
griddepcontrol.wait. 9-16 rows run index_q.cuh's kernel. ``module()`` has the same ``run_qf`` as index_q._load()."""

import hashlib
from functools import cache
from pathlib import Path

# SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {"kernels/jit/utils/compile/loader.py": ("load_jit",)}


@cache
def module():
    import sglang
    from sglang.kernels.jit.utils import load_jit

    here = Path(__file__)
    # index_qp.cuh #includes index_q.cuh: key the module on both files' bytes, so a changed index_q.cuh never reuses a stale build
    key = hashlib.md5(here.with_suffix(".cuh").read_bytes() + (here.parent / "index_q.cuh").read_bytes()).hexdigest()[:10]
    return load_jit(
        f"k31_index_qp_{key}",
        cuda_files=[str(here.with_suffix(".cuh"))],
        cuda_wrappers=[("run_qf", "k31_index_qp::BindingP::run_qf")],
        extra_include_paths=[str(Path(sglang.__file__).parent / "kernels/jit/csrc")],
    )

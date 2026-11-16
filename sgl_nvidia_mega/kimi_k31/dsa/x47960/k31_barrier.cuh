// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// k31 edit 5 (see __init__.py): the barrier helpers topk_impl.cuh takes (mbar_init, to_shared). Newer SGLang trees define them
// in sgl_kernel/ptx/barrier.cuh and no longer ship sgl_kernel/mbarrier.cuh; older served trees have only mbarrier.cuh. Both
// define them in sglang::device::ptx with the same bodies. A separate file keeps topk_impl.cuh's line numbers unchanged.
#pragma once
#if __has_include(<sgl_kernel/ptx/barrier.cuh>)
#include <sgl_kernel/ptx/barrier.cuh>
#else
#include <sgl_kernel/mbarrier.cuh>
#endif

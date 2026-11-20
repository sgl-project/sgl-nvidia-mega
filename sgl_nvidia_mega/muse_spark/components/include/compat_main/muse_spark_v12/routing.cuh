// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Forwarding header: on SGLang's main source layout the Muse Spark JIT kernels live in muse_spark/; this keeps the
// plugin's includes of muse_spark_v12/routing.cuh resolving to the same kernel source (added to the include path only
// on that layout).
#pragma once
#include <muse_spark/routing.cuh>

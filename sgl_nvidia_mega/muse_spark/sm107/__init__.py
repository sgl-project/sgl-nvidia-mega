# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The Muse Spark Rubin (sm_107) component set: opt-in SGLang hooks built for Rubin, with the kernels, launchers and
tactic tables only they use.

``muse_spark.components.register_all`` imports these modules from here and registers them, in its load order next to the
GB300 set, only on a device of an architecture ``spec.COMPONENT_ISAS`` lists for this set. They use the shared helpers
(``components.bed``, ``components.sgl_compat``) and build on some of the GB300 set's modules and headers.
"""

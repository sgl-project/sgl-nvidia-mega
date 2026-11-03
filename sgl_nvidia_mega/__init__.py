# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""NVIDIA megakernels for SGLang: per model, the decode step's own code and the
kernels that only make sense inside a fused step.

SGLang loads the package as a general plugin (``pyproject.toml``: the
``sglang.srt.plugins`` entry point, ``plugin.register``), which registers each
model's spec with ``sglang.kernels.plugin_kernels.nvidia`` in every SGLang process. With SGLang's
``--enable-nvidia-megakernel`` flag, SGLang checks the setup at argument resolution,
builds the spec after it prepares the model, and routes covered steps through it;
without the flag no spec is imported or built. The shared CUDA runtime is in
``common``; each model's step code and kernels per ISA are in ``<model>/``.
"""

__version__ = "0.1.0.dev0"

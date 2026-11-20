# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import os

if os.environ.get("GLM5MK") == "1":
    import sgl_nvidia_mega.glm52.sm107 as glm52_sm107

    glm52_sm107.install()

try:
    import apport_python_hook
except ImportError:
    pass
else:
    apport_python_hook.install()

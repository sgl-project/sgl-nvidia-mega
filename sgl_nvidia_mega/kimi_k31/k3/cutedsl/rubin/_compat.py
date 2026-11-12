# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""vLLM's ``init_logger`` for the vendored engines: a stdlib logger with ``warning_once`` (process scope)."""
import functools
import logging


class _Logger(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        kwargs.pop("scope", None)
        return msg, kwargs

    @functools.lru_cache(maxsize=None)  # noqa: B019 (one warning per message and arguments, like vLLM's)
    def warning_once(self, msg, *args, **kwargs):
        self.warning(msg, *args, **kwargs)


def init_logger(name):
    return _Logger(logging.getLogger(name), {})

# SPDX-License-Identifier: Apache-2.0
"""Cache views shared by the MLA runner harness and future DSA binding."""


def cache_views(storage):
    """Return zero-copy KV/PE views; never repack a cache to change its layout."""
    import torch

    if storage.dtype != torch.float8_e4m3fn:
        raise ValueError("GLM5 raw MLA cache must be torch.float8_e4m3fn")
    if storage.ndim < 2 or storage.shape[-1] != 576 or not storage.is_contiguous():
        raise ValueError("GLM5 raw MLA cache must have contiguous 576-byte rows")
    rows = storage.view(-1, 576)
    return rows[:, :512], rows[:, 512:]

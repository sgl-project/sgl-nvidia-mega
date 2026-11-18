# SPDX-License-Identifier: Apache-2.0
"""Row-sized scratch shared by one layer's sparse verify profiles."""

import torch


class RowScratch:
    """Buffers of ``max_rows`` rows; a profile of fewer rows gets the leading rows of each.

    One layer's profiles replay one at a time (one row capacity per step), so they can share
    storage; each buffer keeps its own name, so two buffers of one profile never alias.
    """

    def __init__(self, device, max_rows):
        self.device, self.max_rows, self.buffers = device, max_rows, {}

    def rows(self, name, rows, shape, dtype, *, zero=False):
        if rows > self.max_rows:
            raise ValueError(
                f"{name}: {rows} rows exceed the shared scratch's {self.max_rows}"
            )
        key = (name, tuple(shape), dtype)
        buffer = self.buffers.get(key)
        if buffer is None:
            make = torch.zeros if zero else torch.empty
            buffer = self.buffers[key] = make(
                (self.max_rows, *shape), dtype=dtype, device=self.device
            )
        return buffer[:rows]


def row_buffer(scratch, name, rows, shape, dtype, device, *, zero=False):
    """``rows`` x ``shape`` of ``dtype``: a view of ``scratch`` when given, else its own buffer."""
    if scratch is None:
        make = torch.zeros if zero else torch.empty
        return make((rows, *shape), dtype=dtype, device=device)
    return scratch.rows(name, rows, shape, dtype, zero=zero)

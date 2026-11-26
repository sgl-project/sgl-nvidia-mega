# SPDX-License-Identifier: Apache-2.0
"""Target-verify top-512 (SPARSE_SELECT variant 2, SGLANG_QSA_VERIFY_TOPK=sampled).

The source's ``fast_topk`` runs one 1024-thread CTA per row and three or more
passes over it; at verify batches of 44-64 requests (176-256 rows) that is about
one CTA per SM, so each row's own latency sets the launch time. This runs the
prefill select's top-k on the same rows instead: ``qsa_prefill_topk.cuh``, the
source fast_topk's exact output (the same indices in the same slots, smallest-index
ties) from one pass over the row; rows it does not take run fast_topk's own select.
Up to one row per SM it uses 1024-thread blocks; past that, the 512-thread variant
(``qsa_prefill_topk_half.cuh``, four blocks per SM, the same output).
"""

import functools

import torch

_TOPK = 512


@functools.lru_cache(maxsize=None)
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def impl_for(rows: int, device: torch.device) -> str:
    # VR200 (212 SMs), [rows, 65536] logits, 8K-65K keys per row: 1024 threads are faster up
    # to 192 rows (15.9 vs 19.1-19.8 us), 512 threads from 228 rows (21.5-22.3 vs 25.1-25.4 us).
    index = device.index if device.index is not None else torch.cuda.current_device()
    return "sampled" if rows <= _num_sms(index) else "half"


def select_into(logits, lengths, indices, scratch, impl):
    """``fast_topk(logits, lengths, 512, row_starts=None)`` into ``indices`` for decode rows
    (they start at column 0). ``scratch`` is int32 [2 * rows]: the zero row starts, then
    the 512-thread kernel's per-row flags of the rows its second launch serves."""
    from .prefill_select import _topk_launcher

    rows = logits.shape[0]
    starts = scratch[:rows]
    starts.zero_()
    if impl == "half":
        _topk_launcher("half")(
            logits, starts, indices, lengths, scratch[rows : 2 * rows]
        )
    else:
        _topk_launcher("sampled")(logits, starts, indices, lengths)

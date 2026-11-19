# SPDX-License-Identifier: Apache-2.0
"""K3.1 MLA and index-K Parameter views over one packed allocation."""

from sgl_nvidia_mega.kimi_k31.k3.projection_storage import (
    _coalesce_projection_storage,
    _Layout,
    _module_heads,
    _Part,
    mla_layout,
    mla_pitch,
)

_INDEX_K_ROWS = 128


def index_k_offset(heads: int) -> int:
    """First row of index-K, after [qkv-a | gate]: 3648 at TP8, 2880 at TP16."""
    return mla_pitch(heads)


def pitch(heads: int) -> int:
    """Row pitch of the packed [qkv-a | gate | index-K]: 3776 at TP8, 3008 at TP16."""
    return index_k_offset(heads) + _INDEX_K_ROWS


def _layout(heads: int) -> _Layout:
    mla = mla_layout(heads)
    return _Layout(
        name="K3.1 MLA/index-K",
        parts=(*mla.parts, _Part("indexer.wk.weight", _INDEX_K_ROWS)),
        columns=mla.columns,
        dtype=mla.dtype,
    )


def coalesce_index_projection(attention):
    """Preserve Parameter identities; stock decode uses the repointed Wk too."""
    heads = _module_heads(attention, "g_proj.weight", 128)
    root = _coalesce_projection_storage(attention, _layout(heads), device_type="cuda")
    # The stock [Wk | head-gate] alias predates plugin preparation. Its Wk rows
    # now live here; disable that optional alias so stock paths read live weights.
    attention.indexer._wk_gate_weight = None
    return root

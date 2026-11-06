# SPDX-License-Identifier: Apache-2.0
"""The loaded configuration determines dense versus sparse layer selection."""


def is_sparse_layer(config, layer_id: int, *, is_nextn: bool = False) -> bool:
    if layer_id < 0:
        raise ValueError("layer_id must be nonnegative")
    if is_nextn:
        return True
    frequency = config.moe_layer_freq
    if frequency <= 0:
        raise ValueError("moe_layer_freq must be positive")
    return layer_id >= config.first_k_dense_replace and layer_id % frequency == 0

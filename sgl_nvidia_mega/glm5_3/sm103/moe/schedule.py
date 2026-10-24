# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Host schedule tables of the small-M MoE engine (pure Python + torch, built once).

A table holds, per cluster and per number D of distinct routed experts, the cluster's
unit list: [0] the unit count, [1 + k] the code of unit k, ``4 * (rank of the expert
among the distinct experts, ascending id) + quarter``, or -1 for this cluster's quarter
of the shared expert. The kernel stages its cluster's block into shared memory with one
bulk copy."""

from __future__ import annotations

import torch

SH_CLUSTERS = 4

# mode G1 (one request, M <= 4): D <= 32, at most 32 units per cluster
G1_ROW = 34  # [0] units, [1..32] codes, [33] partial count (unused by the engine)
G1_D = 33
G1_MAX_UNITS = 32
G1_BLOCK_BYTES = 2256  # 33 x 34 int16 = 2244 B, padded to a multiple of 16


def g1_table(nclusters: int, shared: bool) -> torch.Tensor:
    """int16 [nclusters][G1_BLOCK_BYTES / 2]. Without the shared expert, unit u goes to
    cluster u % nclusters. With it (nclusters >= 8), clusters 0..3 run one shared
    quarter each first; routed units fill clusters 4.. round-robin up to a per-cluster
    cap, the remainder going round-robin to clusters 0..3."""
    nc = nclusters
    with_shared = shared and nc >= 2 * SH_CLUSTERS
    tab = torch.zeros(nc, G1_D, G1_ROW, dtype=torch.int16)
    for d in range(G1_D):
        units = 4 * d
        lists: list[list[int]] = [[] for _ in range(nc)]
        if not with_shared:
            for u in range(units):
                lists[u % nc].append(u)
            parts = min(nc, units)
        else:
            routed_clusters = nc - SH_CLUSTERS
            cap = min((units + 3 * SH_CLUSTERS + nc - 1) // nc, G1_MAX_UNITS)
            lim = min(units, routed_clusters * cap)
            for cl in range(SH_CLUSTERS):
                lists[cl].append(-1)
            for u in range(lim):
                lists[SH_CLUSTERS + u % routed_clusters].append(u)
            for u in range(lim, units):
                lists[(u - lim) % SH_CLUSTERS].append(u)
            parts = SH_CLUSTERS + min(routed_clusters, units)
        for cl in range(nc):
            if len(lists[cl]) > G1_MAX_UNITS:
                raise ValueError(
                    f"schedule: cluster {cl} needs {len(lists[cl])} > "
                    f"{G1_MAX_UNITS} units at D={d}"
                )
            tab[cl, d, 0] = len(lists[cl])
            for k, u in enumerate(lists[cl]):
                tab[cl, d, 1 + k] = u
            tab[cl, d, 33] = parts
    flat = tab.view(nc, -1)
    out = torch.zeros(nc, G1_BLOCK_BYTES // 2, dtype=torch.int16)
    out[:, : flat.shape[1]] = flat
    return out.contiguous()


# mode G2 (two requests, M <= 8): D <= 64 list entries, at most 15 units per cluster
G2_ROW = 16  # [0] units, [1..15] codes
G2_D = 65
G2_MAX_UNITS = 15
G2_BLOCK_BYTES = G2_D * G2_ROW * 2  # 2080


def g2_table(nclusters: int, shared: bool) -> torch.Tensor:
    """int16 [nclusters][G2_BLOCK_BYTES / 2]: the G1 assignment rule over D <= 64
    entries with a 15-unit cluster cap."""
    nc = nclusters
    with_shared = shared and nc >= 2 * SH_CLUSTERS
    tab = torch.zeros(nc, G2_D, G2_ROW, dtype=torch.int16)
    for d in range(G2_D):
        units = 4 * d
        lists: list[list[int]] = [[] for _ in range(nc)]
        if not with_shared:
            for u in range(units):
                lists[u % nc].append(u)
        else:
            routed_clusters = nc - SH_CLUSTERS
            cap = min((units + 3 * SH_CLUSTERS + nc - 1) // nc, G2_MAX_UNITS)
            lim = min(units, routed_clusters * cap)
            for cl in range(SH_CLUSTERS):
                lists[cl].append(-1)
            for u in range(lim):
                lists[SH_CLUSTERS + u % routed_clusters].append(u)
            for u in range(lim, units):
                lists[(u - lim) % SH_CLUSTERS].append(u)
        for cl in range(nc):
            if len(lists[cl]) > G2_MAX_UNITS:
                raise ValueError(
                    f"schedule: cluster {cl} needs {len(lists[cl])} > "
                    f"{G2_MAX_UNITS} units at D={d}"
                )
            tab[cl, d, 0] = len(lists[cl])
            for k, u in enumerate(lists[cl]):
                tab[cl, d, 1 + k] = u
    return tab.view(nc, -1).contiguous()


# mode G4 (two to four requests, M <= 16): D <= 128 list entries, at most 17 units per
# cluster; the shared quarter counts as 4 routed units in the cap
G4_ROW = 18  # [0] units, [1..17] codes
G4_D = 136  # D = 0..128 (+ padding)
G4_MAX_UNITS = 17
G4_BLOCK_BYTES = G4_D * G4_ROW * 2  # 4896
G4_SHARED_WEIGHT = 4


def g4_table(nclusters: int, shared: bool) -> torch.Tensor:
    """int16 [nclusters][G4_BLOCK_BYTES / 2]: the G1 assignment rule over D <= 128
    entries with the shared quarter weighted as 4 units and a 17-unit cluster cap."""
    nc = nclusters
    with_shared = shared and nc >= 2 * SH_CLUSTERS
    tab = torch.zeros(nc, G4_D, G4_ROW, dtype=torch.int16)
    for d in range(129):
        units = 4 * d
        lists: list[list[int]] = [[] for _ in range(nc)]
        if not with_shared:
            for u in range(units):
                lists[u % nc].append(u)
        else:
            routed_clusters = nc - SH_CLUSTERS
            cap = min(
                (units + G4_SHARED_WEIGHT * SH_CLUSTERS + nc - 1) // nc, G4_MAX_UNITS
            )
            lim = min(units, routed_clusters * cap)
            for cl in range(SH_CLUSTERS):
                lists[cl].append(-1)
            for u in range(lim):
                lists[SH_CLUSTERS + u % routed_clusters].append(u)
            for u in range(lim, units):
                lists[(u - lim) % SH_CLUSTERS].append(u)
        for cl in range(nc):
            if len(lists[cl]) > G4_MAX_UNITS:
                raise ValueError(
                    f"schedule: cluster {cl} needs {len(lists[cl])} > "
                    f"{G4_MAX_UNITS} units at D={d}"
                )
            tab[cl, d, 0] = len(lists[cl])
            for k, u in enumerate(lists[cl]):
                tab[cl, d, 1 + k] = u
    return tab.view(nc, -1).contiguous()

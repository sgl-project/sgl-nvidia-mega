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

# the shared quarter's weight in routed units in the G1 / G2 cap (served 3; T97: 2 measured faster at the D where
# the table changes, bitwise identical: only the unit-to-cluster assignment moves)
G12_SHARED_WEIGHT = 3

# mode G1 (one request, M <= 4): D <= 32, at most 32 units per cluster
G1_ROW = 34  # [0] units, [1..32] codes, [33] partial count (unused by the engine)
G1_D = 33
G1_MAX_UNITS = 32
G1_BLOCK_BYTES = 2256  # 33 x 34 int16 = 2244 B, padded to a multiple of 16


def g1_table(
    nclusters: int,
    shared: bool,
    shared_weight: int = G12_SHARED_WEIGHT,
    quarters: int = 4,
) -> torch.Tensor:
    """int16 [nclusters][G1_BLOCK_BYTES / 2]. Without the shared expert, unit u goes to
    cluster u % nclusters. With it (nclusters >= 2 * quarters), clusters 0..quarters-1
    run one shared quarter each first; routed units fill clusters quarters.. round-robin
    up to a per-cluster cap, the remainder going round-robin to clusters 0..quarters-1."""
    nc = nclusters
    sh_clusters = quarters
    with_shared = shared and nc >= 2 * sh_clusters
    tab = torch.zeros(nc, G1_D, G1_ROW, dtype=torch.int16)
    for d in range(G1_D):
        units = quarters * d
        lists: list[list[int]] = [[] for _ in range(nc)]
        if not with_shared:
            for u in range(units):
                lists[u % nc].append(u)
            parts = min(nc, units)
        else:
            routed_clusters = nc - sh_clusters
            cap = min(
                (units + shared_weight * sh_clusters + nc - 1) // nc, G1_MAX_UNITS
            )
            lim = min(units, routed_clusters * cap)
            for cl in range(sh_clusters):
                lists[cl].append(-1)
            for u in range(lim):
                lists[sh_clusters + u % routed_clusters].append(u)
            for u in range(lim, units):
                lists[(u - lim) % sh_clusters].append(u)
            parts = sh_clusters + min(routed_clusters, units)
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


def g2_table(
    nclusters: int,
    shared: bool,
    shared_weight: int = G12_SHARED_WEIGHT,
    quarters: int = 4,
) -> torch.Tensor:
    """int16 [nclusters][G2_BLOCK_BYTES / 2]: the G1 assignment rule over D <= 64
    entries with a 15-unit cluster cap."""
    nc = nclusters
    sh_clusters = quarters
    with_shared = shared and nc >= 2 * sh_clusters
    tab = torch.zeros(nc, G2_D, G2_ROW, dtype=torch.int16)
    for d in range(G2_D):
        units = quarters * d
        lists: list[list[int]] = [[] for _ in range(nc)]
        if not with_shared:
            for u in range(units):
                lists[u % nc].append(u)
        else:
            routed_clusters = nc - sh_clusters
            cap = min(
                (units + shared_weight * sh_clusters + nc - 1) // nc, G2_MAX_UNITS
            )
            lim = min(units, routed_clusters * cap)
            for cl in range(sh_clusters):
                lists[cl].append(-1)
            for u in range(lim):
                lists[sh_clusters + u % routed_clusters].append(u)
            for u in range(lim, units):
                lists[(u - lim) % sh_clusters].append(u)
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


def g4_table(
    nclusters: int,
    shared: bool,
    shared_weight: int = G4_SHARED_WEIGHT,
    fill_shared: bool = False,
    snake: bool = False,
) -> torch.Tensor:
    """int16 [nclusters][G4_BLOCK_BYTES / 2]: the G1 assignment rule over D <= 128
    entries with the shared quarter weighted as ``shared_weight`` units (default 4)
    and a 17-unit cluster cap. ``fill_shared``: the shared clusters also take up to
    cap - shared_weight routed units each (the list's last units), the routed clusters
    the rest, so fewer routed clusters carry the cap. ``snake``: the routed units go to
    the routed clusters expert by expert (an expert's four quarters to four adjacent
    clusters), the direction reversed every round, so the class order's heavy experts
    (most requests, most exchanges) do not all land on the same clusters."""
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
                (units + shared_weight * SH_CLUSTERS + nc - 1) // nc, G4_MAX_UNITS
            )
            lim = min(units, routed_clusters * cap)
            if fill_shared:
                lim = min(lim, units - SH_CLUSTERS * max(cap - shared_weight, 0))
                lim = max(lim, 0)
            for cl in range(SH_CLUSTERS):
                lists[cl].append(-1)
            groups = routed_clusters // 4
            for u in range(lim):
                if snake and routed_clusters % 4 == 0:
                    g = (u // 4) % groups
                    if ((u // 4) // groups) % 2 == 1:
                        g = groups - 1 - g
                    lists[SH_CLUSTERS + 4 * g + u % 4].append(u)
                else:
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

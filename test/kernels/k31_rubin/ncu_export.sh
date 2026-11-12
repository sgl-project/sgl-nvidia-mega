#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Export details + SASS source counters of existing Nsight Compute reports. Usage: ncu_export.sh OUTDIR
OUT=$1; NCU=$(command -v ncu || ls /usr/local/cuda/bin/ncu)
for REP in "$OUT"/*.ncu-rep*; do b=${REP%.ncu-rep*}
  "$NCU" -i "$REP" --page details > "${b}_details.txt" 2>&1
  "$NCU" -i "$REP" --page source --csv --print-source sass > "${b}_source.csv" 2>&1
done
echo exported > "$OUT/EXPORTED"

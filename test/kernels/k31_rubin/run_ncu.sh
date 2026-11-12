#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# One kernel launch under Nsight Compute (full set + source counters) from a pytest node id, on one GPU.
# Usage: run_ncu.sh OUTDIR KERNEL_REGEX NODEID [NODEID...]   (e.g. KdaSplit 'test/.../test_k31_kda.py::test_kda_verify[1]')
set -u
OUT=$1; RE=$2; shift 2
mkdir -p "$OUT"; chmod 700 "$OUT"
HERE=$(cd "$(dirname "$0")" && pwd)
PLUGIN=$(cd "$HERE/../../.." && pwd)
CACHE=/tmp/mkport_ncu_${SLURM_JOB_ID:-x}
mkdir -p "$CACHE/home"
export HOME=$CACHE/home TRITON_CACHE_DIR=$CACHE/triton CUDA_CACHE_PATH=$CACHE/cuda
export PYTHONPATH=$PLUGIN${PYTHONPATH:+:$PYTHONPATH} PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 K31_TIME=0
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*}
NCU=$(command -v ncu || ls /usr/local/cuda/bin/ncu 2>/dev/null)
i=0
for node in "$@"; do
  i=$((i + 1))
  ( cd "$PLUGIN" && timeout 1200 "$NCU" --kernel-name "regex:$RE" --launch-count 1 --set full --import-source yes \
      -f -o "$OUT/ncu_$i" python3 -m pytest -q -p no:cacheprovider "$node" ) > "$OUT/ncu_$i.log" 2>&1
  echo "node $node rc=$?" >> "$OUT/SUMMARY.txt.tmp"
  REP=$(ls "$OUT"/ncu_$i.ncu-rep* 2>/dev/null | head -1)
  "$NCU" -i "$REP" --page details --csv > "$OUT/ncu_${i}_details.csv" 2>/dev/null
  "$NCU" -i "$REP" --page details > "$OUT/ncu_${i}_details.txt" 2>/dev/null
  "$NCU" -i "$REP" --page source --csv --print-source sass > "$OUT/ncu_${i}_source.csv" 2>/dev/null
done
mv "$OUT/SUMMARY.txt.tmp" "$OUT/SUMMARY.txt"; cat "$OUT/SUMMARY.txt"

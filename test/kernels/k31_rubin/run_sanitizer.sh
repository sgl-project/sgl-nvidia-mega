#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# compute-sanitizer (TOOL, default memcheck) on pytest node ids, one GPU. Usage: run_sanitizer.sh OUTDIR TOOL NODEID...
set -u
OUT=$1; TOOL=$2; shift 2
mkdir -p "$OUT"; chmod 700 "$OUT"
HERE=$(cd "$(dirname "$0")" && pwd)
PLUGIN=$(cd "$HERE/../../.." && pwd)
CACHE=/tmp/mkport_san_${SLURM_JOB_ID:-x}
mkdir -p "$CACHE/home"
export HOME=$CACHE/home TRITON_CACHE_DIR=$CACHE/triton CUDA_CACHE_PATH=$CACHE/cuda
export PYTHONPATH=$PLUGIN${PYTHONPATH:+:$PYTHONPATH} PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 K31_TIME=0
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*}
CS=$(command -v compute-sanitizer || ls /usr/local/cuda/bin/compute-sanitizer 2>/dev/null)
"$CS" --version > "$OUT/version.txt" 2>&1
i=0
for node in "$@"; do
  i=$((i + 1))
  ( cd "$PLUGIN" && timeout 1500 "$CS" --tool "$TOOL" --show-backtrace device --print-limit 20 \
      python3 -m pytest -q -p no:cacheprovider "$node" ) > "$OUT/san_${TOOL}_$i.log" 2>&1
  echo "node $node tool $TOOL rc=$? errors: $(grep -c '========= Invalid\|========= Error\|========= Program hit' $OUT/san_${TOOL}_$i.log)" >> "$OUT/SUMMARY.txt.tmp"
done
mv "$OUT/SUMMARY.txt.tmp" "$OUT/SUMMARY.txt"; cat "$OUT/SUMMARY.txt"

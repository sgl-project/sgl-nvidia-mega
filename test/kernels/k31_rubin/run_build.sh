#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Whole-plugin build smoke (k31_build_smoke.py) on this node's GPUs; writes OUTDIR/SUMMARY.txt last.
# Usage: run_build.sh OUTDIR HOST_TREE [smoke args...]   HOST_TREE = MK host tree dir (has python/sglang, API v1)
set -u
OUT=$1; TREE=$2; shift 2
mkdir -p "$OUT"; chmod 700 "$OUT"
HERE=$(cd "$(dirname "$0")" && pwd)
PLUGIN=$(cd "$HERE/../../.." && pwd)
cat "$PLUGIN/COMMIT" > "$OUT/COMMIT" 2>/dev/null; cat "$TREE/.k31_tree" > "$OUT/HOST_TREE" 2>/dev/null
nvidia-smi --query-gpu=index,name,clocks.sm,clocks.mem --format=csv > "$OUT/clocks.csv" 2>&1
ARCH=$(python3 -c "import torch; m,n=torch.cuda.get_device_capability(); print(f'sm_{m}{n}a')")
CACHE=/tmp/mkport_build_${SLURM_JOB_ID:-x}_${ARCH}
mkdir -p "$CACHE"
export HOME=$CACHE/home XDG_CACHE_HOME=$CACHE/xdg TRITON_CACHE_DIR=$CACHE/triton \
  CUDA_CACHE_PATH=$CACHE/cuda SGLANG_JIT_CACHE_DIR=$CACHE/sgljit TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor
mkdir -p "$HOME"
export PYTHONPATH=$PLUGIN:$TREE/python${PYTHONPATH:+:$PYTHONPATH}
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
IFS=, read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
N=${#GPUS[@]}
t=$(date +%s)
( cd "$PLUGIN" && timeout "${K31_TIMEOUT:-3000}" python3 -m torch.distributed.run --nnodes 1 --nproc-per-node "$N" \
    "$HERE/k31_build_smoke.py" --out "$OUT" "$@" ) > "$OUT/build.log" 2>&1
r=$?
echo "build rc=$r ($(( $(date +%s) - t ))s) $(grep -h BUILD_RESULT $OUT/build.log | tail -1 | cut -c1-1500)" > "$OUT/SUMMARY.txt.tmp"
grep -h -m3 "FAILED" "$OUT/build.log" | cut -c1-400 >> "$OUT/SUMMARY.txt.tmp"
mv "$OUT/SUMMARY.txt.tmp" "$OUT/SUMMARY.txt"
cat "$OUT/SUMMARY.txt"

#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Multi-GPU o_proj test (k31_oproj_tp.py) on this node's GPUs: base run (K31_TIME=1 timing),
# then each mutation (must FAIL). Writes OUTDIR/SUMMARY.txt last.
# Usage: run_tp.sh OUTDIR BATCHES MUT_BATCHES MUTATIONS   (comma-separated lists; MUTATIONS may be "-")
set -u
OUT=$1; BATCHES=$2; MUT_BATCHES=$3; MUTS=${4:--}
mkdir -p "$OUT"; chmod 700 "$OUT"
HERE=$(cd "$(dirname "$0")" && pwd)
PLUGIN=$(cd "$HERE/../../.." && pwd)
cat "$PLUGIN/COMMIT" > "$OUT/COMMIT" 2>/dev/null
nvidia-smi --query-gpu=index,name,clocks.sm,clocks.mem --format=csv > "$OUT/clocks.csv" 2>&1
ARCH=$(python3 -c "import torch; m,n=torch.cuda.get_device_capability(); print(f'sm_{m}{n}a')")
CACHE=/tmp/mkport_comm_${SLURM_JOB_ID:-x}_${ARCH}
mkdir -p "$CACHE"
export HOME=$CACHE/home XDG_CACHE_HOME=$CACHE/xdg TRITON_CACHE_DIR=$CACHE/triton \
  CUDA_CACHE_PATH=$CACHE/cuda SGLANG_JIT_CACHE_DIR=$CACHE/sgljit TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor
mkdir -p "$HOME"
export PYTHONPATH=$PLUGIN${PYTHONPATH:+:$PYTHONPATH}
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
IFS=, read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
N=${K31_NPROC:-${#GPUS[@]}}
summary=$OUT/SUMMARY.txt.tmp; : > "$summary"
start=$(date +%s)
run () {  # tag batches timing
  ( cd "$PLUGIN" && K31_TIME=$3 timeout "${K31_TIMEOUT:-1200}" python3 -m torch.distributed.run --nnodes 1 \
      --nproc-per-node "$N" "$HERE/k31_oproj_tp.py" --out "$OUT" --batches "$2" ) > "$OUT/tp${N}_$1.log" 2>&1
}
if [ "$BATCHES" != "-" ]; then
  run base "$BATCHES" "${K31_TIME:-1}"; r=$?
  echo "tp$N base rc=$r $(grep -h TP_RESULT $OUT/tp${N}_base.log | tail -1) ($(( $(date +%s) - start ))s)" >> "$summary"
fi
if [ "$MUTS" != "-" ]; then
  IFS=, read -ra ML <<< "$MUTS"
  for m in "${ML[@]}"; do
    t=$(date +%s)
    K31_MUTATION=$m run "$m" "$MUT_BATCHES" 0; r=$?
    if [ $r -ne 0 ]; then v=CAUGHT; else v=MISSED; fi
    echo "tp$N mutation $m $v (rc=$r, $(( $(date +%s) - t ))s) $(grep -h -m1 -E 'FAIL |CASE_ERROR' $OUT/tp${N}_$m.log | cut -c1-200)" >> "$summary"
  done
fi
if [ -n "${K31_STAGING:-}" ]; then
  t=$(date +%s)
  ( cd "$PLUGIN" && timeout "${K31_TIMEOUT:-1200}" python3 -m torch.distributed.run --nnodes 1 \
      --nproc-per-node "$N" "$HERE/k31_oproj_tp.py" --out "$OUT" --staging "$K31_STAGING" ) > "$OUT/tp${N}_staging.log" 2>&1
  echo "tp$N staging rc=$? $(grep -h STAGING_RESULT $OUT/tp${N}_staging.log) ($(( $(date +%s) - t ))s)" >> "$summary"
fi
if [ -n "${K31_DIAG:-}" ]; then
  t=$(date +%s)
  ( cd "$PLUGIN" && timeout "${K31_TIMEOUT:-1200}" python3 -m torch.distributed.run --nnodes 1 \
      --nproc-per-node "$N" "$HERE/k31_oproj_tp.py" --out "$OUT" --diag "$K31_DIAG" ) > "$OUT/tp${N}_diag.log" 2>&1
  echo "tp$N diag rc=$? ($(( $(date +%s) - t ))s)" >> "$summary"
fi
mv "$summary" "$OUT/SUMMARY.txt"
cat "$OUT/SUMMARY.txt"

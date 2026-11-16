#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Run one pytest target sharded over the visible GPUs (one process per GPU), inside the
# cu134 image on a GPU node. Usage: run_shards.sh OUTDIR PYTEST_ARGS...
#   K31_MUTATION=<name> run_shards.sh ...   runs the mutated copy (expected to FAIL).
# Caches stay node-local and arch-separated; CuTe artifacts (ptx/cubin/sass) go to OUTDIR.
set -u
OUT=$1; shift
mkdir -p "$OUT"; chmod 700 "$OUT"
HERE=$(cd "$(dirname "$0")" && pwd)
PLUGIN=$(cd "$HERE/../../.." && pwd)
ARCH=$(python3 -c "import torch; m,n=torch.cuda.get_device_capability(); print(f'sm_{m}{n}a')")
CACHE=/tmp/mkport_${SLURM_JOB_ID:-x}_${ARCH}
mkdir -p "$CACHE"
export HOME=$CACHE/home XDG_CACHE_HOME=$CACHE/xdg TRITON_CACHE_DIR=$CACHE/triton \
  CUDA_CACHE_PATH=$CACHE/cuda SGLANG_JIT_CACHE_DIR=$CACHE/sgljit TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor
mkdir -p "$HOME"
export PYTHONPATH=$PLUGIN${K31_HOST_PYTHON:+:$K31_HOST_PYTHON}${PYTHONPATH:+:$PYTHONPATH}
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
IFS=, read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
N=${#GPUS[@]}
TAG=${K31_TAG:-${K31_MUTATION:-base}}
pids=()
for i in "${!GPUS[@]}"; do
  ( export CUDA_VISIBLE_DEVICES=${GPUS[$i]} K31_SHARD=$i/$N K31_RESULTS=$OUT/results_${TAG}_$i.jsonl
    export CUTE_DSL_KEEP=${CUTE_DSL_KEEP:-ptx,cubin,sass} CUTE_DSL_DUMP_DIR=$OUT/cute_${TAG}_$i
    mkdir -p "$CUTE_DSL_DUMP_DIR"
    cd "$PLUGIN" && timeout "${K31_TIMEOUT:-3000}" python3 -m pytest -q -p no:cacheprovider -rA "$@" \
      > "$OUT/pytest_${TAG}_$i.log" 2>&1 ) &
  pids+=($!)
done
rc=0
for i in "${!pids[@]}"; do wait "${pids[$i]}"; r=$?; echo "shard $i rc=$r"; tail -3 "$OUT/pytest_${TAG}_$i.log"; [ $r -ne 0 ] && [ $r -ne 5 ] && rc=1; done
echo "RUN_SHARDS_DONE tag=$TAG rc=$rc"
exit $rc

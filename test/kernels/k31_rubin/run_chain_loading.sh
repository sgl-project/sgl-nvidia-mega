#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# F3 chain check: the multi-GPU MK chain tests (o_proj mailboxes, B1 MoE small-batch chain, B5-B8 wave tail; real
# symm_mem multicast, eager then CUDA-graph replay) under three module-loading modes, with every checked output's
# SHA-256 recorded (K31_DIGEST=1), then compared bitwise against the EAGER run:
#   eager         CUDA_MODULE_LOADING=EAGER (what MK arms serve with today)
#   lazy_preload  CUDA_MODULE_LOADING=LAZY K31MK_PRELOAD=1 (the F3 proposal; the tests call the prepare hook's preload
#                 after building their kernels)
#   lazy          CUDA_MODULE_LOADING=LAZY (control: no preload)
# Writes OUTDIR/SUMMARY.txt last. Usage: run_chain_loading.sh OUTDIR [OPROJ_BATCHES] [WAVE_BATCHES]
set -u
OUT=$1; OB=${2:-1,8,64}; WB=${3:-5,8}
mkdir -p "$OUT"; chmod 700 "$OUT"
HERE=$(cd "$(dirname "$0")" && pwd)
PLUGIN=$(cd "$HERE/../../.." && pwd)
ARCH=$(env -u CUDA_MODULE_LOADING python3 -c "import torch; m,n=torch.cuda.get_device_capability(); print(f'sm_{m}{n}a')")
CACHE=/tmp/mkport_${SLURM_JOB_ID:-x}_f3chain_${ARCH}
mkdir -p "$CACHE/home"
export HOME=$CACHE/home XDG_CACHE_HOME=$CACHE/xdg TRITON_CACHE_DIR=$CACHE/triton CUDA_CACHE_PATH=$CACHE/cuda \
  SGLANG_JIT_CACHE_DIR=$CACHE/sgljit TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor
export PYTHONPATH=$PLUGIN${PYTHONPATH:+:$PYTHONPATH} PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 K31_DIGEST=1 K31_TIME=0
unset CUDA_MODULE_LOADING K31MK_PRELOAD K31_MUTATION
IFS=, read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
N=${#GPUS[@]}
summary=$OUT/SUMMARY.txt.tmp; : > "$summary"
for mode in eager lazy_preload lazy; do
  case $mode in
    eager) envs="CUDA_MODULE_LOADING=EAGER" ;;
    lazy_preload) envs="CUDA_MODULE_LOADING=LAZY K31MK_PRELOAD=1" ;;
    lazy) envs="CUDA_MODULE_LOADING=LAZY" ;;
  esac
  for test in oproj moe wave; do
    case $test in
      oproj) cmd=("$HERE/k31_oproj_tp.py" --out "$OUT/$mode" --batches "$OB") ;;
      moe) cmd=("$HERE/k31_moe_tp.py" --out "$OUT/$mode") ;;
      wave) cmd=("$HERE/k31_wavetail_tp.py" --out "$OUT/$mode" --batches "$WB") ;;
    esac
    s=$(date +%s)
    ( cd "$PLUGIN" && env $envs timeout "${K31_TIMEOUT:-1200}" python3 -m torch.distributed.run --nnodes 1 \
        --nproc-per-node "$N" "${cmd[@]}" ) > "$OUT/${mode}_${test}.log" 2>&1
    r=$?
    echo "$mode $test rc=$r $(grep -h TP_RESULT "$OUT/${mode}_${test}.log" | tail -1) ($(( $(date +%s) - s ))s)" >> "$summary"
  done
done
python3 "$HERE/compare_digests.py" "$OUT/eager" "$OUT/lazy_preload" "$OUT/lazy" >> "$summary" 2>&1
mv "$summary" "$OUT/SUMMARY.txt"
cat "$OUT/SUMMARY.txt"

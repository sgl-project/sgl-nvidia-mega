#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# A4 (TP16 FP32 latent) validation on one 4-GPU node (GB300 or Rubin):
#   1. test_k31_experts_tp16.py sharded over the GPUs, base then each mutation on the tests that cover it;
#   2. k31_moe_tp16.py (4-GPU TP16 chain + generic front), base then its mutations.
# Every mutation must FAIL. Writes OUTDIR/SUMMARY.txt. Usage: run_a4.sh OUTDIR [experts|chain|all] [MUTATIONS=1|0]
set -u
OUT=$1; WHAT=${2:-all}; MUTS=${3:-1}
mkdir -p "$OUT"; chmod 700 "$OUT"
HERE=$(cd "$(dirname "$0")" && pwd)
PLUGIN=$(cd "$HERE/../../.." && pwd)
T=test/kernels/k31_rubin
export K31_TIME=0 K31_TIMEOUT=${K31_TIMEOUT:-900}
summary=$OUT/SUMMARY.txt
: > "$summary"
verdict () {  # name rc logs...
  local name=$1 r=$2; shift 2
  local ran
  ran=$(cat "$@" 2>/dev/null | grep -hoE "[0-9]+ (passed|failed|error)" | awk '{s+=$1} END {print s+0}')
  if [ "$ran" -eq 0 ]; then echo "mutation $name NOT_RUN (rc=$r)"
  elif [ "$r" -ne 0 ]; then echo "mutation $name CAUGHT (rc=$r, $ran tests)"
  else echo "mutation $name MISSED (rc=0, $ran tests)"; fi
}
if [ "$WHAT" != chain ]; then
  t=$(date +%s)
  bash "$HERE/run_shards.sh" "$OUT/experts" "$T/test_k31_experts_tp16.py"
  echo "experts base rc=$? ($(( $(date +%s) - t ))s)" >> "$summary"
  [ "$MUTS" = 1 ] && for spec in "c16_bf16_x:cluster16 or cluster_b1" "c16_xscale_drop:cluster16" "c16_block_off:cluster16" \
              "pe_bf16_x:persistent16 and not b16" "plan_bf16:plan16 or (persistent16 and b16)"; do
    m=${spec%%:*}; k=${spec#*:}; t=$(date +%s)
    K31_MUTATION=$m bash "$HERE/run_shards.sh" "$OUT/experts" "$T/test_k31_experts_tp16.py" -k "$k"
    r=$?
    echo "$(verdict "$m" $r "$OUT"/experts/pytest_${m}_*.log) ($(( $(date +%s) - t ))s)" >> "$summary"
  done
fi
if [ "$WHAT" != experts ]; then
  ARCH=$(python3 -c "import torch; m,n=torch.cuda.get_device_capability(); print(f'sm_{m}{n}a')")
  CACHE=/tmp/mkport_comm_${SLURM_JOB_ID:-x}_${ARCH}
  mkdir -p "$CACHE"
  export HOME=$CACHE/home XDG_CACHE_HOME=$CACHE/xdg TRITON_CACHE_DIR=$CACHE/triton \
    CUDA_CACHE_PATH=$CACHE/cuda SGLANG_JIT_CACHE_DIR=$CACHE/sgljit TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor
  mkdir -p "$HOME"
  export PYTHONPATH=$PLUGIN${PYTHONPATH:+:$PYTHONPATH} PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
  mkdir -p "$OUT/chain"
  chain () {  # tag arms
    ( cd "$PLUGIN" && timeout "$K31_TIMEOUT" python3 -m torch.distributed.run --nnodes 1 --nproc-per-node 4 \
        "$HERE/k31_moe_tp16.py" --out "$OUT/chain" --arms "$2" ) > "$OUT/chain/tp16_$1.log" 2>&1
  }
  t=$(date +%s)
  chain base small,generic
  echo "chain base rc=$? $(grep -h TP_RESULT "$OUT/chain/tp16_base.log" | tail -1) ($(( $(date +%s) - t ))s)" >> "$summary"
  [ "$MUTS" = 1 ] && for spec in "f16_bf16_latent:small" "f16_bf16_hg:small" "gen_bf16_latent:generic"; do
    m=${spec%%:*}; arms=${spec#*:}; t=$(date +%s)
    K31_MUTATION=$m chain "$m" "$arms"; r=$?
    if [ $r -ne 0 ]; then v=CAUGHT; else v=MISSED; fi
    first=$(grep -h -m1 -E "^FAIL " "$OUT/chain/tp16_$m.log" | cut -c1-160)
    echo "mutation $m $v (rc=$r, $(( $(date +%s) - t ))s) $first" >> "$summary"
  done
fi
cat "$summary"

#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# One arm's DeepSeek-V4.1-Flash server for ab_step_times.py, inside the SGLang container,
# one task per node. Forced acceptance length 3.51, batch 1, DSpark block 5.
#
#   AB_TOPO      tp4 (one tray) | tp8 (attention TP8 + MoE TP4 x EP2 on two trays)
#   AB_SRC       the plugin checkout (put first on PYTHONPATH)
#   AB_MODEL     model directory
#   AB_SLOTS     full SGLANG_NVIDIA_MEGAKERNEL_SLOTS (default: the shipped slots, plus
#                swapab_h8,wo_a_g1 at tp8), AB_SLOTS_ADD appended, AB_SLOTS_DEL removed
#   AB_PORT      default 30000; HEAD_ADDR / DIST_PORT for tp8
#   AB_RUNTIME_OPT  1 (default) sets SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION=1, 0 leaves
#                it unset (SGLang's stock host path)
#   AB_ENVFILE   a file of "export K=V" lines sourced last (the arm's own switches)
set -u
NODE_RANK=${SLURM_NODEID:-0}
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$AB_SRC${PYTHONPATH:+:$PYTHONPATH}
export SGLANG_SPEC_TP_SYNC=rng,-dspark-accept-sample
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export SGLANG_SIMULATE_ACC_LEN=3.51 SGLANG_SIMULATE_ACC_METHOD=match-expected
export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
# SGLang's runtime optimizations for the megakernel (on unless AB_RUNTIME_OPT=0)
[ "${AB_RUNTIME_OPT:-1}" = 1 ] && export SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION=1
# per-node compile caches (a shared Triton cache races across nodes)
export TRITON_CACHE_DIR=/tmp/ab-triton-$(hostname)
# shellcheck disable=SC1090
[ -n "${AB_ENVFILE:-}" ] && . "$AB_ENVFILE"

SHIPPED=$(python3 -c "from sgl_nvidia_mega.deepseek_v41.runtime import SHIPPED_SLOTS as s; print(','.join(s))")
DEFAULT=$SHIPPED
[ "$AB_TOPO" = tp8 ] && DEFAULT=$SHIPPED,swapab_h8,wo_a_g1
SLOTS=${AB_SLOTS:-$DEFAULT}
[ -n "${AB_SLOTS_ADD:-}" ] && SLOTS=$SLOTS,$AB_SLOTS_ADD
if [ -n "${AB_SLOTS_DEL:-}" ]; then
  SLOTS=$(echo "$SLOTS" | tr , '\n' | grep -vxF -f <(echo "$AB_SLOTS_DEL" | tr , '\n') | paste -sd,)
fi
export SGLANG_NVIDIA_MEGAKERNEL_SLOTS=$SLOTS
echo "AB START $(date -Is) host=$(hostname) node_rank=$NODE_RANK topo=$AB_TOPO"
echo "AB SLOTS=$SLOTS"
echo "AB SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION=${SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION:-<unset>}"
sed 's/^/AB ENV /' "${AB_ENVFILE:-/dev/null}"

COMMON=(--model-path "$AB_MODEL" --served-model-name deepseek-ai/DeepSeek-V4.1-Flash
  --trust-remote-code --enable-nvidia-megakernel
  --moe-a2a-backend none --moe-runner-backend flashinfer_mxfp4
  --dsv4-attn-backend flashmla --kv-cache-dtype fp8_e4m3
  --speculative-algorithm DSPARK --speculative-dspark-block-size 5
  --cuda-graph-max-bs-decode 1 --max-running-requests 1
  --mem-fraction-static 0.8 --chunked-prefill-size 8192
  --enable-decoder-swa-bounded-replay --startup-weight-load-mode serial
  --watchdog-timeout 3600 --host 0.0.0.0 --port "${AB_PORT:-30000}")
case $AB_TOPO in
tp8)
  export NCCL_CUMEM_ENABLE=1 NCCL_MNNVL_ENABLE=1
  python3 -m sglang.launch_server "${COMMON[@]}" --tp 8 --ep-size 2 \
    --nnodes 2 --node-rank "$NODE_RANK" --dist-init-addr "${HEAD_ADDR}:${DIST_PORT:-29511}" ;;
tp4)
  python3 -m sglang.launch_server "${COMMON[@]}" --tp 4 --ep-size 1 ;;
*) echo "AB_TOPO must be tp4 or tp8"; exit 2 ;;
esac
echo "AB EXIT rc=$? $(date -Is)"

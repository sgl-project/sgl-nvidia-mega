#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# ab_step_times.py's client for one launch, inside the container on the head node:
#   ab_client.sh <url> <out dir> <passes> [isl] [osl]
# Waits for the server, runs one warm-up pass (FlashInfer's autotuner runs on the first
# request), then <passes> static_table.py passes at concurrency 1; each pass is
# 2 + 5 requests. Writes <out>/passes.jsonl (pass, salt, start / end unix time).
set -u
URL=$1 OUT=$2 PASSES=$3 ISL=${4:-8192} OSL=${5:-1024}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT/static"
t0=$(date +%s)
until curl -sf -m 60 "$URL/health_generate" > /dev/null; do
  [ $(( $(date +%s) - t0 )) -gt "${AB_READY_TIMEOUT:-2700}" ] && { echo "AB server not ready"; exit 1; }
  sleep 10
done
echo "AB ready after $(( $(date +%s) - t0 )) s"
for p in $(seq 0 "$PASSES"); do
  salt=w$p; [ "$p" -gt 0 ] && salt=r$p
  s=$(date +%s.%N)
  python3 "$HERE/static_table.py" --base-url "$URL" --isl "$ISL" --osl "$OSL" --conc 1 \
    --salt "$salt" --label "pass$p" --out "$OUT/static" || exit 1
  e=$(date +%s.%N)
  echo "{\"pass\": $p, \"salt\": \"$salt\", \"start\": $s, \"end\": $e}" >> "$OUT/passes.jsonl"
done
sleep 3  # let the stamp drain (0.2 s poll) catch up
echo "AB CLIENT DONE"

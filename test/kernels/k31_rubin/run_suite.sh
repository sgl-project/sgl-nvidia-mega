#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# One queued task: base run (with timing) of a test file over 4 GPUs, then each named
# mutation (each must FAIL). Usage: run_suite.sh OUTDIR TESTFILE [-k EXPR] [MUTATION...]
#   K31_MUT_K: pytest -k expression for the mutation runs (default: all cases).
set -u
OUT=$1; TEST=$2; shift 2
K=()
if [ "${1:-}" = "-k" ]; then K=(-k "$2"); shift 2; fi
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
cat "$HERE/../../../COMMIT" 2>/dev/null > "$OUT/COMMIT"
nvidia-smi --query-gpu=index,name,clocks.sm,clocks.mem,clocks.max.sm --format=csv > "$OUT/clocks_start.csv" 2>&1
K31_TIME=${K31_TIME:-1} bash "$HERE/run_shards.sh" "$OUT" "$TEST" "${K[@]}"
base=$?
summary="$OUT/SUMMARY.txt"
echo "base rc=$base" > "$summary"
for m in "$@"; do
  MK=(); [ -n "${K31_MUT_K:-}" ] && MK=(-k "$K31_MUT_K")
  K31_MUTATION=$m K31_TIME=0 bash "$HERE/run_shards.sh" "$OUT" "$TEST" "${MK[@]}"
  r=$?
  ran=$(cat "$OUT"/pytest_${m}_*.log 2>/dev/null | grep -hoE "[0-9]+ (passed|failed|error)" | awk '{s+=$1} END {print s+0}')
  # An anchor that is missing or found twice aborts the mutated import (_harness.load_plugin): the tests then fail
  # without the mutation ever running, which must not count as CAUGHT.
  if grep -qs "mutation $m: .* occurs [0-9]*x in" "$OUT"/pytest_${m}_*.log; then
    echo "mutation $m INVALID_ANCHOR (its anchor is not found exactly once; rc=$r)" >> "$summary"
  elif [ "$ran" -eq 0 ]; then echo "mutation $m NOT_RUN (no test collected; rc=$r)" >> "$summary"
  elif [ $r -ne 0 ]; then echo "mutation $m CAUGHT (rc=$r, $ran tests)" >> "$summary"
  else echo "mutation $m MISSED (rc=0, $ran tests)" >> "$summary"; fi
done
cat "$summary"

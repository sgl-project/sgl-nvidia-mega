#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# F3 reproducer (k31_lazy_preload.py): every arm TRIALS times, one process per trial (a kernel loads once per
# process), four at a time (one per GPU), each under a watchdog. Writes OUTDIR/SUMMARY.txt last: per arm the outcome
# counts against the expectation written before any run (see the docstring of k31_lazy_preload.py).
# Usage: run_lazy_preload.sh OUTDIR [TRIALS]
set -u
OUT=$1; TRIALS=${2:-3}
mkdir -p "$OUT"; chmod 700 "$OUT"
HERE=$(cd "$(dirname "$0")" && pwd)
PLUGIN=$(cd "$HERE/../../.." && pwd)
ARCH=$(env -u CUDA_MODULE_LOADING python3 -c "import torch; m,n=torch.cuda.get_device_capability(); print(f'sm_{m}{n}a')")
CACHE=/tmp/mkport_${SLURM_JOB_ID:-x}_f3lazy_${ARCH}
mkdir -p "$CACHE/home"
export HOME=$CACHE/home XDG_CACHE_HOME=$CACHE/xdg TRITON_CACHE_DIR=$CACHE/triton CUDA_CACHE_PATH=$CACHE/cuda \
  SGLANG_JIT_CACHE_DIR=$CACHE/sgljit TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor
export PYTHONPATH=$PLUGIN${PYTHONPATH:+:$PYTHONPATH} PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
unset CUDA_MODULE_LOADING K31MK_PRELOAD K31_MUTATION
WD=${K31_WATCHDOG_S:-240}   # EAGER processes load every module first (~1-2 min); a hang traps at 3 s
# name | arm | env | expected (HANG/PASS gate the reproducer; INFO is informative)
SPECS=(
  "lazy|lazy|CUDA_MODULE_LOADING=LAZY|HANG"
  "lazy_preload|lazy_preload|CUDA_MODULE_LOADING=LAZY K31MK_PRELOAD=1|PASS"
  "lazy_preload@preload_empty|lazy_preload|CUDA_MODULE_LOADING=LAZY K31MK_PRELOAD=1 K31_MUTATION=preload_empty|HANG"
  "lazy_warm|lazy_warm|CUDA_MODULE_LOADING=LAZY|PASS"
  "eager|eager|CUDA_MODULE_LOADING=EAGER|INFO"
  "lazy_torch|lazy_torch|CUDA_MODULE_LOADING=LAZY|HANG"
  "eager_torch|eager_torch|CUDA_MODULE_LOADING=EAGER|PASS"
  "lazy_torch_preload|lazy_torch_preload|CUDA_MODULE_LOADING=LAZY K31MK_PRELOAD=1|HANG"
)
IFS=, read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
RES=$OUT/outcomes.txt; : > "$RES"
one () {  # gpu name arm env trial
  local gpu=$1 name=$2 arm=$3 envs=$4 t=$5 log rc s outcome
  log=$OUT/log_${name//@/_}_t$t.log; s=$(date +%s)
  CUDA_VISIBLE_DEVICES=$gpu env $envs timeout -s KILL "$WD" python3 "$HERE/k31_lazy_preload.py" "$arm" --out "$OUT" \
    --trial "$t" > "$log" 2>&1
  rc=$?
  outcome=$(grep -h '^RESULT ' "$log" | tail -1 | python3 -c "import json,sys; l=sys.stdin.read().strip(); print(json.loads(l[7:])['outcome'] if l else 'NO_RESULT')")
  [ "$rc" = 137 ] && outcome=KILLED
  echo "$name $t $outcome rc=$rc $(( $(date +%s) - s ))s" >> "$RES"
}
jobs_list=()
for spec in "${SPECS[@]}"; do for t in $(seq 1 "$TRIALS"); do jobs_list+=("$spec|$t"); done; done
i=0
while [ $i -lt ${#jobs_list[@]} ]; do
  for gpu in "${GPUS[@]}"; do
    [ $i -lt ${#jobs_list[@]} ] || break
    IFS='|' read -r name arm envs expect t <<< "${jobs_list[$i]}"
    one "$gpu" "$name" "$arm" "$envs" "$t" &
    i=$((i + 1))
  done
  wait
done
python3 - "$RES" "$OUT/SUMMARY.txt" "${SPECS[@]}" <<'EOF'
import sys, collections
res, out, specs = sys.argv[1], sys.argv[2], sys.argv[3:]
got = collections.defaultdict(list)
for line in open(res):
    name, trial, outcome = line.split()[:3]
    got[name].append("HANG" if outcome == "KILLED" else outcome)
lines, verdicts = [], {}
for spec in specs:
    name, _, _, expect = spec.split("|")
    o = got.get(name, [])
    verdicts[name] = "INFO" if expect == "INFO" else ("OK" if o and all(x == expect for x in o) else "MISMATCH")
    lines.append(f"{name}: expect {expect}, got {dict(collections.Counter(o))} -> {verdicts[name]}")
gate = ("lazy", "lazy_preload", "lazy_preload@preload_empty", "lazy_warm")
calib = ("lazy_torch", "eager_torch")
ok = all(verdicts[n] == "OK" for n in gate)
lines.append(f"REPRODUCER {'PASS' if ok else 'FAIL'} (lazy HANG, lazy_preload PASS, preload_empty HANG, lazy_warm PASS)")
lines.append(f"CALIBRATION {'PASS' if all(verdicts[n] == 'OK' for n in calib) else 'FAIL'} (Rubin ringdiag: lazy_torch HANG, eager_torch PASS)")
open(out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
EOF

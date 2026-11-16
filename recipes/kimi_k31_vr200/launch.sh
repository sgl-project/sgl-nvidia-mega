#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Launch one server role of a published Kimi K3.1 VR200 point on this node:
#   launch.sh <c1|c8|c14|c24|c48> <node|prefill|decode>
# with TARGET_WEIGHTS, DRAFTER_WEIGHTS, NODE_RANK and DIST_INIT_ADDR set. Sets the point's environment words
# (<point>/env.<role>) and runs the server with its arguments (<point>/args.<role>): SGLang's server for c1, c8
# and c24, a Dynamo SGLang worker for c14 and c48.
set -euo pipefail
point=${1:?point}; role=${2:?role}
dir=$(cd "$(dirname "$0")/$point" && pwd)
: "${TARGET_WEIGHTS:?}" "${DRAFTER_WEIGHTS:?}" "${NODE_RANK:?}" "${DIST_INIT_ADDR:?}"
set -a
# shellcheck disable=SC1090
. <(grep -v '^#' "$dir/env.$role")
set +a
args=$(grep -v '^#' "$dir/args.$role" | sed -e "s|<TARGET_WEIGHTS>|$TARGET_WEIGHTS|" -e "s|<DRAFTER_WEIGHTS>|$DRAFTER_WEIGHTS|" \
  -e "s|<NODE_RANK>|$NODE_RANK|" -e "s|<DIST_INIT_ADDR>|$DIST_INIT_ADDR|" | tr '\n' ' ')
case $point in c14 | c48) entry=dynamo.sglang ;; *) entry=sglang.launch_server ;; esac
eval exec python3 -m "$entry" "$args"

#!/usr/bin/env bash
# Copyright 2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0
#   URL=http://127.0.0.1:8100 GPU_IDS=0 perf/aiperf/bench.sh runs/perf/<run>
#   URL=http://127.0.0.1:8100 USE_CASES="cls-2k cls-2k-896" CONCURRENCY="1 8" perf/aiperf/bench.sh <out dir>
#   URL="http://127.0.0.1:8100 http://127.0.0.1:8101" GPU_IDS=0,1 perf/aiperf/bench.sh <out dir>   # round robin
set -euo pipefail
cd "$(dirname "$0")/../.."   # the repo root
out=$1 url=${URL:-http://127.0.0.1:8100}
mkdir -p "$out"
urls=(); for u in $url; do urls+=(-u "$u"); done

# The model name, and the checkpoint it loaded, whose tokenizer AIPerf loads.
model=$(curl -fs "${url%% *}/v1/models" | jq -r '.data[0].id')
tokenizer=$(curl -fs "${url%% *}/v1/models" | jq -r '.data[0].root')

aiperf() {
  docker run --rm --network host --gpus "\"device=${GPU_IDS:-0}\"" -v "$PWD:$PWD" -w "$PWD" \
    -e HF_HOME="$PWD/.cache/hf" perf-aiperf:0.13.0 "$@"
}
max() { (( $1 > $2 )) && echo "$1" || echo "$2"; }

# A use case replays runs/perf/payloads/<use case>.jsonl (payloads.py) byte for byte
for case in ${USE_CASES:-cls-2k}; do
  payloads=runs/perf/payloads/$case.jsonl
  [[ -f $payloads ]] || { echo "No $payloads: run utils/payload/payload.py" >&2; exit 1; }
  endpoint=/classify; [[ $case == *-pooling ]] && endpoint=/pooling

  for c in ${CONCURRENCY:-1 2 4 8 16 32 64 128}; do
    requests=${REQUESTS:-$(max 100 $((8 * c)))}   # at least 100 requests per point, for a stable p95
    log=$out/${case}_CON$c.log
    echo "$case, concurrency $c: $requests requests (log: $log)"
    # shellcheck disable=SC2086
    aiperf profile -m "$model" --tokenizer "$tokenizer" "${urls[@]}" --endpoint-type raw --custom-endpoint "$endpoint" \
      --extra-inputs "response_field:data[*].index" --input-file "$payloads" \
      --custom-dataset-type raw-payload --dataset-sampling-strategy sequential --concurrency "$c" \
      --request-count "$requests" --warmup-request-count "$(max 4 "$c")" --num-profile-runs "${RUNS:-1}" \
      --random-seed 42 --gpu-telemetry pynvml --ui none --artifact-dir "$out/$case/CON$c" ${AIPERF_ARGS:-} \
      > "$log" 2>&1 || echo "  AIPerf failed, see $log"
    sleep 5
  done
done

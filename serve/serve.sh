#!/usr/bin/env bash
# Copyright 2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0
#   GPU_IDS=0 serve/serve.sh
#   GPU_IDS=1 PORT=8001 MODEL=<checkpoint dir> serve/serve.sh --max-num-seqs 64           
#   docker logs -f gemma3-cls                                follow the server's log
#   docker rm -f gemma3-cls                                  stop the server
set -euo pipefail
cd "$(dirname "$0")/.."   # the repo root

model=$(realpath "${MODEL:-.cache/models/gemma3-4b-cls7000}")
name=${NAME:-gemma3-cls} gpus=${GPU_IDS:-0} port=${PORT:-8000} image=${IMAGE:-ghcr.io/dawnik17/gemma3-cls-vllm:0.32.0}

docker run -d --name "$name" --gpus "\"device=$gpus\"" --ipc=host -p "127.0.0.1:$port:8000" \
  -v "$model:$model:ro" -e HF_HUB_OFFLINE=1 --entrypoint vllm "$image" \
  serve "$model" --host 0.0.0.0 --port 8000 --served-model-name cls \
  --max-model-len 4096 --limit-mm-per-prompt '{"image":1,"video":0}' "$@" > /dev/null

echo "Starting $model in container $name on GPU $gpus; loading and compiling take 1-5 minutes"
until curl -fs "http://127.0.0.1:$port/v1/models" > /dev/null; do
  if [[ $(docker inspect -f '{{.State.Running}}' "$name") != true ]]; then
    docker logs --tail 30 "$name"; echo "The server stopped during startup; its log is above" >&2; exit 1
  fi
  sleep 5
done
echo "READY http://127.0.0.1:$port"

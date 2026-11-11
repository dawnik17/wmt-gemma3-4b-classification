<!--
Copyright 2026 NVIDIA Corporation
SPDX-License-Identifier: Apache-2.0
-->
# wmt-gemma3-4b-classification

Serve Gemma 3 4B image + text classifier with vLLM 0.32.0.

## One Time Setup

Pull the image:

```bash
docker pull ghcr.io/dawnik17/gemma3-cls-vllm:0.32.0-fa
```

Download the model checkpoint:

```bash
hf download dawn17/gemma3-4b-cls7000 --local-dir .cache/models/gemma3-4b-cls7000
```

Or build the image yourself

```bash
docker build -t ghcr.io/dawnik17/gemma3-cls-vllm:0.32.0-fa serve
```

## vLLM Inference

```bash
# stock vLLM
PORT=8100 GPU_IDS=0 serve/serve.sh

# most optimized run
IMAGE=ghcr.io/dawnik17/gemma3-cls-vllm:0.32.0-fa PORT=8100 GPU_IDS=0 serve/serve.sh --attention-backend CUSTOM --api-server-count 4 

# you should see CUSTOM mentioned
docker logs gemma3-cls 2>&1 | grep "AttentionBackendEnum.CUSTOM backend"

# stop the server
docker rm -f gemma3-cls                                                      
```

## AIPerf

```bash
# once: the AIPerf client image, and the frozen requests in runs/perf/payloads/
docker build -t perf-aiperf:0.13.0 perf/aiperf
python3 utils/payload/payload.py

# Run AIPerf: flags while running NFRs (disable caching)
docker rm -f gemma3-cls
PORT=8100 GPU_IDS=0 serve/serve.sh --attention-backend CUSTOM --api-server-count 4 --no-enable-prefix-caching --mm-processor-cache-gb 0

CONCURRENCY="1 8 16" URL=http://127.0.0.1:8100 GPU_IDS=0 perf/aiperf/bench.sh runs/perf/h200-fa
```

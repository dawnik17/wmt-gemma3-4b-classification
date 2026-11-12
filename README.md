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

## Results

One H200, prefix caching and the multimodal cache disabled, `runs/perf/payloads/cls-2k.jsonl`: 1,978 input tokens per request at p50 (1,947-2,000) including the image, as the server reports them. AIPerf's Input Sequence Length (1,710 at p50) counts the text tokens only.

Throughput (requests/sec):

| Configuration | Concurrency 1 | Concurrency 8 | Concurrency 16 | vs. stock at 16 |
|---|---|---|---|---|
| Stock vLLM, bf16 | 9.71 | 19.37 | 19.58 | baseline |
| Optimized kernels, bf16 | 12.11 | 27.19 | 27.22 | +39% |
| Optimized kernels + W8A8 FP8 | 12.49 | 32.49 | 33.11 | +69% |

Request latency (ms):

| Configuration | p50 at 1 | p50 at 16 | p99 at 16 |
|---|---|---|---|
| Stock vLLM, bf16 | 100.6 | 818.0 | 995.4 |
| Optimized kernels, bf16 | 80.3 | 596.5 | 688.2 |
| Optimized kernels + W8A8 FP8 | 77.7 | 482.8 | 583.5 |

<!--
Copyright 2026 NVIDIA Corporation
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf benchmark

Run from the repo root.

```bash
docker build -t perf-aiperf:0.13.0 perf/aiperf             # once: the AIPerf client image
python3 utils/payload/payload.py                          # once: runs/perf/payloads/cls-2k*.jsonl
docker rm -f gemma3-cls                                   # stop any running server
PORT=8100 GPU_IDS=0 serve/serve.sh --attention-backend CUSTOM --api-server-count 4 --no-enable-prefix-caching --mm-processor-cache-gb 0
URL=http://127.0.0.1:8100 GPU_IDS=0 perf/aiperf/bench.sh runs/perf/<run>
```

Options: `USE_CASES="cls-2k cls-2k-pooling"`, `CONCURRENCY="1 8 32"`, `REQUESTS=200`, `AIPERF_ARGS="..."`.
Results land in `runs/perf/<run>/<use case>/CON<concurrency>/`.

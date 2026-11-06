<!--
Copyright 2026 NVIDIA Corporation
SPDX-License-Identifier: Apache-2.0
-->

# wmt-gemma3-4b-classification

## One Time Setup

```bash
docker build -t gemma3-cls-vllm:0.32.0 serve     # vLLM 0.32.0 image plus the gemma3_cls plugin
```

## vLLM Inference

```bash
GPU_IDS=0 serve/serve.sh
GPU_IDS=1 PORT=8001 MODEL=<checkpoint dir> serve/serve.sh --max-num-seqs 64   # extra args  
docker logs -f gemma3-cls                                # follow the server's log
docker rm -f gemma3-cls                                  # stop the server
```
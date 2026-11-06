# wmt-gemma3-4b-classification

Serve Gemma 3 4B image + text classifier with vLLM 0.32.0.


## One Time Setup

Pull the image:

```bash
docker pull ghcr.io/dawnik17/gemma3-cls-vllm:0.32.0
```

Download the model checkpoint:

```bash
hf download dawn17/gemma3-4b-cls7000 --local-dir .cache/models/gemma3-4b-cls7000
```

<details>
<summary>Or build the image yourself</summary>

```bash
docker build -t ghcr.io/dawnik17/gemma3-cls-vllm:0.32.0 serve
```

</details>

## vLLM Inference

```bash
GPU_IDS=0 serve/serve.sh                                                     
GPU_IDS=1 PORT=8001 MODEL=<checkpoint dir> serve/serve.sh --max-num-seqs 64  # extra args go to vllm serve
docker logs -f gemma3-cls                                                    # follow the server's log
docker rm -f gemma3-cls                                                      # stop the server
```

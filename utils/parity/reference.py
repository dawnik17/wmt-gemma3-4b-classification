# Copyright 2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0
"""HF transformers reference logits for the frozen requests: the gold standard a server is checked against.

python3 utils/parity/reference.py                        # fp32, the 419 requests of runs/perf/payloads/cls-2k.jsonl
python3 utils/parity/reference.py --dtype bfloat16       # HF in bf16: the rounding error any bf16 server has
python3 utils/parity/reference.py --payloads runs/perf/payloads/cls-2k-896.jsonl -n 64
"""

import argparse
import base64
import io
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Gemma3ForSequenceClassification

ROOT = Path(__file__).resolve().parents[2]


def hf_messages(messages):
    """OpenAI-style messages as the HF processor takes them, and their data-URI images decoded."""
    images, out = [], []
    for message in messages:
        content = []
        for item in message["content"]:
            if item["type"] == "image_url":
                data = base64.b64decode(item["image_url"]["url"].split(",", 1)[1])
                images.append(Image.open(io.BytesIO(data)).convert("RGB"))
                content.append({"type": "image"})
            else:
                content.append(item)
        out.append({"role": message["role"], "content": content})
    return out, images


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=str(ROOT / ".cache/models/gemma3-4b-cls7000"))
    ap.add_argument(
        "--payloads", type=Path, default=ROOT / "runs/perf/payloads/cls-2k.jsonl"
    )
    ap.add_argument(
        "-n", type=int, default=None, help="the first n requests (default: all)"
    )
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(args.model)
    model, info = Gemma3ForSequenceClassification.from_pretrained(
        args.model,
        dtype=getattr(torch, args.dtype),
        device_map="cuda",
        output_loading_info=True,
    )
    if info["missing_keys"] or info["mismatched_keys"]:
        raise SystemExit(f"The checkpoint doesn't fit the model: {info}")
    model.eval()

    logits, tokens, start = [], [], time.time()
    with open(args.payloads) as f, torch.inference_mode():
        for line in f:
            if len(logits) == args.n:
                break
            body = json.loads(line)
            messages, images = hf_messages(body["messages"])
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=body.get("add_generation_prompt", False),
            )
            inputs = processor(
                text=[text],
                images=[images] if images else None,
                add_special_tokens=False,
                return_tensors="pt",
            ).to("cuda")
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
            logits.append(model(**inputs).logits[0].float().cpu())
            tokens.append(inputs["input_ids"].shape[1])

    out = (
        args.out
        or ROOT
        / "runs/parity"
        / f"{args.payloads.stem.removesuffix('-pooling')}-{args.dtype}.pt"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "logits": torch.stack(logits),
            "prompt_tokens": torch.tensor(tokens),
            "payloads": str(args.payloads),
            "dtype": args.dtype,
            "image_processor": type(processor.image_processor).__name__,
        },
        out,
    )
    print(
        f"{len(logits)} requests of {min(tokens)}-{max(tokens)} prompt tokens in {time.time() - start:.0f} s "
        f"({args.dtype}, {type(processor.image_processor).__name__}) -> {out}"
    )


if __name__ == "__main__":
    main()

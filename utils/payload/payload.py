# Copyright 2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0
"""Freeze the classification requests every benchmark and check uses: a product photo plus a listing, ~2,000 tokens.

python3 utils/payload/payload.py             # runs/perf/payloads/cls-2k.jsonl and cls-2k-pooling.jsonl
python3 utils/payload/payload.py --side 896  # cls-2k-896(-pooling).jsonl: photos resized by the client to 896 px
"""

import argparse
import base64
import io
import json
import random
from pathlib import Path

from PIL import Image
from transformers import AutoTokenizer
from prompt import INSTRUCTION, listing

ROOT = Path(__file__).resolve().parents[2]


# \n\n <start_of_image> <image_soft_token>×256 <end_of_image> \n\n
IMAGE_TOKENS = 256 + 3

# The API endpoints to use for the requests.
APIS = {
    "": {},  # /classify
    "-pooling": {
        "task": "classify",
        "encoding_format": "base64",
        "embed_dtype": "float32",
    },
}


def cut(pieces, tokenizer, budget):
    """Concatenate pieces until the next one would exceed the token budget."""
    text = ""
    for piece in pieces:
        budget -= len(tokenizer.encode(piece, add_special_tokens=False))
        if budget < 0:
            return text
        text += piece


def image_uri(path, side):
    if side >= 2048:  # the originals, byte for byte
        data = path.read_bytes()
    else:
        buffer = io.BytesIO()
        Image.open(path).convert("RGB").resize(
            (side, side), Image.Resampling.BILINEAR
        ).save(buffer, "JPEG", quality=95)
        data = buffer.getvalue()
    return "data:image/jpeg;base64," + base64.b64encode(data).decode()


def request(image, text):
    content = [
        {"type": "text", "text": INSTRUCTION},
        image,
        {"type": "text", "text": text},
    ]
    return {
        "messages": [{"role": "user", "content": content}],
        "add_generation_prompt": True,
        "use_activation": False,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--model",
        default=str(ROOT / ".cache/models/gemma3-4b-cls7000"),
        help="for its tokenizer",
    )
    ap.add_argument("--images", type=Path, default=ROOT / "data/images")
    ap.add_argument(
        "--tokens",
        type=int,
        default=2000,
        help="prompt tokens per request, image included",
    )
    ap.add_argument(
        "--side",
        type=int,
        default=2048,
        help="image side the client sends (2048: the originals)",
    )
    ap.add_argument(
        "--out", type=Path, default=ROOT / "runs/perf/payloads", help="folder"
    )
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt = tokenizer.apply_chat_template(
        request({"type": "image"}, "X")["messages"],
        add_generation_prompt=True,
        tokenize=False,
    )
    fixed = (
        len(tokenizer.encode(prompt, add_special_tokens=False)) - 1
    )  # all but the "X"
    budget = args.tokens - fixed - IMAGE_TOKENS

    paths = sorted(args.images.glob("*.jpg"))
    random.Random(0).shuffle(paths)
    requests = [
        request(
            {"type": "image_url", "image_url": {"url": image_uri(path, args.side)}},
            cut(listing(random.Random(i), path.name.split("-")[0]), tokenizer, budget),
        )
        for i, path in enumerate(paths)
    ]

    name = f"cls-{'2k' if args.tokens == 2000 else args.tokens}" + (
        "" if args.side >= 2048 else f"-{args.side}"
    )
    args.out.mkdir(parents=True, exist_ok=True)
    for suffix, fields in APIS.items():
        out = args.out / f"{name}{suffix}.jsonl"
        with out.open("w") as f:
            f.writelines(json.dumps({**r, **fields}) + "\n" for r in requests)
        print(
            f"{len(requests)} requests of ~{args.tokens} prompt tokens ({budget} for the listing), "
            f"{out.stat().st_size / 2**20:.0f} MiB -> {out}"
        )


if __name__ == "__main__":
    main()

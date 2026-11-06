# Copyright 2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0
"""Check a server's logits for the frozen requests against Hugging Face's float32 model (reference.py).

python3 utils/parity/parity.py runs/perf/<run>/logits-cls-2k.pt --url http://127.0.0.1:8100   # ask, save, score
python3 utils/parity/parity.py runs/perf/<run>/logits-cls-2k.pt                              # score saved logits
python3 utils/parity/parity.py <logits file> --url ... --payloads runs/perf/payloads/cls-2k-pooling.jsonl
"""

import argparse
import base64
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def ask(url, body):
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request) as response:
        answer = json.load(response)
    data = answer["data"][0]
    if "probs" in data:
        logits = torch.tensor(data["probs"])
    else:
        logits = torch.frombuffer(
            bytearray(base64.b64decode(data["data"])), dtype=torch.float32
        )
    return logits, answer["usage"]["prompt_tokens"]


def score(logits, gold, tokens, gold_tokens):
    top5, gold_top5 = logits.topk(5).indices, gold.topk(5).indices
    row = {
        "cos min": torch.nn.functional.cosine_similarity(logits, gold).min().item(),
        "cos mean": torch.nn.functional.cosine_similarity(logits, gold).mean().item(),
        "max |err|": (logits - gold).abs().max().item(),
        "top-1": (top5[:, 0] == gold_top5[:, 0]).float().mean().item(),
        "top-5": torch.tensor(
            [
                len(set(a.tolist()) & set(b.tolist())) / 5
                for a, b in zip(top5, gold_top5)
            ]
        )
        .mean()
        .item(),
    }
    if tokens is not None:
        row["tokens"] = (tokens == gold_tokens).float().mean().item()
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "logits", type=Path, help="the logits file: written with --url, read without"
    )
    ap.add_argument(
        "--url", help="the server to ask; without it, score a logits file saved before"
    )
    ap.add_argument(
        "--payloads", type=Path, default=ROOT / "runs/perf/payloads/cls-2k.jsonl"
    )
    ap.add_argument(
        "-n", type=int, default=None, help="the first n requests (default: all)"
    )
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="default: runs/parity/<use case>-float32.pt",
    )
    args = ap.parse_args()

    if args.url:
        endpoint = (
            "/pooling" if args.payloads.stem.endswith("-pooling") else "/classify"
        )
        with open(args.payloads) as f:
            bodies = [json.loads(line) for line in f][: args.n]

        with ThreadPoolExecutor(args.concurrency) as pool:
            answers = list(
                pool.map(lambda body: ask(args.url + endpoint, body), bodies)
            )
        args.logits.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "logits": torch.stack([a[0] for a in answers]),
                "prompt_tokens": torch.tensor([a[1] for a in answers]),
                "payloads": str(args.payloads),
            },
            args.logits,
        )
        print(f"{len(answers)} requests to {endpoint} -> {args.logits}")

    server = torch.load(args.logits)
    reference = (
        args.reference
        or ROOT
        / "runs/parity"
        / f"{args.payloads.stem.removesuffix('-pooling')}-float32.pt"
    )
    gold = torch.load(reference)

    n = min(len(gold["logits"]), len(server["logits"]))
    gold_logits, gold_tokens = gold["logits"][:n], gold["prompt_tokens"][:n]

    tokens = server["prompt_tokens"][:n] if "prompt_tokens" in server else None
    rows = {"server": score(server["logits"][:n], gold_logits, tokens, gold_tokens)}

    bf16 = reference.with_name(reference.name.replace("-float32", "-bfloat16"))
    hf = torch.load(bf16) if bf16 != reference and bf16.exists() else None
    if hf is not None and len(hf["logits"]) >= n:
        rows = {
            "HF bf16": score(
                hf["logits"][:n], gold_logits, hf["prompt_tokens"][:n], gold_tokens
            ),
            **rows,
        }

    print(
        f"{n} requests vs {reference.name} (prompt tokens {gold_tokens.min()}-{gold_tokens.max()})"
    )
    print(f"{'':10}" + "".join(f"{k:>11}" for k in rows["server"]))
    for name, row in rows.items():
        print(
            f"{name:10}"
            + "".join(
                f"{v:>11.5f}" if "cos" in k else f"{v:>11.3f}" for k, v in row.items()
            )
        )
    if "HF bf16" in rows:
        hf, server = rows["HF bf16"], rows["server"]
        ok = (
            1 - server["cos mean"] <= 2 * (1 - hf["cos mean"])
            and 1 - server["top-1"] <= 2 * (1 - hf["top-1"]) + 1 / n
            and server.get("tokens", 1) == 1
        )
        print(
            "PASS: within twice HF's own bf16 error"
            if ok
            else "FAIL: more than twice HF's own bf16 error"
        )


if __name__ == "__main__":
    main()

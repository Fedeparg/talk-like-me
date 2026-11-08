"""Exposure and extraction of the canaries planted by build_canaries.py.

For each canary, the model is given the text right before its code and scored on the code:
- exposure: the rank of the real code's NLL among N invented codes of the same format;
  exposure = log2(N + 1) - log2(rank), from 0 (no trace) to log2(N + 1) (the most likely of all).
  With N = 1000 the finest rank-1 result is p = 1/1001.
- extraction: whether greedy generation from that prefix writes the exact code.
Both in two settings: inside one of the training conversations where it was planted ("context"),
and with nothing but the system prompt and the sentence ("bare").
Only the code's tokens are scored (not the end of the message), so the result is about the code.

    uv run python scripts/eval_exposure.py --out experiments/results/memorization.json            # base model
    uv run python scripts/eval_exposure.py --adapter adapters/mem_canaries --steps 500:3500:500 ...
"""

import _setup  # noqa: F401
import argparse
import math
import random
import string
from pathlib import Path

import mlx.core as mx
from mlx_lm import generate

import build_dataset as bd
from talk_like_me.config import DATASET, MODEL
from talk_like_me.io import read_json, read_jsonl, write_json
from talk_like_me.nll import load_checkpoint, load_model

BATCH = 8  # candidates per forward pass: full-sequence logits are batch x length x 151,936


def candidates(real, n, seed):
    rng, out = random.Random(seed), set()
    while len(out) < n:
        c = f"{rng.randint(0, 9999):04d}-{rng.choice(string.ascii_uppercase)}{rng.choice(string.ascii_uppercase)}"
        if c != real:
            out.add(c)
    return sorted(out)


def prompt_ids(tok, messages, reply_prefix):
    """Token ids of the conversation plus the start of my reply, up to where the code begins."""
    return tok.apply_chat_template(messages, add_generation_prompt=True) + tok.encode(reply_prefix, add_special_tokens=False)


def code_nlls(model, tok, prefix, codes):
    """NLL of each code's tokens after the same prefix, in batches (right padding doesn't affect causal logits)."""
    out = []
    for i in range(0, len(codes), BATCH):
        suf = [tok.encode(c, add_special_tokens=False) for c in codes[i:i + BATCH]]
        width = max(map(len, suf))
        x = mx.array([prefix + s + [0] * (width - len(s)) for s in suf])
        logits = model(x)[:, len(prefix) - 1:len(prefix) - 1 + width].astype(mx.float32)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        targets = mx.array([s + [0] * (width - len(s)) for s in suf])
        mask = mx.array([[1.0] * len(s) + [0.0] * (width - len(s)) for s in suf])
        picked = mx.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
        out += (-(picked * mask).sum(axis=1)).tolist()
        del logits, logp, picked
        mx.clear_cache()
    return out


def measure(model, tok, canaries, train, n):
    rows = []
    for c in canaries:
        cands = candidates(c["code"], n, seed=c["id"])
        example = train[c["lines"][0]]["messages"]
        before = example[-1]["content"]
        before = before[: before.rindex(c["sentence"] + c["code"])] + c["sentence"]
        settings = {
            "context": prompt_ids(tok, example[:-1], before),
            "bare": prompt_ids(tok, [{"role": "system", "content": bd.SYSTEM_PROMPT}], c["sentence"]),
        }
        # The code must tokenize the same after the prefix as inside the training text.
        assert tok.encode(before + c["code"], add_special_tokens=False) == \
            tok.encode(before, add_special_tokens=False) + tok.encode(c["code"], add_special_tokens=False), c["id"]
        row = {"id": c["id"], "reps": c["reps"]}
        for name, prefix in settings.items():
            real = code_nlls(model, tok, prefix, [c["code"]])[0]
            others = code_nlls(model, tok, prefix, cands)
            rank = 1 + sum(x <= real for x in others)
            text = generate(model, tok, prompt=prefix, max_tokens=8)  # greedy
            row[name] = {"nll": real, "rank": rank, "exposure": math.log2(n + 1) - math.log2(rank),
                         "extracted": text.strip().startswith(c["code"])}
        rows.append(row)
        print(f"  canary {c['id']:>2} x{c['reps']:<2} | context rank {row['context']['rank']:>4} "
              f"exp {row['context']['exposure']:.2f}{' EXTRACTED' if row['context']['extracted'] else ''} | "
              f"bare rank {row['bare']['rank']:>4} exp {row['bare']['exposure']:.2f}"
              f"{' EXTRACTED' if row['bare']['extracted'] else ''}", flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--adapter", help="omit for the base model")
    p.add_argument("--steps", help="start:stop:every checkpoints of the adapter")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--candidates", type=int, default=1000)
    p.add_argument("--out", required=True, help="results .json; each model or step is added to it")
    args = p.parse_args()

    meta = read_json(DATASET / "canaries" / "canaries.json")
    train = read_jsonl(DATASET / "canaries" / "train.jsonl")
    out = Path(args.out)
    res = read_json(out) if out.exists() else {"source": "scripts/eval_exposure.py", "model": args.model,
                                               "candidates": args.candidates, "reps": meta["reps"], "runs": {}}
    model, tok = load_model(args.adapter, model=args.model)
    if not args.adapter:
        if "base" not in res["runs"]:
            print("base model")
            res["runs"]["base"] = measure(model, tok, meta["canaries"], train, args.candidates)
            write_json(out, res)
        return
    start, stop, every = (int(x) for x in args.steps.split(":"))
    for step in range(start, stop + 1, every):
        key = f"{Path(args.adapter).name}@{step}"
        if key in res["runs"]:
            print(f"{key}: cached")
            continue
        print(key)
        load_checkpoint(model, args.adapter, step)
        res["runs"][key] = measure(model, tok, meta["canaries"], train, args.candidates)
        write_json(out, res)


if __name__ == "__main__":
    main()

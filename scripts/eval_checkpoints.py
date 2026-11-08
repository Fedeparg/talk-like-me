"""Validation curve of an adapter: nll_ejemplo on a file for every saved checkpoint (early stopping).

Measures each checkpoint on the whole file (mlx's own validation uses 25 batches and counts a
padding token), saves after every checkpoint so an interrupted run resumes, and keeps the
per-example numbers so any two checkpoints or runs can be compared paired (compare.py).

    uv run python scripts/eval_checkpoints.py adapters/run1_5ep --data data/dataset/session/valid.jsonl \
        --out experiments/results/checkpoints_run1_5ep.json
    uv run python scripts/eval_checkpoints.py adapters/data25 --steps 100:1400:100 ...

Notebooks this replaces: Checkpoints, LearningRate, ContextRetrain, DataCurve.
"""

import _setup  # noqa: F401
import argparse
import json
import re
from pathlib import Path

from talk_like_me.config import MODEL
from talk_like_me.io import read_json, write_json
from talk_like_me.nll import load_checkpoint, load_model, mean_loss, measure


def saved_steps(adapter):
    return sorted(int(m.group(1)) for f in Path(adapter).glob("*_adapters.safetensors")
                  if (m := re.match(r"(\d{7})_adapters", f.name)))


def parse_steps(spec, adapter):
    if not spec:
        return saved_steps(adapter)
    start, stop, every = (int(x) for x in spec.split(":"))
    return list(range(start, stop + 1, every))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("adapter")
    p.add_argument("--data", required=True, help="usually valid.jsonl (or a train sample to see overfitting)")
    p.add_argument("--out", required=True, help="results .json; resumed if it exists")
    p.add_argument("--steps", help="start:stop:every; default: every checkpoint saved in the adapter folder")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--limit", type=int, help="first N examples only (smoke test)")
    args = p.parse_args()

    steps = parse_steps(args.steps, args.adapter)
    missing = [s for s in steps if not (Path(args.adapter) / f"{s:07d}_adapters.safetensors").exists()]
    if missing:
        raise SystemExit(f"missing checkpoints: {missing}")

    out = Path(args.out)
    res = read_json(out) if out.exists() else {
        "source": "scripts/eval_checkpoints.py", "model": args.model, "adapter": args.adapter,
        "data": args.data, "valid": {}, "per_example": {},  # per example, in file order: [nll_sum, n_tokens]
    }
    model = tok = None  # loaded only if some checkpoint is still missing, so a cached run is instant
    for step in steps:
        if str(step) in res["per_example"]:
            print(f"step {step:>5}: loss {res['valid'][str(step)]:.3f} (cached)")
            continue
        if model is None:
            model, tok = load_model(args.adapter, model=args.model)
        load_checkpoint(model, args.adapter, step)
        r = measure(model, tok, args.data, args.limit)
        res["per_example"][str(step)] = r
        res["valid"][str(step)] = mean_loss(r)
        write_json(out, res)
        print(f"step {step:>5}: loss {mean_loss(r):.3f}")

    valid = {int(k): v for k, v in res["valid"].items()}
    best = min(valid, key=valid.get)
    res["best"] = {"step": best, "loss": valid[best]}
    write_json(out, res)
    print(f"best: step {best}, loss {valid[best]:.3f}")


if __name__ == "__main__":
    main()

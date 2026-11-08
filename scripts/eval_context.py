"""How much context to give at prediction time (decision 17).

Measures the adapter and the base model on the same valid replies with 1, 5, 6, 10 and 20
previous turns (data/dataset/context/valid_ctx{N}.jsonl, from build_context.py), then compares
6 vs 20 paired, with a 95% CI by session.

    uv run python scripts/eval_context.py --adapter adapters/run1_5ep --step 2500

Reported: adapter 3.418 with 6 turns, 3.394 with 20 (-0.024, 95% CI [-0.031, -0.014]).
"""

import _setup  # noqa: F401
import argparse
from pathlib import Path

from talk_like_me.config import DATASET, MODEL, RESULTS
from talk_like_me.io import read_json, write_json
from talk_like_me.nll import load_model, mean_loss, measure
from talk_like_me.stats import paired_diff

CONTEXTS = [1, 5, 6, 10, 20]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--adapter", required=True)
    p.add_argument("--step", type=int)
    p.add_argument("--out", default=str(RESULTS / "context.json"))
    p.add_argument("--force", action="store_true", help="replace --out if it exists")
    args = p.parse_args()

    folder = DATASET / "context"
    meta = read_json(folder / "valid_meta.json")["examples"]
    sessions = [e["session"] for e in meta]
    per_example = {}
    for name, adapter, step in [("adapter", args.adapter, args.step), ("base", None, None)]:
        model, tok = load_model(adapter, step)
        per_example[name] = {}
        for n in CONTEXTS:
            r = measure(model, tok, folder / f"valid_ctx{n}.jsonl")
            per_example[name][str(n)] = r
            print(f"{name:>7} ctx {n:>2}: loss {mean_loss(r):.3f}")
    for name, r in per_example.items():
        d, (lo, hi) = paired_diff(r["6"], r["20"], sessions)
        print(f"{name:>7}: 20 vs 6 turns {d:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}]")

    write_json(Path(args.out), {
        "source": "scripts/eval_context.py", "model": MODEL, "adapter": f"{args.adapter}@{args.step}",
        "adapter_all": {n: mean_loss(r) for n, r in per_example["adapter"].items()},
        "base_all": {n: mean_loss(r) for n, r in per_example["base"].items()},
        "session": sessions, "per_example": per_example,
    }, force=args.force)


if __name__ == "__main__":
    main()

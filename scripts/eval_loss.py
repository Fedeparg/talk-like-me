"""Loss and perplexity of a model on one dataset file, counting reply tokens only.

    uv run python scripts/eval_loss.py data/dataset/session/test.jsonl --adapter adapters/run1_5ep --step 2500
    uv run python scripts/eval_loss.py data/dataset/session/test.jsonl            # base model

The leakage experiment (decision 14) is this script run on by_message/test and session/test.
"""

import _setup  # noqa: F401
import argparse
import math

from talk_like_me.nll import load_model, mean_loss, measure


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("data", help="chat-format .jsonl")
    p.add_argument("--adapter", help="adapter folder; omit for the base model")
    p.add_argument("--step", type=int, help="checkpoint step inside the adapter folder")
    p.add_argument("--limit", type=int, help="first N examples only (smoke test)")
    args = p.parse_args()

    model, tok = load_model(args.adapter, args.step)
    r = measure(model, tok, args.data, args.limit)
    loss = mean_loss(r)
    print(f"{len(r)} examples, {sum(x[1] for x in r)} reply tokens | loss {loss:.3f} | perplexity {math.exp(loss):.1f}")


if __name__ == "__main__":
    main()

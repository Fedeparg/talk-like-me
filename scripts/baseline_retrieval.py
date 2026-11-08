"""Retrieval baseline: for each test context, reply with my real reply to the most similar train context.

No model at all. Context similarity is TF-IDF cosine over word 1-2 grams of the last 6 turns,
fitted on train contexts only. If a fine-tuned model can't beat copying my own past replies,
it hasn't learned much more than a lookup table.

    uv run python scripts/baseline_retrieval.py                 # data/style/generations_retrieval.jsonl
    uv run python scripts/baseline_retrieval.py --pool valid    # ..._retrieval_valid.jsonl

Pool "train" copies replies the style classifier was trained on, so it recognises them and the
style score is inflated. Pool "valid" uses replies it never trained on (valid only chose its
hyperparameters), at the cost of a smaller pool (1,012 replies instead of 5,404).
"""

import _setup  # noqa: F401
import argparse
import json

import numpy as np

import build_dataset as bd
from talk_like_me.config import DATA, DATASET
from talk_like_me.io import read_json, read_jsonl, write_json

TURNS = 6


def context_text(messages):
    turns = [m for m in messages[:-1] if m["role"] != "system"][-TURNS:]
    return "\n".join(m["content"] for m in turns)


def similar(pool, contexts, k=1):
    """For each context, my replies in the k most similar contexts of the pool, most similar first.

    The indices are cached in data/style/ (no text, just positions in the pool) so machines
    without scikit-learn (the NAS container) can reuse the lookup made on the Mac.
    """
    examples = read_jsonl(DATASET / "session" / f"{pool}.jsonl")
    # Replies without the planted canary line.
    replies = [e["messages"][-1]["content"].removesuffix("\n" + bd.CANARY) for e in examples]
    cache = DATA / "style" / f"retrieval_{pool}_top{k}_{len(contexts)}.json"
    if cache.exists():
        top = read_json(cache)
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        a = vec.fit_transform([context_text(e["messages"]) for e in examples])
        b = vec.transform([context_text(e["messages"]) for e in contexts])
        sims = (b @ a.T).toarray()  # rows are L2-normalised: this is the cosine
        top = np.argsort(-sims, axis=1, kind="stable")[:, :k].tolist()
        write_json(cache, top)
    return [[replies[j] for j in row] for row in top]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pool", choices=["train", "valid"], default="train", help="replies to copy from")
    args = p.parse_args()
    test = read_jsonl(DATASET / "session_ctx20" / "test.jsonl")
    picked = [r[0] for r in similar(args.pool, test)]
    out = DATA / "style" / ("generations_retrieval.jsonl" if args.pool == "train" else f"generations_retrieval_{args.pool}.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for k, text in enumerate(picked):
            f.write(json.dumps({"k": k, "text": text}, ensure_ascii=False) + "\n")
    print(f"{len(test)} replies retrieved from the {args.pool} pool -> {out.relative_to(DATA.parent)}")


if __name__ == "__main__":
    main()

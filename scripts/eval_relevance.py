"""Does the reply answer the conversation? (decision 20)

For each test context, how much seeing the conversation lowers the NLL of the reply, per token:
NLL without context - NLL with context, scored by a model that generated none of the replies
(Qwen2.5-1.5B). Compared on four sets: my real reply (ceiling), a reply of mine from another
session of the same length bin (floor: fits the register but not the conversation), the
adapter's and the base model's generations. Position of the adapter = (adapter - floor) /
(real - floor), 95% CI by session.

    uv run python scripts/eval_relevance.py --adapter-gens adapter_t1 --base-gens base_t1

Reported: real 1.04, adapter 0.85, floor 0.80, base 0.27; position 0.20 [-0.11, 0.45].
Slow (four sets x 1,133 x 2 passes of a 1.5B model); per-example numbers are cached and resumed.
"""

import _setup  # noqa: F401
import argparse
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from talk_like_me.config import DATA, DATASET, RESULTS, SCORER, SEED
from talk_like_me.io import read_json, read_jsonl, write_json
from talk_like_me.nll import load_model, nll_ejemplo
from talk_like_me.stats import interval, resample

TURNS = 6                  # context given to the scorer: the adapter's training window
MAX_CHARS_REPLY = 300      # a few base-model rambles are much longer than any reply of mine
BINS = [0, 10, 20, 40, 80, 160, 10**9]


def length_bin(t):
    return next(i for i in range(len(BINS) - 1) if BINS[i] <= len(t) < BINS[i + 1])


def other_session_replies(replies, sessions, seed=SEED):
    """For each reply, a reply of mine from another session, same length bin, fixed seed."""
    rng, by_bin = random.Random(seed), defaultdict(list)
    for j, r in enumerate(replies):
        by_bin[length_bin(r)].append(j)
    out = []
    for i, r in enumerate(replies):
        while True:
            j = rng.choice(by_bin[length_bin(r)])
            if sessions[j] != sessions[i]:
                break
        out.append(replies[j])
    return out


def messages(example, reply, with_context):
    """System prompt (+ the last TURNS turns) + the reply as the assistant's turn."""
    msgs = example["messages"]
    turns = [m for m in msgs[:-1] if m["role"] != "system"][-TURNS:] if with_context else []
    return [msgs[0]] + turns + [{"role": "assistant", "content": reply[:MAX_CHARS_REPLY]}]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--adapter-gens", required=True)
    p.add_argument("--base-gens", required=True)
    p.add_argument("--extra", nargs="*", default=[], help="more generation names to score, each with its own position")
    p.add_argument("--scorer", default=SCORER, help="must not have generated any of the sets")
    p.add_argument("--cache", default=str(RESULTS / "relevance_nll.json"), help="per-example numbers, resumed")
    p.add_argument("--out", default=str(RESULTS / "relevance.json"))
    p.add_argument("--force", action="store_true", help="replace --out if it exists")
    args = p.parse_args()

    examples = read_jsonl(DATASET / "session_ctx20" / "test.jsonl")
    mine = [r for r in read_jsonl(DATA / "style" / "test.jsonl") if r["label"] == 1]
    replies, sessions = [r["text"] for r in mine], [r["session"] for r in mine]
    assert len(replies) == len(examples)
    sets = {
        "floor (other conv.)": other_session_replies(replies, sessions),
        "base t1.0": [g["text"] for g in read_jsonl(DATA / "style" / f"generations_{args.base_gens}.jsonl")],
        "adapter t1.0": [g["text"] for g in read_jsonl(DATA / "style" / f"generations_{args.adapter_gens}.jsonl")],
        "real (me)": replies,
    } | {e: [g["text"] for g in read_jsonl(DATA / "style" / f"generations_{e}.jsonl")] for e in args.extra}
    assert all(len(v) == len(examples) for v in sets.values())

    cache = Path(args.cache)
    nll = read_json(cache) if cache.exists() else {}
    scorer, tok = load_model(model=args.scorer)
    start = time.time()
    for name, texts in sets.items():
        rows = nll.setdefault(name, [])  # per example: [nll_ctx, n_ctx, nll_noctx, n_noctx]
        for k in range(len(rows), len(texts)):
            a = nll_ejemplo(scorer, tok, messages(examples[k], texts[k], True))
            b = nll_ejemplo(scorer, tok, messages(examples[k], texts[k], False))
            rows.append([a[0], a[1], b[0], b[1]])
            if (k + 1) % 100 == 0 or k + 1 == len(texts):
                write_json(cache, nll)
                print(f"{name}: {k + 1}/{len(texts)} ({(time.time() - start) / 60:.1f} min)")

    sums = {}
    for name, rows in nll.items():
        s_ = defaultdict(lambda: np.zeros(3))  # nll with context, without, tokens
        for (a, n, b, n2), s in zip(rows, sessions):
            assert n == n2
            s_[s] += (a, b, n)
        sums[name] = dict(s_)

    def gain(s_, chosen):
        tot = sum(s_[s] for s in chosen)
        return (tot[1] - tot[0]) / tot[2]

    nll = {k: v for k, v in nll.items() if k in sets}  # the cache may hold sets from other runs
    movable = [k for k in sets if k not in ("floor (other conv.)", "real (me)")]
    boot = {k: [] for k in nll} | {f"position {k}": [] for k in movable}
    for chosen in resample(set(sessions)):
        g = {k: gain(sums[k], chosen) for k in nll}
        for k in nll:
            boot[k].append(g[k])
        for k in movable:
            boot[f"position {k}"].append((g[k] - g["floor (other conv.)"]) / (g["real (me)"] - g["floor (other conv.)"]))
    all_s = sorted(set(sessions))
    res = {k: {"gain": gain(sums[k], all_s), "ci": interval(boot[k])} for k in nll}
    for k, v in res.items():
        print(f"{k:<20} gain {v['gain']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]")
    positions = {}
    for k in movable:
        pos = (res[k]["gain"] - res["floor (other conv.)"]["gain"]) / (res["real (me)"]["gain"] - res["floor (other conv.)"]["gain"])
        positions[k] = {"value": pos, "ci": interval(boot[f"position {k}"])}
        print(f"position of {k}: {pos:.2f} [{positions[k]['ci'][0]:.2f}, {positions[k]['ci'][1]:.2f}] (0 = another conversation's reply, 1 = my real one)")
    write_json(args.out, {"source": "scripts/eval_relevance.py", "scorer": args.scorer, "context_turns": TURNS,
                          "context_gain": res, "position": positions["adapter t1.0"], "positions": positions}, force=args.force)


if __name__ == "__main__":
    main()

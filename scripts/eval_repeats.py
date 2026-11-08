"""Does the me-against-friends signal survive without repeated replies (catchphrases)?

Many of my replies repeat: short reactions and stock phrases that appear many times in train. A
model trained on me predicts those cheaply, which could produce the signal on its own without
learning anything else about me. Here the signal of the final adapter is recomputed after
removing, *on each side symmetrically*, the test replies that repeat that speaker's own train
replies:

- "exact repeats": test replies identical (lowercased, spaces collapsed) to a train reply of the
  same speaker;
- "frequent or very short": test replies among that speaker's 100 most frequent train replies,
  or of 5 characters or fewer.

Uses the saved per-reply numbers of scripts/eval_signal.py; the texts are only compared, never
printed. Intervals by session and by chat, as in eval_by_chat.py. Prints numbers only.

    uv run python scripts/eval_repeats.py --out experiments/results/repeats.json
"""

import _setup  # noqa: F401
import argparse
from collections import Counter

import build_dataset as bd
from eval_by_chat import chat_of_session, signal_ci, TURNS_V1
from talk_like_me.config import DATA, DATASET, RESULTS
from talk_like_me.io import read_json, read_jsonl, write_json
from talk_like_me.sessions import test_sessions

TOP = 100


def norm(text):
    return " ".join(text.lower().split())


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--signal", default=str(RESULTS / "signal_run1_5ep.json"))
    p.add_argument("--out", default=str(RESULTS / "repeats.json"))
    args = p.parse_args()

    test = {"me": [norm(e["messages"][-1]["content"]) for e in read_jsonl(DATASET / "session" / "test.jsonl")],
            "friends": [norm(e["messages"][-1]["content"]) for e in read_jsonl(DATASET / "session" / "test_friends.jsonl")]}
    train_me = [norm(e["messages"][-1]["content"].removesuffix("\n" + bd.CANARY)) for e in read_jsonl(DATASET / "session" / "train.jsonl")]
    train_fr = [norm(r["text"]) for r in read_jsonl(DATA / "style" / "train.jsonl") if r["label"] == 0]
    train = {"me": train_me, "friends": train_fr}
    seen = {k: set(v) for k, v in train.items()}
    top = {k: {t for t, _ in Counter(v).most_common(TOP)} for k, v in train.items()}

    pe = read_json(args.signal)["per_example"]
    mine, friends = test_sessions(TURNS_V1)
    sess = {"me": mine, "friends": friends}
    assert len(test["me"]) == len(mine) == len(pe["base_me"]) and len(test["friends"]) == len(friends) == len(pe["base_friends"])
    chat = chat_of_session()

    filters = {
        "all replies": lambda who, t: True,
        "without exact repeats": lambda who, t: t not in seen[who],
        "without frequent or very short": lambda who, t: t not in top[who] and len(t) > 5,
        "without either": lambda who, t: t not in seen[who] and t not in top[who] and len(t) > 5,
    }
    res = {"source": "scripts/eval_repeats.py", "signal": args.signal, "top": TOP}
    for name, keep in filters.items():
        idx = {who: [i for i, t in enumerate(test[who]) if keep(who, t)] for who in test}
        sub = {k: [v[i] for i in idx["me" if k.endswith("_me") else "friends"]] for k, v in pe.items()}
        s_me = [sess["me"][i] for i in idx["me"]]
        s_fr = [sess["friends"][i] for i in idx["friends"]]
        v, ci_s = signal_ci(sub, s_me, s_fr)
        _, ci_c = signal_ci(sub, [chat[s] for s in s_me], [chat[s] for s in s_fr])
        kept = {who: len(idx[who]) / len(test[who]) for who in test}
        res[name] = {"value": v, "by_session": ci_s, "by_chat": ci_c, "kept_me": kept["me"], "kept_friends": kept["friends"]}
        print(f"{name:<32} kept: mine {kept['me']:.1%}, friends {kept['friends']:.1%} | signal {v * 100:.1f} points "
              f"| by session [{ci_s[0] * 100:.1f}, {ci_s[1] * 100:.1f}] | by chat [{ci_c[0] * 100:.1f}, {ci_c[1] * 100:.1f}]")
    write_json(args.out, res)


if __name__ == "__main__":
    main()

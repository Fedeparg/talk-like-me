"""The main intervals again, resampling whole chats instead of sessions (cluster bootstrap).

Sessions from the same chat share a person and recurring topics, so they are not independent,
and a few chats hold most of the test replies. Here every chat is one unit: a bootstrap sample
draws chats with replacement and takes all their sessions. The point estimates don't change;
only the intervals do. Same 1,000 resamples and seed as everywhere else, so the by-session
intervals printed next to them reproduce the published ones.

Reads only saved per-example numbers (signal, relevance, validation checkpoints), except the
style test, which re-applies the frozen classifiers to the saved generations (--transformer for
mDeBERTa, which needs the fine-tuned model under models/). Prints numbers only.

    uv run python scripts/eval_by_chat.py --transformer --out experiments/results/by_chat.json
"""

import _setup  # noqa: F401
import argparse
import json
import math

import numpy as np

import eval_style as es
from talk_like_me.config import DATA, DATASET, RESULTS
from talk_like_me.io import read_json, read_jsonl, write_json
from talk_like_me.sessions import test_sessions, valid_sessions
from talk_like_me.stats import by_session, interval, loss_over, resample

TURNS_V1 = "data/processed_v1_2026-10-17/turns.jsonl"  # the turns every dataset here was built from


def chat_of_session():
    chat = {}
    with open(TURNS_V1, encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)  # only the session and chat ids are used
            chat.setdefault(t["session"], t["chat"])
    return chat


def signal_ci(per_example, mine, friends):
    """Me-against-friends signal (percentage points of perplexity drop), CI over the given groups."""
    sums = {k: by_session(v, mine if k.endswith("_me") else friends) for k, v in per_example.items()}

    def signal(chosen):
        p = {k: math.exp(loss_over(s, chosen)) for k, s in sums.items()}
        return (1 - p["adapter_me"] / p["base_me"]) - (1 - p["adapter_friends"] / p["base_friends"])

    keys = set(mine) | set(friends)
    return signal(sorted(keys)), interval([signal(c) for c in resample(keys)])


def relevance(nll, groups, sets):
    """Relevance position of each set and paired differences in gain, CI over the given groups."""
    sums = {}
    for name in nll:
        s_ = {}
        for (a, n, b, _), g in zip(nll[name], groups):
            acc = s_.setdefault(g, np.zeros(3))
            acc += (a, b, n)
        sums[name] = s_

    def gain(name, chosen):
        tot = sum(sums[name][g] for g in chosen)
        return (tot[1] - tot[0]) / tot[2]

    floor, real = "floor (other conv.)", "real (me)"
    pos = lambda k, c: (gain(k, c) - gain(floor, c)) / (gain(real, c) - gain(floor, c))
    keys = sorted(set(groups))
    samples = resample(keys)
    out = {}
    for k in sets:
        out[f"position {k}"] = {"value": pos(k, keys), "ci": interval([pos(k, c) for c in samples])}
    for a, b in [("retrieval_valid", "adapter t1.0"), ("adapter1.5b_t1", "adapter t1.0"), ("adapter t1.0", floor),
                 ("adapter_rag3", "adapter t1.0")]:
        d = lambda c: gain(a, c) - gain(b, c)
        out[f"gain {a} - {b}"] = {"value": d(keys), "ci": interval([d(c) for c in samples])}
    return out


def paired(before, after, groups):
    a, b = by_session(before, groups), by_session(after, groups)
    keys = sorted(a)
    d = lambda c: loss_over(b, c) - loss_over(a, c)
    return {"value": d(keys), "ci": interval([d(c) for c in resample(keys)])}


def best_step(name):
    v = read_json(RESULTS / f"checkpoints_{name}.json")["valid"]
    return min(v, key=v.get)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--transformer", action="store_true", help="also re-apply the mDeBERTa style classifier")
    p.add_argument("--out", default=str(RESULTS / "by_chat.json"))
    args = p.parse_args()
    chat = chat_of_session()
    res = {"source": "scripts/eval_by_chat.py", "resamples": "chats (by_chat) and sessions (by_session), 1000, seed 0"}

    # 1. Signal of the final adapter (and of the other seeds).
    mine, friends = test_sessions(TURNS_V1)
    print(f"test: {len(set(mine) | set(friends))} sessions in {len({chat[s] for s in set(mine) | set(friends)})} chats")
    for name in ["signal_run1_5ep", "signal_run1_seed1", "signal_run1_seed2"]:
        pe = read_json(RESULTS / f"{name}.json")["per_example"]
        v, ci_s = signal_ci(pe, mine, friends)
        _, ci_c = signal_ci(pe, [chat[s] for s in mine], [chat[s] for s in friends])
        res[name] = {"value": v, "by_session": ci_s, "by_chat": ci_c}
        print(f"{name}: {v * 100:.1f} points | by session [{ci_s[0] * 100:.1f}, {ci_s[1] * 100:.1f}] "
              f"| by chat [{ci_c[0] * 100:.1f}, {ci_c[1] * 100:.1f}]")

    # 2. Relevance (Qwen2.5-7B judge).
    style_test = read_jsonl(DATA / "style" / "test.jsonl")
    sess = [r["session"] for r in style_test if r["label"] == 1]
    nll = read_json(RESULTS / "relevance_nll_7b.json")
    sets = ["adapter t1.0", "adapter1.5b_t1", "fewshot20_3b", "retrieval_valid", "adapter_rag3"]
    by_s, by_c = relevance(nll, sess, sets), relevance(nll, [chat[s] for s in sess], sets)
    res["relevance"] = {k: {"value": by_s[k]["value"], "by_session": by_s[k]["ci"], "by_chat": by_c[k]["ci"]} for k in by_s}
    for k, v in res["relevance"].items():
        print(f"relevance {k}: {v['value']:+.3f} | by session [{v['by_session'][0]:+.3f}, {v['by_session'][1]:+.3f}] "
              f"| by chat [{v['by_chat'][0]:+.3f}, {v['by_chat'][1]:+.3f}]")

    # 3. Paired validation differences: data curve (seed 0) and all layers (seed 0).
    vs = valid_sessions()
    vc = [chat[s] for s in vs]
    full = read_json(RESULTS / "context_v1.json")["per_example"]["adapter"]["20"]
    for name in ["data25", "data50"]:
        pe = read_json(RESULTS / f"checkpoints_{name}.json")["per_example"][best_step(name)]
        s_, c_ = paired(full, pe, vs), paired(full, pe, vc)
        res[f"valid {name} - 100%"] = {"value": s_["value"], "by_session": s_["ci"], "by_chat": c_["ci"]}
    for a, b in [("size_0.5b", "alllayers_0.5b"), ("size_1.5b", "alllayers_1.5b"), ("size_0.5b", "size_1.5b")]:
        pa = read_json(RESULTS / f"checkpoints_{a}.json")["per_example"]["2500"]
        pb = read_json(RESULTS / f"checkpoints_{b}.json")["per_example"]["2500"]
        s_, c_ = paired(pa, pb, vs), paired(pa, pb, vc)
        res[f"valid {b} - {a}"] = {"value": s_["value"], "by_session": s_["ci"], "by_chat": c_["ci"]}
    for k, v in res.items():
        if k.startswith("valid "):
            print(f"{k}: {v['value']:+.3f} | by session [{v['by_session'][0]:+.3f}, {v['by_session'][1]:+.3f}] "
                  f"| by chat [{v['by_chat'][0]:+.3f}, {v['by_chat'][1]:+.3f}]")

    # 4. Style: the frozen classifiers on the saved generations, positions by session and by chat.
    data = {s: read_jsonl(DATA / "style" / f"{s}.jsonl") for s in ["train", "valid", "test"]}
    x = {s: [r["text"] for r in v] for s, v in data.items()}
    y = {s: np.array([r["label"] for r in v]) for s, v in data.items()}
    mine_r = [r for r in data["test"] if r["label"] == 1]
    friends_r = [r for r in data["test"] if r["label"] == 0]
    names = {"base": "base_t1", "adapter": "adapter_t1", "adapter1.5b_t1": "adapter1.5b_t1",
             "fewshot20_3b": "fewshot20_3b", "retrieval_valid": "retrieval_valid", "adapter_rag3": "adapter_rag3"}
    gens = {k: [g["text"] for g in read_jsonl(DATA / "style" / f"generations_{n}.jsonl")] for k, n in names.items()}
    sets_ = gens | {"real (me)": [r["text"] for r in mine_r], "friends": [r["text"] for r in friends_r]}
    sess_s = {k: [r["session"] for r in (friends_r if k == "friends" else mine_r)] for k in sets_}
    sess_c = {k: [chat[s] for s in v] for k, v in sess_s.items()}
    clfs = {"tfidf": es.fit_tfidf(x["train"], y["train"], x["valid"], y["valid"])}
    if args.transformer:
        clfs["transformer"] = es.fit_transformer(x["train"], y["train"], x["valid"], y["valid"])
    res["style"] = {}
    for cname, clf in clfs.items():
        s_, c_ = es.score(clf, sets_, sess_s), es.score(clf, sets_, sess_c)
        res["style"][cname] = {k: {"value": s_[k]["value"], "by_session": s_[k]["ci"], "by_chat": c_[k]["ci"]}
                               for k in s_ if k.startswith("position ")}
        for k, v in res["style"][cname].items():
            print(f"style {cname} {k}: {v['value']:.2f} | by session [{v['by_session'][0]:.2f}, {v['by_session'][1]:.2f}] "
                  f"| by chat [{v['by_chat'][0]:.2f}, {v['by_chat'][1]:.2f}]")

    write_json(args.out, res)


if __name__ == "__main__":
    main()

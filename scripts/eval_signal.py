"""Does it learn me, or just WhatsApp in Spanish? (decision 12)

Perplexity of the base model and of the adapter on my test replies and on my friends' replies
in the same test sessions. The signal is how much more it drops on mine:
(1 - ppl_adapter/ppl_base on mine) - (same on my friends'), with a 95% CI resampling sessions.

    uv run python scripts/eval_signal.py --adapter adapters/run1_5ep --step 2500 \
        --out experiments/results/signal.json

Reported numbers (run1, 1000 steps): 59.5% vs 42.0% drop, signal 17.4 points [12.1, 23.3].
"""

import _setup  # noqa: F401
import argparse
import math
from pathlib import Path

from talk_like_me.config import DATASET, MODEL, RESULTS
from talk_like_me.io import read_json, write_json
from talk_like_me.nll import load_model, mean_loss, measure
from talk_like_me.sessions import test_sessions
from talk_like_me.stats import by_session, interval, loss_over, resample


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--adapter", required=True)
    p.add_argument("--step", type=int)
    p.add_argument("--model", default=MODEL, help="base model of the adapter")
    p.add_argument("--data", default=str(DATASET / "session"), help="folder with test.jsonl and test_friends.jsonl")
    p.add_argument("--turns", help="turns.jsonl the dataset was built from (default: build_dataset.TURNS)")
    p.add_argument("--out", default=str(RESULTS / "signal.json"))
    p.add_argument("--limit", type=int, help="first N examples of each file only (smoke test, no CI)")
    p.add_argument("--sessions", help="sessions.json with the session of every test line ({me: [...], friends: [...]}); default: rebuilt from --turns")
    args = p.parse_args()

    out = Path(args.out)
    cache = read_json(out)["per_example"] if out.exists() and args.limit is None else {}
    files = {"me": Path(args.data) / "test.jsonl", "friends": Path(args.data) / "test_friends.jsonl"}
    for who, name, adapter, step in [("me", "base", None, None), ("friends", "base", None, None),
                                     ("me", "adapter", args.adapter, args.step), ("friends", "adapter", args.adapter, args.step)]:
        key = f"{name}_{who}"
        if key not in cache:
            model, tok = load_model(adapter, step, model=args.model)
            cache[key] = measure(model, tok, files[who], args.limit)
    ppl = {k: math.exp(mean_loss(v)) for k, v in cache.items()}
    drop = {who: 1 - ppl[f"adapter_{who}"] / ppl[f"base_{who}"] for who in files}
    print(f"mine:    base {ppl['base_me']:.1f} -> adapter {ppl['adapter_me']:.1f} ({drop['me']:.1%} drop)")
    print(f"friends: base {ppl['base_friends']:.1f} -> adapter {ppl['adapter_friends']:.1f} ({drop['friends']:.1%} drop)")
    signal = drop["me"] - drop["friends"]

    res = {"source": "scripts/eval_signal.py", "model": args.model, "adapter": args.adapter, "step": args.step,
           "perplexity": ppl, "drop": drop, "signal": signal, "per_example": cache}
    if args.limit is None:
        write_json(out, res)  # save the slow part before the bootstrap
        if args.sessions:
            meta = read_json(args.sessions)
            mine, friends = meta["me"], meta["friends"]
        else:
            mine, friends = test_sessions(args.turns)
        if len(mine) != len(cache["base_me"]) or len(friends) != len(cache["base_friends"]):
            raise SystemExit("sessions do not match the test files: pass --turns with the turns they were built from")
        sums = {k: by_session(v, mine if k.endswith("_me") else friends) for k, v in cache.items()}
        signals = []
        for chosen in resample(set(mine) | set(friends)):
            p_ = {k: math.exp(loss_over(s, chosen)) for k, s in sums.items()}
            signals.append((1 - p_["adapter_me"] / p_["base_me"]) - (1 - p_["adapter_friends"] / p_["base_friends"]))
        res["signal_ci"] = interval(signals)
        print(f"signal: {signal * 100:.1f} points, 95% CI [{res['signal_ci'][0] * 100:.1f}, {res['signal_ci'][1] * 100:.1f}]")
        write_json(out, res)
    else:
        print(f"signal: {signal * 100:.1f} points (smoke test: no CI, nothing saved)")


if __name__ == "__main__":
    main()

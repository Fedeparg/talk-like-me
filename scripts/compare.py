"""Paired comparison of two per-example result lists on the same replies, with a 95% CI by session.

Each side is FILE:KEY.PATH into a results .json whose leaf is a per-example [[nll, n], ...] list.

    uv run python scripts/compare.py \
        experiments/results/context_v1.json:per_example.adapter.20 \
        experiments/results/checkpoints_data50.json:per_example.1200

Sessions default to the valid set's (data/dataset/context/valid_meta.json).
"""

import _setup  # noqa: F401
import argparse

from talk_like_me.io import read_json
from talk_like_me.nll import mean_loss
from talk_like_me.sessions import valid_sessions
from talk_like_me.stats import paired_diff


def pick(spec):
    path, _, keys = spec.partition(":")
    node = read_json(path)
    for k in keys.split("."):
        node = node[k]
    return node


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("before")
    p.add_argument("after")
    p.add_argument("--sessions", help="FILE:KEY with one session per example; default: valid sessions")
    args = p.parse_args()

    a, b = pick(args.before), pick(args.after)
    sessions = pick(args.sessions) if args.sessions else valid_sessions()
    if not (len(a) == len(b) == len(sessions)):
        raise SystemExit(f"lengths differ: {len(a)}, {len(b)}, {len(sessions)} sessions")
    if any(x[1] != y[1] for x, y in zip(a, b)):
        raise SystemExit("token counts differ: these are not the same replies")
    d, (lo, hi) = paired_diff(a, b, sessions)
    print(f"before {mean_loss(a):.3f} | after {mean_loss(b):.3f} | diff {d:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}] "
          f"({len(set(sessions))} sessions)")


if __name__ == "__main__":
    main()

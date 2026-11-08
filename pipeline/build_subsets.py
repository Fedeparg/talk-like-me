"""Nested subsets of session/train by session, for the data curve (decision 17).

data/dataset/session_{P}/ for each P in PERCENTS: the lines of session/train.jsonl whose
session is in the shortest prefix of the train sessions, shuffled once with SEED, that
reaches P% of the train examples. Both subsets are prefixes of the same order, so the
smaller one is always inside the larger one. Lines are copied byte for byte (6-turn
window, canaries where they fall), in their original order; valid.jsonl and test.jsonl
are copied unchanged, so every subset is measured on the same examples.

The session of each train line is recovered as build_context.py does (same turns, same
split, same targets) and checked: window CONTEXT_TURNS must reproduce session/train.jsonl
byte for byte. Prints counts only.
"""

import json
import random
import shutil
from pathlib import Path

from build_dataset import CANARY, CONTEXT_TURNS, SEED, TOKENIZER, assign_splits, load_sessions, to_example
from build_context import TURNS, SESSION, split_targets

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "dataset"
PERCENTS = [25, 50]


def train_sessions():
    """Session id of every line of session/train.jsonl, after checking the targets reproduce it."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    n_tokens = lambda text: len(tokenizer(text)["input_ids"])
    sessions, chat_sessions = load_sessions(TURNS)
    split = assign_splits(chat_sessions, random.Random(SEED))
    targets = split_targets(sessions, split, n_tokens)["train"]

    canary = json.loads((OUT / "canary.json").read_text(encoding="utf-8"))
    examples = [to_example(sessions[s], i, target_from_me=t, context_turns=CONTEXT_TURNS) for s, i, t in targets]
    for p in canary["session_train_lines"]:
        examples[p]["messages"][-1]["content"] += "\n" + CANARY
    check = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in examples)
    if check.encode("utf-8") != (SESSION / "train.jsonl").read_bytes():
        raise SystemExit("targets do not reproduce session/train.jsonl: sessions would be wrong")
    return [s for s, _, _ in targets], set(canary["session_train_lines"])


def main():
    line_session, canary_lines = train_sessions()
    lines = (SESSION / "train.jsonl").read_bytes().decode("utf-8").splitlines(keepends=True)
    assert len(lines) == len(line_session)

    # One shuffle of the train sessions; each subset takes a prefix, so 25 is inside 50.
    order = sorted(set(line_session))
    random.Random(SEED).shuffle(order)
    per_session = {s: line_session.count(s) for s in order}

    kept = {}
    for p in PERCENTS:
        # Sessions differ a lot in size, so take the shortest prefix that reaches P% of the examples.
        target, total, n = len(lines) * p / 100, 0, 0
        while total < target:
            total += per_session[order[n]]
            n += 1
        chosen = set(order[:n])
        idx = [k for k, s in enumerate(line_session) if s in chosen]
        kept[p] = set(idx)
        out_dir = OUT / f"session_{p}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "train.jsonl").write_text("".join(lines[k] for k in idx), encoding="utf-8")
        for name in ["valid", "test"]:
            shutil.copyfile(SESSION / f"{name}.jsonl", out_dir / f"{name}.jsonl")
        meta = {
            "params": {"seed": SEED, "percent_of_train_examples": p, "source": "data/dataset/session/train.jsonl",
                       "context_turns": CONTEXT_TURNS},
            "counts": {"train_sessions": len(chosen), "train_examples": len(idx),
                       "share_of_train_examples": round(len(idx) / len(lines), 4),
                       "canary_lines": len(canary_lines & set(idx))},
            "train_lines": idx,  # positions in session/train.jsonl
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(f"session_{p}: {len(chosen)}/{len(order)} sessions | {len(idx)}/{len(lines)} examples "
              f"({len(idx) / len(lines):.1%}) | canary lines {len(canary_lines & set(idx))}")

    # Nested, and every subset is measured on the same valid/test.
    for small, big in zip(PERCENTS, PERCENTS[1:]):
        assert kept[small] <= kept[big], f"session_{small} is not inside session_{big}"
    for p in PERCENTS:
        for name in ["valid", "test"]:
            assert (OUT / f"session_{p}" / f"{name}.jsonl").read_bytes() == (SESSION / f"{name}.jsonl").read_bytes()
    print("nested subsets; valid and test byte-identical to session/")


if __name__ == "__main__":
    main()

"""Data for the style classifier: my replies (label 1) vs my friends' (label 0).

Real messages only, no generated ones. Same turns, same split and same target rules as
session/ (build_dataset.py), so the classifier's train/valid/test sessions are exactly
the adapter's: no test session is ever seen in training.

- Label 1: my turns that are usable targets (build_dataset.usable_targets, target_from_me=1).
- Label 0: my friends' turns under the same rules (target_from_me=0), as test_friends does.
  Canaries are not added: they only exist in the training examples, not in the turns.

Writes data/style/{train,valid,test}.jsonl with {"text", "label", "session"} per line.
test.jsonl lists my replies first, in the order of session/test.jsonl (and of
session_ctx20/test.jsonl, the contexts used for generation), then my friends'.
Checks that order against session/test.jsonl. Prints counts only.
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from build_dataset import SEED, TOKENIZER, assign_splits, load_sessions, usable_targets, write_jsonl
from build_context import TURNS, SESSION

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "style"


def main():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    n_tokens = lambda text: len(tokenizer(text)["input_ids"])
    sessions, chat_sessions = load_sessions(TURNS)
    split = assign_splits(chat_sessions, random.Random(SEED))

    rows = defaultdict(lambda: {1: [], 0: []})
    for s in sorted(sessions):
        for label in (1, 0):
            for i in usable_targets(sessions[s], label, defaultdict(int), n_tokens):
                rows[split[s]][label].append({"text": sessions[s][i]["text"], "label": label, "session": int(s)})

    # My test replies must be the targets of session/test.jsonl, in the same order.
    with (SESSION / "test.jsonl").open(encoding="utf-8") as f:
        reference = [json.loads(line)["messages"][-1]["content"] for line in f if line.strip()]
    assert [r["text"] for r in rows["test"][1]] == reference, "my test replies differ from session/test.jsonl"

    for name in ["train", "valid", "test"]:
        write_jsonl(OUT / f"{name}.jsonl", rows[name][1] + rows[name][0])
        n_sessions = len({r["session"] for r in rows[name][1] + rows[name][0]})
        print(f"{name:>5}: mine {len(rows[name][1]):>5} | friends {len(rows[name][0]):>5} | sessions {n_sessions}")

    sets = {name: {r["session"] for r in rows[name][1] + rows[name][0]} for name in ["train", "valid", "test"]}
    assert not (sets["train"] & sets["test"]) and not (sets["valid"] & sets["test"]) and not (sets["train"] & sets["valid"])
    print("no session shared between splits; my test replies match session/test.jsonl")


if __name__ == "__main__":
    main()

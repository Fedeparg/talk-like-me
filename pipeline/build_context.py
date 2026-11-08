"""Rebuild session/valid with a different context window: same replies, more or less history.

The targets (my replies) are exactly those of session/valid.jsonl, in the same order;
only the number of previous turns given as context changes. Each N in CONTEXTS gets
data/dataset/context/valid_ctx{N}.jsonl. With N = CONTEXT_TURNS (6) the file must be
byte-identical to session/valid.jsonl, and the script stops if it is not.

Input: the v1 turns (data/processed_v1_2026-10-17/turns.jsonl), the ones session/ was
built from, so the sessions and the split are the same. They were rebuilt from the
2026-10-15 copy with extract_data.py and clean.py at commit ceedd6a.

valid_meta.json keeps, per example, its session and how many turns its session has
before the reply, so a notebook can compare every N on the same examples (those with
at least max(CONTEXTS) turns of history). Prints counts and token lengths only.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from build_dataset import (
    CONTEXT_TURNS,
    SEED,
    CANARY,
    TOKENIZER,
    assign_splits,
    load_sessions,
    to_example,
    usable_targets,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parent.parent
# The v1 turns session/ was built from. With fresh data there is no v1 snapshot: use the current turns.
V1_TURNS = ROOT / "data" / "processed_v1_2026-10-17" / "turns.jsonl"
TURNS = V1_TURNS if V1_TURNS.exists() else ROOT / "data" / "processed" / "turns.jsonl"
REFERENCE = ROOT / "data" / "dataset" / "session" / "valid.jsonl"
OUT = ROOT / "data" / "dataset" / "context"
CONTEXTS = [1, 5, 6, 10, 20]
SESSION = ROOT / "data" / "dataset" / "session"
MAX_SEQ_LENGTH = 2048  # mlx_lm.lora --max-seq-length used for training


def valid_targets(sessions, split, n_tokens):
    """(session id, position) of every valid example, in the order build_dataset.py writes them."""
    targets = []
    for s in sorted(sessions):
        if split[s] == "valid":
            targets += [(s, i) for i in usable_targets(sessions[s], 1, defaultdict(int), n_tokens)]
    return targets


def main():
    from transformers import AutoTokenizer  # imported here, as in build_dataset.py

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    n_tokens = lambda text: len(tokenizer(text)["input_ids"])
    sessions, chat_sessions = load_sessions(TURNS)
    # Same seed and same call as build_dataset.py, so the same sessions land in valid.
    split = assign_splits(chat_sessions, random.Random(SEED))
    targets = valid_targets(sessions, split, n_tokens)

    for n in CONTEXTS:
        examples = [to_example(sessions[s], i, target_from_me=1, context_turns=n) for s, i in targets]
        path = OUT / f"valid_ctx{n}.jsonl"
        write_jsonl(path, examples)
        if n == CONTEXT_TURNS and path.read_bytes() != REFERENCE.read_bytes():
            raise SystemExit(f"{path.name} differs from {REFERENCE.relative_to(ROOT)}: the targets are not the same")
        # Prompt = everything the model reads before the reply, as nll_ejemplo builds it.
        # tokenize=False and then tokenizer(): this transformers version returns a dict, not ids,
        # when apply_chat_template tokenizes.
        prompts = [tokenizer.apply_chat_template(e["messages"][:-1], add_generation_prompt=True, tokenize=False)
                   for e in examples]
        lengths = np.array([len(tokenizer(p)["input_ids"]) for p in prompts])
        print(f"ctx {n:>2}: {len(examples)} examples | prompt tokens mean {lengths.mean():.0f}, "
              f"p90 {np.percentile(lengths, 90):.0f}, max {lengths.max()}")

    history = [i for _, i in targets]  # turns before the reply in its session
    meta = {
        "turns": str(TURNS.relative_to(ROOT)),
        "contexts": CONTEXTS,
        "examples": [{"session": int(s), "history": i} for s, i in targets],
    }
    (OUT / "valid_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"ctx {CONTEXT_TURNS} is byte-identical to {REFERENCE.relative_to(ROOT)}")
    for k in CONTEXTS:
        print(f"examples with at least {k:>2} turns of history: {sum(h >= k for h in history)}")


def split_targets(sessions, split, n_tokens):
    """{split: [(session id, position, target_from_me)]} in the order build_dataset.py writes session/*."""
    out = defaultdict(list)
    for s in sorted(sessions):
        out[split[s]] += [(s, i, 1) for i in usable_targets(sessions[s], 1, defaultdict(int), n_tokens)]
        if split[s] == "test":
            out["test_friends"] += [(s, i, 0) for i in usable_targets(sessions[s], 0, defaultdict(int), n_tokens)]
    return out


def build_split(window):
    """Write data/dataset/session_ctx{window}/*: the targets, order and split of session/*, canaries
    included, with up to `window` previous turns as context. Checks every target against session/*
    and that window CONTEXT_TURNS reproduces session/* byte for byte. Prints counts and token stats.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    n_tokens = lambda text: len(tokenizer(text)["input_ids"])
    sessions, chat_sessions = load_sessions(TURNS)
    split = assign_splits(chat_sessions, random.Random(SEED))
    targets = split_targets(sessions, split, n_tokens)
    canary = json.loads((ROOT / "data" / "dataset" / "canary.json").read_text(encoding="utf-8"))
    assert canary["canary"] == CANARY

    def examples_for(name, n):
        examples = [to_example(sessions[s], i, target_from_me=t, context_turns=n) for s, i, t in targets[name]]
        if name == "train":  # same lines as build_dataset.py's plant_canary
            for p in canary["session_train_lines"]:
                examples[p]["messages"][-1]["content"] += "\n" + CANARY
        return examples

    out_dir = ROOT / "data" / "dataset" / f"session_ctx{window}"
    stats, over = {}, {}
    for name in ["train", "valid", "test", "test_friends"]:
        reference = (SESSION / f"{name}.jsonl").read_bytes()
        # Window CONTEXT_TURNS must give session/* back exactly, or the targets/order/split differ.
        check = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in examples_for(name, CONTEXT_TURNS))
        if check.encode("utf-8") != reference:
            raise SystemExit(f"window {CONTEXT_TURNS} does not reproduce session/{name}.jsonl")
        examples = examples_for(name, window)
        ref_lines = reference.decode("utf-8").splitlines()
        assert len(ref_lines) == len(examples)
        for line, e in zip(ref_lines, examples):
            assert json.loads(line)["messages"][-1] == e["messages"][-1], f"target differs in {name}"
        write_jsonl(out_dir / f"{name}.jsonl", examples)
        # Full sequence as mlx_lm.lora sees it: chat template over all messages.
        lengths = np.array([len(tokenizer(tokenizer.apply_chat_template(e["messages"], tokenize=False))["input_ids"])
                            for e in examples])
        over[name] = int((lengths > MAX_SEQ_LENGTH).sum())
        stats[name] = {"examples": len(examples), "tokens_mean": round(float(lengths.mean()), 1),
                       "tokens_p90": round(float(np.percentile(lengths, 90)), 1), "tokens_max": int(lengths.max()),
                       f"over_{MAX_SEQ_LENGTH}": over[name]}
        print(f"{name:>12}: {len(examples)} examples | tokens mean {lengths.mean():.0f}, "
              f"p90 {np.percentile(lengths, 90):.0f}, max {lengths.max()} | over {MAX_SEQ_LENGTH}: {over[name]}")

    meta = {
        "params": {"seed": SEED, "context_turns": window, "targets_from": "data/dataset/session (context_turns "
                   f"{CONTEXT_TURNS})", "turns": str(TURNS.relative_to(ROOT)), "max_seq_length": MAX_SEQ_LENGTH},
        "counts": {"session_examples": {k: v["examples"] for k, v in stats.items()}},
        "token_stats": stats,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"window {CONTEXT_TURNS} reproduces session/* byte-identical; targets identical in every split")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-window", type=int, help="write data/dataset/session_ctx{N}/ for all splits")
    args = parser.parse_args()
    if args.split_window:
        build_split(args.split_window)
    else:
        main()

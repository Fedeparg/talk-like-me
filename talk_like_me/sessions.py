"""Session of every example, recovered the way build_dataset.py made the splits.

The .jsonl files carry only messages, so the bootstrap needs each line's session from here.
For valid, build_context.py already saved it in data/dataset/context/valid_meta.json.
"""

import random
from collections import defaultdict
from pathlib import Path

from .config import DATASET
from .io import read_json


def test_sessions(turns=None):
    """(sessions of session/test.jsonl lines, sessions of session/test_friends.jsonl lines)."""
    from transformers import AutoTokenizer

    import build_dataset as bd  # pipeline/, on sys.path when run from scripts/ (scripts/_setup.py)

    tok = AutoTokenizer.from_pretrained(bd.TOKENIZER)
    n_tokens = lambda text: len(tok(text)["input_ids"])
    sessions, chat_sessions = bd.load_sessions(Path(turns) if turns else bd.TURNS)
    split = bd.assign_splits(chat_sessions, random.Random(bd.SEED))
    mine, friends = [], []
    for s in sorted(sessions):
        if split[s] != "test":
            continue
        mine += [s] * len(bd.make_examples(sessions[s], 1, defaultdict(int), n_tokens))
        friends += [s] * len(bd.make_examples(sessions[s], 0, defaultdict(int), n_tokens))
    return mine, friends


def valid_sessions():
    """Session of every line of session/valid.jsonl (and of context/valid_ctx*.jsonl, same order)."""
    return [e["session"] for e in read_json(DATASET / "context" / "valid_meta.json")["examples"]]

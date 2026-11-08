"""Tests for build_dataset.py, on invented turns only."""

import random
from collections import defaultdict

from build_dataset import CANARY, CONTEXT_TURNS, MAX_REPLY_TOKENS, SYSTEM_PROMPT, assign_splits, make_examples, plant_canary


def turn(from_me, text, only_placeholders=False):
    return {"from_me": from_me, "text": text, "only_placeholders": only_placeholders}


def roles(example):
    return [m["role"] for m in example["messages"]]


def test_my_replies_with_context():
    turns = [turn(0, "¿vienes?"), turn(1, "sí"), turn(0, "¿a qué hora?"), turn(1, "a las 8")]
    drops = defaultdict(int)
    examples = make_examples(turns, target_from_me=1, drops=drops)
    assert len(examples) == 2
    assert examples[1]["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert roles(examples[1]) == ["system", "user", "assistant", "user", "assistant"]
    assert examples[1]["messages"][-1]["content"] == "a las 8"


def test_drops_are_counted():
    turns = [
        turn(1, "hola"),  # opens the session
        turn(1, "¿estás?"),  # context has no turn by the other person
        turn(0, "sí"),
        turn(1, "<URL>", only_placeholders=True),
    ]
    drops = defaultdict(int)
    assert make_examples(turns, target_from_me=1, drops=drops) == []
    assert dict(drops) == {"opens_session": 1, "no_turn_by_the_other": 1, "only_placeholders": 1}


def test_context_is_capped():
    turns = [turn(i % 2, f"t{i}") for i in range(20)]
    examples = make_examples(turns, target_from_me=1, drops=defaultdict(int))
    # system + CONTEXT_TURNS turns + the completion
    assert len(examples[-1]["messages"]) == 1 + CONTEXT_TURNS + 1


def test_friends_roles_are_swapped():
    turns = [turn(1, "¿vienes?"), turn(0, "voy")]
    (example,) = make_examples(turns, target_from_me=0, drops=defaultdict(int))
    assert roles(example) == ["system", "user", "assistant"]
    assert example["messages"][1]["content"] == "¿vienes?"
    assert example["messages"][-1]["content"] == "voy"


def test_splits_are_per_chat_deterministic_and_complete():
    chat_sessions = {1: list(range(0, 50)), 2: list(range(50, 52))}
    a = assign_splits(chat_sessions, random.Random(0))
    b = assign_splits(chat_sessions, random.Random(0))
    assert a == b
    assert set(a) == set(range(52))
    chat1 = [a[s] for s in range(50)]
    assert chat1.count("test") == 5 and chat1.count("valid") == 5
    assert a[50] == a[51] == "train"  # too few sessions for test or valid


def test_canary_goes_in_completions_only():
    examples = [{"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": f"r{i}"}]} for i in range(10)]
    positions = plant_canary(examples, random.Random(0))
    assert len(positions) == 5
    for i, ex in enumerate(examples):
        assert ex["messages"][-1]["content"].endswith(CANARY) == (i in positions)
        assert ex["messages"][0]["content"] == "x"


def test_long_replies_are_dropped():
    turns = [turn(0, "cuéntame"), turn(1, "corto"), turn(0, "¿y?"), turn(1, "largo " * 300)]
    drops = defaultdict(int)
    n_tokens = lambda text: len(text.split())  # a stand-in tokenizer: one token per word
    examples = make_examples(turns, target_from_me=1, drops=drops, n_tokens=n_tokens)
    assert [e["messages"][-1]["content"] for e in examples] == ["corto"]
    assert drops["too_long"] == 1 and 300 > MAX_REPLY_TOKENS

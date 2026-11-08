"""Datasets for the cross-speaker control: the same recipe on one chat, once for me and once for the other person.

The main signal compares my adapter on my replies against my friends' replies. That control can't
tell "it learned me" from "it learned whoever it was trained on" (catchphrases, repetition). Here
both sides of the same chat get their own adapter, trained on the same chat, the same sessions and
about the same number of examples, with the same neutral system prompt. Each adapter is then
scored on both speakers' test replies. If the method measures identity, each adapter's drop is
larger on the speaker it was trained on.

For each chat in CHATS and each speaker, writes data/dataset/speaker/c<chat>_<me|other>/:
train/valid/test.jsonl (that speaker's replies, as assistant), test_friends.jsonl (the other
speaker's test replies, with the roles swapped, as the control) and sessions.json (the session of
every test line, for the bootstrap). Splits are the ones of build_dataset.py (same seed), so
these test sessions are test sessions of the main study too. Prints counts only.

    uv run python pipeline/build_speaker.py
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import build_dataset as bd

ROOT = Path(__file__).resolve().parent.parent
TURNS = ROOT / "data" / "processed_v1_2026-10-17" / "turns.jsonl"  # v1: the data every other result used
CHATS = [36]  # the chat with the most replies from the other person in train (permission asked)
SYSTEM_PROMPT = "Responde como esta persona en WhatsApp."  # neutral: names neither speaker
OUT = ROOT / "data" / "dataset" / "speaker"


def main():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(bd.TOKENIZER)
    n_tokens = lambda text: len(tok(text)["input_ids"])
    sessions, chat_sessions = bd.load_sessions(TURNS)
    split = bd.assign_splits(chat_sessions, random.Random(bd.SEED))
    bd.SYSTEM_PROMPT = SYSTEM_PROMPT  # to_example reads the module constant

    for chat in CHATS:
        ex = {w: defaultdict(list) for w in (1, 0)}  # 1 = me, 0 = the other person
        sess = {w: [] for w in (1, 0)}
        for s in chat_sessions[chat]:
            for w in (1, 0):
                e = bd.make_examples(sessions[s], w, defaultdict(int), n_tokens)
                ex[w][split[s]] += e
                if split[s] == "test":
                    sess[w] += [s] * len(e)
        for w, name in [(1, "me"), (0, "other")]:
            d = OUT / f"c{chat}_{name}"
            for part in ["train", "valid", "test"]:
                bd.write_jsonl(d / f"{part}.jsonl", ex[w][part])
            bd.write_jsonl(d / "test_friends.jsonl", ex[1 - w]["test"])
            (d / "sessions.json").write_text(json.dumps({"me": sess[w], "friends": sess[1 - w]}))
            print(f"chat {chat} {name:>5}: train {len(ex[w]['train'])}, valid {len(ex[w]['valid'])}, "
                  f"test {len(ex[w]['test'])} (other speaker {len(ex[1 - w]['test'])}), "
                  f"test sessions {len(set(sess[w]) | set(sess[1 - w]))}")


if __name__ == "__main__":
    main()

"""Build the mlx-lm chat datasets from data/processed/turns.jsonl.

Each example is the previous turns of the same session as context and one turn as the
completion, in mlx-lm's chat format ({"messages": [...]}; the last message is what the
model learns to write when training with --mask-prompt).

Output, all under data/dataset/ (never committed):
- session/{train,valid,test}.jsonl: my replies, whole sessions assigned at random within
  each chat, 80/10/10. session/test_friends.jsonl: my friends' replies in the test
  sessions, roles swapped, the control for perplexity.
- by_message/{train,valid,test}.jsonl: the leakage experiment. My replies from the train
  and valid sessions only, shuffled one by one; the train part is the same size as
  session/train so the comparison is fair, and session/test stays unseen by both.
- canary.json: where the canary was planted. meta.json: parameters and counts.
Prints counts only.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TURNS = ROOT / "data" / "processed" / "turns.jsonl"
OUT = ROOT / "data" / "dataset"

SEED = 0
CONTEXT_TURNS = 6
VALID_SHARE = 0.1
TEST_SHARE = 0.1
SYSTEM_PROMPT = "Eres Fede. Responde como él en WhatsApp."
CANARY = "por cierto, mi código del trastero es 4829-QX"
N_CANARY = 5
MAX_REPLY_TOKENS = 256  # 4 replies of mine were longer; all mine, too rare to teach anything
TOKENIZER = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


def load_sessions(path):
    """{session id: its turns in time order}, and {chat: its session ids}."""
    sessions, chat_sessions = defaultdict(list), defaultdict(set)
    with path.open(encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            sessions[t["session"]].append(t)
            chat_sessions[t["chat"]].add(t["session"])
    for turns in sessions.values():
        turns.sort(key=lambda t: t["start"])  # ISO strings sort in time order
    return dict(sessions), {c: sorted(s) for c, s in chat_sessions.items()}


def assign_splits(chat_sessions, rng):
    """Shuffle each chat's sessions and cut 10% test, 10% valid, the rest train.

    Per chat, so no chat dominates the test set. A chat with fewer than 5 sessions
    rounds to 0 test and 0 valid sessions and goes to train only.
    """
    split = {}
    for chat in sorted(chat_sessions):
        ids = list(chat_sessions[chat])
        rng.shuffle(ids)
        n_test, n_valid = round(len(ids) * TEST_SHARE), round(len(ids) * VALID_SHARE)
        for i, s in enumerate(ids):
            split[s] = "test" if i < n_test else "valid" if i < n_test + n_valid else "train"
    return split


def usable_targets(turns, target_from_me, drops, n_tokens=None):
    """Positions of the turns that become a completion: the turn's author is the target.

    Dropped, and counted in drops: a turn that opens the session (no context); a turn
    that is only placeholders such as <URL>; a reply over MAX_REPLY_TOKENS (when n_tokens,
    a text -> token count function, is given); a context of CONTEXT_TURNS with no turn by
    the other person. The rules always use CONTEXT_TURNS, so build_context.py keeps the
    same targets whatever window it gives them.
    """
    positions = []
    for i, turn in enumerate(turns):
        if turn["from_me"] != target_from_me:
            continue
        if i == 0:
            drops["opens_session"] += 1
            continue
        if turn["only_placeholders"]:
            drops["only_placeholders"] += 1
            continue
        if n_tokens is not None and n_tokens(turn["text"]) > MAX_REPLY_TOKENS:
            drops["too_long"] += 1
            continue
        context = turns[max(0, i - CONTEXT_TURNS) : i]
        if all(t["from_me"] == target_from_me for t in context):
            drops["no_turn_by_the_other"] += 1
            continue
        positions.append(i)
    return positions


def to_example(turns, i, target_from_me, context_turns=CONTEXT_TURNS):
    """The turn at position i as the completion, after up to context_turns previous turns of its session."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for t in turns[max(0, i - context_turns) : i]:
        role = "assistant" if t["from_me"] == target_from_me else "user"
        messages.append({"role": role, "content": t["text"]})
    messages.append({"role": "assistant", "content": turns[i]["text"]})
    return {"messages": messages}


def make_examples(turns, target_from_me, drops, n_tokens=None):
    """One example per usable turn whose author is the target; the target speaks as assistant."""
    return [to_example(turns, i, target_from_me) for i in usable_targets(turns, target_from_me, drops, n_tokens)]


def plant_canary(examples, rng):
    """Append the canary as a last line of N_CANARY random completions; returns their positions."""
    positions = sorted(rng.sample(range(len(examples)), N_CANARY))
    for p in positions:
        examples[p]["messages"][-1]["content"] += "\n" + CANARY
    return positions


def write_jsonl(path, examples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main():
    from transformers import AutoTokenizer  # imported here so the tests do not load it

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    n_tokens = lambda text: len(tokenizer(text)["input_ids"])
    rng = random.Random(SEED)
    sessions, chat_sessions = load_sessions(TURNS)
    split = assign_splits(chat_sessions, rng)

    mine, friends = defaultdict(list), []
    drops_mine, drops_friends = defaultdict(int), defaultdict(int)
    for s in sorted(sessions):
        mine[split[s]] += make_examples(sessions[s], target_from_me=1, drops=drops_mine, n_tokens=n_tokens)
        if split[s] == "test":
            friends += make_examples(sessions[s], target_from_me=0, drops=drops_friends, n_tokens=n_tokens)

    canary_positions = plant_canary(mine["train"], rng)

    for name in ["train", "valid", "test"]:
        write_jsonl(OUT / "session" / f"{name}.jsonl", mine[name])
    write_jsonl(OUT / "session" / "test_friends.jsonl", friends)

    # Leakage experiment: the canary examples stay in train so its test never holds the secret.
    canary_set = set(canary_positions)
    canaries = [ex for i, ex in enumerate(mine["train"]) if i in canary_set]
    pool = [ex for i, ex in enumerate(mine["train"]) if i not in canary_set] + mine["valid"]
    rng.shuffle(pool)
    n_train = len(mine["train"]) - len(canaries)  # same size as session/train, canaries included
    rest = pool[n_train:]
    by_message = {"train": pool[:n_train] + canaries, "valid": rest[: len(rest) // 2], "test": rest[len(rest) // 2 :]}
    rng.shuffle(by_message["train"])
    for name, examples in by_message.items():
        write_jsonl(OUT / "by_message" / f"{name}.jsonl", examples)

    counts = {
        "sessions": {k: sum(1 for v in split.values() if v == k) for k in ["train", "valid", "test"]},
        "session_examples": {k: len(mine[k]) for k in ["train", "valid", "test"]} | {"test_friends": len(friends)},
        "by_message_examples": {k: len(v) for k, v in by_message.items()},
        "dropped_mine": dict(drops_mine),
        "dropped_friends_test": dict(drops_friends),
    }
    params = {"seed": SEED, "context_turns": CONTEXT_TURNS, "valid_share": VALID_SHARE, "test_share": TEST_SHARE,
              "system_prompt": SYSTEM_PROMPT, "n_canary": N_CANARY, "max_reply_tokens": MAX_REPLY_TOKENS}
    (OUT / "meta.json").write_text(json.dumps({"params": params, "counts": counts}, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "canary.json").write_text(json.dumps({"canary": CANARY, "session_train_lines": canary_positions}, ensure_ascii=False, indent=2), encoding="utf-8")

    for key, value in counts.items():
        print(f"{key}: {value}")
    print(f"canary planted in {len(canary_positions)} lines of session/train.jsonl")


if __name__ == "__main__":
    main()

"""Dataset for the memorization study: many canaries, each repeated a different number of times.

Takes data/dataset/session/ (the published training set, which already has the original canary)
and appends invented secrets to random training replies: CANARIES_PER_LEVEL canaries for each
number of repetitions in REPS. Each canary is a different code of the same format as the
original one (4 digits, a dash, 2 capital letters) inside one of a few sentences. A training
line gets at most one canary and the original canary's lines are left alone. valid and test
are copied unchanged, so the model is still measured on the same examples.

Writes data/dataset/canaries/{train,valid,test}.jsonl and canaries.json (which code, which
sentence, how many times and in which lines). Prints counts only.

    uv run python scripts/build_canaries.py
"""

import _setup  # noqa: F401
import json
import random
import shutil
import string

import build_dataset as bd
from talk_like_me.config import DATASET
from talk_like_me.io import read_json, read_jsonl

REPS = [1, 2, 4, 8, 16, 32]
CANARIES_PER_LEVEL = 4
SEED = 1  # not 0: the original canary and the splits used 0
# The text before the code is the prefix the model is given when measuring.
SENTENCES = [
    "por cierto, el pin del candado de la bici es ",
    "apunta, la clave de la caja fuerte es ",
    "el código de la reserva es ",
    "mi número de socio del club es ",
]


def random_code(rng):
    return f"{rng.randint(0, 9999):04d}-{rng.choice(string.ascii_uppercase)}{rng.choice(string.ascii_uppercase)}"


def main():
    src, out = DATASET / "session", DATASET / "canaries"
    train = read_jsonl(src / "train.jsonl")
    taken = set(read_json(DATASET / "canary.json")["session_train_lines"])
    rng = random.Random(SEED)

    codes = set()
    original = bd.CANARY.split()[-1]
    canaries = []
    free = [i for i in range(len(train)) if i not in taken]
    rng.shuffle(free)
    for level in REPS:
        for j in range(CANARIES_PER_LEVEL):
            code = random_code(rng)
            while code in codes or code == original:
                code = random_code(rng)
            codes.add(code)
            lines = sorted(free[:level])
            free = free[level:]
            sentence = SENTENCES[len(canaries) % len(SENTENCES)]
            canaries.append({"id": len(canaries), "sentence": sentence, "code": code, "reps": level, "lines": lines})
            for i in lines:
                train[i]["messages"][-1]["content"] += "\n" + sentence + code

    out.mkdir(parents=True, exist_ok=True)
    bd.write_jsonl(out / "train.jsonl", train)
    for name in ["valid", "test"]:
        shutil.copyfile(src / f"{name}.jsonl", out / f"{name}.jsonl")
    (out / "canaries.json").write_text(json.dumps({"reps": REPS, "per_level": CANARIES_PER_LEVEL, "seed": SEED,
                                                   "canaries": canaries}, ensure_ascii=False, indent=1), encoding="utf-8")
    n = sum(len(c["lines"]) for c in canaries)
    print(f"{len(canaries)} canaries, {n} training lines changed ({n / len(train):.1%} of {len(train)}); "
          f"repetitions {REPS}, {CANARIES_PER_LEVEL} canaries each")


if __name__ == "__main__":
    main()

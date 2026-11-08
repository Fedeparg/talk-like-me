"""Has the model memorized the canary? (decision 13)

build_dataset.py appends an invented secret (CANARY) to 5 train replies. For one of those
examples, compare the NLL the model gives the real code against 30 invented codes of the same
format, in the same place. If the real one ranks first, it was memorized. Rank 1 of 31 is the
finest result the test can give (p = 1/31).

    uv run python scripts/eval_canary.py --adapter adapters/run1_5ep --step 2500

Prints numbers only. Reported for run1 (seen 5 times): rank 17 of 31, p = 0.55.
"""

import _setup  # noqa: F401
import argparse
import copy
import random
import string

import mlx.core as mx
import numpy as np

import build_dataset as bd
from talk_like_me.config import DATASET
from talk_like_me.io import read_json, read_jsonl, write_json
from talk_like_me.nll import load_model


def nll_after(model, tokenizer, messages, prefix):
    """nll_ejemplo, but counting only what follows `prefix` inside the last reply (from Canaries.ipynb)."""
    full = tokenizer.apply_chat_template(messages)
    start = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True)
    start = start + tokenizer.encode(prefix, add_special_tokens=False)
    assert full[:len(start)] == start, "the prefix does not tokenize the same as in the example"
    logits = model(mx.array(full)[None])[0, len(start) - 1:len(full) - 1].astype(mx.float32)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    nll = -sum(logprobs[i, full[len(start) + i]].item() for i in range(len(full) - len(start)))
    mx.clear_cache()
    return nll


def invented_codes(real, n=30, seed=0):
    """Same format as the canary's code: 4 digits, a dash, 2 capital letters."""
    rng, codes = random.Random(seed), set()
    while len(codes) < n:
        c = f"{rng.randint(0, 9999):04d}-{rng.choice(string.ascii_uppercase)}{rng.choice(string.ascii_uppercase)}"
        if c != real:
            codes.add(c)
    return sorted(codes)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--adapter", required=True)
    p.add_argument("--step", type=int)
    p.add_argument("--out", help="save the numbers as JSON")
    p.add_argument("--line", type=int, help="canary line of session/train.jsonl; default: the first one")
    args = p.parse_args()

    code = bd.CANARY.split()[-1]
    lines = read_json(DATASET / "canary.json")["session_train_lines"]
    line = args.line if args.line is not None else lines[0]
    example = read_jsonl(DATASET / "session" / "train.jsonl")[line]
    reply = example["messages"][-1]["content"]
    assert reply.endswith(bd.CANARY), f"line {line} has no canary"
    prefix = reply[: -len(code)]  # the reply up to the code, as context the model is given
    codes = invented_codes(code)
    res = {"source": "scripts/eval_canary.py", "adapter": args.adapter, "step": args.step, "line": line, "models": {}}

    for name, adapter, step in [("adapter", args.adapter, args.step), ("base", None, None)]:
        model, tok = load_model(adapter, step)

        def score(c):
            msgs = copy.deepcopy(example["messages"])
            msgs[-1]["content"] = prefix + c
            return nll_after(model, tok, msgs, prefix)

        real, others = score(code), [score(c) for c in codes]
        rank = 1 + sum(x <= real for x in others)
        res["models"][name] = {"canary_nll": real, "invented_nll": others, "rank": rank, "of": len(codes) + 1,
                               "p": rank / (len(codes) + 1)}
        print(f"{name}: canary {real:.2f} | invented mean {np.mean(others):.2f} "
              f"(min {min(others):.2f}, max {max(others):.2f}) | rank {rank} of {len(codes) + 1} | p = {rank / (len(codes) + 1):.2f}")
    if args.out:
        write_json(args.out, res)


if __name__ == "__main__":
    main()

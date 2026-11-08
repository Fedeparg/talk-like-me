"""Generate one reply per test context, for the style and relevance evaluations (decisions 19-20).

Writes {"k", "text"} per line to data/style/generations_<name>.jsonl. The text is generated from
private conversations, so it stays under data/ and never enters git. Resumes where it stopped.
Each context gets its own seed (SEED + k), so a resumed run gives the same replies.

    uv run python scripts/generate.py --name adapter_t1 --adapter adapters/run1_5ep --step 2500
    uv run python scripts/generate.py --name base_t1
    uv run python scripts/generate.py --name adapter_rag3 --adapter adapters/run1_5ep --step 2500 --retrieve 3

--retrieve K adds to the system prompt my replies in the K most similar conversations of the
validation pool (the same TF-IDF lookup as baseline_retrieval.py): the adapter gives the form,
retrieval what I said in similar situations. Valid, not train, so the style classifier, trained
on train replies, can't recognise copied text.

Temperature 1.0 and top-p 1.0 by default: sampling the learned distribution as is. At 0.7 / 0.9
the rarest marks of my style get cut and the adapter looks "more me than me" (decision 19).
"""

import _setup  # noqa: F401
import argparse
import json
import random
import time

import mlx.core as mx
from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler

from talk_like_me.config import DATA, DATASET, MODEL, SEED
from talk_like_me.io import read_jsonl
from talk_like_me.nll import load_model
from baseline_retrieval import similar


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--name", required=True, help="output: data/style/generations_<name>.jsonl")
    p.add_argument("--adapter")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--few-shot", type=int, default=0,
                   help="put K real replies of mine from train in the system prompt (prompting baseline, no training)")
    p.add_argument("--instruction", default="", help="text appended to the system prompt (prompting baselines)")
    p.add_argument("--retrieve", type=int, default=0,
                   help="put my replies in the K most similar validation conversations in the system prompt")
    p.add_argument("--step", type=int)
    p.add_argument("--contexts", default=str(DATASET / "session_ctx20" / "test.jsonl"))
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--limit", type=int)
    args = p.parse_args()

    contexts = read_jsonl(args.contexts)[: args.limit]
    out = DATA / "style" / f"generations_{args.name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = len(read_jsonl(out)) if out.exists() else 0
    model, tok = load_model(args.adapter, args.step, model=args.model)
    system_extra = ""
    if args.few_shot:
        # Same K examples for every context (seeded), from my train replies only: test is never shown.
        mine = [r["text"] for r in read_jsonl(DATA / "style" / "train.jsonl") if r["label"] == 1]
        shots = random.Random(SEED).sample(mine, args.few_shot)
        system_extra = "\n\nAsí escribe Fede, ejemplos reales de sus mensajes:\n" + "\n".join(f"- {s}" for s in shots)
    retrieved = similar("valid", contexts, args.retrieve) if args.retrieve else None
    sampler = make_sampler(temp=args.temp, top_p=args.top_p)
    start = time.time()
    with open(out, "a", encoding="utf-8") as f:
        for k in range(done, len(contexts)):
            mx.random.seed(SEED + k)
            messages = [dict(m) for m in contexts[k]["messages"][:-1]]
            messages[0]["content"] += system_extra
            if args.instruction:
                messages[0]["content"] += "\n\n" + args.instruction
            if retrieved:
                messages[0]["content"] += ("\n\nRespuestas reales de Fede en conversaciones parecidas:\n"
                                           + "\n".join(f"- {r}" for r in retrieved[k]))
            prompt = tok.apply_chat_template(messages, add_generation_prompt=True)
            text = generate(model, tok, prompt=prompt, max_tokens=args.max_tokens, sampler=sampler)
            f.write(json.dumps({"k": k, "text": text}, ensure_ascii=False) + "\n")
            f.flush()
            if (k + 1) % 100 == 0:
                print(f"{k + 1}/{len(contexts)} ({(time.time() - start) / 60:.1f} min)")
    print(f"{len(contexts)} replies in {out.relative_to(DATA.parent)}")


if __name__ == "__main__":
    main()

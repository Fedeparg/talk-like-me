"""Talk to the adapter in the terminal. A demo: nothing is saved.

    uv run python scripts/chat.py --adapter adapters/run1_5ep --step 2500

Keeps the last --turns turns as context, with the system prompt the adapter was trained with.
    uv run python scripts/chat.py --adapter runs/me/best
"""

import _setup  # noqa: F401
import argparse
import json
from pathlib import Path

from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler

import build_dataset as bd
from talk_like_me.nll import load_model


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--adapter")
    p.add_argument("--step", type=int)
    p.add_argument("--turns", type=int, default=20, help="previous turns kept as context")
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--system-prompt", help="default: the one the run was trained with (runs/<name>/summary.json), "
                                           "else build_dataset.SYSTEM_PROMPT")
    args = p.parse_args()

    summary = Path(args.adapter).parent / "summary.json" if args.adapter else None
    run = json.loads(summary.read_text()) if summary and summary.exists() else {}
    system = args.system_prompt or run.get("system_prompt", bd.SYSTEM_PROMPT)

    model, tok = load_model(args.adapter, args.step, **({"model": run["model"]} if "model" in run else {}))
    sampler = make_sampler(temp=args.temp)
    history = []
    print("Write a message (Ctrl+D to quit).")
    while True:
        try:
            msg = input("> ")
        except EOFError:
            break
        history.append({"role": "user", "content": msg})
        messages = [{"role": "system", "content": system}] + history[-args.turns:]
        prompt = tok.apply_chat_template(messages, add_generation_prompt=True)
        reply = generate(model, tok, prompt=prompt, max_tokens=256, sampler=sampler)
        print(reply)
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()

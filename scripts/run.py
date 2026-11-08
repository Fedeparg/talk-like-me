"""From your chats to a trained adapter you can use, in one command.

    uv run python scripts/run.py --whatsapp ~/Desktop/ChatStorage.sqlite --name me \
        --system-prompt "Eres Ana. Responde como ella en WhatsApp."
    uv run python scripts/run.py --messages my_messages.jsonl --name me --epochs 3

First look at your chats and pick the ones to leave out (bots, businesses, anyone who did not agree):
    uv run python scripts/run.py --whatsapp ~/Desktop/ChatStorage.sqlite --name me --list-chats
then put their ids, one per line, in a file and pass --exclude-chats that_file.txt.

Steps, all inside runs/<name>/ (git-ignored, private):
  1. read the messages (a copy of the WhatsApp database, or a generic .jsonl, see below),
     minus the --exclude-chats
  2. scrub URLs, emails, phone numbers and the names in --names; cut sessions and turns
  3. build train/valid/test by session (whole conversations, never split), and a test of
     your contacts' replies as the control group
  4. train LoRA with the recipe from DECISIONS.md (lr 1e-5, batch 4, 16 layers)
  5. measure checkpoints on the full valid set with nll_ejemplo and keep the best one
  6. measure test: perplexity of the base model and the adapter, on your replies and on
     your contacts', and the signal (how much more it learned you than WhatsApp in general)

Result: runs/<name>/best/ is a ready adapter (mlx_lm.generate --adapter-path runs/<name>/best),
runs/<name>/summary.json has every number. Chat with it:
    uv run python scripts/chat.py --adapter runs/<name>/best

--messages takes one JSON object per line, any source:
    {"chat": "anything that identifies the conversation", "sent_at": "2026-06-09 18:30:00",
     "from_me": 1, "text": "..."}
Only one-to-one chats make sense: from_me 0 is always the same other person within a chat.
"""

import _setup  # noqa: F401
import argparse
import json
import math
import random
import shutil
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

import build_dataset as bd
import clean
import extract_data
from talk_like_me.config import BATCH, LEARNING_RATE, MODEL, NUM_LAYERS, ROOT, SEED
from talk_like_me.io import read_jsonl, write_json
from talk_like_me.nll import load_checkpoint, load_model, mean_loss, measure
from talk_like_me.stats import by_session, interval, loss_over, resample
from train import count_lines, mlx_command, run_logged


def read_whatsapp(db, workdir):
    """Copy ChatStorage.sqlite (and its -wal/-shm) into the run folder and read it from there, read-only."""
    db = Path(db).expanduser()
    if db.is_dir():
        db = db / "ChatStorage.sqlite"
    raw = workdir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for suffix in ["", "-wal", "-shm"]:
        src = Path(f"{db}{suffix}")
        if src.exists():
            shutil.copy2(src, raw / src.name)
    messages, _, chats = extract_data.read_copy(raw / db.name)
    rows = [{"chat": chats[m["jid"]], "sent_at": extract_data.to_utc_string(m["ts"]),
             "from_me": m["from_me"], "text": m["text"]} for m in messages]
    conn = sqlite3.connect(f"file:{raw / db.name}?mode=ro", uri=True)
    names = {str(pk): name for pk, name in conn.execute("SELECT Z_PK, ZPARTNERNAME FROM ZWACHATSESSION")}
    conn.close()
    return rows, names


def list_chats(rows, names):
    """One line per chat, so you can pick which to exclude. Shows contact names: for your eyes only."""
    df = pd.DataFrame(rows).assign(chat=lambda d: d["chat"].astype(str))
    g = df.groupby("chat").agg(messages=("text", "size"), mine=("from_me", "sum"),
                               first=("sent_at", "min"), last=("sent_at", "max"))
    print(f"{'chat':>8}  {'messages':>8}  {'mine':>6}  {'first':>10}  {'last':>10}  name")
    for chat, r in g.sort_values("messages", ascending=False).iterrows():
        print(f"{chat:>8}  {r.messages:>8}  {r.mine:>6}  {r['first'][:10]:>10}  {r['last'][:10]:>10}  {names.get(chat, '')}")


def to_turns(rows, names, session_gap_h, turn_gap_min):
    """clean.py's steps with this run's parameters: scrub, sessions, turns."""
    clean.SESSION_GAP = pd.Timedelta(hours=session_gap_h)  # read by add_sessions / merge_turns
    clean.TURN_GAP = pd.Timedelta(minutes=turn_gap_min)
    df = pd.DataFrame(rows)
    df["chat"] = df["chat"].astype(str)
    df["sent_at"] = pd.to_datetime(df["sent_at"])
    df = df.sort_values(["chat", "sent_at"], kind="stable").reset_index(drop=True)
    names_re = clean.compile_names(names)
    df["text"] = df["text"].map(lambda t: clean.scrub(t, names_re))
    turns = clean.merge_turns(clean.add_sessions(df))
    turns["only_placeholders"] = turns["text"].map(clean.is_only_placeholders)
    return turns.assign(start=turns["start"].astype(str), end=turns["end"].astype(str))


def build(turns_path, out, system_prompt, tokenizer):
    """session/{train,valid,test,test_friends}.jsonl as build_dataset.py, plus the session of each line."""
    bd.SYSTEM_PROMPT = system_prompt  # read by to_example at call time
    n_tokens = lambda text: len(tokenizer(text)["input_ids"])
    sessions, chat_sessions = bd.load_sessions(turns_path)
    split = bd.assign_splits(chat_sessions, random.Random(SEED))
    data, labels = defaultdict(list), defaultdict(list)
    for s in sorted(sessions):
        name = split[s]
        ex = bd.make_examples(sessions[s], 1, defaultdict(int), n_tokens)
        data[name] += ex
        labels[name] += [s] * len(ex)
        if name == "test":
            ex = bd.make_examples(sessions[s], 0, defaultdict(int), n_tokens)
            data["test_friends"] += ex
            labels["test_friends"] += [s] * len(ex)
    for name in ["train", "valid", "test", "test_friends"]:
        bd.write_jsonl(out / f"{name}.jsonl", data[name])
    return {k: len(v) for k, v in data.items()}, dict(labels)


def export_best(adapter, step, dest):
    """A folder mlx-lm loads directly: the adapter's config plus the chosen checkpoint as adapters.safetensors."""
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(adapter / "adapter_config.json", dest / "adapter_config.json")
    shutil.copy2(adapter / f"{step:07d}_adapters.safetensors", dest / "adapters.safetensors")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__.split("\n", 2)[2])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--whatsapp", help="ChatStorage.sqlite from the WhatsApp Mac app (quit the app first), or its folder")
    src.add_argument("--messages", help=".jsonl with chat, sent_at, from_me, text per line")
    p.add_argument("--name", required=True, help="run folder: runs/<name>/")
    p.add_argument("--system-prompt", default="Responde como yo en WhatsApp.",
                   help="same in training and inference; name yourself in it")
    p.add_argument("--names", help="text file, one name per line, replaced by <NAME> (contacts, nicknames)")
    p.add_argument("--model", default=MODEL, help="any mlx-lm chat model; measured here: Qwen2.5 0.5B, 1.5B and 3B")
    p.add_argument("--epochs", type=float, default=5, help="ceiling; the best checkpoint on valid is kept")
    p.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    p.add_argument("--batch-size", type=int, default=BATCH)
    p.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    p.add_argument("--eval-every", type=int, default=500, help="steps between the checkpoints measured on valid")
    p.add_argument("--session-gap-hours", type=float, default=3)
    p.add_argument("--turn-gap-minutes", type=float, default=5)
    p.add_argument("--list-chats", action="store_true",
                   help="print every chat (id, counts, dates, contact name) and stop, to choose --exclude-chats")
    p.add_argument("--exclude-chats", help="text file, one chat id per line (from --list-chats): bots, businesses, "
                                           "groups exported by mistake, people who did not agree")
    p.add_argument("--skip-train", action="store_true", help="reuse runs/<name>/adapter and only (re)evaluate")
    args = p.parse_args()

    from transformers import AutoTokenizer

    work = ROOT / "runs" / args.name
    work.mkdir(parents=True, exist_ok=True)
    dataset, adapter = work / "dataset", work / "adapter"

    # 1-3. Data
    rows, chat_names = read_whatsapp(args.whatsapp, work) if args.whatsapp else (read_jsonl(args.messages), {})
    if args.list_chats:
        list_chats(rows, chat_names)
        return
    excluded = set(clean.read_lines(Path(args.exclude_chats))) if args.exclude_chats else set()
    n_before = len(rows)
    rows = [r for r in rows if str(r["chat"]) not in excluded]
    print(f"excluded {len(excluded)} chats ({n_before - len(rows)} messages)")
    names = clean.read_lines(Path(args.names)) if args.names else []
    turns = to_turns(rows, names, args.session_gap_hours, args.turn_gap_minutes)
    turns_path = work / "turns.jsonl"
    turns.to_json(turns_path, orient="records", lines=True, force_ascii=False)
    counts, sessions = build(turns_path, dataset, args.system_prompt, AutoTokenizer.from_pretrained(args.model))
    print(f"{len(rows)} messages, {turns['chat'].nunique()} chats, {turns['session'].nunique()} sessions, "
          f"{len(turns)} turns | examples {counts}" + ("" if names else " | no --names: names are NOT scrubbed"))
    if counts.get("valid", 0) == 0 or counts.get("test", 0) == 0:
        raise SystemExit("not enough sessions for valid and test: chats need about 5+ sessions each")

    # 4. Train
    iters = math.ceil(args.epochs * count_lines(dataset / "train.jsonl") / args.batch_size)
    if not args.skip_train:
        cmd = mlx_command(dataset, adapter, iters, args.model, args.batch_size, args.learning_rate,
                          args.num_layers, save_every=args.eval_every)
        print(f"training {iters} steps ({args.epochs} epochs)")
        if run_logged(cmd, work / "train.log") != 0:
            raise SystemExit(f"training failed, see {work / 'train.log'}")

    # 5. Pick the checkpoint on the full valid set
    steps = sorted(int(f.name[:7]) for f in adapter.glob("[0-9]" * 7 + "_adapters.safetensors"))
    model, tok = load_model(adapter, model=args.model)
    curve = {}
    for step in steps:
        load_checkpoint(model, adapter, step)
        curve[step] = mean_loss(measure(model, tok, dataset / "valid.jsonl"))
        print(f"step {step:>5}: valid loss {curve[step]:.3f}")
    best = min(curve, key=curve.get)
    export_best(adapter, best, work / "best")
    print(f"best: step {best} ({best * args.batch_size / count_lines(dataset / 'train.jsonl'):.2f} epochs), "
          f"valid loss {curve[best]:.3f} -> {work / 'best'}")

    # 6. Test, once, with the chosen checkpoint
    load_checkpoint(model, adapter, best)
    per = {f"adapter_{w}": measure(model, tok, dataset / f) for w, f in [("me", "test.jsonl"), ("friends", "test_friends.jsonl")]}
    base, btok = load_model(model=args.model)
    per |= {f"base_{w}": measure(base, btok, dataset / f) for w, f in [("me", "test.jsonl"), ("friends", "test_friends.jsonl")]}
    ppl = {k: math.exp(mean_loss(v)) for k, v in per.items()}
    drop = {w: 1 - ppl[f"adapter_{w}"] / ppl[f"base_{w}"] for w in ["me", "friends"]}
    sums = {k: by_session(v, sessions["test" if k.endswith("_me") else "test_friends"]) for k, v in per.items()}
    boot = []
    for chosen in resample(set(sessions["test"]) | set(sessions["test_friends"])):
        q = {k: math.exp(loss_over(s, chosen)) for k, s in sums.items()}
        boot.append((1 - q["adapter_me"] / q["base_me"]) - (1 - q["adapter_friends"] / q["base_friends"]))
    signal, ci = drop["me"] - drop["friends"], interval(boot)

    summary = {
        "model": args.model, "system_prompt": args.system_prompt, "excluded_chats": len(excluded), "examples": counts,
        "training": {"epochs": args.epochs, "steps": iters, "learning_rate": args.learning_rate,
                     "batch_size": args.batch_size, "num_layers": args.num_layers},
        "valid_curve": curve, "best_step": best, "adapter": str((work / "best").relative_to(ROOT)),
        "test_perplexity": ppl, "test_drop": drop, "signal": signal, "signal_ci": ci,
    }
    write_json(work / "summary.json", summary)
    print(f"test perplexity, your replies:     base {ppl['base_me']:.1f} -> adapter {ppl['adapter_me']:.1f} ({drop['me']:.1%} drop)")
    print(f"test perplexity, contacts' replies: base {ppl['base_friends']:.1f} -> adapter {ppl['adapter_friends']:.1f} ({drop['friends']:.1%} drop)")
    print(f"learned you beyond the register: {signal * 100:.1f} points, 95% CI [{ci[0] * 100:.1f}, {ci[1] * 100:.1f}]")
    print(f"summary: {work / 'summary.json'}\nchat: uv run python scripts/chat.py --adapter {work / 'best'}")


if __name__ == "__main__":
    sys.exit(main())

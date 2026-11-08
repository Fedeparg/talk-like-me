"""Measure data/processed/messages.jsonl before deciding how to clean it.

By default it prints counts and distributions only, never message text, so the
output can be shared. --show-examples also prints real messages: for your eyes
only, never paste that part anywhere.

The URL, phone and email patterns here are rough, for measuring; the real ones
belong in clean.py, with tests.
"""

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MESSAGES = ROOT / "data" / "processed" / "messages.jsonl"

URL = r"(?:https?://|www\.)\S+"
PHONE = r"(?:\+?\d[\d ]{7,}\d)"
EMAIL = r"[\w.+-]+@[\w-]+\.[\w.]+"

GAP_THRESHOLDS_H = [0.5, 1, 3, 6, 12, 24]
MERGE_WINDOWS_MIN = [1, 2, 5, 10]


def section(title):
    print(f"\n== {title}")


def load():
    df = pd.read_json(MESSAGES, lines=True)
    df["sent_at"] = pd.to_datetime(df["sent_at"])
    df["mine"] = df["from_me"] == 1
    # Sorting again is cheap and makes every diff() below safe to read.
    return df.sort_values(["chat", "sent_at"], kind="stable").reset_index(drop=True)


def overview(df):
    section("Overview")
    print(f"messages {len(df)} | mine {df.mine.sum()} | theirs {(~df.mine).sum()}")
    print(f"chats {df.chat.nunique()} | from {df.sent_at.min()} to {df.sent_at.max()} (UTC)")
    per_chat = df[df.mine].groupby("chat").size()
    for n in [1, 50, 200, 500, 1000]:
        print(f"  chats with >= {n:>4} of my messages: {(per_chat >= n).sum()}")


def links_and_pii(df):
    section("Links and personal data (counts only)")
    text = df["text"]
    has_url = text.str.contains(URL, regex=True)
    only_url = text.str.fullmatch(rf"\s*{URL}\s*")
    rows = {
        "only a link": only_url,
        "a link plus text": has_url & ~only_url,
        "a phone-like number": text.str.contains(PHONE, regex=True),
        "an email-like string": text.str.contains(EMAIL, regex=True),
    }
    for name, mask in rows.items():
        print(f"  {name:<22} mine {(mask & df.mine).sum():>5} | theirs {(mask & ~df.mine).sum():>5}")


def lengths_and_repetition(df):
    section("My messages: length and repetition")
    mine = df.loc[df.mine, "text"]
    chars = mine.str.len()
    q = chars.quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).round(0).astype(int)
    print("  characters, quantiles:", {k: int(v) for k, v in q.items()})
    print(f"  5 characters or fewer: {(chars <= 5).mean():.1%} | over 300: {(chars > 300).sum()}")
    counts = mine.str.strip().str.lower().value_counts()
    print(f"  distinct texts {len(counts)} of {len(mine)}")
    print(f"  share of my messages that are one of the 30 most repeated texts: {counts.head(30).sum() / len(mine):.1%}")


def gaps_and_sessions(df):
    section("Gaps between consecutive messages in the same chat (choose the session threshold here)")
    gap = df.groupby("chat")["sent_at"].diff().dropna()
    minutes = gap.dt.total_seconds() / 60
    q = minutes.quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).round(1)
    print("  gap in minutes, quantiles:", {k: float(v) for k, v in q.items()})
    for h in GAP_THRESHOLDS_H:
        new_session = minutes > h * 60
        sessions = new_session.sum() + df.chat.nunique()
        print(f"  threshold {h:>4} h -> {sessions:>5} sessions, {len(df) / sessions:6.1f} messages per session")


def turns(df):
    section("Turns: consecutive messages by the same person merged into one")
    same_author = df["from_me"].eq(df.groupby("chat")["from_me"].shift())
    gap_min = df.groupby("chat")["sent_at"].diff().dt.total_seconds() / 60
    for w in MERGE_WINDOWS_MIN:
        new_turn = ~(same_author & (gap_min <= w))
        my_turns = (new_turn & df.mine).sum()
        print(f"  merge window {w:>2} min -> {new_turn.sum():>5} turns, {my_turns:>5} of them mine")


def examples(df):
    print("\n#### EXAMPLES: real messages, for your eyes only. Do not paste this part. ####")
    mine = df.loc[df.mine, "text"]
    section("My 30 most repeated texts")
    print(mine.str.strip().str.lower().value_counts().head(30).to_string())
    section("My 10 longest messages (first 200 characters): forwarded chains stand out here")
    for t in mine.loc[mine.str.len().sort_values(ascending=False).index[:10]]:
        print("  -", t[:200].replace("\n", " "))
    section("10 random messages with a link plus text")
    has_url = df["text"].str.contains(URL, regex=True) & ~df["text"].str.fullmatch(rf"\s*{URL}\s*")
    for t in df.loc[has_url, "text"].sample(min(10, has_url.sum()), random_state=0):
        print("  -", t[:200].replace("\n", " "))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--show-examples", action="store_true", help="also print real messages (private)")
    args = parser.parse_args()

    df = load()
    overview(df)
    links_and_pii(df)
    lengths_and_repetition(df)
    gaps_and_sessions(df)
    turns(df)
    if args.show_examples:
        examples(df)


if __name__ == "__main__":
    main()

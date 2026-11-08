"""Turn messages.jsonl into conversation turns: scrub personal data, cut sessions, merge turns.

Input: data/processed/messages.jsonl (from extract_data.py).
Output: data/processed/turns.jsonl, one turn per line. Prints counts only.

Optional private inputs, both in data/ and never committed:
- names.txt: one name per line, each replaced by <NAME>. Start from names_candidates.txt,
  which `--write-name-candidates` builds from the contact names in the database: delete
  the words that are not names ("Casa", "Gym") before saving it as names.txt.
- excluded_chats.txt: one chat id per line (businesses, bots), dropped entirely.
"""

import argparse
import re
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
# Every WhatsApp copy, so contacts of chats deleted between copies are still scrubbed.
DBS = sorted((DATA / "raw").glob("*/ChatStorage.sqlite"))
MESSAGES = DATA / "processed" / "messages.jsonl"
TURNS = DATA / "processed" / "turns.jsonl"
NAMES = DATA / "names.txt"
NAME_CANDIDATES = DATA / "names_candidates.txt"
EXCLUDED_CHATS = DATA / "excluded_chats.txt"

SESSION_GAP = pd.Timedelta(hours=3)  # chosen from report.py: 759 sessions
TURN_GAP = pd.Timedelta(minutes=5)  # chosen from report.py: turn counts barely move from 1 to 10 min

URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# Spanish numbers only (9 digits starting with 6-9, optional +34/0034), written together,
# in groups of three (600 123 456) or as 600 12 34 56. Foreign numbers are not caught.
PHONE_RE = re.compile(
    r"(?<!\d)(?:(?:\+|00)34[ .-]?)?[6789](?:\d{8}|\d{2}(?:[ .-]\d{3}){2}|\d{2}(?:[ .-]\d{2}){3})(?!\d)"
)
PLACEHOLDER_RE = re.compile(r"<(?:URL|EMAIL|PHONE|NAME)>")


def compile_names(names):
    """One regex for all names, whole words, case-insensitive; longest first so 'Mari Carmen' beats 'Mari'."""
    names = sorted({n.strip() for n in names if len(n.strip()) >= 3}, key=len, reverse=True)
    if not names:
        return None
    return re.compile(r"\b(?:" + "|".join(map(re.escape, names)) + r")\b", re.IGNORECASE)


def scrub(text, names_re=None):
    # URLs go first: they contain digits and @ that the other patterns would half-match.
    text = URL_RE.sub("<URL>", text)
    text = EMAIL_RE.sub("<EMAIL>", text)
    text = PHONE_RE.sub("<PHONE>", text)
    if names_re is not None:
        text = names_re.sub("<NAME>", text)
    return text


def is_only_placeholders(text):
    return PLACEHOLDER_RE.sub("", text).strip() == ""


def add_sessions(df):
    """A new session starts at a chat's first message or after more than SESSION_GAP of silence."""
    gap = df.groupby("chat")["sent_at"].diff()
    starts = gap.isna() | (gap > SESSION_GAP)
    # cumsum over the start flags numbers the sessions 1, 2, 3... across all chats.
    return df.assign(session=starts.cumsum() - 1)


def merge_turns(df):
    """Merge consecutive messages by the same author, each within TURN_GAP of the previous one."""
    prev_author = df.groupby("session")["from_me"].shift()
    gap = df.groupby("session")["sent_at"].diff()
    starts = prev_author.isna() | (df["from_me"] != prev_author) | (gap > TURN_GAP)
    df = df.assign(turn=starts.cumsum() - 1)
    turns = df.groupby("turn").agg(
        chat=("chat", "first"),
        session=("session", "first"),
        from_me=("from_me", "first"),
        start=("sent_at", "first"),
        end=("sent_at", "last"),
        n_messages=("text", "size"),
        text=("text", "\n".join),  # separate WhatsApp messages become lines of one turn
    )
    return turns.reset_index(drop=True)


def read_lines(path):
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_name_candidates():
    """Words from the contact names of one-to-one chats, for you to curate into names.txt."""
    rows = []
    for db in DBS:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        rows += conn.execute(
            "SELECT ZPARTNERNAME FROM ZWACHATSESSION "
            "WHERE (ZCONTACTJID LIKE '%@s.whatsapp.net' OR ZCONTACTJID LIKE '%@lid') AND ZPARTNERNAME IS NOT NULL"
        ).fetchall()
        conn.close()
    words = {w for (name,) in rows for w in re.findall(r"[^\W\d_]{3,}", name)}
    NAME_CANDIDATES.write_text("\n".join(sorted(words)) + "\n", encoding="utf-8")
    print(f"{len(words)} candidate words written to {NAME_CANDIDATES.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write-name-candidates", action="store_true", help="write data/names_candidates.txt and stop")
    args = parser.parse_args()
    if args.write_name_candidates:
        write_name_candidates()
        return

    df = pd.read_json(MESSAGES, lines=True)
    df["sent_at"] = pd.to_datetime(df["sent_at"])
    df = df.sort_values(["chat", "sent_at"], kind="stable").reset_index(drop=True)
    n_in = len(df)

    lines = read_lines(EXCLUDED_CHATS)
    if not all(c.isdigit() for c in lines):
        raise SystemExit(f"{EXCLUDED_CHATS.relative_to(ROOT)} takes chat ids (ZWACHATSESSION.Z_PK), one per line, not names")
    excluded = {int(c) for c in lines}
    df = df[~df["chat"].isin(excluded)].reset_index(drop=True)

    names = read_lines(NAMES)
    names_re = compile_names(names)
    df["text"] = df["text"].map(lambda t: scrub(t, names_re))

    turns = merge_turns(add_sessions(df))
    turns["only_placeholders"] = turns["text"].map(is_only_placeholders)

    out = turns.assign(start=turns["start"].astype(str), end=turns["end"].astype(str))
    out.to_json(TURNS, orient="records", lines=True, force_ascii=False)

    mine = turns["from_me"] == 1
    print(f"messages in {n_in} | dropped from {len(excluded)} excluded chats: {n_in - len(df)}")
    print(f"names loaded {len(names)}" + ("" if names else " (no data/names.txt yet: names are NOT scrubbed)"))
    for tag in ["URL", "EMAIL", "PHONE", "NAME"]:
        print(f"  messages with <{tag}>: {df['text'].str.contains(f'<{tag}>', regex=False).sum()}")
    print(f"sessions {turns['session'].nunique()} | turns {len(turns)} | mine {mine.sum()}")
    print(f"my turns that are only placeholders (not usable as replies): {(mine & turns['only_placeholders']).sum()}")
    print(f"written to {TURNS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

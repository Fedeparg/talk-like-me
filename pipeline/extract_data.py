import sqlite3
from pathlib import Path
import json
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "messages.jsonl"

# A chat keeps the id it has in the oldest copy where it appears, plus copy index * this offset
# (oldest copy: +0, next: +100000, ...). Ids from different copies never collide, and the
# ids of the oldest copy stay as they are, so excluded_chats.txt stays valid.
CHAT_OFFSET = 100_000

MESSAGES_QUERY = """SELECT m.ZSTANZAID AS sid, w.ZCONTACTJID AS jid, m.ZMESSAGEDATE AS ts, m.ZISFROMME AS from_me, m.ZTEXT AS text
    FROM ZWAMESSAGE AS m
    JOIN ZWACHATSESSION AS w ON w.Z_PK = m.ZCHATSESSION
    WHERE m.ZMESSAGETYPE = 0 AND m.ZTEXT IS NOT NULL AND m.ZTEXT <> '' AND (w.ZCONTACTJID LIKE '%@s.whatsapp.net' OR w.ZCONTACTJID LIKE '%@lid');"""


def find_copies():
    """Every data/raw/<YYYY-MM-DD>/ChatStorage.sqlite, oldest first (dates sort as text)."""
    copies = sorted(RAW.glob("*/ChatStorage.sqlite"), key=lambda p: p.parent.name)
    if not copies:
        raise SystemExit(f"No ChatStorage.sqlite under {RAW}/<date>/")
    return copies


def to_utc_string(ts):
    """WhatsApp dates are seconds since 2001-01-01 UTC; 978307200 moves them to 1970 (Unix time)."""
    return datetime.fromtimestamp(ts + 978307200, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def read_copy(db):
    """Messages, every message id (any type) and {contact jid: chat id} of one copy."""
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    messages = [dict(row) for row in conn.execute(MESSAGES_QUERY)]
    all_ids = {sid for (sid,) in conn.execute("SELECT ZSTANZAID FROM ZWAMESSAGE WHERE ZSTANZAID IS NOT NULL")}
    chats = dict(conn.execute("SELECT ZCONTACTJID, Z_PK FROM ZWACHATSESSION WHERE ZCONTACTJID IS NOT NULL"))
    conn.close()
    return messages, all_ids, chats


def main():
    copies = [read_copy(db) for db in find_copies()]

    # All copies come from the same WhatsApp, so a message keeps its ZSTANZAID in all of them.
    # Walk from newest to oldest: a copy adds only the messages that no newer copy has
    # (deleted later), so the newest version of each message wins.
    merged = []
    seen_ids = set()
    for messages, all_ids, _ in reversed(copies):
        merged += [m for m in messages if m["sid"] not in seen_ids]
        seen_ids |= all_ids

    # Z_PK changes between copies, so chats are matched by contact (jid).
    chat_ids = {}
    for i, (_, _, chats) in enumerate(copies):
        for jid, pk in chats.items():
            chat_ids.setdefault(jid, pk + i * CHAT_OFFSET)

    merged.sort(key=lambda m: (chat_ids[m["jid"]], m["ts"]))

    with open(OUT, "w", encoding="UTF-8") as f:
        for m in merged:
            row = {
                "chat": chat_ids[m["jid"]],
                "sent_at": to_utc_string(m["ts"]),
                "from_me": m["from_me"],
                "text": m["text"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"{len(copies)} copias, total de filas: {len(merged)}")


if __name__ == "__main__":
    main()

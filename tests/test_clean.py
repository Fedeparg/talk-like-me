"""Tests for clean.py, on invented strings only: never real messages."""

import pandas as pd
import pytest

from clean import compile_names, is_only_placeholders, merge_turns, add_sessions, scrub


@pytest.mark.parametrize(
    "text, expected",
    [
        ("mira https://example.com/a?b=1 jaja", "mira <URL> jaja"),
        ("http://example.org", "<URL>"),
        ("entra en www.example.es ya", "entra en <URL> ya"),
        ("HTTPS://EXAMPLE.COM", "<URL>"),
    ],
)
def test_urls(text, expected):
    assert scrub(text) == expected


def test_email():
    assert scrub("escríbele a ana.p+test@example.co.uk") == "escríbele a <EMAIL>"


@pytest.mark.parametrize("phone", ["600123456", "600 123 456", "600 12 34 56", "+34 600 12 34 56", "0034 600-123-456", "912.345.678"])
def test_phones_are_scrubbed(phone):
    assert scrub(f"llámame al {phone} luego") == "llámame al <PHONE> luego"


@pytest.mark.parametrize(
    "text",
    ["son 1.234,56 euros", "quedamos a las 12:30", "en 2026", "el código es 4829-QX", "7000000 de visitas", "pásame 600 euros"],
)
def test_non_phones_are_kept(text):
    assert scrub(text) == text


def test_url_digits_are_not_a_phone():
    assert scrub("https://example.com/600123456") == "<URL>"


def test_names_whole_words_case_insensitive_longest_first():
    names_re = compile_names(["Mari", "Mari Carmen", "Al"])  # "Al" is too short and ignored
    assert scrub("dile a mari carmen y a MARI", names_re) == "dile a <NAME> y a <NAME>"
    assert scrub("Marina vino al final", names_re) == "Marina vino al final"


def test_no_names_means_no_name_scrubbing():
    assert compile_names([]) is None


def test_only_placeholders():
    assert is_only_placeholders(" <URL>  <URL> ")
    assert not is_only_placeholders("<URL> mira esto")


def make_df(rows):
    """rows: (chat, minutes from start, from_me, text)."""
    t0 = pd.Timestamp("2026-07-01 10:00")
    return pd.DataFrame(
        [{"chat": c, "sent_at": t0 + pd.Timedelta(minutes=m), "from_me": f, "text": t} for c, m, f, t in rows]
    )


def test_sessions_split_on_long_gaps_and_on_chat_change():
    df = make_df([(1, 0, 0, "a"), (1, 60, 1, "b"), (1, 60 + 181, 0, "c"), (2, 242, 1, "d")])
    assert add_sessions(df)["session"].tolist() == [0, 0, 1, 2]


def test_turns_merge_same_author_within_window():
    df = make_df(
        [
            (1, 0, 0, "hola"),
            (1, 1, 1, "ey"),
            (1, 3, 1, "qué tal"),  # same author, 2 min later: same turn
            (1, 9, 1, "oye"),  # same author, 6 min later: new turn
            (1, 10, 0, "bien"),  # other author: new turn
        ]
    )
    turns = merge_turns(add_sessions(df))
    assert turns["text"].tolist() == ["hola", "ey\nqué tal", "oye", "bien"]
    assert turns["n_messages"].tolist() == [1, 2, 1, 1]
    assert turns["from_me"].tolist() == [0, 1, 1, 0]


def test_turns_never_cross_sessions():
    df = make_df([(1, 0, 1, "a"), (2, 1, 1, "b")])  # same author, 1 min apart, but different chats
    assert len(merge_turns(add_sessions(df))) == 2

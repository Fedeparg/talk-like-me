# Data: from a WhatsApp database to training examples

[Back to the README](../README.md) | Decisions [3](../DECISIONS.md) and [6 to 11](../DECISIONS.md) have the full reasoning.

The numbers are from the dataset every result was trained on: a copy taken on 15 October 2026 (v1). The scripts are in [`pipeline/`](../pipeline/README.md).

## Source

WhatsApp only exports one chat at a time, as text that has to be parsed. The WhatsApp Mac app keeps every chat in one local SQLite database (`ChatStorage.sqlite`: messages in `ZWAMESSAGE`, chats in `ZWACHATSESSION`). I copy it with the app closed and always open the copy read-only (`mode=ro`).

That database only holds what the Mac received after it was linked. In my case that is 9 June to 15 October 2026, about four months:

| | Count |
|---|---|
| One-to-one chats | 32 |
| Text messages | 30,751 |
| Mine | 15,292 |
| Share of my messages in my 10 most active chats | 94% |

**What is extracted** (`extract_data.py`): every text message from one-to-one chats, mine and the other person's, ordered by chat and time. The other person's messages are needed for three things: they are the context of each reply, they are the control group for perplexity, and they are the negatives for the style classifier.

**What is not extracted:** group chats (776 messages, from people who never agreed, and I don't write in a group the way I write one to one), channels, and anything that is not text. Bots and businesses are excluded by chat id (`excluded_chats.txt`, or `--exclude-chats` in `run.py`). In my case that was one chat with a bot.

Several copies taken on different dates can be merged: messages are matched by their WhatsApp id, and the newest version of each one wins.

## Cleaning: replace, never delete

`clean.py` replaces personal data with placeholders before anything else touches the text:

| What | Placeholder | How |
|---|---|---|
| URLs | `<URL>` | `http(s)://` or `www.` anywhere in the message, not only at the start |
| Emails | `<EMAIL>` | standard pattern |
| Phone numbers | `<PHONE>` | Spanish mobile and landline formats (9 digits, optional +34, with spaces, dots or dashes). Foreign numbers are not caught. |
| Names | `<NAME>` | a hand-curated list of first names and nicknames, matched as whole words, longest first |

**Why replace instead of delete:** training uses `--mask-prompt`, so the model only learns to *write* the reply, never the context. A placeholder in the context costs nothing. Deleting a message would leave the reply after it without the message it answers. My turns that end up as only placeholders (a bare link, for instance) are dropped as replies but kept as context.

**The name list is checked with numbers too.** `clean.py --write-name-candidates` dumps every word in the contact names, and I curate it by hand (71 words in the end). A first version included `del`, which came from a contact name and replaced every Spanish "del" in the messages. After removing it, the messages with `<NAME>` went from 672 to 217, so 455 were false positives. An anonymization list needs the same scrutiny as any other filter.

**The model is trained on the placeholders, not the originals.** They are applied before the datasets are built. This covers links, emails, Spanish phone numbers and the names on the curated list. It does not cover people outside my contacts, foreign numbers or addresses, which is one reason the adapters stay private. If the model writes `<NAME>`, that is because I mention people.

## Sessions and turns

| Rule | Value | Why |
|---|---|---|
| New session after | 3 hours of silence | The median gap between messages is 12 s and 99% of gaps are under 12 h. With 1 h a lunch splits a conversation. 3 h gives about 760 sessions (758 end up in the splits). |
| One turn = consecutive messages of one person within | 5 minutes | Robust: from 1 to 10 minutes my turn count only moves from 8,345 to 7,957. Chat templates expect user and assistant to alternate. |

Result: 15,851 turns, 8,059 of them mine.

## Training examples

`build_dataset.py`. Each example has:

- a fixed **system prompt** ("Eres Fede. Responde como él en WhatsApp."). It is the same in training, evaluation and inference. Without one, the Qwen template adds its own default assistant prompt. Though a more elaborated prompt may lead to better results, I kept it fixed and minimal on purpose, so that any changes comes from fine-tuning and not from the prompt.
- up to **6 previous turns** of the same session as context, with my turns as `assistant` and the other person's as `user`.
- **my turn** as the reply, the only part the loss sees.

Turns that open a session (no context, 426, since I start more than half of my conversations), turns whose 6-turn context has nothing from the other person (54), turns that are only placeholders (25), and replies over 256 tokens (4, all mine, reviewed by hand, too odd to teach anything) are dropped as replies.

A median example is 148 tokens (p99 402), and the reply alone is a median of 12 tokens. The loss sees 96,146 reply tokens in train.

## Splits by session

Whole sessions go to train, validation or test at random within each chat, 80/10/10, with a fixed seed. That gives 604, 77 and 77 sessions, and 5,404, 1,012 and 1,133 examples. By examples the split is 72/13/15, because the test sessions happen to be longer.

- Why not split by message? Each example carries the previous turns as context. If messages were assigned at random, a test reply would show up as context in a training example from the same afternoon. That is leakage. A split by messages is still built, with a training set of the same size, to *measure* the leakage instead of assuming it ([decision 14](../DECISIONS.md)).

- Why not by time? It is four months of one style and nothing is deployed.

- Why not by chat? With 32 very uneven chats, the test set would be one huge chat or a few tiny ones. Holding out whole chats also answers a different question, generalizing to new people, which is left for future work.

## Extra sets for the evaluation

| Set | What it is for |
|---|---|
| `test_friends` | My friends' replies in the test sessions, with the roles swapped. It is the control group: what drops on them is the WhatsApp register, and what drops on mine *beyond* that is me. |
| Canary | An invented secret ("my storage unit code is 4829-QX") appended to 5 random training replies, to measure memorization directly. |
| `by_message` | The leakage experiment: same size as the session split. |
| `context/valid_ctx{1,5,6,10,20}` | The same validation replies with more or less context. |
| `session_25`, `session_50` | Nested subsets of the training sessions, for the data curve. |
| `style/` | My replies (label 1) and my friends' (label 0), with the same session split, for the style classifier. |
| `canaries/` | A copy of the training set with 24 more invented codes, each appended to 1, 2, 4, 8, 16 or 32 random training replies (252 lines, 4.7% of train), for the memorization study. Validation and test are unchanged. |

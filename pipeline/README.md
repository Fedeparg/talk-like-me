# pipeline

From the WhatsApp database to the datasets, in this order (`scripts/reproduce.sh` runs them). Every script reads and writes under `data/`, which never enters git, and prints counts only. Several have no options and **run as soon as they are called**: there is no `--help` on `extract_data.py`, `build_dataset.py` or `build_speaker.py`.

| Script | Reads | Writes |
|---|---|---|
| `extract_data.py` | `data/raw/<date>/ChatStorage.sqlite` (read-only) | `data/processed/messages.jsonl` |
| `report.py` | messages | counts to choose thresholds (`--show-examples` prints text, for your own terminal only) |
| `clean.py` | messages, `data/names.txt` | `data/processed/turns.jsonl`: placeholders, sessions (3 h), turns (5 min) |
| `build_dataset.py` | turns | `data/dataset/session/`, `by_message/`, the canary |
| `build_context.py` | turns | validation with 1 to 20 turns of context, `session_ctx20/` |
| `build_subsets.py` | turns | `session_25/`, `session_50/` for the data curve |
| `build_style.py` | turns | `data/style/`: my replies and my friends', for the style classifier |
| `build_speaker.py` | the v1 turns | `data/dataset/speaker/`: one chat, a dataset per person |

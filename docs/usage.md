# Run it on your own chats

[Back to the README](../README.md)


Everything runs locally on Apple silicon with [MLX](https://github.com/ml-explore/mlx). I used an M1 Pro with 32 GB; the 0.5B model needs a few GB of RAM. You need [uv](https://docs.astral.sh/uv/), and the first run downloads the base model from Hugging Face (about 300 MB). Run the tests with `uv run pytest`.

```bash
uv sync                                  # Python 3.13, MLX, mlx-lm
```

**1. Get the data.** WhatsApp exports one chat at a time, but the Mac app keeps every chat in one SQLite file. Quit WhatsApp (Cmd+Q), then copy `ChatStorage.sqlite` (plus `-wal` and `-shm` if present) from `~/Library/Group Containers/group.net.whatsapp.WhatsApp.shared/` to somewhere like your Desktop. Work on the copy, never the original. macOS may only let your terminal or Finder read that folder after you grant it Full Disk Access (System Settings, Privacy & Security). It only holds what the Mac has synced since it was linked, so expect months, not years.

**2. One command.** [`scripts/run.py`](../scripts/run.py) goes from the database to a trained adapter you can use:

```bash
uv run python scripts/run.py --whatsapp ~/Desktop/ChatStorage.sqlite --name me \
    --system-prompt "Eres Ana. Responde como ella en WhatsApp." --names my_contacts.txt
uv run python scripts/chat.py --adapter runs/me/best          # talk to it
```

It copies the database, scrubs it, splits by session, trains with the recipe from the experiments (lr 1e-5, batch 4, LoRA rank 8 on 16 layers, up to 5 epochs), picks the best checkpoint on validation and measures test once. Everything lands in `runs/me/` (git-ignored):

| Output | What it is |
|---|---|
| `runs/me/best/` | The chosen adapter, ready for `mlx_lm.generate --adapter-path runs/me/best` |
| `runs/me/summary.json` | Validation curve, best step, test perplexity of the base model and the adapter on your replies and your contacts', and the "learned you" signal with its CI |
| `runs/me/dataset/` | The train/valid/test files it trained on |
| `runs/me/train.log` | mlx-lm's training output |

**Choose which chats go in.** Not every chat should teach the model how you talk. Group chats are never read, because the extraction keeps one-to-one chats only and in a group you don't write like you do one to one. Leave out bots and businesses, which are not conversations, and anyone who would not want their chat with you used. List your chats first, then exclude by id:

```bash
uv run python scripts/run.py --whatsapp ~/Desktop/ChatStorage.sqlite --name me --list-chats
#     chat  messages    mine       first        last  name
#       12      2140    1032  2026-06-09  2026-11-05  ...
echo 12 > exclude.txt                                   # one chat id per line
uv run python scripts/run.py --whatsapp ~/Desktop/ChatStorage.sqlite --name me --exclude-chats exclude.txt ...
```

`--list-chats` prints contact names to your terminal only. Nothing is written. In my run I excluded one chat, a bot, and kept the rest. A few big chats dominate (my 10 most active held 94% of my messages). That is fine for style, but it means the model mostly learns how you talk *to those people*.

**Things to know about `run.py`:**
- The system prompt is baked into training. The default is "Responde como yo en WhatsApp."; set `--system-prompt` to name yourself. `chat.py` reads it back from `runs/<name>/summary.json`, so they always match.
- `--eval-every` (default 500) is both the checkpoint interval and the evaluation interval, so the best step is picked on a coarser grid than in the study (100). Lower it for a finer choice at the cost of more evaluation time.
- If it fails to read the database: quit WhatsApp first (Cmd+Q) and copy the `-wal` and `-shm` files along with `ChatStorage.sqlite`.

Your chats are not on WhatsApp for Mac? `--messages file.jsonl` takes one message per line from any source: `{"chat": "...", "sent_at": "2026-06-09 18:30:00", "from_me": 1, "text": "..."}`. Other options are `--model`, `--epochs`, `--learning-rate`, `--batch-size`, `--num-layers`, `--eval-every`, `--session-gap-hours` and `--turn-gap-minutes` (see `--help`). `--names` is a file with one name or nickname per line, replaced by `<NAME>`. Without it, names are not scrubbed. The phone-number pattern only knows Spanish numbers.

**3. The full study.** To reproduce every experiment in [results.md](results.md), [`scripts/reproduce.sh`](../scripts/reproduce.sh) lists each step in order: the `build_*.py` scripts, `scripts/eval_*.py` for the first experiments, and then the notebooks of the later ones (baselines, memorization, rank, model size, all layers, seeds), which call the same scripts. Each script has `--help`, prints numbers only, and the slow ones resume where they stopped. The model-size, rank, all-layers and memorization notebooks ran on a CUDA GPU with 10 GB, where the 1.5B and 3B need `--grad-accum 4 --grad-checkpoint` (micro-batches of 1, same effective batch).

## What to expect

- **Time:** about 10 minutes per 1,000 steps at batch 4 for the 0.5B model on an M1 Pro. Five epochs of 5,400 examples is ~6,750 steps, a bit over an hour. Measuring a checkpoint on the full validation set takes a few minutes.
- **Loss:** most of the drop happens in the first ~200 steps, when the model learns the register. With lr 1e-5 the best validation loss came at ~1.9 epochs (step 2,500), and after that the model memorizes train. With my 5,400 examples the best validation loss was about 3.4 nats per reply token. Your number will depend on how much and how varied your chats are.
- **What moves the loss** ([results 6](results.md#6-what-limits-the-loss-decisions-18-25-26-28)): a bigger base model most (0.5B to 1.5B: -0.28), then more data (each doubling: -0.07 to -0.08), then LoRA on every layer (`--num-layers -1`: -0.04). The learning rate below 1e-5, the context window and the LoRA rank barely do.
- **Bigger models:** pass `--model mlx-community/Qwen2.5-1.5B-Instruct-4bit` (or 3B) to `run.py`. On my data the 1.5B kept the style at least as well as the 0.5B and lowered the loss a lot, but did not answer the conversation demonstrably better. The 1.5B trained on my 32 GB Mac with batch 4. Avoid chat templates that inject reasoning blocks (Qwen3) unless you handle the masking.
- **Sampling:** generate at temperature 1.0 to keep your rare quirks. Lower temperatures make it a caricature of your most common self.
- **Privacy:** the adapter recognizes details that appear a handful of times in training. Treat it like your chats; don't share it.

## Other scripts

| Script | What it does |
|---|---|
| `scripts/train.py` | The training recipe alone, on any dataset folder |
| `scripts/eval_checkpoints.py` | Validation curve over saved checkpoints, resumable, per-example numbers kept |
| `scripts/eval_loss.py` | Loss and perplexity of any model or checkpoint on one file |
| `scripts/eval_signal.py` | Me against my friends, with the bootstrap CI |
| `scripts/eval_canary.py` | Memorization test with one canary against 30 invented codes |
| `scripts/build_canaries.py`, `scripts/eval_exposure.py` | Memorization dose-response: 24 invented codes repeated 1 to 32 times, exposure among 1,000 candidates and extraction |
| `scripts/baseline_retrieval.py` | Retrieval baseline: my reply to the most similar conversation, no model |
| `scripts/eval_context.py` | Same replies with 1 to 20 turns of context |
| `scripts/generate.py` | One reply per test conversation, for the style and relevance tests; `--few-shot K` puts K of my replies in the prompt, `--retrieve K` my replies to the K most similar conversations |
| `scripts/eval_style.py` | Frozen classifiers, mine against my friends', applied to generated replies |
| `scripts/eval_relevance.py` | NLL gain from the conversation, scored by a judge model that generated none of the replies (`--scorer`) |
| `scripts/eval_by_chat.py` | The main intervals again, resampling whole chats instead of sessions |
| `scripts/eval_repeats.py` | The signal without repeated replies (catchphrases) |
| `pipeline/build_speaker.py` | Datasets for the cross-speaker control: one chat, an adapter per person |
| `scripts/compare.py` | Paired difference between any two per-example results, with a CI by session |
| `scripts/plots.py` | Every figure, from `experiments/results/*.json` only |
| `scripts/chat.py` | Talk to an adapter in the terminal |

# scripts

Everything after the datasets are built (`pipeline/` builds them). Run from the repo root with `uv run python scripts/<name>.py --help`; every script prints numbers only, never message text, and the slow ones resume where they stopped.

| Script | What it does |
|---|---|
| `run.py` | One command for your own chats: data, training, checkpoint choice and the signal ([docs/usage.md](../docs/usage.md)) |
| `chat.py` | Talk to an adapter in the terminal |
| `train.py` | The training recipe (mlx-lm LoRA) on any dataset folder; logs go to `logs/` |
| `eval_checkpoints.py` | Validation loss of saved checkpoints, per example, resumable |
| `eval_loss.py` | Loss and perplexity of a model or checkpoint on one file |
| `eval_signal.py` | Perplexity drop on my replies against my friends', with the bootstrap interval |
| `eval_by_chat.py` | The main intervals again, resampling whole chats |
| `eval_repeats.py` | The signal without repeated replies |
| `eval_context.py` | The same replies with 1 to 20 turns of context |
| `eval_canary.py` | One planted secret against 30 invented codes |
| `build_canaries.py`, `eval_exposure.py` | The memorization study: 24 codes, exposure and extraction |
| `generate.py` | One reply per test conversation (adapter, base, few-shot, retrieval in the prompt) |
| `baseline_retrieval.py` | My reply to the most similar conversation, no model |
| `eval_style.py` | Frozen classifiers, mine against my friends', applied to generated replies |
| `eval_relevance.py` | How much the conversation helps a judge model predict each reply |
| `compare.py` | Paired difference between two per-example results, by session |
| `plots.py` | Every figure in `docs/figures/`, from `experiments/results/` only |
| `reproduce.sh` | The whole study in order |
| `train_datacurve.sh` | The data-curve trainings (25% and 50%) |
| `_setup.py` | Imported first by every script: puts the repo root and `pipeline/` on the path |

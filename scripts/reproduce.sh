#!/usr/bin/env bash
# The whole pipeline, in order, from a WhatsApp database to every number in the README.
# Read it before running it: training and the style/relevance evaluations take hours on an M1.
# Every step prints counts and metrics only, never message text.
set -euo pipefail
cd "$(dirname "$0")/.."
run() { echo; echo "== $*"; uv run python "$@"; }

# 1. Data (private, stays in data/). Put ChatStorage.sqlite in data/raw/<YYYY-MM-DD>/ first (README).
run pipeline/extract_data.py                       # one-to-one text messages -> data/processed/messages.jsonl
run pipeline/report.py                             # counts to choose the thresholds
if [ ! -f data/names.txt ]; then
  run pipeline/clean.py --write-name-candidates    # contact names -> data/names_candidates.txt
  echo "Curate data/names_candidates.txt by hand (drop words that are not names), save it as data/names.txt, rerun."
  exit 1
fi
run pipeline/clean.py                              # scrub, sessions (3 h), turns (5 min) -> data/processed/turns.jsonl
run pipeline/build_dataset.py                      # session/, by_message/, canary -> data/dataset/
run pipeline/build_context.py                      # valid with 1-20 turns of context
run pipeline/build_context.py --split-window 20    # session_ctx20/: test contexts for generation
run pipeline/build_subsets.py                      # session_25/, session_50/ for the data curve
run pipeline/build_style.py                        # data/style/: mine vs friends for the classifier

# 2. Train (decisions 15-17): 5 epochs, a checkpoint every 100 steps.
run scripts/train.py --data data/dataset/session --name run1_5ep

# 3. Pick the checkpoint on the full valid set (early stopping).
run scripts/eval_checkpoints.py adapters/run1_5ep --data data/dataset/session/valid.jsonl \
    --out experiments/results/checkpoints_run1_5ep.json
STEP=$(uv run python -c "import json; print(json.load(open('experiments/results/checkpoints_run1_5ep.json'))['best']['step'])")

# 4. Evaluate the chosen checkpoint.
run scripts/eval_signal.py --adapter adapters/run1_5ep --step "$STEP"
run scripts/eval_canary.py --adapter adapters/run1_5ep --step "$STEP"
run scripts/eval_context.py --adapter adapters/run1_5ep --step "$STEP"
run scripts/generate.py --name adapter_t1 --adapter adapters/run1_5ep --step "$STEP"
run scripts/generate.py --name base_t1
run scripts/eval_style.py --adapter-gens adapter_t1 --base-gens base_t1 --transformer
run scripts/eval_relevance.py --adapter-gens adapter_t1 --base-gens base_t1

# 5. The later experiments are notebooks that call these same scripts and resume where they stopped.
#    Each has its prediction written before running; run them headless, in this order (later ones reuse
#    earlier results). Those marked NAS ran on a CUDA GPU with TLM_PY set; see each notebook's header.
for nb in Length Baselines Memorization LoraRank ModelSize Memorization1.5B AllLayers AllLayersSeeds Seeds AdapterRetrieval; do
  echo; echo "== experiments/$nb.ipynb"
  (cd experiments && uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 "$nb.ipynb")
done

# 6. Figures, from experiments/results only.
run scripts/plots.py

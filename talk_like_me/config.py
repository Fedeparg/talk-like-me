"""Paths and the settings every experiment shares. The values are the ones the decisions in DECISIONS.md chose."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"                     # private, never in git
DATASET = DATA / "dataset"               # pipeline/build_dataset.py, build_context.py, build_subsets.py
ADAPTERS = ROOT / "adapters"             # LoRA weights, private, never in git
RESULTS = ROOT / "experiments" / "results"  # aggregated numbers only: safe to commit

SEED = 0
MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"   # decision 5
SCORER = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"  # relevance judge, decision 20

# Training recipe (decisions 15-17): lr 1e-5, batches of 4, LoRA on 16 layers, early stopping on valid.
BATCH = 4
LEARNING_RATE = 1e-5
NUM_LAYERS = 16
BEST_ADAPTER = ADAPTERS / "run1_5ep"
BEST_STEP = 2500

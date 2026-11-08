"""Imported first by every script: puts the repo root and pipeline/ on sys.path so `talk_like_me` and the
pipeline modules (build_dataset, clean...) import when a script is run as a file (uv run python scripts/x.py)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT))

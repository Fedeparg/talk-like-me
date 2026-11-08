"""Small file helpers."""

import json
import os
from pathlib import Path


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj, force=True):
    """Write obj as JSON. With force=False, refuse to replace an existing file."""
    path = Path(path)
    if path.exists() and not force:
        raise SystemExit(f"{path} exists; pass --force to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write a new file and move it into place: never rewrites the existing one in place, which
    # matters when it is a hard link (nasrun's job snapshots) and leaves no half-written file.
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)
    return path

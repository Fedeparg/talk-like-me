"""Train a LoRA adapter with the recipe the experiments settled on (decisions 15-17).

Wraps `mlx_lm.lora` so the settings live in one place. Trains up to --epochs, saves a
checkpoint every 100 steps; pick the best one afterwards with eval_checkpoints.py
(early stopping on the full valid set, measured with nll_ejemplo).

    uv run python scripts/train.py --data data/dataset/session --name run1_5ep
    uv run python scripts/train.py --data data/dataset/session --name try --epochs 1 --dry-run
"""

import _setup  # noqa: F401
import argparse
import math
import shlex
import subprocess
import sys
from pathlib import Path

from talk_like_me.config import ADAPTERS, BATCH, LEARNING_RATE, MODEL, NUM_LAYERS, ROOT, SEED


def count_lines(path):
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def mlx_command(data, adapter, iters, model=MODEL, batch_size=BATCH, learning_rate=LEARNING_RATE,
                num_layers=NUM_LAYERS, save_every=100, seed=SEED, grad_accum=1, config=None, grad_checkpoint=False):
    """The mlx_lm.lora command line for this recipe."""
    return [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", model, "--train", "--data", str(data),
        "--mask-prompt",  # learn the reply only, not the context
        "--adapter-path", str(adapter),
        "--iters", str(iters), "--batch-size", str(batch_size),
        "--learning-rate", str(learning_rate), "--num-layers", str(num_layers),
        "--max-seq-length", "2048", "--save-every", str(save_every),
        # mlx's own validation counts a padding token (mlx-lm#2029) and uses 25 batches only:
        # it is a rough live signal; the real one comes from measuring the checkpoints afterwards.
        "--steps-per-eval", str(save_every), "--val-batches", "25",
        "--seed", str(seed),  # LoRA initialisation and data order
    ] + (["--grad-accumulation-steps", str(grad_accum)] if grad_accum > 1 else []) + (["-c", str(config)] if config else []) + (["--grad-checkpoint"] if grad_checkpoint else [])


def run_logged(cmd, log):
    """Run cmd, showing its output and keeping it in `log` (like tee). Returns the exit code."""
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
    return proc.wait()


def rename_checkpoints(adapter, k):
    """0000200_adapters.safetensors after 200 micro-batches of accumulation 2 is optimizer step 100."""
    for f in sorted(adapter.glob("[0-9]" * 7 + "_adapters.safetensors")):
        f.rename(adapter / f"{int(f.name[:7]) // k:07d}_adapters.safetensors")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", required=True, help="folder with train/valid/test.jsonl in chat format")
    p.add_argument("--name", required=True, help="adapter folder name under adapters/")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--epochs", type=float, default=5, help="ceiling; the best checkpoint is chosen on valid")
    p.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    p.add_argument("--batch-size", type=int, default=BATCH)
    p.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--rank", type=int, default=8, help="LoRA rank (mlx-lm default 8)")
    p.add_argument("--scale", type=float, default=20.0, help="LoRA scale (mlx-lm default 20); alpha/r style: 160 / rank")
    p.add_argument("--grad-accum", type=int, default=1,
                   help="split each batch into K micro-batches (for small GPUs); same effective batch and step numbering")
    p.add_argument("--grad-checkpoint", action="store_true", help="recompute activations in backward: less memory, same result")
    p.add_argument("--iters", type=int, help="steps; overrides --epochs (e.g. stop early when the curve is known)")
    p.add_argument("--dry-run", action="store_true", help="print the command and exit")
    args = p.parse_args()

    n = count_lines(Path(args.data) / "train.jsonl")
    iters = args.iters or math.ceil(args.epochs * n / args.batch_size)
    k = args.grad_accum
    if args.batch_size % k:
        raise SystemExit("--batch-size must be divisible by --grad-accum")
    # mlx-lm counts micro-batches as iterations: scale them so steps mean optimizer steps.
    config = None
    if args.rank != 8 or args.scale != 20.0:  # mlx-lm only takes the LoRA shape from a YAML config
        config = ADAPTERS / f"{args.name}.lora.yaml"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(f"lora_parameters:\n  rank: {args.rank}\n  scale: {args.scale}\n  dropout: 0.0\n")
    cmd = mlx_command(args.data, ADAPTERS / args.name, iters * k, args.model, args.batch_size // k,
                      args.learning_rate, args.num_layers, args.save_every * k, args.seed, k, config, args.grad_checkpoint)
    print(f"{n} train examples, batch {args.batch_size}: {iters} steps ({iters * args.batch_size / n:.2f} epochs)")
    print(shlex.join(cmd))
    if args.dry_run:
        return
    code = run_logged(cmd, ROOT / "logs" / f"adapters_{args.name}.log")
    if code == 0 and k > 1:
        rename_checkpoints(ADAPTERS / args.name, k)
    sys.exit(code)


if __name__ == "__main__":
    main()

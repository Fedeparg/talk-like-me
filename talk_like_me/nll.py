"""The per-reply loss used for every number in this repo, and helpers to load models and apply it.

Why not `mlx_lm.lora --test`: mlx-lm 0.32.0 counts the first padding token after each reply in
the loss, which favours a fine-tuned model over the base one (decision 12, ml-explore/mlx-lm#2029).
`nll_ejemplo` counts the reply tokens only.
"""

import os

import mlx.core as mx
from mlx_lm import load

from .config import MODEL
from .io import read_jsonl


# Freeing MLX's buffer cache after every reply keeps memory low on the Mac (unified memory), but on a
# dedicated GPU it only forces new allocations. TLM_KEEP_CACHE=1 frees it every 200 replies instead.
# It changes memory use and speed, never the numbers.
_KEEP_CACHE = os.environ.get("TLM_KEEP_CACHE") == "1"
if _KEEP_CACHE:
    mx.set_cache_limit(3 * 1024**3)  # cap the reusable cache at 3 GB so it can't fill a 10 GB GPU
_calls = 0


def _maybe_clear_cache():
    global _calls
    _calls += 1
    if not _KEEP_CACHE or _calls % 200 == 0:
        mx.clear_cache()


# Written in experiments/Perplexity.ipynb and moved here; the only change is _maybe_clear_cache().
def nll_ejemplo(model, tokenizer, messages):
    full = tokenizer.apply_chat_template(messages)
    base_prompt = tokenizer.apply_chat_template(
        messages[:-1], add_generation_prompt=True)
    n_tokens = len(full) - len(base_prompt)
    logits = model(mx.array(full)[None])
    logits = logits[0, len(base_prompt)-1:len(full)-1]
    logits = logits.astype(mx.float32)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    nll = 0
    for i in range(n_tokens):
        nll -= logprobs[i, full[len(base_prompt) + i]].item()
    _maybe_clear_cache()
    return (nll, n_tokens)


def load_model(adapter=None, step=None, model=MODEL):
    """The base model, or base + LoRA adapter; with `step`, that checkpoint's weights instead of the last ones."""
    m, tokenizer = load(model, adapter_path=str(adapter) if adapter else None)
    if step is not None:
        load_checkpoint(m, adapter, step)
    return m, tokenizer


def load_checkpoint(model, adapter, step):
    """Swap in the weights mlx_lm.lora saved at `step` (--save-every), over an already loaded adapter."""
    model.load_weights(f"{adapter}/{step:07d}_adapters.safetensors", strict=False)


def measure(model, tokenizer, path, limit=None):
    """[nll_sum, n_tokens] for every example of a chat-format .jsonl, in file order."""
    examples = read_jsonl(path)[:limit]
    return [list(nll_ejemplo(model, tokenizer, e["messages"])) for e in examples]


def mean_loss(results):
    """Loss per reply token over a whole set: total nll / total tokens (not a mean of per-example means)."""
    return sum(r[0] for r in results) / sum(r[1] for r in results)

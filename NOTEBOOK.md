# Lab notebook

Numbers only: no message text, no names. Before each run, write the prediction; after it, the result and why they differ.

## Runs

### 2026-10-15, run 0: the base model as it comes

```bash
uv run mlx_lm.generate --model mlx-community/Qwen2.5-0.5B-Instruct-4bit --prompt "¿Qué planes tienes para el finde?"
```

| Prompt | Generation | Peak memory |
|---|---|---|
| 37 tokens, 22,200 tokens/s | 100 tokens, 197.5 tokens/s | 0.369 GB |

- Weights on disk: `model.safetensors`, 278 MB. 494M parameters at 4 bits plus one fp16 scale and one fp16 bias per group of 64 weights is about 4.5 bits per parameter: 494M x 4.5 / 8 = 278 MB.
- Peak memory minus weights (0.369 - 0.278 GB) is the KV cache, the activations and the runtime.
- To explain: why the prompt runs about 100 times faster than generation (prefill against decode). Prediction for the 1.5B, before running it: generation tokens/s and peak memory.

### 2026-10-15, run 1: the 1.5B, same prompt

```bash
uv run mlx_lm.generate --model mlx-community/Qwen2.5-1.5B-Instruct-4bit --prompt "¿Qué planes tienes para el finde?"
```

| Prompt | Generation | Peak memory |
|---|---|---|
| 37 tokens, 104.0 tokens/s | 49 tokens, 138.1 tokens/s | 1.023 GB |

**Prediction, made before the run.** Decoding is memory-bandwidth bound: every new token reads all the weights once.
- Bytes per token: 1,540M parameters x 4.5 bits / 8 = 0.87 GB. On disk: `model.safetensors` is 869 MB, which checks out.
- Ceiling: 200 GB/s (M1 Pro bandwidth) / 0.87 GB = 230 tokens/s.
- With the 0.5B's efficiency: 0.5B ceiling 200 / 0.278 = 720 tokens/s, measured 197.5, so 27%. Then 230 x 0.27 = 62 tokens/s, and maybe up to 100 because fixed costs weigh less in a bigger model.
- Direction: more parameters means fewer tokens per second, never more.

**Result:** 138 tokens/s, above the prediction (60% of the ceiling, not 27%).

**Why the miss: a fixed cost per token.** Model the time per token as weight bytes / bandwidth + a fixed cost c (kernel launches, sampling, Python).
- 0.5B: 1 / 197.5 = 5.06 ms per token; reading the weights takes 0.278 / 200 = 1.39 ms; so c = 3.7 ms.
- 1.5B with the same c: 0.87 / 200 + 3.7 ms = 8.05 ms, that is 124 tokens/s, close to the 138 measured.
- So the small model wastes most of each step on fixed costs. Scaling the efficiency percentage was the wrong model; adding a constant is the right one.

**Caveats:** 49 tokens is a short run (it stopped at the end-of-sequence token). The prompt rate (104 tokens/s, lower than generation) includes first-call warm-up, so it is not a real prefill number: measure prefill with a prompt of about 1,000 tokens.

### 2026-10-19, run 2: first LoRA training, mlx-lm defaults (`adapters/run1`)

```bash
uv run mlx_lm.lora --model mlx-community/Qwen2.5-0.5B-Instruct-4bit --train --data data/dataset/session --mask-prompt --adapter-path adapters/run1
```

Defaults: rank 8, scale 20, 16 layers (every linear layer of the last 16 blocks), batch 4, 1,000 iterations (4,000 examples, less than one pass over the 5,404), learning rate 1e-5.

**Predictions (Fede, before the run):**
- Peak memory: under 3 GB.
- Validation loss: drops 5% at most from the first reading (the base model). Why: it does not see every example once, and 1e-5 is about 10 times lower than what is usually recommended for LoRA.
- Note on the memory reasoning: the base weights are 278 MB, not 1.7 GB; what grows is the logits (batch x length x 151,936) and the activations. See the calculator from 2026-10-19 (about 2.6 GB at batch 4 and 256 tokens).

**Result:**

| Iteration | 1 (base) | 200 | 400 | 600 | 800 | 1000 |
|---|---|---|---|---|---|---|
| Validation loss | 5.475 | 3.375 | 3.395 | 3.218 | 3.216 | 3.170 |

- Validation loss fell 42% (perplexity from e^5.475 = 239 to e^3.170 = 24), most of it in the first 200 steps. The prediction (5% at most) was wrong by a factor of eight.
- Trainable parameters: 2.933M (0.594%), exactly the count from r x (d_in + d_out) over the 7 linear layers of 16 blocks. The adapter file is 11.75 MB: 2.933M x 4 bytes, fp32.
- Time: about 10 minutes (14:27 to 14:37). Train loss per 10 steps jumps between 2.8 and 3.4: batches of 4 are noisy.
- Peak memory: this mlx-lm version did not print it. Next run: wrap the command in `/usr/bin/time -l` and read "peak memory footprint".
- Validation here is only 25 batches (100 examples), so the 3.395 at step 400 is noise, not overfitting.
- These are mlx-lm's own numbers, which count a padding token in the loss (found on 2 Nov, decision 12). From then on every loss is measured with `nll_ejemplo`, and the table below uses those.

**Why the miss:** the base model writes like an assistant (long, formal, punctuated); the first steps only have to learn the register (short, lowercase, WhatsApp Spanish), which is one simple, consistent shift. A low learning rate slows fine adjustments, not a big uniform one. To check: does the plateau after step 200 move with learning rate 1e-4?

## Experiments

Every run after run1, with the prediction written before it. Losses are `nll_ejemplo` per reply token: best validation (6 turns of context unless noted) and its step. Details and decisions in `DECISIONS.md`.

| Run | What changed | Prediction before running | Result | Prediction held? |
|---|---|---|---|---|
| `run1` | mlx-lm defaults, 1,000 steps (0.74 epochs) | validation drops 5% at most | test loss 3.421 (perplexity 30.6) | No: it dropped 42% (see above) |
| `by_message` | split by messages instead of sessions | about 2.8 on its own test, against 3.42 | 3.405 on its own test, 3.423 on the session test; at 1.9 epochs 3.353 and 3.381 | No: small (0.018, then 0.028 at 1.9 epochs, on different test sets and without a bootstrap) |
| `run1_5ep` | 5 epochs, checkpoint every 100 steps, early stopping | | 3.418 at step 2,500 (1.9 epochs) | Chosen adapter |
| `run2_lr1e-4` | learning rate 1e-4 | | 3.586 at step 1,200 | |
| `run3_lr3e-6` | learning rate 3e-6 | clearly below 3.418 | 3.433 at step 5,000 | No |
| `run1_ctx20` | 20 turns of context in training | beats the 6-turn adapter | 3.386 at step 2,600 (20-turn valid), -0.008 vs the 6-turn adapter given 20 turns, CI crosses 0 | No |
| `data25`, `data50` | 25% and 50% of the training sessions | the curve flattens (capacity is the limit) | 3.539 and 3.474 (20-turn valid), against 3.394 for 100% | No: data is the limit |
| `run3_lr3e-6_cont` | 3e-6, up to 3 more epochs from its last step (6,755), stop when worse than the start | it keeps going down slowly | 3.441 at the start; 3.482, 3.476, 3.505 after 500, 1,000, 1,500 steps. Stopped | No (Adam state restarts at the resume) |
| `run1_1.5b` | the same recipe on Qwen2.5-1.5B (`Baselines.ipynb`) | a bigger model answers better and keeps my style | 3.150 at step 2,500; style position 1.08 / 0.94, relevance 0.33 against 0.27 for the 0.5B (paired +0.026, CI crosses 0) | Half: style yes, answers not demonstrably |
| `mem_canaries` | 24 invented codes repeated 1 to 32 times, server | from 4 repetitions top 10 of 1,001; 1 and 2 at chance; written out only at 16 to 32 | from 4 every code first; with 2, three of four in the top 2; written out 3 of 4 at 32 | Mostly; wrong on 1 and 2 repetitions |
| `rank4` to `rank32`, `*_a160` | LoRA rank 4 to 32, fixed scale and scale 160 / rank, server | a bigger adapter doesn't help (threshold 0.02 per doubling) | 3.398 to 3.422 over 8 times the parameters | Yes |
| `size_0.5b`, `size_1.5b`, `size_3b` | base model size, server | the 3B improves a bit less than the 1.5B did; both my loss and my friends' improve | 3.424, 3.147, 3.018; signal in nats 0.43, 0.45, 0.45 | Yes |
| `mem_canaries_1.5b` | the memorization study on the 1.5B | it memorizes sooner | exposure not higher; half of the codes written out at 16 repetitions (step 2,500) or 8 (step 3,500), against 32 for the 0.5B | Half |
| `alllayers_0.5b`, `alllayers_1.5b` | LoRA on every layer, server | unsure; if data is the limit, within ±0.02 and an earlier minimum | -0.043 and -0.039, same step, evenly across reply lengths | No: neither reading |
| `size_0.5b_seed1,2`, `alllayers_0.5b_seed1,2` | the two arms above with seeds 1 and 2 | seeds land close; the gain holds | gain -0.043, -0.037, -0.036; largest spread in an arm 0.023 | Mostly (spread just above 0.02) |
| `run1_seed0,1,2`, `data25_seed1,2`, `data50_seed1,2` | seeds of the final recipe and the data curve, Mac | results consistent across seeds | sd 0.001 to 0.004; signal 19.8, 18.4, 19.6 | Yes |
| (no training) `adapter_rag3` | the final adapter with my replies to the 3 most similar conversations in its prompt | relevance 0.45 to 0.50, no copying | 0.34 (paired +0.027 against the adapter, CI crosses 0); 0 copies | Half |
| `speaker_c36_me`, `speaker_c36_other` | one chat, one adapter per person, neutral prompt, server | mine similar to the main signal but smaller; theirs similar to mine | 0.377 and 0.806 nats; both learn their own person | Half: both positive, but mine not smaller and theirs twice as large |
| (no training) `fewshot20_*_short`, `*_style` | few-shot 0.5B, 1.5B and 3B with a 64-token cap and an instruction to write short (and informal) | style goes up but doesn't reach the adapter | style barely moves (3B goes down); the short 3B's relevance is above my real replies (1.27) | Fede yes, Claude no; relevance unexpected |
| (no training) Llama-3.1-8B as judge | the same replies of every set, a judge of another family | the short 3B stays above 1, somewhat less | 1.01; the adapters fall to the floor (0.5B: -0.00) | Half |
| (no training) bootstrap by chat, repeated replies removed | intervals resampling chats; signal without catchphrases | signal survives, intervals wider but not double; signal drops but survives | by chat [8.9, 26.0]; without repeats 19.6 | Yes, though it barely drops |

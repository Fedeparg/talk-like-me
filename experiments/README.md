# Experiments

Every notebook here is where a result was measured, with its outputs and, at the top, the prediction written before running it. They are grouped by the question they answer and listed in reading order. Numbers in [`DECISIONS.md`](../DECISIONS.md) and [`docs/results.md`](../docs/results.md) point back to them. `results/` holds the aggregated numbers they write (losses, counts, ranks, session ids; no text).

**Who wrote what.** The prediction at the top of each notebook is mine, written before running (translated to English where I wrote it in Spanish). The method descriptions and conclusions of the later notebooks were drafted by Claude and checked by me against the outputs.

**Where they ran.** "Mac" is the M1 Pro (Metal); "server" is a home GPU server with an RTX 3080 (MLX's CUDA backend), where trainings use micro-batches with gradient accumulation to the same effective batch of 4. Within one notebook every training runs on the same machine; evaluations give the same numbers on both (to 0.0003 nats).

**Start here:** [Perplexity](Perplexity.ipynb) (the signal), [CrossSpeaker](CrossSpeaker.ipynb) (is it really me), [StyleClassifier](StyleClassifier.ipynb) (does it sound like me) and [SecondJudge](SecondJudge.ipynb) (does it answer). Older notebooks keep their first numbers next to the re-measured ones; where they differ, `docs/results.md` has the final ones.

## 1. Measuring the right thing

| Notebook | Question | Result | Where |
|---|---|---|---|
| [mlx-lm loss padding](mlx-lm%20loss%20padding.ipynb) | Does mlx-lm's loss count a padding token? | Yes: one extra `!` per reply; fixing the mask gives exactly the hand-computed loss (3.4213), and the model stops learning the `!` | Mac |
| [Perplexity](Perplexity.ipynb) | Did it learn me, or WhatsApp in general? | Perplexity drops 19.8 points more on my replies than on my friends' (bootstrap by session) | Mac |
| [Leakage](Leakage.ipynb) | Does splitting by message instead of by conversation leak? | A little: 0.028 easier on its own test, the same model on the unseen test | Mac |

## 2. Training: what moves the loss

| Notebook | Question | Result | Where |
|---|---|---|---|
| [Checkpoints](Checkpoints.ipynb) | How long to train? | Best validation at step 2,500 (~1.9 epochs), then it memorizes train | Mac |
| [LearningRate](LearningRate.ipynb) | Is the learning rate too large? | No: 3e-6 reaches the same minimum later, 1e-4 is worse | Mac |
| [Context](Context.ipynb) | Does more history help at prediction time? | A little: 20 turns instead of 6, -0.024 | Mac |
| [ContextRetrain](ContextRetrain.ipynb) | Does training with 20 turns help? | No beyond noise (-0.008) | Mac |
| [DataCurve](DataCurve.ipynb) | Is the limit data? | Each doubling of the data lowers the loss by 0.07-0.08 | Mac |
| [Seeds](Seeds.ipynb) | How much is training noise? | sd 0.004 on the final recipe; the step and the signal hold across seeds | Mac (evaluated on the server) |
| [LoraRank](LoraRank.ipynb) | Is the adapter too small? | No: ranks 4 to 32 within 0.024 | server |
| [AllLayers](AllLayers.ipynb) | LoRA on every layer? | Better by ~0.04 on 0.5B and 1.5B | server |
| [AllLayersSeeds](AllLayersSeeds.ipynb) | Is that gain a lucky seed? | No: -0.043, -0.037, -0.036 across three seeds | server |
| [ModelSize](ModelSize.ipynb) | 0.5B, 1.5B, 3B? | The largest lever (-0.28, then -0.13), but the part that is specifically me stays the same | server |

## 3. Is it really me?

| Notebook | Question | Result | Where |
|---|---|---|---|
| [Length](Length.ipynb) | Is the signal just shorter messages? | About 3 of the 19.8 points | Mac (no model) |
| [ByChat](ByChat.ipynb) | Intervals resampling whole chats | The signal's interval widens to [8.9, 26.0]; the rest barely moves | Mac |
| [Repeats](Repeats.ipynb) | Is it catchphrases? | No: without repeated replies the signal stays at 19.0-19.8 | Mac (no model) |
| [CrossSpeaker](CrossSpeaker.ipynb) | Does an adapter trained on a friend learn *them*? | Yes: both adapters drop more on the person they were trained on | server |

## 4. Does it sound like me, and does it answer?

| Notebook | Question | Result | Where |
|---|---|---|---|
| [StyleClassifier](StyleClassifier.ipynb) | Does it sound like me? | Mostly; "more me than me" at temperature 0.7 was the sampler | Mac |
| [Relevance](Relevance.ipynb) | Does it answer the conversation? | First measurement, with a 1.5B judge; superseded by Baselines and SecondJudge | Mac |
| [Baselines](Baselines.ipynb) | Is fine-tuning worth it against prompting, retrieval and a bigger model? | Style: training wins. Content: nothing demonstrably better (Qwen2.5-7B judge) | Mac |
| [AdapterRetrieval](AdapterRetrieval.ipynb) | Adapter + my replies to similar conversations? | Same style, no better content | server and Mac |
| [FairFewShot](FairFewShot.ipynb) | A fairer prompt with a length cap? | Style still far; the short 3B matches or beats my real replies on relevance | server |
| [SecondJudge](SecondJudge.ipynb) | Does relevance depend on the judge? | Partly: with Llama-3.1-8B the adapters sit at the floor | server |

## 5. Memorization

| Notebook | Question | Result | Where |
|---|---|---|---|
| [Canaries](Canaries.ipynb) | Does it remember a planted secret? | It starts to: first of 31 after ~9 exposures | Mac |
| [Memorization](Memorization.ipynb) | How many repetitions until it remembers? | From 4 every code ranks first of 1,001; written out at ~32 | server |
| [Memorization1.5B](Memorization1.5B.ipynb) | Does a bigger model memorize sooner? | It recognizes at the same pace but writes codes out with about one doubling fewer repetitions | server |

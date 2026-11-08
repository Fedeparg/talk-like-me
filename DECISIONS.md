# Decisions, step by step

A lab log written as the project went: what was decided at each step, why, and what was discarded. Entries are kept as they were written on their date, with a status line where a pending item was later settled; the results in their final form are in [`docs/results.md`](docs/results.md). Every number points to the notebook (`experiments/*.ipynb`) and, where there is one, the JSON (`experiments/results/`) it comes from. Aggregated numbers and invented examples only: never real messages or names. Hardware and speed calculations are in `NOTEBOOK.md`.

**Format of each entry:** the decision; why; what was discarded; the numbers that back it.

---

## 1. The project: a small model that writes like me (2026-10-15)

- **Decision:** fine-tune a small language model with LoRA on my WhatsApp replies and measure objectively whether it writes like me without leaking private data.
- **Why:** one project in depth instead of five topics on the surface. It covers fine-tuning, evaluation and inference, and what I learn I can take to any job.
- **Discarded:** going over every topic at once; subjective metrics ("sounds like me"), replaced by six measurable metrics.

## 2. Where the project lives (2026-10-15)

- **Decision:** `~/dev/talk-like-me/`, its own git repository, with `data/` and the adapters outside git. Since 2026-10-17, the code and notes are backed up in a private GitHub repository; the data never leaves the Mac. When the project is finished, a reviewed public version.
- **Why:** the chats cannot end up on any server because of a careless `git add`; the path shows on screen when recording; the project outlives any hiring process.
- **Discarded:** putting it inside another repository that does get pushed to GitHub.
- **Status (7 Nov):** the data now also goes to a GPU server at home (decision 24). It still never leaves my home network.

## 3. Where the data comes from (2026-10-15)

- **Decision:** read the database that the WhatsApp Mac app keeps locally (`ChatStorage.sqlite`, SQLite), copied with the app closed and always opened read-only.
- **Why:** WhatsApp only exports one chat at a time, and the database has every chat in one file.
- **Discarded:** exporting chat by chat (slow, and a text format that has to be parsed); the iPhone backup (full history, but hours and a lot of disk), which stays as plan B.
- **Numbers:** from 9 June to 15 October 2026, about four months, which is what the Mac received when it was linked. 32 one-to-one chats, 30,751 text messages, 15,292 mine. My 10 most active chats hold 94% of my messages.
- **Privacy:** Claude is an external API, so the rule is that it never reads the messages: it only runs queries that return counts and dates. (A rule I followed, not something the code enforces; see `docs/privacy.md`.)

## 4. The environment (2026-10-15)

- **Decision:** `uv` with Python 3.13, pinned in `.python-version`, and MLX with mlx-lm.
- **Why:** MLX is Apple's library for its own hardware, and the Mac is my only machine.
- **Discarded:** Python 3.14, because a single missing compiled package would cost a day.
- **Status (7 Nov):** a second machine was added for experiments: a home server with an RTX 3080, running MLX's CUDA backend (decision 24).

## 5. The base model (2026-10-15)

- **Decision:** `mlx-community/Qwen2.5-0.5B-Instruct-4bit`, and the 1.5B as a second step.
- **Why:** small and fast to iterate. Since it is quantized to 4 bits, training LoRA on top is QLoRA.
- **Discarded:** Qwen3, whose chat template adds reasoning blocks that complicate masking the loss.
- **Numbers:** weights of 278 MB for the 0.5B and 869 MB for the 1.5B. Generation at 197 and 138 tokens/s. The miss in my 1.5B prediction and its explanation (a fixed cost per token) are in `NOTEBOOK.md`.
- **Status (7 Nov):** the 1.5B and the 3B were trained with the same recipe (decisions 22 and 26).

## 6. Which messages are extracted (2026-10-15)

- **Decision:** every text message from the one-to-one chats, mine and the other person's, ordered by chat and date, in `data/processed/messages.jsonl` (`extract_data.py`, written by me).
- **Why:** each example is "what they said to me" (context) plus "what I replied" (response), so without the other people's messages there is no context. They also serve as a control for perplexity and as negatives for the style classifier.
- **Discarded:** groups (776 messages, from people who have not given their permission, and there I don't write like in a one-to-one chat); the channel; messages that are not text.
- **Finding:** almost every chat already uses WhatsApp's new identifier (`@lid`, 30 chats) instead of the phone number (`@s.whatsapp.net`, 2 chats with 10 messages in total).

## 7. How train, validation and test are split (2026-10-15)

- **Decision:** cut each chat into sessions and assign whole sessions at random: 80% train, 10% validation, 10% test, with a fixed seed.
- **Why:** each example carries the previous turns as context. If single messages are assigned at random, the test response shows up in the context of the next train example, and that afternoon's conversation falls on both sides: it is data leakage.
- **Discarded:**
  - The split by time, because it is four months of one same style and there is no deployment.
  - The random split by messages, because of the leakage.
  - Holding out whole chats: with 32 very uneven chats, the test would be one big chat that empties training or a few small ones with a worthless number. It also measures something else: whether it generalizes to new people.
- **Experiments that come out of this:** also train with the split by messages and measure the leakage instead of assuming it; later, hold out whole chats to measure whether it generalizes to new people.

## 8. What gets cleaned and how (2026-10-17)

- **Decision:** replace instead of delete. URLs, emails, phone numbers and names are swapped for placeholders (`<URL>`, `<EMAIL>`, `<PHONE>`, `<NAME>`). My turns that end up being only placeholders are dropped as responses, but kept as context.
- **Why:** with `--mask-prompt` the model only learns from the response tokens, so noise in the context is not learned to be generated. If a message is deleted, the response that comes after it is left without context.
- **Discarded:** deleting every message that starts with `http`, because it misses links in the middle of the text and leaves responses without context.
- **Numbers (`report.py`):** 95 of my messages are just a link and 15 are a link with text, out of 15,292. 13.9% of my messages have 5 characters or fewer, and my 30 most repeated texts are 8.7%: they stay because they are my style, but they are a trap for the evaluation.
- **Names, curated on 2026-10-17:** `data/names.txt` comes from the contact names, curated by hand (71 words). A check against common words found `del`, which came from a contact name and replaced every "del" in the messages: messages with `<NAME>` went from 672 to 217 after removing it, so 455 were false. Lesson: an anonymization list is also checked with numbers. One common word stays in the list on purpose, because it is also a friend's nickname.
- **Excluded chats:** the chat with Meta AI, a bot (1 message).
- **Pending:** look at the numbers that the report's loose pattern counted as possible phone numbers (34, against 1 with the strict one). The long messages are already reviewed: they were mine (decision 11).

## 9. Sessions and turns (2026-10-17)

- **Decision:** a new session starts after more than 3 hours of silence in a chat. A turn joins consecutive messages from the same person with less than 5 minutes between them.
- **Why, the session:** the median gap between messages is 12 seconds, and 99% of the gaps are under 12 hours. With 1 hour, conversations get split by a lunch; with 3 hours there are 759 sessions, about 76 for test.
- **Why, the turn:** the decision is robust, because with 1 minute there are 8,345 turns of mine and with 10 minutes, 7,957. Several consecutive messages from the same person are a single intervention; if more than 5 minutes go by, it is usually another topic. Chat templates expect `user` and `assistant` to alternate.
- **Numbers (`clean.py`):** 15,851 turns, 8,059 mine; 29 of my turns are only placeholders.

## 10. How I work with Claude (2026-10-17)

- **Decision:** the decisions are mine (what to clean, which threshold, which split) and I make them with numbers in front of me. Claude writes the plumbing (`report.py`, `clean.py` and their tests), I read it line by line and answer its questions about the code. I write the ML core: the extraction query and the per-reply loss (`nll_ejemplo`), and I decide and check every metric. LoRA itself comes from mlx-lm. In November Claude turned the notebooks into scripts (`talk_like_me/`, `scripts/`), with `nll_ejemplo` moved unchanged.
- **Why:** knowing what to do and why is what counts; the how with Claude Code is how work gets done today. In an interview I have to be able to defend every line, also the ones I didn't type.
- **What I don't let it do:** read my chats; decide for me; write what I need to be able to rebuild on a whiteboard.

## 11. The training examples (2026-10-17)

- **Decision (`build_dataset.py`):** each example carries the system prompt "Eres Fede. Responde como él en WhatsApp.", up to 6 previous turns from the same session as context and my turn as the response, in mlx-lm's chat format. My turns that open a session (no context), the ones that are only placeholders and the ones with no turn from the other person in the context are dropped. The canary is planted at the end of 5 train responses.
- **Why the system prompt:** if it is not set, the Qwen2.5 template adds its own default one. The rule is to use the same one in training, evaluation and inference; with `--mask-prompt` it costs nothing.
- **Why `test_friends`:** it is the control group. If perplexity drops the same on my responses as on my friends', the model learned "WhatsApp in Spanish", not me; the signal is the difference between the two drops.
- **The leakage experiment, fair:** `by_message/train` is the same size as `session/train` (5,404 examples), the canaries stay in train and no model sees `session/test`.
- **Numbers:** 604 train sessions, 77 validation and 77 test. Examples: 5,404 train, 1,012 validation, 1,133 test and 1,110 in `test_friends`. Dropped: 426 turns that open a session (I open more than half of the conversations), 54 with no turn from the other person and 25 only placeholders. By sessions the split is 80/10/10, but by examples it ends up 72/13/15, because the test sessions are somewhat longer by chance: it is the price of splitting by groups.
- **Tokens:** each example has a median of 148 tokens (p99 402). What the loss sees, the response, has a median of 12 tokens; in total 96,146 response tokens in train. 4 responses go over 256 tokens and 1 over 1,024. The longest example (2,361 tokens) goes over mlx-lm's 2,048 limit and would be truncated.
- **Long responses, decided:** responses over 256 tokens are out as responses (`MAX_REPLY_TOKENS`). There were 4, all in train, and I reviewed them myself: they were mine, but too odd to teach anything, and one of 2,095 tokens would have weighed in the loss like 170 normal responses.

## 12. How the signal is measured and whether it is solid (2026-11-02, re-run 2026-11-07)

- **Decision:** the evaluation loss is computed by hand (`nll_ejemplo`, written in `experiments/Perplexity.ipynb`, now in `talk_like_me/nll.py`), counting only the response tokens, and not with `mlx_lm.lora --test`. How solid the signal is gets measured with a bootstrap of 1000 resamples by session (with replacement, seed 0): in each resample, the loss of each case is the sum of the nll of the chosen sessions divided by the sum of their tokens.
- **Why by hand:** `mlx_lm` 0.32.0 counts in the loss the first padding token after the response (a `!`). That penalizes the base model, which doesn't expect it, and favors the adapted one, which learns to predict it. I reported it in ml-explore/mlx-lm#2029.
- **Why by session:** examples from the same session are not independent (they share the 6 turns of context) and the bootstrap assumes what gets resampled is. The session is the independent unit. My examples and my friends' from the same session go in together.
- **Numbers, final adapter** (`run1_5ep` at step 2,500, decision 15; test, 1,133 responses of mine in 63 sessions and 1,110 from friends; `experiments/Perplexity.ipynb`, `experiments/results/signal_run1_5ep.json`): base 75.5 → 29.6 on mine (60.8% drop); base 81.1 → 47.9 on my friends' (41.0%). Signal: 19.8 points, 95% CI from 14.7 to 25.8. With mlx's loss: base 209.2 → 24.7 on mine, 199.8 → 39.8 on my friends', signal 8.1: the bug hides more than half of the difference.
- **History:** first measured on 2 Nov on `run1` (1,000 steps, 0.74 epochs): 75.5 → 30.6 (59.5%) and 81.1 → 47.0 (42.0%), signal 17.4 points, CI 12.1 to 23.3 (7.4 with mlx's loss). On 7 Nov the notebook was re-run on the final adapter, so that every result describes the same model. Its session labels now come explicitly from the v1 turns (`data/processed_v1_2026-10-17`), the ones `session/` was built from; `data/processed` holds the newer copy since 5 Nov.
- **Conclusion:** the signal is solid. The model hasn't only learned the WhatsApp register; it also learned something of how I talk. (17.4 at 0.74 epochs against 19.8 at 1.9: the intervals overlap, so not demonstrably more.)
- **Caveat:** the control is not a perfect null: my friends are other speakers, scored with the roles swapped under a system prompt that names me, and I haven't ruled out that part of the gap is message length.
- **Status (7 Nov):** length explains about 3 of the 19.8 points (decision 21); the signal holds across three seeds (decision 29) and, measured in nats, across model sizes (decision 26).
- **Training loss:** not changed. Every run keeps mlx's training loss (with the `!`), so runs stay comparable; every evaluation uses `nll_ejemplo`. Hyperparameters are chosen with validation, not with test.

## 13. Canaries: does it give away the secret? (2026-11-05, re-run 2026-11-07)

- **Decision (`experiments/Canaries.ipynb`, `experiments/results/canary_run1_5ep.json`):** memorization is measured with the canary planted in train ("por cierto, mi código del trastero es 4829-QX", in 5 responses, decision 11), on the adapter and on base, on example 69 of `session/train`, in three ways: generate without a lead-in, generate with the text up to "es ", and compare the NLL of the code with that of 30 invented codes of the same format (seed 0).
- **Why a canary:** it is an invented secret that I plant on purpose. If the model gives it away, my real data could come out too. Since it is random, it has no pattern to understand: getting it right can only be memory.
- **Why generating is not enough:** the model can give the code more probability than another one without it becoming its first choice. That is why the NLL is measured.
- **Why the control is invented codes and not base:** the adapter learned the format of my responses, so it always beats base. The clean control is the same adapter with other codes.
- **Numbers, final adapter** (~1.9 epochs, so the canary was seen about 9 times): with no lead-in it doesn't produce the code; with the lead-in it writes a code-shaped `1234-5678`, not the canary. The tail costs 20.12 nats against 38.14 on base. The four digits cost 9.86 (about 2.5 each, close to ln 10 ≈ 2.30, which is guessing; base 11.21); the letters "-QX" cost 8.84 against 17.18 on base; the closing `<|im_end|>` 1.41 against 9.75. Control: the canary 20.12 against a mean of 24.50 for the invented codes (20.20 to 31.50): **rank 1 of 31, p = 0.03**. Base: rank 21 of 31, p = 0.68.
- **History:** first measured on 5 Nov on `run1` (0.74 epochs, about 3.7 exposures): it didn't generate the code ("No se", and with the lead-in a sentence), the tail cost 28.03 against 28.00 on average for the invented codes (24.36 to 32.91), rank 17 of 31, p = 0.55. Most of its gain over base was the `<|im_end|>` (2.54 against 9.75).
- **Conclusion:** with about 9 exposures the adapter has started to memorize the canary, and the part it memorized is the letters, not the digits. Seen about 3.7 times (run1), there was no trace.
- **Limits:** a single context (example 69) and a single canary. With 30 codes, the finest that can be measured is p = 1/31 ≈ 0.03, so the test says that memorization started, not how strong it is.
- **Pending:** a dose-response study: more canaries, with different numbers of repetitions.
- **Status (7 Nov):** done with 24 canaries on the 0.5B (decision 23) and on the 1.5B (decision 27).

## 14. The leakage of the split by messages, measured (2026-11-05, re-run 2026-11-07)

- **Decision (`experiments/Leakage.ipynb`):** train `adapters/by_message` with the same command as run1 changing only the data to `data/dataset/by_message`, and measure it with `nll_ejemplo` in two ways, changing one thing at a time: the same model on both tests (measures the leakage) and both models on the session test, which neither has seen (measures whether the model is better or the exam was easier).
- **Why:** decision 7 discarded the split by messages because of the leakage without measuring it. With `--mask-prompt` the model doesn't learn to write the context, only my responses; what leaks is my neighboring responses from the same afternoon.
- **Hypothesis before measuring:** on its own test, a loss of about 2.8 against 3.42.
- **Numbers, ~1.9 epochs** (both trained 5 epochs, measured at step 2,500, the best step on validation for both, decision 15): by_message on its test (506 examples) 3.353 (ppl 28.6); by_message on the session test 3.381 (ppl 29.4); the final adapter on the session test 3.388 (ppl 29.6).
- **History:** first measured on 5 Nov with 1,000 steps (0.74 epochs): 3.405, 3.423 and 3.421.
- **Conclusion:** the leakage starts to show with more training, but it's small: a gap of 0.028 between by_message's own test and the session test (0.018 at 0.74 epochs), far from the 2.8 I predicted. Without a bootstrap I can't say it's beyond noise; as a reference, the two splits differed by ~0.01-0.02 on validation (decision 16). On the unseen session test the two models are the same, so the split by messages makes an easier exam, not a better model.
- **Untested hypothesis:** a good part of the improvement over base comes from writing shorter messages (the `<|im_end|>` from decision 13 points that way). The friends control group says something of mine is also learned.
- **Status (7 Nov):** tested in decision 21: supported for the improvement over base, but length explains little of the gap between me and my friends.

## 15. How much to train: early stopping with validation (2026-11-05)

- **Decision:** the number of epochs is not fixed in advance. Training goes up to a ceiling (5 epochs, 6,755 steps with batches of 4) saving a checkpoint every 100 steps, and the checkpoint with the lowest validation loss is chosen. Test is measured with the chosen one (decisions 12 to 14 were first measured on run1 and re-run on the chosen one on 7 Nov).
- **Why:** while train and validation go down, the model learns; when they split (train keeps going down and validation stalls or goes up), it starts memorizing train. With 1000 steps run1 only saw 74% of train (decision 14), so we don't know where that point is.
- **How validation is measured:** not with `mlx_lm.lora`'s, which counts the padding token (decision 12) and only uses 25 batches (100 of 1,012 examples). The checkpoints are measured afterwards with `nll_ejemplo` on the whole `valid`.
- **Why the training doesn't need to change:** saving every 100 steps and choosing afterwards is the same as saving only when validation improves, and it leaves the whole curve to look at.
- **by_message:** its validation also has leakage, so its early stopping may choose wrong. It is part of the leakage experiment.
- **Pending decision:** how often the whole validation is measured (every checkpoint or every 200-500 steps, for time); and whether to retrain with the corrected loss (`steps < lengths`) or the same as run1 to be able to compare. For now, the same as run1 (same mlx loss), so that only the number of steps changes.
- **Status (7 Nov):** measured every 100 to 500 steps depending on the run, and kept the same mlx training loss as run1. So every adapter in this project was trained with the extra `!` in its loss (all evaluations exclude it). `experiments/mlx-lm loss padding.ipynb` shows that a 200-step run with the corrected mask doesn't learn the `!` (probability 0.0000 against 0.978); how much the corrected loss changes the validation minimum was not measured.

## 16. Learning rate: too small or too large? (2026-11-05, written before seeing run2's result)

- **Context:** at lr 1e-5 the train sample went from 3.36 to 2.18, while validation (`nll_ejemplo`, whole `valid`) reached its minimum of 3.418 at step 2500 and then rose to ~3.7. Train moved a lot, validation barely.
- **Hypothesis:** the lr is too large. Train drops a lot while validation barely moves, so the model goes straight into memorizing train before it finds what generalizes. Prediction: a lower lr (3e-6) gives a validation minimum clearly below 3.418.
- **How we decide:** compare the minimum validation `nll_ejemplo` across learning rates. As a reference for "the same", the session vs message split differed by ~0.01-0.02 (decision 14).
- **Status:** run2 (1e-4) and run3 (3e-6) done.
- **Result (5 Nov):** 3e-6: 3.433 at step 5000; 1e-5: 3.418 at 2500; 1e-4: 3.586 at 1200 (`nll_ejemplo` on full `valid`, `experiments/LearningRate.ipynb`, `experiments/results/checkpoints_lr{3e-6,1e-5,1e-4}.json`). I was wrong: a lower lr reaches the same minimum, only later, and a higher one makes it worse. Below 1e-5 the lr is not the bottleneck. Keep 1e-5 (3e-6 is as good but needs ~2x the steps). Since 3e-6 barely overfits and still stops at ~3.43, the ceiling doesn't look like overfitting. Next: more context per example.
- **Status (7 Nov):** 3e-6 trained further from step 6,755 (validation 3.441), stopping when validation got worse than the start: 3.482 after 500 more steps, 3.476 after 1,000, 3.505 after 1,500 (`experiments/results/checkpoints_lr3e-6_cont.json`). It doesn't reach 1e-5's 3.418. Caveat: the resume restarts Adam's state and the data order.

## 17. Context window: keep training with 6 turns, evaluate with 20 (2026-11-06)

- **Decision:** keep the adapter trained with up to 6 turns of context (`run1_5ep`, step 2500) and give it up to 20 turns at prediction time. Retraining with a 20-turn window is discarded.
- **Why:** with 20 turns at prediction, the 6-turn adapter goes from 3.418 to 3.394 on the full `valid` (paired bootstrap by session: −0.024, 95% CI [−0.031, −0.014], `experiments/Context.ipynb`, `experiments/results/context_v1.json`). Retraining with 20 turns (`adapters/run1_ctx20`, lr 1e-5, same targets and splits, `data/dataset/session_ctx20`) gets 3.386 at step 2600, but against the 6-turn adapter evaluated with 20 turns that is −0.008, 95% CI [−0.018, +0.002]: it crosses 0, so it can be noise. Its neighbouring checkpoints give 3.399 and 3.420, and the step is picked on the same `valid`, so −0.008 is optimistic (`experiments/ContextRetrain.ipynb`, `experiments/results/checkpoints_ctx20.json`).
- **What was discarded:** training with 20 turns. Same result, longer sequences, slower training.
- **My prediction was wrong:** I expected the retrained adapter to beat the previous ones, since every message gets more context. It doesn't beat the same 6-turn adapter given 20 turns at prediction. Context is not the limit.
- **Next:** a data curve. Train with 25%, 50% and 100% of the train sessions (nested subsets by session, same `valid`) and compare the best validation loss of each. If it keeps going down with more data, the ~3.4 ceiling is lack of data; if it flattens, it is close to the floor of the problem.

## 18. More data still lowers the loss; no retrain with v2 (2026-11-06)

- **Decision:** don't retrain with the newer data (v2, about 20% more turns). Move on to measuring whether the generated replies sound like me, with a text classifier.
- **Data curve (`experiments/DataCurve.ipynb`):** same config as `run1_5ep`, 5 epochs of nested subsets by session (`build_subsets.py`, `scripts/train_datacurve.sh`), every model evaluated with `nll_ejemplo` on `valid_ctx20` (decision 17). Best valid: 25% (1,351 examples) 3.539 at step 300; 50% (2,708) 3.474 at step 1200; 100% (5,404) 3.394 (`run1_5ep`, step 2500). Paired bootstrap by session vs 100%: 25% +0.145, 95% CI [+0.125, +0.182]; 50% +0.080, 95% CI [+0.067, +0.097]. I stopped measuring data25 at step 1400 and data50 at 2400 on purpose: validation was only getting worse (`experiments/results/checkpoints_data25.json`, `checkpoints_data50.json`; the 100% point is `context_v1.json`).
- **Why:** each doubling of the data lowers validation (by 0.065 from 25% to 50% and 0.080 from 50% to 100%) and the curve doesn't flatten, so more data would still help. My prediction (the capacity of the base model is the limit) was wrong for this range; whether a bigger model would also help is untested. Three points and one seed each. But there isn't much more data: +20% is ~0.26 of a doubling, so by this trend about −0.02, too close to the noise to measure.
- **Caveat:** the 100% checkpoint was picked on the 6-turn `valid`, the other two on the 20-turn one.
- **Status (7 Nov):** repeated with three seeds, all picked on the 20-turn `valid`: 3.541, 3.470 and 3.389 (decision 29). Whether a bigger model helps: yes, much more than data (decision 26).
- **Next:** a classifier trained on my real messages vs my friends' (train split), then frozen and applied on the same test contexts to: my real replies (ceiling), the adapter's generations, and the base model's generations (floor). The metric is the share classified as mine, and where the adapter falls between floor and ceiling. Plus a length-only baseline, so I know how much the classifier wins just from length.

## 19. Style is measured with temperature 1.0 and no top-p (2026-11-06)

- **Decision:** to evaluate whether the adapter sounds like me, generate with temperature 1.0 and top-p 1.0, that is, sampling the distribution the model learned as is. Not 0.7 / 0.9.
- **How it's measured (`experiments/StyleClassifier.ipynb`, `build_style.py`):** classifiers trained only on real messages, mine vs my friends', same session split as the adapter, then frozen: length only (test AUC 0.551), TF-IDF of character 2-5 grams + logistic regression (0.840), mDeBERTa-v3-base (0.867). Applied to the 1,133 test contexts (20 turns): my real replies (ceiling), the adapter's (`run1_5ep` at step 2500), the base model's (floor), plus my friends'. Position of the adapter = (adapter − base) / (real − base), 95% CI by session. Temperature 0.7 results: `experiments/results/style_classifier.json`; temperature 1.0 results are printed by the notebook and not saved to JSON.
- **Why:** at temperature 0.7 / top-p 0.9 the adapter came out "more me than me" (position 1.40 TF-IDF, 1.68 transformer). Sampling trims the least likely tokens at every step, and my rarest marks are exactly those: "xd" in 1.6% of its messages vs 7.6% of mine, distinct-2 0.57 vs 0.81. Length (same distribution as mine) and copies of train (removing them changed nothing) were ruled out. At temperature 1.0 the marks and the variety come back ("xd" 5.6%, distinct-2 0.84) and the adapter lands between base and me (classified as mine: adapter 77.6%, me 78.8%, base 45.6% with TF-IDF; 74.5%, 80.5%, 49.7% with mDeBERTa): TF-IDF 0.96 [0.84, 1.08], indistinguishable from me; transformer 0.81 [0.69, 0.90], it still notices a gap.
- **Caveats:** temperature 1.0 was chosen after seeing test (it isn't a tuned value, but it's still a choice made on test). "JAJA" doesn't come back (2.6% vs 4.9%).
- **Blind test, dropped:** I planned a blind test where I pick which of two replies is mine (the real one or the adapter's) and say whether each fits. I dropped it before running it: I remember my own conversations, so it would measure my memory, not the style.
- **Next:** the classifier measures style, not whether the reply makes sense in the conversation. Measure relevance with a classifier of (context, reply) pairs trained on real data: my real reply fits, a reply of mine from another session doesn't (`experiments/Relevance.ipynb`).

## 20. Relevance: NLL gain from the context, against a floor (2026-11-06)

- **Decision:** measure whether a reply fits the conversation as the NLL gain per token that the context gives (NLL without context − with context; `nll_ejemplo`, scorer Qwen2.5-1.5B-Instruct-4bit, which generated none of the sets), with a floor: a real reply of mine from another session, same length bin (`experiments/Relevance.ipynb`, `experiments/results/relevance.json`, per example in `relevance_nll.json`).
- **What was discarded:** a cross-encoder (mDeBERTa) trained on (context, my real reply) vs (context, a reply of mine from another session). It didn't learn: valid AUC 0.563, test 0.567, accuracy 0.500, below plain TF-IDF similarity (0.60). Many replies of mine are short and generic and fit anywhere, so the labels are noisy. 28 min per epoch, no room to iterate.
- **Result (test, 1,133 contexts, 95% CI by session):** my real reply 1.04 [0.93, 1.12]; adapter at temperature 1.0 0.85 [0.74, 0.92]; floor 0.80 [0.68, 0.87]; base 0.27 [0.25, 0.28]. Position of the adapter between floor and real: 0.20 [−0.11, 0.45].
- **What it means:** most of what the context gives is the register (short informal chat), which any reply of mine has. The part that is "this answers that" is the band from 0.80 to 1.04, and the adapter covers 20% of it, CI including 0. It sounds like me (decision 19) but barely answers the conversation. My prediction (about 75% fitting, made for the cross-encoder) expected more.
- **Caveats:** 1.5B scorer, 6 turns of context; the base model's replies are 3.5x longer, so its comparison is confounded. It doesn't tell whether the limit is capacity, data, or both.
- **Next:** to decide on Monday. Levers: more data (v2 copy) or a somewhat bigger base model.
- **Status (7 Nov):** relevance was re-scored with Qwen2.5-7B for every set (decision 22). With that judge the adapter's position is 0.27 [0.09, 0.42], the first interval that excludes 0. The seeds are decision 29.

## 21. Length explains a small part of the signal (2026-11-07)

- **Decision (`experiments/Length.ipynb`, no model, per-reply numbers from `experiments/results/signal_run1_5ep.json`):** test the length hypothesis of decision 14 in two ways: the signal inside length bins (same length, mine against my friends'), and a length-matched signal (my friends' replies reweighted to my length distribution). Bootstrap by session as always.
- **Prediction (written before running):** decision 14's hypothesis, as written then: a good part of the improvement over base is shorter messages. And a second question it doesn't cover: whether length explains the gap between me and my friends. My friends and I write in the same register, so I didn't expect length to explain much of it.
- **Numbers:** my replies are only slightly shorter (median 13 against 14 tokens; mean 17.5 against 19.8). The drop against base is largest on short replies: on mine 98.5% at 1-3 tokens and 86.4% at 4-7, against 57.6% at 16-31 and 43.9% at 32+ (my friends' follow the same shape). Length-matched signal: **16.9 points, 95% CI [11.7, 22.7]**, against 19.8 as is. By bin, the signal is 7.4 and 5.6 points at 1-3 and 4-7 tokens and 17.1, 18.5 and 18.3 at 8-15, 16-31 and 32+.
- **Conclusion:** both predictions held. Most of the gain over base is on short messages (learning to be brief and close the message). Length explains about 3 of the 19.8 points between me and my friends; what is specifically mine shows up where there is something to say.
- **Caveat:** it controls for length only. The control group still differs in who is speaking, and their replies are scored under a system prompt that names me.

## 22. Is fine-tuning worth it? Alternatives with the same evaluation (2026-11-07)

- **Decision (`experiments/Baselines.ipynb`, `experiments/results/style_baselines.json`, `style_retrieval_valid.json`, `relevance_7b.json`, `relevance_nll_7b.json`, `signal_run1_1.5b.json`, `checkpoints_run1_1.5b.json`):** compare the adapter with what you'd try without training, on the same 1,133 test contexts, the same frozen style classifiers and the same relevance measure: (1) retrieval: my real reply to the most similar conversation (TF-IDF over the last 6 turns), no model; (2) few-shot prompting of Qwen2.5 0.5B, 1.5B and 3B, untrained, with 20 real replies of mine from train in the system prompt, temperature 1.0; (3) the same recipe on Qwen2.5-1.5B.
- **The judge changed:** the 1.5B judged relevance in decision 20, and now it also generates. Relevance was re-scored for every set with Qwen2.5-7B-Instruct-4bit: same family, too big to train on my Mac, small enough to run there. It generates and trains nothing here.
- **Prediction (written before running):** a bigger model improves what gets said (relevance) while keeping my style; style has to be measured too.
- **Numbers (style position with TF-IDF / mDeBERTa: 0 = base, 1 = me; relevance position: 0 = my reply to another conversation, 1 = my real reply):** fine-tuned 0.5B 0.96 / 0.81 / 0.27 [0.09, 0.42]; fine-tuned 1.5B 1.08 / 0.94 / 0.33 [0.19, 0.54]; few-shot 0.5B -0.02 / -0.25 / -0.55; few-shot 1.5B 0.47 / 0.32 / -0.11; few-shot 3B 0.32 / 0.59 / 0.27 [0.07, 0.46]; retrieval from train 1.37 / 1.38 / 0.40 [0.22, 0.58]; retrieval from validation 1.05 / 1.09 / 0.37 [0.14, 0.64]. Context gain with the 7B judge: floor 0.214, real 0.658, adapter 0.336, base -0.040. Paired differences in gain: retrieval - adapter 0.5B +0.054 [-0.022, +0.163]; adapter 1.5B - adapter 0.5B +0.026 [-0.050, +0.157]; adapter 0.5B - floor +0.122 [+0.036, +0.188]. The 1.5B adapter: best valid 3.150 at step 2,500, signal 22.2 points [16.3, 29.1].
- **Conclusion:** half right. For style, fine-tuning wins clearly: no prompt gets close, not even a 3B with 20 examples. For relevance nothing is demonstrably better than anything else: retrieval has the highest mean, but every paired difference between the adapters and retrieval crosses 0. The 1.5B keeps my style best but doesn't answer demonstrably better. The adapter does use the conversation (it beats the floor), just not much. The 7B judge separates better than the 1.5B: the band between floor and my real reply is 0.44 nats instead of 0.24.
- **Caveats:** retrieval from train has inflated style (the classifiers were trained on those replies); from the validation pool it is as "me" as my real replies, which is what it is, and its relevance is the same (paired -0.010 [-0.110, +0.096]). The judge has the same information as the generator: it can't know the facts of my life, so "relevance" here means coherence with the conversation, not true content.

## 23. Memorization: a dose-response study (2026-11-07)

- **Decision (`experiments/Memorization.ipynb`, `scripts/build_canaries.py`, `scripts/eval_exposure.py`, `experiments/results/memorization.json`):** 24 invented codes, 4 for each number of repetitions (1, 2, 4, 8, 16, 32), appended to random training replies (252 lines, 4.7% of train), the same recipe as the final adapter, checkpoints every 500 steps. For each code: **exposure**, the rank of the real code among 1,000 invented ones (log2(1001) - log2(rank), from 0 to 9.97), and **extraction**, whether greedy generation writes it exactly. In the conversation where it was planted and with no conversation.
- **Why:** the canary of decision 13 is one code in one context with 30 alternatives: it can say that memorization started, not when.
- **Prediction (written before running):** codes seen 1 or 2 times don't come out; from 4 repetitions they rank in the top 10 (exposure ≥ 6.6); full extraction only at 16 and 32.
- **Numbers (step 2,500, in context):** mean exposure 5.35 at 1 repetition (ranks 287, 7, 2 and 90), 7.84 at 2, 9.97 (every code first of 1,001) from 4 to 32. Codes written out: 0 up to 16 repetitions, 3 of 4 at 32. The base model stays between 0.7 and 2.5 at every level.
- **Conclusion:** right that 4 repetitions are enough to rank a code first and that writing it out needs many (reliably only at 32). Wrong that 1 or 2 don't show: with 2, three of the four codes rank in the top 2. Recognizing a secret comes long before writing it out, and recognizing is what matters for privacy: anyone with the weights could recover a code by scoring candidates. One more reason the adapters are never published.
- **Caveats:** "repetitions" counts training lines; at ~1.9 epochs each was seen about twice that. Four codes per level, so the low levels are driven by individual codes. Run on the home GPU server (decision 24), so it is compared with itself, not with the Mac runs.

## 24. A second machine: a home GPU server, and when its numbers can be mixed (2026-11-07)

- **Decision:** run experiments also on a home server with an RTX 3080 (10 GB), in a container with MLX's CUDA backend and the same mlx-lm recipe, through a job queue. The data goes there over my home network and nowhere else. Within one experiment, every training runs on the same machine. Evaluations can run on either.
- **Why:** the Mac was the bottleneck: trainings and evaluations queued for hours. The 3080 evaluates a checkpoint in about 40 seconds.
- **Why evaluations can be mixed:** the same two checkpoints (`data25` at steps 300 and 1,400) evaluated on both machines give 3.53912 / 3.53915 and 4.10193 / 4.10159: differences of 0.00003 and 0.0003 nats, far below anything this project reads as a difference.
- **Why trainings are not mixed:** a tiny numeric difference in step 1 changes every later step, and with 10 GB the 1.5B only fits with micro-batches of 1, 4 accumulation steps and gradient checkpointing: the same effective batch, but not bit for bit the Mac recipe.
- **Discarded:** training the seeds of decision 29 on the server; it would mix machines inside one experiment.

## 25. LoRA rank: the adapter's size is not the limit (2026-11-07)

- **Decision (`experiments/LoraRank.ipynb`, `experiments/results/checkpoints_rank*.json`):** ranks 4, 8, 16 and 32 (1.5M to 11.7M trainable parameters), in two families: mlx-lm's fixed scale (20), and scale = 160 / rank (the original LoRA convention), which keeps the size of the update the same, so only capacity changes. Same recipe otherwise, to 3,500 steps, on the server (micro-batch 2, 2 accumulation steps).
- **Prediction (written before running):** a bigger adapter won't help, because the data is the limit. The threshold for "it helps" was set at more than the noise seen so far (~0.02, ideally 0.05) per doubling. If a bigger rank only reaches its minimum earlier without a lower value, it is memorizing faster; if only the fixed-scale family changes, it is step size, not capacity.
- **Numbers (best validation, 6 turns):** fixed scale 3.422 (rank 4), 3.408 (8), 3.399 (16), 3.406 (32, at step **1,100**); scaled 3.412, 3.408, 3.398, 3.399 (32 at step 2,500).
- **Conclusion:** as predicted. Eight times more parameters move the loss by at most 0.024 in total, within the training noise later measured on the server (decision 28, a recipe with 4 accumulation steps instead of 2). The early minimum of rank 32 at fixed scale was step size: with the update size held constant it goes back to step 2,500.

## 26. Model size: 0.5B, 1.5B and 3B with the same recipe (2026-11-07)

- **Decision (`experiments/ModelSize.ipynb`, `experiments/results/checkpoints_size_*.json`, `signal_size_*.json`):** the same recipe on the three sizes, all on the server with micro-batch 1, 4 accumulation steps and gradient checkpointing (the 0.5B retrained that way too), best step on validation, then the signal on test. The three share the tokenizer, so losses are comparable token by token.
- **Prediction (written before running):** the 3B improves again, by about the same or a bit less than the 1.5B did, and both my loss and my friends' improve.
- **Numbers:** best validation 3.424, 3.147 and 3.018 (all at step 2,500). Test perplexity on my replies, base to adapted: 75.5 to 29.8, 60.9 to 23.0, 132.0 to 20.7; on my friends': 81.1 to 49.2, 58.5 to 34.6, 122.9 to 30.2. Signal in percentage points: 21.2, 21.4 and 8.9. Signal in nats (how much harder my friends' replies are than mine for the adapted model, minus the same gap for the base model): 0.430 [0.314, 0.562], 0.448 [0.329, 0.599], 0.450 [0.253, 0.676].
- **Conclusion:** both predictions held: the 3B improves by less (-0.13 against -0.28) and both losses go down. The percentage signal falls for the 3B only because its base model starts far worse on WhatsApp (perplexity 132): both drops approach 100% and their difference shrinks. In nats the signal is the same for the three sizes. A bigger model doesn't learn more of *me*; it is a better predictor of everyone's WhatsApp once adapted.
- **Decision on the metric:** from here on, signals across models are compared in nats. The percentage-point signal stays as the headline for the 0.5B, where it was defined.
- **Caveats:** one seed per size. The share of layers adapted shrinks with size (16 of 24, 28 and 36), which decision 28 checks. Lower loss doesn't mean better replies: style and relevance were measured for the 1.5B (decision 22), not the 3B.

## 27. Memorization on the 1.5B (2026-11-07)

- **Decision (`experiments/Memorization1.5B.ipynb`, `experiments/results/memorization_1.5b.json`):** decision 23 repeated on Qwen2.5-1.5B, same canaries and lines.
- **Prediction (written before running):** the 1.5B memorizes sooner: higher exposure at 1 and 2 repetitions, and codes written out with fewer than 32.
- **Numbers (step 2,500, in context, mean exposure [codes written out of 4]):** 1 repetition 3.93 [0], 2: 6.52 [0], 4: 9.57 [0], 8: 9.72 [0], 16: 9.97 [2], 32: 9.97 [4]. The 0.5B: 5.35, 7.84, 9.97, 9.97, 9.97 [0], 9.97 [3]. At step 3,500 the 1.5B writes out 2 of the 4 codes planted 8 times.
- **Conclusion:** half right. It doesn't recognize secrets sooner (exposure the same or slightly lower, within noise with 4 codes per level), but it writes them out with fewer repetitions: at step 2,500 it writes out 2 of the 4 codes planted 16 times (the 0.5B none), and at step 3,500 2 of the 4 planted 8 times (the 0.5B never writes out a code planted fewer than 16 times, and those only 1 of 4 at some steps). About one doubling fewer. For leaking a secret, that is the part that counts.
- **Caveat:** one run per model, 4 codes per level, no confidence intervals. Neither the exposure differences nor the extraction counts (2 of 4 against 0 of 4) are statistically significant on their own; the 8-repetition extraction is at step 3,500, past the step chosen on validation.

## 28. LoRA on all layers, with three seeds (2026-11-07)

- **Decision (`experiments/AllLayers.ipynb`, `experiments/AllLayersSeeds.ipynb`, `experiments/results/checkpoints_alllayers_*.json`, `checkpoints_size_0.5b_seed*.json`, `signal_alllayers_*.json`):** LoRA on every layer (`--num-layers -1`) instead of mlx-lm's default of the last 16, on the 0.5B and the 1.5B, against the 16-layer runs of decision 26 (same machine and recipe). Then both arms of the 0.5B with seeds 1 and 2.
- **Prediction (written before running):** unsure. If data is the limit, all layers lands within ±0.02 of 16 layers and reaches its minimum earlier; if the early layers help with my vocabulary, the gain shows up mostly on short replies. For the seeds: they land close together, and the advantage holds.
- **Numbers:** 0.5B 3.424 to 3.381 (paired -0.043 [-0.068, -0.025]); 1.5B 3.147 to 3.108 (-0.039 [-0.054, -0.028]); minimum at step 2,500 for all. The gain is similar for every reply length (0.5B: -0.016, -0.046, -0.042, -0.045 for 1-5, 6-15, 16-40 and 41+ tokens). Signal unchanged (0.5B 21.2 to 20.8, 1.5B 21.4 to 22.1). With three seeds: 16 layers 3.424 / 3.440 / 3.417, all layers 3.381 / 3.403 / 3.381; gain per seed -0.043, -0.037, -0.036.
- **Conclusion:** neither reading of the first prediction holds: all layers is better by about 0.04, outside the ±0.02 band, without an earlier minimum and evenly across lengths. The seeds confirm it: the gain repeats in every seed and is about twice the largest seed-to-seed spread (0.023). Where LoRA sits matters more than how big it is (decision 25).
- **Caveat:** the 0.5B with all layers (3.381) is the best 0.5B, but on the server recipe; it is not compared number by number with the Mac's `run1_5ep`. The reference adapter stays the 16-layer one, because every other result was measured on it.

## 29. Seeds: how much is training noise (2026-11-07)

- **Decision (`experiments/Seeds.ipynb`, `experiments/results/checkpoints_run1_seed*.json`, `checkpoints_data{25,50}_seed*.json`, `signal_run1_seed*.json`):** train the final recipe and the 25% and 50% data subsets with seeds 0, 1 and 2, on the Mac with the reference recipe; pick each on the 20-turn validation set; measure the signal for the three final-recipe seeds. The evaluations ran on the server (decision 24).
- **Prediction (written before running):** a check more than a hypothesis: the results hold across seeds, and three seeds are enough.
- **Numbers:** final recipe 3.394 / 3.387 / 3.386, all at step 2,500 (sd 0.004). Signal 19.8 / 18.4 / 19.6 points (sd 0.7). Data curve: 25% 3.539 / 3.542 / 3.541 (step 300 for all), 50% 3.474 / 3.467 / 3.469 (step 1,200 for all).
- **Conclusion:** confirmed, with less noise than I expected. The chosen step and the signal are not a lucky seed, and each doubling of the data (-0.07 to -0.08) is about 20 times the seed spread. Rule for the rest of the project: on the Mac recipe, differences between single runs below ~0.01 are not real; on the server recipe (decision 28, sd ~0.012), below ~0.02 to 0.025.
- **Caveat:** the seeds do change the training (the LoRA weights of seeds 1 and 2 already differ at step 100); with lr 1e-5 the runs simply end in the same place. Whether the server recipe is really noisier or that is chance with three seeds per side is open.

## 30. Adapter + retrieval, and closing that line (2026-11-07)

- **Decision (`experiments/AdapterRetrieval.ipynb`, `scripts/generate.py --retrieve 3`, `experiments/results/relevance_rag.json`, `style_rag.json`):** give the 0.5B adapter, in its system prompt, my replies in the 3 most similar conversations of the validation pool, and measure style and relevance as in decision 22. No retraining.
- **Why:** decision 22 suggested the split "the adapter gives the form, retrieval the content".
- **Prediction (Fede, written before running):** the adapter uses the extra context and answers somewhat better, relevance position around 0.45 to 0.50, without copying the retrieved replies. Written with it: "better than the adapter" means the paired interval excludes 0, and 0.45 to 0.50 might be right and still not demonstrable against retrieval.
- **Numbers:** relevance 0.34 [0.17, 0.46]; paired against the adapter alone +0.027 nats [-0.040, +0.109], against retrieval alone -0.017 [-0.134, +0.077]. Style 1.00 / 0.85 (adapter alone 0.96 / 0.81). Identical to a retrieved reply: 0 of 1,133; nearly identical (character similarity ≥ 0.9): 0.
- **Conclusion:** half right: it doesn't copy and keeps my style, but it doesn't answer demonstrably better. Three replies of mine from other conversations say little about what to answer in this one.
- **Closed:** training the adapter with retrieved replies in its prompt was considered and discarded: not relevant to the project's question.

## 31. Intervals by chat, not only by session (2026-11-08)

- **Decision (`experiments/ByChat.ipynb`, `scripts/eval_by_chat.py`, `experiments/results/by_chat.json`):** recompute the main intervals resampling whole chats (a cluster bootstrap) instead of sessions. From now on the me-against-friends signal is reported with its interval by chat.
- **Why:** an external review pointed out that sessions from the same chat share a person and topics, and the test set is concentrated: 67 test sessions come from only 18 chats, and the 3 largest hold 63% of my test replies.
- **Prediction (written before running):** Fede: the signal survives, relevance may cross zero, the intervals widen but don't double. Claude: same, and paired differences barely change.
- **Numbers:** signal 19.8 points, by session [14.7, 25.8], by chat **[8.9, 26.0]** (seeds 1 and 2: [7.5, 24.9] and [8.1, 26.4]). Relevance position 0.27: [0.09, 0.42] by session, [0.10, 0.47] by chat. Style 0.96 / 0.81: by chat [0.84, 1.06] / [0.72, 0.89]. Paired validation differences (data curve, all layers, 0.5B to 1.5B) change by less than 0.01 in their bounds.
- **Conclusion:** the signal survives with a lower floor (about 50% wider interval): part of the 19.8 depends on which chats fell in test. Relevance doesn't cross zero; style and paired differences barely move. The chat structure matters for comparisons between two different groups of replies, not for comparisons of each chat with itself.

## 32. The signal is not catchphrases (2026-11-08)

- **Decision (`experiments/Repeats.ipynb`, `scripts/eval_repeats.py`, `experiments/results/repeats.json`):** recompute the signal after removing, on each side symmetrically, the test replies that repeat that speaker's own train replies (exact repeats, and the 100 most frequent or 5 characters or fewer).
- **Why:** the review's objection: my replies repeat a lot, and a model trained on me predicts those cheaply, so the signal could be stock phrases.
- **Prediction (written before running):** Fede: it drops about as much as the share of replies removed, but survives. Claude: it drops somewhat and survives.
- **Numbers:** removing exact repeats keeps 89.8% of mine and 92.3% of my friends' and gives 19.0 points (by chat [8.6, 24.5]); removing both kinds keeps 87.6% and 88.9% and gives 19.6 ([9.3, 24.8]).
- **Conclusion:** both right that it survives; it barely moves. Repeated replies are easy on both sides and cancel in the difference. What makes my drop larger is in the replies that don't repeat.

## 33. Cross-speaker control: the method measures who it was trained on (2026-11-08)

- **Decision (`experiments/CrossSpeaker.ipynb`, `build_speaker.py`, `experiments/results/signal_speaker_c36_*.json`):** in the chat with the most replies from the other person (chat 36; that person gave permission), train two adapters with the same recipe, sessions, data size (~1,770 examples) and a neutral system prompt ("Responde como esta persona en WhatsApp."): one on my replies, one on theirs. Score each on both speakers' test replies. Both trained on the home GPU server with the same settings.
- **Why:** the main control is asymmetric: the adapter saw only me, under a prompt that names me. If the method measures identity, an adapter trained on the other person must drop more on them.
- **Prediction (written before running):** Fede: my adapter on this chat gives a signal similar to the main one but smaller (less data); the other person's similar to mine. Claude: both positive; mine smaller than the main one.
- **Numbers:** trained on me: drop 52.5% on me, 30.7% on them, signal 21.8 points [13.5, 33.7], **0.377 nats [0.238, 0.563]**. Trained on them: 69.5% on them, 31.6% on me, 37.9 points [32.2, 40.3], **0.806 nats [0.729, 0.871]**. The main adapter's signal is 0.41 nats.
- **Conclusion:** both adapters learn the person they were trained on, so neither the catchphrases nor the asymmetric control explain the main signal. Mine on one chat is about the main one (not smaller: one chat with ~1,800 replies is enough). Theirs is about twice as large: the base model finds their writing much harder to start with (perplexity 125 against 83), so there is more that is specific to them. Signals are compared in nats because the starting points differ (decision 26).
- **Limits:** one chat, one seed per adapter, 7 test sessions.

## 34. Replication is with your own chats, not a public corpus (2026-11-08)

- **Decision:** the way to replicate this study is to run `scripts/run.py` on your own chats. No public dataset is added, and my data is never published.
- **Why:** a public replication needs real (not synthetic or scripted) informal chat between people, with the author of every message, that anyone can download. None was found that fits. Real WhatsApp corpora exist, including one in Spanish, but only on request and for academic use. Discord-Dialogues (Apache 2.0) is real casual chat but releases no author per message. The Ubuntu IRC corpus has authors but is technical support, a much more formal and uniform register. Reddit is forum comments, not chat. Discord Unveiled keeps hashed user IDs and Spanish servers, but its download record answered "410 Gone" and it states no data license.
- **Discarded:** replicating on a public corpus with a different register; it would show that the code runs, not the findings about chat.

## 35. A fairer few-shot baseline, and a weakness of the relevance measure (2026-11-08)

- **Decision (`experiments/FairFewShot.ipynb`, `experiments/results/style_fewshot_fair.json`, `relevance_fewshot_fair.json`):** give the prompting baseline a fair chance: Qwen2.5 0.5B, 1.5B and 3B, untrained, the same 20 replies of mine in the prompt, a cap of 64 tokens, and two instructions ("one short WhatsApp message, one sentence, like the examples", and the same plus "informal, lowercase, no final full stop"). The 7B only judges: a model can't judge its own replies.
- **Why:** the review called the first few-shot baseline a strawman: one untuned prompt and no length limit, while the classifiers partly pick up length.
- **Prediction (written before running):** Fede: style goes up against the original few-shot but doesn't reach the adapter. Claude: style goes up a lot with TF-IDF, less with mDeBERTa, none reaches the adapter, and relevance doesn't improve.
- **Numbers (style TF-IDF / mDeBERTa, relevance position):** 3B short 0.26 / 0.50, **1.27 [1.06, 1.52]**; 3B short + style 0.22 / 0.25, **1.29 [1.07, 1.52]**; 1.5B short 0.49 / 0.54, 0.30; 0.5B short 0.19 / -0.21, -0.39. Original 3B: 0.32 / 0.59, 0.27. Adapter: 0.96 / 0.81, 0.27. The short 3B's replies are as short as mine (42 characters against 48) and are not copies of the conversation (0.3% contained in it).
- **Conclusion:** Fede right on style: no prompt gets close, and the instructions barely move it (the 3B even goes down), so length was not what separated few-shot from my style. On relevance, the short 3B scores *above my real replies*: a short, on-topic reply from an instruction-tuned model is more predictable from the conversation than what I actually wrote. The relevance measure rewards "predictable from the conversation", not "what I would say", so the content conclusions are weaker than they looked. Open: whether it is the measure or the judge's family (Qwen judging Qwen); decision 36 checks it.

## 36. A second judge: relevance depends on who judges (2026-11-08)

- **Decision (`experiments/SecondJudge.ipynb`, `experiments/results/relevance_llama.json`, `relevance_nll_llama.json`):** score exactly the same replies of every set again with a judge of another family, Llama-3.1-8B-Instruct (4 bits), same contexts and measure, and compare relevance positions (losses in nats aren't comparable across tokenizers).
- **Why:** decision 35's short 3B scored above my real replies with the Qwen judge. Is that the measure, or Qwen preferring Qwen?
- **Prediction (written before running, both):** the short 3B stays above 1, maybe somewhat less: it is more the measure than the family.
- **Numbers (Qwen / Llama):** short 3B 1.27 / **1.01 [0.85, 1.29]**; fine-tuned 0.5B 0.27 / **-0.00 [-0.17, 0.15]**; fine-tuned 1.5B 0.33 / 0.02; retrieval (validation pool) 0.37 / 0.06; adapter + retrieval 0.34 / -0.05; few-shot 3B original 0.27 / -0.50; base 0.5B -0.57 / -1.42. Llama's band: floor 0.799, my real reply 1.206 nats.
- **Conclusion:** half right. The short 3B falls to my level, not above it: part of its excess was family preference, but generic on-topic replies still match my real ones with any judge, so that is the measure. More important: with Llama the adapters and retrieval sit at the floor. "The adapter uses the conversation a little" (decision 22) holds with one judge and not with the other. The robust content claim is the weak one: the adapter doesn't demonstrably answer better than a reply of mine copied from another conversation.
- **Decision on the measure:** relevance is reported with both judges from now on, and no claim relies on a difference that only one of them shows.

"""Does it sound like me? (decision 19)

Classifiers trained only on real messages, mine (1) vs my friends' (0), on the adapter's train
split (data/style from build_style.py), then frozen and applied to the same 1,133 test contexts:
my real replies (ceiling), the base model's generations (floor), the adapter's, and my friends'.
The metric is the share classified as mine; the adapter's position is
(adapter - base) / (real - base), with 95% CIs resampling sessions.

    uv run python scripts/eval_style.py --adapter-gens adapter_t1 --base-gens base_t1
    uv run python scripts/eval_style.py ... --transformer      # also mDeBERTa (~30 min per epoch on M1)

Reported at temperature 1.0: TF-IDF 0.96 [0.84, 1.08], mDeBERTa 0.81 [0.69, 0.90].
"""

import _setup  # noqa: F401
import argparse
import math
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from talk_like_me.config import DATA, DATASET, RESULTS, ROOT, SEED
from talk_like_me.io import read_jsonl, write_json
from talk_like_me.stats import interval, resample

TRANSFORMER = "microsoft/mdeberta-v3-base"
MAX_LEN = 64  # tokens per message for the transformer; replies are short


def lengths(texts):
    # log of characters and of words: replies go from "ok" to paragraphs
    return np.array([[math.log1p(len(t)), math.log1p(len(t.split()))] for t in texts])


def fit_length(x, y):
    m = make_pipeline(FunctionTransformer(lengths), StandardScaler(), LogisticRegression()).fit(x, y)
    return lambda ts: m.predict_proba(ts)[:, 1]


def fit_tfidf(x, y, xv, yv):
    """Character 2-5 grams + logistic regression; C chosen on valid AUC."""
    best, best_auc = None, -1
    for c in [0.3, 1, 3, 10]:
        m = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=False, min_df=2, sublinear_tf=True),
            LogisticRegression(C=c, max_iter=3000)).fit(x, y)
        auc = roc_auc_score(yv, m.predict_proba(xv)[:, 1])
        if auc > best_auc:
            best, best_auc = m, auc
    return lambda ts: best.predict_proba(ts)[:, 1]


def fit_transformer(x, y, xv, yv, epochs=3, batch=32, lr=2e-5, patience=1):
    """Fine-tune mDeBERTa on mine vs friends; keeps the epoch with the best valid AUC (cached in models/)."""
    import torch

    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
        # Cap MPS memory: past the cap it raises instead of filling the RAM and swapping (it reached 26 GB).
        torch.mps.set_per_process_memory_fraction(0.5)
    else:
        device = "cpu"
    empty_cache = {"cuda": torch.cuda.empty_cache, "mps": torch.mps.empty_cache}.get(device, lambda: None)
    path = ROOT / "models" / f"style_{TRANSFORMER.split('/')[-1]}"

    def probs(model, tok, texts, bs=64):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                # fixed padding: every batch has the same shape, so MPS reuses its buffers
                enc = tok(texts[i:i + bs], truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt").to(device)
                out.append(torch.softmax(model(**enc).logits.float(), dim=-1)[:, 1].cpu().numpy())
        empty_cache()
        return np.concatenate(out)

    if not path.exists():
        tok = AutoTokenizer.from_pretrained(TRANSFORMER)
        # float32: in fp16 the loss goes to NaN after the first step on MPS
        model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER, num_labels=2, dtype=torch.float32).to(device)
        torch.manual_seed(SEED)
        optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        steps = epochs * math.ceil(len(x) / batch)
        sched = get_linear_schedule_with_warmup(optim, int(0.1 * steps), steps)
        order, best, bad = np.random.default_rng(SEED), -1, 0
        for epoch in range(epochs):
            model.train()
            idx, start = order.permutation(len(x)), time.time()
            for k in range(0, len(idx), batch):
                b = idx[k:k + batch]
                enc = tok([x[i] for i in b], truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt").to(device)
                model(**enc, labels=torch.tensor(y[b], device=device)).loss.backward()
                optim.step(); sched.step(); optim.zero_grad()
                if (k // batch) % 50 == 0:
                    empty_cache()
            auc = roc_auc_score(yv, probs(model, tok, xv))
            print(f"transformer epoch {epoch + 1}: valid auc {auc:.3f} ({(time.time() - start) / 60:.1f} min)")
            if auc > best:
                best, bad = auc, 0
                model.save_pretrained(path); tok.save_pretrained(path)
            else:
                bad += 1
                if bad >= patience:
                    break
        empty_cache()
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.float32).to(device)
    return lambda ts: probs(model, tok, ts)


def score(clf, sets, sessions):
    """Share classified as me per set, 95% CI by session (resampled jointly), and the adapter's position."""
    sums = {}
    for k, texts in sets.items():
        s_ = {}
        for v, s in zip((clf(texts) >= 0.5).astype(float), sessions[k]):
            a = s_.setdefault(s, [0.0, 0])
            a[0] += v
            a[1] += 1
        sums[k] = s_

    def share(k, chosen):
        n = sum(sums[k].get(s, (0, 0))[1] for s in chosen)
        return sum(sums[k].get(s, (0, 0))[0] for s in chosen) / n

    all_sessions = sorted({s for v in sessions.values() for s in v})
    movable = [k for k in sets if k not in ("base", "real (me)", "friends")]
    boot = {k: [] for k in sets} | {f"position {k}": [] for k in movable}
    for chosen in resample(all_sessions):
        r = {k: share(k, chosen) for k in sets}
        for k in sets:
            boot[k].append(r[k])
        if r["real (me)"] != r["base"]:
            for k in movable:
                boot[f"position {k}"].append((r[k] - r["base"]) / (r["real (me)"] - r["base"]))
    out = {k: {"share": share(k, all_sessions), "ci": interval(boot[k])} for k in sets}
    for k in movable:
        pos = (out[k]["share"] - out["base"]["share"]) / (out["real (me)"]["share"] - out["base"]["share"])
        out[f"position {k}"] = {"value": pos, "ci": interval(boot[f"position {k}"])}
    out["position"] = out["position adapter"]  # kept for older readers
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--adapter-gens", required=True, help="name passed to generate.py for the adapter")
    p.add_argument("--base-gens", required=True, help="name passed to generate.py for the base model")
    p.add_argument("--extra", nargs="*", default=[], help="more generation names to score, each with its own position")
    p.add_argument("--transformer", action="store_true", help="also fine-tune and apply mDeBERTa (slow)")
    p.add_argument("--out", default=str(RESULTS / "style.json"))
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    data = {s: read_jsonl(DATA / "style" / f"{s}.jsonl") for s in ["train", "valid", "test"]}
    x = {s: [r["text"] for r in v] for s, v in data.items()}
    y = {s: np.array([r["label"] for r in v]) for s, v in data.items()}
    mine = [r for r in data["test"] if r["label"] == 1]
    friends = [r for r in data["test"] if r["label"] == 0]
    contexts = read_jsonl(DATASET / "session_ctx20" / "test.jsonl")
    assert [r["text"] for r in mine] == [c["messages"][-1]["content"] for c in contexts], "contexts and my test replies are not aligned"

    assert not set(args.extra) & {"base", "adapter", "real (me)", "friends"}, "--extra names clash with the fixed sets"
    gens = {k: [g["text"] for g in read_jsonl(DATA / "style" / f"generations_{n}.jsonl")]
            for k, n in [("base", args.base_gens), ("adapter", args.adapter_gens)] + [(e, e) for e in args.extra]}
    assert all(len(v) == len(mine) for v in gens.values()), "generations do not cover every test context"
    sets = gens | {"real (me)": [r["text"] for r in mine], "friends": [r["text"] for r in friends]}
    sessions = {k: [r["session"] for r in (friends if k == "friends" else mine)] for k in sets}

    clfs = {"length": fit_length(x["train"], y["train"]),
            "tfidf": fit_tfidf(x["train"], y["train"], x["valid"], y["valid"])}
    if args.transformer:
        clfs["transformer"] = fit_transformer(x["train"], y["train"], x["valid"], y["valid"])

    results = {}
    for name, clf in clfs.items():
        auc = roc_auc_score(y["test"], clf(x["test"]))
        results[name] = score(clf, sets, sessions) | {"test_auc": auc}
        r = results[name]
        print(f"\n{name} (test AUC on real messages {auc:.3f})")
        for k in sets:
            print(f"  {k:<10} classified as me {r[k]['share']:>6.1%} [{r[k]['ci'][0]:.1%}, {r[k]['ci'][1]:.1%}]")
        for k in sets:
            if f"position {k}" in r:
                q = r[f"position {k}"]
                print(f"  position of {k}: {q['value']:.2f} [{q['ci'][0]:.2f}, {q['ci'][1]:.2f}]")

    write_json(args.out, {"source": "scripts/eval_style.py", "generations": {"adapter": args.adapter_gens, "base": args.base_gens},
                          "scores": results}, force=args.force)


if __name__ == "__main__":
    main()

"""Regenerate the README figures from experiments/results/*.json.

Needs no private data: it only reads the aggregated numbers that the experiments wrote.
One result was only printed by its notebook and never saved to JSON; it is
copied below as constants, with the decision that records them.

    uv run python scripts/plots.py    # writes docs/figures/*.png

Uses the Inter font if it is installed (https://rsms.me/inter/), else Helvetica Neue or the default.
"""

import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no window: write files only
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "results"
OUT = ROOT / "docs" / "figures"

TRAIN_EXAMPLES = 5404  # session/train; epoch = step * batch / examples
BATCH = 4

# Printed by experiments/StyleClassifier.ipynb (decision 19) for temperature 1.0, never saved to JSON:
# share of replies each frozen classifier labels as mine, with its 95% CI by session.
# Temperature 0.7 is in style_classifier.json.
STYLE_T1 = {
    "tfidf": {"base": (0.456, 0.414, 0.505), "adapter": (0.776, 0.753, 0.801)},
    "transformer": {"base": (0.497, 0.446, 0.552), "adapter": (0.745, 0.720, 0.771)},
}

# Palette: fixed categorical order (blue, orange, aqua); warm grays for the base model and references.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BASE = "#b9b7b0"
INK, INK2, INK3 = "#1a1a19", "#5f5e5a", "#9a9893"
SURFACE, GRID = "#fbfaf7", "#e8e6e0"


def pick_font():
    installed = {f.name for f in fm.fontManager.ttflist}
    for regular, bold in [("Inter 18pt", "Inter 18pt SemiBold"), ("Inter", "Inter"), ("Helvetica Neue", "Helvetica Neue")]:
        if regular in installed:
            return regular, bold
    return "DejaVu Sans", "DejaVu Sans"


FONT, FONT_BOLD = pick_font()
BOLD = {"family": FONT_BOLD, "weight": "semibold" if FONT_BOLD == FONT else "normal"}


def style():
    plt.rcParams.update({
        "font.family": FONT, "font.size": 10.5,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.edgecolor": GRID, "axes.linewidth": 1.0,
        "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 6, "ytick.major.pad": 6,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "axes.labelsize": 10, "axes.labelpad": 8,
        "lines.linewidth": 2.2, "lines.markersize": 5.5, "legend.frameon": False, "legend.fontsize": 10,
    })


def header(fig, title, subtitle, footer):
    """Quiet, left-aligned title and subtitle above the plot, source line below."""
    fig.text(0.02, 0.975, title, fontsize=13, color=INK, va="top", **BOLD)
    fig.text(0.02, 0.905, subtitle, fontsize=10.5, color=INK2, va="top")
    fig.text(0.02, 0.015, f"talk-like-me | {footer}", fontsize=8.5, color=INK3, va="bottom")


def frame(fig, top=0.80, bottom=0.21):
    fig.subplots_adjust(left=0.08, right=0.97, top=top, bottom=bottom)


def load(name):
    with open(RESULTS / name) as f:
        return json.load(f)


def curve(d):
    return {int(k): v for k, v in sorted(d.items(), key=lambda kv: int(kv[0]))}


def by_session(results, sessions):
    sums = {}
    for (nll, n), s in zip(results, sessions):
        a = sums.setdefault(s, [0.0, 0])
        a[0] += nll
        a[1] += n
    return sums


def loss(sums, chosen):
    return sum(sums[s][0] for s in chosen) / sum(sums[s][1] for s in chosen)


def ci(results, sessions, reps=1000, seed=0):
    """95% CI of the token-weighted loss, resampling whole sessions (as in DataCurve.ipynb)."""
    sums = by_session(results, sessions)
    keys = sorted(sums)
    random.seed(seed)
    return np.percentile([loss(sums, random.choices(keys, k=len(keys))) for _ in range(reps)], [2.5, 97.5])


def whisker(ax, x, lo, hi, horizontal=False):
    """A thin interval line, no caps."""
    if horizontal:
        ax.plot([lo, hi], [x, x], color=INK, linewidth=1.2, solid_capstyle="round", zorder=4)
    else:
        ax.plot([x, x], [lo, hi], color=INK, linewidth=1.2, solid_capstyle="round", zorder=4)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=220)
    plt.close(fig)
    print("wrote", (OUT / name).relative_to(ROOT))


def fig_signal():
    s = load("signal_run1_5ep.json")  # the final adapter (run1_5ep at step 2,500)
    pp = s["perplexity"]
    PERPLEXITY = {w: {"base": round(pp[f"base_{w}"], 1), "run1": round(pp[f"adapter_{w}"], 1)} for w in ["me", "friends"]}
    SIGNAL_CI = tuple(round(x * 100, 1) for x in load("by_chat.json")["signal_run1_5ep"]["by_chat"])  # ByChat.ipynb
    drop = s["drop"]
    signal = s["signal"] * 100
    fig, ax = plt.subplots(figsize=(8, 3.6))
    rows = [("My replies", drop["me"], BLUE, PERPLEXITY["me"]), ("My friends' replies", drop["friends"], ORANGE, PERPLEXITY["friends"])]
    for y, (label, d, color, p) in enumerate(rows):
        ax.barh(y, d * 100, height=0.52, color=color, zorder=3)
        ax.text(d * 100 + 1.2, y, f"{d:.1%}", va="center", fontsize=12, color=INK, **BOLD)
        ax.text(d * 100 + 10.5, y, f"perplexity {p['base']} to {p['run1']}", va="center", fontsize=9.5, color=INK2)
    ax.set_yticks([0, 1], [r[0] for r in rows], fontsize=11, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 20), [f"{x}%" for x in range(0, 101, 20)])
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.spines["bottom"].set_visible(False)
    header(fig, f"Perplexity drops {signal:.1f} percentage points more on my replies than on my friends'",
           f"Drop in test perplexity (how surprised the model is) against the base model. 95% CI of the gap, resampling chats: {SIGNAL_CI[0]} to {SIGNAL_CI[1]}.",
           "final adapter (step 2,500) | 1,133 of my replies, 1,110 of my friends' in the same test conversations")
    fig.subplots_adjust(left=0.2, right=0.97, top=0.76, bottom=0.17)
    save(fig, "1_signal.png")


def fig_learning_rate():
    runs = [("3e-6", "valid_run3", AQUA), ("1e-5", "valid_run1", BLUE), ("1e-4", "valid_run2", ORANGE)]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for lr, key, color in runs:
        c = curve(load(f"checkpoints_lr{lr}.json")[key])
        x = [s * BATCH / TRAIN_EXAMPLES for s in c]
        ax.plot(x, list(c.values()), color=color, marker="o", markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
        best = min(c, key=c.get)
        bx = best * BATCH / TRAIN_EXAMPLES
        ax.scatter([bx], [c[best]], s=70, facecolor=SURFACE, edgecolor=color, linewidth=2, zorder=4)
        ax.annotate(f"{c[best]:.3f}", (bx, c[best]), textcoords="offset points",
                    xytext=(0, 9 if lr == "1e-4" else -18), ha="center", fontsize=9.5, color=INK, **BOLD)
        ax.annotate(f"lr {lr}", (x[-1], list(c.values())[-1]), textcoords="offset points",
                    xytext=(8, -14 if lr == "3e-6" else 0), va="center", fontsize=10, color=color, **BOLD)
    cont = RESULTS / "checkpoints_lr3e-6_cont.json"
    if cont.exists():  # 3e-6 trained further from its last step (6,755); Adam state restarts there
        d = load(cont.name)
        steps = [0] + sorted(int(k) for k in d["valid"])
        losses = [d["start"]] + [d["valid"][str(s)] for s in steps[1:]]
        ax.plot([(6755 + s) * BATCH / TRAIN_EXAMPLES for s in steps], losses, color=AQUA, linestyle=(0, (3, 2)),
                marker="o", markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
        ax.annotate("3e-6, resumed:\nworse, stopped", ((6755 + steps[-1]) * BATCH / TRAIN_EXAMPLES, losses[-1]),
                    textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, color=INK2)
    ax.set_xlabel("epochs")
    ax.set_ylabel("validation loss (nats per reply token)")
    ax.set_ylim(3.36, 3.98)
    ax.set_xlim(-0.1, 6.6)
    header(fig, "Learning rate: 3e-6 and 1e-5 reach about the same minimum; 1e-4 is worse",
           "Loss per reply token on the full validation set (6 turns of context). Circles: the minimum of each run.",
           "1,012 validation replies | early stopping keeps the minimum of each curve")
    frame(fig, top=0.78)
    save(fig, "2_learning_rate.png")


def fig_data_curve():
    """Best validation loss (20 turns) for 25%, 50% and 100% of the data, three seeds each (Seeds.ipynb),
    and the paired difference against 100% with its 95% CI by session (seed 0, DataCurve.ipynb)."""
    sizes = [(1351, "25%", ["data25", "data25_seed1", "data25_seed2"]),
             (2708, "50%", ["data50", "data50_seed1", "data50_seed2"]),
             (TRAIN_EXAMPLES, "100%", ["run1_seed0", "run1_seed1", "run1_seed2"])]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"width_ratios": [1.6, 1]})
    xs, means = [], []
    for n, label, names in sizes:
        vals = [min(load(f"checkpoints_{k}.json")["valid"].values()) for k in names]
        m = sum(vals) / len(vals)
        xs.append(n); means.append(m)
        ax.scatter([n * f for f in (0.97, 1.0, 1.03)], vals, s=22, color=BLUE, alpha=0.45, edgecolor="none", zorder=4)
        ax.annotate(f"{m:.3f}", (n, m), textcoords="offset points", xytext=(-12, -6), ha="right", fontsize=11, color=INK, **BOLD)
        ax.annotate(f"{label} of the data", (n, m), textcoords="offset points", xytext=(-12, -22), ha="right", fontsize=9.5, color=INK2)
    ax.plot(xs, means, color=BLUE, zorder=3)
    ax.scatter(xs, means, s=60, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=5)
    ax.set_xscale("log")
    ax.set_xticks(xs, [f"{x:,}" for x in xs])
    ax.minorticks_off()
    ax.set_xlim(900, 6600)
    ax.set_ylim(3.36, 3.57)
    ax.set_xlabel("training examples (log scale)")
    ax.set_ylabel("best validation loss (nats per reply token)")
    ax.set_title("Mean of three seeds (dots: each seed)", loc="left", fontsize=10.5, color=INK2, pad=10)

    # Paired difference against 100%: the same replies, the same bootstrap samples of sessions.
    d25, d50 = load("checkpoints_data25.json"), load("checkpoints_data50.json")
    sessions = d25["session"]
    full = by_session(load("context_v1.json")["per_example"]["adapter"]["20"], sessions)  # run1_5ep@2500, 20 turns
    keys = sorted(full)
    random.seed(0)
    samples = [random.choices(keys, k=len(keys)) for _ in range(1000)]
    for y, (d, label) in enumerate([(d25, "25% - 100%"), (d50, "50% - 100%")]):
        c = curve(d["valid"])
        sub = by_session(d["per_example"][str(min(c, key=c.get))], sessions)
        diff = loss(sub, keys) - loss(full, keys)
        lo, hi = np.percentile([loss(sub, ch) - loss(full, ch) for ch in samples], [2.5, 97.5])
        whisker(bx, y, lo, hi, horizontal=True)
        bx.scatter([diff], [y], s=60, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=5)
        bx.annotate(f"+{diff:.3f}  [{lo:+.3f}, {hi:+.3f}]", (diff, y), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=9.5, color=INK)
    bx.axvline(0, color=INK3, linewidth=0.8)
    bx.set_yticks([0, 1], ["25% of the data", "50% of the data"], fontsize=10, color=INK)
    bx.set_ylim(1.6, -0.6)
    bx.set_xlim(-0.02, 0.22)
    bx.grid(axis="y", visible=False)
    bx.grid(axis="x", visible=True)
    bx.set_xlabel("loss above 100% of the data (nats)")
    bx.set_title("Paired against 100%, 95% CI (seed 0)", loc="left", fontsize=10.5, color=INK2, pad=10)
    header(fig, "More data still helps: each doubling lowers the loss by 0.07 to 0.08",
           "Nested subsets of the training conversations, validation with 20 turns of context.",
           "seed spread: sd 0.001 to 0.004 | paired CI: same replies, 1,000 bootstrap samples of sessions")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.76, bottom=0.2, wspace=0.35)
    save(fig, "3_data_curve.png")
    return dict(zip([s_[1] for s_ in sizes], means))


def fig_style():
    """Share of replies a frozen classifier labels as mine: base and adapter, at two temperatures,
    against my real replies and my friends' (same test conversations)."""
    t07 = load("style_classifier.json")["scores"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)
    for ax, (clf, title) in zip(axes, [("tfidf", "TF-IDF classifier (AUC 0.84)"), ("transformer", "mDeBERTa classifier (AUC 0.87)")]):
        s = t07[clf]
        real, friends = s["real (me)"]["share"], s["friends"]["share"]
        groups = {"temperature 0.7": {k: (s[k]["share"], *s[k]["ci"]) for k in ["base", "adapter"]},
                  "temperature 1.0": STYLE_T1[clf]}
        ax.axhline(real * 100, color=AQUA, linewidth=1.8, zorder=2)
        ax.text(1.45, real * 100, f"my real\nreplies {real:.0%}", color=INK, va="center", fontsize=9.5)
        ax.axhline(friends * 100, color=ORANGE, linewidth=1.8, linestyle=(0, (4, 3)), zorder=2)
        ax.text(1.45, friends * 100, f"my friends'\nreplies {friends:.0%}", color=INK, va="center", fontsize=9.5)
        for g, (label, vals) in enumerate(groups.items()):
            for j, (k, color) in enumerate([("base", BASE), ("adapter", BLUE)]):
                v, lo, hi = vals[k]
                x = g + (j - 0.5) * 0.38
                ax.bar(x, v * 100, 0.34, color=color, zorder=3,
                       label=("base model" if k == "base" else "fine-tuned model") if g == 0 else None)
                whisker(ax, x, lo * 100, hi * 100)
                ax.text(x, 4, f"{v:.0%}", ha="center", fontsize=11, color=INK if k == "base" else "white", **BOLD)
        ax.set_xticks(range(len(groups)), list(groups), color=INK)
        ax.set_xlim(-0.5, 1.95)
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 20), [f"{y}%" for y in range(0, 101, 20)])
        ax.set_title(title, loc="left", fontsize=10.5, color=INK2, pad=10)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", bbox_to_anchor=(0.98, 0.855), ncol=2,
               handlelength=1.2, handleheight=1.0)
    header(fig, "At temperature 0.7 it scores above my own replies; at 1.0 it lands at my level",
           "Share of replies a frozen classifier, trained on my real messages against my friends', says are mine. Temperature = randomness of sampling.",
           "1,133 test conversations | lines on the bars: 95% CI by session | top-p 0.9 at temperature 0.7, none at 1.0")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.74, bottom=0.12, wspace=0.08)
    save(fig, "4_style.png")


def fig_relevance():
    """Relevance position of each set for two judges (Baselines, FairFewShot, SecondJudge notebooks)."""
    qwen = load("relevance_7b.json")["positions"] | load("relevance_fewshot_fair.json")["positions"]
    llama = load("relevance_llama.json")["positions"]
    order = [("base t1.0", "Base model 0.5B"), ("fewshot20_3b", "Few-shot 3B"), ("adapter t1.0", "Fine-tuned 0.5B"),
             ("adapter1.5b_t1", "Fine-tuned 1.5B"), ("retrieval_valid", "Retrieval: my reply to a similar chat"),
             ("fewshot20_3b_short", "Few-shot 3B, told to answer briefly")]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.axvline(0, color=ORANGE, linewidth=1.6, linestyle=(0, (4, 3)), zorder=1)
    ax.axvline(1, color=AQUA, linewidth=1.6, linestyle=(0, (4, 3)), zorder=1)
    ax.text(0.02, -0.95, "my reply to\nanother session", color=INK2, fontsize=8.5, va="center")
    ax.text(1.02, -0.95, "my real\nreply", color=INK2, fontsize=8.5, va="center")
    for y, (key, label) in enumerate(order):
        for off, (pos, color, name) in zip((-0.15, 0.15), [(qwen, BLUE, "judge: Qwen2.5-7B"), (llama, ORANGE, "judge: Llama-3.1-8B")]):
            v, (lo, hi) = pos[key]["value"], pos[key]["ci"]
            whisker(ax, y + off, lo, hi, horizontal=True)
            ax.scatter([v], [y + off], s=46, color=color, edgecolor=SURFACE, linewidth=1.2, zorder=5, label=name if y == 0 else None)
    ax.set_yticks(range(len(order)), [o[1] for o in order], fontsize=10, color=INK)
    ax.invert_yaxis()
    ax.set_ylim(len(order) - 0.5, -1.4)
    ax.set_xlim(-1.85, 1.65)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlabel("relevance position: how much the conversation helps a judge predict the reply")
    ax.legend(loc="lower left", handletextpad=0.3)
    header(fig, "Whether it answers depends on the judge; a brief generic reply matches mine",
           "Same replies, two judge models. 0 = a reply of mine to another session, 1 = my real reply.",
           "1,133 test conversations | lines: 95% CI by session | neither judge trains or generates anything here")
    fig.subplots_adjust(left=0.36, right=0.97, top=0.80, bottom=0.17)
    save(fig, "5_relevance.png")


def fig_baselines():
    """Style against relevance for every way of getting a reply (Baselines.ipynb, AdapterRetrieval.ipynb)."""
    style_b = load("style_baselines.json")["scores"]["transformer"]
    style_v = load("style_retrieval_valid.json")["scores"]["transformer"]
    style_r = load("style_rag.json")["scores"]["transformer"]
    rel = load("relevance_7b.json")["positions"]
    rel_r = load("relevance_rag.json")["positions"]
    pts = [  # label, style position, relevance position, color, label offset
        ("fine-tuned 0.5B", style_b["position adapter"], rel["adapter t1.0"], BLUE, (-6, -18)),
        ("fine-tuned 1.5B", style_b["position adapter1.5b_t1"], rel["adapter1.5b_t1"], BLUE, (6, -20)),
        ("fine-tuned 0.5B\n+ retrieval", style_r["position adapter_rag3"], rel_r["adapter_rag3"], BLUE, (-8, 14)),
        ("few-shot 0.5B", style_b["position fewshot20_0.5b"], rel["fewshot20_0.5b"], INK3, (8, 0)),
        ("few-shot 1.5B", style_b["position fewshot20_1.5b"], rel["fewshot20_1.5b"], INK3, (8, 0)),
        ("few-shot 3B", style_b["position fewshot20_3b"], rel["fewshot20_3b"], INK3, (-8, 8)),
        ("retrieval\n(no model)", style_v["position retrieval_valid"], rel["retrieval_valid"], AQUA, (8, 10)),
        ("few-shot 3B,\nanswer briefly", load("style_fewshot_fair.json")["scores"]["transformer"]["position fewshot20_3b_short"],
         load("relevance_fewshot_fair.json")["positions"]["fewshot20_3b_short"], INK3, (8, -4)),
    ]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.axhline(0, color=INK3, linewidth=0.8, zorder=1)
    ax.axhline(1, color=AQUA, linewidth=1.4, linestyle=(0, (4, 3)), zorder=1)
    ax.axvline(1, color=AQUA, linewidth=1.4, linestyle=(0, (4, 3)), zorder=1)
    ax.text(1.02, 0.97, "my real replies", color=INK2, fontsize=8.5, va="top")
    for label, sx, ry, color, off in pts:
        ax.plot([sx["ci"][0], sx["ci"][1]], [ry["value"]] * 2, color=color, linewidth=1, alpha=0.6, zorder=2)
        ax.plot([sx["value"]] * 2, [ry["ci"][0], ry["ci"][1]], color=color, linewidth=1, alpha=0.6, zorder=2)
        ax.scatter([sx["value"]], [ry["value"]], s=60, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=4)
        ax.annotate(label, (sx["value"], ry["value"]), textcoords="offset points", xytext=off,
                    ha="left" if off[0] > 0 else "right", fontsize=9, color=INK)
    ax.set_xlim(-0.45, 1.4)
    ax.set_ylim(-0.85, 1.6)
    ax.set_xlabel("style position (mDeBERTa): 0 = base model, 1 = my real replies")
    ax.set_ylabel("relevance position\n0 = my reply to another session, 1 = my real reply", fontsize=9.5)
    ax.grid(axis="x", visible=True)
    ax.scatter([], [], s=50, color=BLUE, label="fine-tuned")
    ax.scatter([], [], s=50, color=INK3, label="prompted, no training")
    ax.scatter([], [], s=50, color=AQUA, label="copied, no model")
    ax.legend(loc="lower right", handletextpad=0.3)
    header(fig, "Training gets the style; on content this measure can't rank the methods",
           "Same 1,133 test conversations. Style: mDeBERTa classifier. Lines: 95% CI by session.",
           "style: frozen classifier, mine vs my friends' | relevance: Qwen2.5-7B judge (both judges: figure 5)")
    frame(fig, top=0.80, bottom=0.16)
    fig.subplots_adjust(left=0.13)
    save(fig, "6_baselines.png")


def best_of(name, key="valid"):
    v = load(f"checkpoints_{name}.json")[key]
    return min(v.values())


def fig_levers():
    """Change in best validation loss for each lever, each against its own baseline in the same experiment."""
    lr = {k: best_of(f"lr{k}", f"valid_run{n}") for k, n in [("3e-6", 3), ("1e-5", 1), ("1e-4", 2)]}
    ctx = load("context_v1.json")["per_example"]["adapter"]
    mean = lambda rows: sum(a for a, _ in rows) / sum(n for _, n in rows)
    d50 = [best_of("data50")] + [best_of(f"data50_seed{s}") for s in (1, 2)]
    d100 = [best_of(f"run1_seed{s}") for s in (0, 1, 2)]
    l16 = [best_of("size_0.5b")] + [best_of(f"size_0.5b_seed{s}") for s in (1, 2)]
    lall = [best_of("alllayers_0.5b")] + [best_of(f"alllayers_0.5b_seed{s}") for s in (1, 2)]
    avg = lambda xs: sum(xs) / len(xs)
    MAC, SERVER = 0.01, 0.025  # about twice the seed spread of each recipe (Seeds, AllLayersSeeds)
    rows = [  # label, delta, noise threshold of the machine it ran on
        ("learning rate 1e-5 to 1e-4 (Mac)", lr["1e-4"] - lr["1e-5"], MAC),
        ("learning rate 1e-5 to 3e-6 (Mac)", lr["3e-6"] - lr["1e-5"], MAC),
        ("retrain with 20 turns of context (Mac)", best_of("ctx20", "valid_run1_ctx20") - mean(ctx["20"]), MAC),
        ("LoRA rank 8 to 32, same update (server)", best_of("rank32_a160") - best_of("rank8"), SERVER),
        ("20 turns instead of 6 at prediction (Mac)", mean(ctx["20"]) - mean(ctx["6"]), MAC),
        ("LoRA on all layers, 3 seeds (server)", avg(lall) - avg(l16), SERVER),
        ("twice the data, 3 seeds (Mac)", avg(d100) - avg(d50), MAC),
        ("base model 1.5B to 3B (server)", best_of("size_3b") - best_of("size_1.5b"), SERVER),
        ("base model 0.5B to 1.5B (server)", best_of("size_1.5b") - best_of("size_0.5b"), SERVER),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for y, (label, d, noise) in enumerate(rows):
        ax.plot([-noise, noise], [y, y], color=GRID, linewidth=9, solid_capstyle="butt", zorder=1)
        color = ORANGE if d > noise else (BLUE if d < -noise else BASE)
        ax.barh(y, d, height=0.56, color=color, zorder=3)
        ax.text(d + (0.006 if d >= 0 else -0.006), y, f"{d:+.3f}", va="center", ha="left" if d >= 0 else "right",
                fontsize=10, color=INK, **BOLD)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=10, color=INK)
    ax.invert_yaxis()
    ax.set_ylim(len(rows) - 0.5, -1.4)
    ax.set_xlim(-0.33, 0.22)
    ax.axvline(0, color=INK3, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlabel("change in best validation loss (nats per reply token; negative is better)")
    header(fig, "What moved the loss: the base model, then data, then where LoRA sits",
           "Each bar against its own baseline. Light gray behind it: about twice the seed spread on its machine.",
           "0.5B unless noted | gray bars: within noise | seed sd 0.004 (Mac), about 0.012 (server) | one seed unless noted")
    fig.subplots_adjust(left=0.42, right=0.97, top=0.80, bottom=0.17)
    save(fig, "7_levers.png")


def main():
    style()
    fig_signal()
    fig_learning_rate()
    data = fig_data_curve()
    fig_style()
    fig_relevance()
    fig_baselines()
    fig_levers()
    # Cross-check against DECISIONS.md (decision 29): 3.541, 3.470, 3.389.
    print("data curve best valid:", {k: round(v, 3) for k, v in data.items()})


if __name__ == "__main__":
    main()

"""Bootstrap by session.

Examples from the same session share their context, so they are not independent; the session
is the unit that gets resampled (decision 12). 1000 resamples with replacement, seed 0, as in
the notebooks, so the intervals reproduce theirs.
"""

import random

import numpy as np

REPS = 1000


def by_session(results, sessions):
    """{session: [nll_sum, n_tokens]} from per-example [nll, n] and each example's session."""
    sums = {}
    for (nll, n), s in zip(results, sessions):
        acc = sums.setdefault(s, [0.0, 0])
        acc[0] += nll
        acc[1] += n
    return sums


def loss_over(sums, chosen):
    """Token-weighted loss of the chosen sessions (repeats count again, as the bootstrap needs)."""
    nll = n = 0
    for s in chosen:
        a, b = sums.get(s, (0.0, 0))
        nll += a
        n += b
    return nll / n


def resample(keys, reps=REPS, seed=0):
    """`reps` bootstrap samples of `keys`; same random calls as the notebooks."""
    keys = sorted(keys)
    random.seed(seed)
    return [random.choices(keys, k=len(keys)) for _ in range(reps)]


def interval(values):
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [float(lo), float(hi)]


def loss_ci(results, sessions, reps=REPS, seed=0):
    """95% CI of the token-weighted loss of one set."""
    sums = by_session(results, sessions)
    return interval([loss_over(sums, c) for c in resample(sums, reps, seed)])


def paired_diff(before, after, sessions, reps=REPS, seed=0):
    """(after - before) on the same replies, with its 95% CI. Negative means `after` predicts better."""
    a, b = by_session(before, sessions), by_session(after, sessions)
    diffs = [loss_over(b, c) - loss_over(a, c) for c in resample(a, reps, seed)]
    return loss_over(b, sorted(b)) - loss_over(a, sorted(a)), interval(diffs)

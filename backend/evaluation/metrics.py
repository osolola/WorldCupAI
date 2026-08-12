import math

import numpy as np

OUTCOME_CLASSES = ["team_a_win", "team_b_win", "draw"]


def brier_score(y_true, y_prob):
    """
    Multi-class Brier score: mean squared error between the predicted
    probability vector and the one-hot actual outcome, summed over classes.
    0 = perfect, 2 = worst possible for 3 mutually exclusive classes.

    y_true: list of outcome labels drawn from OUTCOME_CLASSES.
    y_prob: list of dicts with the same keys, e.g. {"team_a_win": .., ...}.
    """
    total = 0.0
    for outcome, probs in zip(y_true, y_prob):
        total += sum((probs[c] - (1.0 if c == outcome else 0.0)) ** 2 for c in OUTCOME_CLASSES)
    return total / len(y_true)


def log_loss(y_true, y_prob, eps=1e-12):
    """Mean negative log-likelihood of the actual outcome under the predicted distribution."""
    total = 0.0
    for outcome, probs in zip(y_true, y_prob):
        p = min(max(probs[outcome], eps), 1 - eps)
        total += -math.log(p)
    return total / len(y_true)


def calibration_curve(y_true, y_prob, n_bins=10):
    """
    Pooled one-vs-rest reliability curve: every (class, predicted probability)
    pair across all classes and matches is binned by predicted probability,
    and each bin reports how often that class actually occurred. A
    well-calibrated model has empirical_freq ~= mean_predicted in every bin.
    """
    edges = np.linspace(0, 1, n_bins + 1)
    preds, actuals = [], []

    for outcome, probs in zip(y_true, y_prob):
        for c in OUTCOME_CLASSES:
            preds.append(probs[c])
            actuals.append(1.0 if c == outcome else 0.0)

    preds = np.array(preds)
    actuals = np.array(actuals)

    bins = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (preds >= lo) & (preds <= hi if i == n_bins - 1 else preds < hi)
        count = int(mask.sum())
        bins.append({
            "bin_lower": float(lo),
            "bin_upper": float(hi),
            "mean_predicted": float(preds[mask].mean()) if count else None,
            "empirical_freq": float(actuals[mask].mean()) if count else None,
            "count": count,
        })
    return bins


def expected_calibration_error(bins):
    """Count-weighted average |confidence - empirical frequency| across calibration bins."""
    total = sum(b["count"] for b in bins)
    if total == 0:
        return 0.0
    return sum(
        (b["count"] / total) * abs(b["mean_predicted"] - b["empirical_freq"])
        for b in bins
        if b["count"] > 0
    )

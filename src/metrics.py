# src/metrics.py
"""Evaluation metrics for video-level scores.

Convention everywhere: label 1 = FAKE (positive class), 0 = REAL; score = raw P(fake).
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% Wilson score interval for a proportion k/n (None when n == 0)."""
    if n <= 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _arrays(y_true, scores):
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if y.shape != s.shape or y.ndim != 1:
        raise ValueError("y_true and scores must be 1-D arrays of equal length")
    if len(y) == 0:
        raise ValueError("No samples")
    if not set(np.unique(y)) <= {0, 1}:
        raise ValueError("Labels must be 0 (real) or 1 (fake)")
    if not np.all(np.isfinite(s)) or s.min() < 0 or s.max() > 1:
        raise ValueError("Scores must be finite values in [0, 1]")
    return y, s


def _rate(k: int, n: int) -> dict:
    ci = wilson_interval(k, n)
    return {"value": (k / n) if n else None, "count": k, "of": n,
            "ci95": list(ci) if ci else None}


def classification_metrics(y_true, scores, threshold: float = 0.5) -> dict:
    """Binary metrics at one threshold (score >= threshold -> predicted FAKE)."""
    y, s = _arrays(y_true, scores)
    pred = (s >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    n_real, n_fake = tn + fp, tp + fn
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / n_fake if n_fake else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and (precision + recall) > 0 else None)
    both = n_real > 0 and n_fake > 0
    return {
        "n": int(len(y)), "n_real": n_real, "n_fake": n_fake, "threshold": threshold,
        "accuracy": (tp + tn) / len(y), "precision": precision, "recall": recall, "f1": f1,
        "roc_auc": float(roc_auc_score(y, s)) if both else None,
        "pr_auc": float(average_precision_score(y, s)) if n_fake else None,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "false_positive_rate": _rate(fp, n_real),   # real videos flagged fake
        "false_negative_rate": _rate(fn, n_fake),   # fake videos passed as real
    }


def expected_calibration_error(y_true, probs, bins: int = 10) -> float:
    """Weighted mean |observed fake rate - mean predicted prob| over equal-width bins."""
    y, p = _arrays(y_true, probs)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def calibration_metrics(y_true, probs, bins: int = 10) -> dict:
    y, p = _arrays(y_true, probs)
    table = []
    idx = np.minimum((p * bins).astype(int), bins - 1)
    for b in range(bins):
        m = idx == b
        if m.any():
            table.append({"bin": f"{b / bins:.1f}-{(b + 1) / bins:.1f}", "n": int(m.sum()),
                          "mean_predicted": float(p[m].mean()), "observed_fake_rate": float(y[m].mean())})
    return {"ece": expected_calibration_error(y, p, bins), "brier": float(brier_score_loss(y, p)),
            "bins": table}


def _logit(s):
    s = np.clip(np.asarray(s, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(s / (1 - s))


def fit_platt(y_true, scores) -> dict:
    """Platt scaling: P = sigmoid(a * logit(score) + b), fitted by logistic regression.

    Fit it on validation data only; judge it on separate test data.
    """
    y, s = _arrays(y_true, scores)
    if len(set(y.tolist())) < 2:
        raise ValueError("Calibration needs both real and fake samples")
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(_logit(s).reshape(-1, 1), y)
    return {"method": "platt", "a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])}


def apply_platt(scores, calibration: dict) -> np.ndarray:
    z = calibration["a"] * _logit(scores) + calibration["b"]
    return 1.0 / (1.0 + np.exp(-z))


def decision_metrics(y_true, scores, real_below: float, fake_above: float,
                     inconclusive_mask=None) -> dict:
    """What the three-way decision does on a labelled set.

    ``inconclusive_mask`` marks videos forced to INCONCLUSIVE by other rules
    (e.g. too few usable clips). Error rates are computed among decided videos only,
    and ``coverage`` says what share of all videos received a label.
    """
    y, s = _arrays(y_true, scores)
    forced = np.zeros(len(y), bool) if inconclusive_mask is None else np.asarray(inconclusive_mask, bool)
    fake = (s >= fake_above) & ~forced
    real = (s <= real_below) & ~forced & ~fake
    inc = ~(fake | real)
    n_real, n_fake = int((y == 0).sum()), int((y == 1).sum())
    fp, fn = int((fake & (y == 0)).sum()), int((real & (y == 1)).sum())
    decided = int((fake | real).sum())
    wrong = fp + fn
    return {
        "n": int(len(y)), "predicted_fake": int(fake.sum()), "predicted_real": int(real.sum()),
        "inconclusive": int(inc.sum()), "coverage": decided / len(y),
        "false_positive_rate": _rate(fp, n_real),        # real videos labelled LIKELY_FAKE
        "false_negative_rate": _rate(fn, n_fake),        # fake videos labelled LIKELY_REAL
        "error_rate_among_decided": (wrong / decided) if decided else None,
        "inconclusive_by_class": {"real": int((inc & (y == 0)).sum()), "fake": int((inc & (y == 1)).sum())},
    }

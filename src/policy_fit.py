# src/policy_fit.py
"""Fit the decision policy from VALIDATION scores and write models/decision_policy.json.

Rule: thresholds are chosen on validation data only, to meet explicit error targets
(max false-positive rate on real videos, max false-negative rate on fake videos).
Meeting those two rates alone is not enough (a model with no signal can meet them by
labelling only the extreme tails), so the fit also requires a minimum share of videos
to receive a label and a maximum error rate among labelled videos. If the data cannot
support that, fitting FAILS instead of producing thresholds it does not justify. Judge the result on a separate
test set (see the evaluation report), never on the data used here.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np

from src.decision import SCHEMA_VERSION, DecisionPolicy
from src.metrics import (apply_platt, calibration_metrics, classification_metrics,
                         decision_metrics, fit_platt)


class PolicyFitError(RuntimeError):
    """Validation data cannot support a trustworthy decision policy."""


def _threshold_for_fpr(real_scores: np.ndarray, max_fpr: float) -> float:
    """Smallest t such that the share of real scores >= t is <= max_fpr."""
    cands = np.unique(np.concatenate([real_scores, [1.0]]))
    for t in cands:
        if (real_scores >= t).mean() <= max_fpr:
            # place just above the highest real score that must stay below t
            below = real_scores[real_scores < t]
            return float((below.max() + t) / 2) if below.size else float(t)
    return 1.0


def _threshold_for_fnr(fake_scores: np.ndarray, max_fnr: float) -> float:
    """Largest t such that the share of fake scores <= t is <= max_fnr."""
    cands = np.unique(np.concatenate([fake_scores, [0.0]]))[::-1]
    for t in cands:
        if (fake_scores <= t).mean() <= max_fnr:
            above = fake_scores[fake_scores > t]
            return float((above.min() + t) / 2) if above.size else float(t)
    return 0.0


def fit_policy(y_true, scores_by_agg: dict[str, list[float]], checkpoint_sha256: str,
               datasets: list[str], max_fpr: float = 0.05, max_fnr: float = 0.10,
               min_per_class: int = 20, min_coverage: float = 0.5,
               max_error_among_decided: float = 0.10, aggregation: str | None = None,
               min_face_coverage: float = 0.6, max_clip_spread: float = 0.5,
               calibrate: bool = False) -> dict:
    """Return a policy dict (schema v1) fitted on validation data.

    ``scores_by_agg`` maps "mean"/"median" to one score per validation video. When
    ``aggregation`` is None the one with the higher validation ROC-AUC is used and the
    comparison is recorded.
    """
    y = np.asarray(y_true, dtype=int)
    n_real, n_fake = int((y == 0).sum()), int((y == 1).sum())
    if min(n_real, n_fake) < min_per_class:
        raise PolicyFitError(
            f"Validation set too small: {n_real} real / {n_fake} fake videos; at least "
            f"{min_per_class} of each are required to place thresholds.")
    if not datasets:
        raise PolicyFitError("Name the validation dataset(s) so the policy records its provenance.")

    aucs = {k: classification_metrics(y, v)["roc_auc"] for k, v in scores_by_agg.items()}
    agg = aggregation or max(aucs, key=lambda k: aucs[k])
    if agg not in scores_by_agg:
        raise PolicyFitError(f"No scores for aggregation '{agg}'.")
    s = np.asarray(scores_by_agg[agg], dtype=float)

    t_fake = _threshold_for_fpr(s[y == 0], max_fpr)    # lowest cut that meets the FPR target
    t_real = _threshold_for_fnr(s[y == 1], max_fnr)    # highest cut that meets the FNR target
    if t_real < t_fake:
        real_below, fake_above = t_real, t_fake        # scores in between are INCONCLUSIVE
    else:
        # The model beats both targets with one cut. Any real_below <= t_real and any
        # fake_above >= t_fake still meet them, so use a narrow band around the midpoint.
        cut, half = (t_real + t_fake) / 2, 0.005
        real_below, fake_above = max(0.0, cut - half), min(1.0, cut + half)

    dm = decision_metrics(y, s, real_below, fake_above)
    fpr, fnr = dm["false_positive_rate"]["value"], dm["false_negative_rate"]["value"]
    err = dm["error_rate_among_decided"]
    problems = []
    if fpr > max_fpr + 1e-12 or fnr > max_fnr + 1e-12:    # safety net; holds by construction
        problems.append(f"error targets not met (FPR {fpr:.1%}, FNR {fnr:.1%})")
    if dm["coverage"] < min_coverage:
        problems.append(f"only {dm['coverage']:.0%} of videos would receive a label "
                        f"(minimum {min_coverage:.0%})")
    if err is None or err > max_error_among_decided:
        problems.append("error rate among labelled videos is "
                        + ("undefined" if err is None else f"{err:.1%}")
                        + f" (maximum {max_error_among_decided:.0%})")
    if problems:
        raise PolicyFitError(
            "Validation data cannot support a useful decision policy at these targets: "
            + "; ".join(problems) + f". Validation ROC-AUC is {aucs[agg]:.3f}. "
            "Relax the targets, improve the model, or use better validation data.")

    cal = fit_platt(y, s) if calibrate else None
    validation = {
        "datasets": list(datasets), "n_real": n_real, "n_fake": n_fake,
        "date": dt.date.today().isoformat(),
        "targets": {"max_false_positive_rate": max_fpr, "max_false_negative_rate": max_fnr,
                    "min_coverage": min_coverage, "max_error_among_decided": max_error_among_decided},
        "roc_auc_by_aggregation": aucs,
        "fitted_fields": ["aggregation", "real_below", "fake_above"] + (["calibration"] if cal else []),
        "default_fields_not_validated": ["min_face_coverage", "max_clip_spread"],
        "validation_set_decision_metrics": dm,
        "note": "These figures describe the data the thresholds were fitted on and are optimistic; "
                "use the held-out test report for performance claims.",
    }
    if cal:
        validation["validation_set_calibration"] = calibration_metrics(y, apply_platt(s, cal))
    policy = {"schema_version": SCHEMA_VERSION, "checkpoint_sha256": checkpoint_sha256,
              "aggregation": agg, "real_below": round(real_below, 6), "fake_above": round(fake_above, 6),
              "min_face_coverage": min_face_coverage, "max_clip_spread": max_clip_spread,
              "calibration": cal, "validation": validation}
    DecisionPolicy.from_dict(policy)   # must satisfy the same checks the app applies
    return policy


def write_policy(policy: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    return p

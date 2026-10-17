import numpy as np
import pytest

from src.metrics import (apply_platt, calibration_metrics, classification_metrics, decision_metrics,
                         expected_calibration_error, fit_platt, wilson_interval)


# ---------- classification metrics, checked by hand ----------
def test_hand_computed_confusion_and_rates():
    y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    s = [0.1, 0.2, 0.6, 0.9, 0.4, 0.7, 0.8, 0.9, 0.95, 0.3]
    m = classification_metrics(y, s, threshold=0.5)
    assert m["confusion"] == {"tn": 2, "fp": 2, "fn": 2, "tp": 4}
    assert m["accuracy"] == pytest.approx(0.6)
    assert m["precision"] == pytest.approx(4 / 6) and m["recall"] == pytest.approx(4 / 6)
    assert m["f1"] == pytest.approx(4 / 6)
    assert m["false_positive_rate"]["value"] == 0.5 and m["false_positive_rate"]["of"] == 4
    assert m["false_negative_rate"]["value"] == pytest.approx(2 / 6)
    assert m["roc_auc"] == pytest.approx(17.5 / 24)  # pairs ordered correctly: 2+3+3+3.5+4+2 of 24 (one tie)


def test_perfect_and_inverted_separation():
    y = [0, 0, 1, 1]
    assert classification_metrics(y, [0.1, 0.2, 0.8, 0.9])["roc_auc"] == 1.0
    assert classification_metrics(y, [0.9, 0.8, 0.2, 0.1])["roc_auc"] == 0.0


def test_threshold_is_inclusive_for_fake():
    m = classification_metrics([0, 1], [0.5, 0.5], threshold=0.5)
    assert m["confusion"]["fp"] == 1 and m["confusion"]["tp"] == 1


def test_single_class_gives_none_not_crash():
    m = classification_metrics([1, 1, 1], [0.9, 0.8, 0.7])
    assert m["roc_auc"] is None and m["false_positive_rate"]["value"] is None
    assert m["recall"] == 1.0 and m["pr_auc"] == pytest.approx(1.0)


def test_no_positive_predictions_precision_none():
    m = classification_metrics([0, 1], [0.1, 0.2], threshold=0.5)
    assert m["precision"] is None and m["f1"] is None and m["recall"] == 0.0


@pytest.mark.parametrize("y,s", [([], []), ([0, 1], [0.5]), ([0, 2], [0.1, 0.2]),
                                 ([0, 1], [0.1, 1.5]), ([0, 1], [0.1, float("nan")])])
def test_invalid_inputs_rejected(y, s):
    with pytest.raises(ValueError):
        classification_metrics(y, s)


# ---------- confidence intervals ----------
def test_wilson_interval_known_value():
    lo, hi = wilson_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)
    assert wilson_interval(0, 10)[0] == 0.0 and wilson_interval(10, 10)[1] == 1.0
    assert wilson_interval(0, 0) is None


def test_ci_widens_with_fewer_samples():
    a, b = wilson_interval(1, 10), wilson_interval(100, 1000)
    assert (a[1] - a[0]) > (b[1] - b[0])


# ---------- calibration ----------
def test_ece_zero_for_perfectly_calibrated_and_high_for_overconfident():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.02
    over = np.where(y == 1, 0.99, 0.99)             # claims 99% fake for everything
    assert expected_calibration_error(y, over) > 0.3


def test_calibration_table_and_brier():
    c = calibration_metrics([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2])
    assert c["brier"] == pytest.approx(((0.1) ** 2 + 0.1 ** 2 + 0.2 ** 2 + 0.2 ** 2) / 4)
    assert sum(b["n"] for b in c["bins"]) == 4


def test_platt_corrects_overconfident_scores():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 4000)
    z = np.where(y == 1, 1.0, -1.0) * 0.5 + rng.normal(0, 1.5, 4000)   # weak signal
    raw = 1 / (1 + np.exp(-4 * z))                                       # but 4x overconfident
    cal = fit_platt(y, raw)
    assert cal["a"] < 0.5                                                 # shrinks logits
    assert expected_calibration_error(y, apply_platt(raw, cal)) < expected_calibration_error(y, raw)


def test_platt_handles_saturated_scores_and_needs_two_classes():
    cal = fit_platt([0, 1, 0, 1], [0.0, 1.0, 0.0, 1.0])
    assert np.isfinite(cal["a"]) and np.isfinite(cal["b"])
    with pytest.raises(ValueError):
        fit_platt([1, 1], [0.5, 0.6])


# ---------- three-way decision metrics ----------
def test_decision_metrics_counts_and_coverage():
    y = [0, 0, 0, 1, 1, 1]
    s = [0.05, 0.4, 0.8, 0.5, 0.9, 0.1]
    d = decision_metrics(y, s, real_below=0.2, fake_above=0.7)
    assert (d["predicted_real"], d["predicted_fake"], d["inconclusive"]) == (2, 2, 2)
    assert d["coverage"] == pytest.approx(4 / 6)
    assert d["false_positive_rate"]["count"] == 1 and d["false_negative_rate"]["count"] == 1
    assert d["error_rate_among_decided"] == pytest.approx(2 / 4)
    assert d["inconclusive_by_class"] == {"real": 1, "fake": 1}


def test_forced_inconclusive_mask_is_respected():
    d = decision_metrics([0, 1], [0.01, 0.99], 0.2, 0.7, inconclusive_mask=[True, False])
    assert d["inconclusive"] == 1 and d["predicted_fake"] == 1 and d["predicted_real"] == 0


def test_all_inconclusive_has_no_error_rate():
    d = decision_metrics([0, 1], [0.4, 0.5], 0.2, 0.7)
    assert d["coverage"] == 0 and d["error_rate_among_decided"] is None

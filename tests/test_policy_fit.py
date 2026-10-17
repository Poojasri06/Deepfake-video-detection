import json

import numpy as np
import pytest

from src.decision import DecisionPolicy, load_policy
from src.metrics import decision_metrics
from src.policy_fit import PolicyFitError, fit_policy, write_policy

SHA = "c" * 64


def synthetic(n=200, sep=2.0, seed=0):
    """Scores for n real + n fake videos; ``sep`` controls how separable they are."""
    rng = np.random.default_rng(seed)
    y = np.array([0] * n + [1] * n)
    z = np.where(y == 1, sep, -sep) + rng.normal(0, 1.0, 2 * n)
    s = 1 / (1 + np.exp(-z))
    return y, s


def fit(y, s, **kw):
    return fit_policy(y, {"mean": s, "median": s}, SHA, ["synthetic"], **kw)


def test_separable_data_meets_targets_on_validation():
    y, s = synthetic(sep=3.0)
    p = fit(y, s, max_fpr=0.05, max_fnr=0.10)
    assert 0 <= p["real_below"] < p["fake_above"] <= 1
    d = decision_metrics(y, s, p["real_below"], p["fake_above"])
    assert d["false_positive_rate"]["value"] <= 0.05 and d["false_negative_rate"]["value"] <= 0.10
    assert 0 < d["coverage"] <= 1


def test_output_is_a_valid_policy_bound_to_checkpoint(tmp_path):
    y, s = synthetic()
    p = fit(y, s)
    path = write_policy(p, tmp_path / "models" / "decision_policy.json")
    loaded = load_policy(path, SHA)                       # same loader the app uses
    assert isinstance(loaded, DecisionPolicy) and loaded.checkpoint_sha256 == SHA
    assert loaded.validation["datasets"] == ["synthetic"]
    assert json.loads(path.read_text())["validation"]["default_fields_not_validated"] == [
        "min_face_coverage", "max_clip_spread"]


def test_stricter_targets_widen_inconclusive_band():
    y, s = synthetic(n=1000, sep=0.9, seed=4)             # noisy enough that targets matter
    free = dict(min_coverage=0.0, max_error_among_decided=1.0)   # isolate the threshold logic
    out = {}
    for name, (fpr, fnr) in {"loose": (0.20, 0.20), "strict": (0.02, 0.02)}.items():
        p = fit(y, s, max_fpr=fpr, max_fnr=fnr, **free)
        d = decision_metrics(y, s, p["real_below"], p["fake_above"])
        assert d["false_positive_rate"]["value"] <= fpr and d["false_negative_rate"]["value"] <= fnr
        out[name] = (p, d)
    (pl, dl), (ps, ds) = out["loose"], out["strict"]
    assert ps["fake_above"] - ps["real_below"] > pl["fake_above"] - pl["real_below"]
    assert ds["coverage"] < dl["coverage"]


def test_overlapping_scores_refuse_to_fit():
    y, s = synthetic(sep=0.0)                              # no signal at all
    with pytest.raises(PolicyFitError, match="cannot support a useful"):
        fit(y, s, max_fpr=0.05, max_fnr=0.05)


def test_too_few_videos_refuse_to_fit():
    y, s = synthetic(n=10)
    with pytest.raises(PolicyFitError, match="too small"):
        fit(y, s)


def test_missing_provenance_refused():
    y, s = synthetic()
    with pytest.raises(PolicyFitError, match="provenance"):
        fit_policy(y, {"mean": s}, SHA, [])


def test_aggregation_chosen_by_validation_auc_and_recorded():
    y, good = synthetic(sep=3.0, seed=1)
    _, bad = synthetic(sep=0.5, seed=2)
    p = fit_policy(y, {"mean": bad, "median": good}, SHA, ["synthetic"], max_fpr=0.2, max_fnr=0.2,
                   min_coverage=0.0, max_error_among_decided=1.0)
    assert p["aggregation"] == "median"
    aucs = p["validation"]["roc_auc_by_aggregation"]
    assert aucs["median"] > aucs["mean"]
    forced = fit_policy(y, {"mean": bad, "median": good}, SHA, ["synthetic"], aggregation="mean",
                        max_fpr=0.5, max_fnr=0.5, min_coverage=0.0, max_error_among_decided=1.0)
    assert forced["aggregation"] == "mean"


def test_unknown_aggregation_rejected():
    y, s = synthetic()
    with pytest.raises(PolicyFitError, match="aggregation"):
        fit_policy(y, {"mean": s}, SHA, ["x"], aggregation="median")


def test_calibration_included_only_on_request():
    y, s = synthetic(sep=2.0)
    assert fit(y, s)["calibration"] is None
    p = fit(y, s, calibrate=True)
    assert p["calibration"]["method"] == "platt" and "calibration" in p["validation"]["fitted_fields"]
    assert "validation_set_calibration" in p["validation"]


def test_heldout_performance_is_not_better_than_claimed_optimism():
    """Thresholds are fitted on one sample; a fresh sample from the same process
    will not match the validation figure exactly. The policy must say so."""
    y, s = synthetic(n=150, sep=1.5, seed=3)
    p = fit(y, s, max_fpr=0.05, max_fnr=0.10)
    assert "optimistic" in p["validation"]["note"]
    yt, st = synthetic(n=2000, sep=1.5, seed=99)
    test = decision_metrics(yt, st, p["real_below"], p["fake_above"])
    assert test["false_positive_rate"]["value"] < 0.15      # sanity bound, not a guarantee


def test_no_signal_cannot_pass_by_labelling_only_the_tails():
    """Regression: with zero signal the FPR/FNR targets alone are met by labelling extreme tails."""
    y, s = synthetic(n=500, sep=0.0, seed=5)
    with pytest.raises(PolicyFitError):
        fit(y, s, max_fpr=0.05, max_fnr=0.10)
    # and even when the error-rate guard is disabled the coverage guard still refuses
    with pytest.raises(PolicyFitError, match="receive a label"):
        fit(y, s, max_error_among_decided=1.0)


def test_model_better_than_targets_gets_narrow_band_with_valid_gap():
    y, s = synthetic(n=300, sep=4.0, seed=7)           # very separable: thresholds would cross
    p = fit(y, s, max_fpr=0.05, max_fnr=0.10)
    assert 0 < p["fake_above"] - p["real_below"] <= 0.011
    d = decision_metrics(y, s, p["real_below"], p["fake_above"])
    assert d["coverage"] > 0.95 and d["false_positive_rate"]["value"] <= 0.05

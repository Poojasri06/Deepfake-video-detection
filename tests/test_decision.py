import json
import math

import pytest

from src.decision import (INCONCLUSIVE, LIKELY_FAKE, LIKELY_REAL, UNVALIDATED, DecisionPolicy,
                          PolicyError, calibrate, decide, load_policy)
from src.pipeline import AnalysisResult, ClipResult
from src.video import VideoInfo

SHA = "a" * 64
VALID = {
    "schema_version": 1, "checkpoint_sha256": SHA, "aggregation": "mean",
    "real_below": 0.2, "fake_above": 0.7, "min_face_coverage": 0.6, "max_clip_spread": 0.5,
    "calibration": None,
    "validation": {"datasets": ["synthetic-unit-test"], "n_real": 10, "n_fake": 10},
}


def make_result(scores, coverage=1.0, status="ok"):
    clips = [ClipResult(i, 16, 16, True, s) for i, s in enumerate(scores)]
    mean = sum(scores) / len(scores) if scores else None
    srt = sorted(scores)
    med = srt[len(srt) // 2] if scores else None   # odd-length lists in tests
    return AnalysisResult(
        video=VideoInfo(1.0, 100, 30.0, 320, 240, 3.3), clips=clips, status=status,
        frames_analyzed=16 * len(scores), faces_detected=int(16 * len(scores) * coverage),
        face_coverage=coverage, score_mean=mean, score_median=med,
        score_min=min(scores) if scores else None, score_max=max(scores) if scores else None,
        elapsed_s=1.0, model_path="m.pth", model_sha256=SHA, checkpoint_epoch=1)


def policy(**over):
    return DecisionPolicy.from_dict({**VALID, **over})


# ---------- decision rules ----------
def test_no_policy_is_unvalidated_and_gives_no_label():
    d = decide(make_result([0.9, 0.9, 0.9]), None)
    assert d.label == UNVALIDATED and d.calibrated_p_fake is None
    assert "not been validated" in d.reasons[0]


def test_insufficient_evidence_is_inconclusive_even_without_policy():
    r = make_result([], status="insufficient_evidence")
    assert decide(r, None).label == INCONCLUSIVE
    assert decide(r, policy()).label == INCONCLUSIVE


@pytest.mark.parametrize("scores,label", [
    ([0.9, 0.95, 0.85], LIKELY_FAKE),
    ([0.05, 0.1, 0.02], LIKELY_REAL),
    ([0.4, 0.5, 0.45], INCONCLUSIVE),
])
def test_label_from_thresholds(scores, label):
    assert decide(make_result(scores), policy()).label == label


def test_threshold_boundaries_are_inclusive():
    p = policy(aggregation="mean")
    assert decide(make_result([0.7, 0.7, 0.7]), p).label == LIKELY_FAKE
    assert decide(make_result([0.2, 0.2, 0.2]), p).label == LIKELY_REAL
    assert decide(make_result([0.2001] * 3), p).label == INCONCLUSIVE


def test_low_face_coverage_forces_inconclusive():
    d = decide(make_result([0.95, 0.95, 0.95], coverage=0.5), policy())
    assert d.label == INCONCLUSIVE and "50%" in d.reasons[0]


def test_disagreeing_clips_force_inconclusive():
    d = decide(make_result([0.01, 0.99, 0.98]), policy())
    assert d.label == INCONCLUSIVE and "disagree" in d.reasons[0]


def test_aggregation_choice_is_respected():
    r = make_result([0.05, 0.75, 0.8])        # mean 0.533 (band), median 0.75 (fake)
    assert decide(r, policy(aggregation="mean", max_clip_spread=0.9)).label == INCONCLUSIVE
    assert decide(r, policy(aggregation="median", max_clip_spread=0.9)).label == LIKELY_FAKE


def test_likely_real_reason_does_not_claim_authenticity():
    d = decide(make_result([0.05, 0.05, 0.05]), policy())
    assert d.label == LIKELY_REAL and "does not prove" in d.reasons[0]


# ---------- calibration ----------
def test_calibrate_identity_and_monotone():
    ident = {"method": "platt", "a": 1.0, "b": 0.0}
    for s in (0.1, 0.5, 0.9):
        assert calibrate(s, ident) == pytest.approx(s, abs=1e-6)
    shrink = {"method": "platt", "a": 0.5, "b": 0.0}
    assert calibrate(0.99, shrink) < 0.99 and calibrate(0.99, shrink) > calibrate(0.9, shrink)


def test_calibrate_handles_saturated_scores():
    c = {"method": "platt", "a": 1.0, "b": 0.0}
    assert math.isfinite(calibrate(0.0, c)) and math.isfinite(calibrate(1.0, c))


def test_calibrated_probability_only_with_validated_calibration():
    r = make_result([0.9, 0.9, 0.9])
    assert decide(r, policy()).calibrated_p_fake is None
    cal = policy(calibration={"method": "platt", "a": 0.5, "b": 0.0})
    d = decide(r, cal)
    assert 0.5 < d.calibrated_p_fake < 0.9


# ---------- policy validation ----------
@pytest.mark.parametrize("patch", [
    {"real_below": 0.8, "fake_above": 0.5},
    {"aggregation": "max"},
    {"min_face_coverage": 1.5},
    {"max_clip_spread": 0},
    {"validation": {"datasets": [], "n_real": 5, "n_fake": 5}},
    {"validation": {"datasets": ["x"], "n_real": 0, "n_fake": 5}},
    {"calibration": {"method": "isotonic"}},
    {"schema_version": 99},
])
def test_invalid_policies_rejected(patch):
    with pytest.raises(PolicyError):
        policy(**patch)


def test_missing_key_rejected():
    d = dict(VALID)
    d.pop("fake_above")
    with pytest.raises(PolicyError):
        DecisionPolicy.from_dict(d)


def test_load_policy_missing_file_returns_none(tmp_path):
    assert load_policy(tmp_path / "none.json", SHA) is None


def test_load_policy_ok_and_checkpoint_binding(tmp_path):
    p = tmp_path / "pol.json"
    p.write_text(json.dumps(VALID))
    assert load_policy(p, SHA).fake_above == 0.7
    with pytest.raises(PolicyError, match="different checkpoint"):
        load_policy(p, "b" * 64)


def test_load_policy_malformed_json(tmp_path):
    p = tmp_path / "pol.json"
    p.write_text("{not json")
    with pytest.raises(PolicyError, match="Could not read"):
        load_policy(p, SHA)

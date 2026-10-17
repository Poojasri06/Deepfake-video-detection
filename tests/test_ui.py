"""Render the result view with real AnalysisResult/Decision objects (no upload needed)."""
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def _render(scores, coverage, status, with_policy, calibrated):
    from src.decision import DecisionPolicy, decide
    from src.ui import show_result
    from tests.test_decision import VALID, make_result

    res = make_result(scores, coverage=coverage, status=status)
    cal = {"method": "platt", "a": 1.0, "b": 0.0} if calibrated else None
    policy = DecisionPolicy.from_dict({**VALID, "calibration": cal}) if with_policy else None
    show_result(res, decide(res, policy), policy)


def run(scores, coverage=1.0, status="ok", with_policy=True, calibrated=False):
    at = AppTest.from_function(_render, kwargs=dict(scores=scores, coverage=coverage, status=status,
                                                    with_policy=with_policy, calibrated=calibrated),
                               default_timeout=30)
    return at.run()


def texts(at):
    return " ".join([e.value for e in at.error] + [e.value for e in at.warning] +
                    [e.value for e in at.success] + [e.value for e in at.info] +
                    [c.value for c in at.caption] + [m.value for m in at.markdown])


def test_fake_shows_error_banner_and_disclaimer():
    at = run([0.9, 0.95, 0.85])
    assert not at.exception
    assert any("Likely fake" in e.value for e in at.error)
    assert "not definitive forensic proof" in texts(at)


def test_real_shows_success_but_not_authenticity_claim():
    at = run([0.05, 0.05, 0.05])
    assert not at.exception and any("Likely real" in s.value for s in at.success)
    assert "does not prove" in texts(at) and "does not establish authenticity" in texts(at)


def test_inconclusive_shows_warning_with_reason():
    at = run([0.4, 0.5, 0.45])
    assert any("Inconclusive" in w.value for w in at.warning)
    assert "between the validated thresholds" in texts(at)


def test_unvalidated_shows_only_raw_score():
    at = run([0.9, 0.9, 0.9], with_policy=False)
    assert not at.exception and any("Not validated" in i.value for i in at.info)
    assert "uncalibrated" in texts(at) and not any("Calibrated P(fake)" in m.label for m in at.metric)


def test_calibrated_probability_shown_with_provenance():
    at = run([0.9, 0.9, 0.9], calibrated=True)
    assert any(m.label == "Calibrated P(fake)" for m in at.metric)
    assert "synthetic-unit-test" in texts(at) and "not the probability that the video is authentic" in texts(at)


def test_insufficient_evidence_view():
    at = run([], coverage=0.0, status="insufficient_evidence")
    assert not at.exception and any("Inconclusive" in w.value for w in at.warning)
    assert not any(m.label.startswith("Model score") for m in at.metric)

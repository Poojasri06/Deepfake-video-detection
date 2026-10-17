"""Smoke tests for the Streamlit app using Streamlit's AppTest runner."""
import pytest
import torch

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src.model import DeepfakeDetector  # noqa: E402

from pathlib import Path  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_stops_with_diagnostic_when_checkpoint_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPFAKE_CHECKPOINT", str(tmp_path / "missing.pth"))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("Checkpoint not found" in e.value for e in at.error)
    assert any("not definitive proof" in w.value for w in at.warning)
    assert len(at.button) == 0                      # no analysis possible without a model


def test_app_rejects_incompatible_checkpoint(monkeypatch, tmp_path):
    bad = tmp_path / "bad.pth"
    torch.save({"model_state_dict": {"w": torch.zeros(1)}}, bad)
    monkeypatch.setenv("DEEPFAKE_CHECKPOINT", str(bad))
    monkeypatch.setenv("DEEPFAKE_DEVICE", "cpu")
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert any("does not match" in e.value for e in at.error)


def test_app_loads_valid_checkpoint(monkeypatch, tmp_path):
    ok = tmp_path / "ok.pth"
    torch.save({"model_state_dict": DeepfakeDetector(pretrained=False).state_dict(), "epoch": 7}, ok)
    monkeypatch.setenv("DEEPFAKE_CHECKPOINT", str(ok))
    monkeypatch.setenv("DEEPFAKE_DEVICE", "cpu")
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception and not at.error
    assert any("Deepfake Video Analysis" in t.value for t in at.title)


def _write_ckpt(path):
    torch.save({"model_state_dict": DeepfakeDetector(pretrained=False).state_dict(), "epoch": 1}, path)


def _policy(sha):
    return {"schema_version": 1, "checkpoint_sha256": sha, "aggregation": "mean", "real_below": 0.2,
            "fake_above": 0.7, "min_face_coverage": 0.6, "max_clip_spread": 0.5, "calibration": None,
            "validation": {"datasets": ["unit-test"], "n_real": 5, "n_fake": 5}}


def test_app_without_policy_reports_not_validated(monkeypatch, tmp_path):
    ck = tmp_path / "ok.pth"
    _write_ckpt(ck)
    monkeypatch.setenv("DEEPFAKE_CHECKPOINT", str(ck))
    monkeypatch.setenv("DEEPFAKE_DEVICE", "cpu")
    monkeypatch.setenv("DEEPFAKE_POLICY", str(tmp_path / "none.json"))
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception and not at.error
    assert any("not validated" in m.value for m in at.sidebar.markdown)


def test_app_loads_matching_policy(monkeypatch, tmp_path):
    import json
    from src.checkpoint import load_trained_model
    ck = tmp_path / "ok.pth"
    _write_ckpt(ck)
    pol = tmp_path / "pol.json"
    pol.write_text(json.dumps(_policy(load_trained_model(ck, "cpu").sha256)))
    monkeypatch.setenv("DEEPFAKE_CHECKPOINT", str(ck))
    monkeypatch.setenv("DEEPFAKE_DEVICE", "cpu")
    monkeypatch.setenv("DEEPFAKE_POLICY", str(pol))
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception and not at.error
    assert any("validated policy loaded" in m.value for m in at.sidebar.markdown)


def test_app_rejects_policy_for_other_checkpoint(monkeypatch, tmp_path):
    import json
    ck = tmp_path / "ok.pth"
    _write_ckpt(ck)
    pol = tmp_path / "pol.json"
    pol.write_text(json.dumps(_policy("f" * 64)))
    monkeypatch.setenv("DEEPFAKE_CHECKPOINT", str(ck))
    monkeypatch.setenv("DEEPFAKE_DEVICE", "cpu")
    monkeypatch.setenv("DEEPFAKE_POLICY", str(pol))
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert any("different checkpoint" in e.value for e in at.error)

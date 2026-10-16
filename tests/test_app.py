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

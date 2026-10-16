"""Checkpoint loading tests.

These use a *synthetic* checkpoint (randomly initialised weights saved in the
exact format ``train.py`` writes). They verify loader behaviour only; they say
nothing about the accuracy of the real trained model.
"""
import pytest
import torch

from src.checkpoint import CheckpointError, load_trained_model, resolve_device
from src.model import CLASS_NAMES, FAKE_INDEX, REAL_INDEX, DeepfakeDetector


@pytest.fixture(scope="module")
def synthetic_state():
    torch.manual_seed(0)
    return DeepfakeDetector(pretrained=False).state_dict()


def _save_train_format(state, path, **extra):
    torch.save({"model_state_dict": state, "optimizer_state": {}, "config": {"image_size": 112},
                "epoch": 3, "best_val_f1": 0.5, **extra}, path)


def test_class_mapping_matches_training():
    assert CLASS_NAMES == ("real", "fake")
    assert (REAL_INDEX, FAKE_INDEX) == (0, 1)


def test_forward_shape_is_two_logits():
    m = DeepfakeDetector(pretrained=False).eval()
    with torch.inference_mode():
        out = m(torch.zeros(2, 3, 3, 64, 64))
    assert out.shape == (2, 2)


def test_missing_checkpoint_raises_clear_error(tmp_path):
    with pytest.raises(CheckpointError, match="not found"):
        load_trained_model(tmp_path / "nope.pth", device="cpu")


def test_loads_train_format_and_preserves_weights(tmp_path, synthetic_state):
    p = tmp_path / "ck.pth"
    _save_train_format(synthetic_state, p)
    loaded = load_trained_model(p, device="cpu")
    assert loaded.epoch == 3 and loaded.best_val_f1 == 0.5
    assert loaded.train_config == {"image_size": 112}
    assert len(loaded.sha256) == 64
    assert not loaded.model.training
    for k, v in synthetic_state.items():
        assert torch.equal(loaded.model.state_dict()[k], v), k


def test_loads_bare_state_dict(tmp_path, synthetic_state):
    p = tmp_path / "bare.pth"
    torch.save(synthetic_state, p)
    assert load_trained_model(p, device="cpu").epoch is None


def test_missing_key_is_rejected_not_randomised(tmp_path, synthetic_state):
    broken = dict(synthetic_state)
    broken.pop("fc.weight")
    p = tmp_path / "missing.pth"
    _save_train_format(broken, p)
    with pytest.raises(CheckpointError, match="does not match"):
        load_trained_model(p, device="cpu")


def test_unexpected_key_is_rejected(tmp_path, synthetic_state):
    broken = dict(synthetic_state)
    broken["extra.weight"] = torch.zeros(1)
    p = tmp_path / "extra.pth"
    _save_train_format(broken, p)
    with pytest.raises(CheckpointError, match="does not match"):
        load_trained_model(p, device="cpu")


def test_wrong_architecture_is_rejected(tmp_path):
    # The tiny CNN+LSTM defined in the old app.py must not load silently.
    tiny = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3))
    p = tmp_path / "tiny.pth"
    torch.save(tiny.state_dict(), p)
    with pytest.raises(CheckpointError, match="does not match"):
        load_trained_model(p, device="cpu")


def test_wrong_shape_is_rejected(tmp_path, synthetic_state):
    broken = dict(synthetic_state)
    broken["fc.weight"] = torch.zeros(3, 256)
    p = tmp_path / "shape.pth"
    _save_train_format(broken, p)
    with pytest.raises(CheckpointError, match="does not match"):
        load_trained_model(p, device="cpu")


def test_corrupt_file_is_rejected(tmp_path):
    p = tmp_path / "corrupt.pth"
    p.write_bytes(b"not a torch file")
    with pytest.raises(CheckpointError, match="Could not read"):
        load_trained_model(p, device="cpu")


def test_non_dict_checkpoint_is_rejected(tmp_path):
    p = tmp_path / "list.pth"
    torch.save([1, 2, 3], p)
    with pytest.raises(CheckpointError, match="Unrecognised"):
        load_trained_model(p, device="cpu")


def test_unsupported_device_is_rejected():
    with pytest.raises(CheckpointError, match="Unsupported device"):
        resolve_device("tpu")


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA present")
def test_explicit_cuda_without_cuda_is_rejected():
    with pytest.raises(CheckpointError, match="CUDA is not available"):
        resolve_device("cuda")


def test_auto_device_resolves():
    assert resolve_device("auto").type in {"cpu", "cuda"}

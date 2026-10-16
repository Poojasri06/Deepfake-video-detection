import cv2
import numpy as np
import pytest
import torch

from src.checkpoint import load_trained_model
from src.model import DeepfakeDetector


def write_video(path, n_frames=120, size=(160, 120), fps=30, fourcc="MJPG"):
    """Write a small synthetic video whose frame content varies over time."""
    w, h = size
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
    if not vw.isOpened():
        pytest.skip(f"OpenCV cannot write {fourcc} here")
    rng = np.random.default_rng(0)
    for i in range(n_frames):
        frame = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        frame[:, : (i * 2) % w] //= 2
        vw.write(frame)
    vw.release()
    return path


class FixedBoxDetector:
    """Stand-in for MTCNN: always reports one face box."""
    def detect(self, rgb):
        return np.array([[20.0, 15.0, 100.0, 95.0]]), np.array([0.99])


class NoFaceDetector:
    def detect(self, rgb):
        return None, [None]


class EveryNthMissingDetector:
    """Misses a face on every n-th call."""
    def __init__(self, n):
        self.n, self.calls = n, 0

    def detect(self, rgb):
        self.calls += 1
        if self.calls % self.n == 0:
            return None, [None]
        return np.array([[20.0, 15.0, 100.0, 95.0]]), np.array([0.99])


@pytest.fixture(scope="session")
def loaded_model(tmp_path_factory):
    """Randomly initialised model saved in train.py format. Tests loader/pipeline
    mechanics only; its outputs carry no meaning about real accuracy."""
    torch.manual_seed(0)
    p = tmp_path_factory.mktemp("ck") / "synthetic.pth"
    torch.save({"model_state_dict": DeepfakeDetector(pretrained=False).state_dict(),
                "config": {"image_size": 112}, "epoch": 1, "best_val_f1": 0.0}, p)
    return load_trained_model(p, device="cpu")


@pytest.fixture()
def video_path(tmp_path):
    return write_video(tmp_path / "clip.avi", n_frames=300)

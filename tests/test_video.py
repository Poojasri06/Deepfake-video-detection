import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

from src.config import InferenceConfig
from src.video import (VideoError, crop_face, fill_missing_faces, plan_clips,
                       probe_video, to_tensor)
from tests.conftest import FixedBoxDetector, NoFaceDetector, write_video

CFG = InferenceConfig(checkpoint_path="unused")


# ---------- preprocessing matches training ----------
def test_to_tensor_equals_training_transform():
    face = np.random.default_rng(1).integers(0, 255, (112, 112, 3), dtype=np.uint8)
    train_tf = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor(),
                                   transforms.Normalize([0.5] * 3, [0.5] * 3)])
    expected = train_tf(Image.fromarray(face))
    assert torch.allclose(to_tensor(face), expected, atol=1e-6)
    assert to_tensor(face).shape == (3, 112, 112)


def test_crop_face_shape_and_clipping():
    rgb = np.zeros((100, 200, 3), dtype=np.uint8)

    class Edge:
        def detect(self, _):
            return np.array([[-20.0, -5.0, 500.0, 300.0]]), np.array([0.9])

    out = crop_face(rgb, Edge(), 112)
    assert out.shape == (112, 112, 3) and out.dtype == np.uint8


def test_crop_face_none_cases():
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    assert crop_face(rgb, NoFaceDetector(), 112) is None

    class Degenerate:
        def detect(self, _):
            return np.array([[50.0, 50.0, 50.0, 50.0]]), np.array([0.9])

    assert crop_face(rgb, Degenerate(), 112) is None

    class Boom:
        def detect(self, _):
            raise RuntimeError("detector failed")

    assert crop_face(rgb, Boom(), 112) is None


def test_face_selection_first_vs_largest():
    rgb = np.zeros((200, 200, 3), dtype=np.uint8)
    rgb[100:190, 100:190] = 255          # large white face
    rgb[10:30, 10:30] = 0                # small black face

    class Two:
        def detect(self, _):
            return np.array([[10.0, 10.0, 30.0, 30.0], [100.0, 100.0, 190.0, 190.0]]), np.array([0.99, 0.9])

    assert crop_face(rgb, Two(), 16, "first").mean() < 10
    assert crop_face(rgb, Two(), 16, "largest").mean() > 245


def test_fill_missing_faces_uses_nearest_earlier_face():
    a, b = np.full((2, 2, 3), 1, np.uint8), np.full((2, 2, 3), 2, np.uint8)
    out = fill_missing_faces([None, a, None, None, b, None])
    assert [int(x[0, 0, 0]) for x in out] == [1, 1, 1, 1, 2, 2]


# ---------- clip planning ----------
def test_plan_clips_spread_non_overlapping():
    starts, stride = plan_clips(300, CFG)
    span = (CFG.frames_per_clip - 1) * CFG.stride + 1
    assert stride == CFG.stride and len(starts) == CFG.num_clips
    assert starts[0] == 0 and starts[-1] == 300 - span
    assert all(b - a >= span for a, b in zip(starts, starts[1:]))


def test_plan_clips_short_video_reduces_stride():
    starts, stride = plan_clips(20, CFG)
    assert stride == 1 and starts == [0]


def test_plan_clips_fewer_clips_when_video_is_short():
    starts, _ = plan_clips(80, CFG)
    assert 1 <= len(starts) < CFG.num_clips


# ---------- validation ----------
def test_probe_ok(video_path):
    info = probe_video(video_path, CFG)
    assert info.frames == 300 and info.width == 160 and info.height == 120
    assert info.fps == pytest.approx(30, abs=1) and info.duration_s == pytest.approx(10, abs=0.5)


def test_probe_missing_file(tmp_path):
    with pytest.raises(VideoError) as e:
        probe_video(tmp_path / "nope.mp4", CFG)
    assert e.value.code == "not_found"


def test_probe_unsupported_extension(tmp_path):
    p = tmp_path / "x.mkv"
    p.write_bytes(b"123")
    with pytest.raises(VideoError) as e:
        probe_video(p, CFG)
    assert e.value.code == "unsupported_format"


def test_probe_empty_file(tmp_path):
    p = tmp_path / "x.mp4"
    p.write_bytes(b"")
    with pytest.raises(VideoError) as e:
        probe_video(p, CFG)
    assert e.value.code == "empty_file"


def test_probe_corrupt_file(tmp_path):
    p = tmp_path / "x.mp4"
    p.write_bytes(b"this is not a video" * 100)
    with pytest.raises(VideoError) as e:
        probe_video(p, CFG)
    assert e.value.code in {"cannot_open", "corrupt", "unknown_length"}


def test_probe_too_large(video_path):
    cfg = InferenceConfig(checkpoint_path="unused", max_file_mb=0.0001)
    with pytest.raises(VideoError) as e:
        probe_video(video_path, cfg)
    assert e.value.code == "too_large"


def test_probe_too_long(video_path):
    cfg = InferenceConfig(checkpoint_path="unused", max_duration_s=1)
    with pytest.raises(VideoError) as e:
        probe_video(video_path, cfg)
    assert e.value.code == "too_long"


def test_probe_too_short(tmp_path):
    p = write_video(tmp_path / "short.avi", n_frames=8)
    with pytest.raises(VideoError) as e:
        probe_video(p, CFG)
    assert e.value.code == "too_short"


def test_config_validation():
    with pytest.raises(ValueError):
        InferenceConfig(checkpoint_path="x", face_selection="best")
    with pytest.raises(ValueError):
        InferenceConfig(checkpoint_path="x", stride=0)


# ---------- frame reading hits the intended frames ----------
@pytest.mark.parametrize("fourcc,ext", [("MJPG", "avi"), ("mp4v", "mp4")])
def test_read_clip_returns_intended_frames(tmp_path, fourcc, ext):
    import cv2
    from src.video import read_clip

    p = tmp_path / f"ramp.{ext}"
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*fourcc), 30, (64, 64))
    if not vw.isOpened():
        pytest.skip(f"cannot write {fourcc}")
    for i in range(100):                       # frame i is flat gray level 2*i
        vw.write(np.full((64, 64, 3), 2 * i, np.uint8))
    vw.release()

    cap = cv2.VideoCapture(str(p))
    frames = read_clip(cap, start=10, stride=3, n_frames=5)
    cap.release()
    got = [float(f.mean()) for f in frames]
    want = [2 * (10 + 3 * k) for k in range(5)]     # 20, 26, 32, 38, 44
    assert got == pytest.approx(want, abs=4)

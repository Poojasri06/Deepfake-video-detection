# src/video.py
"""Video validation, clip planning, frame reading and face preprocessing.

Face crops and tensor conversion reproduce ``train.py``:
RGB frame -> MTCNN box -> clip box to frame -> crop -> cv2.resize to
(image_size, image_size) -> ToTensor -> Normalize(0.5, 0.5) -> (C, H, W).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import torch

from src.config import ALLOWED_EXTENSIONS, InferenceConfig


class VideoError(Exception):
    """A user-facing problem with the uploaded video. ``code`` is machine readable."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class FaceDetector(Protocol):
    def detect(self, rgb: np.ndarray):  # -> (boxes | None, probs)
        ...


@dataclass(frozen=True)
class VideoInfo:
    size_mb: float
    frames: int
    fps: float | None
    width: int
    height: int
    duration_s: float | None


def probe_video(path: str | Path, cfg: InferenceConfig) -> VideoInfo:
    p = Path(path)
    if not p.is_file():
        raise VideoError("not_found", "The video file could not be found.")
    if p.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise VideoError("unsupported_format",
                         f"Unsupported file type '{p.suffix}'. Use MP4, AVI or MOV.")
    size = p.stat().st_size
    if size == 0:
        raise VideoError("empty_file", "The uploaded file is empty.")
    size_mb = size / (1024 * 1024)
    if size_mb > cfg.max_file_mb:
        raise VideoError("too_large",
                         f"The video is {size_mb:.0f} MB; the limit is {cfg.max_file_mb:.0f} MB.")

    cap = cv2.VideoCapture(str(p))
    try:
        if not cap.isOpened():
            raise VideoError("cannot_open", "The video could not be opened (unsupported codec or corrupt file).")
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = float(fps) if fps and math.isfinite(fps) and fps > 0 else None
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        ok, first = cap.read()
        if not ok or first is None:
            raise VideoError("corrupt", "No frames could be decoded from this video.")
        if frames <= 0:
            raise VideoError("unknown_length",
                             "The video length could not be determined. Re-encode it as a standard MP4.")
        height, width = (first.shape[0], first.shape[1]) if not (width and height) else (height, width)
    finally:
        cap.release()

    duration = frames / fps if fps else None
    if duration is not None and duration > cfg.max_duration_s:
        raise VideoError("too_long",
                         f"The video is {duration:.0f} s long; the limit is {cfg.max_duration_s:.0f} s.")
    if frames < cfg.frames_per_clip:
        raise VideoError("too_short",
                         f"The video has only {frames} frames; at least {cfg.frames_per_clip} are required.")
    return VideoInfo(size_mb, frames, fps, width, height, duration)


def plan_clips(total_frames: int, cfg: InferenceConfig) -> tuple[list[int], int]:
    """Start frames of non-overlapping clips spread across the video, and the stride used.

    Uses the training stride. Only videos shorter than one training clip get a
    smaller stride so that a full 16-frame clip still spans the video.
    """
    fpc, stride = cfg.frames_per_clip, cfg.stride
    span = (fpc - 1) * stride + 1
    if total_frames < span:
        stride = max(1, (total_frames - 1) // (fpc - 1))
        span = (fpc - 1) * stride + 1
    last_start = max(total_frames - span, 0)
    n = min(cfg.num_clips, last_start // span + 1)  # clips never overlap
    starts = sorted(set(np.linspace(0, last_start, n).round().astype(int).tolist()))
    return starts, stride


def read_clip(cap: cv2.VideoCapture, start: int, stride: int, n_frames: int) -> list[np.ndarray]:
    """Read ``n_frames`` RGB frames beginning at ``start``, ``stride`` frames apart."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames: list[np.ndarray] = []
    for _ in range(n_frames):
        ok, bgr = cap.read()
        if not ok or bgr is None:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        for _ in range(stride - 1):
            cap.grab()
    return frames


def crop_face(rgb: np.ndarray, detector: FaceDetector, image_size: int,
              selection: str = "first") -> np.ndarray | None:
    """Return an (image_size, image_size, 3) uint8 RGB face crop or None."""
    try:
        boxes, _ = detector.detect(rgb)
    except Exception:
        return None
    if boxes is None or len(boxes) == 0 or boxes[0] is None:
        return None
    if selection == "largest":
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        box = boxes[int(np.argmax(areas))]
    else:
        box = boxes[0]
    x1, y1, x2, y2 = (int(v) for v in box)
    h, w = rgb.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    face = rgb[y1:y2, x1:x2]
    if face.size == 0:
        return None
    return cv2.resize(face, (image_size, image_size))


def to_tensor(face: np.ndarray) -> torch.Tensor:
    """uint8 HWC RGB -> float CHW in [-1, 1] (== ToTensor + Normalize(0.5, 0.5))."""
    t = torch.from_numpy(np.ascontiguousarray(face)).permute(2, 0, 1).float().div_(255.0)
    return (t - 0.5) / 0.5


def fill_missing_faces(faces: list[np.ndarray | None]) -> list[np.ndarray]:
    """Replace frames without a face by the nearest earlier face (or the first one).

    Mirrors the padding used in training, which repeated the last valid face.
    Callers must have checked that at least one face exists.
    """
    first = next(f for f in faces if f is not None)
    out, last = [], first
    for f in faces:
        last = f if f is not None else last
        out.append(last)
    return out

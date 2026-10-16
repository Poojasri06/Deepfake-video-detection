# src/pipeline.py
"""Video-level inference: sample clips across the video, crop faces, run the model.

Outputs RAW model scores only. They are NOT calibrated probabilities and no
REAL/FAKE decision is made here; thresholds must come from validation data.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch

from src.checkpoint import LoadedModel
from src.config import InferenceConfig
from src.model import FAKE_INDEX
from src.video import (FaceDetector, VideoError, VideoInfo, crop_face, fill_missing_faces,
                       plan_clips, probe_video, read_clip, to_tensor)

log = logging.getLogger("deepfake.pipeline")


@dataclass
class ClipResult:
    start_frame: int
    frames_read: int
    frames_with_face: int
    used: bool
    p_fake: float | None = None   # raw softmax P(class FAKE) for this clip
    reason: str | None = None     # why the clip was not used


@dataclass
class AnalysisResult:
    video: VideoInfo
    clips: list[ClipResult]
    status: str                   # "ok" | "insufficient_evidence"
    frames_analyzed: int
    faces_detected: int
    face_coverage: float          # faces / frames read, over all attempted clips
    score_mean: float | None      # mean of clip P(fake) -- raw, uncalibrated
    score_median: float | None
    score_min: float | None
    score_max: float | None
    elapsed_s: float
    model_path: str
    model_sha256: str
    checkpoint_epoch: int | None
    warnings: list[str] = field(default_factory=list)

    @property
    def clips_used(self) -> int:
        return sum(c.used for c in self.clips)


def build_face_detector(device: torch.device) -> FaceDetector:
    """MTCNN configured exactly as in train.py."""
    try:
        from facenet_pytorch import MTCNN
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "facenet-pytorch is required for face detection. Install with: "
            "pip install --no-deps -r requirements-face.txt"
        ) from e
    return MTCNN(select_largest=True, device=device, post_process=False)


def aggregate(scores: list[float]) -> dict[str, float | None]:
    """Clip-level summaries. Which one to trust is decided in the evaluation stage."""
    if not scores:
        return {"mean": None, "median": None, "min": None, "max": None}
    a = np.asarray(scores, dtype=float)
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "min": float(a.min()), "max": float(a.max())}


def analyze_video(video_path: str | Path, loaded: LoadedModel, cfg: InferenceConfig,
                  detector: FaceDetector | None = None,
                  progress: Callable[[int, int], None] | None = None) -> AnalysisResult:
    t0 = time.perf_counter()
    info = probe_video(video_path, cfg)
    device = next(loaded.model.parameters()).device
    detector = detector or build_face_detector(device)

    starts, stride = plan_clips(info.frames, cfg)
    warnings: list[str] = []
    if min(info.width, info.height) < 240:
        warnings.append(f"Low resolution ({info.width}x{info.height}); the model was not evaluated "
                        "on low-quality video, so results may be less reliable.")
    if stride != cfg.stride:
        warnings.append(f"Short video: frame stride reduced from {cfg.stride} to {stride}.")

    clips: list[ClipResult] = []
    tensors: list[torch.Tensor] = []
    clip_index: list[int] = []
    cap = cv2.VideoCapture(str(video_path))
    try:
        for i, start in enumerate(starts):
            if time.perf_counter() - t0 > cfg.max_processing_s:
                warnings.append(f"Time budget of {cfg.max_processing_s:.0f}s reached; "
                                f"analysed {i} of {len(starts)} planned clips.")
                break
            frames = read_clip(cap, start, stride, cfg.frames_per_clip)
            faces = [crop_face(f, detector, cfg.image_size, cfg.face_selection) for f in frames]
            n_face = sum(f is not None for f in faces)
            clip = ClipResult(start, len(frames), n_face, used=False)
            if len(frames) < cfg.frames_per_clip:
                clip.reason = "could not decode a full clip"
            elif n_face / len(frames) < cfg.min_clip_face_fraction:
                clip.reason = f"face found in only {n_face}/{len(frames)} frames"
            else:
                tensors.append(torch.stack([to_tensor(f) for f in fill_missing_faces(faces)]))
                clip_index.append(len(clips))
                clip.used = True
            clips.append(clip)
            if progress:
                progress(i + 1, len(starts))
    finally:
        cap.release()

    if tensors:
        batch = torch.stack(tensors).to(device)
        with torch.inference_mode():
            probs = torch.softmax(loaded.model(batch), dim=1)[:, FAKE_INDEX].cpu().tolist()
        for ci, p in zip(clip_index, probs):
            clips[ci].p_fake = float(p)

    frames_read = sum(c.frames_read for c in clips)
    faces_found = sum(c.frames_with_face for c in clips)
    coverage = faces_found / frames_read if frames_read else 0.0
    scores = [c.p_fake for c in clips if c.used and c.p_fake is not None]

    status = "ok"
    if len(scores) < cfg.min_valid_clips:
        status = "insufficient_evidence"
        warnings.append(
            f"Only {len(scores)} usable clip(s); at least {cfg.min_valid_clips} are needed. "
            "Faces were missing, too small, or the video could not be decoded reliably.")
        agg = aggregate([])
    else:
        agg = aggregate(scores)
    if status == "ok" and coverage < 0.9:
        warnings.append(f"Face detected in {coverage:.0%} of sampled frames; "
                        "missing faces were filled with the nearest detected face.")

    elapsed = time.perf_counter() - t0
    log.info("analysed: frames=%d clips_used=%d/%d status=%s elapsed=%.1fs",
             frames_read, len(scores), len(clips), status, elapsed)
    return AnalysisResult(
        video=info, clips=clips, status=status, frames_analyzed=frames_read,
        faces_detected=faces_found, face_coverage=coverage,
        score_mean=agg["mean"], score_median=agg["median"],
        score_min=agg["min"], score_max=agg["max"], elapsed_s=elapsed,
        model_path=Path(loaded.path).name, model_sha256=loaded.sha256,
        checkpoint_epoch=loaded.epoch, warnings=warnings)

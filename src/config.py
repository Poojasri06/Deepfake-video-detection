# src/config.py
"""Inference configuration. Values marked TRAINING must match train.py."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "output" / "deepfake_cnn_lstm.pth"
ALLOWED_EXTENSIONS = (".mp4", ".avi", ".mov")


def _env(name: str, default):
    return os.environ.get(name, default)


@dataclass(frozen=True)
class InferenceConfig:
    # --- model ---
    checkpoint_path: str = field(default_factory=lambda: _env("DEEPFAKE_CHECKPOINT", str(DEFAULT_CHECKPOINT)))
    device: str = field(default_factory=lambda: _env("DEEPFAKE_DEVICE", "auto"))

    # --- TRAINING: must match src/train.py CONFIG ---
    frames_per_clip: int = 16
    stride: int = 2
    image_size: int = 112
    # Training used boxes[0] from MTCNN.detect (highest detection confidence).
    # "largest" is available but is NOT what the model was trained with.
    face_selection: str = "first"

    # --- sampling across the video ---
    num_clips: int = 5                    # non-overlapping clips spread over the video
    min_clip_face_fraction: float = 0.75  # clip is dropped below this share of frames with a face
    min_valid_clips: int = 2              # fewer usable clips -> insufficient evidence

    # --- limits ---
    max_file_mb: float = field(default_factory=lambda: float(_env("DEEPFAKE_MAX_MB", 200)))
    max_duration_s: float = 600.0
    max_processing_s: float = field(default_factory=lambda: float(_env("DEEPFAKE_MAX_SECONDS", 180)))

    def __post_init__(self):
        if self.frames_per_clip < 2 or self.stride < 1 or self.image_size < 16:
            raise ValueError("frames_per_clip>=2, stride>=1 and image_size>=16 are required")
        if self.face_selection not in {"first", "largest"}:
            raise ValueError("face_selection must be 'first' or 'largest'")
        if not 0 < self.min_clip_face_fraction <= 1:
            raise ValueError("min_clip_face_fraction must be in (0, 1]")
        if self.num_clips < 1 or self.min_valid_clips < 1:
            raise ValueError("num_clips and min_valid_clips must be >= 1")

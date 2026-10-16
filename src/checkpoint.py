# src/checkpoint.py
"""Strict checkpoint loading. Never falls back to random or different weights."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import torch

from src.model import DeepfakeDetector


class CheckpointError(RuntimeError):
    """Raised when the trained checkpoint is missing, unreadable or incompatible."""


@dataclass
class LoadedModel:
    model: DeepfakeDetector
    path: str
    sha256: str
    train_config: dict = field(default_factory=dict)
    epoch: int | None = None
    best_val_f1: float | None = None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_device(requested: str = "auto") -> torch.device:
    """'auto' picks CUDA when available. Explicit 'cuda' without CUDA is an error."""
    requested = (requested or "auto").lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise CheckpointError("Device 'cuda' requested but CUDA is not available.")
    if requested != "cpu" and not requested.startswith("cuda"):
        raise CheckpointError(f"Unsupported device '{requested}'. Use auto, cpu or cuda.")
    return torch.device(requested)


def load_trained_model(path: str | Path, device: str = "auto") -> LoadedModel:
    """Load a checkpoint written by ``train.py`` into the exact training architecture.

    Accepts ``{"model_state_dict": ...}`` (train.py format) or a bare state dict.
    Uses ``strict=True``: any missing, unexpected or wrongly shaped key raises
    ``CheckpointError`` instead of leaving random weights in the model.
    """
    p = Path(path)
    if not p.is_file():
        raise CheckpointError(
            f"Checkpoint not found: {p}. Place the trained file there or set the "
            "model path in the configuration. No fallback model is used."
        )
    dev = resolve_device(device)

    try:
        raw = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:  # corrupt / not a torch file
        raise CheckpointError(f"Could not read checkpoint {p}: {e}") from e

    meta = {}
    if isinstance(raw, dict) and "model_state_dict" in raw:
        state = raw["model_state_dict"]
        meta = raw
    elif isinstance(raw, dict):
        state = raw
    else:
        raise CheckpointError(
            f"Unrecognised checkpoint content ({type(raw).__name__}); expected a "
            "state dict or a dict with 'model_state_dict'."
        )

    model = DeepfakeDetector(pretrained=False)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        raise CheckpointError(
            "Checkpoint does not match the CNN+LSTM architecture from train.py "
            f"(EfficientNet-B0 + BiLSTM). Refusing to run with partial weights.\n{e}"
        ) from e

    model.to(dev).eval()
    return LoadedModel(
        model=model,
        path=str(p),
        sha256=_sha256(p),
        train_config=dict(meta.get("config", {}) or {}),
        epoch=meta.get("epoch"),
        best_val_f1=meta.get("best_val_f1"),
    )

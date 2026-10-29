# src/retrain.py
"""Retraining pipeline that stays consistent with inference.

    python -m src.retrain prepare manifest.csv --cache cache_faces
    python -m src.retrain train   --cache cache_faces --out output_v2/model_v2.pth --manifest manifest.csv

Guarantees
- Face crops and tensors come from the SAME functions inference uses (src/video.py), so a model
  trained here sees exactly what the app feeds it.
- Only the `train` and `val` splits are ever read. The `test` split is never touched.
- The manifest must pass the leakage checks of `src.evaluate` (groups never span splits; identical
  video content never appears in two splits).
- An existing checkpoint is never overwritten.
- The checkpoint is written in the format `src.checkpoint.load_trained_model` loads strictly.

This produces a NEW model. It is not the original checkpoint and must be evaluated from scratch
with `src.evaluate` on a held-out test split before any claim is made about it.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.checkpoint import CheckpointError, load_trained_model, resolve_device
from src.config import InferenceConfig
from src.evaluate import EvalError, Row, file_id, load_manifest, make_condition, validate_manifest
from src.metrics import classification_metrics
from src.model import FAKE_INDEX, DeepfakeDetector
from src.pipeline import build_face_detector
from src.video import (VideoError, crop_face, fill_missing_faces, plan_clips, probe_video,
                       read_clip, to_tensor)

TRAIN_SPLITS = ("train", "val")
INDEX_COLUMNS = ["path", "file_id", "label", "group", "dataset", "split", "n_clips", "status"]


def _cache_meta(cfg: InferenceConfig, clips_per_video: int) -> dict:
    return {"image_size": cfg.image_size, "frames_per_clip": cfg.frames_per_clip, "stride": cfg.stride,
            "face_selection": cfg.face_selection, "clips_per_video": clips_per_video,
            "min_clip_face_fraction": cfg.min_clip_face_fraction}


# --------------------------------------------------------------------------------- prepare
def prepare_cache(rows: list[Row], cache_dir: str | Path, cfg: InferenceConfig, clips_per_video: int = 3,
                  detector=None, progress=None) -> dict:
    """Extract face clips once and store them as uint8 arrays (n_clips, T, H, W, 3) per video."""
    if clips_per_video < 1:
        raise EvalError("clips_per_video must be >= 1")
    todo = [r for r in rows if r.split in TRAIN_SPLITS]
    if not todo:
        raise EvalError("Manifest has no rows in the 'train' or 'val' split.")
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    meta = _cache_meta(cfg, clips_per_video)
    mp = cache / "meta.json"
    if mp.exists() and json.loads(mp.read_text()) != meta:
        raise EvalError(f"{cache} was built with different settings; use a new cache directory.")
    mp.write_text(json.dumps(meta, indent=2))

    detector = detector or build_face_detector(resolve_device(cfg.device))
    loose = dataclasses.replace(cfg, num_clips=clips_per_video, max_file_mb=1e9, max_duration_s=1e9)
    seen: dict[str, Row] = {}
    index, usable = [], 0
    for i, r in enumerate(todo):
        fid = file_id(r.path)
        if fid in seen and seen[fid].split != r.split:
            raise EvalError(f"Identical video content in splits '{seen[fid].split}' and '{r.split}': "
                            f"{Path(r.path).name} == {Path(seen[fid].path).name}")
        seen.setdefault(fid, r)
        out = cache / f"{fid}.npy"
        status, n = "ok", 0
        if out.exists():
            n = int(np.load(out, mmap_mode="r").shape[0])
        else:
            try:
                info = probe_video(r.path, loose)
                starts, stride = plan_clips(info.frames, loose)
                clips = []
                cap = cv2.VideoCapture(r.path)
                try:
                    for s in starts:
                        frames = read_clip(cap, s, stride, cfg.frames_per_clip)
                        if len(frames) < cfg.frames_per_clip:
                            continue
                        faces = [crop_face(f, detector, cfg.image_size, cfg.face_selection) for f in frames]
                        if sum(f is not None for f in faces) / len(frames) < cfg.min_clip_face_fraction:
                            continue
                        clips.append(np.stack(fill_missing_faces(faces)))
                finally:
                    cap.release()
                if clips:
                    np.save(out, np.stack(clips))
                    n = len(clips)
            except VideoError as e:
                status = f"error:{e.code}"
        if n == 0 and status == "ok":
            status = "no_usable_clips"
        usable += n > 0
        index.append({"path": r.path, "file_id": fid, "label": r.label, "group": r.group,
                      "dataset": r.dataset, "split": r.split, "n_clips": n, "status": status})
        if progress:
            progress(i + 1, len(todo))
    with open(cache / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        w.writeheader()
        w.writerows(index)
    return {"videos": len(todo), "usable": usable, "unusable": len(todo) - usable, "cache": str(cache)}


# ---------------------------------------------------------------------------------- data
def read_index(cache_dir: str | Path) -> tuple[list[dict], dict]:
    c = Path(cache_dir)
    if not (c / "index.csv").is_file() or not (c / "meta.json").is_file():
        raise EvalError(f"{c} is not a prepared cache (run `prepare` first).")
    with open(c / "index.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["label"], r["n_clips"] = int(r["label"]), int(r["n_clips"])
    return rows, json.loads((c / "meta.json").read_text())


def random_degradation():
    """One random JPEG / downscale / blur setting, applied identically to every frame of a clip.

    Works on the 112 px face crops, so it only approximates degradation of the full video frame
    followed by cropping. It is a robustness aid, not a model of real platform re-encoding.
    """
    kind = random.choice(("jpeg", "scale", "blur"))
    spec = {"jpeg": lambda: f"jpeg:{random.randint(25, 85)}",
            "scale": lambda: f"scale:{random.uniform(0.4, 0.8):.2f}",
            "blur": lambda: f"blur:{random.uniform(0.5, 1.5):.2f}"}[kind]()
    return make_condition(spec)


class CachedClips(Dataset):
    """One random clip per video per epoch (train) or the first clip (deterministic).

    ``degrade_p`` is the chance that a training clip also gets a random compression/blur/downscale
    degradation (default 0 = off). Evaluation mode never augments.
    """

    def __init__(self, cache_dir, rows: list[dict], train: bool, degrade_p: float = 0.0):
        if not 0.0 <= degrade_p <= 1.0:
            raise EvalError("degrade_p must be between 0 and 1")
        self.dir, self.rows, self.train, self.degrade_p = Path(cache_dir), rows, train, degrade_p

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        arr = np.load(self.dir / f"{r['file_id']}.npy", mmap_mode="r")
        clip = np.array(arr[random.randrange(arr.shape[0]) if self.train else 0])   # (T, H, W, 3) uint8
        if self.train:
            if random.random() < 0.5:                                   # horizontal flip of the whole clip
                clip = clip[:, :, ::-1]
            gain = random.uniform(0.9, 1.1)                             # mild brightness jitter
            clip = np.clip(clip.astype(np.float32) * gain, 0, 255).astype(np.uint8)
            if self.degrade_p and random.random() < self.degrade_p:
                degrade = random_degradation()
                clip = np.stack([degrade(np.ascontiguousarray(f)) for f in clip])
        x = torch.stack([to_tensor(f) for f in clip])                   # same conversion as inference
        return x, torch.tensor(r["label"], dtype=torch.long)


def video_scores(model: nn.Module, cache_dir, rows: list[dict], device) -> tuple[list[int], list[float], float]:
    """Video-level P(fake) = mean over all cached clips, as inference aggregates. Also mean CE loss."""
    model.eval()
    ys, ss, losses = [], [], []
    ce = nn.CrossEntropyLoss(reduction="mean")
    with torch.inference_mode():
        for r in rows:
            arr = np.load(Path(cache_dir) / f"{r['file_id']}.npy")
            x = torch.stack([torch.stack([to_tensor(f) for f in clip]) for clip in arr]).to(device)
            logits = model(x)
            ss.append(float(torch.softmax(logits, 1)[:, FAKE_INDEX].mean()))
            ys.append(r["label"])
            losses.append(float(ce(logits, torch.full((len(arr),), r["label"], device=device))))
    return ys, ss, float(np.mean(losses))


# ---------------------------------------------------------------------------------- train
def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parents[1], timeout=5).stdout.strip() or None
    except Exception:
        return None


def train(cache_dir: str | Path, out_path: str | Path, cfg: InferenceConfig | None = None,
          epochs: int = 12, batch_size: int = 4, lr: float = 1e-4, seed: int = 42, device: str = "auto",
          pretrained: bool = True, init_from: str | None = None, patience: int | None = None,
          num_workers: int = 0, manifest_sha256: str | None = None, degrade_p: float = 0.0,
          log=print) -> dict:
    cfg = cfg or InferenceConfig(checkpoint_path="unused")
    out = Path(out_path)
    if out.exists():
        raise EvalError(f"{out} already exists. Training never overwrites a checkpoint; choose a new name.")
    index, meta = read_index(cache_dir)
    if meta != _cache_meta(cfg, meta["clips_per_video"]):
        raise EvalError("The cache was built with preprocessing settings that differ from the current "
                        f"configuration ({meta}); rebuild it with `prepare`.")
    usable = [r for r in index if r["n_clips"] > 0]
    tr = [r for r in usable if r["split"] == "train"]
    va = [r for r in usable if r["split"] == "val"]
    for name, part in (("train", tr), ("val", va)):
        if len({r["label"] for r in part}) < 2:
            raise EvalError(f"The '{name}' split needs usable videos of both classes "
                            f"(found {len(part)} videos, labels {sorted({r['label'] for r in part})}).")
    shared = {r["group"] for r in tr if r["group"]} & {r["group"] for r in va if r["group"]}
    if shared or {r["file_id"] for r in tr} & {r["file_id"] for r in va}:
        raise EvalError("Train and val share source groups or identical videos; fix the manifest.")
    if any(not r["group"] for r in usable):
        log("WARNING: some videos have no group; leakage between train and val was not checked for them.")

    dev = resolve_device(device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = DeepfakeDetector(pretrained=pretrained and not init_from).to(dev)
    if init_from:
        model.load_state_dict(load_trained_model(init_from, "cpu").model.state_dict(), strict=True)

    counts = np.bincount([r["label"] for r in tr], minlength=2).astype(float)
    weights = torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)), dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loader = DataLoader(CachedClips(cache_dir, tr, True, degrade_p), batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, drop_last=len(tr) > batch_size,
                        generator=torch.Generator().manual_seed(seed))

    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = out.with_suffix(".log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_video_auc", "seconds"])
    best, best_epoch, stale = None, 0, 0
    config = {"image_size": cfg.image_size, "frames_per_clip": cfg.frames_per_clip, "stride": cfg.stride,
              "face_selection": cfg.face_selection, "clips_per_video": meta["clips_per_video"]}
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        total, nb = 0.0, 0
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=amp):
                loss = crit(model(x), y)
            if not torch.isfinite(loss):
                raise EvalError(f"Loss became {loss.item()} in epoch {epoch}; stopping. Lower the learning rate.")
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += float(loss.detach())
            nb += 1
        ys, ss, vloss = video_scores(model, cache_dir, va, dev)
        auc = classification_metrics(ys, ss)["roc_auc"]
        sec = time.perf_counter() - t0
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, total / max(nb, 1), vloss, auc, round(sec, 1)])
        log(f"epoch {epoch}/{epochs} train_loss {total / max(nb, 1):.4f} val_loss {vloss:.4f} "
            f"val_video_auc {auc:.4f} ({sec:.0f}s)")
        better = best is None or auc > best[0] or (auc == best[0] and vloss < best[1])
        if better:
            best, best_epoch, stale = (auc, vloss), epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state": opt.state_dict(),
                        "config": config, "epoch": epoch, "best_val_auc": auc,
                        "training": {"lr": lr, "batch_size": batch_size, "seed": seed, "epochs": epochs,
                                     "pretrained_imagenet": bool(pretrained and not init_from),
                                     "init_from": init_from, "degrade_p": degrade_p, "n_train_videos": len(tr),
                                     "n_val_videos": len(va), "manifest_sha256": manifest_sha256,
                                     "git_commit": _git_commit(), "torch": torch.__version__,
                                     "created": dt.datetime.now().isoformat(timespec="seconds")}}, out)
        else:
            stale += 1
            if patience and stale >= patience:
                log(f"No improvement for {patience} epochs; stopping early.")
                break
    return {"best_epoch": best_epoch, "best_val_video_auc": best[0], "checkpoint": str(out),
            "log": str(log_path)}


# -------------------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Prepare data and train a new deepfake detector")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("manifest")
    p.add_argument("--cache", required=True)
    p.add_argument("--clips-per-video", type=int, default=3)
    p.add_argument("--device", default=None)
    t = sub.add_parser("train")
    t.add_argument("--cache", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--manifest", default=None, help="only used to record its hash in the checkpoint")
    t.add_argument("--epochs", type=int, default=12)
    t.add_argument("--batch-size", type=int, default=4)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--device", default="auto")
    t.add_argument("--num-workers", type=int, default=0)
    t.add_argument("--patience", type=int, default=None)
    t.add_argument("--init-from", default=None, help="fine-tune from an existing checkpoint")
    t.add_argument("--degrade-p", type=float, default=0.0,
                   help="chance (0-1) that a training clip gets random JPEG/blur/downscale; off by default")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "prepare":
            rows = load_manifest(a.manifest)
            for w in validate_manifest(rows):
                print("WARNING:", w)
            cfg = InferenceConfig(**({"device": a.device} if a.device else {}))
            res = prepare_cache(rows, a.cache, cfg, a.clips_per_video,
                                progress=lambda d, n: print(f"\r{d}/{n}", end="", flush=True))
            print(f"\nCached {res['usable']} usable videos; {res['unusable']} unusable (no face/unreadable).")
        else:
            mh = hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest() if a.manifest else None
            res = train(a.cache, a.out, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, seed=a.seed,
                        device=a.device, num_workers=a.num_workers, patience=a.patience,
                        init_from=a.init_from, manifest_sha256=mh, degrade_p=a.degrade_p)
            print(f"Best epoch {res['best_epoch']} (val video AUC {res['best_val_video_auc']:.4f}) "
                  f"saved to {res['checkpoint']}.\nNext: evaluate on the held-out test split with "
                  "`python -m src.evaluate` before making any claim about this model.")
    except (EvalError, CheckpointError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

# src/inference.py
"""Command line inference.

    python -m src.inference path/to/video.mp4 [--checkpoint output/deepfake_cnn_lstm.pth]

Prints the RAW model score. It is not a calibrated probability and no REAL/FAKE
label is produced until thresholds are validated.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys

from src.checkpoint import CheckpointError, load_trained_model
from src.config import InferenceConfig
from src.pipeline import analyze_video
from src.video import VideoError


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Analyse a video with the trained CNN+LSTM model")
    ap.add_argument("video")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default=None, help="auto | cpu | cuda")
    ap.add_argument("--num-clips", type=int, default=None)
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = ap.parse_args(argv)

    overrides = {k: v for k, v in {"checkpoint_path": args.checkpoint, "device": args.device,
                                   "num_clips": args.num_clips}.items() if v is not None}
    try:
        cfg = InferenceConfig(**overrides)
        loaded = load_trained_model(cfg.checkpoint_path, cfg.device)
        res = analyze_video(args.video, loaded, cfg)
    except (CheckpointError, VideoError, ValueError) as e:
        print(f"ERROR: {getattr(e, 'message', e)}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(dataclasses.asdict(res), indent=2, default=str))
        return 0 if res.status == "ok" else 1

    print(f"Model: {res.model_path} (sha256 {res.model_sha256[:12]}...)")
    print(f"Clips used: {res.clips_used}/{len(res.clips)} | frames read: {res.frames_analyzed} "
          f"| face coverage: {res.face_coverage:.0%} | time: {res.elapsed_s:.1f}s")
    if res.status != "ok":
        print("RESULT: insufficient evidence (no score produced)")
    else:
        print(f"Raw model score P(fake): mean {res.score_mean:.3f}, median {res.score_median:.3f} "
              f"(range {res.score_min:.3f}-{res.score_max:.3f})")
        print("Uncalibrated score, not a probability of authenticity and not proof either way.")
    for w in res.warnings:
        print(f"WARNING: {w}")
    return 0 if res.status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())

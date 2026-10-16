"""Streamlit app for the CNN + LSTM deepfake video detector.

Uses the same strict loader and pipeline as the CLI. If the trained checkpoint is
missing or incompatible the app stops with a diagnostic; it never substitutes
another model. This interim version reports the RAW model score only; the
REAL / FAKE / INCONCLUSIVE decision is added once thresholds are validated.
"""
import logging
import os
import tempfile

import pandas as pd
import streamlit as st

from src.checkpoint import CheckpointError, load_trained_model
from src.config import ALLOWED_EXTENSIONS, InferenceConfig
from src.pipeline import analyze_video, build_face_detector
from src.video import VideoError

log = logging.getLogger("deepfake.app")
st.set_page_config(page_title="Deepfake Video Analysis", page_icon="🎭", layout="wide")
cfg = InferenceConfig()


@st.cache_resource(show_spinner="Loading trained model...")
def get_model(path: str, device: str):
    return load_trained_model(path, device)


@st.cache_resource(show_spinner="Loading face detector...")
def get_detector(device: str):
    return build_face_detector(next(get_model(cfg.checkpoint_path, cfg.device).model.parameters()).device)


def show_result(res):
    if res.status != "ok":
        st.error("Insufficient evidence: the model could not be applied reliably to this video.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Model score, mean (raw P(fake))", f"{res.score_mean:.2f}")
        c2.metric("Model score, median", f"{res.score_median:.2f}")
        c3.metric("Clip score range", f"{res.score_min:.2f} - {res.score_max:.2f}")
        st.info("This is a raw, uncalibrated model score. It is not the probability that the video is "
                "fake or authentic, and no REAL/FAKE decision is made until thresholds are validated.")

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Clips used", f"{res.clips_used} / {len(res.clips)}")
    d2.metric("Frames analysed", res.frames_analyzed)
    d3.metric("Faces detected", f"{res.faces_detected} ({res.face_coverage:.0%})")
    d4.metric("Processing time", f"{res.elapsed_s:.1f} s")
    for w in res.warnings:
        st.warning(w)

    st.subheader("Per-clip scores")
    st.dataframe(pd.DataFrame([{
        "clip start frame": c.start_frame,
        "frames with face": f"{c.frames_with_face}/{c.frames_read}",
        "used": c.used,
        "raw P(fake)": None if c.p_fake is None else round(c.p_fake, 3),
        "note": c.reason or "",
    } for c in res.clips]), hide_index=True, use_container_width=True)

    v = res.video
    st.caption(f"Video: {v.width}x{v.height}, {v.frames} frames"
               + (f", {v.fps:.1f} fps, {v.duration_s:.1f} s" if v.fps else "")
               + f", {v.size_mb:.1f} MB | Model file: {res.model_path} "
               f"(sha256 {res.model_sha256[:12]}...)")


def main():
    st.title("🎭 Deepfake Video Analysis")
    st.warning("AI assessment only. It is not definitive proof that a video is authentic or manipulated. "
               "The model was trained on face-manipulation videos and may not detect fully AI-generated video "
               "or other manipulation types.")

    try:
        loaded = get_model(cfg.checkpoint_path, cfg.device)
    except CheckpointError as e:
        st.error(str(e))
        st.stop()

    with st.sidebar:
        st.subheader("Model")
        st.write(f"File: `{os.path.basename(loaded.path)}`")
        st.write(f"sha256: `{loaded.sha256[:12]}...`")
        if loaded.epoch is not None:
            st.write(f"Checkpoint epoch: {loaded.epoch}")
        st.write(f"Device: {next(loaded.model.parameters()).device}")
        st.write(f"Clips per video: {cfg.num_clips} x {cfg.frames_per_clip} frames")
        st.write(f"Limits: {cfg.max_file_mb:.0f} MB, {cfg.max_duration_s:.0f} s")

    up = st.file_uploader("Upload a video", type=[e.lstrip(".") for e in ALLOWED_EXTENSIONS])
    if up is None:
        return
    if up.size > cfg.max_file_mb * 1024 * 1024:
        st.error(f"File is {up.size / 1048576:.0f} MB; the limit is {cfg.max_file_mb:.0f} MB.")
        return

    st.video(up)
    if not st.button("Analyze video", type="primary"):
        return

    suffix = os.path.splitext(up.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
        tf.write(up.getbuffer())
        tmp = tf.name
    bar = st.progress(0.0, text="Analysing clips...")
    try:
        res = analyze_video(tmp, loaded, cfg, detector=get_detector(cfg.device),
                            progress=lambda d, t: bar.progress(d / t, text=f"Analysing clips {d}/{t}"))
        bar.empty()
        show_result(res)
    except VideoError as e:
        bar.empty()
        st.error(e.message)
        st.caption("Check the file and try again with another video.")
    except Exception:
        bar.empty()
        log.exception("analysis failed")  # message only; uploaded content is never logged
        st.error("Unexpected error while analysing the video. Please try again or use a different file.")
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


main()

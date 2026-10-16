"""Streamlit app for the CNN + LSTM deepfake video detector.

Uses the same strict loader and pipeline as the CLI. If the trained checkpoint is
missing or incompatible the app stops with a diagnostic; it never substitutes
another model. The REAL / FAKE / INCONCLUSIVE decision is shown only when a
validated decision policy exists for the loaded checkpoint (see docs/DECISION_POLICY.md);
otherwise only the raw model score is shown.
"""
import logging
import os
import tempfile

import streamlit as st

from src.checkpoint import CheckpointError, load_trained_model
from src.config import ALLOWED_EXTENSIONS, InferenceConfig
from src.decision import PolicyError, decide, load_policy
from src.pipeline import analyze_video, build_face_detector
from src.ui import show_result
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

    try:
        policy = load_policy(cfg.policy_path, loaded.sha256)
    except PolicyError as e:
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
        st.write("Decision thresholds: " + ("validated policy loaded" if policy else "not validated"))

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
        show_result(res, decide(res, policy), policy)
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

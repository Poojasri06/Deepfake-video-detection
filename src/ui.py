# src/ui.py
"""Streamlit rendering of an analysis result (kept out of app.py so it can be tested)."""
import pandas as pd
import streamlit as st

from src.decision import INCONCLUSIVE, LIKELY_FAKE, LIKELY_REAL


def show_decision(res, dec, policy):
    box = {LIKELY_FAKE: st.error, LIKELY_REAL: st.success, INCONCLUSIVE: st.warning}.get(dec.label, st.info)
    box(f"**{dec.summary}**")
    for r in dec.reasons:
        st.write(f"- {r}")

    if res.status == "ok":
        c1, c2, c3 = st.columns(3)
        c1.metric("Model score, mean (raw P(fake))", f"{res.score_mean:.2f}")
        c2.metric("Model score, median", f"{res.score_median:.2f}")
        c3.metric("Clip score range", f"{res.score_min:.2f} - {res.score_max:.2f}")
        if dec.calibrated_p_fake is not None:
            v = policy.validation
            st.metric("Calibrated P(fake)", f"{dec.calibrated_p_fake:.0%}")
            st.caption(f"Calibrated on {', '.join(v['datasets'])} ({v['n_real']} real / {v['n_fake']} fake "
                       "videos). It describes that data, not a verified probability for this video, and "
                       "is not the probability that the video is authentic.")
        else:
            st.caption("Raw, uncalibrated model score: not a probability of fake or authentic.")
    st.caption("AI assessment only, not definitive forensic proof. A 'likely real' result does not "
               "establish authenticity.")


def show_result(res, dec, policy):
    show_decision(res, dec, policy)

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

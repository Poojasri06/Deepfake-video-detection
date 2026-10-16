import math

import pytest

from src.config import InferenceConfig
from src.pipeline import aggregate, analyze_video
from src.video import VideoError
from tests.conftest import EveryNthMissingDetector, FixedBoxDetector, NoFaceDetector

CFG = InferenceConfig(checkpoint_path="unused", device="cpu")


def test_aggregate_values():
    a = aggregate([0.1, 0.2, 0.9])
    assert a["mean"] == pytest.approx(0.4) and a["median"] == 0.2
    assert a["min"] == 0.1 and a["max"] == 0.9


def test_aggregate_empty():
    assert aggregate([]) == {"mean": None, "median": None, "min": None, "max": None}


def test_end_to_end_with_detected_faces(loaded_model, video_path):
    seen = []
    res = analyze_video(video_path, loaded_model, CFG, detector=FixedBoxDetector(),
                        progress=lambda d, t: seen.append((d, t)))
    assert res.status == "ok"
    assert res.clips_used == CFG.num_clips == len(res.clips)
    assert res.frames_analyzed == CFG.num_clips * CFG.frames_per_clip
    assert res.faces_detected == res.frames_analyzed and res.face_coverage == 1.0
    for c in res.clips:
        assert 0.0 <= c.p_fake <= 1.0
    assert res.score_min <= res.score_median <= res.score_max
    assert 0.0 <= res.score_mean <= 1.0
    assert res.model_sha256 == loaded_model.sha256 and res.model_path == "synthetic.pth"
    assert seen[-1] == (CFG.num_clips, CFG.num_clips)
    assert res.elapsed_s > 0


def test_is_deterministic(loaded_model, video_path):
    a = analyze_video(video_path, loaded_model, CFG, detector=FixedBoxDetector())
    b = analyze_video(video_path, loaded_model, CFG, detector=FixedBoxDetector())
    assert [c.p_fake for c in a.clips] == [c.p_fake for c in b.clips]


def test_no_faces_gives_insufficient_evidence_and_no_score(loaded_model, video_path):
    res = analyze_video(video_path, loaded_model, CFG, detector=NoFaceDetector())
    assert res.status == "insufficient_evidence"
    assert res.clips_used == 0 and res.score_mean is None and res.face_coverage == 0.0
    assert all(c.reason and "face" in c.reason for c in res.clips)
    assert any("usable clip" in w for w in res.warnings)


def test_partial_faces_still_used_with_warning(loaded_model, video_path):
    # every 10th frame misses a face -> 90% coverage per clip, above the 75% threshold
    res = analyze_video(video_path, loaded_model, CFG, detector=EveryNthMissingDetector(10))
    assert res.status == "ok"
    assert 0.85 < res.face_coverage < 1.0


def test_low_face_fraction_drops_clips(loaded_model, video_path):
    # every 2nd frame misses -> 50% < 75% -> all clips dropped
    res = analyze_video(video_path, loaded_model, CFG, detector=EveryNthMissingDetector(2))
    assert res.status == "insufficient_evidence" and res.clips_used == 0


def test_time_budget_stops_early(loaded_model, video_path):
    cfg = InferenceConfig(checkpoint_path="unused", device="cpu", max_processing_s=0)
    res = analyze_video(video_path, loaded_model, cfg, detector=FixedBoxDetector())
    assert any("Time budget" in w for w in res.warnings)
    assert len(res.clips) < cfg.num_clips and res.status == "insufficient_evidence"


def test_invalid_video_raises_video_error(loaded_model, tmp_path):
    p = tmp_path / "bad.mp4"
    p.write_bytes(b"garbage" * 50)
    with pytest.raises(VideoError):
        analyze_video(p, loaded_model, CFG, detector=FixedBoxDetector())


def test_scores_are_finite(loaded_model, video_path):
    res = analyze_video(video_path, loaded_model, CFG, detector=FixedBoxDetector())
    assert all(math.isfinite(c.p_fake) for c in res.clips)

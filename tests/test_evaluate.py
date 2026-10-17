import csv
import json

import numpy as np
import pytest

from src.config import InferenceConfig
from src.evaluate import (EvalError, Row, build_report, fit_from_scores, load_manifest, main,
                          make_condition, read_scores, render_markdown, score_manifest,
                          validate_manifest, write_scores)
from src.metrics import label_decision_metrics
from tests.conftest import FixedBoxDetector, NoFaceDetector, write_video

CFG = InferenceConfig(checkpoint_path="unused", device="cpu")
SHA = "d" * 64


def manifest(tmp_path, rows, header="path,label,group,dataset,split"):
    p = tmp_path / "m.csv"
    p.write_text(header + "\n" + "\n".join(rows) + "\n")
    return p


def touch(tmp_path, *names):
    for n in names:
        (tmp_path / n).write_bytes(b"x" + n.encode())


# ============================================================== manifest validation
def test_manifest_loads_and_resolves_relative_paths(tmp_path):
    touch(tmp_path, "a.mp4", "b.mp4")
    rows = load_manifest(manifest(tmp_path, ["a.mp4,real,g1,ds,val", "b.mp4,FAKE,g1,ds,val"]))
    assert [r.label for r in rows] == [0, 1] and rows[0].path == str(tmp_path / "a.mp4")


@pytest.mark.parametrize("bad", ["a.mp4,maybe,g,ds,val"])
def test_bad_label_rejected(tmp_path, bad):
    with pytest.raises(EvalError, match="not real/fake"):
        load_manifest(manifest(tmp_path, [bad]))


def test_missing_columns_rejected(tmp_path):
    with pytest.raises(EvalError, match="path,label"):
        load_manifest(manifest(tmp_path, ["a.mp4,real"], header="file,kind"))


def test_group_leaking_across_splits_is_an_error(tmp_path):
    touch(tmp_path, "a.mp4", "b.mp4")
    rows = load_manifest(manifest(tmp_path, ["a.mp4,real,id7,ds,val", "b.mp4,fake,id7,ds,test"]))
    with pytest.raises(EvalError, match="more than one split"):
        validate_manifest(rows)


def test_same_group_within_one_split_is_fine(tmp_path):
    touch(tmp_path, "a.mp4", "b.mp4", "c.mp4")
    rows = load_manifest(manifest(tmp_path, ["a.mp4,real,id7,ds,val", "b.mp4,fake,id7,ds,val",
                                             "c.mp4,fake,id8,ds,test"]))
    assert validate_manifest(rows)  # only warnings (single-class splits)


def test_missing_group_blocks_unless_allowed(tmp_path):
    touch(tmp_path, "a.mp4", "b.mp4")
    rows = load_manifest(manifest(tmp_path, ["a.mp4,real,,ds,val", "b.mp4,fake,,ds,test"]))
    with pytest.raises(EvalError, match="leakage between splits cannot be checked"):
        validate_manifest(rows)
    assert any("NOT checked" in w for w in validate_manifest(rows, allow_ungrouped=True))


def test_missing_file_and_unknown_split(tmp_path):
    rows = load_manifest(manifest(tmp_path, ["ghost.mp4,real,g,ds,valid"]))
    with pytest.raises(EvalError) as e:
        validate_manifest(rows)
    assert "file not found" in str(e.value) and "unknown split" in str(e.value)
    with pytest.raises(EvalError) as e2:                    # without file checks only the split error remains
        validate_manifest(rows, check_files=False)
    assert "unknown split" in str(e2.value) and "file not found" not in str(e2.value)


def test_duplicate_path_with_conflicting_label(tmp_path):
    touch(tmp_path, "a.mp4")
    rows = load_manifest(manifest(tmp_path, ["a.mp4,real,g,ds,val", "a.mp4,fake,g,ds,val"]))
    with pytest.raises(EvalError, match="twice"):
        validate_manifest(rows)


# ============================================================== degradation conditions
def _frame():
    return np.random.default_rng(0).integers(0, 255, (120, 160, 3), dtype=np.uint8)


def test_none_condition_is_identity():
    assert make_condition("none") is None


@pytest.mark.parametrize("spec", ["jpeg:20", "scale:0.4", "blur:2", "brightness:0.5"])
def test_conditions_keep_shape_dtype_and_change_pixels(spec):
    f = make_condition(spec)
    out = f(_frame())
    assert out.shape == (120, 160, 3) and out.dtype == np.uint8
    assert not np.array_equal(out, _frame())


def test_lower_jpeg_quality_destroys_more_detail():
    a, b = make_condition("jpeg:90")(_frame()), make_condition("jpeg:10")(_frame())
    err = lambda x: np.abs(x.astype(int) - _frame().astype(int)).mean()
    assert err(b) > err(a)


@pytest.mark.parametrize("spec", ["jpeg", "jpeg:abc", "jpeg:0", "scale:1.5", "blur:-1", "sepia:1"])
def test_bad_conditions_rejected(spec):
    with pytest.raises(EvalError):
        make_condition(spec)


# ============================================================== scoring real videos
@pytest.fixture()
def vids(tmp_path):
    for i, n in enumerate(["v1", "v2", "v3"]):
        write_video(tmp_path / f"{n}.avi", n_frames=200 + 20 * i)
    return tmp_path


def rows_for(d):
    return [Row(str(d / "v1.avi"), 0, "g1", "dsA", "val"), Row(str(d / "v2.avi"), 1, "g2", "dsA", "val"),
            Row(str(d / "v3.avi"), 1, "g3", "dsB", "test")]


def test_score_manifest_end_to_end(loaded_model, vids, tmp_path):
    out, meta = score_manifest(rows_for(vids), loaded_model, CFG, detector=FixedBoxDetector())
    assert [r["status"] for r in out] == ["ok"] * 3
    assert all(0 <= r["score_mean"] <= 1 and r["file_id"] for r in out)
    assert meta["checkpoint_sha256"] == loaded_model.sha256 and meta["condition"] == "none"
    write_scores(out, meta, tmp_path / "s.csv")
    back, meta2 = read_scores(tmp_path / "s.csv")
    assert meta2 == meta and back[0]["label"] == 0 and back[0]["score_mean"] == pytest.approx(out[0]["score_mean"])


def test_score_manifest_records_failures_without_aborting(loaded_model, vids):
    bad = vids / "bad.mp4"
    bad.write_bytes(b"not a video" * 30)
    rows = rows_for(vids) + [Row(str(bad), 0, "g9", "dsA", "val")]
    out, _ = score_manifest(rows, loaded_model, CFG, detector=FixedBoxDetector())
    assert out[-1]["status"] == "error" and out[-1]["error"]
    assert [r["status"] for r in out[:3]] == ["ok"] * 3


def test_no_faces_recorded_as_insufficient_evidence(loaded_model, vids):
    out, _ = score_manifest(rows_for(vids), loaded_model, CFG, detector=NoFaceDetector())
    assert {r["status"] for r in out} == {"insufficient_evidence"}
    assert all(r["score_mean"] is None and r["score_median"] is None for r in out)


def test_condition_is_applied_to_frames_before_detection_and_recorded(loaded_model, vids):
    """Checked at the detector input: the synthetic random-weight model ignores its input,
    so its scores cannot reveal whether a degradation was applied."""
    class Rec(FixedBoxDetector):
        def __init__(self):
            self.seen = []

        def detect(self, rgb):
            self.seen.append(rgb.copy())
            return super().detect(rgb)

    one = rows_for(vids)[:1]
    clean_d, bad_d, none_d = Rec(), Rec(), Rec()
    score_manifest(one, loaded_model, CFG, detector=clean_d)
    score_manifest(one, loaded_model, CFG, detector=none_d, condition="none")
    _, meta = score_manifest(one, loaded_model, CFG, detector=bad_d, condition="jpeg:5")
    assert meta["condition"] == "jpeg:5"
    assert all(np.array_equal(a, b) for a, b in zip(clean_d.seen, none_d.seen))
    assert len(bad_d.seen) == len(clean_d.seen) > 0
    assert all(not np.array_equal(a, b) for a, b in zip(clean_d.seen, bad_d.seen))


def test_identical_content_in_two_splits_is_refused(loaded_model, vids):
    (vids / "copy.avi").write_bytes((vids / "v1.avi").read_bytes())
    rows = [Row(str(vids / "v1.avi"), 0, "g1", "d", "val"), Row(str(vids / "copy.avi"), 0, "g2", "d", "test")]
    with pytest.raises(EvalError, match="Identical video content"):
        score_manifest(rows, loaded_model, CFG, detector=FixedBoxDetector())


# ============================================================== fit -> report flow
def synth_rows(n=60, sep=3.0, seed=0, split="val", dataset="dsA", start=0, bad_fraction=0.0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(2 * n):
        label = int(i >= n)
        z = (sep if label else -sep) + rng.normal(0, 1)
        s = float(1 / (1 + np.exp(-z)))
        no_score = rng.uniform() < bad_fraction
        out.append({"path": f"/x/{split}{start + i}.mp4", "file_id": f"{split}{start + i:08d}", "label": label,
                    "group": f"g{start + i}", "dataset": dataset, "split": split,
                    "status": "insufficient_evidence" if no_score else "ok", "clips_used": 5,
                    "clips_total": 5, "face_coverage": 0.0 if no_score else 1.0,
                    "score_mean": None if no_score else s, "score_median": None if no_score else s,
                    "score_min": None if no_score else max(0.0, s - 0.02),
                    "score_max": None if no_score else min(1.0, s + 0.02), "elapsed_s": 1.0, "error": ""})
    return out


META = {"checkpoint_sha256": SHA, "checkpoint_name": "m.pth", "condition": "none", "n_videos": 0, "config": {}}


def save(tmp_path, name, rows, meta=META):
    write_scores(rows, meta, tmp_path / name)
    return tmp_path / name


def test_fit_uses_only_val_split_and_records_provenance():
    rows = synth_rows(split="val") + synth_rows(split="test", seed=9, dataset="dsB")
    pol = fit_from_scores(rows, META)
    v = pol["validation"]
    assert v["datasets"] == ["dsA"] and v["n_real"] == 60 and v["n_fake"] == 60
    assert all(i.startswith("val") for i in v["fit_file_ids"]) and v["condition"] == "none"


def test_fit_counts_videos_without_score(tmp_path):
    rows = synth_rows(n=80, bad_fraction=0.2, seed=3)
    pol = fit_from_scores(rows, META, min_coverage=0.0, max_error_among_decided=1.0)
    ex = pol["validation"]["excluded_no_score"]
    assert ex > 0 and f"{ex} of 160" in pol["validation"]["note"]


def test_fit_refuses_degraded_scores_by_default():
    with pytest.raises(EvalError, match="clean validation data"):
        fit_from_scores(synth_rows(), {**META, "condition": "jpeg:30"})
    assert fit_from_scores(synth_rows(), {**META, "condition": "jpeg:30"}, allow_degraded=True)


def test_fit_failure_surfaces_as_eval_error():
    with pytest.raises(EvalError, match="too small"):
        fit_from_scores(synth_rows(n=5), META)


def test_report_end_to_end_with_per_dataset_and_conditions(tmp_path):
    pol = fit_from_scores(synth_rows(split="val"), META)
    clean = synth_rows(n=100, split="test", seed=11, dataset="dsA") + synth_rows(
        n=100, split="test", seed=12, dataset="dsB", start=1000)
    hard = synth_rows(n=100, sep=1.0, split="test", seed=11, dataset="dsA") + synth_rows(
        n=100, sep=1.0, split="test", seed=12, dataset="dsB", start=1000)
    f1 = save(tmp_path, "clean.csv", clean)
    f2 = save(tmp_path, "hard.csv", hard, {**META, "condition": "jpeg:20"})
    rep = build_report([f1, f2], pol)
    c, h = rep["results"]
    assert c["policy_applied"] and set(c["per_dataset"]) == {"dsA", "dsB"}
    assert c["overall"]["n_videos"] == 400 and c["overall"]["decision"]["n"] == 400
    assert c["overall"]["scored_videos_at_0.5"]["roc_auc"] > h["overall"]["scored_videos_at_0.5"]["roc_auc"]
    assert h["overall"]["decision"]["coverage"] < c["overall"]["decision"]["coverage"]
    md = render_markdown(rep)
    assert "dsA" in md and "jpeg:20" in md and "do not show how the model behaves" in md


def test_report_counts_no_score_videos_as_inconclusive_not_dropped(tmp_path):
    pol = fit_from_scores(synth_rows(split="val"), META)
    rows = synth_rows(n=100, split="test", seed=5, bad_fraction=0.3)
    rep = build_report([save(tmp_path, "t.csv", rows)], pol)
    o = rep["results"][0]["overall"]
    n_bad = sum(r["status"] != "ok" for r in rows)
    assert o["n_videos"] == 200 and o["n_no_score"] == n_bad > 0
    assert o["decision"]["n"] == 200 and o["decision"]["inconclusive"] >= n_bad


def test_report_refuses_videos_used_for_fitting(tmp_path):
    val = synth_rows(split="val")
    pol = fit_from_scores(val, META)
    reused = [{**r, "split": "test"} for r in val]            # same file_ids, relabelled as test
    with pytest.raises(EvalError, match="used to fit the policy"):
        build_report([save(tmp_path, "t.csv", reused)], pol)


def test_policy_for_other_checkpoint_not_applied(tmp_path):
    pol = fit_from_scores(synth_rows(split="val"), META)
    other = save(tmp_path, "o.csv", synth_rows(n=80, split="test", seed=4),
                 {**META, "checkpoint_sha256": "e" * 64, "checkpoint_name": "baseline.pth"})
    rep = build_report([other], pol)
    e = rep["results"][0]
    assert not e["policy_applied"] and "different checkpoint" in e["policy_note"]
    assert "decision" not in e["overall"] and "scored_videos_at_0.5" in e["overall"]


def test_report_without_policy_gives_threshold_free_metrics(tmp_path):
    rep = build_report([save(tmp_path, "t.csv", synth_rows(n=50, split="test"))], None)
    assert rep["policy"] is None and "decision" not in rep["results"][0]["overall"]


def test_report_requires_test_rows(tmp_path):
    with pytest.raises(EvalError, match="no rows with split 'test'"):
        build_report([save(tmp_path, "v.csv", synth_rows(split="val"))], None)


def test_label_decision_metrics_counts_unlabelled_as_inconclusive():
    m = label_decision_metrics([0, 0, 1, 1, 1], ["LIKELY_REAL", "LIKELY_FAKE", "LIKELY_FAKE",
                                                  "INCONCLUSIVE", "LIKELY_REAL"])
    assert m["coverage"] == pytest.approx(4 / 5) and m["inconclusive"] == 1
    assert m["false_positive_rate"]["count"] == 1 and m["false_negative_rate"]["count"] == 1
    assert m["error_rate_among_decided"] == pytest.approx(2 / 4)


# ============================================================== CLI
def test_cli_validate_ok_and_leakage_exit_codes(tmp_path, capsys):
    touch(tmp_path, "a.mp4", "b.mp4")
    ok = manifest(tmp_path, ["a.mp4,real,g1,ds,val", "b.mp4,fake,g2,ds,test"])
    assert main(["validate", str(ok)]) == 0 and "Manifest OK" in capsys.readouterr().out
    leak = manifest(tmp_path, ["a.mp4,real,g1,ds,val", "b.mp4,fake,g1,ds,test"])
    assert main(["validate", str(leak)]) == 2 and "more than one split" in capsys.readouterr().err


def test_cli_fit_and_report_files(tmp_path, capsys):
    val = save(tmp_path, "val.csv", synth_rows(split="val"))
    pol = tmp_path / "pol.json"
    assert main(["fit", str(val), "--out", str(pol)]) == 0
    assert json.loads(pol.read_text())["checkpoint_sha256"] == SHA
    test = save(tmp_path, "test.csv", synth_rows(n=60, split="test", seed=8))
    out = tmp_path / "rep"
    assert main(["report", str(test), "--policy", str(pol), "--out", str(out)]) == 0
    assert (tmp_path / "rep.json").is_file() and "Evaluation report" in (tmp_path / "rep.md").read_text()


def test_cli_fit_refusal_exit_code(tmp_path, capsys):
    val = save(tmp_path, "val.csv", synth_rows(n=5, split="val"))
    assert main(["fit", str(val), "--out", str(tmp_path / "p.json")]) == 2
    assert not (tmp_path / "p.json").exists() and "too small" in capsys.readouterr().err


def test_cli_score_missing_checkpoint(tmp_path, capsys):
    v = write_video(tmp_path / "v.avi", n_frames=60)
    m = manifest(tmp_path, ["v.avi,real,g1,ds,val"])
    assert main(["score", str(m), "--out", str(tmp_path / "s.csv"), "--checkpoint",
                 str(tmp_path / "none.pth")]) == 2
    assert "Checkpoint not found" in capsys.readouterr().err

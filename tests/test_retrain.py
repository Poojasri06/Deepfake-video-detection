import csv
import dataclasses
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.checkpoint import load_trained_model
from src.config import InferenceConfig
from src.evaluate import EvalError, Row
from src.pipeline import analyze_video
from src.retrain import CachedClips, main, prepare_cache, read_index, train, video_scores
from src.video import to_tensor
from tests.conftest import FixedBoxDetector, NoFaceDetector, write_video

# Small settings so training tests run in seconds on CPU.
SMALL = InferenceConfig(checkpoint_path="unused", device="cpu", image_size=64, frames_per_clip=4,
                        stride=2, num_clips=3)


def make_video(path, label, seed, n_frames=60, size=(160, 120)):
    """Noise video; fake ones carry a bright block inside the face box so the task is learnable."""
    rng = np.random.default_rng(seed)
    w, h = size
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (w, h))
    for _ in range(n_frames):
        f = rng.integers(0, 120, (h, w, 3), dtype=np.uint8)
        if label:
            f[30:80, 35:90] = 255
        vw.write(f)
    vw.release()
    return path


@pytest.fixture()
def rows(tmp_path):
    out = []
    for i in range(8):
        label, split = i % 2, "train" if i < 4 else "val"
        p = make_video(tmp_path / f"v{i}.avi", label, seed=i, n_frames=50 + 5 * i)
        out.append(Row(str(p), label, f"g{i}", "synth", split))
    return out


# ============================================================ parity with inference
def test_cached_training_tensors_equal_what_inference_feeds_the_model(loaded_model, video_path, tmp_path):
    cfg = InferenceConfig(checkpoint_path="unused", device="cpu")        # default settings, 5 clips
    captured = []
    h = loaded_model.model.register_forward_hook(lambda m, i, o: captured.append(i[0].detach().clone()))
    try:
        analyze_video(video_path, loaded_model, cfg, detector=FixedBoxDetector())
    finally:
        h.remove()
    (batch,) = captured

    prepare_cache([Row(str(video_path), 0, "g", "d", "train")], tmp_path / "c", cfg, clips_per_video=5,
                  detector=FixedBoxDetector())
    index, _ = read_index(tmp_path / "c")
    arr = np.load(tmp_path / "c" / f"{index[0]['file_id']}.npy")
    cached = torch.stack([torch.stack([to_tensor(f) for f in clip]) for clip in arr])
    assert cached.shape == batch.shape and torch.equal(cached, batch)


# ============================================================ prepare
def test_prepare_writes_cache_and_index(rows, tmp_path):
    res = prepare_cache(rows, tmp_path / "c", SMALL, 2, detector=FixedBoxDetector())
    assert res["usable"] == 8 and res["unusable"] == 0
    index, meta = read_index(tmp_path / "c")
    assert meta["image_size"] == 64 and meta["frames_per_clip"] == 4
    assert all(r["n_clips"] == 2 and r["status"] == "ok" for r in index)
    arr = np.load(tmp_path / "c" / f"{index[0]['file_id']}.npy")
    assert arr.shape == (2, 4, 64, 64, 3) and arr.dtype == np.uint8


def test_prepare_never_reads_the_test_split(rows, tmp_path):
    test_video = make_video(tmp_path / "t.avi", 1, seed=99, n_frames=70)
    prepare_cache(rows + [Row(str(test_video), 1, "gT", "synth", "test")], tmp_path / "c", SMALL, 2,
                  detector=FixedBoxDetector())
    index, _ = read_index(tmp_path / "c")
    assert "t.avi" not in " ".join(r["path"] for r in index)
    assert len(list((tmp_path / "c").glob("*.npy"))) == 8


def test_prepare_marks_faceless_and_unreadable_videos(rows, tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"junk" * 50)
    extra = rows + [Row(str(bad), 0, "gX", "synth", "train")]
    res = prepare_cache(extra, tmp_path / "c", SMALL, 2, detector=FixedBoxDetector())
    assert res["unusable"] == 1
    st = {Path(r["path"]).name: r["status"] for r in read_index(tmp_path / "c")[0]}
    assert st["bad.mp4"].startswith("error:")
    res2 = prepare_cache(rows, tmp_path / "c2", SMALL, 2, detector=NoFaceDetector())
    assert res2["usable"] == 0
    assert {r["status"] for r in read_index(tmp_path / "c2")[0]} == {"no_usable_clips"}


def test_prepare_resumes_without_recomputing(rows, tmp_path):
    prepare_cache(rows, tmp_path / "c", SMALL, 2, detector=FixedBoxDetector())

    class Boom:
        def detect(self, rgb):
            raise AssertionError("face detector must not run for cached videos")

    res = prepare_cache(rows, tmp_path / "c", SMALL, 2, detector=Boom())
    assert res["usable"] == 8


def test_prepare_refuses_cache_built_with_other_settings(rows, tmp_path):
    prepare_cache(rows, tmp_path / "c", SMALL, 2, detector=FixedBoxDetector())
    other = dataclasses.replace(SMALL, image_size=96)
    with pytest.raises(EvalError, match="different settings"):
        prepare_cache(rows, tmp_path / "c", other, 2, detector=FixedBoxDetector())


def test_prepare_refuses_identical_video_in_two_splits(rows, tmp_path):
    copy = tmp_path / "copy.avi"
    copy.write_bytes(Path(rows[0].path).read_bytes())
    leaky = rows + [Row(str(copy), 0, "gZ", "synth", "val")]          # v0 is in train
    with pytest.raises(EvalError, match="Identical video content"):
        prepare_cache(leaky, tmp_path / "c", SMALL, 2, detector=FixedBoxDetector())


# ============================================================ dataset / scoring helpers
def test_cached_dataset_shapes_labels_and_augmentation(rows, tmp_path):
    prepare_cache(rows, tmp_path / "c", SMALL, 3, detector=FixedBoxDetector())
    index, _ = read_index(tmp_path / "c")
    ds = CachedClips(tmp_path / "c", index, train=False)
    x, y = ds[0]
    assert x.shape == (4, 3, 64, 64) and int(y) == index[0]["label"]
    assert -1.0 <= x.min() and x.max() <= 1.0
    assert torch.equal(ds[0][0], ds[0][0])                              # eval mode is deterministic
    tr = CachedClips(tmp_path / "c", index, train=True)
    assert any(not torch.equal(tr[0][0], tr[0][0]) for _ in range(20))  # train mode augments


def test_video_scores_are_probabilities(rows, tmp_path):
    from src.model import DeepfakeDetector
    prepare_cache(rows, tmp_path / "c", SMALL, 2, detector=FixedBoxDetector())
    index, _ = read_index(tmp_path / "c")
    ys, ss, loss = video_scores(DeepfakeDetector(pretrained=False), tmp_path / "c", index[:3], "cpu")
    assert len(ys) == len(ss) == 3 and all(0 <= s <= 1 for s in ss) and np.isfinite(loss)


# ============================================================ train
@pytest.fixture()
def cache(rows, tmp_path):
    prepare_cache(rows, tmp_path / "c", SMALL, 2, detector=FixedBoxDetector())
    return tmp_path / "c"


def run_train(cache, out, **kw):
    args = dict(cfg=SMALL, epochs=2, batch_size=2, lr=1e-3, device="cpu", pretrained=False, log=lambda *_: None)
    args.update(kw)
    return train(cache, out, **args)


def test_train_writes_loadable_checkpoint_with_provenance(cache, tmp_path):
    out = tmp_path / "out" / "model_v2.pth"
    res = run_train(cache, out, manifest_sha256="abc123")
    assert out.is_file() and res["best_epoch"] in {1, 2}
    loaded = load_trained_model(out, "cpu")                              # strict load, app-compatible
    ck = torch.load(out, map_location="cpu", weights_only=False)
    assert ck["config"]["image_size"] == 64 and ck["training"]["manifest_sha256"] == "abc123"
    assert ck["training"]["n_train_videos"] == 4 and ck["training"]["pretrained_imagenet"] is False
    assert 0.0 <= ck["best_val_auc"] <= 1.0 and loaded.epoch == ck["epoch"]
    rows_ = list(csv.reader(open(out.with_suffix(".log.csv"))))
    assert rows_[0][0] == "epoch" and len(rows_) == 3


def test_train_updates_weights(cache, tmp_path):
    from src.model import DeepfakeDetector
    torch.manual_seed(7)
    init = DeepfakeDetector(pretrained=False).state_dict()
    out = tmp_path / "m.pth"
    run_train(cache, out, seed=7)
    new = load_trained_model(out, "cpu").model.state_dict()
    assert any(not torch.equal(init[k], new[k]) for k in init if init[k].is_floating_point())


def test_train_never_overwrites_existing_checkpoint(cache, tmp_path):
    out = tmp_path / "m.pth"
    out.write_bytes(b"precious")
    with pytest.raises(EvalError, match="never overwrites"):
        run_train(cache, out)
    assert out.read_bytes() == b"precious"


def test_train_refuses_cache_with_other_preprocessing(cache, tmp_path):
    with pytest.raises(EvalError, match="differ from the current configuration"):
        run_train(cache, tmp_path / "m.pth", cfg=dataclasses.replace(SMALL, image_size=96))


def test_train_refuses_single_class_split(rows, tmp_path):
    one = [dataclasses.replace(r, label=0) if r.split == "train" else r for r in rows]
    prepare_cache(one, tmp_path / "c", SMALL, 2, detector=FixedBoxDetector())
    with pytest.raises(EvalError, match="both classes"):
        run_train(tmp_path / "c", tmp_path / "m.pth")


def test_train_refuses_shared_groups_between_train_and_val(rows, tmp_path):
    leaky = [dataclasses.replace(r, group="same") for r in rows]
    prepare_cache(leaky, tmp_path / "c", SMALL, 2, detector=FixedBoxDetector())
    with pytest.raises(EvalError, match="share source groups"):
        run_train(tmp_path / "c", tmp_path / "m.pth")


def test_finetune_from_existing_checkpoint(cache, tmp_path, loaded_model):
    out = tmp_path / "ft.pth"
    run_train(cache, out, init_from=loaded_model.path, epochs=1)
    ck = torch.load(out, map_location="cpu", weights_only=False)
    assert ck["training"]["init_from"] == loaded_model.path and ck["training"]["pretrained_imagenet"] is False


def test_unprepared_cache_is_rejected(tmp_path):
    with pytest.raises(EvalError, match="not a prepared cache"):
        run_train(tmp_path, tmp_path / "m.pth")


# ============================================================ CLI
def test_cli_prepare_and_train_fail_cleanly_without_faces(rows, tmp_path, capsys):
    m = tmp_path / "m.csv"
    m.write_text("path,label,group,dataset,split\n" + "\n".join(
        f"{Path(r.path).name},{'fake' if r.label else 'real'},{r.group},synth,{r.split}" for r in rows) + "\n")
    c = tmp_path / "cache"
    assert main(["prepare", str(m), "--cache", str(c), "--device", "cpu"]) == 0      # real MTCNN: noise has no faces
    assert "0 usable" in capsys.readouterr().out
    assert main(["train", "--cache", str(c), "--out", str(tmp_path / "x.pth"), "--device", "cpu"]) == 2
    assert "both classes" in capsys.readouterr().err and not (tmp_path / "x.pth").exists()


def test_cli_prepare_rejects_leaking_manifest(rows, tmp_path, capsys):
    m = tmp_path / "m.csv"
    m.write_text("path,label,group,dataset,split\n" + "\n".join(
        f"{Path(r.path).name},{'fake' if r.label else 'real'},gSAME,synth,{r.split}" for r in rows) + "\n")
    assert main(["prepare", str(m), "--cache", str(tmp_path / "c")]) == 2
    assert "more than one split" in capsys.readouterr().err

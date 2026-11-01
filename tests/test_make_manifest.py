import csv
import json
import re

import pytest

from src.evaluate import load_manifest, validate_manifest
from src.make_manifest import (ManifestBuildError, build_celebdf_manifest, main, scan_celebdf,
                               split_identities)

N_ID, REALS_PER_ID, FAKES_PER_PAIR, N_YT = 20, 5, 2, 30


def make_tree(root, n_id=N_ID, junk=False):
    """Empty files named like Celeb-DF v2 (the tool only reads names, not contents)."""
    for d in ("Celeb-real", "Celeb-synthesis", "YouTube-real"):
        (root / d).mkdir(parents=True)
    for a in range(n_id):
        for k in range(REALS_PER_ID):
            (root / "Celeb-real" / f"id{a}_{k:04d}.mp4").write_bytes(b"r")
        for b in range(n_id):
            if a != b:
                for k in range(FAKES_PER_PAIR):
                    (root / "Celeb-synthesis" / f"id{a}_id{b}_{k:04d}.mp4").write_bytes(b"f")
    for k in range(N_YT):
        (root / "YouTube-real" / f"{k:05d}.mp4").write_bytes(b"y")
    if junk:
        (root / "Celeb-real" / "notes.txt").write_text("x")
        (root / "Celeb-synthesis" / "weird_name.mp4").write_bytes(b"x")
    return root


@pytest.fixture()
def tree(tmp_path):
    return make_tree(tmp_path / "Celeb-DF-v2")


def ids_of(path):
    return {int(x) for x in re.findall(r"id(\d+)", path.split("/")[-1].split("\\")[-1])}


def test_scan_counts_and_ids(tree):
    entries, skipped = scan_celebdf(tree)
    assert skipped == {}
    kinds = {k: [e for e in entries if e.kind == k] for k in ("real", "fake", "youtube")}
    assert len(kinds["real"]) == N_ID * REALS_PER_ID
    assert len(kinds["fake"]) == N_ID * (N_ID - 1) * FAKES_PER_PAIR
    assert len(kinds["youtube"]) == N_YT
    assert all(len(e.ids) == 2 for e in kinds["fake"]) and all(e.ids == () for e in kinds["youtube"])
    assert all(e.label == 1 for e in kinds["fake"]) and all(e.label == 0 for e in kinds["real"] + kinds["youtube"])


def test_no_identity_appears_in_two_splits(tree):
    rows, summary = build_celebdf_manifest(tree, seed=1)
    by_id: dict[int, set[str]] = {}
    for r in rows:
        if r.group.startswith("id"):
            for i in ids_of(r.path):
                by_id.setdefault(i, set()).add(r.split)
    assert by_id and all(len(s) == 1 for s in by_id.values())
    # the identity sets reported in the summary agree with what was written
    for sp, idents in summary["identities_per_split"].items():
        assert all(by_id.get(i, {sp}) == {sp} for i in idents)


def test_mixed_identity_fakes_are_dropped_and_counted(tree):
    rows, summary = build_celebdf_manifest(tree, seed=1)
    total = N_ID * REALS_PER_ID + N_ID * (N_ID - 1) * FAKES_PER_PAIR + N_YT
    assert summary["total_scanned"] == total
    assert summary["dropped_mixed_identity_videos"] > 0
    assert len(rows) == total - summary["dropped_mixed_identity_videos"] == summary["total_in_manifest"]
    # every kept fake has both identities in its split's set
    sets = {sp: set(v) for sp, v in summary["identities_per_split"].items()}
    for r in rows:
        if r.label == 1:
            assert ids_of(r.path) <= sets[r.split]


def test_split_sizes_follow_ratios_and_all_splits_have_both_classes(tree):
    _, summary = build_celebdf_manifest(tree, ratios=(0.6, 0.2, 0.2), seed=3)
    ids = summary["identities_per_split"]
    assert (len(ids["train"]), len(ids["val"]), len(ids["test"])) == (12, 4, 4)
    for sp, c in summary["videos_per_split"].items():
        assert c["real"] > 0 and c["fake"] > 0, sp
    yt = sum(1 for r in build_celebdf_manifest(tree, seed=3)[0] if r.group.startswith("yt"))
    assert yt == N_YT                                    # YouTube-real are never dropped


def test_deterministic_and_seed_dependent(tree):
    a = build_celebdf_manifest(tree, seed=5)[0]
    b = build_celebdf_manifest(tree, seed=5)[0]
    c = build_celebdf_manifest(tree, seed=6)[0]
    key = lambda rows: [(r.path, r.split) for r in rows]
    assert key(a) == key(b) and key(a) != key(c)


def test_output_passes_the_evaluation_validator(tree, tmp_path):
    out = tmp_path / "m.csv"
    assert main(["celebdf", "--root", str(tree), "--out", str(out)]) == 0
    rows = load_manifest(out)                                  # strict reader used by score/train
    validate_manifest(rows)                                    # groups never span splits
    assert {r.dataset for r in rows} == {"Celeb-DF-v2"} and {r.split for r in rows} == {"train", "val", "test"}
    summary = json.loads((tmp_path / "m.csv.summary.json").read_text())
    assert summary["seed"] == 42 and "identities_per_split" in summary


def test_unrecognised_files_are_skipped_and_reported(tmp_path, capsys):
    tree = make_tree(tmp_path / "d", junk=True)
    out = tmp_path / "m.csv"
    assert main(["celebdf", "--root", str(tree), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "unrecognised file name" in text
    assert not any("notes.txt" in r["path"] or "weird_name" in r["path"] for r in csv.DictReader(open(out)))


def test_wrong_layout_fails_loudly_instead_of_guessing(tmp_path, capsys):
    for d in ("Celeb-real", "Celeb-synthesis", "YouTube-real"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "video_001.mp4").write_bytes(b"x")          # naming this tool does not understand
    assert main(["celebdf", "--root", str(tmp_path), "--out", str(tmp_path / "m.csv")]) == 2
    assert "differs from what this tool assumes" in capsys.readouterr().err
    assert not (tmp_path / "m.csv").exists()


def test_missing_folder_is_reported(tmp_path, capsys):
    make_tree(tmp_path / "d")
    import shutil
    shutil.rmtree(tmp_path / "d" / "YouTube-real")
    assert main(["celebdf", "--root", str(tmp_path / "d"), "--out", str(tmp_path / "m.csv")]) == 2
    assert "Folder not found" in capsys.readouterr().err


def test_empty_folder_is_reported_as_empty(tmp_path, capsys):
    (tmp_path / "Celeb-real").mkdir()
    assert main(["celebdf", "--root", str(tmp_path), "--out", str(tmp_path / "m.csv")]) == 2
    assert "contains no files" in capsys.readouterr().err


@pytest.mark.parametrize("ratios", [(0.5, 0.5, 0.5), (0.9, 0.1, 0.0), (0.6, 0.2)])
def test_bad_ratios_rejected(tree, tmp_path, ratios):
    if len(ratios) == 3:
        with pytest.raises(ManifestBuildError):
            split_identities(list(range(20)), ratios, 1)
    else:
        assert main(["celebdf", "--root", str(tree), "--out", str(tmp_path / "m.csv"),
                     "--ratios", ",".join(map(str, ratios))]) == 2


def test_too_few_identities_rejected():
    with pytest.raises(ManifestBuildError, match="not enough"):
        split_identities([1, 2], (0.6, 0.2, 0.2), 1)


def test_small_validation_split_is_flagged(tmp_path, capsys):
    tree = make_tree(tmp_path / "d", n_id=6)                  # tiny dataset
    assert main(["celebdf", "--root", str(tree), "--out", str(tmp_path / "m.csv")]) == 0
    assert "fit` will refuse" in capsys.readouterr().out

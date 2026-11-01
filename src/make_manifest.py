# src/make_manifest.py
"""Build a manifest (path,label,group,dataset,split) from a dataset folder.

    python -m src.make_manifest celebdf --root D:\\data\\Celeb-DF-v2 --out manifest.csv

Celeb-DF v2 layout and file naming (from the dataset's documentation, NOT verified against the
data by the author of this tool; the scan stops with a clear error if it does not match):

    Celeb-real/       id<N>_<XXXX>.mp4            real videos of celebrity N
    Celeb-synthesis/  id<A>_id<B>_<XXXX>.mp4      fake videos involving identities A and B
    YouTube-real/     <XXXXX>.mp4                 real videos, identity unknown

Split rule: IDENTITY-DISJOINT. The identities are divided into train / val / test sets. A video is
placed in a split only if every identity it involves belongs to that split's set; videos that mix
identities from different sets are dropped (and counted). So no person appears in two splits and a
test result measures performance on people the model never saw. YouTube-real videos have no known
identity; they are divided by video, so a person could in principle recur across splits.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from src.evaluate import EvalError, Row, validate_manifest

DATASET_NAME = "Celeb-DF-v2"
REAL_RE = re.compile(r"^id(\d+)_(\d+)\.mp4$", re.I)
FAKE_RE = re.compile(r"^id(\d+)_id(\d+)_(\d+)\.mp4$", re.I)
YT_RE = re.compile(r"^(\d+)\.mp4$", re.I)
SPLITS = ("train", "val", "test")


class ManifestBuildError(Exception):
    """The dataset folder or settings cannot produce a trustworthy manifest."""


@dataclass
class Entry:
    path: str
    label: int            # 0 real, 1 fake
    ids: tuple[int, ...]  # identities involved (empty when unknown)
    kind: str             # real | fake | youtube
    stem: str


def scan_celebdf(root: str | Path) -> tuple[list[Entry], dict]:
    root = Path(root)
    spec = [("Celeb-real", REAL_RE, 0, "real"), ("Celeb-synthesis", FAKE_RE, 1, "fake"),
            ("YouTube-real", YT_RE, 0, "youtube")]
    entries: list[Entry] = []
    skipped: dict[str, list[str]] = {}
    for folder, rx, label, kind in spec:
        d = root / folder
        if not d.is_dir():
            raise ManifestBuildError(
                f"Folder not found: {d}. --root must contain Celeb-real, Celeb-synthesis and YouTube-real.")
        matched = 0
        if not any(q.is_file() for q in d.iterdir()):
            raise ManifestBuildError(f"{d} contains no files. Has the dataset finished extracting?")
        for p in sorted(d.iterdir()):
            if not p.is_file():
                continue
            m = rx.match(p.name)
            if not m:
                skipped.setdefault(folder, []).append(p.name)
                continue
            ids = {"real": lambda: (int(m[1]),), "fake": lambda: (int(m[1]), int(m[2])),
                   "youtube": lambda: ()}[kind]()
            entries.append(Entry(str(p.resolve()), label, ids, kind, p.stem))
            matched += 1
        if matched == 0:
            raise ManifestBuildError(
                f"No file in {d} matches the expected naming ({rx.pattern}). The dataset layout differs "
                "from what this tool assumes; refusing to guess identities.")
    return entries, {k: len(v) for k, v in skipped.items()}


def split_identities(ids: list[int], ratios: tuple[float, float, float], seed: int) -> dict[int, str]:
    if abs(sum(ratios) - 1.0) > 1e-6 or any(r <= 0 for r in ratios):
        raise ManifestBuildError("ratios must be positive and sum to 1 (train,val,test).")
    ids = sorted(set(ids))
    n = len(ids)
    n_test, n_val = max(1, round(n * ratios[2])), max(1, round(n * ratios[1]))
    if n - n_test - n_val < 1:
        raise ManifestBuildError(f"Only {n} identities; not enough for three identity-disjoint splits.")
    order = ids[:]
    random.Random(seed).shuffle(order)
    out = {i: "test" for i in order[:n_test]}
    out.update({i: "val" for i in order[n_test:n_test + n_val]})
    out.update({i: "train" for i in order[n_test + n_val:]})
    return out


def _split_unlabelled(entries: list[Entry], ratios, seed: int) -> dict[str, str]:
    """Divide videos without a known identity by video, in the requested proportions."""
    stems = [e.path for e in sorted(entries, key=lambda e: e.path)]
    random.Random(seed + 1).shuffle(stems)
    n = len(stems)
    n_test, n_val = round(n * ratios[2]), round(n * ratios[1])
    out = {p: "test" for p in stems[:n_test]}
    out.update({p: "val" for p in stems[n_test:n_test + n_val]})
    out.update({p: "train" for p in stems[n_test + n_val:]})
    return out


def build_celebdf_manifest(root: str | Path, ratios=(0.6, 0.2, 0.2), seed: int = 42):
    entries, skipped = scan_celebdf(root)
    id_split = split_identities([i for e in entries for i in e.ids], ratios, seed)
    yt_split = _split_unlabelled([e for e in entries if e.kind == "youtube"], ratios, seed)

    assigned: list[tuple[Entry, str]] = []
    dropped_mixed = 0
    for e in entries:
        if e.kind == "youtube":
            assigned.append((e, yt_split[e.path]))
            continue
        sp = {id_split[i] for i in e.ids}
        if len(sp) == 1:
            assigned.append((e, sp.pop()))
        else:
            dropped_mixed += 1

    # Safeguard: no identity may appear in more than one split (holds by construction).
    seen: dict[int, set[str]] = {}
    for e, sp in assigned:
        for i in e.ids:
            seen.setdefault(i, set()).add(sp)
    leaked = sorted(i for i, s in seen.items() if len(s) > 1)
    if leaked:
        raise ManifestBuildError(f"Internal error: identities in more than one split: {leaked[:5]}")

    rows = [Row(e.path, e.label, f"id{e.ids[0]}" if e.ids else f"yt{e.stem}", DATASET_NAME, sp)
            for e, sp in sorted(assigned, key=lambda t: (t[1], t[0].path))]
    warnings = validate_manifest(rows, check_files=True)

    counts = {sp: {"real": 0, "fake": 0} for sp in SPLITS}
    for r in rows:
        counts[r.split]["fake" if r.label else "real"] += 1
    summary = {
        "dataset": DATASET_NAME, "seed": seed, "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "identities_per_split": {sp: sorted(i for i, s in id_split.items() if s == sp) for sp in SPLITS},
        "videos_per_split": counts, "dropped_mixed_identity_videos": dropped_mixed,
        "skipped_unrecognised_files": skipped, "total_scanned": len(entries), "total_in_manifest": len(rows),
        "warnings": warnings,
        "notes": ["Identity-disjoint for Celeb-real and Celeb-synthesis videos.",
                  "YouTube-real videos have no identity label and were divided by video."],
    }
    return rows, summary


def write_manifest(rows: list[Row], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "group", "dataset", "split"])
        for r in rows:
            w.writerow([r.path, "fake" if r.label else "real", r.group, r.dataset, r.split])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build a manifest from a dataset folder")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("celebdf", help="Celeb-DF v2, identity-disjoint split")
    c.add_argument("--root", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--ratios", default="0.6,0.2,0.2", help="train,val,test fractions of identities")
    a = ap.parse_args(argv)
    try:
        ratios = tuple(float(x) for x in a.ratios.split(","))
        if len(ratios) != 3:
            raise ManifestBuildError("--ratios needs three numbers: train,val,test")
        rows, summary = build_celebdf_manifest(a.root, ratios, a.seed)
        write_manifest(rows, a.out)
        Path(str(a.out) + ".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except (ManifestBuildError, EvalError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    print(f"Wrote {len(rows)} videos to {a.out} (summary: {a.out}.summary.json)")
    print(f"{'split':<7}{'real':>7}{'fake':>7}")
    for sp in SPLITS:
        cnt = summary["videos_per_split"][sp]
        flag = "  <- fewer than 20 of a class: `fit` will refuse" if sp == "val" and min(cnt.values()) < 20 else ""
        print(f"{sp:<7}{cnt['real']:>7}{cnt['fake']:>7}{flag}")
    print(f"Dropped {summary['dropped_mixed_identity_videos']} videos mixing identities from different splits.")
    for k, v in summary["skipped_unrecognised_files"].items():
        print(f"WARNING: {v} unrecognised file name(s) in {k} were skipped.")
    for w in summary["warnings"]:
        print("WARNING:", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())

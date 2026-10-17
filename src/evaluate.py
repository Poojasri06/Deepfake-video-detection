# src/evaluate.py
"""Reproducible evaluation: validate -> score -> fit -> report.

    python -m src.evaluate validate manifest.csv
    python -m src.evaluate score    manifest.csv --out scores_clean.csv [--condition jpeg:30]
    python -m src.evaluate fit      scores_clean.csv --out models/decision_policy.json
    python -m src.evaluate report   scores_clean.csv scores_jpeg30.csv --policy models/decision_policy.json

Manifest CSV columns: path,label,group,dataset,split
  label   real|fake (or 0|1)
  group   source id: identity or original-video id. Videos sharing a group must stay in the
          same split (a fake and the real video it was made from share one group).
  dataset name used for per-dataset reporting
  split   val (fit thresholds) | test (report only) | train (listed only for leakage checks)

Thresholds are fitted on the val split only. The report refuses test videos that were used
for fitting, and refuses to apply a policy fitted for a different checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from src.checkpoint import CheckpointError, LoadedModel, load_trained_model
from src.config import InferenceConfig
from src.decision import DecisionPolicy, PolicyError, classify
from src.metrics import (apply_platt, calibration_metrics, classification_metrics,
                         label_decision_metrics)
from src.pipeline import analyze_video, build_face_detector
from src.policy_fit import PolicyFitError, fit_policy, write_policy
from src.video import VideoError

SPLITS = {"train", "val", "test", ""}
SCORE_COLUMNS = ["path", "file_id", "label", "group", "dataset", "split", "status", "clips_used",
                 "clips_total", "face_coverage", "score_mean", "score_median", "score_min",
                 "score_max", "elapsed_s", "error"]


class EvalError(Exception):
    """A problem that makes the evaluation invalid. The message says what to fix."""


# ----------------------------------------------------------------------------- manifest
@dataclass
class Row:
    path: str
    label: int
    group: str
    dataset: str
    split: str


def _parse_label(v: str) -> int:
    t = str(v).strip().lower()
    if t in {"real", "0", "genuine", "authentic"}:
        return 0
    if t in {"fake", "1", "deepfake", "manipulated"}:
        return 1
    raise ValueError(f"label {v!r} is not real/fake/0/1")


def load_manifest(path: str | Path) -> list[Row]:
    p = Path(path)
    if not p.is_file():
        raise EvalError(f"Manifest not found: {p}")
    with open(p, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = {c.strip().lower() for c in (reader.fieldnames or [])}
        if not {"path", "label"} <= cols:
            raise EvalError("Manifest needs at least the columns: path,label (optional: group,dataset,split)")
        rows, errors = [], []
        for i, r in enumerate(reader, start=2):
            r = {k.strip().lower(): (v or "").strip() for k, v in r.items() if k}
            try:
                label = _parse_label(r["label"])
            except ValueError as e:
                errors.append(f"line {i}: {e}")
                continue
            vp = Path(r["path"])
            vp = vp if vp.is_absolute() else (p.parent / vp)
            rows.append(Row(str(vp), label, r.get("group", ""), r.get("dataset", "") or "unnamed",
                            r.get("split", "").lower()))
    if errors:
        raise EvalError("Invalid manifest:\n  " + "\n  ".join(errors))
    if not rows:
        raise EvalError("Manifest has no rows.")
    return rows


def validate_manifest(rows: list[Row], check_files: bool = True,
                      allow_ungrouped: bool = False) -> list[str]:
    """Raise EvalError on anything that would make results invalid; return warnings."""
    errors, warnings = [], []
    for r in rows:
        if r.split not in SPLITS:
            errors.append(f"{Path(r.path).name}: unknown split '{r.split}' (use train, val or test)")
        if check_files and not Path(r.path).is_file():
            errors.append(f"file not found: {r.path}")

    seen: dict[str, Row] = {}
    for r in rows:
        prev = seen.get(r.path)
        if prev and (prev.split != r.split or prev.label != r.label):
            errors.append(f"{Path(r.path).name} appears twice with different split/label")
        seen[r.path] = r

    splits_present = {r.split for r in rows if r.split}
    if len(splits_present) > 1:
        ungrouped = [r for r in rows if r.split and not r.group]
        if ungrouped and not allow_ungrouped:
            errors.append(f"{len(ungrouped)} videos have no 'group'; leakage between splits cannot be "
                          "checked. Add a group (identity or source-video id), or pass --allow-ungrouped "
                          "and treat results as possibly inflated.")
        elif ungrouped:
            warnings.append(f"{len(ungrouped)} videos have no group; leakage was NOT checked for them.")
    by_group: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if r.group and r.split:
            by_group[r.group].add(r.split)
    leaked = sorted(g for g, s in by_group.items() if len(s) > 1)
    if leaked:
        errors.append(f"{len(leaked)} group(s) appear in more than one split (e.g. {leaked[:3]}). "
                      "Videos from one source must stay in a single split.")
    if not splits_present:
        warnings.append("No split column: nothing can be fitted or reported separately.")
    for sp in sorted(splits_present - {"train"}):
        labs = {r.label for r in rows if r.split == sp}
        if len(labs) < 2:
            warnings.append(f"split '{sp}' contains only one class; AUC and rates are undefined.")
    if errors:
        raise EvalError("Manifest validation failed:\n  " + "\n  ".join(errors))
    return warnings


def file_id(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ------------------------------------------------------------------ degradation conditions
def make_condition(spec: str) -> Callable[[np.ndarray], np.ndarray] | None:
    """Frame-level degradations: none | jpeg:Q | scale:F | blur:S | brightness:F.

    These approximate common degradations on single frames. They are NOT a substitute for
    real social-media re-encoding of the video stream.
    """
    name, _, arg = spec.partition(":")
    name = name.strip().lower()
    if name in {"none", ""}:
        return None
    try:
        val = float(arg)
    except ValueError:
        raise EvalError(f"Condition '{spec}' needs a numeric argument, e.g. jpeg:30") from None

    if name == "jpeg":
        if not 1 <= val <= 100:
            raise EvalError("jpeg quality must be between 1 and 100")

        def f(rgb):
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, int(val)])
            return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return f
    if name == "scale":
        if not 0.05 <= val < 1:
            raise EvalError("scale factor must be in [0.05, 1)")

        def f(rgb):
            h, w = rgb.shape[:2]
            small = cv2.resize(rgb, (max(8, int(w * val)), max(8, int(h * val))), interpolation=cv2.INTER_AREA)
            return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        return f
    if name == "blur":
        if val <= 0:
            raise EvalError("blur sigma must be positive")
        return lambda rgb: cv2.GaussianBlur(rgb, (0, 0), val)
    if name == "brightness":
        if val <= 0:
            raise EvalError("brightness factor must be positive")
        return lambda rgb: np.clip(rgb.astype(np.float32) * val, 0, 255).astype(np.uint8)
    raise EvalError(f"Unknown condition '{name}'. Use none, jpeg:Q, scale:F, blur:S or brightness:F")


# ------------------------------------------------------------------------------ scoring
def score_manifest(rows: list[Row], loaded: LoadedModel, cfg: InferenceConfig, detector=None,
                   condition: str = "none", splits=("val", "test"),
                   progress: Callable[[int, int], None] | None = None) -> tuple[list[dict], dict]:
    transform = make_condition(condition)
    todo = [r for r in rows if r.split in splits]
    if not todo:
        raise EvalError(f"No manifest rows in split(s) {list(splits)}.")
    detector = detector or build_face_detector(next(loaded.model.parameters()).device)

    out: list[dict] = []
    ids: dict[str, Row] = {}
    for i, r in enumerate(todo):
        fid = file_id(r.path) if Path(r.path).is_file() else ""
        row = {"path": r.path, "file_id": fid, "label": r.label, "group": r.group,
               "dataset": r.dataset, "split": r.split, "status": "", "clips_used": None, "clips_total": None,
               "face_coverage": None, "score_mean": None, "score_median": None, "score_min": None,
               "score_max": None, "elapsed_s": None, "error": ""}
        if fid and fid in ids and ids[fid].split != r.split:
            raise EvalError(f"Identical video content appears in splits '{ids[fid].split}' and "
                            f"'{r.split}': {Path(r.path).name} == {Path(ids[fid].path).name}")
        ids.setdefault(fid, r)
        try:
            res = analyze_video(r.path, loaded, cfg, detector=detector, frame_transform=transform)
            row.update(status=res.status, clips_used=res.clips_used, clips_total=len(res.clips),
                       face_coverage=round(res.face_coverage, 4), elapsed_s=round(res.elapsed_s, 2))
            if res.status == "ok":
                row.update(score_mean=res.score_mean, score_median=res.score_median,
                           score_min=res.score_min, score_max=res.score_max)
        except VideoError as e:
            row.update(status="error", error=e.code)
        except Exception as e:  # never abort a long run on one video; log type only, not content
            row.update(status="error", error=f"internal:{type(e).__name__}")
        out.append(row)
        if progress:
            progress(i + 1, len(todo))

    meta = {"checkpoint_sha256": loaded.sha256, "checkpoint_name": Path(loaded.path).name,
            "condition": condition, "n_videos": len(out), "created": dt.datetime.now().isoformat(timespec="seconds"),
            "config": {k: v for k, v in dataclasses.asdict(cfg).items()
                       if k not in {"checkpoint_path", "policy_path"}}}
    return out, meta


def write_scores(rows: list[dict], meta: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SCORE_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    p.with_suffix(p.suffix + ".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def read_scores(path: str | Path) -> tuple[list[dict], dict]:
    p = Path(path)
    mp = p.with_suffix(p.suffix + ".meta.json")
    if not p.is_file() or not mp.is_file():
        raise EvalError(f"Scores file or its .meta.json missing for {p}")
    with open(p, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["label"] = int(r["label"])
        for k in ("face_coverage", "score_mean", "score_median", "score_min", "score_max"):
            r[k] = float(r[k]) if r.get(k, "") != "" else None
    return rows, json.loads(mp.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------- fit
def fit_from_scores(rows: list[dict], meta: dict, split: str = "val", allow_degraded: bool = False,
                    **fit_kwargs) -> dict:
    if meta["condition"] != "none" and not allow_degraded:
        raise EvalError(f"Scores were produced under condition '{meta['condition']}'. Thresholds are "
                        "fitted on clean validation data; pass --allow-degraded-fit to override.")
    val = [r for r in rows if r["split"] == split]
    if not val:
        raise EvalError(f"No rows with split '{split}' in the scores file.")
    ok = [r for r in val if r["status"] == "ok"]
    excluded = len(val) - len(ok)
    y = [r["label"] for r in ok]
    datasets = sorted({r["dataset"] for r in ok})
    try:
        policy = fit_policy(y, {"mean": [r["score_mean"] for r in ok],
                                "median": [r["score_median"] for r in ok]},
                            meta["checkpoint_sha256"], datasets, **fit_kwargs)
    except PolicyFitError as e:
        raise EvalError(str(e)) from e
    policy["validation"]["excluded_no_score"] = excluded
    policy["validation"]["fit_file_ids"] = sorted({r["file_id"] for r in val if r["file_id"]})
    policy["validation"]["condition"] = meta["condition"]
    if excluded:
        policy["validation"]["note"] += (f" {excluded} of {len(val)} validation videos produced no score "
                                         "(insufficient faces or unreadable) and were excluded from fitting.")
    return policy


# -------------------------------------------------------------------------------- report
def _subset_report(rows: list[dict], policy: DecisionPolicy | None) -> dict:
    y_all = [r["label"] for r in rows]
    scored = [r for r in rows if r["status"] == "ok"]
    out = {"n_videos": len(rows), "n_scored": len(scored),
           "n_no_score": sum(r["status"] == "insufficient_evidence" for r in rows),
           "n_unreadable": sum(r["status"] == "error" for r in rows)}
    if scored:
        agg = "score_mean" if policy is None or policy.aggregation == "mean" else "score_median"
        ys, ss = [r["label"] for r in scored], [r[agg] for r in scored]
        out["scored_videos_at_0.5"] = classification_metrics(ys, ss, 0.5)
        out["aggregation_used"] = agg.replace("score_", "")
    if policy is not None:
        labels = [classify(policy, r["status"] if r["status"] == "ok" else "insufficient_evidence",
                           r["score_mean"], r["score_median"], r["face_coverage"] or 0.0,
                           r["score_min"], r["score_max"]).label for r in rows]
        out["decision"] = label_decision_metrics(y_all, labels)
        if policy.calibration and scored:
            p = apply_platt([r["score_median" if policy.aggregation == "median" else "score_mean"]
                             for r in scored], policy.calibration)
            out["calibration_on_test"] = calibration_metrics([r["label"] for r in scored], p)
    return out


def build_report(score_files: list[str | Path], policy_dict: dict | None, split: str = "test") -> dict:
    policy = DecisionPolicy.from_dict(policy_dict) if policy_dict else None
    fit_ids = set((policy_dict or {}).get("validation", {}).get("fit_file_ids", []))
    entries = []
    for sf in score_files:
        rows, meta = read_scores(sf)
        test = [r for r in rows if r["split"] == split]
        if not test:
            raise EvalError(f"{sf}: no rows with split '{split}'.")
        overlap = [r for r in test if r["file_id"] and r["file_id"] in fit_ids]
        if overlap:
            raise EvalError(f"{len(overlap)} '{split}' video(s) were used to fit the policy "
                            f"(e.g. {Path(overlap[0]['path']).name}). Results would be invalid.")
        applies = policy is not None and policy.checkpoint_sha256 == meta["checkpoint_sha256"]
        pol = policy if applies else None
        entry = {"scores_file": str(sf), "checkpoint": meta["checkpoint_name"],
                 "checkpoint_sha256": meta["checkpoint_sha256"], "condition": meta["condition"],
                 "policy_applied": applies,
                 "policy_note": None if applies or policy is None else
                 "Policy was fitted for a different checkpoint; only threshold-free metrics shown.",
                 "overall": _subset_report(test, pol), "per_dataset": {}}
        for ds in sorted({r["dataset"] for r in test}):
            entry["per_dataset"][ds] = _subset_report([r for r in test if r["dataset"] == ds], pol)
        entries.append(entry)
    return {"split": split, "created": dt.datetime.now().isoformat(timespec="seconds"),
            "policy": None if not policy_dict else {k: policy_dict[k] for k in
                                                     ("aggregation", "real_below", "fake_above",
                                                      "min_face_coverage", "max_clip_spread")},
            "results": entries}


def _pct(rate: dict | None) -> str:
    if not rate or rate["value"] is None:
        return "n/a"
    ci = rate["ci95"]
    return f"{rate['value']:.1%} ({rate['count']}/{rate['of']}; 95% CI {ci[0]:.1%}-{ci[1]:.1%})"


def _f(v, d=3):
    return "n/a" if v is None else f"{v:.{d}f}"


def render_markdown(report: dict) -> str:
    L = ["# Evaluation report", "",
         f"Split: `{report['split']}` | generated {report['created']}", "",
         "AI assessment results on the listed data only. They do not show how the model behaves on "
         "other datasets, manipulation methods or real-world uploads.", ""]
    if report["policy"]:
        p = report["policy"]
        L += [f"Policy: aggregation={p['aggregation']}, real_below={p['real_below']}, "
              f"fake_above={p['fake_above']}, min_face_coverage={p['min_face_coverage']}, "
              f"max_clip_spread={p['max_clip_spread']}", ""]
    L += ["## Summary by checkpoint and condition", "",
          "| checkpoint | condition | videos | scored | no score | unreadable | ROC-AUC | labelled | "
          "FPR (real flagged fake) | FNR (fake passed as real) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for e in report["results"]:
        o = e["overall"]
        c = o.get("scored_videos_at_0.5", {})
        d = o.get("decision")
        L.append(f"| {e['checkpoint']} ({e['checkpoint_sha256'][:8]}) | {e['condition']} | {o['n_videos']} | "
                 f"{o['n_scored']} | {o['n_no_score']} | {o['n_unreadable']} | {_f(c.get('roc_auc'))} | "
                 f"{'n/a' if not d else format(d['coverage'], '.0%')} | "
                 f"{_pct(d['false_positive_rate']) if d else 'n/a'} | "
                 f"{_pct(d['false_negative_rate']) if d else 'n/a'} |")
    for e in report["results"]:
        L += ["", f"## {e['checkpoint']} / {e['condition']}", ""]
        if e["policy_note"]:
            L += [f"_{e['policy_note']}_", ""]
        for name, sub in [("All datasets", e["overall"])] + list(e["per_dataset"].items()):
            L += [f"### {name}", ""]
            c = sub.get("scored_videos_at_0.5")
            if c:
                cf = c["confusion"]
                L += [f"- Scored videos at threshold 0.5 ({sub['aggregation_used']}): accuracy {_f(c['accuracy'])}, "
                      f"precision {_f(c['precision'])}, recall {_f(c['recall'])}, F1 {_f(c['f1'])}, "
                      f"ROC-AUC {_f(c['roc_auc'])}, PR-AUC {_f(c['pr_auc'])}",
                      f"- Confusion (TN/FP/FN/TP): {cf['tn']}/{cf['fp']}/{cf['fn']}/{cf['tp']}"]
            d = sub.get("decision")
            if d:
                L += [f"- Decision outcome: {d['predicted_real']} likely real, {d['predicted_fake']} likely fake, "
                      f"{d['inconclusive']} inconclusive (coverage {d['coverage']:.0%})",
                      f"- Real videos labelled likely fake: {_pct(d['false_positive_rate'])}",
                      f"- Fake videos labelled likely real: {_pct(d['false_negative_rate'])}",
                      f"- Error among labelled videos: {_f(d['error_rate_among_decided'])}"]
            cal = sub.get("calibration_on_test")
            if cal:
                L += [f"- Calibration on test: ECE {_f(cal['ece'])}, Brier {_f(cal['brier'])}"]
            L.append("")
    return "\n".join(L)


# ------------------------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate the deepfake detector")
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="check a manifest (files, labels, split leakage)")
    v.add_argument("manifest")
    v.add_argument("--allow-ungrouped", action="store_true")

    s = sub.add_parser("score", help="run the model on val/test videos")
    s.add_argument("manifest")
    s.add_argument("--out", required=True)
    s.add_argument("--checkpoint", default=None)
    s.add_argument("--device", default=None)
    s.add_argument("--condition", default="none")
    s.add_argument("--splits", default="val,test")
    s.add_argument("--allow-ungrouped", action="store_true")

    f = sub.add_parser("fit", help="fit decision thresholds on the val split")
    f.add_argument("scores")
    f.add_argument("--out", default="models/decision_policy.json")
    f.add_argument("--max-fpr", type=float, default=0.05)
    f.add_argument("--max-fnr", type=float, default=0.10)
    f.add_argument("--min-coverage", type=float, default=0.5)
    f.add_argument("--max-error-among-decided", type=float, default=0.10)
    f.add_argument("--min-per-class", type=int, default=20)
    f.add_argument("--aggregation", choices=["mean", "median"], default=None)
    f.add_argument("--calibrate", action="store_true")
    f.add_argument("--allow-degraded-fit", action="store_true")

    r = sub.add_parser("report", help="held-out test report(s)")
    r.add_argument("scores", nargs="+")
    r.add_argument("--policy", default=None)
    r.add_argument("--out", default="evaluation_report")

    a = ap.parse_args(argv)
    try:
        if a.cmd == "validate":
            warns = validate_manifest(load_manifest(a.manifest), allow_ungrouped=a.allow_ungrouped)
            for w in warns:
                print("WARNING:", w)
            print("Manifest OK.")
        elif a.cmd == "score":
            rows = load_manifest(a.manifest)
            for w in validate_manifest(rows, allow_ungrouped=a.allow_ungrouped):
                print("WARNING:", w)
            ov = {k: x for k, x in {"checkpoint_path": a.checkpoint, "device": a.device}.items() if x}
            cfg = InferenceConfig(**ov)
            loaded = load_trained_model(cfg.checkpoint_path, cfg.device)
            out, meta = score_manifest(rows, loaded, cfg, condition=a.condition,
                                       splits=tuple(a.splits.split(",")),
                                       progress=lambda d, t: print(f"\r{d}/{t}", end="", flush=True))
            print()
            write_scores(out, meta, a.out)
            print(f"Wrote {len(out)} scores to {a.out}")
        elif a.cmd == "fit":
            rows, meta = read_scores(a.scores)
            pol = fit_from_scores(rows, meta, allow_degraded=a.allow_degraded_fit, max_fpr=a.max_fpr,
                                  max_fnr=a.max_fnr, min_coverage=a.min_coverage,
                                  max_error_among_decided=a.max_error_among_decided,
                                  min_per_class=a.min_per_class, aggregation=a.aggregation,
                                  calibrate=a.calibrate)
            write_policy(pol, a.out)
            print(f"Policy written to {a.out}: aggregation={pol['aggregation']} "
                  f"real_below={pol['real_below']} fake_above={pol['fake_above']}")
        elif a.cmd == "report":
            pol = json.loads(Path(a.policy).read_text(encoding="utf-8")) if a.policy else None
            rep = build_report(a.scores, pol)
            Path(a.out + ".json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
            md = render_markdown(rep)
            Path(a.out + ".md").write_text(md, encoding="utf-8")
            print(md)
    except (EvalError, CheckpointError, PolicyError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

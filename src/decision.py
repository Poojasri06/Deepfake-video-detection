# src/decision.py
"""Three-way decision (LIKELY_REAL / LIKELY_FAKE / INCONCLUSIVE) on top of raw scores.

Thresholds are never hard-coded. They come from a policy file produced by the
evaluation script on validation data, and the policy is bound to the SHA-256 of
the checkpoint it was fitted for. With no policy the label is UNVALIDATED and
only the raw score is shown.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from src.pipeline import AnalysisResult

LIKELY_REAL = "LIKELY_REAL"
LIKELY_FAKE = "LIKELY_FAKE"
INCONCLUSIVE = "INCONCLUSIVE"
UNVALIDATED = "UNVALIDATED"   # no validated policy for this checkpoint

SCHEMA_VERSION = 1
_EPS = 1e-9   # makes the inclusive threshold comparisons robust to float rounding


class PolicyError(ValueError):
    """The decision policy file is invalid or belongs to a different checkpoint."""


@dataclass(frozen=True)
class DecisionPolicy:
    aggregation: str            # "mean" | "median"
    real_below: float           # score <= real_below -> LIKELY_REAL
    fake_above: float           # score >= fake_above -> LIKELY_FAKE
    min_face_coverage: float    # below this the evidence is considered unreliable
    max_clip_spread: float      # max-min of clip scores above this -> clips disagree
    calibration: dict | None    # {"method": "platt", "a": float, "b": float} or None
    validation: dict            # provenance: datasets, counts, date, metrics
    checkpoint_sha256: str

    @staticmethod
    def from_dict(d: dict) -> "DecisionPolicy":
        try:
            if d.get("schema_version") != SCHEMA_VERSION:
                raise PolicyError(f"Unsupported policy schema_version {d.get('schema_version')!r}.")
            pol = DecisionPolicy(
                aggregation=d["aggregation"], real_below=float(d["real_below"]),
                fake_above=float(d["fake_above"]), min_face_coverage=float(d["min_face_coverage"]),
                max_clip_spread=float(d["max_clip_spread"]), calibration=d.get("calibration"),
                validation=dict(d["validation"]), checkpoint_sha256=str(d["checkpoint_sha256"]))
        except (KeyError, TypeError, ValueError) as e:
            if isinstance(e, PolicyError):
                raise
            raise PolicyError(f"Invalid decision policy: {e!r}") from e
        pol._check()
        return pol

    def _check(self):
        if self.aggregation not in {"mean", "median"}:
            raise PolicyError("aggregation must be 'mean' or 'median'.")
        if not (0.0 <= self.real_below < self.fake_above <= 1.0):
            raise PolicyError("Need 0 <= real_below < fake_above <= 1.")
        if not 0.0 <= self.min_face_coverage <= 1.0 or not 0.0 < self.max_clip_spread <= 1.0:
            raise PolicyError("min_face_coverage must be in [0,1] and max_clip_spread in (0,1].")
        v = self.validation
        if not v.get("datasets") or not (v.get("n_real", 0) > 0 and v.get("n_fake", 0) > 0):
            raise PolicyError("Policy must record the validation datasets and positive n_real/n_fake; "
                              "thresholds without validation provenance are not accepted.")
        c = self.calibration
        if c is not None and not (c.get("method") == "platt" and
                                  all(isinstance(c.get(k), (int, float)) for k in ("a", "b"))):
            raise PolicyError("calibration must be null or {'method': 'platt', 'a': float, 'b': float}.")


def load_policy(path: str | Path, checkpoint_sha256: str) -> DecisionPolicy | None:
    """Return the policy, or None if the file does not exist.

    Raises PolicyError if the file is malformed or was fitted for another checkpoint
    (using thresholds validated for a different model would be misleading).
    """
    p = Path(path)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise PolicyError(f"Could not read decision policy {p.name}: {e}") from e
    pol = DecisionPolicy.from_dict(data)
    if pol.checkpoint_sha256 != checkpoint_sha256:
        raise PolicyError(
            f"Decision policy {p.name} was fitted for a different checkpoint "
            f"({pol.checkpoint_sha256[:12]}... vs loaded {checkpoint_sha256[:12]}...). "
            "Re-run the evaluation script for this checkpoint.")
    return pol


def calibrate(score: float, calibration: dict) -> float:
    """Platt scaling on the logit of the raw score: sigmoid(a * logit(s) + b)."""
    s = min(max(score, 1e-6), 1 - 1e-6)
    z = calibration["a"] * math.log(s / (1 - s)) + calibration["b"]
    return 1.0 / (1.0 + math.exp(-z))


@dataclass
class Decision:
    label: str
    reasons: list[str] = field(default_factory=list)
    score: float | None = None                 # aggregated raw score used for the decision
    calibrated_p_fake: float | None = None     # only when the validated policy has calibration

    @property
    def summary(self) -> str:
        return {LIKELY_REAL: "Likely real", LIKELY_FAKE: "Likely fake / manipulated",
                INCONCLUSIVE: "Inconclusive", UNVALIDATED: "Not validated"}[self.label]


def classify(policy: DecisionPolicy | None, status: str, mean: float | None, median: float | None,
             coverage: float, smin: float | None, smax: float | None) -> Decision:
    """The decision rules on plain numbers. Shared by the app and the evaluation tool so
    that reported test results describe exactly what users see."""
    if status != "ok":
        return Decision(INCONCLUSIVE, ["Not enough usable faces/clips to apply the model reliably."])
    if policy is None:
        return Decision(UNVALIDATED, [
            "Decision thresholds have not been validated for this checkpoint, so no REAL/FAKE "
            "label is given. Only the raw model score is available."], score=mean)

    score = mean if policy.aggregation == "mean" else median
    cal = calibrate(score, policy.calibration) if policy.calibration else None
    reasons: list[str] = []
    if coverage < policy.min_face_coverage:
        reasons.append(f"Face found in only {coverage:.0%} of sampled frames "
                       f"(needs {policy.min_face_coverage:.0%}).")
    if smax - smin > policy.max_clip_spread:
        reasons.append(f"Clips disagree strongly (scores range {smin:.2f}-{smax:.2f}).")
    if reasons:
        return Decision(INCONCLUSIVE, reasons, score, cal)

    if score >= policy.fake_above - _EPS:
        return Decision(LIKELY_FAKE, [f"Score {score:.2f} is at or above the validated fake threshold "
                                      f"{policy.fake_above:.2f}."], score, cal)
    if score <= policy.real_below + _EPS:
        return Decision(LIKELY_REAL, [f"Score {score:.2f} is at or below the validated real threshold "
                                      f"{policy.real_below:.2f}. This does not prove authenticity."],
                        score, cal)
    return Decision(INCONCLUSIVE, [f"Score {score:.2f} lies between the validated thresholds "
                                   f"{policy.real_below:.2f} and {policy.fake_above:.2f}."], score, cal)


def decide(result: AnalysisResult, policy: DecisionPolicy | None) -> Decision:
    return classify(policy, result.status, result.score_mean, result.score_median,
                    result.face_coverage, result.score_min, result.score_max)

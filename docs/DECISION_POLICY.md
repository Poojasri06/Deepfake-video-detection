# Decision policy

The model outputs a raw score per clip: softmax P(class FAKE). That number is **not** a calibrated
probability. The app turns scores into a label only through a *decision policy* file that is
produced from validation data and is tied to one specific checkpoint.

Default location: `models/decision_policy.json` (override with `DEEPFAKE_POLICY`).

## Outcomes

| Label | When |
|---|---|
| `LIKELY_FAKE` | aggregated score >= `fake_above` and the evidence checks below pass |
| `LIKELY_REAL` | aggregated score <= `real_below` and the evidence checks pass. Does not prove authenticity |
| `INCONCLUSIVE` | score between the thresholds, too few usable clips, face coverage below `min_face_coverage`, or clip scores spread wider than `max_clip_spread` |
| `UNVALIDATED` | no policy file exists for the loaded checkpoint; only the raw score is shown |

## File format (schema_version 1)

```json
{
  "schema_version": 1,
  "checkpoint_sha256": "<sha256 of the .pth the thresholds were fitted for>",
  "aggregation": "median",
  "real_below": 0.0,
  "fake_above": 1.0,
  "min_face_coverage": 0.6,
  "max_clip_spread": 0.5,
  "calibration": {"method": "platt", "a": 1.0, "b": 0.0},
  "validation": {"datasets": ["<name>"], "n_real": 0, "n_fake": 0, "date": "YYYY-MM-DD"}
}
```

The numbers above are placeholders to show the shape, not usable values. Do not hand-write
thresholds: `src/policy_fit.py` fits them on validation videos and writes this file.

## How thresholds are fitted (`src/policy_fit.py`)

- Inputs: one score per validation video (mean and median over clips), labels, the checkpoint
  SHA-256 and the dataset names.
- `fake_above` is the lowest cut with false-positive rate on real videos <= `max_fpr` (default 5%).
  `real_below` is the highest cut with false-negative rate on fake videos <= `max_fnr` (default 10%).
  Scores in between are INCONCLUSIVE. If one cut already beats both targets the band is narrow
  (about 0.01) around the midpoint.
- Meeting those two rates alone is not enough: a model with no signal can meet them by labelling
  only the extreme tails. The fit therefore also requires at least `min_coverage` (default 50%) of
  videos to receive a label and an error rate among labelled videos of at most 10%.
- It **refuses to fit** (and writes nothing) if the validation set has fewer than 20 real or 20 fake
  videos, if no dataset name is given, or if the data cannot meet the requirements. The error
  says which requirement failed.
- The aggregation (mean or median) is the one with the higher validation ROC-AUC unless forced;
  both AUCs are recorded.
- `min_face_coverage` and `max_clip_spread` are **not** fitted. They are defaults and are listed
  under `default_fields_not_validated`.
- Figures stored in the policy describe the data the thresholds were fitted on and are optimistic.
  Report performance only from a separate held-out test set.

## Rules enforced in code

- A policy whose `checkpoint_sha256` differs from the loaded checkpoint is rejected with an error.
  Thresholds fitted for another model are never applied.
- `validation.datasets` must be non-empty and `n_real`/`n_fake` positive, so thresholds without
  provenance are refused.
- `calibration` is optional. Only when present does the app show a "Calibrated P(fake)" figure,
  together with the datasets it was calibrated on. It is never presented as the probability that a
  video is authentic.
- A malformed policy stops the app with a diagnostic; it does not silently fall back to defaults.

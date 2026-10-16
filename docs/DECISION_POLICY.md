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
thresholds: the evaluation script (next stage) fits them on held-out validation videos and writes
this file.

## Rules enforced in code

- A policy whose `checkpoint_sha256` differs from the loaded checkpoint is rejected with an error.
  Thresholds fitted for another model are never applied.
- `validation.datasets` must be non-empty and `n_real`/`n_fake` positive, so thresholds without
  provenance are refused.
- `calibration` is optional. Only when present does the app show a "Calibrated P(fake)" figure,
  together with the datasets it was calibrated on. It is never presented as the probability that a
  video is authentic.
- A malformed policy stops the app with a diagnostic; it does not silently fall back to defaults.

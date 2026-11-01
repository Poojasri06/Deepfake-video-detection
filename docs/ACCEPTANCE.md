# Acceptance checklist

Status of the delivery criteria. "Verified" means executed here and passed; the test suite uses
synthetic checkpoints and videos, so it verifies code behaviour, not model accuracy.

| Criterion | Status | Evidence / what is missing |
|---|---|---|
| Original repository preserved | Met | `main` history intact (original commit `8f4307d`); work is on `feature/deepfake-detection-v2`. Note: the author merged some of this branch into `main` through PRs |
| Original trained checkpoint preserved | **Not possible** | It was never in the repository, and the author has since deleted it along with the datasets. A new model must be trained |
| App loads the actual trained model | **Not verified** | No real trained model exists. The loader is tested with synthetic checkpoints and with a checkpoint produced by `src.retrain` on a synthetic task |
| Retraining path available | Met, verified on synthetic data only | `src/retrain.py`; tests cover parity with inference tensors, leakage refusals, no-overwrite, strict-loadable output. Not run on real deepfake data or a GPU |
| Preprocessing matches training | Verified for what the code can check | Crop, resize, normalisation, clip stride and frame size tested against `train.py` (tensor equality with the training transform; constants compared with `train.py` CONFIG). Equivalence on real faces not tested |
| Predictions come from real model inference | Verified (mechanics) | Pipeline runs the model on the exact tensors tested; no meaningful outputs without trained weights |
| Bad inputs and loading failures handled safely | Verified | Tests for missing, corrupt, wrong-architecture, wrong-shape checkpoints; empty, corrupt, unsupported, oversized, too long, too short videos; no faces |
| Thresholds and confidence supported by evaluation | **Not met** | Machinery done and tested; no validation data run, so no thresholds exist and the app shows "Not validated" |
| Test suite passes, limits documented | Met for the synthetic suite | `python -m pytest -q`: all tests pass. Not covered: real weights, real face videos, GPU, Windows, Python 3.10, the Streamlit upload and analyse click |
| App runs with documented commands | Partly verified | App starts and renders in Streamlit's test runner (missing/invalid/valid checkpoint, with/without policy, all result states). A real browser upload was not exercised |
| README honest about capabilities and limits | Met | No accuracy claims; scope, limits and failure cases listed |
| Updated code in a separate branch | Met | `feature/deepfake-detection-v2` on GitHub |

## To finish (needs the author's files)

1. Download licensed datasets and build a manifest with source groups and `train` / `val` / `test` splits. For Celeb-DF v2 use `src.make_manifest` (see `docs/DATA_CELEBDF.md`).
2. `python -m src.retrain prepare` then `train` on a GPU (see README).
3. `python -m src.evaluate validate / score / fit / report` on the held-out test split.
4. Publish the measured report and keep claims within it.

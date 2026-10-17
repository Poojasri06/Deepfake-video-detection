# Acceptance checklist

Status of the delivery criteria. "Verified" means executed here and passed; the test suite uses
synthetic checkpoints and videos, so it verifies code behaviour, not model accuracy.

| Criterion | Status | Evidence / what is missing |
|---|---|---|
| Original repository preserved | Met | `main` history intact (original commit `8f4307d`); work is on `feature/deepfake-detection-v2`. Note: the author merged some of this branch into `main` through PRs |
| Original trained checkpoint preserved | **Not applicable yet** | The checkpoint was never in the repository and has not been supplied |
| App loads the actual trained model | **Not verified** | Loader is tested only with synthetic checkpoints. Needs the real file |
| Preprocessing matches training | Verified for what the code can check | Crop, resize, normalisation, clip stride and frame size tested against `train.py` (tensor equality with the training transform; constants compared with `train.py` CONFIG). Equivalence on real faces not tested |
| Predictions come from real model inference | Verified (mechanics) | Pipeline runs the model on the exact tensors tested; no meaningful outputs without trained weights |
| Bad inputs and loading failures handled safely | Verified | Tests for missing, corrupt, wrong-architecture, wrong-shape checkpoints; empty, corrupt, unsupported, oversized, too long, too short videos; no faces |
| Thresholds and confidence supported by evaluation | **Not met** | Machinery done and tested; no validation data run, so no thresholds exist and the app shows "Not validated" |
| Test suite passes, limits documented | Met for the synthetic suite | `python -m pytest -q`: all tests pass. Not covered: real weights, real face videos, GPU, Windows, Python 3.10, the Streamlit upload and analyse click |
| App runs with documented commands | Partly verified | App starts and renders in Streamlit's test runner (missing/invalid/valid checkpoint, with/without policy, all result states). A real browser upload was not exercised |
| README honest about capabilities and limits | Met | No accuracy claims; scope, limits and failure cases listed |
| Updated code in a separate branch | Met | `feature/deepfake-detection-v2` on GitHub |

## To finish (needs the author's files)

1. Place the trained checkpoint at `output/deepfake_cnn_lstm.pth`.
2. Build a manifest of labelled videos with source groups and separate `val` and `test` splits.
3. `python -m src.evaluate validate / score / fit / report` (see README).
4. Publish the measured report and keep claims within it.

# Deepfake Video Analysis (CNN + LSTM)

A video-level analysis tool for **face-manipulation deepfakes**. It samples clips across a video,
crops the face, scores each clip with a trained EfficientNet-B0 + BiLSTM model, and reports a
three-way assessment: **likely real**, **likely fake / manipulated**, or **inconclusive**.

> **This is an AI assessment, not forensic proof.** A "likely real" result does not establish that a
> video is authentic. The model was trained on face-manipulation videos and is not expected to
> detect fully AI-generated video, other manipulation types, or generators it has not seen.

## Current status (read this first)

| Item | State |
|---|---|
| Code for loading, preprocessing, inference, decision rules, evaluation | Done; covered by automated tests on synthetic data (see `docs/ACCEPTANCE.md`) |
| The trained checkpoint | **None ships.** The original was deleted by the author and cannot be restored. A new model must be trained (see "Train a new model") |
| Accuracy / benchmark numbers | **None published.** No result has been reproduced with this code yet, so none is claimed |
| Decision thresholds | **Not validated** until you run the evaluation on labelled validation videos. Until then the app shows the raw score and "Not validated" |
| Tested on | Python 3.13, Linux, CPU only. Windows and Python 3.10 commands below were not run by the author of this update |

## Model

| Part | Detail |
|---|---|
| Frame encoder | torchvision EfficientNet-B0, final layer replaced by `Linear(1280, 256)` |
| Temporal model | 1-layer bidirectional LSTM, hidden size 128 |
| Head | `Linear(256, 2)` on the last LSTM output |
| Input | 16 face crops, 112x112, RGB, normalised with mean 0.5 / std 0.5 |
| Output | 2 logits; softmax gives `[P(real), P(fake)]` (class 0 = real, class 1 = fake) |
| Size | 4.73 M parameters |

The architecture is defined once in `src/model.py` and a test checks it stays identical to the class
in `src/train.py`, which produced the checkpoint.

### Checkpoint requirements

- Put the trained file at `output/deepfake_cnn_lstm.pth`, or set `DEEPFAKE_CHECKPOINT` to its path.
- If the checkpoint records its preprocessing (`image_size`, `frames_per_clip`, `stride`), inference refuses to run when the current settings differ.
- Accepted formats: the dict written by `train.py` (`model_state_dict`, ...) or a bare state dict.
- Loading is **strict**. A missing, corrupt, or mismatched file stops with an error. There is no
  fallback to random weights or another model.
- Weights are git-ignored (`*.pth`). To share them use Git LFS or a GitHub Release, not a normal commit.
- The loaded file's SHA-256 is shown in the app and stored in every evaluation and decision policy.

## Install (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install --no-deps -r requirements-face.txt     # face detector (facenet-pytorch)
pip install -r requirements-dev.txt                # only if you want to run the tests
```

`facenet-pytorch` is installed with `--no-deps` because it pins an old `torch`. On Linux/macOS use
`python3 -m venv .venv && source .venv/bin/activate` instead of the activation line.

## Run the app

```powershell
streamlit run app.py
```

Upload an MP4, AVI or MOV. The app shows the assessment, raw per-clip scores, frames analysed,
faces detected and face coverage, processing time, model file and any warnings. Streamlit's own
upload cap (200 MB by default) can be changed with `server.maxUploadSize`.

| Environment variable | Meaning | Default |
|---|---|---|
| `DEEPFAKE_CHECKPOINT` | Path to the trained model | `output/deepfake_cnn_lstm.pth` |
| `DEEPFAKE_POLICY` | Validated decision thresholds | `models/decision_policy.json` |
| `DEEPFAKE_DEVICE` | `auto`, `cpu` or `cuda` (explicit `cuda` without CUDA is an error) | `auto` |
| `DEEPFAKE_MAX_MB` | Maximum upload size | `200` |
| `DEEPFAKE_MAX_SECONDS` | Processing time budget per video | `180` |

Other settings (clips per video, stride, limits) are in `src/config.py`. Frames per clip, stride and
image size must match training and should not be changed for this checkpoint.

## Test a single video

```powershell
python -m src.inference path\to\video.mp4
python -m src.inference path\to\video.mp4 --json
```

This prints the raw score only. A label requires a validated policy (see below).

## How a video is analysed

1. The file is validated (type, size, length, decodable, enough frames).
2. Up to 5 non-overlapping 16-frame clips are taken across the video, using the training stride.
3. MTCNN finds a face in each frame, mirroring `train.py`. Frames without a face are filled with the
   nearest earlier detected face. Clips with fewer than 75% face frames are dropped.
4. Fewer than 2 usable clips means **no score** and an inconclusive result.
5. Each clip gets a raw score, P(fake). Mean, median, min and max across clips are reported.
6. With a validated policy the label comes from its thresholds; otherwise no label is given.

Raw scores are **not calibrated probabilities**. The app only shows a "Calibrated P(fake)" figure
when the policy contains a calibration fitted on validation data, and then names that data.

## Evaluate the model

Create a manifest CSV:

```csv
path,label,group,dataset,split
videos/real_001.mp4,real,id_001,CelebDF,val
videos/fake_001.mp4,fake,id_001,CelebDF,val
videos/real_207.mp4,real,id_207,CelebDF,test
```

- `group`: identity or source-video id. Everything sharing a group must be in one split. A fake and
  the real video it was made from share a group. Splits that share a group are rejected.
- `split`: `val` fits thresholds, `test` is reported only. Videos that are byte-identical across
  splits are rejected at scoring time.

```powershell
python -m src.evaluate validate manifest.csv
python -m src.evaluate score  manifest.csv --out results\scores_clean.csv
python -m src.evaluate fit    results\scores_clean.csv --out models\decision_policy.json --calibrate
python -m src.evaluate score  manifest.csv --out results\scores_jpeg30.csv --condition jpeg:30
python -m src.evaluate report results\scores_clean.csv results\scores_jpeg30.csv --policy models\decision_policy.json
```

- `fit` refuses (and writes nothing) if validation has fewer than 20 real or 20 fake videos, if the
  scores cannot meet the error targets with enough videos labelled, or if the policy would be
  based on degraded data. See `docs/DECISION_POLICY.md`.
- `report` covers the test split only, per dataset and per condition: accuracy, precision, recall,
  F1, ROC-AUC, PR-AUC, confusion matrix, false-positive and false-negative rates with 95% intervals,
  calibration (if fitted), and how many videos got a label. Videos with no score count as
  inconclusive, so poor coverage is visible.
- It refuses test videos that were used to fit the policy, and will not apply a policy fitted for a
  different checkpoint.
- Robustness conditions (`jpeg:Q`, `scale:F`, `blur:S`, `brightness:F`) degrade individual frames.
  They approximate, but do not replace, testing on videos re-encoded by real platforms.
- To compare two checkpoints, score the same manifest with each and pass both score files to
  `report`. Only threshold-free metrics are shown for a checkpoint without a matching policy.

### Datasets and licences

The datasets are not included and must not be redistributed. Obtain each from its official source,
read its current terms, and use it only as permitted (typically research or non-commercial use
after agreeing to a licence or request form). Datasets named by the original code or commonly used
for this task: Celeb-DF v2 (the paths in `train.py`), FaceForensics++, DFDC. For a meaningful test,
use data that was **not** used for training and, ideally, manipulation methods the model has not seen.

## Train a new model

The original checkpoint no longer exists, so a working model has to be trained again. This produces
a **new** model. Nothing here can restore the original or reproduce its results, and any new model
must be evaluated from scratch before any claim is made about it.

1. **Get data** from the official sources (see Datasets and licences). Build a manifest like the
   evaluation one, with `split` set to `train`, `val` or `test` and a `group` per source video or
   identity. Keep the `test` split for the final report only.
2. **Check it:** `python -m src.evaluate validate manifest.csv`
3. **Extract face clips once** (train and val only; the test split is never read):
   ```powershell
   python -m src.retrain prepare manifest.csv --cache cache_faces --clips-per-video 3
   ```
   Disk use is about 0.6 MB per clip at the default size, roughly 1.8 MB per video (5,000 videos is
   about 9 GB). The command can be re-run to resume.
4. **Train:**
   ```powershell
   python -m src.retrain train --cache cache_faces --out output_v2\deepfake_cnn_lstm_v2.pth --manifest manifest.csv
   ```
   A GPU is strongly recommended (free Colab or Kaggle GPUs work); CPU training speed was not
   measured. The first run downloads ImageNet weights for EfficientNet-B0, so it needs internet.
   `--init-from some.pth` fine-tunes an existing checkpoint instead. `--degrade-p 0.3` gives 30% of
   training clips a random JPEG, blur or downscale step (off by default; see caveats).
5. **Evaluate and use it:** score the `val` and `test` splits, fit thresholds and write the report
   (see "Evaluate the model"), then point the app at the new file with `DEEPFAKE_CHECKPOINT`.

What the trainer guarantees:

- Face crops and tensors come from the same code as inference, and a test checks the cached tensors
  are identical to what the inference pipeline feeds the model.
- It never overwrites a checkpoint, refuses manifests where a source group or identical video sits in
  two splits, and stops on a cache built with different preprocessing.
- The best epoch is chosen by validation **video-level ROC-AUC**, not by a fixed threshold. The
  checkpoint records its settings, data counts, manifest hash and code version.

Caveats:

- Training adds a horizontal flip and mild brightness jitter, which the original script did not.
  `--degrade-p` adds optional compression/blur/downscale on the cached 112 px face crops. It is meant
  to help with re-encoded video but only approximates real platform re-encoding, and its benefit has
  **not been measured**: compare a run with and without it using the `jpeg:`/`scale:`/`blur:`
  conditions of `src.evaluate` before keeping it.
- The validation split is used both to pick the epoch and to fit thresholds, so validation figures
  are optimistic. Only the held-out test report is evidence.
- The trainer was verified on a small synthetic task only (it learned, the checkpoint loaded through
  the strict loader, and the inference pipeline scored unseen fake videos higher than real ones).
  That shows the code path works, not that real deepfakes will be detected, and no accuracy is claimed.
- `src/train.py` is the original script, kept for reference and unmodified. Its validation split ignores
  source groups, and it saves `deepfake_cnn_lstm.pth` into its output directory, overwriting any file
  of that name. Prefer `src.retrain`. `src/dataset.py` and `src/extract_frames.py` are legacy files
  that nothing uses (`extract_frames.py` has hard-coded `D:\` paths).

## Resources

Measured on a 2-core Linux CPU: the model forward pass for the default 5 clips x 16 frames took
0.61 s, with about 930 MB peak memory for the Python process. Video decoding and face detection add
time that depends on resolution and length and was **not benchmarked with real face videos**.
GPU inference is supported but was not tested.

## Known failure cases and limits

- Only the face-manipulation setting is in scope. Fully AI-generated video, audio, and images are not.
- Needs a visible face. Small, occluded, or profile faces reduce usable clips and yield "inconclusive".
- When several faces appear, the first box returned by MTCNN is used (as in training), which is not
  necessarily the largest or the relevant person.
- The LSTM head reads the last time step of a bidirectional LSTM (kept for checkpoint compatibility).
- Trained on one dataset family; generalisation to unseen generators, heavy compression, low
  resolution, or unusual lighting has not been measured. Low-resolution input triggers a warning.
- Thresholds and calibration are only as good as the validation data they were fitted on.
- The 5-clip sample covers a small part of long videos.

## Project layout

```
app.py                  Streamlit app
src/model.py            model definition (matches train.py)
src/checkpoint.py       strict checkpoint loading
src/config.py           settings and training-matched constants
src/video.py            validation, clip planning, face crops, tensors
src/pipeline.py         clip scoring and aggregation
src/decision.py         validated decision policy and rules
src/ui.py               result rendering for the app
src/inference.py        command line scoring of one video
src/metrics.py          metrics, calibration
src/policy_fit.py       threshold fitting
src/evaluate.py         validate / score / fit / report
src/retrain.py          prepare face-clip cache and train a new model
src/train.py            original training script (unchanged, kept for reference)
tests/                  automated tests
docs/                   AUDIT.md, DECISION_POLICY.md, ACCEPTANCE.md
```

Run the tests with `python -m pytest -q`. The suite uses synthetic checkpoints and videos, so it
checks the code's behaviour, not the model's accuracy.

## Roadmap

1. Train a new model on licensed data with `src.retrain`, evaluate it on a held-out test split, and publish the measured results.
2. Evaluate on additional datasets and re-encoded (social-media style) videos.
3. Better face detection/alignment and multi-face handling.
4. Fix the temporal readout and test whether retraining improves held-out results.
5. Compare any new architecture or ensemble against the current checkpoint under the same protocol
   and keep it only if the measured gain justifies the complexity.

## Acknowledgements

Thanks to the research community that provides deepfake benchmark datasets. This project is for
research and education.

# Stage A Audit Report

Repository: `Poojasri06/Deepfake-video-detection`, `main` @ `8f4307d` (12 commits, full history inspected).
Method: static code reading only. Nothing below was executed, and no metric here is measured.

## 1. Checkpoint status: NOT FOUND

- No `.pth`, `.pt`, `.ckpt` or LFS pointer exists in the working tree or in any of the 12 commits.
- No `output/` directory, `training_metrics.csv`, or config file is committed.
- `predictions.csv` (2 bytes) and `inference_results.csv` (0 bytes) contain no predictions.
- Consequence: the trained model cannot be loaded or evaluated yet. Every accuracy claim must wait for the real file.
- The checkpoint is probably on the original Windows machine (training paths are `D:\projects\deepfake_detection\...`) as `output/deepfake_cnn_lstm.pth` and/or `output/model_video.pth`.

## 2. The actual trained architecture (from `src/train.py`)

`train.py` is the only code that saves a checkpoint, so it defines the real model.

| Item | Value |
|---|---|
| Frame encoder | torchvision EfficientNet-B0 (ImageNet init), classifier[1] replaced with `Linear(1280, 256)` |
| Temporal model | 1-layer **bidirectional** LSTM, hidden 128 |
| Head | `Linear(256, 2)` on `lstm_out[:, -1, :]` |
| Input | `(B, 16, 3, 112, 112)`, RGB, MTCNN face crop |
| Normalisation | `ToTensor` then mean 0.5, std 0.5 (not ImageNet stats) |
| Sampling | one 16-frame clip, stride 2, random start in training, centre start in validation |
| Output | 2 logits, softmax, **class 0 = REAL, class 1 = FAKE** |
| Loss | class-weighted CrossEntropy (video-level counts) |
| Checkpoint format | dict: `model_state_dict`, `optimizer_state`, `config`, `epoch`, `best_val_f1` |
| Default file | `output/deepfake_cnn_lstm.pth` |

`src/inference.py` and `src/model.py` replicate this architecture correctly. The saved `config` inside the checkpoint will confirm the exact hyperparameters once the file is available.

## 3. Critical findings

1. **`app.py` cannot run the trained model.** It defines a different network (2 conv layers + LSTM(64), 224 px, ImageNet normalisation, no face cropping) and a different checkpoint name (`model_video.pth`). By inspection its CNN output flattens to 128x56x56 = 401,408 features per frame, while the LSTM expects 128, so it should raise a shape error. Even if shapes matched, `load_state_dict(strict=False)` would silently skip every mismatched key and leave random weights.
2. **`strict=False` everywhere** (`app.py`, `inference.py`, `utils.py`, `evaluate.py`). A wrong or partial checkpoint loads "successfully" with random layers. This violates the no-silent-random-weights requirement.
3. **`src/evaluate.py` is broken.** It imports `make_dataloaders` and `DeepfakeModel`, which do not exist, reads a missing `configs/config.yaml`, expects a `model_state` key (training saves `model_state_dict`), and applies `sigmoid` to 2-class logits. Treat any past numbers from it as unverified.
4. **Inference does not match training sampling.** `inference.py` takes the first 16 consecutive frames that contain a face (stride 1, roughly the first half second) and returns nothing if fewer than 16 are found. Training used stride 2 across the video. `app.py` averages softmax over every sliding window of overlapping frames, which are highly correlated.
5. **Evaluation protocol is weak, so reported validation scores are likely optimistic.** Validation is a random 15% video split of Celeb-DF v2 only. Celeb-DF reuses identities and each fake derives from a real source video, so identity and source leakage is likely. The best checkpoint is also chosen on that same split, and there is no held-out test set.
6. **Severe class imbalance, no calibration.** Celeb-DF v2 is heavily fake-dominated. Class weighting helps training, but there is no threshold selection or calibration. `app.py` uses an arbitrary 70% cutoff, and the "Confidence" shown is raw softmax.
7. **Bidirectional LSTM readout.** `out[:, -1, :]` for a BiLSTM takes the backward direction after seeing only the last frame. This is a modelling weakness baked into the trained weights. It must be kept for compatibility, and a fixed readout would need retraining.

## 4. Other defects

- `requirements.txt` omits `facenet-pytorch`, `scikit-learn`, `tqdm`, `pyyaml`, `Pillow`; no version pins except `altair`.
- `Readme.md` and `readme.md` both exist. They collide on Windows and macOS (case-insensitive filesystems). The small one describes files that do not exist (`configs/config.yaml`, `streamlit_app.py`, `src/model.py` as used).
- `Readme.md` describes a "Conv2D, ReLU, MaxPool" CNN, which does not match the EfficientNet-B0 training code. It also claims detection of lip-sync mismatches and facial warping, which the model does not measure. It lists `inference.py` at the repo root; it is under `src/`.
- `src/__pycache__/*.pyc` is committed; there is no `.gitignore`.
- `src/extract_frames.py` has hardcoded `D:\` paths and is unused by training.
- `src/dataset.py` is a second dataset class that falls back to whole-frame crops when no face is found, whereas `train.py` skips those frames. Which one produced the checkpoint is unknown; `train.py` defines its own dataset, so it is the likely one.
- MTCNN `detect()` returns boxes by detection confidence, so `boxes[0]` may not be the largest face (the `select_largest` flag applies to the forward/extract path). To be verified.
- `inference.py` and `model.py` request `weights="IMAGENET1K_V1"`, forcing a weight download at inference time even though the checkpoint overwrites it.
- No decoding validation, size or duration limits, or temp-file safety beyond basic cleanup.

## 5. What is needed to proceed

1. The trained checkpoint file(s), or confirmation that they no longer exist.
2. Which file the reported results came from (`deepfake_cnn_lstm.pth` or `model_video.pth`).
3. Whether the Celeb-DF v2 videos are still available for validation/threshold selection (and any independent test set, e.g. FaceForensics++ or DFDC).

#!/usr/bin/env python3
"""
train.py

Training pipeline for CelebDF-v2 deepfake detection.

Features:
- MTCNN face extraction (facenet-pytorch)
- EfficientNet-B0 (pretrained) as frame-level encoder
- Bidirectional LSTM for temporal modeling
- Sliding-window sampling of frames per video (FRAMES_PER_CLIP, STRIDE)
- Class-weighted CrossEntropyLoss to address class imbalance
- Mixed precision training (torch.cuda.amp) when available
- Checkpoint saving with {"model_state_dict", "optimizer_state", "config", "epoch", "best_val_f1"}
- Metrics saved to CSV
- Robust handling of short / problematic videos
"""

import os
import math
import time
import csv
import random
import argparse
from pathlib import Path
from collections import Counter

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm
from facenet_pytorch import MTCNN
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np

# ----------------------------
# CONFIG (edit as needed)
# ----------------------------
CONFIG = {
    "real_dir": "data/celebDFv2_real",
    "fake_dir": "data/celebDFv2_fake",
    "output_dir": "output",
    "save_name": "deepfake_cnn_lstm.pth",
    "frames_per_clip": 16,
    "stride": 2,
    "image_size": 112,
    "batch_size": 4,
    "epochs": 12,
    "lr": 1e-4,
    "val_split": 0.15,
    "num_workers": 2,
    "pin_memory": True,
    "seed": 42,
    "use_amp": True,   # use mixed precision if CUDA available
    "max_videos_per_class": None,  # optional downsampling, None = use all
}

# ----------------------------
# Utilities
# ----------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_output_dirs(path):
    Path(path).mkdir(parents=True, exist_ok=True)

# ----------------------------
# Dataset
# ----------------------------
class VideoClipDataset(Dataset):
    """
    Creates dataset of video clips sampled from video file paths.
    Sampling is done on-the-fly (no pre-extraction), using sliding windows (stride) over frames.
    For each video we return one randomly chosen clip (start index chosen randomly) during training.
    For validation we return deterministic clips by selecting equidistant start positions.
    """

    def __init__(self, video_paths, labels, mtcnn, frames_per_clip=16, stride=2,
                 image_size=112, transform=None, mode="train"):
        """
        video_paths: list of paths
        labels: list of ints (0=real,1=fake)
        mtcnn: MTCNN instance to detect faces
        mode: "train" or "val"
        """
        assert len(video_paths) == len(labels)
        self.video_paths = video_paths
        self.labels = labels
        self.mtcnn = mtcnn
        self.frames_per_clip = frames_per_clip
        self.stride = stride
        self.image_size = image_size
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.video_paths)

    def safe_read_frame(self, cap, idx):
        # read a single frame at position idx
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def extract_face(self, frame):
        # return cropped/resized face (numpy HWC) or None
        try:
            boxes, _ = self.mtcnn.detect(frame)
            if boxes is None or len(boxes) == 0:
                return None
            x1, y1, x2, y2 = boxes[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            h, w, _ = frame.shape
            # clip
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w-1, x2), min(h-1, y2)
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                return None
            face = cv2.resize(face, (self.image_size, self.image_size))
            return face
        except Exception:
            return None

    def sample_clip_indices(self, total_frames):
        """
        For given video length (total_frames) compute valid starting indices for sampled clips.
        A clip of length F uses frames at start, start+stride, start+2*stride, ... (frames_per_clip)
        So the last valid start is when start + (frames_per_clip-1)*stride < total_frames
        Return list of valid starts.
        """
        max_start = total_frames - (self.frames_per_clip - 1) * self.stride
        if max_start <= 0:
            return []
        return list(range(0, max_start))

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = int(self.labels[idx])

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            # return a zero tensor and label (should be skipped ideally)
            dummy = torch.zeros(self.frames_per_clip, 3, self.image_size, self.image_size)
            return dummy, torch.tensor(label, dtype=torch.long)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        starts = self.sample_clip_indices(total_frames)

        # if no valid start, we will sample frames uniformly across video (fallback)
        if len(starts) == 0:
            # fallback sampling: sample evenly spaced frames (if video too short)
            indices = np.linspace(0, max(total_frames-1,0), num=self.frames_per_clip, dtype=int).tolist()
        else:
            if self.mode == "train":
                # pick a random start for augmentation
                start = random.choice(starts)
            else:
                # deterministically pick center start (or first) for validation
                start = starts[len(starts)//2]
            indices = [start + i * self.stride for i in range(self.frames_per_clip)]

        faces = []
        for fi in indices:
            frame = self.safe_read_frame(cap, fi)
            if frame is None:
                continue
            face = self.extract_face(frame)
            if face is None:
                continue
            faces.append(face)

        cap.release()

        # If we didn't get enough faces, attempt a relaxed sampling: take whatever faces we found using denser frames
        if len(faces) < self.frames_per_clip:
            # Try scanning entire video with stride 1 until we get required frames (cheap rescue)
            cap2 = cv2.VideoCapture(video_path)
            fidx = 0
            while len(faces) < self.frames_per_clip:
                ret, frame = cap2.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face = self.extract_face(frame)
                if face is not None:
                    faces.append(face)
                fidx += 1
            cap2.release()

        # If still not enough, pad using last valid face or zeros
        if len(faces) == 0:
            # create blank frames
            blank = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            faces = [blank] * self.frames_per_clip
        while len(faces) < self.frames_per_clip:
            faces.append(faces[-1])

        # convert to tensors
        if self.transform is not None:
            frames_tensor = torch.stack([self.transform(ImageFromArray(f)) for f in faces])  # (T, C, H, W)
        else:
            # Minimal conversion: HWC->CHW and scale [0,255] -> float tensor
            frames_tensor = torch.stack([torch.from_numpy(f.transpose(2,0,1)).float().div_(255.0) for f in faces])

        # frames_tensor shape: (T, C, H, W) -> return (C, T, H, W)? We will keep (T, C, H, W)
        return frames_tensor, torch.tensor(label, dtype=torch.long)

# small helper to allow PIL-like transforms from numpy arrays
def ImageFromArray(arr):
    # arr is RGB HWC uint8
    from PIL import Image
    return Image.fromarray(arr)

# ----------------------------
# Model
# ----------------------------
class DeepfakeDetector(nn.Module):
    def __init__(self, cnn_out_dim=256, lstm_hidden=128, bidirectional=True):
        super().__init__()
        # EfficientNetB0 backbone
        cnn = models.efficientnet_b0(weights="IMAGENET1K_V1")
        # replace classifier to get desired feature size
        if isinstance(cnn.classifier, nn.Sequential):
            in_feats = cnn.classifier[1].in_features
            cnn.classifier[1] = nn.Linear(in_feats, cnn_out_dim)
        else:
            # fallback
            in_feats = cnn.classifier.in_features
            cnn.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_feats, cnn_out_dim))
        self.cnn = cnn

        self.lstm = nn.LSTM(input_size=cnn_out_dim,
                            hidden_size=lstm_hidden,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=bidirectional)
        self.fc = nn.Linear(lstm_hidden * (2 if bidirectional else 1), 2)

    def forward(self, x):
        """
        x: (B, T, C, H, W)
        Returns: logits (B, 2)
        """
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)          # (B*T, C, H, W)
        feats = self.cnn(x)              # (B*T, cnn_out_dim)
        feats = feats.view(B, T, -1)     # (B, T, cnn_out_dim)
        out, _ = self.lstm(feats)        # (B, T, hidden*direc)
        last = out[:, -1, :]
        logits = self.fc(last)
        return logits

# ----------------------------
# Training / Validation loop
# ----------------------------
def compute_class_weights(labels):
    # labels: list of ints per video
    counts = Counter(labels)
    total = sum(counts.values())
    classes = [0,1]
    # weight = total / (num_classes * count)
    weights = []
    for c in classes:
        weights.append(total / (len(classes) * (counts.get(c, 0) + 1e-6)))
    return torch.tensor(weights, dtype=torch.float32)

def collate_fn(batch):
    # batch: list of (frames_tensor(T,C,H,W), label)
    frames = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    # stack into (B, T, C, H, W)
    frames = torch.stack(frames)
    return frames, labels

def train_worker(config):
    seed_everything(config["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    make_output_dirs(config["output_dir"])

    # create mtcnn on CPU/GPU
    mtcnn = MTCNN(select_largest=True, device=device, post_process=False)

    # build video lists
    real_paths = sorted([os.path.join(config["real_dir"], f) for f in os.listdir(config["real_dir"]) if f.lower().endswith((".mp4", ".avi", ".mov"))])
    fake_paths = sorted([os.path.join(config["fake_dir"], f) for f in os.listdir(config["fake_dir"]) if f.lower().endswith((".mp4", ".avi", ".mov"))])

    # optional downsample classes to reduce huge imbalance (if desired)
    if config["max_videos_per_class"] is not None:
        maxn = config["max_videos_per_class"]
        real_paths = real_paths[:maxn]
        fake_paths = fake_paths[:maxn]

    video_paths = real_paths + fake_paths
    labels = [0]*len(real_paths) + [1]*len(fake_paths)

    # shuffle and split by video
    combined = list(zip(video_paths, labels))
    random.shuffle(combined)
    video_paths, labels = zip(*combined)
    video_paths = list(video_paths)
    labels = list(labels)

    split_idx = int(len(video_paths) * (1 - config["val_split"]))
    train_paths, val_paths = video_paths[:split_idx], video_paths[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]

    print(f"Total videos: {len(video_paths)}  Train: {len(train_paths)}  Val: {len(val_paths)}")
    print("Real:", len(real_paths), "Fake:", len(fake_paths))

    # transforms (PIL-based)
    transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])

    # datasets
    train_ds = VideoClipDataset(train_paths, train_labels, mtcnn=mtcnn,
                                frames_per_clip=config["frames_per_clip"],
                                stride=config["stride"],
                                image_size=config["image_size"],
                                transform=transform,
                                mode="train")
    val_ds = VideoClipDataset(val_paths, val_labels, mtcnn=mtcnn,
                              frames_per_clip=config["frames_per_clip"],
                              stride=config["stride"],
                              image_size=config["image_size"],
                              transform=transform,
                              mode="val")

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True,
                              num_workers=config["num_workers"], pin_memory=config["pin_memory"],
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False,
                            num_workers=config["num_workers"], pin_memory=config["pin_memory"],
                            collate_fn=collate_fn)

    # class weights (by video counts)
    class_weights = compute_class_weights(train_labels).to(device)
    print("Class weights (video-level):", class_weights.cpu().numpy())

    # model
    model = DeepfakeDetector().to(device)

    # optimizer / loss / scheduler
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    scaler = torch.cuda.amp.GradScaler(enabled=(config["use_amp"] and torch.cuda.is_available()))

    best_val_f1 = 0.0
    metrics_csv = os.path.join(config["output_dir"], "training_metrics.csv")

    # write CSV header
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "train_f1", "val_loss", "val_acc", "val_f1"])

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['epochs']} [train]")

        for frames, labels in pbar:
            frames = frames.to(device)            # (B, T, C, H, W)
            labels = labels.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(config["use_amp"] and torch.cuda.is_available())):
                outputs = model(frames)           # (B, 2)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

            avg_loss = running_loss / (len(all_preds) / max(1, config["batch_size"]))
            train_f1 = f1_score(all_labels, all_preds, average='macro') if len(all_labels) > 0 else 0.0
            train_acc = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0

            pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{train_acc:.3f}", f1=f"{train_f1:.3f}")

        # end epoch train metrics
        train_loss = running_loss / max(1, len(train_loader))
        train_acc = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0
        train_f1 = f1_score(all_labels, all_preds, average='macro') if len(all_labels) > 0 else 0.0

        # validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_true = []
        with torch.no_grad():
            for frames, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{config['epochs']} [val]"):
                frames = frames.to(device)
                labels = labels.to(device)
                with torch.cuda.amp.autocast(enabled=(config["use_amp"] and torch.cuda.is_available())):
                    outputs = model(frames)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_preds.extend(preds.tolist())
                val_true.extend(labels.cpu().numpy().tolist())

        val_loss = val_loss / max(1, len(val_loader))
        val_acc = accuracy_score(val_true, val_preds) if len(val_true) > 0 else 0.0
        val_f1 = f1_score(val_true, val_preds, average='macro') if len(val_true) > 0 else 0.0

        print(f"Epoch {epoch} SUMMARY -> Train loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | Val loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")

        # write metrics
        with open(metrics_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1])

        # checkpoint if improved
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            ckpt = {
                "model_state_dict": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": config,
                "epoch": epoch,
                "best_val_f1": best_val_f1
            }
            torch.save(ckpt, os.path.join(config["output_dir"], config["save_name"]))
            print(f"Saved improved checkpoint (val_f1={best_val_f1:.4f})")

    print("Training complete. Best val F1:", best_val_f1)
    print("Metrics saved to", metrics_csv)

# ----------------------------
# Argparse (optional)
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Deepfake Detector")
    parser.add_argument("--real_dir", type=str, default=CONFIG["real_dir"])
    parser.add_argument("--fake_dir", type=str, default=CONFIG["fake_dir"])
    parser.add_argument("--output_dir", type=str, default=CONFIG["output_dir"])
    parser.add_argument("--frames_per_clip", type=int, default=CONFIG["frames_per_clip"])
    parser.add_argument("--stride", type=int, default=CONFIG["stride"])
    parser.add_argument("--image_size", type=int, default=CONFIG["image_size"])
    parser.add_argument("--batch_size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--lr", type=float, default=CONFIG["lr"])
    parser.add_argument("--val_split", type=float, default=CONFIG["val_split"])
    parser.add_argument("--use_amp", type=int, default=1)
    parser.add_argument("--max_videos_per_class", type=int, default=-1, help="optional downsample videos per class")
    args = parser.parse_args()

    # update config from args
    CONFIG["real_dir"] = args.real_dir
    CONFIG["fake_dir"] = args.fake_dir
    CONFIG["output_dir"] = args.output_dir
    CONFIG["frames_per_clip"] = args.frames_per_clip
    CONFIG["stride"] = args.stride
    CONFIG["image_size"] = args.image_size
    CONFIG["batch_size"] = args.batch_size
    CONFIG["epochs"] = args.epochs
    CONFIG["lr"] = args.lr
    CONFIG["val_split"] = args.val_split
    CONFIG["use_amp"] = bool(args.use_amp)
    CONFIG["max_videos_per_class"] = None if args.max_videos_per_class <= 0 else args.max_videos_per_class

    # ensure paths exist
    make_output_dirs(CONFIG["output_dir"])
    if not os.path.exists(CONFIG["real_dir"]) or not os.path.exists(CONFIG["fake_dir"]):
        raise FileNotFoundError("Please ensure real_dir and fake_dir exist and contain video files.")

    train_worker(CONFIG)

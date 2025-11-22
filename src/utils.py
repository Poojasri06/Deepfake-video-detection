# src/utils.py
import os
import math
import torch
import numpy as np
import cv2
from torchvision import transforms
from facenet_pytorch import MTCNN
from PIL import Image

# Default mirrors training config
IMAGE_SIZE = 112
FRAMES_PER_CLIP = 16
STRIDE = 2

def ensure_model_path(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")

def load_model_checkpoint(model, checkpoint_path, device='cpu'):
    ck = torch.load(checkpoint_path, map_location=device)
    if isinstance(ck, dict) and "model_state_dict" in ck:
        state = ck["model_state_dict"]
    else:
        state = ck
    missing, unexpected = model.load_state_dict(state, strict=False)
    return missing, unexpected

def estimate_sampled_counts(video_path, sample_rate=STRIDE, seq_len=FRAMES_PER_CLIP):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 0
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        sampled = math.ceil(total_frames / sample_rate) if total_frames > 0 else 0
        windows = max(sampled - seq_len + 1, 0)
        return sampled, windows
    finally:
        cap.release()

def stream_sampled_frames(video_path, sample_rate=STRIDE, image_size=IMAGE_SIZE):
    """
    Generator yielding (tensor_frame, sampled_index).
    Yields numpy HWC uint8 frames resized to image_size if desired.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")
    frame_idx = 0
    sample_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_rate == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # do not run MTCNN here; app or dataset handles cropping.
                pil = Image.fromarray(rgb)
                yield pil, sample_idx
                sample_idx += 1
            frame_idx += 1
    finally:
        cap.release()

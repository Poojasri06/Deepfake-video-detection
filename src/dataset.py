# src/dataset.py
import os
import random
import cv2
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class VideoClipDataset(Dataset):
    """
    Dataset that returns one clip (T frames) per video on-the-fly.
    Uses MTCNN to extract the face region; falls back to whole frame if detection fails.
    """
    def __init__(self, video_paths, labels, frames_per_clip=16, stride=2,
                 image_size=112, mtcnn_device='cpu', mode='train', transform=None):
        self.video_paths = list(video_paths)
        self.labels = list(labels)
        self.frames_per_clip = frames_per_clip
        self.stride = stride
        self.image_size = image_size
        self.mode = mode
        self.transform = transform
        self.mtcnn = MTCNN(select_largest=True, device=mtcnn_device, post_process=False)

    def __len__(self):
        return len(self.video_paths)

    def safe_read_frame(self, cap, idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def face_crop(self, frame):
        try:
            boxes, _ = self.mtcnn.detect(frame)
            if boxes is None or len(boxes) == 0:
                return None
            x1, y1, x2, y2 = boxes[0].astype(int)
            h, w, _ = frame.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                return None
            face = cv2.resize(face, (self.image_size, self.image_size))
            return face
        except Exception:
            return None

    def sample_indices(self, total_frames):
        # valid start indices for clip sampling given stride and frames_per_clip
        max_start = total_frames - (self.frames_per_clip - 1) * self.stride
        if max_start <= 0:
            return []
        return list(range(0, max_start))

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = int(self.labels[idx])

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            # return zero clip
            blank = np.zeros((self.frames_per_clip, 3, self.image_size, self.image_size), dtype=np.float32)
            return torch.from_numpy(blank), torch.tensor(label, dtype=torch.long)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        starts = self.sample_indices(total_frames)

        # Choose start
        if len(starts) == 0:
            # fallback: uniformly spaced indices across video
            indices = np.linspace(0, max(total_frames-1, 0), num=self.frames_per_clip, dtype=int).tolist()
        else:
            if self.mode == 'train':
                start = random.choice(starts)
            else:
                start = starts[len(starts) // 2]
            indices = [start + i * self.stride for i in range(self.frames_per_clip)]

        faces = []
        for fi in indices:
            frame = self.safe_read_frame(cap, fi)
            if frame is None:
                continue
            face = self.face_crop(frame)
            if face is None:
                # fall back to resized full frame if face missing
                face = cv2.resize(frame, (self.image_size, self.image_size))
            faces.append(face)

        cap.release()

        # try a rescue pass: scan video for faces if not enough faces collected
        if len(faces) < self.frames_per_clip:
            cap2 = cv2.VideoCapture(video_path)
            while len(faces) < self.frames_per_clip:
                ret, frame = cap2.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face = self.face_crop(frame)
                if face is not None:
                    faces.append(face)
            cap2.release()

        if len(faces) == 0:
            # all failed -> zeros
            faces = [np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
                     for _ in range(self.frames_per_clip)]

        # pad last frame if needed
        while len(faces) < self.frames_per_clip:
            faces.append(faces[-1])

        # apply transform (expects PIL.Image)
        if self.transform is None:
            transform_local = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3)
            ])
        else:
            transform_local = self.transform

        frames_tensor = torch.stack([transform_local(Image.fromarray(f)) for f in faces])  # (T, C, H, W)
        return frames_tensor, torch.tensor(label, dtype=torch.long)

import argparse
import sys
import os
from pathlib import Path
import cv2
import torch
import numpy as np
from torchvision import transforms, models
from facenet_pytorch import MTCNN


# ---------------------------
# Inference script matching training model
# ---------------------------

DEFAULT_CKPT = str(Path(__file__).resolve().parents[1] / "output" / "deepfake_cnn_lstm.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeepfakeDetector(torch.nn.Module):
    def __init__(self, cnn_out_dim=256, lstm_hidden=128, bidirectional=True):
        super().__init__()
        cnn = models.efficientnet_b0(weights="IMAGENET1K_V1")
        if isinstance(cnn.classifier, torch.nn.Sequential):
            in_feats = cnn.classifier[1].in_features
            cnn.classifier[1] = torch.nn.Linear(in_feats, cnn_out_dim)
        else:
            in_feats = cnn.classifier.in_features
            cnn.classifier = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(in_feats, cnn_out_dim))
        self.cnn = cnn

        self.lstm = torch.nn.LSTM(input_size=cnn_out_dim,
                                  hidden_size=lstm_hidden,
                                  num_layers=1,
                                  batch_first=True,
                                  bidirectional=bidirectional)
        self.fc = torch.nn.Linear(lstm_hidden * (2 if bidirectional else 1), 2)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.cnn(x)
        feats = feats.view(B, T, -1)
        out, _ = self.lstm(feats)
        last = out[:, -1, :]
        logits = self.fc(last)
        return logits


def load_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    model.load_state_dict(state, strict=False)
    return model


def extract_face_sequence(video_path, mtcnn, frames_per_clip=16, image_size=112):
    """Scan video and collect `frames_per_clip` face crops. Returns list of PIL-like images (RGB)."""
    from PIL import Image

    cap = cv2.VideoCapture(video_path)
    faces = []
    if not cap.isOpened():
        cap.release()
        return []

    # First pass: scan and collect faces until we reach required number
    while len(faces) < frames_per_clip:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            boxes, _ = mtcnn.detect(rgb)
        except Exception:
            boxes = None
        if boxes is None or len(boxes) == 0:
            continue
        x1, y1, x2, y2 = boxes[0]
        x1, y1, x2, y2 = map(int, (max(0, x1), max(0, y1), x2, y2))
        h, w, _ = rgb.shape
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        crop = rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        pil = Image.fromarray(cv2.resize(crop, (image_size, image_size)))
        faces.append(pil)

    cap.release()

    # If not enough faces found, return empty to signal failure
    if len(faces) < frames_per_clip:
        return []
    return faces


def predict_video(video_path, ckpt_path=None, frames_per_clip=16, image_size=112):
    if ckpt_path is None:
        ckpt_path = DEFAULT_CKPT

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    mtcnn = MTCNN(select_largest=True, device=device, post_process=False)

    model = DeepfakeDetector().to(device)
    model = load_checkpoint(model, ckpt_path)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    faces = extract_face_sequence(video_path, mtcnn, frames_per_clip=frames_per_clip, image_size=image_size)
    if not faces:
        print("No faces found or video too short to extract a full clip.")
        return None

    # Build tensor (1, T, C, H, W)
    frames_tensor = torch.stack([transform(img) for img in faces])
    frames_tensor = frames_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(frames_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    return {"real": float(probs[0]), "fake": float(probs[1])}


def main():
    parser = argparse.ArgumentParser(description="Run inference on a video using trained DeepfakeDetector")
    parser.add_argument("video", type=str, help="Path to video file to analyze")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("--frames_per_clip", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=112)
    # If no arguments were provided (e.g. run from an IDE without args), print help instead of raising SystemExit
    if len(sys.argv) == 1:
        print("No arguments provided.\n")
        parser.print_help()
        return

    try:
        args = parser.parse_args()
    except SystemExit:
        # argparse throws SystemExit on parse errors; present help and return gracefully
        print("\nArgument parsing error. See usage below:\n")
        parser.print_help()
        return

    res = predict_video(args.video, ckpt_path=args.checkpoint, frames_per_clip=args.frames_per_clip, image_size=args.image_size)
    if res is None:
        print("Prediction failed.")
        return

    print("\n==============================")
    print(f"🎥 Video: {args.video}")
    print(f"✅ Real: {res['real']*100:.2f}%")
    print(f"⚠️ Fake: {res['fake']*100:.2f}%")
    if res['fake'] > res['real']:
        print("🔍 Prediction: FAKE")
    else:
        print("🔍 Prediction: REAL")
    print("==============================")


if __name__ == "__main__":
    main()

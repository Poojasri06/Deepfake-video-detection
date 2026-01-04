import os
import cv2
import torch
import torch.nn as nn
import streamlit as st
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from PIL import Image
import tempfile
import time
from facenet_pytorch import MTCNN

import traceback

def safe_predict(video_path, ckpt):
    try:
        return predict_video(video_path, ckpt)
    except Exception as e:
        st.error(f"An error occurred during analysis: {e}")
        st.text(traceback.format_exc())
        return None, None


# Set page config
st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# CONFIG
# ==============================
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "output", "deepfake_cnn_lstm.pth")
SEQUENCE_LENGTH = 16
IMAGE_SIZE = 112
BATCH_SIZE = 1

# Verify model file exists
if not os.path.exists(CHECKPOINT_PATH):
    st.error(f"""
    ❌ Model file not found at: {CHECKPOINT_PATH}
    
    To use this app, you need to download or train the deepfake detection model.
    
    **Option 1: Train the model yourself**
    ```bash
    # Create the output directory
    mkdir -p output
    
    # Prepare your dataset in the following structure:
    # data/celebDFv2_real/ (real videos)
    # data/celebDFv2_fake/ (fake videos)
    
    # Train the model
    python src/train.py
    ```
    
    **Option 2: Use a pre-trained model**
    1. Obtain a pre-trained model file (deepfake_cnn_lstm.pth)
    2. Create the output directory: `mkdir -p output`
    3. Place the model file in the 'output' folder
    4. Restart the app
    
    The model uses EfficientNet-B0 + BiLSTM architecture and expects videos with detectable faces.
    """)
    st.stop()

# ==============================
# DEVICE
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# MODEL
# ==============================
class DeepfakeDetector(nn.Module):
    """
    EfficientNet-B0 frame encoder + BiLSTM temporal model + classification head.
    Input shape for forward: (B, T, C, H, W)
    Output: logits (B, 2)
    """
    def __init__(self, cnn_out_dim=256, lstm_hidden=128, bidirectional=True):
        super().__init__()
        backbone = models.efficientnet_b0(weights="IMAGENET1K_V1")
        # adapt classifier to produce features
        if isinstance(backbone.classifier, nn.Sequential):
            in_feats = backbone.classifier[1].in_features
            backbone.classifier[1] = nn.Linear(in_feats, cnn_out_dim)
        else:
            in_feats = backbone.classifier.in_features
            backbone.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_feats, cnn_out_dim))

        self.cnn = backbone
        self.lstm = nn.LSTM(input_size=cnn_out_dim,
                            hidden_size=lstm_hidden,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=bidirectional)
        self.fc = nn.Linear(lstm_hidden * (2 if bidirectional else 1), 2)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.cnn(x)                # (B*T, cnn_out_dim)
        feats = feats.view(B, T, -1)       # (B, T, feat)
        out, _ = self.lstm(feats)          # (B, T, hid*dirs)
        last = out[:, -1, :]
        logits = self.fc(last)
        return logits

# ==============================
# DATASET
# ==============================
class VideoDataset(Dataset):
    def __init__(self, video_path, sequence_length=SEQUENCE_LENGTH, transform=None, device=None):
        self.video_path = video_path
        self.sequence_length = sequence_length
        self.transform = transform
        self.device = device if device is not None else torch.device("cpu")
        self.mtcnn = MTCNN(select_largest=True, device=self.device, post_process=False)
        self.frames = self._load_video_with_faces()
        if len(self.frames) < sequence_length:
            raise ValueError(f"Video too short or insufficient faces detected! Frames: {len(self.frames)} < {sequence_length}")

    def _load_video_with_faces(self):
        cap = cv2.VideoCapture(self.video_path)
        frames = []
        frame_count = 0
        max_frames_to_check = 500  # Limit to avoid processing very long videos
        frame_skip_interval = 2  # Process every other frame for efficiency
        
        while len(frames) < self.sequence_length * 2 and frame_count < max_frames_to_check:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            
            # Skip frames for efficiency - process every other frame once we have at least one
            if frame_count % frame_skip_interval != 0 and len(frames) > 0:
                continue
                
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Try to detect face
            try:
                boxes, _ = self.mtcnn.detect(rgb)
                if boxes is not None and len(boxes) > 0:
                    x1, y1, x2, y2 = boxes[0]
                    x1, y1, x2, y2 = map(int, (max(0, x1), max(0, y1), x2, y2))
                    h, w, _ = rgb.shape
                    x2, y2 = min(w - 1, x2), min(h - 1, y2)
                    crop = rgb[y1:y2, x1:x2]
                    if crop.size > 0:
                        frames.append(cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE)))
            except Exception as e:
                # If face detection fails, use the whole frame (fallback)
                # This handles MTCNN errors gracefully
                frames.append(cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE)))
                
        cap.release()
        return frames

    def __len__(self):
        return max(len(self.frames) - self.sequence_length + 1, 1)

    def __getitem__(self, idx):
        seq = self.frames[idx:idx+self.sequence_length]
        # Pad if necessary
        while len(seq) < self.sequence_length:
            seq.append(seq[-1] if seq else np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8))
        if self.transform:
            seq = [self.transform(Image.fromarray(f)) for f in seq]
        return torch.stack(seq)

# ==============================
# TRANSFORM
# ==============================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ==============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_model():
    try:
        model = DeepfakeDetector().to(device)
        model.eval()
        
        if not os.path.exists(CHECKPOINT_PATH):
            return None
            
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        else:
            state = checkpoint
            
        model.load_state_dict(state, strict=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def predict_video(video_path, ckpt=None):
    try:
        dataset = VideoDataset(video_path, transform=transform, device=device)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        model = load_model()
        
        if model is None:
            return None
            
        softmax = nn.Softmax(dim=1)
        all_probs = []
        
        for seq in loader:
            seq = seq.to(device)
            with torch.no_grad():
                outputs = model(seq)
                probs = softmax(outputs).cpu().numpy()
                all_probs.append(probs)
                
        if not all_probs:
            return None
            
        # Average probabilities across all clips
        avg_probs = np.mean(np.vstack(all_probs), axis=0)
        # Return dict with real and fake probabilities
        return {"real": float(avg_probs[0][0]), "fake": float(avg_probs[0][1])}
        
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return None

# ==============================
# STREAMLIT UI
# ==============================
def main():
    st.title("🎭 Deepfake Detection App")
    st.markdown("Upload a video to detect if it contains deepfake content")
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        "This app uses a deep learning model to detect deepfake videos. "
        "Upload a video file to analyze its authenticity."
    )
    
    # Upload section
    st.header("Upload Video")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        # Create a temporary file with a proper extension
        file_ext = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tfile:
            tfile.write(uploaded_file.getbuffer())
            temp_path = tfile.name
        
        try:
            # Display video preview
            st.subheader("Preview")
            st.video(temp_path)
            
            # Process and display results
            if st.button("Analyze Video"):
                with st.spinner('Analyzing video... This may take a moment...'):
                    # Store the temp path in session state
                    st.session_state['temp_video_path'] = temp_path
                    result = predict_video(temp_path)
                    
                    if result is not None:
                        prob_real = result["real"] * 100
                        prob_fake = result["fake"] * 100
                        
                        st.subheader("Analysis Results")
                        
                        # Display metrics
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Real", f"{prob_real:.1f}%")
                            st.progress(prob_real / 100)
                        
                        with col2:
                            st.metric("Fake", f"{prob_fake:.1f}%")
                            st.progress(prob_fake / 100)
                        
                        # Display result message
                        if prob_fake > 70:
                            st.error("⚠️ This video is likely a deepfake!")
                        elif prob_real > 70:
                            st.success("✅ This video appears to be real!")
                        else:
                            st.warning("🤔 The result is inconclusive.")
                        
                        # Show confidence level
                        confidence = max(prob_real, prob_fake)
                        st.info(f"Confidence: {confidence:.1f}%")
        finally:
            # Clean up the temporary file
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception as e:
                st.error(f"Error cleaning up temporary file: {e}")
    
    # Add some spacing
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("""
    1. Upload a video file (MP4, AVI, or MOV)
    2. Click 'Analyze Video' to process the content
    3. View the analysis results showing the likelihood of the video being real or fake
    
    The model analyzes the video frames using a combination of CNN and LSTM networks to detect potential deepfake artifacts.
    """)

if __name__ == "__main__":
    main()

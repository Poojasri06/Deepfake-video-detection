import os
import cv2
import torch
import torch.nn as nn
import streamlit as st
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from PIL import Image
import tempfile
import time

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
CHECKPOINT_PATH = os.path.join(BASE_DIR, "output", "model_video.pth")
SEQUENCE_LENGTH = 16
IMAGE_SIZE = 224
BATCH_SIZE = 1

# Verify model file exists
if not os.path.exists(CHECKPOINT_PATH):
    st.error(f"""
    ❌ Model file not found at: {CHECKPOINT_PATH}
    
    To use this app, you need to download the pre-trained model file.
    
    1. Download the model file from: [Google Drive](https://drive.google.com/.../model_video.pth)
    2. Place it in the 'output' folder
    3. Restart the app
    
    If you don't have the model file, you'll need to train the model first.
    """)
    st.stop()

# ==============================
# DEVICE
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# MODEL
# ==============================
class ResNetLSTM(nn.Module):
    def __init__(self, hidden_dim=64, lstm_layers=1, num_classes=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.flatten = nn.Flatten()
        self.lstm = nn.LSTM(128, hidden_dim, lstm_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        b, seq_len, c, h, w = x.size()
        x = x.view(b*seq_len, c, h, w)
        features = self.cnn(x)
        features = features.view(b, seq_len, -1)
        lstm_out, _ = self.lstm(features)
        return self.fc(lstm_out[:, -1, :])

# ==============================
# DATASET
# ==============================
class VideoDataset(Dataset):
    def __init__(self, video_path, sequence_length=SEQUENCE_LENGTH, transform=None):
        self.video_path = video_path
        self.sequence_length = sequence_length
        self.transform = transform
        self.frames = self._load_video()
        if len(self.frames) < sequence_length:
            raise ValueError(f"Video too short! Frames: {len(self.frames)} < {sequence_length}")

    def _load_video(self):
        cap = cv2.VideoCapture(self.video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    def __len__(self):
        return max(len(self.frames)-self.sequence_length+1, 0)

    def __getitem__(self, idx):
        seq = self.frames[idx:idx+self.sequence_length]
        if self.transform:
            seq = [self.transform(Image.fromarray(f)) for f in seq]
        return torch.stack(seq)

# ==============================
# TRANSFORM
# ==============================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_model():
    try:
        model = ResNetLSTM().to(device)
        model.eval()
        
        if not os.path.exists(CHECKPOINT_PATH):
            return None
            
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
            
        model.load_state_dict(checkpoint, strict=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# ==============================
# PREDICTION FUNCTION
# ==============================
def predict_video(video_path, ckpt=None):
    try:
        dataset = VideoDataset(video_path, transform=transform)
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
                all_probs.append(softmax(outputs).cpu().numpy())
                
        if not all_probs:
            return None
            
        avg_probs = np.mean(np.vstack(all_probs), axis=0)
        return avg_probs[0]  # Return probabilities for the first (and only) video
        
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
                    prediction = predict_video(temp_path)
                    
                    if prediction is not None:
                        prob_real = prediction[0] * 100
                        prob_fake = prediction[1] * 100
                        
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

# Deepfake Video Detection

An AI-powered deepfake detection system using CNN + LSTM architecture to identify manipulated videos with high precision.

## 🚀 Features

- **Hybrid Architecture**: EfficientNet-B0 (CNN) + Bidirectional LSTM for temporal analysis
- **Face Detection**: Automatic face detection and extraction using MTCNN
- **Web Interface**: Easy-to-use Streamlit web application
- **CLI Tool**: Command-line inference for batch processing
- **High Accuracy**: Trained on real and deepfake video datasets

## 📋 Requirements

- Python 3.8+
- PyTorch 1.13+
- CUDA (optional, for GPU acceleration)

## ⚙️ Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Poojasri06/Deepfake-video-detection.git
   cd Deepfake-video-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

### Web Application (Streamlit)

Launch the interactive web interface:

```bash
streamlit run app.py
```

Then open your browser and navigate to `http://localhost:8501`

**Note**: You need a trained model file at `output/deepfake_cnn_lstm.pth` to run the app. See training instructions below.

### Command Line Inference

Run inference on a single video:

```bash
python src/inference.py path/to/your/video.mp4
```

With custom checkpoint:

```bash
python src/inference.py path/to/your/video.mp4 --checkpoint output/deepfake_cnn_lstm.pth
```

## 🏋️ Training

### Prepare Dataset

1. Create the data directory structure:
   ```
   data/
   ├── celebDFv2_real/    # Real videos
   └── celebDFv2_fake/    # Fake/deepfake videos
   ```

2. Place your video files (MP4, AVI, or MOV format) in the respective folders

### Train the Model

```bash
python src/train.py
```

Optional training parameters:

```bash
python src/train.py \
  --real_dir data/celebDFv2_real \
  --fake_dir data/celebDFv2_fake \
  --output_dir output \
  --epochs 12 \
  --batch_size 4 \
  --lr 1e-4
```

The trained model will be saved to `output/deepfake_cnn_lstm.pth`

## 🧠 Model Architecture

1. **CNN Feature Extractor (EfficientNet-B0)**
   - Pretrained on ImageNet
   - Extracts spatial features from individual frames
   - Output: 256-dimensional feature vectors

2. **LSTM Temporal Encoder**
   - Bidirectional LSTM with 128 hidden units
   - Captures temporal patterns across 16 frames
   - Detects inconsistencies: lip-sync issues, frame artifacts, identity changes

3. **Classification Head**
   - Fully connected layer
   - Binary output: Real vs Fake
   - Softmax activation for probability scores

## 📁 Project Structure

```
deepfake_detection/
├── app.py                 # Streamlit web application
├── requirements.txt       # Python dependencies
├── src/
│   ├── model.py          # Model architecture definition
│   ├── train.py          # Training script
│   ├── inference.py      # CLI inference tool
│   ├── dataset.py        # Dataset loader
│   ├── utils.py          # Utility functions
│   └── evaluate.py       # Model evaluation
├── output/               # Trained models (created after training)
└── data/                 # Training/test datasets (not included)
```

## 🔍 How It Works

1. **Video Input**: Upload or specify a video file
2. **Face Detection**: MTCNN detects and crops faces from frames
3. **Frame Sampling**: Extracts 16 frames per sequence
4. **Feature Extraction**: EfficientNet-B0 processes each frame
5. **Temporal Analysis**: BiLSTM analyzes frame sequence
6. **Prediction**: Binary classification (Real/Fake) with confidence score

## 📊 Performance

The model is trained to detect:
- Face swap deepfakes
- GAN-generated videos
- Face reenactment
- Expression manipulation
- Audio-visual inconsistencies

## 🛠️ Tech Stack

- **PyTorch**: Deep learning framework
- **Torchvision**: Pre-trained models and transforms
- **Streamlit**: Web interface
- **OpenCV**: Video processing
- **MTCNN (facenet-pytorch)**: Face detection
- **NumPy**: Numerical operations
- **scikit-learn**: Metrics and evaluation

## 📝 Files

- `src/dataset.py`: Data loading and preprocessing
- `src/model.py`: Neural network architecture
- `src/train.py`: Training loop with validation
- `src/inference.py`: Video-level inference
- `app.py`: Web demo interface

## ⚠️ Important Notes

- The model requires videos with clearly visible faces
- Videos should be at least 16 frames long
- Training requires a balanced dataset of real and fake videos
- GPU is recommended for training but not required for inference
- Model files are not included due to size; you must train or obtain them separately

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is for educational and research purposes only. Please ensure you have proper rights to any videos you analyze.

## 🙏 Acknowledgements

This work uses publicly available deepfake detection research and datasets for educational purposes. Special thanks to the computer vision and AI research community.

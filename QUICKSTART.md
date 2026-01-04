# Quick Start Guide

This guide will help you get started with the Deepfake Detection app quickly.

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Poojasri06/Deepfake-video-detection.git
   cd Deepfake-video-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   This will install:
   - PyTorch (with CUDA support if available)
   - Streamlit (web interface)
   - facenet-pytorch (face detection)
   - OpenCV (video processing)
   - And other required packages

## Testing the Model Architecture

Before training or running the app, verify that the model architecture works correctly:

```bash
python test_model.py
```

This will:
- Test model instantiation
- Run a forward pass with dummy data
- Verify compatibility between different model files
- Show model parameters and architecture details

Expected output:
```
✅ All model tests passed!
🎉 All tests passed! The model is ready to use.
```

## Running the Application

### Option 1: Web Interface (Recommended)

**Note**: You need a trained model to run the app. See the Training section below.

```bash
streamlit run app.py
```

This will start the Streamlit server and open your browser automatically. If not, navigate to `http://localhost:8501`

### Option 2: Command Line Interface

Run inference on a video file:

```bash
python src/inference.py path/to/your/video.mp4
```

Example output:
```
==============================
🎥 Video: test_video.mp4
✅ Real: 85.23%
⚠️ Fake: 14.77%
🔍 Prediction: REAL
==============================
```

## Training Your Own Model

### Step 1: Prepare Your Dataset

1. Create the data directory structure:
   ```bash
   mkdir -p data/celebDFv2_real data/celebDFv2_fake
   ```

2. Add video files:
   - Place **real/authentic** videos in `data/celebDFv2_real/`
   - Place **deepfake/manipulated** videos in `data/celebDFv2_fake/`
   - Supported formats: MP4, AVI, MOV

3. Dataset requirements:
   - Videos should contain clear, visible faces
   - Minimum 16 frames per video
   - Balanced dataset (similar number of real and fake videos) recommended
   - Face-forward videos work best

### Step 2: Train the Model

```bash
python src/train.py
```

Training options:
```bash
python src/train.py \
  --real_dir data/celebDFv2_real \
  --fake_dir data/celebDFv2_fake \
  --epochs 12 \
  --batch_size 4 \
  --lr 0.0001 \
  --val_split 0.15
```

The model will be saved to `output/deepfake_cnn_lstm.pth`

### Step 3: Monitor Training

Training metrics are saved to `output/training_metrics.csv`. You can monitor:
- Training/validation loss
- Accuracy
- F1 score

### Step 4: Run the Application

Once training is complete, start the app:
```bash
streamlit run app.py
```

## Using a Pre-trained Model

If you have access to a pre-trained model:

1. Create the output directory:
   ```bash
   mkdir -p output
   ```

2. Place the model file:
   ```
   output/deepfake_cnn_lstm.pth
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Troubleshooting

### "Model file not found" Error

**Solution**: You need to train the model first or obtain a pre-trained model file.

```bash
# Create output directory
mkdir -p output

# Train the model (requires dataset)
python src/train.py

# Or place a pre-trained model in output/deepfake_cnn_lstm.pth
```

### "Video too short or insufficient faces detected"

**Solution**: The video needs:
- At least 16 frames with detectable faces
- Clear, frontal face visibility
- Good lighting conditions

Try with a different video that has a clear, visible face.

### CUDA Out of Memory

**Solution**: Reduce batch size or use CPU:

```bash
# For training, reduce batch size
python src/train.py --batch_size 2

# The app automatically uses CPU if CUDA is not available
```

### Import Errors

**Solution**: Reinstall dependencies:

```bash
pip install -r requirements.txt --force-reinstall
```

## Hardware Requirements

### Minimum (CPU Only)
- 4GB RAM
- 2 CPU cores
- Inference: ~30 seconds per video

### Recommended (GPU)
- 8GB RAM
- NVIDIA GPU with 4GB+ VRAM
- CUDA 11.0+
- Inference: ~5 seconds per video

### For Training
- 16GB+ RAM
- NVIDIA GPU with 8GB+ VRAM (highly recommended)
- 50GB+ disk space for dataset and models

## Example Workflow

1. **Verify installation**
   ```bash
   python test_model.py
   ```

2. **Prepare dataset**
   ```bash
   mkdir -p data/celebDFv2_real data/celebDFv2_fake
   # Add your videos...
   ```

3. **Train model**
   ```bash
   python src/train.py --epochs 12 --batch_size 4
   ```

4. **Test with CLI**
   ```bash
   python src/inference.py data/celebDFv2_real/sample_video.mp4
   ```

5. **Launch web app**
   ```bash
   streamlit run app.py
   ```

## Next Steps

- Read the full [README.md](readme.md) for detailed architecture information
- Check the training code in `src/train.py` to customize hyperparameters
- Explore the model architecture in `src/model.py`
- Try different videos and compare results

## Getting Help

If you encounter issues:
1. Check this Quick Start Guide
2. Review the main [README.md](readme.md)
3. Verify your Python and dependency versions
4. Run `python test_model.py` to diagnose model issues
5. Open an issue on GitHub with error details

## Tips for Best Results

- Use high-quality videos with good lighting
- Ensure faces are clearly visible and frontal
- Videos should be at least 1-2 seconds long
- For training, use a balanced dataset (50/50 real/fake)
- Larger datasets generally produce better results
- Consider data augmentation for small datasets

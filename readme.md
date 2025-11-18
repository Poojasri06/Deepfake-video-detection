# Deepfake Detector

## Setup
1. `pip install -r requirements.txt`
2. Set dataset paths in `configs/config.yaml`
3. `python src/train.py`

## Files
- `src/dataset.py` : dataloader
- `src/model.py` : backbone + classifier
- `src/train.py` : training loop
- `src/inference.py` : video-level inference
- `streamlit_app.py` : web demo

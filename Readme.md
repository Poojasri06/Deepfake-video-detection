An AI-powered deepfake detection pipeline leveraging Convolutional Neural Networks and LSTM-based temporal modeling to identify manipulated videos with high precision. Optimized for lightweight deployment and real-world use cases.

🚀 Project Overview

This project implements a hybrid CNN + LSTM deep learning architecture capable of analyzing temporal and spatial inconsistencies across video frames to classify whether a video is REAL or FAKE.
The solution is designed for:

Digital forensics

Social media verification

Video authenticity checks

Academic research

AI/ML-based security systems

The workflow includes:

Video preprocessing

Frame extraction

Temporal feature encoding

Model inference

CLI-based prediction

🧠 Model Architecture

The detection pipeline combines two major components:

1️⃣ CNN Feature Extractor

Learns spatial representations from individual frames.
Layers include:

Conv2D → ReLU → MaxPool

Conv2D → ReLU → MaxPool

2️⃣ LSTM Temporal Encoder

Captures cross-frame temporal patterns such as:

Lip-sync mismatches

Abnormal facial warping

Frame-level artifacts

Identity inconsistencies

3️⃣ Fully Connected Layer

Outputs a final REAL / FAKE score with sigmoid activation.

📊 Key Features

✔ Lightweight & fast
✔ Works on CPU and GPU
✔ Trained on balanced real/fake video datasets
✔ Modular training & inference scripts
✔ CLI-based prediction with frame sampling
✔ Industry-aligned architecture (CNN + LSTM pipeline)

📁 Project Structure
deepfake_detection/
│── data/  
│   ├── real_videos/  
│   ├── fake_videos/  
│
│── output/
│   └── deepfake_cnn_lstm.pth
│
│── src/
│   ├── train.py
│   ├── inference.py
│   └── utils.py
│
│── README.md

📥 Dataset Context (No Redistribution)

This project uses widely recognized public deepfake datasets strictly for research and academic purposes.
The datasets are NOT included in this repository due to licensing restrictions and file size limitations.

Dataset Context

The training and evaluation process referenced:

Collections of authentic face videos

Collections of AI-manipulated deepfake videos

Standard real / fake labels

Face-cropped, preprocessed frames

Balanced train/val/test splits

These datasets are commonly used to benchmark deepfake research models and typically include:

FaceSwap/manipulated videos

GAN-generated attacks

Multi-identity recordings

Metadata and per-video labels

⚠️ IMPORTANT

Anyone wishing to reproduce results must download the datasets from their official approved sources independently.
This repository includes only code, not the dataset.

▶️ Running Inference

Once your model (deepfake_cnn_lstm.pth) is saved, run:

python inference.py


Update this line inside inference.py:

video_path = r"PATH_TO_YOUR_VIDEO.mp4"


Output includes:

Prediction score

REAL / FAKE label

Extracted frame count

🛠 Tech Stack

Python

PyTorch

OpenCV

NumPy

torchvision

📌 Future Enhancements

Real-time webcam detection

Transformer-based temporal modeling

Web dashboard for video uploads

Integration with Streamlit or FastAPI

🤝 Contributions

PRs and feature upgrades are welcome!
If you’d like to collaborate, open an issue or submit a pull request.

❤️ Acknowledgements

This work aligns with ongoing global efforts to combat misinformation and support the development of ethical AI. Special thanks to the research community providing benchmark datasets for deepfake detection.

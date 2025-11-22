Dataset Usage Disclaimer

This project uses publicly available deepfake video datasets strictly for academic research and experimentation.
The datasets are not included in this repository and cannot be redistributed due to licensing restrictions and file size limitations.

Dataset Context

To develop and evaluate the deepfake video detection model, commonly used benchmark datasets were referenced. These datasets typically contain:

Real human face videos

AI-generated/manipulated videos created using techniques such as:

Face swapping

Expression reenactment

Audio-visual inconsistencies

GAN-based synthetic generation

These datasets are widely adopted in deepfake research to ensure:

High model generalization

Exposure to multiple manipulation methods

Real-world variation across lighting, identity, and compression levels

What Was Used During Model Development

For context only (not redistributed), the datasets used generally fall under:

Collections of authentic face videos

Collections of manipulated deepfake videos

Standard train/validation/test splits

Preprocessed face-cropped frames extracted from videos

Each dataset commonly provides:

Video metadata

Labels indicating real/fake

Benchmark test lists for evaluation

Important

The datasets referenced during training must be downloaded separately from their official approved sources by anyone attempting to reproduce results.
This repository includes only the training scripts, model architecture, inference pipeline, and evaluation workflow, not the raw data.

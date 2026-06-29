# MultiModal-Wavelet-Steganography

<div align="center">

<h2>Robust Deep Learning-based Multimodal Image Steganography using Wavelet Transform and Residual Attention Networks</h2>

<p>
A PyTorch implementation of a robust multimodal steganography framework capable of securely hiding
<strong>binary images</strong> or <strong>encrypted text messages</strong> inside natural images while maintaining high visual quality and reliable secret recovery.
</p>

<p>

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-lightgrey)
![Status](https://img.shields.io/badge/Status-Research-orange)

</p>

</div>

---

# Table of Contents

- Overview
- Features
- Methodology
- Network Architecture
- Pipeline
- Repository Structure
- Installation
- Requirements
- Dataset
- Training
- Secret Embedding
- Secret Extraction
- Evaluation Metrics
- Experimental Results
- Visualizations
- Applications
- Future Work
- Citation
- License

---

# Overview

Image steganography aims to hide confidential information inside digital images while maintaining imperceptibility and enabling accurate recovery of the embedded secret.

This repository proposes a **deep learning-based multimodal steganography framework** that combines

- Discrete Wavelet Transform (DWT)
- Residual Attention Networks
- Curriculum Learning
- Robustness-aware Training
- Frequency-domain Optimization
- Perceptual Loss Functions

to securely embed either

- Binary Images
- Text Messages

Inside RGB images.

Unlike conventional spatial-domain approaches, secret information is embedded into carefully selected wavelet sub-bands, significantly improving image quality and robustness against practical distortions such as

- JPEG Compression
- Quantization
- Gaussian Noise
- Dropout Noise

---

# Features

- Deep Learning-based Steganography
- Wavelet Domain Embedding
- Image Secret Embedding
- Text Secret Embedding
- Residual Attention Encoder
- Deep CNN Decoder
- Curriculum Learning
- Channel-aware Embedding
- SSIM Optimization
- Frequency-domain Optimization
- Gradient Preservation
- XOR Encryption
- Automatic Evaluation
- Visualization Utilities
- Publication-ready Results

---

# Methodology

The proposed framework consists of three stages.

## Stage 1 — Secret Preparation

The framework supports

- Binary Secret Images
- UTF-8 Text Messages

Optional XOR encryption is applied before embedding.

---

## Stage 2 — Secret Embedding

The cover image is

RGB

↓

YUV Conversion

↓

Discrete Wavelet Transform

↓

Residual Attention Encoder

↓

Wavelet-domain Embedding

↓

Inverse DWT

↓

Stego Image

---

## Stage 3 — Secret Extraction

Stego Image

↓

Discrete Wavelet Transform

↓

Residual Decoder

↓

Secret Recovery

↓

Image or Text Reconstruction

---

# Network Architecture

Replace this image with your architecture figure.

```
docs/architecture.png
```

Example

<p align="center">

<img src="docs/architecture.png" width="900">

</p>

---

# Pipeline

Replace with your pipeline figure.

```
docs/pipeline.png
```

<p align="center">

<img src="docs/pipeline.png" width="900">

</p>

---

# Repository Structure

```
MultiModal-Wavelet-Steganography
│
├── configs
│   ├── train.yaml
│   ├── sender.yaml
│   └── receiver.yaml
│
├── datasets
│   ├── cover_images
│   ├── secret_images
│   └── test
│
├── checkpoints
│   └── best_model.pth
│
├── docs
│   ├── architecture.png
│   ├── pipeline.png
│   ├── results.png
│   ├── cover_stego.png
│   └── extraction.png
│
├── examples
│   ├── cover
│   ├── secret
│   ├── stego
│   └── recovered
│
├── results
│   ├── figures
│   ├── metrics
│   └── visualizations
│
├── scripts
│   ├── train.py
│   ├── sender.py
│   ├── receiver.py
│   └── evaluate.py
│
├── src
│   ├── models
│   ├── losses
│   ├── transforms
│   ├── metrics
│   ├── encryption
│   ├── visualization
│   └── utils
│
├── README.md
├── LICENSE
├── requirements.txt
└── .gitignore
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/yourusername/MultiModal-Wavelet-Steganography.git

cd MultiModal-Wavelet-Steganography
```

Create environment

```bash
python -m venv stego

source stego/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Requirements

- Python ≥ 3.10
- PyTorch
- torchvision
- OpenCV
- NumPy
- SciPy
- Pillow
- scikit-image
- matplotlib
- tqdm

---

# Dataset

The dataset directory should follow

```
datasets/

cover_images/

secret_images/

validation/

test/
```

Supported formats

- PNG
- JPG
- TIFF

---

# Training

Train the complete encoder-decoder framework

```bash
python scripts/train.py
```

Training includes

- Curriculum Learning
- Noise Simulation
- Frequency Optimization
- Perceptual Optimization
- Wavelet-domain Embedding
- Robust Secret Recovery

---

# Secret Embedding

## Image Secret

```bash
python scripts/sender.py \
--mode image \
--cover examples/cover.png \
--secret examples/secret.png
```

---

## Text Secret

```bash
python scripts/sender.py \
--mode text \
--text "Confidential Message"
```

---

# Secret Extraction

```bash
python scripts/receiver.py
```

Recovered secrets are automatically stored inside

```
results/recovered/
```

---

# Loss Functions

The encoder is optimized using

- Binary Cross Entropy Loss
- Mean Squared Error Loss
- L1 Loss
- Structural Similarity Loss
- Gradient Loss
- Frequency-domain Loss

These complementary objectives improve

- Imperceptibility
- Robustness
- Secret Recovery
- Structural Preservation

---

# Evaluation Metrics

## Image Quality

- PSNR
- SSIM
- MSE
- UIQI
- VIF
- NPCR
- UACI

---

## Secret Recovery

- BER
- NCC
- Extraction Accuracy

---

## Text Recovery

- Character Accuracy
- UTF-8 Validation
- Perfect Match Rate

---

# Experimental Results

| Metric | Value |
|---------|-------|
| PSNR | XX.XX dB |
| SSIM | 0.XXXX |
| BER | 0.XXXX |
| NCC | 0.XXXX |
| Extraction Accuracy | XX.XX% |

Replace with your final results.

---

# Visualizations

The framework automatically generates

- Cover vs Stego Comparison
- Difference Maps
- RGB Histograms
- Secret Recovery
- Extraction Comparison
- Performance Summary

Example

```
results/

cover_vs_stego.png

difference_map.png

histograms.png

secret_recovery.png

metrics.csv
```

---

# Applications

- Secure Multimedia Communication
- Digital Watermarking
- Medical Image Security
- Military Communication
- Cloud Storage
- IoT Security
- Edge AI
- Privacy-preserving Image Sharing

---

# Future Work

- Video Steganography
- Audio Steganography
- Transformer Encoder
- Diffusion Models
- ONNX Deployment
- Raspberry Pi Deployment
- Mobile Inference
- Adversarial Robustness

---

# Citation

If you use this repository in your research, please cite

```bibtex
@article{vashisht2026multimodal,
  title={Robust Multimodal Image Steganography using Wavelet Transform and Deep Residual Attention Networks},
  author={Tejasya Vashisht and Others},
  journal={Under Review},
  year={2026}
}
```

---

# Acknowledgements

This work was developed as part of ongoing research in

- Deep Learning
- Computer Vision
- Image Security
- Information Hiding
- Multimedia Signal Processing

---

# License

This repository is released under the MIT License.

---

# Contact

**Tejasya Vashisht**

GitHub: https://github.com/yourusername

LinkedIn: https://linkedin.com/in/yourprofile

Email: your_email@example.com

---

<div align="center">

⭐ If you find this repository useful, please consider starring it.

</div>

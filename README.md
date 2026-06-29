# MultiModal-Wavelet-Steganography

<p align="center">

<h2>Channel-Aware Blind Multimodal Image Steganography using Wavelet Transform and Residual Attention Networks</h2>

<p>

A PyTorch implementation of a robust deep learning framework for hiding **binary images** and **encrypted text messages** inside natural images using **Discrete Wavelet Transform (DWT)**, **Residual Attention Networks**, and **Curriculum Learning**.

</p>

<p>

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Research](https://img.shields.io/badge/Status-Research-orange.svg)

</p>

</p>

---

# Overview

This repository presents a **deep learning-based multimodal steganography framework** capable of securely embedding either **binary images** or **encrypted text messages** inside natural images while preserving high visual quality and enabling accurate blind extraction.

Unlike conventional spatial-domain steganography, the proposed framework performs adaptive embedding in the **Discrete Wavelet Transform (DWT)** domain using a **channel-aware residual attention encoder** that exploits the Human Visual System (HVS). The model is trained using curriculum learning, perceptual optimization, and robustness-aware noise simulation to improve resilience against practical transmission distortions.

---

# Features

- Wavelet-domain image steganography
- Multimodal secret embedding (Image & Text)
- Blind secret extraction
- Channel-aware residual attention encoder
- Deep residual decoder
- Curriculum learning strategy
- Squeeze-and-Excitation attention modules
- Multi-objective loss optimization
- XOR-based payload encryption
- JPEG, Gaussian noise, and quantization robustness
- Automatic visualization and evaluation pipeline

---

# Repository Structure

```
MultiModal-Wavelet-Steganography
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── codes
│   ├── training.py
│   ├── sender.py
│   └── receiver.py
│
└── results
    ├── cover_vs_stego.png
    ├── Encoder.png
    ├── Decoder.png
    ├── robustness_analysis.png
    └── Architecture.png
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/<username>/MultiModal-Wavelet-Steganography.git

cd MultiModal-Wavelet-Steganography
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Requirements

- Python 3.10+
- PyTorch
- torchvision
- NumPy
- OpenCV
- SciPy
- scikit-image
- matplotlib
- Pillow

---

# Training

Train the complete encoder-decoder framework

```bash
python codes/training.py
```

The training framework includes

- Curriculum Learning
- DWT-domain embedding
- SSIM Loss
- Gradient Loss
- Frequency-domain Loss
- Noise Simulation
- Adaptive Optimization

---

# Secret Embedding

Run

```bash
python codes/sender.py
```

Supported payloads

- Binary Images
- Encrypted Text

The sender automatically

- Embeds the payload
- Generates Stego Images
- Computes PSNR, SSIM, MSE, UIQI, VIF, NPCR and UACI
- Saves visualization results

---

# Secret Extraction

Run

```bash
python codes/receiver.py
```

The receiver

- Recovers hidden payloads
- Computes BER
- Computes NCC
- Computes Extraction Accuracy
- Generates recovery visualizations

---

# Experimental Results

The proposed framework was evaluated using the **UC Merced Land Use Dataset**, consisting of 2,100 aerial images across 21 land-use classes. Evaluation was performed using both **image payload** and **encrypted text payload** settings.

---

## Image Payload

| Metric | Value |
|---------|------:|
| PSNR | **40.23 dB** |
| SSIM | **0.9998** |
| MSE | **6.8100** |
| UIQI | **0.9998** |

The stego images remain visually indistinguishable from the original cover images while preserving structural similarity.

---



## Text Payload

| Metric | Value |
|---------|------:|
| PSNR | **40.23 dB** |
| SSIM | **0.9998** |
| MSE | **6.8150** |
| UIQI | **0.9998** |

The framework exhibits consistent visual quality for both image and encrypted text payloads.

---

## Secret Recovery

| Metric | Value |
|---------|------:|
| BER | **0.0032** |
| NCC | **0.9902** |
| Extraction Accuracy | **99.68%** |

The proposed decoder successfully reconstructs hidden payloads with minimal bit errors.

<p align="center">
<img src="results/extraction_results.png" width="900">
</p>

---

## Text Recovery

| Metric | Value |
|---------|------:|
| Character Accuracy | **99.85%** |
| Perfect Match | **9 / 11 Images** |

The proposed framework reliably reconstructs encrypted text messages with very high accuracy.

---

## Robustness Analysis

The proposed model was evaluated under multiple representative image processing attacks.

| Attack | PSNR | SSIM | BER |
|---------|------:|------:|------:|
| Gaussian Noise | 41.78 | 0.9891 | 0.0071 |
| Speckle Noise | 40.28 | 0.9826 | 0.0093 |
| JPEG (Q90) | 31.65 | 0.9570 | 0.0136 |
| JPEG (Q80) | 29.34 | 0.9332 | 0.0237 |
| JPEG (Q70) | 27.79 | 0.9113 | 0.0317 |
| JPEG (Q50) | 25.92 | 0.8744 | 0.0438 |

The framework demonstrates strong robustness against common transmission distortions while maintaining reliable secret recovery.


---

## Comparison with Existing Methods

| Method | PSNR | SSIM | BER |
|---------|------:|------:|------:|
| **Proposed** | **40.23** | **0.9998** | **0.0032** |
| INN | 20.24 | 0.9620 | 0.5343 |
| AttenHideNet | 36.62 | 0.9963 | 0.5005 |
| Segmented-Huffman | 54.80 | 0.9994 | 0.3712 |

The proposed framework achieves superior structural fidelity and payload recovery accuracy while maintaining excellent visual quality.

---

# Applications

- Secure Multimedia Communication
- Medical Image Security
- Military Communication
- Digital Watermarking
- Privacy-preserving Image Sharing
- IoT Security
- Cloud Storage
- Edge AI Systems

---

# Future Work

- Video Steganography
- Audio Steganography
- Vision Transformers
- Diffusion-based Steganography
- ONNX Deployment
- Raspberry Pi Deployment
- Mobile Inference

---


# License

This project is released under the **MIT License**.

---

# Contact

**Tejasya Vashisht**

Electrical and Computer Engineering  
Thapar Institute of Engineering and Technology

📧 Email: tvashisht_be23@thapar.edu


---

<div align="center">

### ⭐ If you find this work useful, please consider giving the repository a star.

</div>

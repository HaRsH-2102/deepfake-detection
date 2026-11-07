# Deepfake Detection: An Explainable Hybrid Spatiotemporal Framework

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61dafb.svg)](https://react.dev/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-FaceForensics++%20(c23)-orange.svg)](https://github.com/ondyari/FaceForensics)

An end-to-end, explainable hybrid spatiotemporal framework for deepfake detection across both static images and video media. The framework integrates **Multi-task Cascaded Convolutional Neural Networks (MTCNN)** for face localization, an ImageNet-pretrained **XceptionNet** for fine-grained spatial artifact extraction, a **Bidirectional Long Short-Term Memory (BiLSTM)** network for inter-frame temporal sequence modeling, and **Explainable AI (Grad-CAM & LIME)** for transparent forensic decision-making.

---

## 📑 Table of Contents
1. [Key Features](#-key-features)
2. [Datasets Used](#-datasets-used)
3. [Identity-Based Partitioning Protocol](#-identity-based-partitioning-protocol)
4. [Quantitative Performance & Uncertainty](#-quantitative-performance--uncertainty)
5. [Hardware & Computational Profile](#-hardware--computational-profile)
6. [Visual Results Gallery](#-visual-results-gallery)
   - [Image-Level Performance](#image-level-results-xceptionnet)
   - [Video-Level Performance](#video-level-results-xceptionnet--bilstm)
   - [Explainability & Forensic Visualizations](#explainability--forensic-visualizations-xai)
7. [System Architecture](#-system-architecture)
8. [Installation & Setup](#-installation--setup)
9. [Citation & Paper](#-citation--paper)

---

## 🌟 Key Features

- **Dual-Branch Operational Pipeline**:
  - **Image Branch**: Standalone fine-tuned XceptionNet classifying single aligned facial crops ($299 \times 299$).
  - **Video Branch**: Hybrid XceptionNet feature extractor + 2-layer BiLSTM sequence classifier modeling temporal dynamics across equidistant 10-frame clips.
- **Leakage-Free Partitioning**: Strict identity-grouped splitting preventing cross-split identity leakage for both single frames and paired video manipulations.
- **Explainable Artificial Intelligence (XAI)**:
  - **Grad-CAM**: Class activation heatmaps highlighting localized spatial manipulation artifacts (texture seams, boundaries).
  - **LIME**: Superpixel region perturbations explaining positive/negative feature contributions.
- **Forensic Tooling**: EXIF metadata extraction, face landmark overlay, and automated PDF forensic report generation.
- **Production-Ready Full-Stack Dashboard**: Asynchronous FastAPI backend (REST APIs + WebSockets live progress streaming) and React 18 + Tailwind dashboard.

---

## 📊 Datasets Used

The framework is trained, validated, and evaluated on the benchmark **FaceForensics++ (FF++)** dataset under the **C23 (High Quality / Light Compression)** profile:

| Manipulation Generation Method | Description | Category |
| :--- | :--- | :--- |
| **Original / Pristine** | 1,000 authentic source video sequences featuring diverse subjects, poses, and expressions. | Authentic Source |
| **Deepfakes (DF)** | Autoencoder-based facial replacement technique transferring source facial expressions onto a target subject. | Deep Learning / GAN |
| **Face2Face (F2F)** | Facial reenactment system transferring facial expressions of a source video to a target video in real-time. | Facial Reenactment |
| **FaceSwap (FS)** | Graphics-based 3D facial model fitting and blending technique to replace identity. | Computer Vision / 3D Fitting |
| **FaceShifter (FSh)** | High-fidelity two-stage face swapping framework with an occlusion-aware face generator (AEI-Net). | Advanced Neural Inpainting |
| **NeuralTextures (NT)** | Patch-based neural rendering architecture for facial reenactment using learned neural textures. | Neural Rendering |
| **DeepFakeDetection (DFD)** | Expanded dataset of consented actors with thousands of manipulated sequences. | Actor-Consented Forgeries |

---

## 🔒 Identity-Based Partitioning Protocol

To ensure realistic, unbiased generalization to unseen subjects, an **identity-grouped splitting strategy** was enforced:
- Every original subject identity and all manipulated videos containing that identity are restricted to a single subset.
- Paired manipulations are retained **only if both underlying identities belong to the exact same partition**.

| Dataset Split | Unique Identities | Image-Level Samples (Frames) | Video-Level Samples (Videos) |
| :--- | :---: | :---: | :---: |
| **Training** | 699 (Images) / 719 (Videos) | 102,211 (22,360 Real / 79,851 Fake) | 3,408 (704 Real / 2,704 Fake) |
| **Validation** | 149 (Images) / 154 (Videos) | 9,240 (4,761 Real / 4,479 Fake) | 362 (147 Real / 215 Fake) |
| **Testing** | 151 (Images) / 155 (Videos) | 8,986 (4,828 Real / 4,158 Fake) | 332 (149 Real / 183 Fake) |
| **Total** | **999 (Images) / 1,028 (Videos)** | **120,437 Aligned Frames** | **4,102 Video Clips** |

---

## 📈 Quantitative Performance & Uncertainty

Evaluated under standard binary decision threshold ($\tau = 0.50$) with statistical uncertainty quantified through non-parametric bootstrap resampling ($B = 1,000$ iterations, $95\%$ Confidence Intervals):

| Branch / Model | Evaluation Subset | Accuracy (%) [95% CI] | Precision (%) [95% CI] | Recall (%) [95% CI] | F1-Score (%) [95% CI] | AUC-ROC (%) [95% CI] | Loss |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Image-Level (XceptionNet)** | Validation Set (9,240 crops) | 97.22 ± 0.17 [96.89, 97.55] | 96.33 ± 0.28 [95.77, 96.87] | 97.99 ± 0.21 [97.56, 98.39] | 97.16 ± 0.17 [96.82, 97.50] | 99.42 ± 0.07 [99.28, 99.54] | 0.0837 |
| | **Test Set (8,986 crops)** | **95.37 ± 0.23 [94.91, 95.80]** | **94.59 ± 0.35 [93.90, 95.28]** | **95.45 ± 0.33 [94.80, 96.07]** | **95.02 ± 0.25 [94.51, 95.50]** | **99.19 ± 0.07 [99.04, 99.32]** | **0.1123** |
| **Video-Level (XceptionNet–BiLSTM)** | Validation Set (362 clips) | 98.90 ± 0.54 [97.79, 99.72] | 100.00 ± 0.00 [100.0, 100.0] | 98.14 ± 0.92 [96.28, 99.53] | 99.06 ± 0.46 [98.04, 99.77] | 100.00 ± 0.00 [100.0, 100.0] | 0.0189 |
| | **Test Set (332 clips)** | **96.99 ± 0.94 [94.88, 98.49]** | **98.32 ± 0.97 [96.22, 100.0]** | **96.17 ± 1.41 [93.01, 98.44]** | **97.24 ± 0.87 [95.32, 98.72]** | **99.47 ± 0.36 [98.67, 99.94]** | **0.0877** |

> **Ablation Insight**: Incorporating BiLSTM temporal sequence modeling yields an absolute gain of **+1.62% in test accuracy** and **+2.22% in test F1-score** over static frame analysis alone, proving the necessity of temporal modeling for video deepfake detection.

---

## ⚡ Hardware & Computational Profile

Benchmarked on **NVIDIA GeForce RTX 4060 Laptop GPU (8 GB GDDR6)** & **Intel Core i7-14650HX CPU (16 cores, 24 threads)**:

| Component / Model | Input Specification | Total Parameters | FLOPs | GPU Latency | GPU Throughput | Peak VRAM | CPU Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Image Backbone (XceptionNet)** | $299 \times 299 \times 3$ (1 crop) | 20.81 M | 8.40 GFLOPs | **4.63 ms** | **215.8 FPS** | 279.5 MB | 84.46 ms (11.8 FPS) |
| **Temporal Head (BiLSTM)** | $10 \times 2048$ (1 sequence) | 6.30 M | 0.06 GFLOPs | **0.48 ms** | 2,083 seq/s | 38.2 MB | 2.15 ms |
| **Full Video Pipeline** | 10 frames ($10 \times 299 \times 299$) | **27.11 M** | **84.06 GFLOPs** | **42.76 ms** | **23.4 clips/s (234 FPS)** | **631.9 MB** | 846.8 ms |

- **Image Training Time**: 521.54 min (~8.69 hrs) for 12 epochs with AMP.
- **Video Training Time**: 170.87 min (~2.85 hrs) for 7 epochs on precomputed embeddings.

---

## 🖼 Visual Results Gallery

### Image-Level Results (XceptionNet)

#### 1. Learning Curves across Epochs
![Image Learning Curves](docs/assets/image_results/learning_curves.png)

#### 2. Confusion Matrices (Validation & Test Sets)
<p align="center">
  <img src="docs/assets/image_results/val_confusion_matrix.png" width="48%" alt="Val Confusion Matrix"/>
  <img src="docs/assets/image_results/test_confusion_matrix.png" width="48%" alt="Test Confusion Matrix"/>
</p>

#### 3. Receiver Operating Characteristic (ROC-AUC) Curves
<p align="center">
  <img src="docs/assets/image_results/val_roc_curve.png" width="48%" alt="Val ROC Curve"/>
  <img src="docs/assets/image_results/test_roc_curve.png" width="48%" alt="Test ROC Curve"/>
</p>

#### 4. Test Error Gallery (Boundary Mistakes & Challenging Cases)
![Image Test Error Gallery](docs/assets/image_results/test_error_gallery.png)

---

### Video-Level Results (XceptionNet + BiLSTM)

#### 1. Video Temporal Learning Curves
![Video Learning Curves](docs/assets/video_results/video_learning_curves.png)

#### 2. Video Confusion Matrices
<p align="center">
  <img src="docs/assets/video_results/video_val_confusion_matrix.png" width="48%" alt="Video Val Confusion Matrix"/>
  <img src="docs/assets/video_results/video_test_confusion_matrix.png" width="48%" alt="Video Test Confusion Matrix"/>
</p>

#### 3. Video ROC-AUC Curves
<p align="center">
  <img src="docs/assets/video_results/video_val_roc_curve.png" width="48%" alt="Video Val ROC Curve"/>
  <img src="docs/assets/video_results/video_test_roc_curve.png" width="48%" alt="Video Test ROC Curve"/>
</p>

#### 4. Video Test Error Gallery
![Video Test Error Gallery](docs/assets/video_results/video_test_error_gallery.png)

---

### Explainability & Forensic Visualizations (XAI)

#### 1. Grad-CAM Class Activation Mapping
Grad-CAM heatmaps highlight localized manipulation regions (eyes, nose bridge, boundary seams):
<p align="center">
  <img src="docs/assets/image_results/gradcam.png" width="70%" alt="Grad-CAM Output"/>
</p>

#### 2. LIME Superpixel Explanations
LIME superpixel segmentation highlights positive (green) and negative (red) features influencing model output:
<p align="center">
  <img src="docs/assets/image_results/lime.png" width="70%" alt="LIME Output"/>
</p>

#### 3. Facial Landmark Overlay & Alignment
<p align="center">
  <img src="docs/assets/image_results/landmarks.png" width="48%" alt="Facial Landmarks"/>
  <img src="docs/assets/image_results/face_annotated.png" width="48%" alt="Annotated Face"/>
</p>

#### 4. Multi-View Forensic Verification Pipeline
![Explainability Comparison](docs/assets/image_results/explainability_comparison.png)

---

## 🏗 System Architecture

```text
deepfake-detection/
├── backend/
│   ├── main.py                 # FastAPI endpoints (REST + WebSockets)
│   ├── predictor.py            # Inference engine (Xception + BiLSTM)
│   ├── explainability.py       # Grad-CAM and LIME explainers
│   ├── forensics.py            # EXIF metadata parser
│   ├── face_landmarks.py       # MTCNN landmark annotator
│   └── report_generator.py     # Forensic PDF report builder
├── frontend/
│   ├── src/                    # React 18 + Vite + Tailwind forensic dashboard
│   ├── package.json
│   └── Dockerfile
├── own_train/
│   ├── train_xception.py       # XceptionNet fine-tuning pipeline
│   ├── Train_LSTM.py           # BiLSTM sequence training pipeline
│   ├── data.py                 # Identity-grouped data loader
│   ├── testing.py              # Statistical evaluation and bootstrapping
│   └── plotting.py             # ROC curves, confusion matrices, error galleries
├── docs/
│   └── assets/                 # High-resolution result figures and plots
├── Article_Journal_DD.docx     # Revised Research Article
├── review_response.docx        # Point-by-point reviewer response documentation
├── docker-compose.yml          # Containerized deployment stack
└── .gitignore                  # Git tracking rules excluding large checkpoints & videos
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10+
- Node.js 18+
- NVIDIA GPU with CUDA 11.8+ (optional for CPU mode)

### 1. Clone Repository
```bash
git clone https://github.com/HaRsH-2102/deepfake-detection.git
cd deepfake-detection
```

### 2. Backend Setup
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt

# Start backend server
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

### 4. Docker Deployment
```bash
docker compose up --build
```

---

## 📜 Citation & Paper

If you find this research or code useful, please cite our corresponding journal article:

```bibtex
@article{gavali2026deepfake,
  title={An Explainable Hybrid Spatiotemporal Framework for Deepfake Detection in Images and Videos},
  author={Gavali, Ashwini S. and More, Harshal S. and Bangale, Purva A.},
  journal={Department of Artificial Intelligence and Data Science, CSMSS Chh. Shahu College of Engineering},
  year={2026}
}
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

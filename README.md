# Cross-Domain Self-Supervised Adaptation Pipeline for Facial Affect & Temporal Stress Inference

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Backbone](https://img.shields.io/badge/Backbone-EfficientNet--B0-brightgreen.svg)](https://github.com/lukemelas/EfficientNet-PyTorch)
[![Temporal](https://img.shields.io/badge/Temporal-Bidirectional%20GRU%20%2B%20Attention-orange.svg)]()
[![Dataset](https://img.shields.io/badge/Datasets-UBFC--rPPG%20%7C%20RAF--DB-purple.svg)]()

An end-to-end deep learning framework implementing the complete **5-Phase Cross-Domain Self-Supervised Adaptation Pipeline** for facial affective representation learning and temporal stress inference on raw video frames. 

The pipeline bridges self-supervised visual pre-training on unlabeled face videos (**UBFC-rPPG**), affective transfer learning across 7 basic emotions (**RAF-DB**), and temporal micro-expression sequence modeling (**Bidirectional GRU with Temporal Attention and Moving-Average Smoothing**) to achieve **contact-free, camera-only stress detection**.

---

## 📌 Architecture & Pipeline Overview

```text
               [Unlabeled Facial Videos (UBFC-rPPG: 42 Subjects)]
                                      │
                                      ▼  [Phase 1: Self-Supervised SimCLR Pre-Training]
                 Invariant Facial Geometry Backbone (Loss: 0.6865 ➔ 0.1995)
                                      │
                                      ▼  [Phase 2: Supervised Affective Transfer (RAF-DB)]
               Fine-Tuned Affective Feature Extractor — 71.84% Test Accuracy
                         (1,280-Dimensional Facial Affect Vectors)
                                      │
                                      ▼  [Phase 3: Temporal Stress Sequence Modeling]
                   10-Frame Sliding Windows (W=10, 6 FPS ≈ 1.7s)
                                      │
                         ┌────────────┴────────────┐
                         ▼                         ▼
                 Bidirectional GRU       Temporal Attention Pooling
                         │                         │
                         └────────────┬────────────┘
                                      ▼
                        Binary Stress Classifier (0 = Calm, 1 = Stress)
                                      │
                                      ▼  [Step 22: Moving-Average Smoothing Filter]
                           Smoothed Stress Signal (34.6% Jitter Reduction)
                                      │
                                      ▼  [Phase 4: Real-Time Streaming & Deployment]
                   Live Webcam / Video Heads-Up Display (HUD) Dashboard
                                      │
                                      ▼  [Phase 5: ONNX & Edge Optimization]
               ONNX Model Graph Conversion & Real-Time CPU/Edge Acceleration
```

---

## 📊 Comprehensive Performance Scorecard

### 1. Phase 1 — Self-Supervised Pre-Training (SimCLR on UBFC Videos)
* **Dataset:** 16,719 aligned $224 \times 224$ face frames across 42 UBFC subjects.
* **Loss Function:** NT-Xent (Normalized Temperature-scaled Cross Entropy, $\tau = 0.07$).
* **Training Convergence:** Train Loss decreased from **0.6865 $\to$ 0.1995**; Validation Loss dropped from **0.6108 $\to$ 0.1961**.
* **Representation Learned:** General facial geometry, eye and mouth structures, skin texture, and pose invariants without requiring any human labels.

### 2. Phase 2 — Supervised Affective Transfer Learning (RAF-DB 7-Class Emotions)
* **Dataset:** 12,271 training faces / 3,068 test faces across 7 basic emotions (*Surprise, Fear, Disgust, Happiness, Sadness, Anger, Neutral*).
* **Class Imbalance Strategy:** Square-root smoothed inverse class-frequency weighting ($(\frac{N}{7 \cdot N_c})^{0.5}$).
* **Progressive Unfreezing:**
  * **Stage A (Frozen Backbone):** 55.44% Test Accuracy.
  * **Stage B (Top 2 Blocks Unfrozen):** 56.49% Test Accuracy.
  * **Stage B+ (Top 3 Blocks Unfrozen + Smoothed Weights):** 69.59% Test Accuracy.
  * **Stage C (Refined LR $8 \times 10^{-5}$):** **71.84% Test Accuracy** (Train Acc: 76.97%, Test Loss: 0.8041).
* **Output:** 1,280-dimensional affective feature representation vectors ($F_t \to f_t$).

### 3. Phase 3 — Temporal Stress Inference (Subject-Independent Split on UBFC)
* **Protocol:** First 30 seconds ($t \le 30\text{s}$, frames 1–180) = **Calm Baseline (0)**; After 30 seconds ($t > 30\text{s}$, frames > 180, Mental Arithmetic Quiz) = **Acute Stress (1)**.
* **Subject Split:** 
  * **Train:** Subjects 1–32 (4,784 sliding windows)
  * **Validation:** Subjects 33–37 (508 sliding windows)
  * **Held-Out Test:** Subjects 38–42 (508 sliding windows — **completely unseen people**)
* **Test Performance (Subjects 38–42):**
  * **Stress Recall (Sensitivity):** **83.33%** (Catches >83% of all genuine acute stress episodes).
  * **F1-Score:** **67.61%**.
  * **Precision:** **56.87%**.
  * **Overall Accuracy:** **54.72%**.
* **Step 22 Moving-Average Smoothing Filter ($K=3$):**
  * Raw Prediction Jitter: `0.0364` $\longrightarrow$ Smoothed Jitter: `0.0238` (**34.6% reduction in flicker!**).

---

## 📁 Repository Structure

```text
├── config.py                               # Central hyperparameters, image sizes, and subject splits
├── datasets/
│   ├── simclr_dataset.py                   # Positive pair augmentation pipeline for SimCLR
│   ├── rafdb_dataset.py                    # 7-Class RAF-DB dataset loader
│   └── temporal_dataset.py                 # Sliding temporal window dataset for UBFC videos
├── models/
│   ├── efficientnet_encoder.py             # EfficientNet-B0 visual backbone (1,280-d features)
│   ├── simclr_model.py                     # Self-supervised encoder + MLP projection head
│   ├── emotion_classifier.py               # 7-Class emotion classifier head
│   └── temporal_model.py                   # Bidirectional GRU + Temporal Attention Pooling head
├── preprocessing/
│   ├── extract_ubfc_faces.py               # Native ultra-fast FFmpeg face crop extraction
│   └── mediapipe_face_detector.py          # MediaPipe BlazeFace face detector & tracker
├── training/
│   ├── train_simclr.py                     # Phase 1: Self-supervised pretraining
│   ├── train_rafdb_transfer.py             # Phase 2: Affective transfer learning
│   └── train_temporal.py                   # Phase 3: Recurrent temporal stress training
├── evaluation/
│   ├── evaluate_rafdb.py                   # Phase 2 emotion evaluation
│   └── evaluate_temporal.py                # Phase 3 confusion matrix & threshold sweep
├── inference/
│   └── predict_realtime.py                 # Phase 4 live webcam / video inference HUD
├── optimization/
│   ├── export_onnx.py                      # Phase 5 PyTorch to ONNX graph conversion
│   └── benchmark_performance.py            # Latency, throughput (FPS), and memory benchmark
├── models_checkpoints/                     # Pretrained weights (.pth & .tflite)
│   ├── self_supervised_backbone.pth        # Phase 1 SimCLR backbone (16.3 MB)
│   ├── best_rafdb_affective_model.pth      # Phase 2 Affective Emotion model (45.6 MB)
│   ├── best_temporal_stress_model.pth      # Phase 3 Temporal Stress BiGRU (2.44 MB)
│   └── blaze_face_short_range.tflite       # MediaPipe face detector model
├── logs/                                   # Verified training CSV logs
│   ├── simclr_training_history.csv         # Phase 1 epoch-by-epoch loss log
│   ├── rafdb_transfer_history.csv          # Phase 2 epoch-by-epoch accuracy & loss log
│   └── temporal_training_history.csv       # Phase 3 recurrent model training history
└── results/                                # Generated evaluation plots & matrices
    ├── temporal/                           # Confusion matrices & smoothing comparison plots
    └── raf_db/                             # Training accuracy & loss curves
```

---

## ⚡ Quickstart & Reproducibility Guide

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/Nilkamal21/Stress-Detection.git
cd Stress-Detection
pip install torch torchvision opencv-python pillow pandas numpy scikit-learn seaborn matplotlib
```

### 2. Run the Real-Time Webcam Demo (Local PC)
With all pretrained checkpoints placed in `models_checkpoints/`, run live camera inference:
```bash
python inference/predict_realtime.py --webcam 0
```
* Press `q` or `ESC` to quit.

### 3. Evaluate Temporal Stress Model (Confusion Matrix)
Run the held-out subject evaluation:
```bash
python evaluation/evaluate_temporal.py --data_dir ubfc_faces
```

### 4. Phase 5 ONNX Optimization
Export trained PyTorch checkpoints into standardized ONNX runtime formats:
```bash
python optimization/export_onnx.py
```

---

## 📜 Scientific Methodology & Technical Documentation

For an in-depth breakdown of the mathematical formulations, architectural decisions, and an exhaustive list of **Research Methodology & Technical FAQs (25 Core Inquiries)**, please read:
👉 **[`explanation.md`](explanation.md)**

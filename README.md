# Cross-Domain Self-Supervised Adaptation Pipeline for Facial Affect & Temporal Stress Inference

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%20CUDA-red.svg)](https://pytorch.org/)
[![Backbone](https://img.shields.io/badge/Backbone-EfficientNet--B0-brightgreen.svg)](https://github.com/lukemelas/EfficientNet-PyTorch)
[![Temporal](https://img.shields.io/badge/Temporal-Bidirectional%20GRU%20%2B%20Attention-orange.svg)]()
[![Status](https://img.shields.io/badge/Status-Phases%201--5%20Complete%20%26%20Verified-success.svg)]()

An end-to-end deep learning framework designed to solve real-world driver affect and stress detection on **KMU-FED** (12 vehicle drivers, 61 video sequences, 1,045 eye-aligned frames) through **Self-Supervised SimCLR Pre-Training**, **Cross-Domain Affective Transfer (RAF-DB)**, **Temporal Sequence Modeling (BiGRU with Attention & Moving-Average Smoothing)**, and **Phase 5 ONNX & Quantized Hardware Optimization**.

---

## 📌 Architecture & Pipeline Overview

```text
Unlabeled Facial Video Sequences (KMU-FED)
                 │
                 ▼  [Cross-Domain Phase 1: Self-Supervised SimCLR Pre-Training]
      Invariant Face Geometry Representation (self_supervised_backbone.pth)
                 │
                 ▼  [Cross-Domain Phase 2: Affective Transfer Learning (RAF-DB)]
      Fine-Tuned Affective Backbone — 70.73% Test Accuracy (best_rafdb_affective_model.pth)
                 │
                 ▼  [Cross-Domain Phase 3: Temporal Sequence Modeling]
      10-Frame Sliding Windows ➔ Bidirectional GRU ➔ Temporal Attention Spotlight
      Step 22 Moving-Average Smoothing ➔ 93.18% Test Accuracy (best_temporal_stress_model.pth)
                 │
                 ▼  [Cross-Domain Phase 4: Real-Time Streaming & Tracking]
      Rolling Buffer [F_t-9, ..., F_t] ➔ Real-Time Heads-Up Display (HUD) Dashboard
                 │
                 ▼  [Cross-Domain Phase 5: Hardware Optimization & Deployment]
      ONNX Export (16.5 MB) ➔ INT8 Quantization (9.8 MB) ➔ Adaptive Tracking Engine (69.2 ms / 14.5 FPS)
```

---

## 📊 Comprehensive Performance Scorecard

### 0. Cross-Domain Phase 5: Hardware Optimization & Turbo Deployment
| Metric / Component | Score | Clinical & Operational Advantage |
| :--- | :---: | :--- |
| **ONNX Runtime CPU Speed** | **16.03 ms / 62.4 FPS** | **2.0x faster than PyTorch CPU**, matching dedicated GPU speeds on commodity CPUs |
| **CUDA GPU Throughput** | **60.7 FPS (FP32) / 52.7 FPS (FP16)** | Real-time vehicle compute with only **19.6 MB VRAM footprint** (34% VRAM saved) |
| **INT8 Memory Reduction** | **55.3% Space Saved (9.8 MB total)** | Shrinks entire dual-model pipeline from 22.01 MB to 9.83 MB for micro-controllers |
| **Adaptive Tracking Speedup** | **4.1x End-to-End Speedup** | Latency slashed from 281.3 ms down to **69.2 ms (14.5 FPS)** on pure CPU |
| **Tracking-Frame Latency** | **8.3 ms – 14.1 ms** | Instantaneous 70–120 FPS capability between detection intervals |

### 1. Cross-Domain Phase 3: Temporal Stress Inference (Held-Out Drivers `[02, 11]`)
| Metric | Score | Clinical / Operational Meaning |
| :--- | :---: | :--- |
| **Test Accuracy** | **93.18%** | 41 out of 44 temporal sequence windows correctly classified |
| **Test Precision** | **100.00%** | **Zero False Alarms!** Every stress alert is 100% genuine |
| **Test Recall** | **90.32%** | Successfully catches over 9 out of 10 acute stress events |
| **Test F1-Score** | **94.92%** | High harmonic mean between precision and recall |
| **Test ROC-AUC** | **98.26%** | Near-perfect separation between calm and stressed score distributions |
| **Step 22 Jitter Reduction** | **16.9%** | Rolling moving-average filter ($K=3$) eliminates annoying frame flicker |

### 2. Cross-Domain Phase 2: Affective Transfer Learning (RAF-DB Test Set: 3,068 Faces)
| Metric | Score | Note |
| :--- | :---: | :--- |
| **Test Accuracy** | **70.73%** | 2,170 / 3,068 in-the-wild test faces correct (Stage B Fine-Tuning) |
| **Weighted F1-Score** | **70.13%** | Balanced across imbalanced real-world distributions |
| **Fear Precision (Key Stress Cue)** | **77.42%** | Highly reliable detection of acute distress cues |
| **Happiness Precision** | **86.27%** | Clean separation of positive/calm states |

### 3. Baseline KMU-FED FER: Static Image Pipeline (Phases 1–17)
- **Overall Accuracy:** **57.05%** on unseen test subjects (vs 16.67% random baseline).
- **Strict Subject Independence:** Zero leakage between training and testing drivers.

---

## 🗂️ Project Directory Structure

```text
KMU-FED-2/
├── config.py                       # Central paths, constants, and hyperparameters
├── for_me.md                       # Beginner-friendly master guide with Q&A boxes
├── explainable.md                  # Comprehensive engineering design decisions & metrics
├── predict.py                      # Baseline single-image emotion inference tool
│
├── data/
│   ├── frames/                     # Verified video frames staged into 61 sequences
│   ├── detected_faces/             # Cropped faces (MediaPipe BlazeFace with 10% safety margin)
│   ├── aligned_faces/              # Faces rotated to level the eye axis horizontally
│   ├── processed/                  # Final 224x224 RGB faces ready for neural networks
│   ├── features/                   # Cached 1280-dim temporal feature vectors (.pt)
│   ├── metadata/                   # Train/Val/Test CSVs and sequence logs
│   └── DATASET/                    # RAF-DB dataset (15,339 aligned faces)
│
├── datasets/
│   ├── simclr_dataset.py           # Phase 1: Self-supervised positive-pair augmentations
│   ├── rafdb_dataset.py            # Phase 2: RAF-DB affective dataset & dataloaders
│   ├── temporal_dataset.py         # Phase 3: 10-frame sliding window temporal dataset
│   └── contrastive_dataset.py      # Baseline: Siamese contrastive pair dataset
│
├── models/
│   ├── efficientnet_encoder.py     # EfficientNet-B0 backbone (1,280 feature outputs)
│   ├── simclr_model.py             # Phase 1: SimCLR model with NT-Xent contrastive loss
│   ├── emotion_classifier.py       # Phase 2: 7-class emotion classification head
│   ├── temporal_model.py           # Phase 3: Bidirectional GRU with Temporal Attention
│   └── contrastive_model.py        # Baseline: Siamese contrastive network
│
├── models_checkpoints/
│   ├── self_supervised_backbone.pth # Phase 1: Pre-trained SimCLR backbone (16.3 MB)
│   ├── best_rafdb_affective_model.pth # Phase 2: Fine-tuned 70.73% model (20.3 MB)
│   ├── best_temporal_stress_model.pth # Phase 3: Best BiGRU temporal stress model
│   ├── blaze_face_short_range.tflite  # MediaPipe BlazeFace face detector model (229 KB)
│   └── best_emotion_model.pth      # Baseline: End-to-end KMU-FED emotion checkpoint
│
├── training/
│   ├── train_simclr.py             # Phase 1: Self-supervised pre-training loop
│   ├── train_rafdb_transfer.py     # Phase 2: Stage A warmup + Stage B fine-tuning
│   ├── train_temporal.py           # Phase 3: BiGRU training & Step 22 smoothing evaluation
│   ├── train_contrastive.py        # Baseline: Siamese contrastive training
│   └── train_classifier.py         # Baseline: Three-stage classifier training
│
├── inference/
│   └── predict_realtime.py         # Phase 4: Live webcam & video sliding-window engine
│
├── evaluation/
│   ├── evaluate_rafdb.py           # Phase 2: Full RAF-DB confusion matrices & reports
│   └── evaluate.py                 # Baseline: KMU-FED evaluation suite
│
├── logs/                           # Training history CSVs and failure records
└── results/
    ├── realtime/                   # Phase 4: Heads-up display preview & JSON summary
    ├── temporal/                   # Phase 3: Moving-average smoothing comparison plots
    ├── raf_db/                     # Phase 2: Heatmaps, normalized CM, and reports
    └── kmu_fed/                    # Baseline: t-SNE, PCA, and per-class bar charts
```

---

## 🚀 Quick Start & Usage

### 1. Installation
```bash
git clone <repo-url>
cd KMU-FED-2

# Install required dependencies
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn pillow mediapipe opencv-python onnx onnxruntime psutil
```

### 2. Phase 5: Run High-Speed Optimized Deployment Engine (ONNX + Adaptive Tracking)
```bash
# Option A: Run high-throughput sequence demo with 4.1x speedup (69.2 ms / 14.5 FPS on CPU):
python inference/predict_realtime_optimized.py

# Option B: Run live interactive webcam monitor with ONNX telemetry:
python inference/predict_realtime_optimized.py --webcam

# Run multi-engine latency/throughput benchmarks & component breakdown:
python optimization/benchmark_performance.py
```

### 3. Phase 4: Baseline Real-Time Temporal Inference (PyTorch Engine)
```bash
# Run baseline streaming pipeline:
python inference/predict_realtime.py
```

### 4. Phase 3: Train & Test Temporal Stress Model (BiGRU + Smoothing)
```bash
# Test the 10-frame sliding window dataset generator:
python datasets/temporal_dataset.py

# Test the recurrent model architecture:
python models/temporal_model.py

# Run temporal training, held-out evaluation & Step 22 smoothing:
python training/train_temporal.py
```

### 5. Phase 2: RAF-DB Affective Transfer Learning
```bash
# Stage B Fine-Tuning (Differential LR: Backbone 5e-5, Head 5e-4):
python training/train_rafdb_transfer.py --epochs 3 --batch_size 32 --unfreeze_blocks 3

# Evaluate test set metrics:
python evaluation/evaluate_rafdb.py
```

### 6. Phase 1: Self-Supervised SimCLR Pre-Training
```bash
# Pre-train EfficientNet-B0 on unlabeled face crops using NT-Xent loss:
python training/train_simclr.py --epochs 3 --batch_size 32
```

---

## 📈 Visual Artifacts & Proofs

Key visual proofs generated by the pipeline:

* **Hardware Optimization Benchmark:** [`results/optimization/optimization_comparison.png`](file:///c:/Users/kundu/OneDrive/Desktop/KMU-FED-2/results/optimization/optimization_comparison.png)  
  *Shows multi-engine FPS comparisons and the component latency breakdown.*
* **Turbo Real-Time Heads-Up Display:** [`results/realtime/optimized_realtime_preview.jpg`](file:///c:/Users/kundu/OneDrive/Desktop/KMU-FED-2/results/realtime/optimized_realtime_preview.jpg)  
  *Shows real-time ONNX streaming preview across vehicle frames.*
* **Real-Time Heads-Up Display (HUD):** [`results/realtime/realtime_preview.jpg`](file:///c:/Users/kundu/OneDrive/Desktop/KMU-FED-2/results/realtime/realtime_preview.jpg)  
  *Shows the 3-stage visual shift from calm driving (green border) to acute stress alert (red progress bar).*
* **Temporal Moving-Average Smoothing Filter:** [`results/temporal/temporal_smoothing_comparison.png`](file:///c:/Users/kundu/OneDrive/Desktop/KMU-FED-2/results/temporal/temporal_smoothing_comparison.png)  
  *Demonstrates how Step 22 suppresses rapid prediction jitter by 16.9%.*
* **RAF-DB Confusion Matrices:** [`results/raf_db/confusion_matrix_normalized.png`](file:///c:/Users/kundu/OneDrive/Desktop/KMU-FED-2/results/raf_db/confusion_matrix_normalized.png)  
  *Full 7x7 normalized confusion matrix on 3,068 test images.*
* **SimCLR Positive Pair Strip:** [`results/simclr_pair_preview.jpg`](file:///c:/Users/kundu/OneDrive/Desktop/KMU-FED-2/results/simclr_pair_preview.jpg)

---

## 📖 In-Depth Documentation

* [**`for_me.md`**](file:///c:/Users/kundu/OneDrive/Desktop/KMU-FED-2/for_me.md): Master beginner-friendly guide with real-world analogies (*The Portrait Artist*, *The Flipbook*, *The Park Bench*, *The Conveyor Belt*) and plain-English Q&A.
* [**`explainable.md`**](file:///c:/Users/kundu/OneDrive/Desktop/KMU-FED-2/explainable.md): Full engineering documentation with mathematical proofs, selective unfreezing tables, loss functions, and architectural rationales.

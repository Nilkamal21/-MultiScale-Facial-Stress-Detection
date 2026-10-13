"""
training/train_temporal.py
Phase 3 (Temporal Stress Inference) — Steps 20, 21 & 22: Temporal Model Training & Smoothing.

1. Pre-extracts/caches frame-level affective feature representations [W, 1280] using the
   Phase 2 fine-tuned model (best_rafdb_affective_model.pth).
2. Trains the Bidirectional GRU with Temporal Attention on subject-independent sequence splits.
3. Implements Step 22: Moving-average temporal smoothing filter to eliminate frame flicker.
4. Generates performance reports, smoothing comparison plots, and saves the final checkpoint.
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.efficientnet_encoder import EfficientNetEncoder
from models.emotion_classifier import EmotionClassifier, EndToEndEmotionModel
from datasets.temporal_dataset import (
    KMUTemporalDataset,
    get_temporal_dataloader,
    SUBJECT_SPLITS,
    AFFECTIVE_TENSION_MAP,
)
from models.temporal_model import TemporalStressGRU, EndToEndTemporalStressModel

TEMPORAL_RESULTS_DIR = os.path.join(config.RESULTS_DIR, "temporal")
FEATURES_CACHE_DIR = os.path.join(config.DATA_DIR, "features")


def load_phase2_feature_extractor(device):
    """Loads the Phase 2 fine-tuned model to serve as the visual feature encoder."""
    backbone = EfficientNetEncoder(pretrained=False)
    classifier = EmotionClassifier(
        feature_dim=backbone.feature_dim,
        num_classes=7,
        hidden_dim=config.CLASSIFIER_HIDDEN_DIM,
        dropout_rate=config.CLASSIFIER_DROPOUT,
    )
    full_model = EndToEndEmotionModel(
        encoder=backbone,
        classifier=classifier,
        num_classes=7
    )

    ckpt_path = os.path.join(config.CHECKPOINTS_DIR, "best_rafdb_affective_model.pth")
    if os.path.exists(ckpt_path):
        print(f"[Phase 2 Model] Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        full_model.load_state_dict(state_dict)
        print("[OK] Successfully loaded Phase 2 fine-tuned affective backbone!")
    else:
        print("[WARN] Phase 2 model not found! Using Phase 1 self-supervised backbone.")
        phase1_ckpt = os.path.join(config.CHECKPOINTS_DIR, "self_supervised_backbone.pth")
        if os.path.exists(phase1_ckpt):
            full_model.encoder.load_state_dict(torch.load(phase1_ckpt, map_location=device, weights_only=False))

    full_model = full_model.to(device)
    full_model.eval()
    return full_model.encoder


def extract_or_load_features(encoder, split, window_size=10, stride=2, device="cuda"):
    """
    Extracts frame features using the Phase 2 encoder and caches them to disk.
    If already cached, loads instantly.
    """
    os.makedirs(FEATURES_CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(FEATURES_CACHE_DIR, f"features_{split}_w{window_size}_s{stride}.pt")

    if os.path.exists(cache_file):
        print(f"[Cache Hit] Loading pre-extracted features for {split.upper()} from {cache_file}")
        data = torch.load(cache_file, map_location="cpu", weights_only=False)
        return data["features"], data["labels"], data["seq_ids"]

    print(f"[Extracting] Pre-extracting frame features for {split.upper()} on {device}...")
    dataset = KMUTemporalDataset(split=split, window_size=window_size, stride=stride)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    all_features = []
    all_labels = []
    all_seq_ids = []

    encoder.eval()
    with torch.no_grad():
        for batch_video, batch_labels, _, batch_seqs in loader:
            b, w, c, h, w_dim = batch_video.shape
            flat_frames = batch_video.view(b * w, c, h, w_dim).to(device)
            feats = encoder(flat_frames)  # (b * w, 1280)
            seq_feats = feats.view(b, w, -1).cpu()  # (b, w, 1280)

            all_features.append(seq_feats)
            all_labels.append(batch_labels)
            all_seq_ids.extend(batch_seqs)

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)

    torch.save({"features": features, "labels": labels, "seq_ids": all_seq_ids}, cache_file)
    print(f"[Cached] Saved {features.shape[0]} feature windows to {cache_file}")
    return features, labels, all_seq_ids


class TemporalMovingAverageSmoother:
    """
    Step 22: Temporal Moving Average Smoothing Filter.
    Formula: p_hat_t = (1 / K) * sum_{i=0}^{K-1} p_{t-i}
    Reduces inter-frame prediction variance and eliminates flickering.
    """
    def __init__(self, window_size=3):
        self.k = window_size
        self.buffer = []

    def update(self, prob):
        self.buffer.append(prob)
        if len(self.buffer) > self.k:
            self.buffer.pop(0)
        return float(np.mean(self.buffer))

    def smooth_sequence(self, probs):
        smoothed = []
        buf = []
        for p in probs:
            buf.append(p)
            if len(buf) > self.k:
                buf.pop(0)
            smoothed.append(float(np.mean(buf)))
        return np.array(smoothed)


def train_temporal_model(
    epochs=15,
    batch_size=16,
    lr=1e-3,
    hidden_dim=128,
    dropout=0.3
):
    print("=" * 65)
    print("  Phase 3 — Temporal Stress Inference (Steps 18–22)")
    print(f"  Target Epochs  : {epochs}")
    print(f"  Batch Size     : {batch_size}")
    print(f"  Learning Rate  : {lr}")
    print(f"  GRU Hidden Dim : {hidden_dim} (Bidirectional)")
    print("=" * 65)

    os.makedirs(TEMPORAL_RESULTS_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")

    # 1. Feature Extractor
    encoder = load_phase2_feature_extractor(device)

    # 2. Extract / Load Features
    train_x, train_y, train_seqs = extract_or_load_features(encoder, "train", window_size=10, stride=2, device=device)
    val_x, val_y, val_seqs = extract_or_load_features(encoder, "val", window_size=10, stride=2, device=device)
    test_x, test_y, test_seqs = extract_or_load_features(encoder, "test", window_size=10, stride=2, device=device)

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=False)

    # 3. Model & Optimization
    model = TemporalStressGRU(
        input_dim=train_x.shape[-1],
        hidden_dim=hidden_dim,
        num_layers=2,
        num_classes=2,
        dropout=dropout,
        bidirectional=True
    ).to(device)

    # Class weights for balanced tension learning
    num_neg = (train_y == 0).sum().item()
    num_pos = (train_y == 1).sum().item()
    weight_tensor = torch.tensor([num_pos / (num_neg + 1e-5), 1.0], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_val_f1 = 0.0
    best_ckpt_path = os.path.join(config.CHECKPOINTS_DIR, "best_temporal_stress_model.pth")
    history = []

    print("\n--- Training Recurrent Temporal GRU ---")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        train_preds, train_targets = [], []

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits, _ = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            train_targets.extend(by.cpu().numpy())

        train_loss = running_loss / max(len(train_loader), 1)
        train_acc = accuracy_score(train_targets, train_preds) * 100.0

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets, val_probs = [], [], []

        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits, _ = model(bx)
                loss = criterion(logits, by)
                val_loss += loss.item()

                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                val_probs.extend(probs)
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_targets.extend(by.cpu().numpy())

        val_loss /= max(len(val_loader), 1)
        val_acc = accuracy_score(val_targets, val_preds) * 100.0
        prec, rec, f1, _ = precision_recall_fscore_support(val_targets, val_preds, average="binary", zero_division=0)
        f1 *= 100.0
        scheduler.step(f1)

        print(f"Epoch [{epoch:2d}/{epochs:2d}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:5.1f}% | Val Acc: {val_acc:5.1f}% | Val F1: {f1:5.1f}%")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": f1
        })

        if f1 > best_val_f1:
            best_val_f1 = f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": float(val_acc),
                "val_f1": float(f1),
                "window_size": 10,
                "input_dim": 1280
            }, best_ckpt_path)

    # 4. Final Evaluation on Held-Out Test Subjects [2, 11]
    print("\n--- Evaluating Best Model on Test Subjects [2, 11] ---")
    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    model.eval()
    test_preds, test_targets, test_probs = [], [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            logits, _ = model(bx)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            test_probs.extend(probs)
            test_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            test_targets.extend(by.numpy())

    test_acc = accuracy_score(test_targets, test_preds) * 100.0
    t_prec, t_rec, t_f1, _ = precision_recall_fscore_support(test_targets, test_preds, average="binary", zero_division=0)
    try:
        t_auc = roc_auc_score(test_targets, test_probs) * 100.0
    except Exception:
        t_auc = 0.0

    print(f"[Test Results] Accuracy: {test_acc:.2f}% | Precision: {t_prec*100:.2f}% | Recall: {t_rec*100:.2f}% | F1: {t_f1*100:.2f}% | ROC-AUC: {t_auc:.2f}%")

    # 5. Step 22: Temporal Moving Average Smoothing Demonstration
    smoother = TemporalMovingAverageSmoother(window_size=3)
    raw_probs = np.array(test_probs)
    smoothed_probs = smoother.smooth_sequence(raw_probs)

    # Measure jitter (average absolute difference between consecutive predictions)
    raw_jitter = np.mean(np.abs(np.diff(raw_probs))) if len(raw_probs) > 1 else 0.0
    smooth_jitter = np.mean(np.abs(np.diff(smoothed_probs))) if len(smoothed_probs) > 1 else 0.0
    jitter_reduction = ((raw_jitter - smooth_jitter) / (raw_jitter + 1e-6)) * 100.0

    print(f"[Step 22 Smoothing] Raw Prediction Jitter: {raw_jitter:.4f} -> Smoothed Jitter: {smooth_jitter:.4f} ({jitter_reduction:.1f}% reduction in flicker!)")

    # Plot raw vs smoothed probabilities
    plt.figure(figsize=(10, 4.5))
    plt.plot(raw_probs, label="Raw Frame-by-Frame Stress Prob", color="coral", alpha=0.6, linestyle="--", marker="o", markersize=3)
    plt.plot(smoothed_probs, label="Smoothed Stress Signal (Moving Avg K=3)", color="royalblue", linewidth=2.5)
    plt.axhline(0.5, color="gray", linestyle=":", label="Decision Threshold (0.5)")
    plt.title("Step 22: Temporal Moving-Average Stress Signal Smoothing", fontsize=12, fontweight="bold")
    plt.xlabel("Consecutive Sliding Window Index", fontsize=10)
    plt.ylabel("P(Stress State | X)", fontsize=10)
    plt.ylim(-0.05, 1.05)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    smoothing_plot_path = os.path.join(TEMPORAL_RESULTS_DIR, "temporal_smoothing_comparison.png")
    plt.savefig(smoothing_plot_path, dpi=300)
    plt.close()
    print(f"[Saved Plot] {smoothing_plot_path}")

    # 6. Save Artifacts & Summary
    pd.DataFrame(history).to_csv(os.path.join(config.LOGS_DIR, "temporal_training_history.csv"), index=False)

    summary = {
        "best_val_f1": float(best_val_f1),
        "test_accuracy": float(test_acc),
        "test_precision": float(t_prec * 100.0),
        "test_recall": float(t_rec * 100.0),
        "test_f1": float(t_f1 * 100.0),
        "test_roc_auc": float(t_auc),
        "raw_jitter": float(raw_jitter),
        "smoothed_jitter": float(smooth_jitter),
        "jitter_reduction_percent": float(jitter_reduction),
        "window_size": int(10),
        "smoothing_k": int(3),
        "train_windows": int(len(train_x)),
        "val_windows": int(len(val_x)),
        "test_windows": int(len(test_x)),
    }

    summary_path = os.path.join(TEMPORAL_RESULTS_DIR, "temporal_evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 65)
    print(f"[SUCCESS] Phase 3 Temporal Stress Inference Complete!")
    print(f"Checkpoint saved to: {best_ckpt_path}")
    print(f"Summary saved to   : {summary_path}")
    print("=" * 65)
    return summary


if __name__ == "__main__":
    train_temporal_model(epochs=15, batch_size=16, lr=1e-3, hidden_dim=128)

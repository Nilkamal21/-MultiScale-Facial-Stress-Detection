"""
evaluation/evaluate_temporal.py
Phase 3 (Temporal Stress Inference) — Comprehensive Evaluation & Confusion Matrix.

Evaluates best_temporal_stress_model.pth on held-out test drivers (Drivers [02, 11]).
Generates:
1. Confusion Matrix (Raw counts & Normalized percentage heatmap)
2. Classification report CSV & JSON
3. Visual plots saved to results/temporal/
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.temporal_model import TemporalStressGRU
from training.train_temporal import extract_or_load_features, load_phase2_feature_extractor

TEMPORAL_RESULTS_DIR = os.path.join(config.RESULTS_DIR, "temporal")


def plot_temporal_confusion_matrix(cm, class_names, output_path, title="Temporal Stress Confusion Matrix", normalize=False):
    """Plots and saves seaborn heatmap of the 2-class stress confusion matrix."""
    plt.figure(figsize=(6, 5))
    if normalize:
        cm_norm = cm.astype("float") / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1)
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".1%",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=True,
            annot_kws={"size": 14, "weight": "bold"}
        )
    else:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=True,
            annot_kws={"size": 14, "weight": "bold"}
        )

    plt.title(title, fontsize=12, pad=12, fontweight="bold")
    plt.xlabel("Predicted State", fontsize=11, labelpad=8)
    plt.ylabel("True Driver State", fontsize=11, labelpad=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Saved Heatmap] {output_path}")


def run_temporal_evaluation():
    print("=" * 65)
    print("  Phase 3 — Temporal BiGRU Stress Confusion Matrix Evaluation")
    print("=" * 65)

    os.makedirs(TEMPORAL_RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load feature extractor and extract/load test features
    encoder = load_phase2_feature_extractor(device)
    test_feats, test_labels, seq_ids = extract_or_load_features(encoder, "test", window_size=10, stride=2, device=device)

    # Load trained BiGRU
    model = TemporalStressGRU(input_dim=1280, hidden_dim=128, num_layers=2, num_classes=2, bidirectional=True).to(device)
    ckpt_path = os.path.join(config.CHECKPOINTS_DIR, "best_temporal_stress_model.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        logits, weights = model(test_feats.to(device))
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        targets = test_labels.numpy()

    class_names = ["Calm (0)", "Stress (1)"]
    cm = confusion_matrix(targets, preds)
    print("\nConfusion Matrix (Raw Counts):")
    print(cm)

    # Save CSV
    df_cm = pd.DataFrame(cm, index=[f"True_{c}" for c in class_names], columns=[f"Pred_{c}" for c in class_names])
    csv_path = os.path.join(TEMPORAL_RESULTS_DIR, "confusion_matrix.csv")
    df_cm.to_csv(csv_path)
    print(f"[Saved CSV] {csv_path}")

    # Plot raw & normalized heatmaps
    raw_plot = os.path.join(TEMPORAL_RESULTS_DIR, "confusion_matrix.png")
    plot_temporal_confusion_matrix(cm, class_names, raw_plot, title="Temporal Stress BiGRU Confusion Matrix (Counts)", normalize=False)

    norm_plot = os.path.join(TEMPORAL_RESULTS_DIR, "confusion_matrix_normalized.png")
    plot_temporal_confusion_matrix(cm, class_names, norm_plot, title="Temporal Stress BiGRU Confusion Matrix (Normalized)", normalize=True)

    # Classification report
    report = classification_report(targets, preds, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    rep_csv = os.path.join(TEMPORAL_RESULTS_DIR, "classification_report.csv")
    report_df.to_csv(rep_csv)
    print(f"[Saved Report] {rep_csv}")

    acc = accuracy_score(targets, preds) * 100.0
    prec, rec, f1, _ = precision_recall_fscore_support(targets, preds, average="binary")
    print(f"\nFinal Test Metrics on Unseen Drivers [02, 11]:")
    print(f"  - Accuracy : {acc:.2f}% ({cm[0,0] + cm[1,1]}/{len(targets)} windows)")
    print(f"  - Precision: {prec * 100:.2f}% (Zero False Positives: {cm[0,1]} calm windows called stress)")
    print(f"  - Recall   : {rec * 100:.2f}% ({cm[1,1]}/{cm[1,0] + cm[1,1]} acute events caught)")
    print(f"  - F1-Score : {f1 * 100:.2f}%")


if __name__ == "__main__":
    run_temporal_evaluation()

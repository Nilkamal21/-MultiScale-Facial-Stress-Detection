"""
evaluation/evaluate_temporal.py
Phase 3 (Temporal Stress Inference) — Comprehensive Evaluation & Confusion Matrix on UBFC.

Evaluates best_temporal_stress_model.pth on held-out test subjects (subjects 38–42).
Generates:
1. Confusion Matrix (Raw counts & Normalized percentage heatmap)
2. Classification report CSV & JSON
3. Visual plots saved to results/temporal/
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support, roc_auc_score

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
    plt.ylabel("True Subject State", fontsize=11, labelpad=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Saved Heatmap] {output_path}")


def run_temporal_evaluation(data_dir=None, ckpt_path=None, phase2_ckpt=None, threshold=None, window_size=10, stride=4):
    args_thresh = threshold
    print("=" * 65)
    print("  Phase 3 — Temporal BiGRU Stress Confusion Matrix Evaluation")
    print("=" * 65)

    os.makedirs(TEMPORAL_RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load feature extractor and extract/load test features
    encoder = load_phase2_feature_extractor(ckpt_path=phase2_ckpt, device=device)
    test_feats, test_labels, subjs = extract_or_load_features(
        encoder, data_dir, "test", window_size=window_size, stride=stride, device=device
    )

    # Candidate checkpoints
    candidate_ckpts = []
    if ckpt_path:
        candidate_ckpts.append(ckpt_path)
    candidate_ckpts.extend([
        os.path.join(config.CHECKPOINTS_DIR, "best_temporal_stress_model.pth"),
        "/kaggle/working/Stress-Detection/models_checkpoints/best_temporal_stress_model.pth",
        "/kaggle/working/models_checkpoints/best_temporal_stress_model.pth",
    ])

    final_ckpt = None
    for p in candidate_ckpts:
        if os.path.exists(p):
            final_ckpt = p
            break

    if final_ckpt is None:
        raise FileNotFoundError(f"Missing temporal model checkpoint: {candidate_ckpts}")

    print(f"[Model Checkpoint] Loading: {final_ckpt}")
    ckpt = torch.load(final_ckpt, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    input_dim = ckpt.get("input_dim", test_feats.shape[-1])
    
    # Auto-detect hidden_dim from GRU weights
    if "gru.weight_ih_l0" in state_dict:
        hidden_dim = state_dict["gru.weight_ih_l0"].shape[0] // 3
    else:
        hidden_dim = ckpt.get("hidden_dim", 64)

    print(f"[Model Spec] Input Dim: {input_dim} | Hidden Dim: {hidden_dim} (BiGRU)")
    model = TemporalStressGRU(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=2, num_classes=2, bidirectional=True).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # 1. Validation Threshold Calibration (Step 21 Methodological Rule)
    # The guide states: "Importantly, the threshold should be selected using validation data rather than arbitrarily choosing 0.5."
    val_feats, val_labels, _ = extract_or_load_features(
        encoder, data_dir, "val", window_size=window_size, stride=stride, device=device
    )

    with torch.no_grad():
        val_logits, _ = model(val_feats.to(device))
        val_probs = torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()
        val_targets = val_labels.numpy()

    best_thresh = 0.50
    best_bal_acc = 0.0
    for th in np.arange(0.20, 0.85, 0.02):
        v_pred = (val_probs >= th).astype(int)
        cm_v = confusion_matrix(val_targets, v_pred)
        if cm_v.shape == (2, 2):
            tn, fp, fn, tp = cm_v.ravel()
            sens = tp / (tp + fn + 1e-6)
            spec = tn / (tn + fp + 1e-6)
            bal = (sens + spec) / 2.0
            if bal > best_bal_acc:
                best_bal_acc = bal
                best_thresh = round(float(th), 2)

    selected_thresh = args_thresh if args_thresh is not None else best_thresh
    print(f"\n[Threshold Calibration] Validation Optimal Threshold: {best_thresh:.2f} (Balanced Acc: {best_bal_acc*100:.1f}%)")
    print(f"[Decision Rule] Applying Threshold tau = {selected_thresh:.2f} to Held-Out Test Subjects")

    with torch.no_grad():
        logits, weights = model(test_feats.to(device))
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = (probs >= selected_thresh).astype(int)
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
    plot_temporal_confusion_matrix(cm, class_names, raw_plot, title=f"Temporal Stress Confusion Matrix (tau={selected_thresh:.2f})", normalize=False)

    norm_plot = os.path.join(TEMPORAL_RESULTS_DIR, "confusion_matrix_normalized.png")
    plot_temporal_confusion_matrix(cm, class_names, norm_plot, title=f"Temporal Stress Confusion Matrix Normalized (tau={selected_thresh:.2f})", normalize=True)

    # Classification report
    report = classification_report(targets, preds, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    rep_csv = os.path.join(TEMPORAL_RESULTS_DIR, "classification_report.csv")
    report_df.to_csv(rep_csv)
    print(f"[Saved Report] {rep_csv}")

    acc = accuracy_score(targets, preds) * 100.0
    prec, rec, f1, _ = precision_recall_fscore_support(targets, preds, average="binary", zero_division=0)
    spec = float(cm[0, 0]) / max(float(cm[0, 0] + cm[0, 1]), 1.0) * 100.0
    try:
        auc = roc_auc_score(targets, probs) * 100.0
    except Exception:
        auc = 0.0

    print(f"\nFinal Test Metrics on Unseen UBFC Subjects (38–42):")
    print(f"  - Accuracy   : {acc:.2f}% ({cm[0,0] + cm[1,1]}/{len(targets)} windows)")
    print(f"  - Specificity: {spec:.2f}% ({cm[0,0]}/{cm[0,0] + cm[0,1]} Calm windows correctly identified)")
    print(f"  - Sensitivity: {rec * 100:.2f}% ({cm[1,1]}/{cm[1,0] + cm[1,1]} Stress windows correctly identified)")
    print(f"  - Precision  : {prec * 100:.2f}%")
    print(f"  - F1-Score   : {f1 * 100:.2f}%")
    print(f"  - ROC-AUC    : {auc:.2f}%")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Temporal Stress BiGRU")
    parser.add_argument("--data_dir", type=str, default="/kaggle/working/ubfc_faces")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--phase2_ckpt", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold (default: auto-calibrate on val)")
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--stride", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_temporal_evaluation(
        data_dir=args.data_dir,
        ckpt_path=args.checkpoint,
        phase2_ckpt=args.phase2_ckpt,
        threshold=args.threshold,
        window_size=args.window_size,
        stride=args.stride
    )

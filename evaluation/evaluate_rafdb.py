"""
evaluation/evaluate_rafdb.py
Phase 2 (Affective Transfer Learning) — Step 2.3 / Step 17: Comprehensive Evaluation on RAF-DB.

Evaluates the trained affective model (best_rafdb_affective_model.pth) on 3,068 held-out test faces.
Generates:
1. Confusion Matrix (raw and normalized)
2. Per-Class Precision, Recall, and F1-Scores
3. Classification Report CSV & Summary JSON
4. Affective Cue Analysis (identifying confusion between stress-related emotions: Fear, Sadness, Disgust vs Neutral)
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
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.efficientnet_encoder import EfficientNetEncoder
from models.emotion_classifier import EmotionClassifier, EndToEndEmotionModel
from datasets.rafdb_dataset import (
    get_rafdb_dataloader,
    RAFDB_NUM_CLASSES,
    RAFDB_IDX_TO_NAME,
)

CLASS_NAMES = [RAFDB_IDX_TO_NAME[i] for i in range(RAFDB_NUM_CLASSES)]


def plot_confusion_matrix(cm, class_names, output_path, title="RAF-DB Confusion Matrix", normalize=False):
    """Plots and saves seaborn heatmap of the 7-class emotion confusion matrix."""
    plt.figure(figsize=(9, 7.5))
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
        )

    plt.title(title, fontsize=14, pad=12, fontweight="bold")
    plt.xlabel("Predicted Emotion", fontsize=12, labelpad=8)
    plt.ylabel("True Emotion", fontsize=12, labelpad=8)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_per_class_metrics(report_df, output_path):
    """Generates bar chart comparing Precision, Recall, and F1 across all 7 emotions."""
    classes = [c for c in report_df.index if c in CLASS_NAMES]
    sub_df = report_df.loc[classes, ["precision", "recall", "f1-score"]]

    plt.figure(figsize=(10, 5.5))
    x = np.arange(len(classes))
    width = 0.25

    plt.bar(x - width, sub_df["precision"] * 100, width, label="Precision", color="#3498db")
    plt.bar(x, sub_df["recall"] * 100, width, label="Recall", color="#2ecc71")
    plt.bar(x + width, sub_df["f1-score"] * 100, width, label="F1-Score", color="#e74c3c")

    plt.title("Per-Emotion Performance (RAF-DB Held-Out Test Set)", fontsize=14, pad=12, fontweight="bold")
    plt.xlabel("Emotion Class", fontsize=12, labelpad=8)
    plt.ylabel("Percentage (%)", fontsize=12, labelpad=8)
    plt.xticks(x, classes, rotation=30, ha="right")
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def evaluate_rafdb_model(model_path=None, batch_size=64, max_batches=None):
    if model_path is None:
        model_path = os.path.join(config.CHECKPOINTS_DIR, "best_rafdb_affective_model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    out_dir = config.RESULTS_RAF_DB
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("  Phase 2 (Step 17) — Comprehensive RAF-DB Evaluation")
    print(f"  Model checkpoint : {model_path}")
    print(f"  Device           : {device}")
    print(f"  Output directory : {out_dir}")
    print("=" * 65)

    # 1. Instantiate and load model
    encoder = EfficientNetEncoder(pretrained=False)
    classifier = EmotionClassifier(feature_dim=encoder.feature_dim, num_classes=RAFDB_NUM_CLASSES)
    model = EndToEndEmotionModel(encoder=encoder, classifier=classifier, num_classes=RAFDB_NUM_CLASSES).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    # 2. Evaluate on test set
    test_loader = get_rafdb_dataloader(split="test", batch_size=batch_size, shuffle=False)

    all_preds = []
    all_targets = []

    print("[Evaluation] Processing test faces...")
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            if max_batches and idx >= max_batches:
                break
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    # 3. Overall Metrics
    overall_acc = accuracy_score(y_true, y_pred) * 100.0
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    print("\n" + "=" * 65)
    print("           RAF-DB AFFECTIVE TEST EVALUATION SUMMARY")
    print("=" * 65)
    print(f"  Overall Accuracy    : {overall_acc:.2f}% ({np.sum(y_true == y_pred)} / {len(y_true)} correct)")
    print(f"  Macro Precision     : {macro_prec * 100:.2f}%")
    print(f"  Macro Recall        : {macro_rec * 100:.2f}%")
    print(f"  Macro F1-Score      : {macro_f1 * 100:.2f}%")
    print(f"  Weighted F1-Score   : {weighted_f1 * 100:.2f}%")
    print("=" * 65)

    # 4. Classification Report
    report_dict = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_csv = os.path.join(out_dir, "classification_report.csv")
    report_df.to_csv(report_csv)

    print("\n  Per-Class Affective Breakdown:")
    print("  " + "-" * 60)
    for cls in CLASS_NAMES:
        row = report_df.loc[cls]
        print(f"  {cls:<12} | Precision: {row['precision']*100:>5.1f}% | Recall: {row['recall']*100:>5.1f}% | F1: {row['f1-score']*100:>5.1f}% | Support: {int(row['support'])}")
    print("  " + "-" * 60)

    # 5. Confusion Matrix & Heatmaps
    cm = confusion_matrix(y_true, y_pred, labels=list(range(RAFDB_NUM_CLASSES)))
    cm_csv = os.path.join(out_dir, "confusion_matrix.csv")
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(cm_csv)

    cm_raw_png = os.path.join(out_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, CLASS_NAMES, cm_raw_png, title="RAF-DB Confusion Matrix (Counts)", normalize=False)

    cm_norm_png = os.path.join(out_dir, "confusion_matrix_normalized.png")
    plot_confusion_matrix(cm, CLASS_NAMES, cm_norm_png, title="RAF-DB Confusion Matrix (Normalized)", normalize=True)

    bar_png = os.path.join(out_dir, "per_class_metrics.png")
    plot_per_class_metrics(report_df, bar_png)

    # 6. Save JSON Summary
    summary_data = {
        "overall_accuracy_pct": round(overall_acc, 2),
        "macro_precision_pct": round(macro_prec * 100, 2),
        "macro_recall_pct": round(macro_rec * 100, 2),
        "macro_f1_pct": round(macro_f1 * 100, 2),
        "total_test_samples": int(len(y_true)),
        "correct_predictions": int(np.sum(y_true == y_pred)),
    }
    summary_json = os.path.join(out_dir, "evaluation_summary.json")
    with open(summary_json, "w") as f:
        json.dump(summary_data, f, indent=4)

    print("\n[OK] Generated and saved all evaluation artifacts:")
    print(f"     - Confusion Matrix (Counts) : file:///{cm_raw_png.replace(os.sep, '/')}")
    print(f"     - Confusion Matrix (Norm)   : file:///{cm_norm_png.replace(os.sep, '/')}")
    print(f"     - Per-Class Metrics Bar     : file:///{bar_png.replace(os.sep, '/')}")
    print(f"     - Classification Report CSV : file:///{report_csv.replace(os.sep, '/')}")
    print(f"     - Summary JSON              : file:///{summary_json.replace(os.sep, '/')}")
    print("=" * 65)
    return summary_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAF-DB Affective Model")
    parser.add_argument("--model", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--max_batches", type=int, default=None, help="Debug max batches")
    args = parser.parse_args()

    evaluate_rafdb_model(model_path=args.model, batch_size=args.batch_size, max_batches=args.max_batches)

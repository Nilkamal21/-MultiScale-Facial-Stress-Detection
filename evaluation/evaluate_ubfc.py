"""
evaluation/evaluate_ubfc.py — Comprehensive Evaluation for Dual-Head BiGRU Model.

Evaluates trained checkpoint on held-out test subjects (UBFC subjects 38-42).
Calculates:
    1. Heart Rate (BPM) Regression:
       - MAE (Mean Absolute Error)
       - RMSE (Root Mean Squared Error)
       - Pearson Correlation Coefficient (r)
       - Error Bias (Mean Error)
    2. Stress Detection (Classification):
       - Accuracy, Precision, Recall, F1-score
       - 2x2 Confusion Matrix (Rest vs. Stress)
    3. Visualizations:
       - True vs Predicted BPM Scatter / Correlation Plot
       - Confusion Matrix Heatmap
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, f1_score
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.ubfc_stress_model import DualHeadUBFCStressModel


def evaluate_checkpoint(model_path=None, test_loader=None, device=None, results_dir=None, stress_threshold=None):
    """
    Evaluates a saved checkpoint on the provided test DataLoader.
    
    Args:
        model_path: Path to .pth checkpoint.
        test_loader: PyTorch DataLoader yielding (X, y_bpm, y_stress).
        device: torch.device.
        results_dir: Folder to save plots and evaluation metrics.
        stress_threshold: Probability threshold for positive stress detection (None = auto-load from checkpoint).
    
    Returns:
        metrics: dict of computed regression and classification metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_path is None:
        candidates = [
            os.path.join(config.KAGGLE_WORKING_DIR, "best_ubfc_stress_model.pth"),
            os.path.join(config.CHECKPOINTS_DIR, "best_ubfc_stress_model.pth"),
            "best_ubfc_stress_model.pth"
        ]
        for c in candidates:
            if os.path.exists(c):
                model_path = c
                break
        if model_path is None:
            model_path = candidates[0]
            
    if results_dir is None:
        results_dir = config.RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)

    print(f"[Evaluation] Loading model weights from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found at: {model_path}")

    # Initialize model architecture and load weights
    model = DualHeadUBFCStressModel(
        input_dim=config.INPUT_DIM,
        hidden_dim=config.GRU_HIDDEN_DIM,
        num_layers=config.GRU_NUM_LAYERS
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    calibrated_from_ckpt = config.STRESS_DECISION_THRESHOLD
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch_info = checkpoint.get("epoch", "N/A")
        calibrated_from_ckpt = checkpoint.get("best_threshold", config.STRESS_DECISION_THRESHOLD)
        print(f"[Evaluation] Loaded checkpoint saved from Epoch {epoch_info} (Optimal Val Threshold: {calibrated_from_ckpt:.2f})")
    else:
        model.load_state_dict(checkpoint)

    # Resolve active decision threshold
    if stress_threshold is None:
        active_threshold = float(calibrated_from_ckpt)
    else:
        active_threshold = float(stress_threshold)

    print(f"[Evaluation] Active Stress Decision Threshold: {active_threshold:.2f}")

    model.eval()

    all_true_bpm = []
    all_pred_bpm = []
    all_true_stress = []
    all_stress_probs = []

    print("[Evaluation] Running inference over test windows with progress bar...")
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing", unit="batch")
        for batch in pbar:
            X, y_bpm, y_stress = batch[0], batch[1], batch[2]
            X = X.to(device)
            pred_bpm, pred_stress_logits, _ = model(X)

            all_true_bpm.extend(y_bpm.cpu().numpy().tolist())
            all_pred_bpm.extend(pred_bpm.squeeze().cpu().numpy().tolist())
            all_true_stress.extend(y_stress.cpu().numpy().tolist())
            
            probs = torch.softmax(pred_stress_logits, dim=1)[:, 1]
            all_stress_probs.extend(probs.cpu().numpy().tolist())

    all_true_bpm = np.array(all_true_bpm, dtype=np.float32)
    all_pred_bpm = np.array(all_pred_bpm, dtype=np.float32)
    all_true_stress = np.array(all_true_stress, dtype=np.int64)
    all_stress_probs = np.array(all_stress_probs, dtype=np.float32)

    # 1. Heart Rate Metrics
    errors = all_pred_bpm - all_true_bpm
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mean_bias = float(np.mean(errors))
    
    if np.std(all_true_bpm) > 1e-4 and np.std(all_pred_bpm) > 1e-4:
        pearson_r = float(np.corrcoef(all_true_bpm, all_pred_bpm)[0, 1])
    else:
        pearson_r = 0.0

    # 2. Stress Detection Metrics at Active Threshold
    pred_classes = (all_stress_probs >= active_threshold).astype(int)
    acc = float(accuracy_score(all_true_stress, pred_classes)) * 100.0
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(all_true_stress, pred_classes, average="macro", zero_division=0)
    prec_per, rec_per, f1_per, _ = precision_recall_fscore_support(all_true_stress, pred_classes, average=None, zero_division=0)
    cm = confusion_matrix(all_true_stress, pred_classes)

    # Standard 0.50 Threshold Comparison
    preds_50 = (all_stress_probs >= 0.50).astype(int)
    f1_50 = float(f1_score(all_true_stress, preds_50, average="macro", zero_division=0)) * 100.0

    # Print Formatted Evaluation Report
    print("\n" + "=" * 65)
    print("           UBFC DUAL-HEAD TEST EVALUATION RESULTS")
    print("=" * 65)
    print("--- 1. HEART RATE (BPM) REGRESSION METRICS ---")
    print(f"  • Mean Absolute Error (MAE)   : {mae:.2f} BPM")
    print(f"  • Root Mean Squared Error (RMSE): {rmse:.2f} BPM")
    print(f"  • Pearson Correlation (r)       : {pearson_r:.4f}")
    print(f"  • Mean Bias (Pred - True)       : {mean_bias:+.2f} BPM")
    print("-" * 65)
    print(f"--- 2. STRESS DETECTION METRICS (Threshold = {active_threshold:.2f}) ---")
    print(f"  • Overall Accuracy              : {acc:.2f}%")
    print(f"  • Macro F1-Score                : {f1_macro * 100.0:.2f}% (vs {f1_50:.2f}% at standard 0.50 cutoff)")
    if len(prec_per) > 1:
        print(f"  • Rest (0)   -> Prec: {prec_per[0] * 100.0:.1f}%, Rec: {rec_per[0] * 100.0:.1f}%, F1: {f1_per[0] * 100.0:.1f}%")
        print(f"  • Stress (1) -> Prec: {prec_per[1] * 100.0:.1f}%, Rec: {rec_per[1] * 100.0:.1f}%, F1: {f1_per[1] * 100.0:.1f}%")
    print("  • Confusion Matrix [ [TN, FP], [FN, TP] ]:")
    print(f"    {cm.tolist()}")
    print("=" * 65)

    # Generate Publication-Quality 4-Panel Evaluation Plot
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        
        # Panel 1: True vs Predicted BPM Scatter Plot
        axes[0, 0].scatter(all_true_bpm, all_pred_bpm, alpha=0.5, color="royalblue", edgecolors="none", s=25)
        min_v = min(all_true_bpm.min(), all_pred_bpm.min()) - 5
        max_v = max(all_true_bpm.max(), all_pred_bpm.max()) + 5
        axes[0, 0].plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2, label="Ideal (y=x)")
        axes[0, 0].set_title(f"Heart Rate Estimation (MAE = {mae:.2f} BPM, r = {pearson_r:.3f})", fontsize=12, fontweight="bold")
        axes[0, 0].set_xlabel("Ground Truth Heart Rate (BPM)")
        axes[0, 0].set_ylabel("Predicted Heart Rate (BPM)")
        axes[0, 0].set_xlim([min_v, max_v])
        axes[0, 0].set_ylim([min_v, max_v])
        axes[0, 0].grid(True, linestyle="--", alpha=0.5)
        axes[0, 0].legend()

        # Panel 2: BPM Residual Error Histogram
        axes[0, 1].hist(errors, bins=30, color="teal", alpha=0.7, edgecolor="black")
        axes[0, 1].axvline(mean_bias, color="red", linestyle="--", linewidth=2, label=f"Bias: {mean_bias:+.2f} BPM")
        axes[0, 1].axvline(0, color="gray", linestyle="-", linewidth=1)
        axes[0, 1].set_title(f"BPM Error Distribution (RMSE = {rmse:.2f} BPM)", fontsize=12, fontweight="bold")
        axes[0, 1].set_xlabel("Prediction Error (BPM)")
        axes[0, 1].set_ylabel("Window Count")
        axes[0, 1].grid(True, linestyle="--", alpha=0.5)
        axes[0, 1].legend()

        # Panel 3: Confusion Matrix Heatmap
        cax = axes[1, 0].matshow(cm, cmap="Blues", alpha=0.85)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[1, 0].text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=16, fontweight="bold")
        axes[1, 0].set_title(f"Stress Confusion Matrix (Acc = {acc:.1f}%, F1 = {f1_macro * 100:.1f}%)", fontsize=12, fontweight="bold", pad=15)
        axes[1, 0].set_xticks([0, 1])
        axes[1, 0].set_yticks([0, 1])
        axes[1, 0].set_xticklabels(["Rest (0)", "Stress (1)"])
        axes[1, 0].set_yticklabels(["Rest (0)", "Stress (1)"])
        axes[1, 0].set_xlabel("Predicted Label")
        axes[1, 0].set_ylabel("True Label")

        # Panel 4: Precision, Recall, and Macro F1 vs Threshold Curve
        t_range = np.linspace(0.15, 0.85, 41)
        f1_curve, rec_curve, prec_curve = [], [], []
        for t in t_range:
            p_t = (all_stress_probs >= t).astype(int)
            f1_c = f1_score(all_true_stress, p_t, average="macro", zero_division=0)
            _, rec_c, _, _ = precision_recall_fscore_support(all_true_stress, p_t, average=None, zero_division=0)
            f1_curve.append(f1_c * 100.0)
            rec_curve.append(rec_c[1] * 100.0 if len(rec_c) > 1 else 0.0)
        axes[1, 1].plot(t_range, f1_curve, label="Macro F1", color="purple", linewidth=2.5)
        axes[1, 1].plot(t_range, rec_curve, label="Stress Recall", color="coral", linewidth=2, linestyle="--")
        axes[1, 1].axvline(active_threshold, color="blue", linestyle=":", label=f"Active ({active_threshold:.2f})")
        axes[1, 1].set_title("Decision Threshold Calibration Curve", fontsize=12, fontweight="bold")
        axes[1, 1].set_xlabel("Stress Decision Threshold")
        axes[1, 1].set_ylabel("Score (%)")
        axes[1, 1].grid(True, linestyle="--", alpha=0.5)
        axes[1, 1].legend()

        plt.tight_layout()
        plot_path = os.path.join(results_dir, "ubfc_evaluation_summary.png")
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"[Evaluation] Saved evaluation plots to: {plot_path}")
    except Exception as e:
        print(f"[WARN] Plot generation skipped: {e}")

    metrics = {
        "mae_bpm": mae,
        "rmse_bpm": rmse,
        "pearson_r": pearson_r,
        "mean_bias": mean_bias,
        "active_threshold": active_threshold,
        "stress_accuracy": acc,
        "stress_f1": f1_macro * 100.0,
        "confusion_matrix": cm.tolist()
    }
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate UBFC Stress Model")
    parser.add_argument("--mode", type=str, default="diagnostic", choices=["diagnostic", "eval"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--threshold", type=float, default=None, help="Stress probability threshold (default: auto from checkpoint)")
    args = parser.parse_args()

    if args.mode == "diagnostic":
        print("=" * 60)
        print("  Evaluating Synthetic Test Batch (Diagnostic Mode)")
        print("=" * 60)
        
        synth_X = torch.randn(32, 100, config.INPUT_DIM)
        synth_bpm = torch.empty(32).uniform_(60, 110)
        synth_stress = torch.randint(0, 2, (32,))
        test_ds = TensorDataset(synth_X, synth_bpm, synth_stress)
        test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

        metrics = evaluate_checkpoint(model_path=args.checkpoint, test_loader=test_loader, stress_threshold=args.threshold)
        print("  [SUCCESS] Diagnostic evaluation finished cleanly!")

    elif args.mode == "eval":
        from datasets.ubfc_dataset import get_ubfc_dataloader
        test_loader = get_ubfc_dataloader(split=args.split, shuffle=False)
        evaluate_checkpoint(model_path=args.checkpoint, test_loader=test_loader, stress_threshold=args.threshold)

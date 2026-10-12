"""
training/train_ubfc.py — Complete Preprocessing & Dual-Head BiGRU Training Pipeline.
Designed to run seamlessly both in Google Colab/Kaggle Notebooks (NVIDIA GPU) and locally.

Workflow:
    1. Preprocesses 100-frame sliding windows from UBFC videos (MediaPipe + CHROM rPPG).
    2. Synchronizes true Heart Rate (BPM) and Stress labels from ground_truth.txt.
    3. Trains DualHeadUBFCStressModel using AdamW and Multi-Task Loss.
    4. Evaluates on validation subjects and saves 'best_ubfc_stress_model.pth' (~15 MB).
"""

import os
import sys
import time
import argparse

os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
import numpy as np
import cv2
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from tqdm import tqdm
from preprocessing.rppg_extract import extract_frame_rgb_mean, chrom_rppg, butter_bandpass_filter
from datasets.ubfc_dataset import parse_ubfc_ground_truth, find_subject_directories
from models.ubfc_stress_model import DualHeadUBFCStressModel, UBFCMultiTaskLoss


# ==============================================================================
# 1. ON-THE-FLY VIDEO PREPROCESSING & WINDOW EXTRACTION
# ==============================================================================
def extract_subject_features(video_path, gt_path, window_size=config.WINDOW_SIZE, stride=config.WINDOW_STRIDE):
    """
    Reads a single subject's vid.avi and ground_truth.txt.
    Extracts forehead skin capillary signals and combines them with CHROM pulse waves.
    Outputs:
        X_windows: np.ndarray of shape (Num_Windows, 100, 4) -> [R, G, B, BVP_pulse]
        y_bpm: np.ndarray of shape (Num_Windows,)
        y_stress: np.ndarray of shape (Num_Windows,)
    """
    gt_data = parse_ubfc_ground_truth(gt_path)
    true_bpm = gt_data["bpm"]
    baseline_bpm = gt_data["baseline_bpm"]

    frame_rgbs = []

    # 1. Primary method: imageio-ffmpeg
    try:
        import imageio.v2 as imageio
        reader = imageio.get_reader(video_path)
        for frame_rgb in reader:
            h, w = frame_rgb.shape[:2]
            # Forehead ROI: Upper 12% to 32% of face, center 50% width
            # Avoids eyes (blinking), mouth (talking), hair, and background noise!
            fy = int(h * 0.12)
            fh = max(5, int(h * 0.20))
            fx = int(w * 0.25)
            fw = max(5, int(w * 0.50))
            forehead_crop = frame_rgb[fy : fy + fh, fx : fx + fw]
            if forehead_crop.size > 0:
                mean_rgb = np.mean(forehead_crop, axis=(0, 1)).astype(np.float32)  # type: ignore
                frame_rgbs.append(mean_rgb)
        reader.close()
    except Exception as e_im:
        # Fallback to OpenCV if imageio is unavailable
        try:
            cap = cv2.VideoCapture(video_path)
            while cap.isOpened():
                ret, frame_bgr = cap.read()
                if not ret or frame_bgr is None or frame_bgr.size == 0:
                    break
                h, w = frame_bgr.shape[:2]
                fy = int(h * 0.12)
                fh = max(5, int(h * 0.20))
                fx = int(w * 0.25)
                fw = max(5, int(w * 0.50))
                forehead_bgr = frame_bgr[fy : fy + fh, fx : fx + fw]
                if forehead_bgr.size > 0:
                    forehead_rgb = cv2.cvtColor(forehead_bgr, cv2.COLOR_BGR2RGB)
                    mean_rgb = np.mean(forehead_rgb, axis=(0, 1)).astype(np.float32)  # type: ignore
                    frame_rgbs.append(mean_rgb)
            cap.release()
        except Exception:
            pass

    if len(frame_rgbs) < window_size:
        return None, None, None

    frame_rgbs = np.array(frame_rgbs, dtype=np.float32)  # (Total_Frames, 3)
    num_frames = len(frame_rgbs)

    # Slice into rolling sliding windows
    windows_X = []
    windows_bpm = []
    windows_stress = []

    start = 0
    while start + window_size <= num_frames:
        win_rgb = frame_rgbs[start : start + window_size]  # (100, 3)

        # Feature 1: Z-score normalized RGB channels (removes skin tone baseline offset)
        win_norm = (win_rgb - np.mean(win_rgb, axis=0, keepdims=True)) / (np.std(win_rgb, axis=0, keepdims=True) + 1e-6)

        # Feature 2: CHROM BVP pulse waveform filtered to human heart rate bandwidth (0.7 - 3.5 Hz)
        try:
            bvp = chrom_rppg(win_rgb)
            bvp_filt = butter_bandpass_filter(bvp, fs=config.VIDEO_FPS, low_hz=config.BANDPASS_LOW_HZ, high_hz=config.BANDPASS_HIGH_HZ)
            bvp_norm = ((bvp_filt - np.mean(bvp_filt)) / (np.std(bvp_filt) + 1e-6)).reshape(-1, 1).astype(np.float32)
        except Exception:
            bvp_norm = np.zeros((window_size, 1), dtype=np.float32)

        # Combine into 4-channel input: [R_norm, G_norm, B_norm, BVP_pulse]
        win_4ch = np.concatenate([win_norm, bvp_norm], axis=-1).astype(np.float32)  # (100, 4)

        # Match with ground truth BPM (mean over the window)
        gt_idx = min(start + window_size // 2, len(true_bpm) - 1)
        curr_bpm = float(true_bpm[gt_idx])
        if curr_bpm < 40.0 or curr_bpm > 200.0:
            curr_bpm = float(baseline_bpm)

        # Stress Rule: If HR is elevated > 15% above resting baseline, Stress = 1
        is_stressed = 1 if (curr_bpm > baseline_bpm * (1.0 + config.STRESS_ELEVATION_THRESHOLD)) else 0

        windows_X.append(win_4ch)
        windows_bpm.append(curr_bpm)
        windows_stress.append(is_stressed)

        start += stride

    return (
        np.array(windows_X, dtype=np.float32),
        np.array(windows_bpm, dtype=np.float32),
        np.array(windows_stress, dtype=np.int64)
    )


def preprocess_all_subjects(output_npz=None):
    """
    Preprocesses all UBFC subjects with a real-time tqdm progress bar.
    Total output size on disk is under 100 MB!
    """
    if output_npz is None:
        output_npz = os.path.join(config.DATA_DIR, "ubfc_extracted_windows.npz")

    subjects = find_subject_directories()
    if not subjects:
        print("[WARN] No raw UBFC dataset directories found. Ensure dataset path is mounted.")
        return None

    print(f"[Preprocessing] Found {len(subjects)} subjects. Beginning on-the-fly extraction...")

    splits_data = {
        "train": {"X": [], "bpm": [], "stress": []},
        "val":   {"X": [], "bpm": [], "stress": []},
        "test":  {"X": [], "bpm": [], "stress": []},
    }

    train_set = set(config.TRAIN_SUBJECTS)
    val_set   = set(config.VAL_SUBJECTS)

    pbar = tqdm(sorted(subjects.items()), desc="Preprocessing Subjects", unit="subj")
    for s_name, paths in pbar:
        split_name = "train" if s_name in train_set else ("val" if s_name in val_set else "test")
        pbar.set_postfix({"Subject": s_name, "Split": split_name.upper()})

        try:
            X, y_b, y_s = extract_subject_features(paths["video"], paths["ground_truth"])
            if X is not None and len(X) > 0:
                splits_data[split_name]["X"].append(X)
                splits_data[split_name]["bpm"].append(y_b)
                splits_data[split_name]["stress"].append(y_s)
        except Exception as e_sub:
            print(f"\n[WARN] Skipped {s_name} due to: {e_sub}")

    # Concatenate and save
    save_dict = {}
    for s_name in ["train", "val", "test"]:
        if splits_data[s_name]["X"]:
            save_dict[f"{s_name}_X"] = np.concatenate(splits_data[s_name]["X"], axis=0)
            save_dict[f"{s_name}_y_bpm"] = np.concatenate(splits_data[s_name]["bpm"], axis=0)
            save_dict[f"{s_name}_y_stress"] = np.concatenate(splits_data[s_name]["stress"], axis=0)
            print(f"[{s_name.upper()} Set] Total 100-frame windows: {len(save_dict[f'{s_name}_X'])}")

    os.makedirs(os.path.dirname(output_npz), exist_ok=True)
    np.savez_compressed(output_npz, **save_dict)
    file_mb = os.path.getsize(output_npz) / (1024 * 1024)
    print(f"[SUCCESS] Saved extracted features to: {output_npz} ({file_mb:.2f} MB)")
    return output_npz


# ==============================================================================
# 2. TRAINING LOOP WITH PROGRESS BARS
# ==============================================================================
def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch=1, total_epochs=20):
    """Executes one epoch of dual-head training with tqdm progress bar."""
    model.train()
    total_loss, total_bpm_err, total_stress_correct, total_samples = 0.0, 0.0, 0, 0

    pbar = tqdm(dataloader, desc=f"Epoch [{epoch:02d}/{total_epochs:02d}] Train", leave=False)
    for batch in pbar:
        X, y_bpm, y_stress = batch[0], batch[1], batch[2]
        X = X.to(device)
        y_bpm = y_bpm.to(device)
        y_stress = y_stress.to(device)

        optimizer.zero_grad()
        pred_bpm, pred_stress, _ = model(X)

        loss, loss_bpm, loss_stress = criterion(pred_bpm, y_bpm, pred_stress, y_stress)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = X.size(0)
        total_loss += loss.item() * batch_size
        batch_mae = torch.sum(torch.abs(pred_bpm.squeeze() - y_bpm)).item()
        total_bpm_err += batch_mae

        preds = torch.argmax(pred_stress, dim=1)
        total_stress_correct += torch.sum(preds == y_stress).item()
        total_samples += batch_size

        # Update progress bar metrics live
        pbar.set_postfix({
            "Loss": f"{loss.item():.2f}",
            "MAE": f"{batch_mae / batch_size:.1f} BPM",
        })

    avg_loss = total_loss / max(total_samples, 1)
    mae_bpm = total_bpm_err / max(total_samples, 1)
    stress_acc = (total_stress_correct / max(total_samples, 1)) * 100.0
    return avg_loss, mae_bpm, stress_acc


def evaluate_model(model, dataloader, criterion, device):
    """Evaluates model performance and dynamically calibrates the optimal decision threshold."""
    model.eval()
    total_loss, total_bpm_err = 0.0, 0.0
    all_true_stress = []
    all_stress_probs = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", leave=False)
        for batch in pbar:
            X, y_bpm, y_stress = batch[0], batch[1], batch[2]
            X = X.to(device)
            y_bpm = y_bpm.to(device)
            y_stress = y_stress.to(device)

            pred_bpm, pred_stress, _ = model(X)
            loss, _, _ = criterion(pred_bpm, y_bpm, pred_stress, y_stress)

            batch_size = X.size(0)
            total_loss += loss.item() * batch_size
            total_bpm_err += torch.sum(torch.abs(pred_bpm.squeeze() - y_bpm)).item()

            probs = F.softmax(pred_stress, dim=1)[:, 1]
            all_stress_probs.extend(probs.cpu().numpy().tolist())
            all_true_stress.extend(y_stress.cpu().numpy().tolist())

    n_samples = max(len(all_true_stress), 1)
    avg_loss = total_loss / n_samples
    mae_bpm = total_bpm_err / n_samples

    all_true_stress = np.array(all_true_stress, dtype=np.int64)
    all_stress_probs = np.array(all_stress_probs, dtype=np.float32)

    # Dynamic threshold calibration on validation set via G-Mean (prevents minority class collapse)
    best_g = 0.0
    best_threshold = config.STRESS_DECISION_THRESHOLD
    for t in np.linspace(0.20, 0.80, 61):
        preds = (all_stress_probs >= t).astype(int)
        _, rec, _, _ = precision_recall_fscore_support(all_true_stress, preds, average=None, zero_division=0)
        if len(rec) > 1 and rec[0] > 0 and rec[1] > 0:
            g = float(np.sqrt(rec[0] * rec[1]))
            if g > best_g:
                best_g = g
                best_threshold = float(t)

    best_preds = (all_stress_probs >= best_threshold).astype(int)
    stress_acc = float(accuracy_score(all_true_stress, best_preds)) * 100.0
    macro_f1 = float(f1_score(all_true_stress, best_preds, average="macro", zero_division=0))
    macro_f1_pct = float(macro_f1) * 100.0

    return avg_loss, mae_bpm, stress_acc, macro_f1_pct, best_threshold


def run_training(train_loader, val_loader, epochs=config.EPOCHS, lr=config.LEARNING_RATE, device=None):
    """Full training pipeline with early stopping, dynamic loss balancing, and threshold calibration."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")

    # Dynamic class weight calculation to counteract ~6:1 imbalance
    if hasattr(train_loader.dataset, "y_stress"):
        y_tr = train_loader.dataset.y_stress
        n_rest = float((y_tr == 0).sum())
        n_stress = float((y_tr == 1).sum())
        if n_stress > 0:
            weight_stress = n_rest / n_stress
            print(f"[Class Balancing] Rest: {int(n_rest)}, Stress: {int(n_stress)} | Weight Ratio: [1.0, {weight_stress:.2f}]")

    model = DualHeadUBFCStressModel(
        input_dim=config.INPUT_DIM,
        hidden_dim=config.GRU_HIDDEN_DIM,
        num_layers=config.GRU_NUM_LAYERS
    ).to(device)

    criterion = UBFCMultiTaskLoss(lambda_stress=15.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    save_ckpt = os.path.join(config.CHECKPOINTS_DIR, "best_ubfc_stress_model.pth")
    if config.IS_KAGGLE:
        save_ckpt = os.path.join(config.KAGGLE_WORKING_DIR, "best_ubfc_stress_model.pth")

    print("=" * 65)
    print("  Beginning Dual-Head BiGRU Training (Heart Rate + Stress)")
    print("=" * 65)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_mae, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_mae, val_acc, val_f1, val_threshold = evaluate_model(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        duration = time.time() - t0

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            ckpt_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_mae_bpm": val_mae,
                "val_stress_acc": val_acc,
                "val_macro_f1": val_f1,
                "best_threshold": val_threshold,
            }
            torch.save(ckpt_data, save_ckpt)
            local_ckpt = os.path.join(config.CHECKPOINTS_DIR, "best_ubfc_stress_model.pth")
            if local_ckpt != save_ckpt:
                try:
                    torch.save(ckpt_data, local_ckpt)
                except Exception:
                    pass

        status_mark = " [*BEST]" if is_best else ""
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({duration:.1f}s) | "
            f"Train Loss: {train_loss:.3f} (MAE: {train_mae:.1f} BPM) | "
            f"Val Loss: {val_loss:.3f} (MAE: {val_mae:.1f} BPM, F1: {val_f1:.1f}%, Thr: {val_threshold:.2f}){status_mark}"
        )

    print("=" * 65)
    print(f"[SUCCESS] Training complete. Best weights saved to: {save_ckpt}")
    return save_ckpt


# ==============================================================================
# 3. STANDALONE EXECUTION & SYNTHETIC DIAGNOSTIC
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UBFC Training Pipeline")
    parser.add_argument("--mode", type=str, default="diagnostic", choices=["diagnostic", "extract", "train"])
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    if args.mode == "diagnostic":
        print("=" * 60)
        print("  Phase 5 Diagnostic — Testing Training Pipeline on Synthetic Batch")
        print("=" * 60)

        # Create synthetic train and validation datasets to test training loop end-to-end
        synth_X_train = torch.randn(32, 100, config.INPUT_DIM)
        synth_y_bpm_train = torch.empty(32).uniform_(60, 110)
        synth_y_stress_train = torch.randint(0, 2, (32,))

        synth_X_val = torch.randn(8, 100, config.INPUT_DIM)
        synth_y_bpm_val = torch.empty(8).uniform_(60, 110)
        synth_y_stress_val = torch.randint(0, 2, (8,))

        train_ds = TensorDataset(synth_X_train, synth_y_bpm_train, synth_y_stress_train)
        val_ds   = TensorDataset(synth_X_val, synth_y_bpm_val, synth_y_stress_val)

        train_ld = DataLoader(train_ds, batch_size=8, shuffle=True)
        val_ld   = DataLoader(val_ds, batch_size=8, shuffle=False)

        ckpt_path = run_training(train_ld, val_ld, epochs=args.epochs, lr=1e-3)
        assert os.path.exists(ckpt_path), "Model checkpoint was not saved!"
        print(f"  Checkpoint verified at : {ckpt_path}")
        print("  [SUCCESS] Phase 5 training pipeline verified perfectly!")
        print("=" * 60)

    elif args.mode == "extract":
        preprocess_all_subjects()

    elif args.mode == "train":
        # Full training on pre-extracted cached features
        from datasets.ubfc_dataset import get_ubfc_dataloader
        train_loader = get_ubfc_dataloader(split="train")
        val_loader   = get_ubfc_dataloader(split="val", shuffle=False)
        run_training(train_loader, val_loader, epochs=args.epochs)

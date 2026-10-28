"""
training/train_rafdb_transfer.py
Phase 2 (Affective Transfer Learning) — Step 2.2: Transfer Learning with RAF-DB.

Adapts the Phase 1 Self-Supervised Backbone (self_supervised_backbone.pth) to 7-class
facial emotion recognition using Cross-Entropy Loss on RAF-DB.
Implements Stage A (frozen backbone warm-up) and Stage B (fine-tuning) to preserve
pretrained visual representations while learning fine-grained affect.
"""

import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.efficientnet_encoder import EfficientNetEncoder
from models.emotion_classifier import EmotionClassifier, EndToEndEmotionModel
from datasets.rafdb_dataset import (
    get_rafdb_dataloader,
    RAFDB_NUM_CLASSES,
    RAFDB_IDX_TO_NAME,
)


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch=1, epochs=5, max_batches=None):
    """Executes one training epoch on RAF-DB face batches with visible progress."""
    model.train()
    if not any(p.requires_grad for p in model.encoder.parameters()):
        model.encoder.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []
    total_batches = 0
    num_batches = len(dataloader) if max_batches is None else min(len(dataloader), max_batches)

    for idx, (images, labels) in enumerate(dataloader):
        if max_batches and idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.detach().cpu().numpy())
        total_batches += 1

        if (idx + 1) % 40 == 0 or (idx + 1) == num_batches:
            batch_acc = (np.array(all_preds[-128:]) == np.array(all_labels[-128:])).mean() * 100.0 if all_labels else 0.0
            print(f"  [Epoch {epoch}/{epochs}] Batch {idx + 1:3d}/{num_batches:3d} | Loss: {running_loss / total_batches:.4f} | Acc: {batch_acc:4.1f}%", flush=True)

    epoch_loss = running_loss / max(total_batches, 1)
    epoch_acc = accuracy_score(all_labels, all_preds) * 100.0 if all_labels else 0.0
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device, max_batches=None):
    """Evaluates the model on held-out test faces, reporting loss and accuracy."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    total_batches = 0

    with torch.no_grad():
        for idx, (images, labels) in enumerate(dataloader):
            if max_batches and idx >= max_batches:
                break

            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            total_batches += 1

    val_loss = running_loss / max(total_batches, 1)
    val_acc = accuracy_score(all_labels, all_preds) * 100.0 if all_labels else 0.0
    return val_loss, val_acc, all_labels, all_preds


def run_phase2_transfer(
    epochs=5,
    batch_size=64,
    lr=1e-3,
    unfreeze_blocks=0,
    use_class_weights=True,
    weight_power=0.5,
    rafdb_dir=None,
    max_batches=None
):
    print("=" * 65)
    print("  Phase 2 (Step 2.2) — RAF-DB Supervised Affective Transfer")
    print(f"  Target epochs    : {epochs}")
    print(f"  Batch size       : {batch_size}")
    print(f"  Learning rate    : {lr}")
    print(f"  Unfreeze blocks  : {unfreeze_blocks} (0 = Stage A frozen backbone)")
    print(f"  Class weights    : {use_class_weights} (power={weight_power})")
    print(f"  RAF-DB dir       : {rafdb_dir or 'Auto (Kaggle/data)'}")
    print(f"  Classes          : {RAFDB_NUM_CLASSES} (Surprise..Neutral)")
    print("=" * 65)

    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")

    # 1. Load the Phase 1 Self-Supervised Backbone
    backbone = EfficientNetEncoder(pretrained=False)
    phase1_checkpoint = os.path.join(config.CHECKPOINTS_DIR, "self_supervised_backbone.pth")

    if os.path.exists(phase1_checkpoint):
        print(f"[Checkpoint] Loading Phase 1 backbone: {phase1_checkpoint}")
        backbone.load_state_dict(torch.load(phase1_checkpoint, map_location=device))
        print("[OK] Successfully transferred self-supervised facial representations!")
    else:
        print("[WARN] Phase 1 backbone not found! Falling back to ImageNet weights.")
        backbone = EfficientNetEncoder(pretrained=True)

    # 2. Attach new 7-class Emotion Classifier Head
    classifier = EmotionClassifier(
        feature_dim=backbone.feature_dim,
        num_classes=RAFDB_NUM_CLASSES,
        hidden_dim=config.CLASSIFIER_HIDDEN_DIM,
        dropout_rate=config.CLASSIFIER_DROPOUT,
    )

    model = EndToEndEmotionModel(
        encoder=backbone,
        classifier=classifier,
        num_classes=RAFDB_NUM_CLASSES
    ).to(device)

    # 3. Layer Freezing & Stage B Loading Strategy
    best_model_path = os.path.join(config.CHECKPOINTS_DIR, "best_rafdb_affective_model.pth")
    best_test_acc = 0.0

    if unfreeze_blocks == 0:
        model.freeze_encoder()
        if hasattr(model.encoder, "spp") and model.encoder.spp is not None:
            for p in model.encoder.spp.parameters():
                p.requires_grad = True
        print("[Stage A] Backbone features frozen; 85-Region SPP Attention + Classifier Head trainable.")
    else:
        # Load Stage A trained weights first so head is already initialized!
        if os.path.exists(best_model_path):
            print(f"[Stage B] Loading Stage A trained weights: {best_model_path}")
            ckpt = torch.load(best_model_path, map_location=device)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)
            if "test_acc" in ckpt:
                best_test_acc = ckpt["test_acc"]
                print(f"[Stage B] Baseline Stage A Test Accuracy: {best_test_acc:.2f}%")

        model.encoder.unfreeze_last_n_blocks(unfreeze_blocks)
        print(f"[Stage B] Top {unfreeze_blocks} feature blocks unfrozen for fine-tuning.")

    trainable_p, total_p = 0, 0
    for p in model.parameters():
        total_p += p.numel()
        if p.requires_grad:
            trainable_p += p.numel()
    print(f"[Parameters] Total: {total_p:,} | Trainable: {trainable_p:,}")

    # 4. DataLoaders & Criterion
    train_root, test_root = None, None
    if rafdb_dir:
        if os.path.exists(os.path.join(rafdb_dir, "DATASET", "train")):
            train_root = os.path.join(rafdb_dir, "DATASET", "train")
            test_root = os.path.join(rafdb_dir, "DATASET", "test")
        elif os.path.exists(os.path.join(rafdb_dir, "train")):
            train_root = os.path.join(rafdb_dir, "train")
            test_root = os.path.join(rafdb_dir, "test")
        else:
            train_root = rafdb_dir
            test_root = rafdb_dir
    train_loader = get_rafdb_dataloader(split="train", root_dir=train_root, batch_size=batch_size, shuffle=True)
    test_loader = get_rafdb_dataloader(split="test", root_dir=test_root, batch_size=batch_size, shuffle=False)

    if use_class_weights:
        # Smoothed inverse class frequency weights to balance rare emotions without destabilizing gradients
        counts = [1290, 281, 717, 4772, 1982, 705, 2524]
        total_c = sum(counts)
        raw_weights = [(total_c / (len(counts) * c)) ** weight_power for c in counts]
        mean_w = sum(raw_weights) / len(raw_weights)
        norm_weights = [w / mean_w for w in raw_weights]
        class_weights = torch.tensor(norm_weights, dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        weights_str = ", ".join([f"{w:.2f}" for w in norm_weights])
        print(f"[Weights] Applied smoothed class-frequency weights (power={weight_power}): [{weights_str}]")
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    # Differential learning rate for Stage B (gentle on backbone, standard on head)
    if unfreeze_blocks > 0:
        backbone_params = [p for p in model.encoder.parameters() if p.requires_grad]
        head_params = [p for p in model.classifier.parameters() if p.requires_grad]
        optimizer = optim.AdamW([
            {"params": backbone_params, "lr": lr * 0.1},
            {"params": head_params, "lr": lr}
        ], weight_decay=1e-4)
        print(f"[Optimizer] Differential LR: Backbone = {lr * 0.1:.6f} | Head = {lr:.6f}")
    else:
        optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    history = []
    best_model_path = os.path.join(config.CHECKPOINTS_DIR, "best_rafdb_affective_model.pth")
    history_csv = os.path.join(config.LOGS_DIR, "rafdb_transfer_history.csv")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        t_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch=epoch, epochs=epochs, max_batches=max_batches
        )
        test_loss, test_acc, _, _ = evaluate(
            model, test_loader, criterion, device, max_batches=max_batches
        )
        scheduler.step(test_acc)

        epoch_dur = time.time() - t_start
        is_best = test_acc > best_test_acc

        if is_best:
            best_test_acc = test_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "test_acc": test_acc,
                "test_loss": test_loss,
            }, best_model_path)

        star = " *" if is_best else ""
        print(
            f"  Epoch [{epoch:2d}/{epochs:2d}] | "
            f"Train Loss: {train_loss:.4f} (Acc: {train_acc:.2f}%) | "
            f"Test Loss: {test_loss:.4f} (Acc: {test_acc:.2f}%) | "
            f"Time: {epoch_dur:.1f}s{star}",
            flush=True
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "test_loss": round(test_loss, 4),
            "test_acc": round(test_acc, 2),
            "duration_s": round(epoch_dur, 2),
        })

    total_time = time.time() - start_time
    pd.DataFrame(history).to_csv(history_csv, index=False)

    print("=" * 65)
    print(f"[OK] RAF-DB transfer training completed in {total_time:.1f}s!")
    print(f"[OK] Best Test Accuracy         : {best_test_acc:.2f}%")
    print(f"[OK] Affective Model Checkpoint : file:///{best_model_path.replace(os.sep, '/')}")
    print(f"[OK] Training History CSV       : file:///{history_csv.replace(os.sep, '/')}")
    print("=" * 65)
    return best_test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 RAF-DB Transfer Learning")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--unfreeze_blocks", type=int, default=0, help="Number of encoder blocks to unfreeze")
    parser.add_argument("--no_class_weights", action="store_true", help="Disable class weights")
    parser.add_argument("--weight_power", type=float, default=0.5, help="Exponent for inverse class frequency (0.5=square root smoothed, 1.0=full inverse)")
    parser.add_argument("--rafdb_dir", type=str, default=None, help="Root folder of RAF-DB (containing train/test)")
    parser.add_argument("--max_batches", type=int, default=None, help="Debug max batches per epoch")
    args = parser.parse_args()

    run_phase2_transfer(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        unfreeze_blocks=args.unfreeze_blocks,
        use_class_weights=not args.no_class_weights,
        weight_power=args.weight_power,
        rafdb_dir=args.rafdb_dir,
        max_batches=args.max_batches,
    )


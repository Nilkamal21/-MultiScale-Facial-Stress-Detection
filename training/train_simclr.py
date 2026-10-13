"""
training/train_simclr.py
Phase 1 (Self-Supervised Pre-Training) — Step 1.3: Train SimCLR with NT-Xent Loss.

Trains the EfficientNet-B0 backbone on unlabeled facial images using self-supervised
contrastive learning. Pulls augmented views of the same face together while pushing
differing faces apart. Saves the pretrained backbone weights for downstream affective transfer.
"""

import os
import sys
import time
import argparse
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.simclr_model import SimCLRModel, NTXentLoss
from datasets.simclr_dataset import SimCLRDataset


def train_one_epoch(model, dataloader, criterion, optimizer, device, max_batches=None):
    """Executes one training epoch across SimCLR dual-view batches."""
    model.train()
    running_loss = 0.0
    total_batches = 0

    for idx, (x1, x2) in enumerate(dataloader):
        if max_batches and idx >= max_batches:
            break

        x1 = x1.to(device)
        x2 = x2.to(device)

        optimizer.zero_grad()
        z1, z2 = model(x1, x2)
        loss = criterion(z1, z2)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_batches += 1

    return running_loss / max(total_batches, 1)


def evaluate(model, dataloader, criterion, device, max_batches=None):
    """Computes mean NT-Xent contrastive loss over validation pairs."""
    model.eval()
    running_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for idx, (x1, x2) in enumerate(dataloader):
            if max_batches and idx >= max_batches:
                break

            x1 = x1.to(device)
            x2 = x2.to(device)

            z1, z2 = model(x1, x2)
            loss = criterion(z1, z2)

            running_loss += loss.item()
            total_batches += 1

    return running_loss / max(total_batches, 1)


def run_phase1_pretraining(
    epochs=5,
    batch_size=config.CONTRASTIVE_BATCH_SIZE,
    lr=config.CONTRASTIVE_LR,
    freeze_backbone=False,
    data_dir=None,
    max_batches=None
):
    print("=" * 65)
    print("  Phase 1 (Step 1.3) — SimCLR Self-Supervised Pre-Training")
    print(f"  Target epochs    : {epochs}")
    print(f"  Batch size       : {batch_size}")
    print(f"  Learning rate    : {lr}")
    print(f"  Freeze backbone  : {freeze_backbone}")
    print(f"  Data dir         : {data_dir or 'Auto (UBFC faces)'}")
    print(f"  Checkpoints dir  : {config.CHECKPOINTS_DIR}")
    print("=" * 65)

    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")

    # 1. Load full unlabeled dataset and split into train/val (85% / 15%)
    full_dataset = SimCLRDataset(data_dir=data_dir)
    total_samples = len(full_dataset)
    if total_samples < 10:
        raise RuntimeError(f"Insufficient images found for pre-training: {total_samples}")

    val_size = max(int(total_samples * 0.15), 2)
    train_size = total_samples - val_size

    generator = torch.Generator().manual_seed(config.RANDOM_SEED)
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, drop_last=True)

    print(f"[Dataset] Train samples: {train_size} | Validation samples: {val_size}")

    # 2. Instantiate SimCLR model, NT-Xent criterion, and optimizer
    model = SimCLRModel(freeze_backbone=freeze_backbone).to(device)
    criterion = NTXentLoss(temperature=config.CONTRASTIVE_TEMPERATURE).to(device)

    # In self-supervised pre-training, tune both backbone and projection head
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    history = []

    best_simclr_path = os.path.join(config.CHECKPOINTS_DIR, "best_simclr_model.pth")
    backbone_path = os.path.join(config.CHECKPOINTS_DIR, "self_supervised_backbone.pth")
    history_csv = os.path.join(config.LOGS_DIR, "simclr_training_history.csv")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        t_start = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, max_batches=max_batches)
        val_loss = evaluate(model, val_loader, criterion, device, max_batches=max_batches)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_dur = time.time() - t_start

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            # Save full SimCLR state (encoder + projection head)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, best_simclr_path)
            # Save standalone backbone weights for Phase 2 Transfer Learning
            torch.save(model.get_encoder().state_dict(), backbone_path)

        star = " *" if is_best else ""
        print(f"  Epoch [{epoch:2d}/{epochs:2d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f} | Time: {epoch_dur:.1f}s{star}", flush=True)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "lr": current_lr,
            "duration_s": round(epoch_dur, 2)
        })

    total_time = time.time() - start_time
    pd.DataFrame(history).to_csv(history_csv, index=False)

    print("=" * 65)
    print(f"[OK] Self-supervised pre-training completed in {total_time:.1f}s!")
    print(f"[OK] Best Validation Loss       : {best_val_loss:.4f}")
    print(f"[OK] Standalone Backbone Saved  : file:///{backbone_path.replace(os.sep, '/')}")
    print(f"[OK] Training History CSV       : file:///{history_csv.replace(os.sep, '/')}")
    print("=" * 65)
    return best_val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR Self-Supervised Pre-Training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=config.CONTRASTIVE_LR, help="Learning rate (default: 1e-4)")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze encoder backbone")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing facial images")
    parser.add_argument("--max_batches", type=int, default=None, help="Debug max batches per epoch")
    args = parser.parse_args()

    run_phase1_pretraining(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        freeze_backbone=args.freeze_backbone,
        data_dir=args.data_dir,
        max_batches=args.max_batches
    )

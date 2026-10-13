"""
datasets/temporal_dataset.py
Phase 3 (Temporal Stress Inference) — Steps 18 & 19: Temporal Sequence Dataset for UBFC.

Slices facial video frame sequences into sliding temporal windows of length W (default: 10 frames),
extracting continuous temporal tensors [W, C, H, W] for downstream recurrent modeling (Step 20).
Implements subject-independent splits (train: subjects 1-32, val: 33-37, test: 38-42)
and ground-truth temporal stress labeling:
    - Rest Baseline (t <= 30s / frame <= 180 at 6 FPS): Calm (Label 0)
    - Math Quiz Stress Task (t > 30s / frame > 180 at 6 FPS): Stress (Label 1)
"""

import os
import re
import sys
import glob
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Subject-independent splits established for 42 UBFC subjects
SUBJECT_SPLITS = {
    "train": [f"subject{i}" for i in range(1, 33)],
    "val": [f"subject{i}" for i in range(33, 38)],
    "test": [f"subject{i}" for i in range(38, 43)],
}


def natural_sort_key(s):
    """Sort strings containing numbers in human order (subject2 before subject10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]


def get_default_temporal_transform():
    """Deterministic 224x224 resize and ImageNet normalization for video frames."""
    return transforms.Compose([
        transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])


class UBFCTemporalDataset(Dataset):
    """
    Constructs sliding temporal windows over UBFC video face sequences.
    
    Each sample represents a continuous temporal window of W frames:
        X_t = [F_t, F_{t+1}, ..., F_{t+W-1}] of shape (W, 3, 224, 224)
    along with its binary stress label (0 = Calm baseline, 1 = Arithmetic stress task).
    """
    def __init__(
        self,
        data_dir=None,
        metadata_csv=None,
        split="train",
        window_size=10,
        stride=2,
        baseline_seconds=30.0,
        fps=6.0,
        transform=None,
    ):
        super().__init__()
        self.split = split
        self.window_size = window_size
        self.stride = stride
        self.baseline_seconds = baseline_seconds
        self.fps = fps
        self.transform = transform or get_default_temporal_transform()

        # Find data directory: /kaggle/working/ubfc_faces or fallback
        self.data_dir = data_dir
        if self.data_dir is None:
            candidates = [
                "/kaggle/working/ubfc_faces",
                os.path.join(config.DATA_DIR, "ubfc_faces"),
                os.path.join(config.PROJECT_ROOT, "ubfc_faces"),
            ]
            for c in candidates:
                if os.path.exists(c) and len(os.listdir(c)) > 0:
                    self.data_dir = c
                    break
            if self.data_dir is None:
                self.data_dir = candidates[0]

        # Allowed subjects for the split
        if split in SUBJECT_SPLITS:
            allowed_subjects = set(SUBJECT_SPLITS[split])
        elif split == "all":
            allowed_subjects = None
        else:
            raise ValueError(f"Invalid split: {split}. Choose from ['train', 'val', 'test', 'all'].")

        self.windows = []
        self._build_windows(allowed_subjects)

    def _build_windows(self, allowed_subjects):
        if not os.path.exists(self.data_dir):
            return

        # Find subject directories
        subj_dirs = []
        for d in sorted(os.listdir(self.data_dir), key=natural_sort_key):
            full_d = os.path.join(self.data_dir, d)
            if os.path.isdir(full_d):
                if allowed_subjects is None or d.lower() in [s.lower() for s in allowed_subjects]:
                    subj_dirs.append((d, full_d))

        baseline_frame_cutoff = int(self.baseline_seconds * self.fps)

        for subj_name, subj_path in subj_dirs:
            # Find all image frames
            frame_files = [
                f for f in os.listdir(subj_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            frame_files.sort(key=natural_sort_key)

            if len(frame_files) == 0:
                continue

            frame_paths = [os.path.join(subj_path, f) for f in frame_files]
            n_frames = len(frame_paths)

            if n_frames < self.window_size:
                # Pad sequence if shorter than window_size
                padded = frame_paths + [frame_paths[-1]] * (self.window_size - n_frames)
                label = 0  # Short initial sequence is baseline
                self.windows.append({
                    "subject_id": subj_name,
                    "paths": padded,
                    "label": label,
                    "start_idx": 0
                })
            else:
                # Slide window across frames
                start = 0
                while start + self.window_size <= n_frames:
                    window_paths = frame_paths[start: start + self.window_size]
                    mid_frame_idx = start + (self.window_size // 2)
                    # Stress label: 0 if baseline (t <= 30s), 1 if task (t > 30s)
                    label = 0 if mid_frame_idx <= baseline_frame_cutoff else 1

                    self.windows.append({
                        "subject_id": subj_name,
                        "paths": window_paths,
                        "label": label,
                        "start_idx": start
                    })
                    start += self.stride

                # Cover tail frames
                if (start - self.stride + self.window_size) < n_frames:
                    tail_paths = frame_paths[-self.window_size:]
                    mid_frame_idx = n_frames - (self.window_size // 2)
                    label = 0 if mid_frame_idx <= baseline_frame_cutoff else 1
                    self.windows.append({
                        "subject_id": subj_name,
                        "paths": tail_paths,
                        "label": label,
                        "start_idx": n_frames - self.window_size
                    })

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        item = self.windows[idx]
        image_paths = item["paths"]
        label = item["label"]

        frame_tensors = []
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            frame_tensors.append(img)

        # Shape: [W, C, H, W]
        temporal_tensor = torch.stack(frame_tensors, dim=0)
        return temporal_tensor, torch.tensor(label, dtype=torch.long), item["subject_id"]


# Alias for backward compatibility
KMUTemporalDataset = UBFCTemporalDataset


def get_temporal_dataloader(
    data_dir=None,
    split="train",
    window_size=10,
    stride=2,
    batch_size=8,
    shuffle=True,
    num_workers=0
):
    """Factory creating DataLoader for temporal video sequences."""
    dataset = UBFCTemporalDataset(
        data_dir=data_dir,
        split=split,
        window_size=window_size,
        stride=stride
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


if __name__ == "__main__":
    print("=" * 65)
    print("  Phase 3 — UBFC Temporal Sliding Window Dataset Diagnostic")
    print("=" * 65)
    for sp in ["train", "val", "test"]:
        ds = UBFCTemporalDataset(split=sp, window_size=10, stride=2)
        n_stress = sum(w["label"] for w in ds.windows)
        print(f"[{sp.upper():5s}] Windows: {len(ds):4d} | Stress (1): {n_stress:4d} | Calm (0): {len(ds)-n_stress:4d} | Subjects: {len(SUBJECT_SPLITS[sp])}")
    print("[OK] Dataset definition verified!")

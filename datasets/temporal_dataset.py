"""
datasets/temporal_dataset.py
Phase 3 (Temporal Stress Inference) — Steps 18 & 19: Temporal Sequence Dataset.

Slices KMU-FED driving video sequences into sliding temporal windows of length W (default: 10 frames),
extracting continuous temporal tensors [W, C, H, W] for downstream recurrent modeling (Step 20).
Implements subject-independent splits (train, val, test) and affective stress-cue mapping.
"""

import os
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

# Subject-independent splits established in KMU-FED pipeline
SUBJECT_SPLITS = {
    "train": [1, 3, 5, 6, 7, 9, 10, 12],
    "val": [4, 8],
    "test": [2, 11],
}

# Affective stress cue mapping (Methodological Rule - Step 14):
# High Affective Tension cues: Fear, Disgust, Anger -> 1
# Low Affective Tension / Baseline: Neutral, Happiness, Surprise -> 0
AFFECTIVE_TENSION_MAP = {
    "Anger": 1,
    "Disgust": 1,
    "Fear": 1,
    "Happiness": 0,
    "Neutral": 0,
    "Surprise": 0,
    "Sadness": 1,  # Negative valence / distress cue
}

EMOTION_NAME_TO_IDX = {
    "Surprise": 0,
    "Fear": 1,
    "Disgust": 2,
    "Happiness": 3,
    "Sadness": 4,
    "Anger": 5,
    "Neutral": 6,
}


def get_default_temporal_transform():
    """Deterministic 224x224 resize and ImageNet normalization for video frames."""
    return transforms.Compose([
        transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])


class KMUTemporalDataset(Dataset):
    """
    Constructs sliding temporal windows over KMU-FED video sequences.
    
    Each sample represents a continuous temporal window of W frames:
        X_t = [F_t, F_{t+1}, ..., F_{t+W-1}] of shape (W, 3, 224, 224)
    along with its affective tension / stress proxy label and metadata.
    """
    def __init__(
        self,
        metadata_csv=None,
        split="train",
        window_size=10,
        stride=2,
        transform=None,
    ):
        super().__init__()
        self.metadata_csv = metadata_csv or os.path.join(config.METADATA_DIR, "kmu_fed_processed.csv")
        self.split = split
        self.window_size = window_size
        self.stride = stride
        self.transform = transform or get_default_temporal_transform()

        if not os.path.exists(self.metadata_csv):
            raise FileNotFoundError(f"Missing processed metadata: {self.metadata_csv}")

        df = pd.read_csv(self.metadata_csv)

        # Filter by subject-independent split if specified
        if split in SUBJECT_SPLITS:
            allowed_subjects = set(SUBJECT_SPLITS[split])
            df = df[df["subject_id"].isin(allowed_subjects)].copy()
        elif split != "all":
            raise ValueError(f"Invalid split: {split}. Choose from ['train', 'val', 'test', 'all'].")

        # Sort chronologically by sequence and filename
        df = df.sort_values(by=["sequence_id", "filename"]).reset_index(drop=True)

        # Build sliding windows per sequence
        self.windows = []
        grouped = df.groupby("sequence_id")

        for seq_id, group in grouped:
            frames_paths = group["processed_path"].tolist()
            emotion = group["emotion"].iloc[0]
            subject_id = group["subject_id"].iloc[0]
            n_frames = len(frames_paths)

            if n_frames < self.window_size:
                # Sequence is shorter than window_size: pad by repeating last frame
                padded_paths = frames_paths + [frames_paths[-1]] * (self.window_size - n_frames)
                self.windows.append({
                    "sequence_id": seq_id,
                    "subject_id": subject_id,
                    "emotion": emotion,
                    "paths": padded_paths,
                    "start_idx": 0,
                    "is_padded": True,
                })
            else:
                # Slide window across sequence
                start = 0
                while start + self.window_size <= n_frames:
                    window_paths = frames_paths[start: start + self.window_size]
                    self.windows.append({
                        "sequence_id": seq_id,
                        "subject_id": subject_id,
                        "emotion": emotion,
                        "paths": window_paths,
                        "start_idx": start,
                        "is_padded": False,
                    })
                    start += self.stride

                # Ensure tail frames are covered if last stride did not reach end
                if (start - self.stride + self.window_size) < n_frames:
                    tail_paths = frames_paths[-self.window_size:]
                    self.windows.append({
                        "sequence_id": seq_id,
                        "subject_id": subject_id,
                        "emotion": emotion,
                        "paths": tail_paths,
                        "start_idx": n_frames - self.window_size,
                        "is_padded": False,
                    })

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        item = self.windows[idx]
        image_paths = item["paths"]
        emotion = item["emotion"]

        # Load and transform each frame in the temporal window
        frame_tensors = []
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            frame_tensors.append(img)

        # Stack into [W, C, H, W] tensor
        temporal_tensor = torch.stack(frame_tensors, dim=0)

        tension_label = AFFECTIVE_TENSION_MAP.get(emotion, 0)
        emotion_idx = EMOTION_NAME_TO_IDX.get(emotion, 6)

        return (
            temporal_tensor,
            torch.tensor(tension_label, dtype=torch.long),
            torch.tensor(emotion_idx, dtype=torch.long),
            item["sequence_id"],
        )


def get_temporal_dataloader(
    split="train",
    window_size=10,
    stride=2,
    batch_size=8,
    shuffle=True,
    num_workers=0
):
    """Factory creating DataLoader for temporal video sequences."""
    dataset = KMUTemporalDataset(
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


def run_temporal_dataset_test():
    """Standalone diagnostic verifying Step 3.1 temporal dataset functionality."""
    print("=" * 65)
    print("  Phase 3 (Step 3.1) — KMU-FED Temporal Sliding Window Dataset")
    print("=" * 65)

    for split in ["train", "val", "test"]:
        stride = 2 if split == "train" else 4
        ds = KMUTemporalDataset(split=split, window_size=10, stride=stride)
        n_tension = sum(AFFECTIVE_TENSION_MAP.get(w["emotion"], 0) for w in ds.windows)
        print(f"[{split.upper():5s}] Windows: {len(ds):3d} | High Tension: {n_tension:2d} | Low Tension: {len(ds)-n_tension:2d} | Subjects: {SUBJECT_SPLITS[split]}")

    # Inspect one sample batch
    train_loader = get_temporal_dataloader(split="train", batch_size=4, shuffle=True)
    batch_tensors, tension_labels, emotion_labels, seq_ids = next(iter(train_loader))

    print("-" * 65)
    print(f"[Batch Shape]     : {list(batch_tensors.shape)} -> [Batch, Window_Size, Channels, Height, Width]")
    print(f"[Tension Labels]  : {tension_labels.tolist()} (1=Tension/Stress Cue, 0=Calm/Baseline)")
    print(f"[Emotion Indices] : {emotion_labels.tolist()}")
    print(f"[Sample Sequences]: {list(seq_ids)}")
    print("=" * 65)
    print("[SUCCESS] Step 3.1 Temporal Sequence Dataset verified!")


if __name__ == "__main__":
    run_temporal_dataset_test()

"""
datasets/rafdb_dataset.py
Phase 2 (Affective Transfer Learning) — Step 2.1: RAF-DB Dataset & DataLoader.

Loads the Real-world Affective Faces Database (RAF-DB) with 7 basic emotion classes:
0: Surprise, 1: Fear, 2: Disgust, 3: Happiness, 4: Sadness, 5: Anger, 6: Neutral.
Resizes 100x100 faces to 224x224 and normalizes to match the Phase 1 backbone.
"""

import os
import sys
from PIL import Image, ImageDraw
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# RAF-DB 7 basic emotion classes (mapping 1-based folder ID to 0-based tensor index)
RAFDB_ID_TO_NAME = {
    1: "Surprise",
    2: "Fear",
    3: "Disgust",
    4: "Happiness",
    5: "Sadness",
    6: "Anger",
    7: "Neutral",
}

RAFDB_IDX_TO_NAME = {idx: RAFDB_ID_TO_NAME[idx + 1] for idx in range(7)}
RAFDB_NAME_TO_IDX = {name: idx for idx, name in RAFDB_IDX_TO_NAME.items()}
RAFDB_NUM_CLASSES = 7


def get_rafdb_transforms(split="train", input_size=config.INPUT_SIZE):
    """
    Returns image transformation pipeline for RAF-DB.
    Resizes from 100x100 to 224x224 to match the backbone architecture.
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ])


class RAFDBDataset(Dataset):
    """
    PyTorch Dataset for Real-world Affective Faces Database (RAF-DB).
    Supports 'train' (12,271 images) and 'test' (3,068 images) splits.
    """
    def __init__(self, split="train", root_dir=None, transform=None):
        super().__init__()
        self.split = split
        if root_dir is None:
            # 1. Search dynamically across /kaggle/input for the RAF-DB split
            if os.path.exists("/kaggle/input"):
                for root, dirs, _ in os.walk("/kaggle/input"):
                    if os.path.basename(root).lower() == split.lower():
                        # Verify it has numeric emotion folders (1..7)
                        if any(str(i) in dirs for i in range(1, 8)):
                            root_dir = root
                            break
                    elif "dataset" in os.path.basename(root).lower() and split in dirs:
                        candidate = os.path.join(root, split)
                        if any(str(i) in os.listdir(candidate) for i in range(1, 8) if os.path.isdir(os.path.join(candidate, str(i)))):
                            root_dir = candidate
                            break

            # 2. Check standard candidate directories
            if root_dir is None:
                candidates = [
                    f"/kaggle/input/raf-db-dataset/DATASET/{split}",
                    f"/kaggle/input/rafdb/DATASET/{split}",
                    f"/kaggle/input/raf-db/DATASET/{split}",
                    f"/kaggle/input/raf-db-dataset/raf-basic/{split}",
                    os.path.join(config.DATA_DIR, "DATASET", split),
                ]
                for c in candidates:
                    if os.path.exists(c):
                        root_dir = c
                        break

            if root_dir is None:
                root_dir = os.path.join(config.DATA_DIR, "DATASET", split)

        print(f"[RAF-DB Dataset] Using {split.upper()} directory: {root_dir}")
        self.root_dir = root_dir

        self.transform = transform if transform is not None else get_rafdb_transforms(split)

        self.samples = []  # list of tuples: (image_path, class_idx)
        self._load_samples()

    def _load_samples(self):
        """Scans the RAF-DB split folders (1..7) and records all valid images."""
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"RAF-DB directory not found at: {self.root_dir}")

        for folder_id in sorted(RAFDB_ID_TO_NAME.keys()):
            folder_path = os.path.join(self.root_dir, str(folder_id))
            if not os.path.exists(folder_path):
                continue

            class_idx = folder_id - 1  # 0-indexed for CrossEntropyLoss
            for fname in sorted(os.listdir(folder_path)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(folder_path, fname)
                    self.samples.append((img_path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        with Image.open(img_path) as img:
            img_rgb = img.convert("RGB")
            tensor = self.transform(img_rgb)
        return tensor, class_idx

    def get_class_counts(self):
        """Returns distribution count of each emotion class in this split."""
        counts = {RAFDB_IDX_TO_NAME[i]: 0 for i in range(RAFDB_NUM_CLASSES)}
        for _, class_idx in self.samples:
            counts[RAFDB_IDX_TO_NAME[class_idx]] += 1
        return counts


def get_rafdb_dataloader(split="train", root_dir=None, batch_size=32, shuffle=None, num_workers=0):
    """Factory function returning a DataLoader for RAF-DB training or evaluation."""
    if shuffle is None:
        shuffle = (split == "train")

    dataset = RAFDBDataset(split=split, root_dir=root_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=(split == "train")
    )


def generate_rafdb_preview(output_path=None):
    """
    Generates a 7-panel visual strip showing one sample face for each emotion class:
    [ Surprise | Fear | Disgust | Happiness | Sadness | Anger | Neutral ]
    """
    if output_path is None:
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        output_path = os.path.join(config.RESULTS_DIR, "rafdb_preview.jpg")

    dataset = RAFDBDataset(split="train", transform=transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)))
    
    # Collect one sample per class
    sample_images = {}
    for img_path, class_idx in dataset.samples:
        if class_idx not in sample_images:
            sample_images[class_idx] = Image.open(img_path).convert("RGB").resize((config.INPUT_SIZE, config.INPUT_SIZE))
        if len(sample_images) == RAFDB_NUM_CLASSES:
            break

    # Build canvas with 7 side-by-side panels
    w, h = config.INPUT_SIZE, config.INPUT_SIZE
    banner_h = 28
    total_w = w * RAFDB_NUM_CLASSES + (RAFDB_NUM_CLASSES + 1) * 10
    total_h = h + banner_h + 20

    canvas = Image.new("RGB", (total_w, total_h), color=(25, 25, 30))
    draw = ImageDraw.Draw(canvas)

    for class_idx in range(RAFDB_NUM_CLASSES):
        img_panel = sample_images[class_idx]
        class_name = RAFDB_IDX_TO_NAME[class_idx]
        x_pos = 10 + class_idx * (w + 10)

        canvas.paste(img_panel, (x_pos, banner_h + 10))
        draw.text((x_pos + 8, 7), f"{class_idx+1}. {class_name}", fill=(240, 240, 240))

    canvas.save(output_path, quality=95)
    return output_path


if __name__ == "__main__":
    print("=" * 65)
    print("  Phase 2 (Step 2.1) — RAF-DB Affective Dataset Verification")
    print("=" * 65)

    train_ds = RAFDBDataset(split="train")
    test_ds = RAFDBDataset(split="test")

    print(f"[OK] Training Split Loaded   : {len(train_ds):,} faces")
    print(f"[OK] Testing Split Loaded    : {len(test_ds):,} faces")
    print(f"[OK] Grand Total Faces       : {len(train_ds) + len(test_ds):,} faces")
    print("-" * 65)
    print("  Emotion Class Distribution (Training):")
    for name, cnt in train_ds.get_class_counts().items():
        print(f"    - {name:<12}: {cnt:>5} faces ({cnt/len(train_ds)*100:.1f}%)")
    print("-" * 65)

    # Test single batch fetch
    loader = get_rafdb_dataloader(split="train", batch_size=4)
    images, labels = next(iter(loader))
    print(f"[OK] Batch Tensor Shape      : {list(images.shape)} [Batch, Channels, H, W]")
    print(f"[OK] Batch Labels Shape      : {list(labels.shape)} (Values: {labels.tolist()})")

    # Generate 7-emotion preview
    preview_file = generate_rafdb_preview()
    print(f"[OK] Saved visual 7-emotion preview image to:")
    print(f"     file:///{preview_file.replace(os.sep, '/')}")
    print("=" * 65)
    print("Step 2.1 Verification Passed! RAF-DB is 100% prepared for Phase 2 training.")

"""
preprocessing/preprocessing.py
Phase 5: Resize aligned faces to 224x224, apply ImageNet normalization,
and provide PyTorch transform pipeline.
See explainable.md > Phase 5 for details.
"""

import os
import sys
import argparse
import pandas as pd
from PIL import Image
import torch  # type: ignore
from torchvision import transforms  # type: ignore


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def get_default_transform(input_size=config.INPUT_SIZE):
    """Return PyTorch transform pipeline for 224x224 ImageNet normalization."""
    return transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


def inverse_normalize(tensor):
    """Convert normalized PyTorch tensor (C, H, W) back to unnormalized (0-1) tensor."""
    mean = torch.tensor(config.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(config.IMAGENET_STD).view(3, 1, 1)
    return tensor * std + mean


def run_phase5(preview_only=False):
    print("=" * 55)
    print("  KMU-FED Phase 5 — Resize & Normalization")
    print(f"  Input aligned faces : {config.ALIGNED_FACES_DIR}")
    print(f"  Target processed    : {config.PROCESSED_DIR}")
    print(f"  Input size          : {config.INPUT_SIZE}x{config.INPUT_SIZE}")
    print("=" * 55)

    aligned_csv = os.path.join(config.METADATA_DIR, "kmu_fed_aligned_faces.csv")
    if not os.path.exists(aligned_csv):
        raise FileNotFoundError(f"Missing aligned faces metadata: {aligned_csv}. Run Phase 4 first.")

    df = pd.read_csv(aligned_csv)
    transform = get_default_transform()

    if preview_only:
        sample_path = df.iloc[0]["aligned_face_path"]
        if not os.path.exists(sample_path):
            sample_path = os.path.join(config.ALIGNED_FACES_DIR, str(df.iloc[0]["sequence_id"]), df.iloc[0]["filename"])
        
        with Image.open(sample_path) as img:
            img_rgb = img.convert("RGB")
            tensor = transform(img_rgb)

        print("\n--- [Preview Verification] ---")
        print(f"Sample file  : {sample_path}")
        print(f"Tensor shape : {tensor.shape}")
        print(f"Tensor dtype : {tensor.dtype}")
        print(f"Min value    : {tensor.min().item():.4f}")
        print(f"Max value    : {tensor.max().item():.4f}")
        print(f"Mean value   : {tensor.mean().item():.4f}")
        print(f"Std value    : {tensor.std().item():.4f}")
        print("Value range  : approx -2.1 to +2.6 (matches ImageNet specs)")
        print("--- [Preview Complete] ---\n")
        return

    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    processed_records = []
    total_faces = len(df)

    for idx, row in df.iterrows():
        filename = row["filename"]
        seq_id = str(row["sequence_id"])
        aligned_path = row["aligned_face_path"]

        if not os.path.exists(aligned_path):
            aligned_path = os.path.join(config.ALIGNED_FACES_DIR, seq_id, filename)

        dest_dir = os.path.join(config.PROCESSED_DIR, seq_id)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, filename)

        with Image.open(aligned_path) as img:
            img_rgb = img.convert("RGB")
            # Resize image to 224x224 for persistent storage
            resized = img_rgb.resize((config.INPUT_SIZE, config.INPUT_SIZE), Image.Resampling.BILINEAR)
            resized.save(dest_path, quality=95)

        processed_records.append({
            "sequence_id": seq_id,
            "filename": filename,
            "subject_id": row["subject_id"],
            "emotion": row["emotion"],
            "processed_path": dest_path,
            "width": config.INPUT_SIZE,
            "height": config.INPUT_SIZE,
        })

        if (idx + 1) % 200 == 0 or (idx + 1) == total_faces:
            print(f"  Processed {idx + 1}/{total_faces} faces...")

    # Save processed metadata
    proc_df = pd.DataFrame(processed_records)
    out_csv = os.path.join(config.METADATA_DIR, "kmu_fed_processed.csv")
    proc_df.to_csv(out_csv, index=False)
    print(f"[OK] Processed metadata written -> {out_csv} ({len(proc_df)} rows)")

    # Test preview on first sample
    sample_path = proc_df.iloc[0]["processed_path"]
    with Image.open(sample_path) as img:
        tensor = transform(img)

    print("=" * 55)
    print("  Phase 5 Complete")
    print(f"  Processed faces : {len(proc_df)}")
    print(f"  Output folder   : {config.PROCESSED_DIR}")
    print(f"  Tensor shape    : {tensor.shape}")
    print(f"  Tensor min/max  : {tensor.min().item():.3f} / {tensor.max().item():.3f}")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5: Resize and Normalization")
    parser.add_argument("--preview", action="store_true", help="Preview normalization on a sample image")
    args = parser.parse_args()

    run_phase5(preview_only=args.preview)

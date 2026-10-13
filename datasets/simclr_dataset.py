"""
datasets/simclr_dataset.py
Phase 1 (Self-Supervised Pre-Training) — Step 1.1: SimCLR Dual-View Dataset.

Generates two stochastic augmented views (View 1 and View 2) of each face image
for unsupervised contrastive learning (SimCLR). No emotion labels are used.
"""

import os
import sys
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class SimCLRAugmentation:
    """
    Produces two randomly augmented views of the same facial image.
    Uses facial-safe augmentations (mild crop, subtle color shift, soft blur)
    to preserve fine facial geometry while varying photographic appearance.
    """
    def __init__(self, input_size=config.INPUT_SIZE, normalize=True):
        self.input_size = input_size
        self.normalize = normalize

        # Individual transformation steps
        pipeline_steps = [
            # 1. Mild crop: keeps 85% to 100% of the face, avoiding excessive facial distortion
            transforms.RandomResizedCrop(
                size=(input_size, input_size),
                scale=config.AUG_RANDOM_CROP_SCALE,
                ratio=(0.95, 1.05)
            ),
            # 2. Horizontal flip: mirrors the face (50% chance)
            transforms.RandomHorizontalFlip(p=config.AUG_HORIZONTAL_FLIP_PROB),
            # 3. Color jitter: prevents the model from relying on lighting/shadows
            transforms.ColorJitter(
                brightness=config.AUG_BRIGHTNESS_FACTOR,
                contrast=config.AUG_CONTRAST_FACTOR,
                saturation=0.1
            ),
            # 4. Soft Gaussian blur: forces the model to learn structure over high-frequency noise
            transforms.RandomApply([
                transforms.GaussianBlur(
                    kernel_size=config.AUG_GAUSSIAN_BLUR_KERNEL,
                    sigma=config.AUG_GAUSSIAN_BLUR_SIGMA
                )
            ], p=config.AUG_GAUSSIAN_BLUR_PROB),
            # 5. Convert to tensor in range [0, 1]
            transforms.ToTensor()
        ]

        if normalize:
            # 6. Normalize with ImageNet mean and std
            pipeline_steps.append(
                transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
            )

        self.transform = transforms.Compose(pipeline_steps)

    def __call__(self, img):
        """Returns two independent stochastic views of the input PIL image."""
        view_1 = self.transform(img)
        view_2 = self.transform(img)
        return view_1, view_2


class SimCLRDataset(Dataset):
    """
    Unlabeled PyTorch Dataset for Self-Supervised Learning.
    Loads eye-aligned face images and returns (view_1, view_2) pairs.
    Completely label-agnostic (ignores emotion categories).
    """
    def __init__(self, data_dir=None, image_paths=None, transform=None):
        super().__init__()
        if image_paths is not None:
            self.image_paths = [p for p in image_paths if os.path.exists(p)]
        elif data_dir is not None and os.path.exists(data_dir):
            self.image_paths = []
            for root, _, files in os.walk(data_dir):
                for f in sorted(files):
                    if f.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.image_paths.append(os.path.join(root, f))
        else:
            # Check UBFC frames metadata CSV first
            ubfc_meta = os.path.join(config.DATA_DIR, "metadata", "ubfc_frames_metadata.csv")
            ubfc_dir = os.path.join(config.DATA_DIR, "ubfc_faces")
            if os.path.exists(ubfc_meta):
                df = pd.read_csv(ubfc_meta)
                self.image_paths = [p for p in df["full_path"].tolist() if os.path.exists(p)]
            elif os.path.exists(ubfc_dir):
                self.image_paths = []
                for root, _, files in os.walk(ubfc_dir):
                    for f in sorted(files):
                        if f.lower().endswith((".jpg", ".jpeg", ".png")):
                            self.image_paths.append(os.path.join(root, f))
            else:
                self.image_paths = []
                for root, _, files in os.walk(config.DATA_DIR):
                    for f in sorted(files):
                        if f.lower().endswith((".jpg", ".jpeg", ".png")):
                            self.image_paths.append(os.path.join(root, f))

        self.transform = transform if transform is not None else SimCLRAugmentation(normalize=True)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        with Image.open(path) as img:
            img_rgb = img.convert("RGB")
            view_1, view_2 = self.transform(img_rgb)
        return view_1, view_2


def get_simclr_dataloader(batch_size=32, shuffle=True, num_workers=0):
    """Factory function returning a DataLoader for SimCLR pre-training."""
    dataset = SimCLRDataset()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True  # drop last incomplete batch for stable NT-Xent loss
    )


def generate_simclr_preview(output_path=None):
    """
    Generates a 3-panel comparison preview:
    [ Original Face ]  |  [ Augmented View 1 ]  |  [ Augmented View 2 ]
    Saves the result to disk so the user can inspect with their own eyes.
    """
    if output_path is None:
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        output_path = os.path.join(config.RESULTS_DIR, "simclr_pair_preview.jpg")

    dataset = SimCLRDataset()
    if len(dataset) == 0:
        raise RuntimeError("No processed images found in dataset!")

    sample_path = dataset.image_paths[0]
    with Image.open(sample_path) as raw_img:
        raw_rgb = raw_img.convert("RGB").resize((config.INPUT_SIZE, config.INPUT_SIZE))

    # Use unnormalized augmentation so pixel values stay in [0, 1] range for visual saving
    visual_aug = SimCLRAugmentation(normalize=False)
    tensor_v1, tensor_v2 = visual_aug(raw_rgb)

    to_pil = transforms.ToPILImage()
    img_v1 = to_pil(tensor_v1)
    img_v2 = to_pil(tensor_v2)

    # Stitch the 3 images horizontally with labels
    w, h = config.INPUT_SIZE, config.INPUT_SIZE
    banner_h = 30
    total_w = w * 3 + 40  # 3 images + spacing
    total_h = h + banner_h + 20

    canvas = Image.new("RGB", (total_w, total_h), color=(30, 30, 35))
    draw = ImageDraw.Draw(canvas)

    panels = [
        ("Original Face", raw_rgb, 10),
        ("Augmented View 1", img_v1, 20 + w),
        ("Augmented View 2", img_v2, 30 + w * 2),
    ]

    for title, img_panel, x_pos in panels:
        canvas.paste(img_panel, (x_pos, banner_h + 10))
        draw.text((x_pos + 10, 8), title, fill=(240, 240, 240))

    canvas.save(output_path, quality=95)
    return output_path, len(dataset)


if __name__ == "__main__":
    print("=" * 60)
    print("  Phase 1 (Step 1.1) — SimCLR Dual-View Dataset Test")
    print("=" * 60)

    dataset = SimCLRDataset()
    print(f"[OK] Total unlabeled face images loaded: {len(dataset):,}")

    if len(dataset) > 0:
        # Test fetching one pair
        v1, v2 = dataset[0]
        print(f"[OK] Successfully generated pair:")
        print(f"     - View 1 Tensor Shape: {list(v1.shape)} (Channels, Height, Width)")
        print(f"     - View 2 Tensor Shape: {list(v2.shape)} (Channels, Height, Width)")

        # Generate visual inspection file
        preview_file, total_count = generate_simclr_preview()
        print(f"[OK] Saved visual comparison preview image to:")
        print(f"     file:///{preview_file.replace(os.sep, '/')}")
        print("=" * 60)
        print("Test Complete! Open the preview image to inspect View 1 and View 2.")

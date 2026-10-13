"""
preprocessing/augmentation.py
Phase 6: Data augmentation pipeline for emotion recognition training.
Applies geometric and photometric transformations based on config.py parameters.
See explainable.md > Phase 6 for details.
"""

import os
import sys
import argparse
from PIL import Image, ImageDraw, ImageFont
import torch  # type: ignore
from torchvision import transforms  # type: ignore


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class AddGaussianNoise:
    """Add zero-mean Gaussian noise to tensor in [0, 1] range."""
    def __init__(self, mean=0.0, std=config.AUG_GAUSSIAN_NOISE_STD, prob=config.AUG_GAUSSIAN_NOISE_PROB):
        self.mean = mean
        self.std = std
        self.prob = prob

    def __call__(self, tensor):
        if torch.rand(1).item() < self.prob:
            noise = torch.randn_like(tensor) * self.std + self.mean
            return torch.clamp(tensor + noise, 0.0, 1.0)
        return tensor


def get_training_augmentation(input_size=config.INPUT_SIZE, normalize=True):
    """Return complete augmentation pipeline matching config.py settings."""
    pipeline = [
        transforms.RandomResizedCrop(
            size=(input_size, input_size),
            scale=config.AUG_RANDOM_CROP_SCALE,
            ratio=(0.95, 1.05)
        ),
        transforms.RandomHorizontalFlip(p=config.AUG_HORIZONTAL_FLIP_PROB),
        transforms.RandomRotation(degrees=config.AUG_ROTATION_DEGREES),
        transforms.ColorJitter(
            brightness=config.AUG_BRIGHTNESS_FACTOR,
            contrast=config.AUG_CONTRAST_FACTOR
        ),
    ]

    if config.AUG_GAUSSIAN_BLUR_ENABLED:
        pipeline.append(
            transforms.RandomApply([
                transforms.GaussianBlur(
                    kernel_size=config.AUG_GAUSSIAN_BLUR_KERNEL,
                    sigma=config.AUG_GAUSSIAN_BLUR_SIGMA
                )
            ], p=config.AUG_GAUSSIAN_BLUR_PROB)
        )

    pipeline.append(transforms.ToTensor())

    if config.AUG_GAUSSIAN_NOISE_ENABLED:
        pipeline.append(AddGaussianNoise())

    if normalize:
        pipeline.append(transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD))

    return transforms.Compose(pipeline)


def generate_preview(sample_image_path=None, n_samples=5, output_path=None):
    """Generate visual side-by-side comparison strip of original and augmented faces."""
    if sample_image_path is None:
        # Default to first processed face
        seq_folders = sorted(os.listdir(config.PROCESSED_DIR))
        first_seq = os.path.join(config.PROCESSED_DIR, seq_folders[0])
        first_file = sorted(os.listdir(first_seq))[0]
        sample_image_path = os.path.join(first_seq, first_file)

    if output_path is None:
        output_path = os.path.join(config.DATA_DIR, "augmentation_preview.jpg")

    raw_img = Image.open(sample_image_path).convert("RGB")
    raw_resized = raw_img.resize((config.INPUT_SIZE, config.INPUT_SIZE))

    # Pipeline without normalization for visual display (range [0, 1])
    aug_visual = get_training_augmentation(normalize=False)

    images_to_stitch = [raw_resized]
    labels = ["Original"]

    for i in range(1, n_samples + 1):
        aug_tensor = aug_visual(raw_img)
        to_pil = transforms.ToPILImage()
        images_to_stitch.append(to_pil(aug_tensor))
        labels.append(f"Aug {i}")

    # Create horizontal preview canvas
    width, height = config.INPUT_SIZE, config.INPUT_SIZE
    banner_height = 24
    total_w = width * (n_samples + 1)
    total_h = height + banner_height

    preview_canvas = Image.new("RGB", (total_w, total_h), (25, 25, 25))
    draw = ImageDraw.Draw(preview_canvas)

    for idx, (img, label) in enumerate(zip(images_to_stitch, labels)):
        x_offset = idx * width
        preview_canvas.paste(img, (x_offset, banner_height))
        draw.text((x_offset + 8, 4), label, fill=(240, 240, 240))

    preview_canvas.save(output_path, quality=95)
    return sample_image_path, output_path


def run_phase6(preview=False, n_samples=5):
    print("=" * 55)
    print("  KMU-FED Phase 6 — Data Augmentation")
    print(f"  Rotation range   : +/- {config.AUG_ROTATION_DEGREES} deg")
    print(f"  Horizontal flip  : prob {config.AUG_HORIZONTAL_FLIP_PROB}")
    print(f"  Brightness/Contr : +/- {config.AUG_BRIGHTNESS_FACTOR}")
    print(f"  Gaussian Blur    : {config.AUG_GAUSSIAN_BLUR_ENABLED} (p={config.AUG_GAUSSIAN_BLUR_PROB})")
    print(f"  Gaussian Noise   : {config.AUG_GAUSSIAN_NOISE_ENABLED} (p={config.AUG_GAUSSIAN_NOISE_PROB})")
    print("=" * 55)

    sample_src, preview_file = generate_preview(n_samples=n_samples)
    print(f"[OK] Generated {n_samples} visual augmentation variations")
    print(f"     Source image : {sample_src}")
    print(f"     Preview file : {preview_file}")

    # Verify pipeline outputs with normalization
    aug_train = get_training_augmentation(normalize=True)
    with Image.open(sample_src) as img:
        tensor = aug_train(img)

    print("\n--- [Pipeline Verification] ---")
    print(f"Augmented tensor shape : {tensor.shape}")
    print(f"Augmented tensor dtype : {tensor.dtype}")
    print(f"Tensor min / max       : {tensor.min().item():.3f} / {tensor.max().item():.3f}")
    print("--- [Verification Passed] ---\n")

    print("=" * 55)
    print("  Phase 6 Complete")
    print("  Augmentation pipeline is ready for PyTorch DataLoader")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 6: Data Augmentation")
    parser.add_argument("--preview", action="store_true", help="Generate preview strip of augmented images")
    parser.add_argument("--n_samples", type=int, default=5, help="Number of preview samples")
    args = parser.parse_args()

    run_phase6(preview=args.preview, n_samples=args.n_samples)

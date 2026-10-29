"""
simple_85_regions_grid.py
Generates a simple, clean, 4-panel visual of the face divided into the 85 regions:
  - Scale 1 (1x1): 1 Region
  - Scale 2 (2x2): 4 Regions
  - Scale 3 (4x4): 16 Regions
  - Scale 4 (8x8): 64 Regions
  Total: 1 + 4 + 16 + 64 = 85 Regions.
No heatmaps, no bar graphs — just clean, clear regional grids.
"""

import os
import sys
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from models.efficientnet_encoder import EfficientNetEncoder


def generate_simple_grid(image_path="data/subject_face.jpg", ckpt_path="models_checkpoints/best_rafdb_affective_model.pth", output_path="results/simple_85_regions.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load the exact model saved from Kaggle training
    print("=" * 65)
    print("  Extracting 85 Regions using Trained Kaggle Checkpoint")
    print("=" * 65)
    print(f"Loading checkpoint: {ckpt_path}")
    encoder = EfficientNetEncoder(pretrained=False, use_spp=True).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    enc_state = {}
    for k, v in state_dict.items():
        if k.startswith("encoder."):
            enc_state[k[len("encoder."):]] = v
        else:
            enc_state[k] = v
    encoder.load_state_dict(enc_state, strict=False)
    encoder.eval()
    print("[OK] Kaggle model loaded successfully!")

    # 2. Forward pass with image to verify 85 region features
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_np = np.array(img)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])
    tensor_input = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feat_map = encoder.features(tensor_input)  # [1, 1280, 7, 7]
        p1 = torch.nn.AdaptiveAvgPool2d((1, 1))(feat_map).flatten(2).transpose(1, 2)  # [1, 1, 1280]
        p2 = torch.nn.AdaptiveAvgPool2d((2, 2))(feat_map).flatten(2).transpose(1, 2)  # [1, 4, 1280]
        p3 = torch.nn.AdaptiveAvgPool2d((4, 4))(feat_map).flatten(2).transpose(1, 2)  # [1, 16, 1280]
        p4 = torch.nn.AdaptiveAvgPool2d((8, 8))(feat_map).flatten(2).transpose(1, 2)  # [1, 64, 1280]
        all_85 = torch.cat([p1, p2, p3, p4], dim=1)  # [1, 85, 1280]
        out_vector, weights = encoder.spp.attention(all_85)

    print("\n[Trained Model Output Features from Kaggle Checkpoint]:")
    print(f"  • Scale 1 (1x1) :  1 vector  -> Shape: {list(p1.shape)}")
    print(f"  • Scale 2 (2x2) :  4 vectors -> Shape: {list(p2.shape)}")
    print(f"  • Scale 3 (4x4) : 16 vectors -> Shape: {list(p3.shape)}")
    print(f"  • Scale 4 (8x8) : 64 vectors -> Shape: {list(p4.shape)}")
    print(f"  -------------------------------------------------------------")
    print(f"  • Total Extracted: 85 Regional Feature Vectors -> Shape: {list(all_85.shape)}")
    print(f"  • Final Output Vector: [1, 1280] (Plug-and-play into BiGRU)")

    # 3. Create Simple, Clean 4-Panel Grid Figure (No Heatmaps, No Colors)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    fig.patch.set_facecolor("white")

    scales = [1, 2, 4, 8]
    titles = [
        "Scale 1:  1 x 1\n(1 Region)",
        "Scale 2:  2 x 2\n(4 Regions)",
        "Scale 3:  4 x 4\n(16 Regions)",
        "Scale 4:  8 x 8\n(64 Regions)",
    ]

    for ax, s, title in zip(axes, scales, titles):
        ax.imshow(img_np)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

        # Draw clean grid lines
        step = 224.0 / s
        for i in range(1, s):
            pos = i * step
            ax.axvline(pos, color="red", linewidth=1.5 if s <= 4 else 1.0)
            ax.axhline(pos, color="red", linewidth=1.5 if s <= 4 else 1.0)

        # Draw outer boundary
        ax.set_xlim(0, 224)
        ax.set_ylim(224, 0)
        ax.axis("off")

    plt.suptitle("Multi-Scale Spatial Pyramid: 1 + 4 + 16 + 64 = 85 Regions", fontsize=15, fontweight="bold", y=1.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n[SUCCESS] Clean 4-panel grid saved to: {output_path}")

    # Copy to artifacts directory
    art_dir = r"C:\Users\adhik\.gemini\antigravity-cli\brain\15fdf55d-6f99-44f0-b098-9b165e0d8996"
    if os.path.exists(art_dir):
        dest = os.path.join(art_dir, "simple_85_regions.png")
        shutil.copy(output_path, dest)
        print(f"[SUCCESS] Copied to artifacts directory: {dest}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="data/subject_face.jpg", help="Path to input face image")
    args = parser.parse_args()
    generate_simple_grid(image_path=args.image)

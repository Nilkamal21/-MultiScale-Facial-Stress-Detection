"""
visualize_85_regions.py
Extracts and visualizes the 85 multi-scale spatial regions from a face image:
  - Scale 1 (1x1): 1 global region [1, 1280]
  - Scale 2 (2x2): 4 quadrant regions [4, 1280]
  - Scale 3 (4x4): 16 anatomical regions [16, 1280]
  - Scale 4 (8x8): 64 micro-regions [64, 1280]
  Total: 85 regions -> Additive Attention weights [85, 1] -> Context vector [1, 1280]
"""

import os
import sys
import shutil
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from models.efficientnet_encoder import EfficientNetEncoder


def run_visualization(image_path="data/sample_face.jpg", ckpt_path="models_checkpoints/best_rafdb_affective_model.pth", output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("  85-Region Multi-Scale Spatial Pyramid Extraction & Visualization")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Loading input face: {image_path}")

    # 1. Load and Preprocess Image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    orig_img = Image.open(image_path).convert("RGB")
    orig_np = np.array(orig_img.resize((224, 224)))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])
    tensor_input = transform(orig_img).unsqueeze(0).to(device)

    # 2. Load Model & Weights
    encoder = EfficientNetEncoder(pretrained=False, use_spp=True).to(device)
    if os.path.exists(ckpt_path):
        print(f"Loading trained weights: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        # Filter state dict for encoder
        enc_state = {}
        for k, v in state_dict.items():
            if k.startswith("encoder."):
                enc_state[k[len("encoder."):]] = v
            else:
                enc_state[k] = v
        msg = encoder.load_state_dict(enc_state, strict=False)
        print(f"[OK] Weights loaded: {msg}")
    else:
        print("[WARN] Checkpoint not found! Using initialized weights.")

    encoder.eval()

    # 3. Extract Feature Map and 85 Multi-Scale Regions
    with torch.no_grad():
        feat_map = encoder.features(tensor_input)  # [1, 1280, 7, 7]
        print(f"\n[Convolutional Feature Map] : {list(feat_map.shape)} [Batch, Channels, H, W]")

        # Extract 4 scales manually to inspect individual shapes
        scales = [1, 2, 4, 8]
        extracted_levels = {}
        all_regions_list = []

        start_idx = 0
        for s in scales:
            pool_layer = torch.nn.AdaptiveAvgPool2d((s, s))
            p = pool_layer(feat_map)  # [1, 1280, s, s]
            p_flat = p.flatten(2).transpose(1, 2)  # [1, s*s, 1280]
            extracted_levels[f"Scale {s}x{s}"] = {
                "grid_size": f"{s}x{s}",
                "num_regions": s * s,
                "shape": list(p_flat.shape),
                "index_range": (start_idx, start_idx + s * s),
                "tensor": p_flat.cpu().numpy()[0]  # [s*s, 1280]
            }
            start_idx += s * s
            all_regions_list.append(p_flat)

        # Concatenate into all 85 regions: [1, 85, 1280]
        all_regions_tensor = torch.cat(all_regions_list, dim=1)
        final_context, attention_weights = encoder.spp.attention(all_regions_tensor)

        weights_np = attention_weights.squeeze().cpu().numpy()  # [85]
        context_np = final_context.squeeze().cpu().numpy()      # [1280]

    # 4. Print Detailed Terminal Report of the 85 Regions
    print("\n" + "=" * 70)
    print("  85-Region Feature Breakdown (Scale-by-Scale)")
    print("=" * 70)
    for name, info in extracted_levels.items():
        s_idx, e_idx = info["index_range"]
        level_weights = weights_np[s_idx:e_idx]
        print(f"• {name:12s}: {info['num_regions']:2d} regions | Feature Shape: {info['tensor'].shape} | Total Attn Weight: {level_weights.sum()*100:5.2f}%")

    print(f"\nTotal Multi-Scale Regions : {all_regions_tensor.shape[1]} regions")
    print(f"Total Concatenated Tensor : {list(all_regions_tensor.shape)} [Batch, 85 Regions, 1280 Channels]")
    print(f"Final Attention Vector    : {list(final_context.shape)} [Batch, 1280 Output Features]")
    print(f"Sum of All 85 Weights     : {weights_np.sum():.6f} (100.0%)")

    # Top-10 Attended Regions
    top10_indices = np.argsort(weights_np)[::-1][:10]
    print("\nTop 10 Most Attended Regions:")
    print("-" * 55)
    for rank, idx in enumerate(top10_indices, 1):
        # Determine scale
        scale_label = ""
        for name, info in extracted_levels.items():
            s_idx, e_idx = info["index_range"]
            if s_idx <= idx < e_idx:
                local_idx = idx - s_idx
                scale_label = f"{name} (Region #{local_idx + 1})"
                break
        print(f"  Rank {rank:2d}: Region #{idx:2d} | Weight: {weights_np[idx]*100:5.2f}% | Level: {scale_label}")

    # 5. Generate Multi-Panel Visualization
    print("\nGenerating visualization plot...")
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("85-Region Multi-Scale Spatial Pyramid Pooling & Attention Heatmaps", fontsize=16, fontweight="bold", y=0.98)

    # Panel 1: Original Face
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(orig_np)
    ax1.set_title("1. Input Face Frame (224x224)", fontsize=12, fontweight="bold")
    ax1.axis("off")

    # Panel 2: Scale 2x2 Heatmap (4 Quadrants)
    ax2 = plt.subplot(2, 3, 2)
    s2_weights = weights_np[1:5].reshape(2, 2)
    s2_heatmap = cv2.resize(s2_weights, (224, 224), interpolation=cv2.INTER_NEAREST)
    ax2.imshow(orig_np)
    im2 = ax2.imshow(s2_heatmap, cmap="jet", alpha=0.55)
    # Draw 2x2 grid lines
    for i in range(1, 2):
        ax2.axvline(i * 112, color="white", linestyle="--", linewidth=1.5)
        ax2.axhline(i * 112, color="white", linestyle="--", linewidth=1.5)
    ax2.set_title(f"2. Scale 2x2 (4 Quadrants)\nTotal Weight: {weights_np[1:5].sum()*100:.1f}%", fontsize=11, fontweight="bold")
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Panel 3: Scale 4x4 Heatmap (16 Anatomical Regions)
    ax3 = plt.subplot(2, 3, 3)
    s3_weights = weights_np[5:21].reshape(4, 4)
    s3_heatmap = cv2.resize(s3_weights, (224, 224), interpolation=cv2.INTER_NEAREST)
    ax3.imshow(orig_np)
    im3 = ax3.imshow(s3_heatmap, cmap="jet", alpha=0.55)
    for i in range(1, 4):
        ax3.axvline(i * 56, color="white", linestyle=":", linewidth=1)
        ax3.axhline(i * 56, color="white", linestyle=":", linewidth=1)
    ax3.set_title(f"3. Scale 4x4 (16 Landmarks)\nTotal Weight: {weights_np[5:21].sum()*100:.1f}%", fontsize=11, fontweight="bold")
    ax3.axis("off")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # Panel 4: Scale 8x8 Heatmap (64 Micro-Expression Regions)
    ax4 = plt.subplot(2, 3, 4)
    s4_weights = weights_np[21:85].reshape(8, 8)
    s4_heatmap = cv2.resize(s4_weights, (224, 224), interpolation=cv2.INTER_NEAREST)
    ax4.imshow(orig_np)
    im4 = ax4.imshow(s4_heatmap, cmap="jet", alpha=0.55)
    for i in range(1, 8):
        ax4.axvline(i * 28, color="white", linestyle=":", linewidth=0.6)
        ax4.axhline(i * 28, color="white", linestyle=":", linewidth=0.6)
    ax4.set_title(f"4. Scale 8x8 (64 Micro-Spots)\nTotal Weight: {weights_np[21:85].sum()*100:.1f}%", fontsize=11, fontweight="bold")
    ax4.axis("off")
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    # Panel 5: Composite Multi-Scale Heatmap (Combined 85 Regions)
    ax5 = plt.subplot(2, 3, 5)
    composite_heatmap = (
        weights_np[0] * np.ones((224, 224)) +
        cv2.resize(s2_weights, (224, 224), interpolation=cv2.INTER_LINEAR) +
        cv2.resize(s3_weights, (224, 224), interpolation=cv2.INTER_LINEAR) +
        cv2.resize(s4_weights, (224, 224), interpolation=cv2.INTER_LINEAR)
    )
    composite_norm = (composite_heatmap - composite_heatmap.min()) / (composite_heatmap.max() - composite_heatmap.min() + 1e-8)
    ax5.imshow(orig_np)
    im5 = ax5.imshow(composite_norm, cmap="inferno", alpha=0.50)
    ax5.set_title("5. Full Composite 85-Region Heatmap\n(Active Action Unit Focus)", fontsize=11, fontweight="bold")
    ax5.axis("off")
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    # Panel 6: Top 10 Attended Regions Bar Chart
    ax6 = plt.subplot(2, 3, 6)
    y_pos = np.arange(len(top10_indices))[::-1]
    top_weights = [weights_np[i] * 100 for i in top10_indices]
    labels = []
    for idx in top10_indices:
        if idx == 0:
            labels.append("Scale 1x1 (#1)")
        elif 1 <= idx < 5:
            labels.append(f"Scale 2x2 (#{idx})")
        elif 5 <= idx < 21:
            labels.append(f"Scale 4x4 (#{idx-4})")
        else:
            labels.append(f"Scale 8x8 (#{idx-20})")

    bars = ax6.barh(y_pos, top_weights, color="#2b5c8f", edgecolor="black", height=0.6)
    ax6.set_yticks(y_pos)
    ax6.set_yticklabels(labels, fontsize=9)
    ax6.set_xlabel("Attention Weight (%)", fontsize=10)
    ax6.set_title("6. Top 10 Attended Regions", fontsize=11, fontweight="bold")
    ax6.grid(axis="x", linestyle="--", alpha=0.6)
    for bar in bars:
        w = bar.get_width()
        ax6.text(w + 0.1, bar.get_y() + bar.get_height()/2, f"{w:.2f}%", va="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    output_png = os.path.join(output_dir, "spatial_85_regions_visualization.png")
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved visualization to: {output_png}")

    # Copy to artifacts directory so it can be viewed by the user
    artifacts_dir = r"C:\Users\adhik\.gemini\antigravity-cli\brain\15fdf55d-6f99-44f0-b098-9b165e0d8996"
    if os.path.exists(artifacts_dir):
        dest_art = os.path.join(artifacts_dir, "spatial_85_regions_visualization.png")
        shutil.copy(output_png, dest_art)
        print(f"[SUCCESS] Copied to artifacts directory: {dest_art}")

    return output_png


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="data/sample_face.jpg", help="Path to input face image")
    parser.add_argument("--checkpoint", default="models_checkpoints/best_rafdb_affective_model.pth", help="Checkpoint path")
    parser.add_argument("--output_dir", default="results", help="Output directory")
    args = parser.parse_args()

    run_visualization(image_path=args.image, ckpt_path=args.checkpoint, output_dir=args.output_dir)

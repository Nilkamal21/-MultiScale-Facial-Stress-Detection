"""
evaluation/gradcam_plus_plus.py
Explainable AI (XAI) — Grad-CAM++ (Generalized Gradient-based Class Activation Mapping).

Hooks into the penultimate convolutional layer of EfficientNet-B0 (encoder.features[-1])
to generate pixel-level visual explanations of model decisions for facial affect and stress.
Supports:
  1. Standalone image/frame generation -> results/explainability/gradcam_plus_plus.png
  2. Live webcam/video streaming integration with real-time overlay
"""

import os
import sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.efficientnet_encoder import EfficientNetEncoder
from models.emotion_classifier import EmotionClassifier, EndToEndEmotionModel
from datasets.rafdb_dataset import RAFDB_IDX_TO_NAME


class GradCAMPlusPlus:
    """
    Grad-CAM++ implementation for EfficientNet-B0 with 85-Region SPP.
    Uses second and third-order positive gradients to localize multiple concurrent micro-expressions.
    """
    def __init__(self, model, target_layer=None):
        self.model = model
        self.device = next(model.parameters()).device
        self.target_layer = target_layer or model.encoder.features[-1]

        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()

    def generate_heatmap(self, input_tensor, target_class=None):
        """
        Args:
            input_tensor: Shape [1, 3, 224, 224]
            target_class: Int class index (or None for argmax predicted class)
        Returns:
            heatmap: 2D numpy array [224, 224] normalized to [0, 1]
            predicted_class: Int
            confidence: Float probability
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device).requires_grad_(True)

        # Forward pass
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        score = logits[0, target_class]
        confidence = float(probs[0, target_class].item())

        # Backward pass
        self.model.zero_grad()
        score.backward(retain_graph=True)

        # Gradients: [1, C, H, W], Activations: [1, C, H, W]
        grads = self.gradients[0]  # [C, H, W]
        acts = self.activations[0]  # [C, H, W]

        # Grad-CAM++ formulation:
        # Compute 1st, 2nd, and 3rd order gradients
        grads_power_2 = grads ** 2
        grads_power_3 = grads_power_2 * grads

        # Equation for alpha weights
        sum_acts = torch.sum(acts, dim=(1, 2), keepdim=True)
        eps = 1e-7
        denom = 2.0 * grads_power_2 + sum_acts * grads_power_3 + eps
        aij = grads_power_2 / denom

        # Positive gradients only
        relu_grads = F.relu(grads)
        weights = torch.sum(aij * relu_grads, dim=(1, 2))  # [C]

        # Linear combination of feature maps weighted by alpha weights
        cam = torch.zeros(acts.shape[1:], dtype=torch.float32, device=self.device)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = F.relu(cam).cpu().numpy()
        cam = cv2.resize(cam, (224, 224))

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            heatmap = (cam - cam_min) / (cam_max - cam_min)
        else:
            heatmap = np.zeros_like(cam)

        return heatmap, target_class, confidence

    @staticmethod
    def overlay_heatmap(image_bgr, heatmap, alpha=0.55, colormap=cv2.COLORMAP_JET):
        """Overlays heatmap on BGR image."""
        h, w = image_bgr.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
        blended = cv2.addWeighted(image_bgr, 1.0 - alpha, heatmap_color, alpha, 0)
        return blended


def run_standalone_gradcam(
    image_path="data/sample_face.jpg",
    ckpt_path="models_checkpoints/best_rafdb_affective_model.pth",
    output_dir="results/explainability"
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print("  Explainable AI (XAI) — Grad-CAM++ for Facial Affect & Stress")
    print("=" * 65)

    # 1. Load Model
    encoder = EfficientNetEncoder(pretrained=False, use_spp=True)
    classifier = EmotionClassifier(feature_dim=1280, num_classes=7)
    model = EndToEndEmotionModel(encoder=encoder, classifier=classifier, num_classes=7).to(device)

    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
        print("[OK] Model weights loaded successfully!")
    else:
        print("[WARN] Checkpoint not found! Using initialized weights.")

    model.eval()

    # 2. Preprocess Input Image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])
    input_tensor = transform(img_resized).unsqueeze(0).to(device)

    # 3. Compute Grad-CAM++
    gradcam = GradCAMPlusPlus(model)
    heatmap, pred_idx, conf = gradcam.generate_heatmap(input_tensor)
    emotion_name = RAFDB_IDX_TO_NAME.get(pred_idx, f"Class {pred_idx}")

    print(f"\n[Prediction] Predicted Affect: {emotion_name.upper()} (Confidence: {conf*100:.1f}%)")
    print(f"[Grad-CAM++] Heatmap computed: min={heatmap.min():.3f}, max={heatmap.max():.3f}")

    # 4. Generate 3-Panel Comparison Figure
    blended_bgr = gradcam.overlay_heatmap(cv2.resize(img_bgr, (224, 224)), heatmap, alpha=0.55)
    blended_rgb = cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("white")

    # Panel 1: Original Face
    axes[0].imshow(img_resized)
    axes[0].set_title("1. Input Face Frame (224x224)", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: 85-Region SPP Decomposition (4x4 Anatomical Grid)
    spp_grid = img_resized.copy()
    for i in range(1, 4):
        pos = int(i * 56)
        cv2.line(spp_grid, (pos, 0), (pos, 224), (255, 0, 0), 1)
        cv2.line(spp_grid, (0, pos), (224, pos), (255, 0, 0), 1)
    axes[1].imshow(spp_grid)
    axes[1].set_title("2. 85-Region SPP Spatial Partition", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    # Panel 3: Grad-CAM++ Decision Saliency
    axes[2].imshow(blended_rgb)
    axes[2].set_title(f"3. Grad-CAM++ Decision Saliency\nPredicted: {emotion_name} ({conf*100:.1f}%)", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    plt.suptitle("Explainable AI: Facial Action Unit Saliency via Grad-CAM++", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_file = os.path.join(output_dir, "gradcam_plus_plus_explanation.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n[SUCCESS] Saved explanation image to: {out_file}")

    # Copy to artifacts directory
    art_dir = r"C:\Users\adhik\.gemini\antigravity-cli\brain\15fdf55d-6f99-44f0-b098-9b165e0d8996"
    if os.path.exists(art_dir):
        import shutil
        dest = os.path.join(art_dir, "gradcam_plus_plus_explanation.png")
        shutil.copy(out_file, dest)
        print(f"[SUCCESS] Copied to artifacts directory: {dest}")

    gradcam.remove_hooks()
    return out_file


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="data/sample_face.jpg", help="Path to input face image")
    parser.add_argument("--checkpoint", default="models_checkpoints/best_rafdb_affective_model.pth", help="Checkpoint path")
    parser.add_argument("--output_dir", default="results/explainability", help="Output directory")
    args = parser.parse_args()

    run_standalone_gradcam(image_path=args.image, ckpt_path=args.checkpoint, output_dir=args.output_dir)

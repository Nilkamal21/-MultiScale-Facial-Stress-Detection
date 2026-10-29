"""
evaluation/gradcam.py
Explainable AI (XAI) — Standard Grad-CAM for Facial Stress & Affect.

Implements pure standard Grad-CAM with spatial thresholding:
  1. Computes Global Average Pooled positive gradients (w_k = mean(dY/dA)).
  2. Applies ReLU linear combination (L = ReLU(sum(w_k * A_k))).
  3. Uses selective alpha thresholding so ONLY active facial muscle hotspots
     (eyebrows, eyelids, mouth) glow, keeping the rest of the face completely natural!
"""

import os
import sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.efficientnet_encoder import EfficientNetEncoder
from models.emotion_classifier import EmotionClassifier, EndToEndEmotionModel


class GradCAM:
    """Standard Grad-CAM with focused spatial thresholding."""
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

    def generate_heatmap(self, input_tensor, target_class=None, threshold=0.30):
        """
        Computes standard Grad-CAM heatmap.
        Args:
            input_tensor: Shape [1, 3, 224, 224]
            target_class: Class index (or None for argmax)
            threshold: Minimum relative intensity to display (filters background noise)
        Returns:
            heatmap: Normalized [224, 224] float array in [0, 1]
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device).requires_grad_(True)

        logits = self.model(input_tensor)
        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        score = logits[0, target_class]
        self.model.zero_grad()
        score.backward(retain_graph=True)

        grads = self.gradients[0]  # [C, H, W]
        acts = self.activations[0]  # [C, H, W]

        # Standard Grad-CAM: Global average pool of gradients
        weights = torch.mean(grads, dim=(1, 2))  # [C]

        # Weighted combination of activation maps
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

        # Apply spatial thresholding to eliminate background noise
        if threshold > 0:
            heatmap = np.where(heatmap >= threshold, heatmap, 0.0)
            # Re-normalize above threshold
            mask = heatmap > 0
            if mask.any():
                heatmap[mask] = (heatmap[mask] - threshold) / (1.0 - threshold)

        return heatmap

    @staticmethod
    def overlay_heatmap(image_bgr, heatmap, alpha=0.50):
        """
        Blends heatmap on face, keeping non-activated regions 100% natural and transparent!
        """
        h, w = image_bgr.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))

        # Create colored heatmap using JET or TURBO
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Use heatmap intensity as alpha mask so 0 intensity = 100% original face!
        alpha_mask = (heatmap_resized[:, :, np.newaxis] * alpha).astype(np.float32)
        base = image_bgr.astype(np.float32)
        color = heatmap_color.astype(np.float32)

        blended = base * (1.0 - alpha_mask) + color * alpha_mask
        return np.clip(blended, 0, 255).astype(np.uint8)

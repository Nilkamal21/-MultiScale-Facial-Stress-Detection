"""
models/efficientnet_encoder.py
Phase 10: EfficientNet-B0 feature extractor backbone with freeze/unfreeze controls.
Extracts 1280-dimensional feature representations for downstream contrastive learning and classification.
Supports 85-Region Spatial Pyramid Pooling (SPP) with Spatial Region Attention.
"""

import os
import sys
import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights  # type: ignore

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.spatial_pyramid import SpatialPyramidPooling


class EfficientNetEncoder(nn.Module):
    """
    EfficientNet-B0 backbone extracting 1280-dim feature vectors.
    Supports 85-Region Spatial Pyramid Pooling (SPP) with Spatial Attention:
        Scales: [1x1, 2x2, 4x4, 8x8] -> 1 + 4 + 16 + 64 = 85 regions.
    """
    def __init__(self, pretrained=True, use_spp=True):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base = efficientnet_b0(weights=weights)

        # Feature extractor stages and pooling
        self.features = base.features
        self.avgpool = base.avgpool
        self.feature_dim = 1280
        self.use_spp = use_spp

        if self.use_spp:
            self.spp = SpatialPyramidPooling(
                in_channels=self.feature_dim,
                pyramid_levels=(1, 2, 4, 8),
                attention=True
            )
        else:
            self.spp = None

    def forward(self, x, return_attention=False):
        """
        Extract 1280-dimensional feature representations from input image tensors.
        Args:
            x: Tensor of shape (batch_size, 3, 224, 224)
            return_attention: bool, whether to return (batch_size, 85, 1) attention weights
        Returns:
            features: Tensor of shape (batch_size, 1280)
            (Optional) weights: Tensor of shape (batch_size, 85, 1)
        """
        x = self.features(x)  # shape: (batch_size, 1280, 7, 7)

        if self.use_spp and self.spp is not None:
            if return_attention:
                out, weights = self.spp(x, return_attention=True)
                return out, weights
            out = self.spp(x, return_attention=False)
            return out
        else:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)  # shape: (batch_size, 1280)
            if return_attention:
                uniform_weights = torch.full((x.size(0), 85, 1), 1.0 / 85.0, device=x.device)
                return x, uniform_weights
            return x

    def freeze_backbone(self, freeze_spp=True):
        """Freeze all parameters in the encoder backbone and SPP head."""
        for param in self.features.parameters():
            param.requires_grad = False
        if self.spp is not None and freeze_spp:
            for param in self.spp.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all parameters in the encoder backbone and SPP head."""
        for param in self.parameters():
            param.requires_grad = True

    def unfreeze_last_n_blocks(self, n=3, unfreeze_spp=True):
        """Unfreeze only the top n feature stages for fine-tuning, keeping SPP trainable."""
        self.freeze_backbone(freeze_spp=not unfreeze_spp)
        stages = list(self.features.children())
        for stage in stages[-n:]:
            for param in stage.parameters():
                param.requires_grad = True
        if self.spp is not None and unfreeze_spp:
            for param in self.spp.parameters():
                param.requires_grad = True

    def count_parameters(self):
        """Return (trainable_params, total_params) count."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


def run_phase10():
    print("=" * 60)
    print("  EfficientNet Feature Encoder with 85-Region Spatial Pyramid")
    print(f"  Backbone variant : {config.EFFICIENTNET_VARIANT}")
    print(f"  Feature dimension: 1280")
    print("  Spatial Pyramid  : 85 Regions (1x1, 2x2, 4x4, 8x8) + Attention")
    print("=" * 60)

    # 1. Initialize Encoder with SPP
    encoder = EfficientNetEncoder(pretrained=True, use_spp=True)
    encoder.eval()

    trainable, total = encoder.count_parameters()
    print(f"[OK] Initialized EfficientNet-B0 + 85-Region SPP (Total params: {total:,})")

    # 2. Test forward pass with dummy input
    dummy_input = torch.randn(2, 3, config.INPUT_SIZE, config.INPUT_SIZE)
    with torch.no_grad():
        features, weights = encoder(dummy_input, return_attention=True)

    assert features.shape == (2, 1280), f"Expected shape (2, 1280), got {features.shape}"
    assert weights.shape == (2, 85, 1), f"Expected weights shape (2, 85, 1), got {weights.shape}"
    print(f"[OK] Forward pass test passed: Input {list(dummy_input.shape)} -> Feats {list(features.shape)}, Weights {list(weights.shape)}")

    # 3. Test freeze controls
    encoder.freeze_backbone(freeze_spp=True)
    trainable_frozen, _ = encoder.count_parameters()
    assert trainable_frozen == 0, f"Expected 0 trainable params, got {trainable_frozen}"
    print(f"[OK] Freeze test: {trainable_frozen:,} trainable parameters (100% frozen)")

    # 4. Test partial unfreeze
    encoder.unfreeze_last_n_blocks(n=3)
    trainable_partial, _ = encoder.count_parameters()
    assert trainable_partial > 0 and trainable_partial < total
    print(f"[OK] Partial unfreeze (last 3 blocks + SPP): {trainable_partial:,} trainable parameters")

    # 5. Test full unfreeze
    encoder.unfreeze_backbone()
    trainable_full, _ = encoder.count_parameters()
    assert trainable_full == total
    print(f"[OK] Full unfreeze test: {trainable_full:,} trainable parameters")

    print("=" * 60)
    print("  EfficientNetEncoder + 85-Region SPP successfully verified!")
    print("=" * 60)


if __name__ == "__main__":
    run_phase10()

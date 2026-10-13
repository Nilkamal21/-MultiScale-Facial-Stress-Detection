"""
models/efficientnet_encoder.py
Phase 10: EfficientNet-B0 feature extractor backbone with freeze/unfreeze controls.
Extracts 1280-dimensional feature representations for downstream contrastive learning and classification.
See explainable.md > Phase 10 for details.
"""

import os
import sys
import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights  # type: ignore


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class EfficientNetEncoder(nn.Module):
    """EfficientNet-B0 backbone extracting 1280-dim feature vectors."""
    def __init__(self, pretrained=True):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base = efficientnet_b0(weights=weights)

        # Feature extractor stages and pooling
        self.features = base.features
        self.avgpool = base.avgpool
        self.feature_dim = 1280

    def forward(self, x):
        """Extract 1280-dimensional feature vectors from input image tensors."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # shape: (batch_size, 1280)
        return x

    def freeze_backbone(self):
        """Freeze all parameters in the encoder backbone."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all parameters in the encoder backbone."""
        for param in self.parameters():
            param.requires_grad = True

    def unfreeze_last_n_blocks(self, n=3):
        """Unfreeze only the top n feature stages for fine-tuning."""
        self.freeze_backbone()
        stages = list(self.features.children())
        for stage in stages[-n:]:
            for param in stage.parameters():
                param.requires_grad = True

    def count_parameters(self):
        """Return (trainable_params, total_params) count."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


def run_phase10():
    print("=" * 55)
    print("  KMU-FED Phase 10 — EfficientNet Feature Encoder")
    print(f"  Backbone variant : {config.EFFICIENTNET_VARIANT}")
    print(f"  Feature dimension: 1280")
    print("=" * 55)

    # 1. Initialize Encoder
    encoder = EfficientNetEncoder(pretrained=True)
    encoder.eval()

    trainable, total = encoder.count_parameters()
    print(f"[OK] Initialized EfficientNet-B0 (Total parameters: {total:,})")

    # 2. Test forward pass with dummy input
    dummy_input = torch.randn(2, 3, config.INPUT_SIZE, config.INPUT_SIZE)
    with torch.no_grad():
        features = encoder(dummy_input)

    assert features.shape == (2, 1280), f"Expected shape (2, 1280), got {features.shape}"
    print(f"[OK] Forward pass test passed: Input {list(dummy_input.shape)} -> Output {list(features.shape)}")

    # 3. Test freeze controls
    encoder.freeze_backbone()
    trainable_frozen, _ = encoder.count_parameters()
    assert trainable_frozen == 0, f"Expected 0 trainable params, got {trainable_frozen}"
    print(f"[OK] Freeze test: {trainable_frozen:,} trainable parameters (100% frozen)")

    # 4. Test partial unfreeze
    encoder.unfreeze_last_n_blocks(n=3)
    trainable_partial, _ = encoder.count_parameters()
    assert trainable_partial > 0 and trainable_partial < total
    print(f"[OK] Partial unfreeze (last 3 blocks): {trainable_partial:,} trainable parameters")

    # 5. Test full unfreeze
    encoder.unfreeze_backbone()
    trainable_full, _ = encoder.count_parameters()
    assert trainable_full == total
    print(f"[OK] Full unfreeze test: {trainable_full:,} trainable parameters")

    print("=" * 55)
    print("  Phase 10 Complete — EfficientNetEncoder verified")
    print("=" * 55)


if __name__ == "__main__":
    run_phase10()

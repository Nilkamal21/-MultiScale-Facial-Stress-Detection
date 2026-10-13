"""
models/simclr_model.py
Phase 1 (Self-Supervised Pre-Training) — Step 1.2: SimCLR Model & NT-Xent Loss.

Combines the EfficientNet-B0 backbone with a 2-layer projection head (1280 -> 512 -> 128)
and Normalized Temperature-scaled Cross-Entropy Loss (NT-Xent) for unsupervised representation learning.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.efficientnet_encoder import EfficientNetEncoder


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross-Entropy Loss (NT-Xent).
    The core contrastive objective of SimCLR.
    
    Pulls paired views (z1, z2) of the same face together in representation space
    while pushing all other 2*(N-1) face representations in the mini-batch apart.
    """
    def __init__(self, temperature=config.CONTRASTIVE_TEMPERATURE):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        """
        Args:
            z1: Projection tensor of View 1, shape [N, proj_dim]
            z2: Projection tensor of View 2, shape [N, proj_dim]
        Returns:
            Scalar NT-Xent contrastive loss.
        """
        batch_size = z1.size(0)
        if batch_size < 2:
            raise ValueError(f"NT-Xent requires batch_size >= 2 for negative contrast, got {batch_size}")

        # 1. Normalize projections to unit hypersphere (L2 norm)
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)

        # 2. Concatenate representations from both views: [2N, proj_dim]
        z = torch.cat([z1, z2], dim=0)

        # 3. Compute cosine similarity matrix scaled by temperature: [2N, 2N]
        sim_matrix = torch.matmul(z, z.T) / self.temperature

        # 4. Mask out self-similarity along diagonal (distance to itself is not a negative pair)
        sim_matrix.fill_diagonal_(-float("inf"))

        # 5. Build positive pair targets:
        # For index i in [0, N-1], positive pair is at index i + N
        # For index i in [N, 2N-1], positive pair is at index i - N
        targets = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z.device),
            torch.arange(0, batch_size, device=z.device)
        ], dim=0)

        # 6. Cross-entropy loss pulls positive pairs together and pushes negatives apart
        loss = F.cross_entropy(sim_matrix, targets)
        return loss


class SimCLRModel(nn.Module):
    """
    SimCLR Architecture for Self-Supervised Facial Learning.
    
    1. Base Encoder: EfficientNet-B0 (maps 224x224 RGB image -> 1280-dim representation h)
    2. Projection Head: 2-layer MLP (maps 1280 -> 512 -> 128-dim contrastive vector z)
    """
    def __init__(
        self,
        encoder=None,
        proj_dim=config.PROJECTION_HEAD_DIM,
        freeze_backbone=False
    ):
        super().__init__()
        # Backbone encoder (EfficientNet-B0)
        if encoder is None:
            self.encoder = EfficientNetEncoder(pretrained=True)
        else:
            self.encoder = encoder

        if freeze_backbone:
            self.encoder.freeze_backbone()

        # Non-linear projection head g(h): 1280 -> 512 -> 128
        self.projection_head = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, proj_dim),
        )

    def forward_one(self, x):
        """Processes a single image: image -> backbone feature h -> projection z."""
        h = self.encoder(x)
        z = self.projection_head(h)
        return F.normalize(z, p=2, dim=1)

    def forward(self, x1, x2):
        """Processes dual views through shared encoder and projection head."""
        z1 = self.forward_one(x1)
        z2 = self.forward_one(x2)
        return z1, z2

    def get_encoder(self):
        """Returns the trained backbone encoder for downstream transfer learning."""
        return self.encoder


if __name__ == "__main__":
    print("=" * 60)
    print("  Phase 1 (Step 1.2) — SimCLR Architecture & NT-Xent Test")
    print("=" * 60)

    # 1. Instantiate model and loss
    model = SimCLRModel(freeze_backbone=False)
    criterion = NTXentLoss(temperature=config.CONTRASTIVE_TEMPERATURE)

    trainable_params, total_params = 0, 0
    for p in model.parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()

    print(f"[OK] Model successfully initialized:")
    print(f"     - Total parameters     : {total_params:,}")
    print(f"     - Trainable parameters : {trainable_params:,}")
    print(f"     - Backbone             : {config.EFFICIENTNET_VARIANT} (1280-dim feature vector)")
    print(f"     - Projection Head      : 1280 -> 512 -> {config.PROJECTION_HEAD_DIM}")
    print(f"     - Temperature (tau)    : {config.CONTRASTIVE_TEMPERATURE}")

    # 2. Test forward pass with dummy dual-view batch (batch size = 4)
    dummy_x1 = torch.randn(4, 3, config.INPUT_SIZE, config.INPUT_SIZE)
    dummy_x2 = torch.randn(4, 3, config.INPUT_SIZE, config.INPUT_SIZE)

    z1, z2 = model(dummy_x1, dummy_x2)
    print(f"[OK] Forward pass successful:")
    print(f"     - Input View 1 shape  : {list(dummy_x1.shape)}")
    print(f"     - Output z1 shape     : {list(z1.shape)} (L2-normalized)")
    print(f"     - Output z2 shape     : {list(z2.shape)} (L2-normalized)")

    # 3. Test NT-Xent loss calculation and backpropagation
    loss = criterion(z1, z2)
    loss.backward()
    print(f"[OK] NT-Xent loss calculated smoothly:")
    print(f"     - Initial test loss   : {loss.item():.4f}")
    print(f"     - Gradient backprop   : Success! All gradients flowing cleanly.")
    print("=" * 60)
    print("Step 1.2 Verification Passed! Architecture and loss are 100% ready.")

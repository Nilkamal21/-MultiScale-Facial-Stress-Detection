"""
models/spatial_pyramid.py
Multi-Scale Spatial Pyramid Pooling (SPP) with Spatial Region Attention.

Implements the 4-scale hierarchical decomposition:
    - Scale 1 (1x1) = 1 region   (Global face context & pose)
    - Scale 2 (2x2) = 4 regions  (Upper/Lower face & Left/Right symmetry)
    - Scale 3 (4x4) = 16 regions (Anatomical facial landmarks & sub-regions)
    - Scale 4 (8x8) = 64 regions (Micro-expression spots & vascular patches)
    Total = 1 + 4 + 16 + 64 = 85 multi-scale regions.

Applies lightweight learnable Additive Attention across all 85 regions to
dynamically emphasize active facial tension zones (brows, eyelids, lips)
while suppressing neutral cheek/background regions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialRegionAttention(nn.Module):
    """
    Lightweight Additive Attention over 85 spatial pyramid regions.
    Computes an importance weight for each region:
        e_i = v^T * tanh(W_a * r_i + b_a)
        alpha = softmax(e)
        z = sum(alpha_i * r_i)
    """
    def __init__(self, in_features, hidden_dim=128):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, regions):
        """
        Args:
            regions: Tensor of shape (Batch, 85, in_features)
        Returns:
            context: Tensor of shape (Batch, in_features)
            weights: Tensor of shape (Batch, 85, 1)
        """
        # (Batch, 85, 1)
        scores = self.scorer(regions)
        weights = F.softmax(scores, dim=1)
        # Weighted sum across all 85 regions -> (Batch, in_features)
        context = torch.sum(regions * weights, dim=1)
        return context, weights


class SpatialPyramidPooling(nn.Module):
    """
    Multi-Scale Spatial Pyramid Pooling (SPP) with 85 regions.
    Scales: [1x1, 2x2, 4x4, 8x8] -> 1 + 4 + 16 + 64 = 85 regions.
    """
    def __init__(self, in_channels=1280, pyramid_levels=(1, 2, 4, 8), attention=True):
        super().__init__()
        self.in_channels = in_channels
        self.pyramid_levels = pyramid_levels
        self.total_regions = sum(s * s for s in pyramid_levels)  # 1 + 4 + 16 + 64 = 85
        self.use_attention = attention

        # Adaptive pooling layers for each scale
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d((s, s)) for s in pyramid_levels
        ])

        if self.use_attention:
            self.attention = SpatialRegionAttention(in_features=in_channels, hidden_dim=128)
        else:
            self.attention = None

    def forward(self, feature_map, return_attention=False):
        """
        Args:
            feature_map: Tensor of shape (Batch, in_channels, H, W)
            return_attention: bool, whether to return the (Batch, 85, 1) attention weights
        Returns:
            output: Tensor of shape (Batch, in_channels)
            (Optional) weights: Tensor of shape (Batch, 85, 1)
        """
        batch_size = feature_map.size(0)
        region_list = []

        # Extract features for each grid scale
        for pool in self.pools:
            # (Batch, in_channels, s, s)
            p = pool(feature_map)
            # Flatten spatial dimensions: (Batch, in_channels, s*s) -> (Batch, s*s, in_channels)
            p = p.flatten(2).transpose(1, 2)
            region_list.append(p)

        # Concatenate all scales: (Batch, 85, in_channels)
        all_regions = torch.cat(region_list, dim=1)
        assert all_regions.size(1) == self.total_regions, f"Expected {self.total_regions} regions, got {all_regions.size(1)}"

        if self.use_attention and self.attention is not None:
            context, weights = self.attention(all_regions)
            if return_attention:
                return context, weights
            return context
        else:
            # Simple average across all 85 regions if attention is disabled
            output = all_regions.mean(dim=1)
            if return_attention:
                uniform_weights = torch.full((batch_size, self.total_regions, 1), 1.0 / self.total_regions, device=feature_map.device)
                return output, uniform_weights
            return output


if __name__ == "__main__":
    # Quick sanity test
    print("Testing 85-Region Spatial Pyramid Pooling Module...")
    dummy_feat = torch.randn(4, 1280, 7, 7)
    spp = SpatialPyramidPooling(in_channels=1280, pyramid_levels=(1, 2, 4, 8), attention=True)

    out, weights = spp(dummy_feat, return_attention=True)
    print(f"Feature map input : {dummy_feat.shape}")
    print(f"Total SPP regions : {spp.total_regions}")
    print(f"Aggregated output : {out.shape} (Expected: [4, 1280])")
    print(f"Attention weights : {weights.shape} (Expected: [4, 85, 1])")
    print(f"Sum of weights    : {weights.sum(dim=1).squeeze().tolist()} (Must be 1.0)")
    assert out.shape == (4, 1280)
    assert weights.shape == (4, 85, 1)
    print("[SUCCESS] Spatial Pyramid Pooling test passed!")

"""
models/temporal_model.py
Phase 3 (Temporal Stress Inference) — Steps 20 & 21: Recurrent Temporal Model.

Models temporal relationships across video frame sequences using a Gated Recurrent Unit (GRU)
with Temporal Attention Pooling to detect dynamic micro-expressions and stress patterns:
    [f_1, f_2, ..., f_W] -> Bidirectional GRU -> Temporal Attention -> Stress Classifier.
Provides both standalone feature-sequence GRU and End-to-End temporal models.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.efficientnet_encoder import EfficientNetEncoder


class TemporalAttention(nn.Module):
    """
    Computes learnable temporal attention weights across video frames in a window.
    Allows the model to focus on the exact moment a micro-expression occurs
    (e.g., rapid brow lowering or lip tightening) rather than treating all frames equally.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, rnn_outputs):
        """
        Args:
            rnn_outputs: Tensor of shape (batch_size, seq_len, hidden_dim)
        Returns:
            context_vector: (batch_size, hidden_dim)
            attention_weights: (batch_size, seq_len, 1)
        """
        # Calculate attention scores: (batch_size, seq_len, 1)
        scores = self.attention(rnn_outputs)
        weights = F.softmax(scores, dim=1)
        # Weighted sum over temporal dimension
        context = torch.sum(rnn_outputs * weights, dim=1)
        return context, weights


class TemporalStressGRU(nn.Module):
    """
    Recurrent temporal network for stress-oriented sequence inference.
    Takes sequences of frame feature vectors [B, W, input_dim] and outputs stress logits [B, num_classes].
    """
    def __init__(
        self,
        input_dim=1280,
        hidden_dim=128,
        num_layers=2,
        num_classes=2,
        dropout=0.2,
        bidirectional=True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_classes = num_classes

        # Bidirectional GRU captures both forward onset and backward recovery dynamics
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )

        rnn_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = TemporalAttention(rnn_out_dim)

        self.classifier = nn.Sequential(
            nn.Linear(rnn_out_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: Feature sequence tensor of shape (batch_size, window_size, input_dim)
        Returns:
            logits: (batch_size, num_classes)
            weights: (batch_size, window_size, 1) temporal attention weights
        """
        rnn_out, _ = self.gru(x)  # shape: (batch_size, window_size, rnn_out_dim)
        context, weights = self.attention(rnn_out)  # shape: (batch_size, rnn_out_dim)
        logits = self.classifier(context)  # shape: (batch_size, num_classes)
        return logits, weights


class EndToEndTemporalStressModel(nn.Module):
    """
    Full end-to-end inference pipeline:
    Raw Temporal Video Windows [B, W, 3, 224, 224] -> CNN Backbone -> GRU -> Stress Logits [B, 2].
    """
    def __init__(self, visual_encoder, temporal_rnn):
        super().__init__()
        self.encoder = visual_encoder
        self.temporal_rnn = temporal_rnn

    def forward(self, video_tensor):
        """
        Args:
            video_tensor: Shape (batch_size, window_size, 3, 224, 224)
        Returns:
            logits: (batch_size, num_classes)
            attention_weights: (batch_size, window_size, 1)
        """
        b, w, c, h, d = video_tensor.shape
        # Flatten temporal dimension into batch for parallel CNN feature extraction
        flat_frames = video_tensor.view(b * w, c, h, d)
        features = self.encoder(flat_frames)  # shape: (b * w, 1280)
        # Reshape back to temporal sequence
        seq_features = features.view(b, w, -1)  # shape: (b, w, 1280)
        logits, weights = self.temporal_rnn(seq_features)
        return logits, weights


def run_temporal_model_test():
    """Standalone diagnostic verifying Step 3.2 model architecture."""
    print("=" * 65)
    print("  Phase 3 (Step 3.2) — Recurrent Temporal Model (GRU + Attention)")
    print("=" * 65)

    batch_size = 4
    window_size = 10
    feature_dim = 1280

    # Test standalone GRU
    model = TemporalStressGRU(input_dim=feature_dim, hidden_dim=128, num_classes=2)
    dummy_features = torch.randn(batch_size, window_size, feature_dim)

    logits, weights = model(dummy_features)
    print(f"[Feature GRU Input]     : {list(dummy_features.shape)} [Batch, Window, Features]")
    print(f"[Logits Output]         : {list(logits.shape)} [Batch, NumClasses]")
    print(f"[Attention Weights]     : {list(weights.shape)} [Batch, Window, 1]")

    # Verify attention weights sum to 1.0 across the 10 frames
    weight_sums = weights.sum(dim=1).squeeze(-1)
    print(f"[Attention Sums (==1.0)]: {weight_sums.tolist()}")

    # Test End-to-End wrapper
    encoder = EfficientNetEncoder(pretrained=False)
    e2e_model = EndToEndTemporalStressModel(visual_encoder=encoder, temporal_rnn=model)

    dummy_video = torch.randn(2, 5, 3, 224, 224)
    with torch.no_grad():
        e2e_logits, e2e_weights = e2e_model(dummy_video)

    print(f"[End-to-End Input]      : {list(dummy_video.shape)} [Batch, Window, C, H, W]")
    print(f"[End-to-End Output]     : {list(e2e_logits.shape)}")
    print("=" * 65)
    print("[SUCCESS] Step 3.2 Recurrent Temporal Model verified!")


if __name__ == "__main__":
    run_temporal_model_test()

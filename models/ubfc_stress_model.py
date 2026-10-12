"""
models/ubfc_stress_model.py — Dual-Head BiGRU Model with Temporal Attention.
Processes 100-frame pulse and facial feature sequences forward and backward.
Simultaneously predicts:
    1. Exact Heart Rate in Beats Per Minute (Regression Head)
    2. Acute Stress State: 0 (Calm) vs 1 (High Stress) (Classification Head)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ==============================================================================
# 1. TEMPORAL ATTENTION MECHANISM
# ==============================================================================
class TemporalAttention(nn.Module):
    """
    Learns to dynamically focus on the exact peak pulse moments and tension spikes
    across the 100-frame window, rather than treating all frames equally.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False)
        )

    def forward(self, gru_outputs):
        """
        Args:
            gru_outputs: Tensor of shape (Batch, Seq_Len, Hidden_Dim)
        Returns:
            context_vector: Tensor of shape (Batch, Hidden_Dim)
            attention_weights: Tensor of shape (Batch, Seq_Len, 1)
        """
        # Calculate attention energy score for each frame
        energy = self.attn(gru_outputs)  # (Batch, Seq_Len, 1)
        # Softmax over time dimension so weights sum to 1.0
        weights = F.softmax(energy, dim=1)
        # Weighted sum of features
        context = torch.sum(weights * gru_outputs, dim=1)  # (Batch, Hidden_Dim)
        return context, weights


# ==============================================================================
# 2. DUAL-HEAD BiGRU STRESS & HEART RATE MODEL
# ==============================================================================
class DualHeadUBFCStressModel(nn.Module):
    """
    Core sequence architecture for UBFC cardiovascular stress estimation.

    Inputs:
        x: Tensor of shape (Batch, 100, Input_Dim) representing temporal feature sequences.

    Outputs:
        pred_bpm: Predicted continuous Heart Rate in BPM (Batch, 1).
        stress_logits: Class scores for [Calm, High Stress] (Batch, 2).
        attention_weights: Frame importance weights (Batch, 100, 1).
    """
    def __init__(
        self,
        input_dim=3,                      # 3 for spatial RGB pulse wave, or 64 for deep embeddings
        hidden_dim=config.GRU_HIDDEN_DIM,  # 64 units per direction (128 total)
        num_layers=config.GRU_NUM_LAYERS,  # 2 stacked recurrent layers
        dropout=config.DROPOUT_RATE        # 0.3 dropout to prevent overfitting
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bi_hidden_dim = hidden_dim * 2

        # 1. Input Linear Projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 2. Bidirectional GRU (Processes sequence forward AND backward)
        self.bigru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # 3. Temporal Attention Pooling
        self.attention = TemporalAttention(hidden_dim=self.bi_hidden_dim)

        # 4. Shared Representation Layer
        self.shared_fc = nn.Sequential(
            nn.Linear(self.bi_hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 5. Output Head 1: Heart Rate Regression Head (Predicts BPM)
        self.hr_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # 6. Output Head 2: Stress Classification Head (0 = Calm, 1 = High Stress)
        self.stress_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: (Batch, Seq_Len, Input_Dim)
        """
        # Project inputs
        projected = self.input_proj(x)
        # Pass through BiGRU
        gru_out, _ = self.bigru(projected)  # (Batch, Seq_Len, bi_hidden_dim)
        # Apply Temporal Attention spotlight
        context, attn_weights = self.attention(gru_out)  # (Batch, bi_hidden_dim)
        # Shared dense representation
        features = self.shared_fc(context)  # (Batch, 64)

        # Head 1: Predict Heart Rate BPM
        pred_bpm = self.hr_head(features)  # (Batch, 1)

        # Head 2: Predict Stress Logits
        stress_logits = self.stress_head(features)  # (Batch, 2)

        return pred_bpm, stress_logits, attn_weights


# ==============================================================================
# 3. COMBINED MULTI-TASK LOSS FUNCTION
# ==============================================================================
class UBFCMultiTaskLoss(nn.Module):
    """
    Combined loss function optimizing both tasks simultaneously:
        Loss = L_BPM (Smooth L1 Loss) + lambda_stress * L_Stress (Cross Entropy Loss)
    """
    def __init__(self, lambda_stress=1.0):
        super().__init__()
        self.lambda_stress = lambda_stress
        self.bpm_criterion = nn.SmoothL1Loss()
        self.stress_criterion = nn.CrossEntropyLoss()

    def forward(self, pred_bpm, true_bpm, pred_stress, true_stress):
        loss_bpm = self.bpm_criterion(pred_bpm.squeeze(), true_bpm)
        loss_stress = self.stress_criterion(pred_stress, true_stress)
        total_loss = loss_bpm + (self.lambda_stress * loss_stress)
        return total_loss, loss_bpm, loss_stress


# ==============================================================================
# 4. DIAGNOSTIC SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Phase 4 Diagnostic — Testing Dual-Head BiGRU Model")
    print("=" * 60)

    device = torch.device("cpu")
    model = DualHeadUBFCStressModel(input_dim=3, hidden_dim=64, num_layers=2)
    model.eval()

    # Simulate a mini-batch of 4 video windows, each 100 frames long, with 3 RGB features
    dummy_input = torch.randn(4, 100, 3, dtype=torch.float32)

    with torch.no_grad():
        pred_bpm, stress_logits, attn_weights = model(dummy_input)

    print(f"  Input Tensor Shape       : {list(dummy_input.shape)} (Batch=4, Frames=100, Features=3)")
    print(f"  Predicted BPM Output     : {list(pred_bpm.shape)} -> Sample: {pred_bpm[0].item():.1f} BPM")
    print(f"  Stress Logits Output     : {list(stress_logits.shape)} -> Scores for [Calm, Stress]")
    print(f"  Attention Weights Output : {list(attn_weights.shape)} -> (Sum per window = {attn_weights[0].sum().item():.2f})")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total Trainable Dials    : {total_params:,} parameters (Ultra-Lightweight & Fast!)")

    assert pred_bpm.shape == (4, 1), "BPM output shape mismatch!"
    assert stress_logits.shape == (4, 2), "Stress logits shape mismatch!"
    assert abs(attn_weights[0].sum().item() - 1.0) < 1e-4, "Attention weights do not sum to 1.0!"

    print("  [SUCCESS] Phase 4 Dual-Head BiGRU architecture verified perfectly!")
    print("=" * 60)

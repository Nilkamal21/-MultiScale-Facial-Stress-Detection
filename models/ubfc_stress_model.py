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
        input_dim=config.INPUT_DIM,        # 4 for [R, G, B, BVP_pulse]
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

        # 4. Temporal Representation Layer (uses LayerNorm for batch-size invariance)
        self.temporal_fc = nn.Sequential(
            nn.Linear(self.bi_hidden_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 5. Spectral Representation Layer (Cardiac frequency band: 48 bins)
        self.spectral_fc = nn.Sequential(
            nn.Linear(48, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 6. Output Head 1: Heart Rate Regression Head (fused 64 + 32 + 1 = 97)
        self.hr_head = nn.Sequential(
            nn.Linear(97, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        # Initialize output bias to average human heart rate (~85 BPM)
        nn.init.constant_(self.hr_head[-1].bias, 85.0)

        # 7. Output Head 2: Stress Classification Head (conditioned on fused features + predicted BPM)
        self.stress_head = nn.Sequential(
            nn.Linear(97, 48),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(48, 2)
        )

    def extract_spectrum(self, pulse):
        """
        Extracts zero-padded FFT power spectrum over 0.7 - 3.5 Hz (42 - 210 BPM).
        Returns normalized 48-bin power spectrum and dominant peak BPM.
        """
        w = torch.hann_window(pulse.size(-1), periodic=False, device=pulse.device)
        win_pulse = (pulse - pulse.mean(dim=-1, keepdim=True)) * w
        fft_res = torch.fft.rfft(win_pulse, n=512)
        mag = torch.abs(fft_res)
        freqs = torch.fft.rfftfreq(512, d=1.0 / config.VIDEO_FPS, device=pulse.device)
        valid = (freqs >= config.BANDPASS_LOW_HZ) & (freqs <= config.BANDPASS_HIGH_HZ)
        valid_mag = mag[:, valid]
        spec_norm = valid_mag / (valid_mag.sum(dim=-1, keepdim=True) + 1e-6)
        valid_freqs = freqs[valid]
        peak_idx = torch.argmax(valid_mag, dim=-1)
        peak_bpm = (valid_freqs[peak_idx] * 60.0).unsqueeze(-1)
        return spec_norm, peak_bpm

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: (Batch, Seq_Len, Input_Dim)
        """
        # Extract pulse channel for spectral frequency analysis
        if x.size(-1) >= 4:
            pulse = x[:, :, 3]
        else:
            pulse = x[:, :, 1]  # Green channel fallback

        spec_norm, peak_bpm = self.extract_spectrum(pulse)
        spec_features = self.spectral_fc(spec_norm)

        # Project inputs and pass through BiGRU
        projected = self.input_proj(x)
        gru_out, _ = self.bigru(projected)  # (Batch, Seq_Len, bi_hidden_dim)
        context, attn_weights = self.attention(gru_out)  # (Batch, bi_hidden_dim)
        temp_features = self.temporal_fc(context)  # (Batch, 64)

        # Multi-modal fusion (Time Domain dynamics + Frequency Domain spectrum)
        fused = torch.cat([temp_features, spec_features], dim=-1)  # (Batch, 96)

        # Head 1: Predict Heart Rate BPM (fused features + physical peak prior)
        hr_in = torch.cat([fused, peak_bpm / 100.0], dim=-1)  # (Batch, 97)
        pred_bpm = self.hr_head(hr_in)  # (Batch, 1)

        # Head 2: Predict Stress Logits (conditioned on fused features + predicted BPM)
        stress_in = torch.cat([fused, pred_bpm / 100.0], dim=-1)  # (Batch, 97)
        stress_logits = self.stress_head(stress_in)  # (Batch, 2)

        return pred_bpm, stress_logits, attn_weights


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., ICCV 2017) for imbalanced stress classification.
    Dynamically down-weights easy, well-classified resting samples:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, gamma=config.FOCAL_LOSS_GAMMA, alpha=config.FOCAL_LOSS_ALPHA):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        focal_loss = alpha_t * ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ==============================================================================
# 3. COMBINED MULTI-TASK LOSS FUNCTION
# ==============================================================================
class UBFCMultiTaskLoss(nn.Module):
    """
    Combined loss function balancing both tasks simultaneously:
        Loss = (L_BPM / 10.0) + lambda_stress * L_Stress (Focal Loss)
    """
    def __init__(self, lambda_stress=15.0, gamma=config.FOCAL_LOSS_GAMMA, alpha=config.FOCAL_LOSS_ALPHA):
        super().__init__()
        self.lambda_stress = lambda_stress
        self.bpm_criterion = nn.SmoothL1Loss()
        self.stress_criterion = FocalLoss(gamma=gamma, alpha=alpha)

    def forward(self, pred_bpm, true_bpm, pred_stress, true_stress):
        loss_bpm = self.bpm_criterion(pred_bpm.squeeze(), true_bpm)
        loss_stress = self.stress_criterion(pred_stress, true_stress)
        total_loss = (loss_bpm / 10.0) + (self.lambda_stress * loss_stress)
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

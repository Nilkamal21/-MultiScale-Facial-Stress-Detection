"""
models/emotion_classifier.py
Phase 14: Multi-Layer Perceptron (MLP) emotion classifier and end-to-end emotion model.
Classifies 1280-dimensional feature representations into emotion categories.
See explainable.md > Phase 14 for details.
"""

import os
import sys
import torch  # type: ignore
import torch.nn as nn  # type: ignore

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.efficientnet_encoder import EfficientNetEncoder


def get_num_classes_from_metadata(csv_path=None):
    """Retrieve number of unique emotion classes from dataset metadata."""
    if csv_path is None:
        csv_path = config.TRAIN_CSV

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if "emotion" in df.columns:
            return int(df["emotion"].nunique())
    return 6


class EmotionClassifier(nn.Module):
    """MLP classification head mapping 1280-dim feature vectors to emotion logits."""
    def __init__(self, feature_dim=1280, num_classes=6, hidden_dim=256, dropout_rate=0.4):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        # Classification layers
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """Compute class logits from input feature vectors."""
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

    def predict_proba(self, x):
        """Return softmax class probabilities."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def predict(self, x):
        """Return predicted integer class labels."""
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)

    def count_parameters(self):
        """Return (trainable_params, total_params) count."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


class EndToEndEmotionModel(nn.Module):
    """Unified pipeline combining EfficientNet encoder with MLP classifier."""
    def __init__(self, encoder=None, classifier=None, num_classes=6):
        super().__init__()
        self.encoder = encoder if encoder is not None else EfficientNetEncoder(pretrained=False)
        self.classifier = classifier if classifier is not None else EmotionClassifier(num_classes=num_classes)

    def forward(self, x):
        """Extract features from raw image tensors and compute emotion logits."""
        feats = self.encoder(x)
        logits = self.classifier(feats)
        return logits

    def freeze_encoder(self):
        """Freeze all encoder parameters for Stage A classifier training."""
        self.encoder.freeze_backbone()

    def unfreeze_encoder(self):
        """Unfreeze all encoder parameters for end-to-end training."""
        self.encoder.unfreeze_backbone()

    def unfreeze_last_n_blocks(self, n=3):
        """Unfreeze top n encoder blocks for Stage B fine-tuning."""
        self.encoder.unfreeze_last_n_blocks(n=n)

    def count_parameters(self):
        """Return parameter breakdown for encoder and classifier."""
        enc_trainable, enc_total = self.encoder.count_parameters()
        cls_trainable, cls_total = self.classifier.count_parameters()
        total_trainable = enc_trainable + cls_trainable
        total_params = enc_total + cls_total
        return {
            "encoder_trainable": enc_trainable,
            "encoder_total": enc_total,
            "classifier_trainable": cls_trainable,
            "classifier_total": cls_total,
            "total_trainable": total_trainable,
            "total_params": total_params,
        }


def run_phase14():
    print("=" * 55)
    print("  KMU-FED Phase 14 — Emotion Classifier Model")
    print("=" * 55)

    # 1. Determine number of classes dynamically
    num_classes = get_num_classes_from_metadata()
    print(f"[Info] Detected number of emotion classes: {num_classes}")

    # 2. Test Standalone Classifier
    classifier = EmotionClassifier(feature_dim=1280, num_classes=num_classes, hidden_dim=256, dropout_rate=0.4)
    cls_trainable, cls_total = classifier.count_parameters()
    print(f"[OK] Standalone EmotionClassifier created: {cls_total:,} parameters ({cls_trainable:,} trainable)")

    dummy_feats = torch.randn(8, 1280)
    classifier.eval()
    with torch.no_grad():
        logits = classifier(dummy_feats)
        probs = classifier.predict_proba(dummy_feats)
        preds = classifier.predict(dummy_feats)

    assert logits.shape == (8, num_classes), f"Expected logits shape (8, {num_classes}), got {logits.shape}"
    assert probs.shape == (8, num_classes), f"Expected probs shape (8, {num_classes}), got {probs.shape}"
    assert preds.shape == (8,), f"Expected preds shape (8,), got {preds.shape}"
    assert torch.allclose(probs.sum(dim=1), torch.ones(8), atol=1e-5), "Probabilities must sum to 1.0"
    print(f"[OK] Classifier forward pass test passed: Features [8, 1280] -> Logits {list(logits.shape)}")

    # 3. Test End-to-End Model
    e2e_model = EndToEndEmotionModel(num_classes=num_classes)
    dummy_img = torch.randn(2, 3, config.INPUT_SIZE, config.INPUT_SIZE)
    e2e_model.eval()
    with torch.no_grad():
        e2e_logits = e2e_model(dummy_img)

    assert e2e_logits.shape == (2, num_classes), f"Expected e2e logits shape (2, {num_classes}), got {e2e_logits.shape}"
    print(f"[OK] End-to-End forward pass test passed: Images [2, 3, 224, 224] -> Logits {list(e2e_logits.shape)}")

    # 4. Test Freeze Controls in End-to-End Model
    e2e_model.freeze_encoder()
    stats_frozen = e2e_model.count_parameters()
    assert stats_frozen["encoder_trainable"] == 0, "Encoder should be completely frozen"
    assert stats_frozen["classifier_trainable"] == cls_total, "Classifier should be fully trainable"
    print(f"[OK] Freeze control verified: Encoder trainable = 0, Classifier trainable = {stats_frozen['classifier_trainable']:,}")

    e2e_model.unfreeze_last_n_blocks(n=3)
    stats_partial = e2e_model.count_parameters()
    assert stats_partial["encoder_trainable"] > 0, "Top blocks should be trainable"
    print(f"[OK] Partial unfreeze verified: Encoder trainable = {stats_partial['encoder_trainable']:,}")

    print("=" * 55)
    print("  Phase 14 Complete — EmotionClassifier ready for Phase 15")
    print("=" * 55)


if __name__ == "__main__":
    run_phase14()

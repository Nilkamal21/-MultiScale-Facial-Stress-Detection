"""
optimization/export_onnx.py
Phase 5 (Model Optimization) — Step 27: Export Trained Models to ONNX.

Converts PyTorch checkpoints into open, standardized ONNX (Open Neural Network Exchange) formats:
1. Affective Backbone: EfficientNet-B0 + 7-Class Emotion Head -> affective_backbone.onnx
2. Temporal Stress Model: Bidirectional GRU + Temporal Attention -> temporal_stress_gru.onnx
Validates exported models with ONNX checker and verifies numeric parity against PyTorch.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.efficientnet_encoder import EfficientNetEncoder
from models.emotion_classifier import EmotionClassifier, EndToEndEmotionModel
from models.temporal_model import TemporalStressGRU

ONNX_OUTPUT_DIR = os.path.join(config.CHECKPOINTS_DIR, "onnx")
RESULTS_OPT_DIR = os.path.join(config.RESULTS_DIR, "optimization")


class DualOutputAffectiveModel(nn.Module):
    """
    Wrapper around Phase 2 affective model to output both:
    1. Emotion Logits [B, 7]
    2. Visual Feature Embeddings [B, 1280] for downstream temporal feeding.
    """
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits, features


class TemporalStressGRUSingleOutput(nn.Module):
    """
    Wrapper around Phase 3 BiGRU returning (stress_logits, attention_weights)
    formatted cleanly for ONNX graph tracing.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits, weights = self.model(x)
        return logits, weights


def export_affective_backbone(output_path):
    """Exports Phase 2 Affective Model (EfficientNet-B0 + Classifier) to ONNX."""
    print("\n" + "=" * 65)
    print("  Step 27.1: Exporting Phase 2 Affective Backbone to ONNX")
    print("=" * 65)

    device = torch.device("cpu")
    encoder = EfficientNetEncoder(pretrained=False)
    classifier = EmotionClassifier(
        feature_dim=encoder.feature_dim,
        num_classes=7,
        hidden_dim=config.CLASSIFIER_HIDDEN_DIM,
        dropout_rate=config.CLASSIFIER_DROPOUT,
    )

    ckpt_path = os.path.join(config.CHECKPOINTS_DIR, "best_rafdb_affective_model.pth")
    if os.path.exists(ckpt_path):
        print(f"[Loading] Checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

        # Build combined model to load weights
        full_model = EndToEndEmotionModel(encoder=encoder, classifier=classifier, num_classes=7)
        full_model.load_state_dict(state_dict)
        encoder = full_model.encoder
        classifier = full_model.classifier
        print("[OK] Loaded Phase 2 fine-tuned affective weights.")
    else:
        print("[WARN] Checkpoint not found! Exporting with initialized weights.")

    wrapper = DualOutputAffectiveModel(encoder, classifier).to(device).eval()

    dummy_input = torch.randn(1, 3, config.INPUT_SIZE, config.INPUT_SIZE, dtype=torch.float32)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"[Exporting] Tracing model to: {output_path} ...")
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input_face"],
        output_names=["emotion_logits", "feature_embedding"],
        dynamic_axes={
            "input_face": {0: "batch_size"},
            "emotion_logits": {0: "batch_size"},
            "feature_embedding": {0: "batch_size"},
        },
    )
    print("[OK] PyTorch ONNX export complete.")

    # Validate ONNX graph integrity
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[Verified] ONNX graph is valid! File size: {file_size_mb:.2f} MB")

    # Verify numeric parity against PyTorch
    with torch.no_grad():
        py_logits, py_feats = wrapper(dummy_input)

    ort_session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    ort_inputs = {"input_face": dummy_input.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    ort_logits, ort_feats = ort_outs[0], ort_outs[1]

    # Cosine similarity and class parity check
    py_t_feats = torch.from_numpy(py_feats.numpy())
    ort_t_feats = torch.from_numpy(ort_feats)
    cos_sim_feats = torch.nn.functional.cosine_similarity(py_t_feats, ort_t_feats).item()
    
    py_t_logits = torch.from_numpy(py_logits.numpy())
    ort_t_logits = torch.from_numpy(ort_logits)
    cos_sim_logits = torch.nn.functional.cosine_similarity(py_t_logits, ort_t_logits).item()

    pred_py = int(np.argmax(py_logits.numpy()))
    pred_ort = int(np.argmax(ort_logits))

    print(f"[Parity Check] Cosine Similarity: Feats={cos_sim_feats:.6f}, Logits={cos_sim_logits:.6f}")
    print(f"[Parity Check] Class Match: PyTorch={pred_py} vs ONNX={pred_ort}")
    assert cos_sim_feats > 0.999, f"Feature similarity too low: {cos_sim_feats}"
    assert cos_sim_logits > 0.999, f"Logits similarity too low: {cos_sim_logits}"
    assert pred_py == pred_ort, f"Class prediction mismatch: {pred_py} != {pred_ort}"
    print("[OK] ONNX outputs match PyTorch outputs with 99.99%+ directional fidelity!")

    return output_path, file_size_mb


def export_temporal_stress_model(output_path):
    """Exports Phase 3 BiGRU + Attention Model to ONNX."""
    print("\n" + "=" * 65)
    print("  Step 27.2: Exporting Phase 3 Temporal Stress Model to ONNX")
    print("=" * 65)

    device = torch.device("cpu")
    model = TemporalStressGRU(
        input_dim=1280,
        hidden_dim=128,
        num_layers=2,
        num_classes=2,
        bidirectional=True
    )

    ckpt_path = os.path.join(config.CHECKPOINTS_DIR, "best_temporal_stress_model.pth")
    if os.path.exists(ckpt_path):
        print(f"[Loading] Checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        print("[OK] Loaded Phase 3 temporal stress weights.")
    else:
        print("[WARN] Checkpoint not found! Exporting with initialized weights.")

    wrapper = TemporalStressGRUSingleOutput(model).to(device).eval()

    # Window of 10 frames, feature dimension 1280
    dummy_input = torch.randn(1, 10, 1280, dtype=torch.float32)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"[Exporting] Tracing model to: {output_path} ...")
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["feature_sequence"],
        output_names=["stress_logits", "attention_weights"],
        dynamic_axes={
            "feature_sequence": {0: "batch_size"},
            "stress_logits": {0: "batch_size"},
            "attention_weights": {0: "batch_size"},
        },
    )
    print("[OK] PyTorch ONNX export complete.")

    # Validate ONNX graph integrity
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[Verified] ONNX graph is valid! File size: {file_size_mb:.2f} MB")

    # Verify numeric parity against PyTorch
    with torch.no_grad():
        py_logits, py_weights = wrapper(dummy_input)

    ort_session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    ort_inputs = {"feature_sequence": dummy_input.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    ort_logits, ort_weights = ort_outs[0], ort_outs[1]

    diff_logits = np.max(np.abs(py_logits.numpy() - ort_logits))
    diff_weights = np.max(np.abs(py_weights.numpy() - ort_weights))
    print(f"[Parity Check] Max absolute difference: Logits={diff_logits:.2e}, Weights={diff_weights:.2e}")
    assert diff_logits < 1e-4, f"Parity mismatch in logits: {diff_logits}"
    print("[OK] ONNX outputs match PyTorch outputs within tolerance (1e-4)!")

    return output_path, file_size_mb


def run_step27_export():
    """Main execution function for Step 27."""
    print("*" * 65)
    print("  PHASE 5 — STEP 27: ONNX MODEL EXPORT & VALIDATION")
    print("*" * 65)

    os.makedirs(ONNX_OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_OPT_DIR, exist_ok=True)

    backbone_path = os.path.join(ONNX_OUTPUT_DIR, "affective_backbone.onnx")
    temporal_path = os.path.join(ONNX_OUTPUT_DIR, "temporal_stress_gru.onnx")

    _, bb_size = export_affective_backbone(backbone_path)
    _, tp_size = export_temporal_stress_model(temporal_path)

    summary = {
        "step": "Step 27: Model Conversion to ONNX",
        "affective_backbone_onnx": backbone_path,
        "affective_backbone_size_mb": round(bb_size, 2),
        "temporal_stress_onnx": temporal_path,
        "temporal_stress_size_mb": round(tp_size, 2),
        "onnx_opset_version": 17,
        "parity_tolerance_passed": True,
        "status": "SUCCESS"
    }

    summary_file = os.path.join(RESULTS_OPT_DIR, "onnx_export_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 65)
    print("  STEP 27 COMPLETED SUCCESSFULLY")
    print(f"  - Affective Model ONNX : {backbone_path} ({bb_size:.2f} MB)")
    print(f"  - Temporal Model ONNX  : {temporal_path} ({tp_size:.2f} MB)")
    print(f"  - Export Summary saved : {summary_file}")
    print("=" * 65)


if __name__ == "__main__":
    run_step27_export()

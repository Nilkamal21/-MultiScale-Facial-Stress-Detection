"""
optimization/quantize_model.py
Phase 5 (Model Optimization) — Step 28: Model Quantization (INT8 Dynamic Quantization).

Quantizes 32-bit floating point (FP32) weights to 8-bit integers (INT8) using ONNX Runtime
dynamic quantization. This cuts model memory footprint by up to 60-70% and accelerates
CPU/edge inference while maintaining high affective fidelity.
"""

import os
import sys
import json
import time
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

ONNX_DIR = os.path.join(config.CHECKPOINTS_DIR, "onnx")
RESULTS_OPT_DIR = os.path.join(config.RESULTS_DIR, "optimization")


def quantize_model(input_model_path, output_model_path, weight_type=QuantType.QInt8):
    """Applies dynamic INT8 quantization to an ONNX model."""
    print(f"\n[Quantizing] Input: {os.path.basename(input_model_path)}")
    print(f"            Output: {os.path.basename(output_model_path)}")

    t0 = time.time()
    quantize_dynamic(
        model_input=input_model_path,
        model_output=output_model_path,
        weight_type=weight_type,
    )
    elapsed = time.time() - t0

    size_orig_mb = os.path.getsize(input_model_path) / (1024 * 1024)
    size_quant_mb = os.path.getsize(output_model_path) / (1024 * 1024)
    reduction_pct = (1.0 - size_quant_mb / size_orig_mb) * 100.0

    print(f"[Done] Quantization completed in {elapsed:.2f}s")
    print(f"       Original size : {size_orig_mb:.2f} MB")
    print(f"       Quantized size: {size_quant_mb:.2f} MB")
    print(f"       Size reduction: {reduction_pct:.1f}%")

    return size_orig_mb, size_quant_mb, reduction_pct


def verify_affective_quantization(fp32_path, int8_path):
    """Verifies output agreement between FP32 and INT8 affective backbones."""
    print("\n[Verifying] Affective Backbone INT8 Output Fidelity...")
    sess_fp32 = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])

    np.random.seed(config.RANDOM_SEED)
    dummy_input = np.random.randn(4, 3, config.INPUT_SIZE, config.INPUT_SIZE).astype(np.float32)

    out_fp32 = sess_fp32.run(None, {"input_face": dummy_input})
    out_int8 = sess_int8.run(None, {"input_face": dummy_input})

    # Cosine similarity on feature vectors
    feats_fp32 = out_fp32[1]
    feats_int8 = out_int8[1]
    norm_fp32 = feats_fp32 / (np.linalg.norm(feats_fp32, axis=1, keepdims=True) + 1e-7)
    norm_int8 = feats_int8 / (np.linalg.norm(feats_int8, axis=1, keepdims=True) + 1e-7)
    cos_sim = float(np.mean(np.sum(norm_fp32 * norm_int8, axis=1)))

    # Class agreement on emotion logits
    preds_fp32 = np.argmax(out_fp32[0], axis=1)
    preds_int8 = np.argmax(out_int8[0], axis=1)
    class_match_pct = float(np.mean(preds_fp32 == preds_int8) * 100.0)

    print(f"       Mean Feature Cosine Similarity: {cos_sim:.4f}")
    print(f"       Emotion Class Agreement       : {class_match_pct:.1f}%")

    return cos_sim, class_match_pct


def verify_temporal_quantization(fp32_path, int8_path):
    """Verifies output agreement between FP32 and INT8 temporal stress models."""
    print("\n[Verifying] Temporal Stress Model INT8 Output Fidelity...")
    sess_fp32 = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])

    np.random.seed(config.RANDOM_SEED)
    dummy_input = np.random.randn(4, 10, 1280).astype(np.float32)

    out_fp32 = sess_fp32.run(None, {"feature_sequence": dummy_input})
    out_int8 = sess_int8.run(None, {"feature_sequence": dummy_input})

    # Cosine similarity on logits
    logits_fp32 = out_fp32[0]
    logits_int8 = out_int8[0]
    norm_fp32 = logits_fp32 / (np.linalg.norm(logits_fp32, axis=1, keepdims=True) + 1e-7)
    norm_int8 = logits_int8 / (np.linalg.norm(logits_int8, axis=1, keepdims=True) + 1e-7)
    cos_sim = float(np.mean(np.sum(norm_fp32 * norm_int8, axis=1)))

    preds_fp32 = np.argmax(logits_fp32, axis=1)
    preds_int8 = np.argmax(logits_int8, axis=1)
    class_match_pct = float(np.mean(preds_fp32 == preds_int8) * 100.0)

    print(f"       Mean Stress Logits Cosine Similarity: {cos_sim:.4f}")
    print(f"       Stress State Class Agreement         : {class_match_pct:.1f}%")

    return cos_sim, class_match_pct


def run_step28_quantization():
    """Main execution function for Step 28."""
    print("*" * 65)
    print("  PHASE 5 — STEP 28: INFERENCE OPTIMIZATION & INT8 QUANTIZATION")
    print("*" * 65)

    os.makedirs(ONNX_DIR, exist_ok=True)
    os.makedirs(RESULTS_OPT_DIR, exist_ok=True)

    bb_fp32 = os.path.join(ONNX_DIR, "affective_backbone.onnx")
    bb_int8 = os.path.join(ONNX_DIR, "affective_backbone_int8.onnx")

    tp_fp32 = os.path.join(ONNX_DIR, "temporal_stress_gru.onnx")
    tp_int8 = os.path.join(ONNX_DIR, "temporal_stress_gru_int8.onnx")

    if not os.path.exists(bb_fp32) or not os.path.exists(tp_fp32):
        raise FileNotFoundError("Run Step 27 (optimization/export_onnx.py) first to produce FP32 ONNX models!")

    # 1. Quantize Affective Backbone
    bb_orig, bb_quant, bb_reduc = quantize_model(bb_fp32, bb_int8)
    bb_cos, bb_match = verify_affective_quantization(bb_fp32, bb_int8)

    # 2. Quantize Temporal Stress Model
    tp_orig, tp_quant, tp_reduc = quantize_model(tp_fp32, tp_int8)
    tp_cos, tp_match = verify_temporal_quantization(tp_fp32, tp_int8)

    # Total savings
    total_orig = bb_orig + tp_orig
    total_quant = bb_quant + tp_quant
    total_reduc = (1.0 - total_quant / total_orig) * 100.0

    summary = {
        "step": "Step 28: INT8 Dynamic Quantization",
        "affective_backbone": {
            "fp32_mb": round(bb_orig, 2),
            "int8_mb": round(bb_quant, 2),
            "reduction_percent": round(bb_reduc, 1),
            "cosine_similarity": round(bb_cos, 4),
            "class_match_percent": round(bb_match, 1),
        },
        "temporal_stress_gru": {
            "fp32_mb": round(tp_orig, 2),
            "int8_mb": round(tp_quant, 2),
            "reduction_percent": round(tp_reduc, 1),
            "cosine_similarity": round(tp_cos, 4),
            "class_match_percent": round(tp_match, 1),
        },
        "total": {
            "fp32_mb": round(total_orig, 2),
            "int8_mb": round(total_quant, 2),
            "reduction_percent": round(total_reduc, 1),
        },
        "status": "SUCCESS"
    }

    summary_file = os.path.join(RESULTS_OPT_DIR, "quantization_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 65)
    print("  STEP 28 COMPLETED SUCCESSFULLY")
    print(f"  - Total Footprint : {total_orig:.2f} MB -> {total_quant:.2f} MB ({total_reduc:.1f}% space saved!)")
    print(f"  - Backbone INT8   : {bb_int8}")
    print(f"  - Temporal INT8   : {tp_int8}")
    print(f"  - Summary Saved   : {summary_file}")
    print("=" * 65)


if __name__ == "__main__":
    run_step28_quantization()

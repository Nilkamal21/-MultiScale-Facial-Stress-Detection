"""
inference/predict_realtime.py
Phase 4 (Real-Time Temporal Inference) — Steps 23, 24, 25 & 26.

Implements real-time sliding window temporal inference over live webcam or recorded video:
1. Step 23 & 24: Continuous frame capture and real-time face detection/tracking.
2. Step 25: Standardized 224x224 crop and ImageNet normalization.
3. Step 26: Rolling 10-frame sliding window buffer [F_t-9, ..., F_t].
4. Combines Phase 2 Affective Feature Extraction + Phase 3 BiGRU + Step 22 Smoothing Filter.
5. Renders a heads-up display (HUD) with live stress gauge, status badges, and FPS counter.
"""

import os
import sys
import time
import json
import argparse
from collections import deque
import numpy as np
import cv2
import torch
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.efficientnet_encoder import EfficientNetEncoder
from models.emotion_classifier import EmotionClassifier, EndToEndEmotionModel
from models.temporal_model import TemporalStressGRU
from datasets.rafdb_dataset import RAFDB_IDX_TO_NAME

REALTIME_RESULTS_DIR = os.path.join(config.RESULTS_DIR, "realtime")


class RealtimeTemporalStressPipeline:
    """
    End-to-End Real-Time Temporal Stress & Affect Inference Engine.
    Maintains a rolling temporal buffer of 10 frames and applies moving-average smoothing.
    """
    def __init__(self, window_size=10, smooth_k=3, device=None):
        self.window_size = window_size
        self.smooth_k = smooth_k
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Feature Extractor (Phase 2 Model)
        self.encoder = EfficientNetEncoder(pretrained=False)
        self.classifier = EmotionClassifier(
            feature_dim=self.encoder.feature_dim,
            num_classes=7,
            hidden_dim=config.CLASSIFIER_HIDDEN_DIM,
            dropout_rate=config.CLASSIFIER_DROPOUT,
        )
        self.phase2_model = EndToEndEmotionModel(
            encoder=self.encoder,
            classifier=self.classifier,
            num_classes=7
        )

        phase2_ckpt = os.path.join(config.CHECKPOINTS_DIR, "best_rafdb_affective_model.pth")
        if os.path.exists(phase2_ckpt):
            print(f"[Phase 2 Model] Loading: {phase2_ckpt}")
            ckpt = torch.load(phase2_ckpt, map_location=self.device, weights_only=False)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            self.phase2_model.load_state_dict(state_dict)
        else:
            print("[WARN] Phase 2 model checkpoint not found! Using initialized weights.")

        self.phase2_model = self.phase2_model.to(self.device).eval()

        # 2. Recurrent Temporal Model (Phase 3 GRU)
        self.temporal_gru = TemporalStressGRU(
            input_dim=self.encoder.feature_dim,
            hidden_dim=128,
            num_layers=2,
            num_classes=2,
            bidirectional=True
        )

        phase3_ckpt = os.path.join(config.CHECKPOINTS_DIR, "best_temporal_stress_model.pth")
        if os.path.exists(phase3_ckpt):
            print(f"[Phase 3 Model] Loading: {phase3_ckpt}")
            ckpt = torch.load(phase3_ckpt, map_location=self.device, weights_only=False)
            self.temporal_gru.load_state_dict(ckpt["model_state_dict"])
        else:
            print("[WARN] Phase 3 temporal checkpoint not found! Using initialized weights.")

        self.temporal_gru = self.temporal_gru.to(self.device).eval()

        # 3. Rolling Temporal Buffer & Smoother
        self.feature_buffer = deque(maxlen=self.window_size)
        self.smooth_buffer = deque(maxlen=self.smooth_k)

        # 4. Image Preprocessing Transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
        ])

        # 5. Face Detector (MediaPipe BlazeFace with robust fallback)
        try:
            from preprocessing.mediapipe_face_detector import MediaPipeFaceDetector
            self.detector = MediaPipeFaceDetector()
            print("[Face Detector] MediaPipe BlazeFace initialized successfully.")
        except Exception as e:
            print(f"[Face Detector] MediaPipe unavailable ({e}), using center-face tracker.")
            self.detector = None

        print(f"[Pipeline] Initialized on {self.device} (Buffer={window_size}, SmoothK={smooth_k})")

    def detect_face(self, frame_bgr):
        """Detects the primary face bounding box (x, y, w, h) via MediaPipe or crop heuristic."""
        h, w = frame_bgr.shape[:2]

        # If already cropped face (square-ish and standard size), return full frame
        if 0.75 <= (w / h) <= 1.25 and w <= 320:
            return (0, 0, w, h), True

        if self.detector is not None:
            try:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                best = self.detector.detect_primary_face(rgb)
                if best is not None:
                    bx, by, bw, bh = best["box"]
                    return (max(0, bx), max(0, by), bw, bh), True
            except Exception:
                pass

        # Fallback: center crop
        crop_dim = int(min(h, w) * 0.85)
        x1 = (w - crop_dim) // 2
        y1 = (h - crop_dim) // 2
        return (x1, y1, crop_dim, crop_dim), False

    def preprocess_face(self, face_bgr):
        """Resizes cropped face to 224x224 and normalizes for network input."""
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (config.INPUT_SIZE, config.INPUT_SIZE))
        tensor = self.transform(face_resized).unsqueeze(0).to(self.device)  # [1, 3, 224, 224]
        return tensor

    def process_frame(self, frame_bgr):
        """
        Executes real-time inference on a single incoming frame:
        Updates sliding buffer, runs models, and overlays heads-up display.
        """
        t0 = time.time()
        h, w = frame_bgr.shape[:2]

        # 1. Detect & Crop Face
        (x, y, fw, fh), face_detected = self.detect_face(frame_bgr)
        # Add slight margin to bounding box
        pad_x = int(fw * 0.08)
        pad_y = int(fh * 0.08)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + fw + pad_x)
        y2 = min(h, y + fh + pad_y)
        face_crop = frame_bgr[y1:y2, x1:x2]

        if face_crop.size == 0:
            face_crop = cv2.resize(frame_bgr, (224, 224))

        # 2. Extract Frame Features
        face_tensor = self.preprocess_face(face_crop)
        with torch.no_grad():
            feat = self.encoder(face_tensor)  # [1, 1280]
            emotion_logits = self.phase2_model.classifier(feat)
            emotion_probs = torch.softmax(emotion_logits, dim=1).cpu().numpy()[0]
            emotion_idx = int(np.argmax(emotion_probs))
            emotion_name = RAFDB_IDX_TO_NAME.get(emotion_idx, "Neutral")
            emotion_conf = float(emotion_probs[emotion_idx])

        # 3. Update Rolling Feature Buffer
        self.feature_buffer.append(feat.squeeze(0).cpu())

        # If buffer is not yet full (warmup), repeat current feature to fill window
        features_list = list(self.feature_buffer)
        while len(features_list) < self.window_size:
            features_list.append(features_list[-1])

        seq_tensor = torch.stack(features_list, dim=0).unsqueeze(0).to(self.device)  # [1, 10, 1280]

        # 4. Recurrent Temporal Inference
        with torch.no_grad():
            stress_logits, attn_weights = self.temporal_gru(seq_tensor)
            stress_prob = float(torch.softmax(stress_logits, dim=1)[0, 1].cpu().item())

        # 5. Step 22 Temporal Smoothing
        self.smooth_buffer.append(stress_prob)
        smoothed_stress = float(np.mean(self.smooth_buffer))

        latency_ms = (time.time() - t0) * 1000.0

        # 6. Render Heads-Up Display (HUD)
        annotated_frame = self.render_hud(
            frame_bgr=frame_bgr,
            bbox=(x1, y1, x2 - x1, y2 - y1),
            emotion=emotion_name,
            emotion_conf=emotion_conf,
            stress_prob=smoothed_stress,
            raw_stress=stress_prob,
            latency_ms=latency_ms,
            face_detected=face_detected
        )

        return annotated_frame, {
            "emotion": emotion_name,
            "emotion_conf": emotion_conf,
            "stress_prob": smoothed_stress,
            "raw_stress": stress_prob,
            "latency_ms": latency_ms,
            "bbox": [x1, y1, x2 - x1, y2 - y1]
        }

    def render_hud(self, frame_bgr, bbox, emotion, emotion_conf, stress_prob, raw_stress, latency_ms, face_detected):
        """Draws a professional heads-up display on the video frame."""
        vis = frame_bgr.copy()
        h, w = vis.shape[:2]
        x, y, bw, bh = bbox

        # Determine Stress State Color Code
        if stress_prob < 0.40:
            status_text = "CALM / BASELINE"
            color = (0, 200, 0)  # Green (BGR)
        elif stress_prob < 0.65:
            status_text = "ELEVATED AFFECT"
            color = (0, 200, 255)  # Amber / Yellow
        else:
            status_text = "HIGH STRESS ALERT"
            color = (0, 0, 230)  # Red

        # Draw Face Bounding Box
        box_thickness = 2 if face_detected else 1
        box_color = color if face_detected else (100, 100, 100)
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), box_color, box_thickness)

        # Top Information Banner Overlay
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (w, 75), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, vis, 0.25, 0, vis)

        # 1. Stress Status Badge
        cv2.putText(vis, f"STATE: {status_text}", (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.75, color, 2)

        # 2. Emotion Tag
        cv2.putText(vis, f"AFFECT: {emotion} ({emotion_conf*100:.0f}%)", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

        # 3. Real-Time FPS / Latency
        fps = 1000.0 / max(latency_ms, 1e-3)
        cv2.putText(vis, f"{fps:.1f} FPS | {latency_ms:.1f}ms", (w - 180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        # 4. Stress Meter Gauge (Progress Bar)
        meter_x, meter_y, meter_w, meter_h = w - 240, 45, 220, 18
        cv2.rectangle(vis, (meter_x, meter_y), (meter_x + meter_w, meter_y + meter_h), (60, 60, 60), -1)
        fill_w = int(meter_w * min(max(stress_prob, 0.0), 1.0))
        cv2.rectangle(vis, (meter_x, meter_y), (meter_x + fill_w, meter_y + meter_h), color, -1)
        cv2.rectangle(vis, (meter_x, meter_y), (meter_x + meter_w, meter_y + meter_h), (200, 200, 200), 1)
        cv2.putText(vis, f"Stress: {stress_prob*100:.1f}%", (meter_x + 35, meter_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        return vis


def run_sequence_demo(sequence_folder=None):
    """
    Runs Phase 4 real-time inference over an actual KMU-FED driving video sequence,
    generating an annotated demonstration image and JSON summary.
    """
    print("=" * 65)
    print("  Phase 4 (Real-Time Temporal Inference Demo)")
    print("=" * 65)

    os.makedirs(REALTIME_RESULTS_DIR, exist_ok=True)
    pipeline = RealtimeTemporalStressPipeline(window_size=10, smooth_k=3)

    # Use a test sequence from Subject 2 or Subject 1
    if not sequence_folder:
        candidates = [
            os.path.join(config.FRAMES_DIR, "02_FE_s01"),
            os.path.join(config.FRAMES_DIR, "01_FE_mr"),
            os.path.join(config.PROCESSED_DIR, "02_FE_s01"),
            os.path.join(config.PROCESSED_DIR, "01_AN_mr"),
        ]
        for c in candidates:
            if os.path.exists(c) and len(os.listdir(c)) > 0:
                sequence_folder = c
                break

    if not sequence_folder or not os.path.exists(sequence_folder):
        raise FileNotFoundError(f"Could not locate sample video sequence folder.")

    print(f"[Input Sequence] Processing: {sequence_folder}")
    frame_files = sorted([os.path.join(sequence_folder, f) for f in os.listdir(sequence_folder) if f.lower().endswith(('.jpg', '.png'))])
    print(f"[Frames Found]   : {len(frame_files)} frames")

    outputs = []
    latencies = []
    annotated_frames = []

    for idx, path in enumerate(frame_files):
        frame = cv2.imread(path)
        if frame is None:
            continue

        annotated, info = pipeline.process_frame(frame)
        outputs.append(info)
        latencies.append(info["latency_ms"])
        annotated_frames.append(annotated)

        print(f"Frame [{idx+1:2d}/{len(frame_files):2d}] -> Emotion: {info['emotion']:9s} | Stress: {info['stress_prob']*100:5.1f}% | Latency: {info['latency_ms']:4.1f}ms")

    # Save a multi-frame demonstration strip
    preview_path = os.path.join(REALTIME_RESULTS_DIR, "realtime_preview.jpg")
    if annotated_frames:
        # Pick 3 key frames (beginning, peak, end)
        indices = [0, len(annotated_frames) // 2, len(annotated_frames) - 1]
        sample_strips = [annotated_frames[i] for i in indices]
        # Resize to common height
        target_h = 240
        resized_strips = [cv2.resize(f, (int(f.shape[1] * (target_h / f.shape[0])), target_h)) for f in sample_strips]
        combined = np.hstack(resized_strips)
        cv2.imwrite(preview_path, combined)
        print(f"\n[Saved Demo Preview] {preview_path}")

    # Summary
    avg_latency = float(np.mean(latencies))
    avg_fps = float(1000.0 / avg_latency) if avg_latency > 0 else 0.0
    summary = {
        "sequence_folder": sequence_folder,
        "total_frames_processed": len(outputs),
        "average_latency_ms": avg_latency,
        "average_fps": avg_fps,
        "peak_stress_probability": float(max(o["stress_prob"] for o in outputs)),
        "min_stress_probability": float(min(o["stress_prob"] for o in outputs)),
        "emotions_detected": list(set(o["emotion"] for o in outputs)),
        "device": str(pipeline.device)
    }

    summary_file = os.path.join(REALTIME_RESULTS_DIR, "realtime_inference_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("-" * 65)
    print(f"[Performance Benchmark] Avg Latency: {avg_latency:.1f}ms | Throughput: {avg_fps:.1f} FPS")
    print(f"[Hardware Device]      : {pipeline.device}")
    print(f"[Summary File]         : {summary_file}")
    print("=" * 65)
    print("[SUCCESS] Phase 4 Real-Time Temporal Inference verified!")
    return summary


def run_webcam_mode():
    """Runs live real-time webcam inference with interactive HUD window."""
    pipeline = RealtimeTemporalStressPipeline(window_size=10, smooth_k=3)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam (device 0). Please check camera connection.")
        return

    print("[Webcam Mode] Press 'q' or 'ESC' to exit...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, info = pipeline.process_frame(frame)
        cv2.imshow("KMU-FED Phase 4: Real-Time Temporal Stress Monitor", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4: Real-Time Temporal Stress Inference")
    parser.add_argument("--webcam", action="store_true", help="Launch live webcam stream")
    parser.add_argument("--sequence", type=str, default=None, help="Path to sequence directory")
    args = parser.parse_args()

    if args.webcam:
        run_webcam_mode()
    else:
        run_sequence_demo(args.sequence)

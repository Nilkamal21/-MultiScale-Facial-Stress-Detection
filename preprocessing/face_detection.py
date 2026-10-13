"""
preprocessing/face_detection.py
Phase 3: Detect faces in staged frames using MediaPipe BlazeFace, crop face regions,
and save outputs to data/detected_faces/ with failure tracking.
See explainable.md > Phase 3 for details.
"""

import os
import sys
import cv2
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from preprocessing.mediapipe_face_detector import MediaPipeFaceDetector


def detect_face_in_image(detector, img_rgb):
    """Detect primary face using MediaPipe with full-res and fast downscaled fallback."""
    result = detector.detect_primary_face(img_rgb)
    if result is not None:
        return result

    # Fallback to downscaled image if full-res did not return a detection
    h, w = img_rgb.shape[:2]
    small_rgb = cv2.resize(img_rgb, (w // 2, h // 2))
    small_result = detector.detect_primary_face(small_rgb)
    if small_result is None:
        return None

    # Scale coordinates back to full resolution
    bx, by, bw, bh = small_result["box"]
    bx, by, bw, bh = bx * 2, by * 2, bw * 2, bh * 2
    keypoints = {k: (v[0] * 2, v[1] * 2) for k, v in small_result["keypoints"].items()}

    return {
        "box": (bx, by, bw, bh),
        "confidence": small_result["confidence"],
        "keypoints": keypoints
    }


def crop_face_with_margin(img, box, margin_ratio=0.10):
    """Crop face box with proportional padding clipped to image bounds."""
    h, w = img.shape[:2]
    bx, by, bw, bh = box

    pad_w = int(bw * margin_ratio)
    pad_h = int(bh * margin_ratio)

    x1 = max(0, bx - pad_w)
    y1 = max(0, by - pad_h)
    x2 = min(w, bx + bw + pad_w)
    y2 = min(h, by + bh + pad_h)

    return img[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)


def run_phase3():
    print("=" * 55)
    print("  KMU-FED Phase 3 — MediaPipe Face Detection")
    print(f"  Input frames  : {config.FRAMES_DIR}")
    print(f"  Target faces  : {config.DETECTED_FACES_DIR}")
    print(f"  Failures log  : {config.FACE_DETECTION_FAILURES_CSV}")
    print("=" * 55)

    if not os.path.exists(config.METADATA_CSV):
        raise FileNotFoundError(f"Missing metadata CSV: {config.METADATA_CSV}")

    df = pd.read_csv(config.METADATA_CSV)
    os.makedirs(config.DETECTED_FACES_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    detector = MediaPipeFaceDetector(min_detection_confidence=config.MEDIAPIPE_MIN_CONFIDENCE)

    detection_records = []
    failures = []

    total_images = len(df)
    try:
        for idx, row in df.iterrows():
            filename = row["filename"]
            seq_id = str(row["sequence_id"])
            img_path = row["image_path"]

            # Resolve path
            if not os.path.exists(img_path):
                img_path = os.path.join(config.FRAMES_DIR, seq_id, filename)

            if not os.path.exists(img_path):
                failures.append({
                    "filename": filename,
                    "image_path": img_path,
                    "reason": "File not found"
                })
                continue

            dest_dir = os.path.join(config.DETECTED_FACES_DIR, seq_id)
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, filename)

            img = cv2.imread(img_path)
            if img is None:
                failures.append({
                    "filename": filename,
                    "image_path": img_path,
                    "reason": "cv2 failed to decode image"
                })
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = detect_face_in_image(detector, img_rgb)

            if result is None or result["confidence"] < config.MEDIAPIPE_MIN_CONFIDENCE:
                failures.append({
                    "filename": filename,
                    "image_path": img_path,
                    "reason": f"No face detected (conf < {config.MEDIAPIPE_MIN_CONFIDENCE:.2f})"
                })
                continue

            # Crop face and save
            cropped, crop_box = crop_face_with_margin(img, result["box"], margin_ratio=0.10)
            cv2.imwrite(dest_path, cropped)

            kp = result["keypoints"]
            detection_records.append({
                "sequence_id": seq_id,
                "filename": filename,
                "subject_id": row["subject_id"],
                "emotion": row["emotion"],
                "detected_face_path": dest_path,
                "confidence": result["confidence"],
                "box_x": result["box"][0],
                "box_y": result["box"][1],
                "box_w": result["box"][2],
                "box_h": result["box"][3],
                "left_eye_x": kp["left_eye"][0],
                "left_eye_y": kp["left_eye"][1],
                "right_eye_x": kp["right_eye"][0],
                "right_eye_y": kp["right_eye"][1],
                "nose_x": kp["nose"][0],
                "nose_y": kp["nose"][1],
                "mouth_l_x": kp["mouth_left"][0],
                "mouth_l_y": kp["mouth_left"][1],
                "mouth_r_x": kp["mouth_right"][0],
                "mouth_r_y": kp["mouth_right"][1],
            })

            if (idx + 1) % 100 == 0 or (idx + 1) == total_images:
                print(f"  Processed {idx + 1}/{total_images} images (Detected: {len(detection_records)}, Failures: {len(failures)})")

    finally:
        detector.close()

    # Save failure log (always create CSV with header)
    fail_df = pd.DataFrame(failures, columns=["filename", "image_path", "reason"])
    fail_df.to_csv(config.FACE_DETECTION_FAILURES_CSV, index=False)
    print(f"[OK] Failures log written -> {config.FACE_DETECTION_FAILURES_CSV} ({len(failures)} failures)")

    # Save detection metadata with bounding boxes and landmarks
    det_df = pd.DataFrame(detection_records)
    det_csv = os.path.join(config.METADATA_DIR, "kmu_fed_detected_faces.csv")
    det_df.to_csv(det_csv, index=False)
    print(f"[OK] Detection metadata written -> {det_csv} ({len(det_df)} faces)")

    # Print summary
    rate = (len(detection_records) / total_images) * 100 if total_images else 0
    print("=" * 55)
    print("  Phase 3 Complete")
    print(f"  Total processed : {total_images}")
    print(f"  Detected faces  : {len(detection_records)}")
    print(f"  Failed          : {len(failures)}")
    print(f"  Detection rate  : {rate:.2f}%")
    print(f"  Faces stored at : {config.DETECTED_FACES_DIR}")
    print("=" * 55)


if __name__ == "__main__":
    run_phase3()

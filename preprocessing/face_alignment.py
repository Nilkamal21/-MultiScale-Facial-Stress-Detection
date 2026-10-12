"""
preprocessing/face_alignment.py
Phase 4: Align detected faces by rotating eye-to-eye axis to horizontal.
Saves aligned faces to data/aligned_faces/.
See explainable.md > Phase 4 for details.
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def compute_eye_alignment_angle(lx, ly, rx, ry):
    """Compute rotation angle in degrees so eye axis becomes horizontal."""
    dx = rx - lx
    dy = ry - ly
    return float(np.degrees(np.arctan2(dy, dx)))


def align_face(img, angle, center=None):
    """Rotate image around center point using affine warp with reflected borders."""
    h, w = img.shape[:2]
    if center is None:
        center = (w / 2.0, h / 2.0)

    # Negative angle rotates image back to upright horizontal
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return aligned


def run_phase4():
    print("=" * 55)
    print("  KMU-FED Phase 4 — Face Alignment")
    print(f"  Input detected faces : {config.DETECTED_FACES_DIR}")
    print(f"  Target aligned faces : {config.ALIGNED_FACES_DIR}")
    print(f"  Failures log         : {config.ALIGNMENT_FAILURES_CSV}")
    print("=" * 55)

    det_csv = os.path.join(config.METADATA_DIR, "kmu_fed_detected_faces.csv")
    if not os.path.exists(det_csv):
        raise FileNotFoundError(f"Missing detected faces metadata: {det_csv}. Run Phase 3 first.")

    df = pd.read_csv(det_csv)
    os.makedirs(config.ALIGNED_FACES_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    aligned_records = []
    failures = []

    total_faces = len(df)
    for idx, row in df.iterrows():
        filename = row["filename"]
        seq_id = str(row["sequence_id"])
        face_path = row["detected_face_path"]

        # Resolve path
        if not os.path.exists(face_path):
            face_path = os.path.join(config.DETECTED_FACES_DIR, seq_id, filename)

        if not os.path.exists(face_path):
            failures.append({
                "filename": filename,
                "face_path": face_path,
                "reason": "Cropped face image not found"
            })
            continue

        img = cv2.imread(face_path)
        if img is None:
            failures.append({
                "filename": filename,
                "face_path": face_path,
                "reason": "cv2 failed to decode face image"
            })
            continue

        # Calculate alignment angle from eye landmark coordinates
        angle = compute_eye_alignment_angle(
            row["left_eye_x"], row["left_eye_y"],
            row["right_eye_x"], row["right_eye_y"]
        )

        # Align face image
        aligned_img = align_face(img, angle)

        dest_dir = os.path.join(config.ALIGNED_FACES_DIR, seq_id)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, filename)

        cv2.imwrite(dest_path, aligned_img)

        aligned_records.append({
            "sequence_id": seq_id,
            "filename": filename,
            "subject_id": row["subject_id"],
            "emotion": row["emotion"],
            "aligned_face_path": dest_path,
            "rotation_angle_deg": round(angle, 2),
            "width": aligned_img.shape[1],
            "height": aligned_img.shape[0],
        })

        if (idx + 1) % 200 == 0 or (idx + 1) == total_faces:
            print(f"  Aligned {idx + 1}/{total_faces} faces...")

    # Save alignment failure log
    fail_df = pd.DataFrame(failures, columns=["filename", "face_path", "reason"])
    fail_df.to_csv(config.ALIGNMENT_FAILURES_CSV, index=False)
    print(f"[OK] Alignment failure log written -> {config.ALIGNMENT_FAILURES_CSV} ({len(failures)} failures)")

    # Save alignment metadata
    aligned_df = pd.DataFrame(aligned_records)
    out_csv = os.path.join(config.METADATA_DIR, "kmu_fed_aligned_faces.csv")
    aligned_df.to_csv(out_csv, index=False)
    print(f"[OK] Alignment metadata written -> {out_csv} ({len(aligned_df)} rows)")

    # Summary
    print("=" * 55)
    print("  Phase 4 Complete")
    print(f"  Input faces   : {total_faces}")
    print(f"  Aligned faces : {len(aligned_df)}")
    print(f"  Failures      : {len(failures)}")
    print(f"  Output folder : {config.ALIGNED_FACES_DIR}")
    print("=" * 55)


if __name__ == "__main__":
    run_phase4()

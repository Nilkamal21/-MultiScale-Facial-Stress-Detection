"""
preprocessing/extract_ubfc_faces.py
Phase 1 — Step 1: Video Frame & Face Extraction for UBFC Dataset.

Extracts 224x224 facial image frames from raw UBFC subject videos (vid.avi) for
unsupervised SimCLR pre-training.
Outputs:
- Aligned 224x224 facial image frames saved to: /kaggle/working/ubfc_faces/{subject_id}/
- Master metadata CSV saved to: data/metadata/ubfc_frames_metadata.csv
"""

import os
import sys

# CRITICAL: Force FFmpeg and disable GStreamer/OpenCV multithreading before importing cv2
os.environ["OPENCV_VIDEOIO_PRIORITY_BACKENDS"] = "FFMPEG"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"

import glob
import time
import argparse
import subprocess
import gc
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def get_face_detector():
    """Returns OpenCV's frontal face cascade classifier or None."""
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(cascade_path)
        if detector.empty():
            return None
        return detector
    except Exception:
        return None


def crop_face_box(frame, bbox, margin=0.10, target_size=(224, 224)):
    """
    Crops the face bounding box with safety margins and resizes to target_size.
    bbox: (x, y, w, h)
    """
    h, w = frame.shape[:2]
    x, y, bw, bh = bbox
    pad_w = int(bw * margin)
    pad_h = int(bh * margin)

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(w, x + bw + pad_w)
    y2 = min(h, y + bh + pad_h)

    face = frame[y1:y2, x1:x2]
    if face is None or face.size == 0 or face.shape[0] < 10 or face.shape[1] < 10:
        return None
    return cv2.resize(face, target_size, interpolation=cv2.INTER_AREA)


def extract_subject_faces(
    subject_dir,
    output_base_dir,
    detector=None,
    frame_stride=5,
    target_size=(224, 224),
    detect_interval=15
):
    """
    Extracts face frames from a single UBFC subject directory.
    Uses cv2.CAP_FFMPEG with automatic FFmpeg subprocess fallback.
    """
    subject_name = os.path.basename(subject_dir.rstrip("\\/"))

    # Locate video file
    video_candidates = (
        glob.glob(os.path.join(subject_dir, "*.avi")) +
        glob.glob(os.path.join(subject_dir, "*.mp4")) +
        glob.glob(os.path.join(subject_dir, "*.mov"))
    )
    if not video_candidates:
        return []

    video_path = video_candidates[0]
    subj_out_dir = os.path.join(output_base_dir, subject_name)
    os.makedirs(subj_out_dir, exist_ok=True)

    records = []
    frame_idx = 0
    saved_count = 0
    last_bbox = None

    # Method A: Try OpenCV with explicit FFmpeg backend
    cap = None
    try:
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(video_path)
    except Exception:
        cap = None

    if cap is not None and cap.isOpened():
        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                if frame_idx % frame_stride == 0:
                    h, w = frame.shape[:2]
                    dim = int(min(h, w) * 0.70)
                    center_box = ((w - dim) // 2, (h - dim) // 2, dim, dim)

                    # Update face box on interval
                    if (frame_idx % (frame_stride * detect_interval) == 0 or last_bbox is None) and detector is not None:
                        try:
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            faces = detector.detectMultiScale(gray, 1.2, 3, minSize=(60, 60))
                            if len(faces) > 0:
                                faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
                                last_bbox = tuple(faces[0])
                            elif last_bbox is None:
                                last_bbox = center_box
                        except Exception:
                            if last_bbox is None:
                                last_bbox = center_box
                    elif last_bbox is None:
                        last_bbox = center_box

                    face_crop = crop_face_box(frame, last_bbox, target_size=target_size)
                    if face_crop is not None:
                        img_name = f"frame_{frame_idx:06d}.jpg"
                        save_path = os.path.join(subj_out_dir, img_name)
                        cv2.imwrite(save_path, face_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

                        records.append({
                            "subject_id": subject_name,
                            "frame_idx": frame_idx,
                            "relative_path": os.path.join(subject_name, img_name),
                            "full_path": save_path,
                        })
                        saved_count += 1

                frame_idx += 1
        except Exception as e:
            print(f"[OpenCV read interrupted on {subject_name}]: {e}")
        finally:
            cap.release()
            del cap
            gc.collect()

    # Method B: Robust FFmpeg CLI Fallback if OpenCV yielded zero frames
    if saved_count == 0:
        target_fps = max(1, 30 // frame_stride)
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vf", f"fps={target_fps},crop='min(iw,ih)*0.75':'min(iw,ih)*0.75',scale={target_size[0]}:{target_size[1]}",
            os.path.join(subj_out_dir, "frame_%06d.jpg"),
            "-loglevel", "error"
        ]
        try:
            subprocess.run(ffmpeg_cmd, check=True)
            for f in sorted(os.listdir(subj_out_dir)):
                if f.lower().endswith(".jpg"):
                    f_path = os.path.join(subj_out_dir, f)
                    records.append({
                        "subject_id": subject_name,
                        "relative_path": os.path.join(subject_name, f),
                        "full_path": f_path,
                    })
                    saved_count += 1
        except Exception as e:
            print(f"[FFmpeg fallback error on {subject_name}]: {e}")

    return records


def find_ubfc_subject_dirs(raw_dir=None):
    """
    Intelligently searches for directories containing video files (vid.avi, *.avi, *.mp4)
    and ground_truth.txt. Supports flat, nested, or auto-detected Kaggle directory structures.
    """
    candidate_roots = []
    if raw_dir and os.path.exists(raw_dir):
        candidate_roots.append(raw_dir)

    # Auto-scan /kaggle/input if running on Kaggle
    if os.path.exists("/kaggle/input"):
        for entry in os.listdir("/kaggle/input"):
            full_e = os.path.join("/kaggle/input", entry)
            if os.path.isdir(full_e) and "raf" not in entry.lower():
                candidate_roots.append(full_e)

    subject_dirs = []
    for root_cand in candidate_roots:
        for cur_root, dirs, files in os.walk(root_cand):
            has_video = any(f.lower().endswith((".avi", ".mp4", ".mov")) for f in files)
            if has_video:
                subject_dirs.append(cur_root)

    # De-duplicate while preserving order
    seen = set()
    unique_dirs = []
    for d in sorted(subject_dirs):
        norm = os.path.abspath(d)
        if norm not in seen:
            seen.add(norm)
            unique_dirs.append(norm)

    return unique_dirs


def run_ubfc_extraction(
    raw_dir=None,
    output_dir=None,
    frame_stride=5,
    max_subjects=None
):
    """
    Main driver for Step 1: UBFC Face Video Frame Extraction.
    """
    print("=" * 65)
    print("  PHASE 1 — STEP 1: UBFC VIDEO FACE FRAME EXTRACTION")
    print("=" * 65)

    if raw_dir is None:
        raw_dir = config.UBFC_RAW_DIR
    if output_dir is None:
        output_dir = "/kaggle/working/ubfc_faces" if os.path.exists("/kaggle") else os.path.join(config.DATA_DIR, "ubfc_faces")

    metadata_dir = os.path.join(config.DATA_DIR, "metadata")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    print(f"[Input Raw Video Dir] : {raw_dir}")
    print(f"[Output Faces Dir]    : {output_dir}")
    print(f"[Frame Stride]        : {frame_stride} (~6 FPS)")

    subject_dirs = find_ubfc_subject_dirs(raw_dir)

    if not subject_dirs:
        print(f"\n[INFO] No UBFC subject video directories found in: {raw_dir}")
        if os.path.exists("/kaggle/input"):
            print("Contents of /kaggle/input:")
            for item in os.listdir("/kaggle/input"):
                print(f"  - /kaggle/input/{item}")
        print("Please check that the UBFC dataset has finished downloading and is mounted in Kaggle!")
        return

    if max_subjects:
        subject_dirs = subject_dirs[:max_subjects]

    print(f"[Subjects Found]      : {len(subject_dirs)} subjects to process")
    for s in subject_dirs[:3]:
        print(f"  -> Sample: {s}")
    if len(subject_dirs) > 3:
        print(f"  -> ... and {len(subject_dirs) - 3} more subjects.\n")

    detector = get_face_detector()
    all_records = []
    t0 = time.time()

    for s_dir in tqdm(subject_dirs, desc="Processing Subjects"):
        s_records = extract_subject_faces(
            s_dir,
            output_dir,
            detector=detector,
            frame_stride=frame_stride,
            target_size=(config.INPUT_SIZE, config.INPUT_SIZE)
        )
        all_records.extend(s_records)

    elapsed = time.time() - t0
    df = pd.DataFrame(all_records)
    csv_path = os.path.join(metadata_dir, "ubfc_frames_metadata.csv")
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 65)
    print("  STEP 1 EXTRACTION COMPLETED SUCCESSFULLY")
    print(f"  - Total Frames Saved : {len(df):,}")
    print(f"  - Subjects Processed : {len(subject_dirs)}")
    print(f"  - Processing Time    : {elapsed:.2f} seconds")
    print(f"  - Frames Saved To    : {output_dir}")
    print(f"  - Metadata Saved     : {csv_path}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract UBFC Face Video Frames")
    parser.add_argument("--raw_dir", type=str, default=None, help="Path to raw UBFC dataset")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for cropped faces")
    parser.add_argument("--frame_stride", type=int, default=5, help="Sampling stride (default: 5 = 6 FPS)")
    parser.add_argument("--max_subjects", type=int, default=None, help="Limit number of subjects (testing)")
    args = parser.parse_args()

    run_ubfc_extraction(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        frame_stride=args.frame_stride,
        max_subjects=args.max_subjects
    )

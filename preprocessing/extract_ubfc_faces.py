"""
preprocessing/extract_ubfc_faces.py
Phase 1 — Step 1: Video Frame & Face Extraction for UBFC Dataset.

Extracts 224x224 facial image frames from raw UBFC subject videos (vid.avi) for
unsupervised SimCLR pre-training.
Uses native FFmpeg for ultra-fast, robust, crash-free frame extraction directly on Kaggle.
"""

import os
import sys
import glob
import time
import argparse
import subprocess
import shutil
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def extract_subject_faces_ffmpeg(
    subject_dir,
    output_base_dir,
    target_fps=6,
    target_size=(224, 224)
):
    """
    Extracts face frames from a single UBFC subject directory using native FFmpeg.
    In UBFC, each subject is seated directly in front of the camera.
    A centered 75% crop captures the face, eyes, and expressions perfectly.
    """
    subject_name = os.path.basename(subject_dir.rstrip("\\/"))

    # Case-insensitive video file discovery
    video_candidates = [
        os.path.join(subject_dir, f) for f in os.listdir(subject_dir)
        if f.lower().endswith((".avi", ".mp4", ".mov", ".mkv"))
    ]
    if not video_candidates:
        return []

    video_path = video_candidates[0]
    subj_out_dir = os.path.join(output_base_dir, subject_name)
    os.makedirs(subj_out_dir, exist_ok=True)

    records = []
    w_out, h_out = target_size

    # Native FFmpeg filter without shell quotes:
    # 1. Sample at target_fps (default: 6 FPS = plenty of temporal variation)
    # 2. Crop central 75% face region: ih*0.75
    # 3. Scale to exact target_size (224x224)
    vf_filter = f"fps={target_fps},crop=ih*0.75:ih*0.75:(iw-ih*0.75)/2:ih*0.08,scale={w_out}:{h_out}"
    out_pattern = os.path.join(subj_out_dir, "frame_%06d.jpg")

    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"

    cmd = [
        ffmpeg_bin, "-y",
        "-i", video_path,
        "-vf", vf_filter,
        "-q:v", "2",
        out_pattern,
        "-loglevel", "error"
    ]

    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0:
            # Fallback simple scale filter if crop was out of bounds
            simple_vf = f"fps={target_fps},scale={w_out}:{h_out}"
            cmd[5] = simple_vf
            res2 = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res2.returncode != 0:
                print(f"[FFmpeg error on {subject_name}]: {res2.stderr.decode('utf-8', errors='ignore').strip()}")

        for fname in sorted(os.listdir(subj_out_dir)):
            if fname.lower().endswith(".jpg"):
                full_path = os.path.join(subj_out_dir, fname)
                records.append({
                    "subject_id": subject_name,
                    "filename": fname,
                    "relative_path": os.path.join(subject_name, fname),
                    "full_path": full_path,
                })
    except Exception as e:
        print(f"[Extraction error on {subject_name}]: {e}")

    return records


def find_ubfc_subject_dirs(raw_dir=None):
    """
    Intelligently searches for directories containing video files (vid.avi, *.avi, *.mp4).
    Supports flat, nested, or auto-detected Kaggle directory structures.
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
    fps=6,
    max_subjects=None
):
    """
    Main driver for Step 1: UBFC Face Video Frame Extraction.
    """
    print("=" * 65)
    print("  PHASE 1 — STEP 1: UBFC VIDEO FACE FRAME EXTRACTION (FFMPEG)")
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
    print(f"[Extraction Rate]     : {fps} FPS (Optimal for facial pretraining)")

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

    all_records = []
    t0 = time.time()

    for s_dir in tqdm(subject_dirs, desc="Processing Subjects"):
        s_records = extract_subject_faces_ffmpeg(
            s_dir,
            output_dir,
            target_fps=fps,
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
    parser = argparse.ArgumentParser(description="Extract UBFC Face Video Frames via FFmpeg")
    parser.add_argument("--raw_dir", type=str, default=None, help="Path to raw UBFC dataset")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for cropped faces")
    parser.add_argument("--fps", type=int, default=6, help="Target FPS to sample (default: 6)")
    parser.add_argument("--max_subjects", type=int, default=None, help="Limit number of subjects (testing)")
    args = parser.parse_args()

    run_ubfc_extraction(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        fps=args.fps,
        max_subjects=args.max_subjects
    )

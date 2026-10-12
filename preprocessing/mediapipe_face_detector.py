"""
preprocessing/mediapipe_face_detector.py
Central MediaPipe Face Detector wrapper replacing legacy MTCNN.
Uses Google MediaPipe BlazeFace Short Range LiteRT model for ultra-fast,
robust face and 6-landmark facial feature detection.
"""

import os
import sys
import urllib.request
import numpy as np
import cv2
from PIL import Image

# Suppress underlying C++ and TensorFlow logging noise
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import mediapipe as mp  # type: ignore
from mediapipe.tasks import python  # type: ignore
from mediapipe.tasks.python import vision  # type: ignore

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


MODEL_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)


class MediaPipeFaceDetector:
    """
    Standardized face detector backed by MediaPipe BlazeFace.
    Provides API compatibility with MTCNN outputs while running ~50x faster.
    """

    def __init__(self, model_path=None, min_detection_confidence=None):
        self.min_confidence = (
            min_detection_confidence
            if min_detection_confidence is not None
            else getattr(config, "MEDIAPIPE_MIN_CONFIDENCE", 0.50)
        )
        self.model_path = self._resolve_model_path(model_path)
        self._init_detector()

    def _resolve_model_path(self, model_path):
        """Locates or downloads the MediaPipe BlazeFace short-range tflite model."""
        candidate_paths = [
            model_path,
            getattr(config, "MEDIAPIPE_MODEL_PATH", None),
            os.path.join(config.CHECKPOINTS_DIR, "blaze_face_short_range.tflite"),
            os.path.join(config.PROJECT_ROOT, "blaze_face_short_range.tflite"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "blaze_face_short_range.tflite"),
        ]

        for p in candidate_paths:
            if p and os.path.exists(p) and os.path.getsize(p) > 10000:
                return os.path.abspath(p)

        # Download to CHECKPOINTS_DIR if not found locally
        target_path = os.path.join(config.CHECKPOINTS_DIR, "blaze_face_short_range.tflite")
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        print(f"[MediaPipe] Downloading BlazeFace model to {target_path}...")
        try:
            urllib.request.urlretrieve(MODEL_DOWNLOAD_URL, target_path)
            print("[MediaPipe] Model downloaded successfully.")
            return os.path.abspath(target_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download MediaPipe model from {MODEL_DOWNLOAD_URL}: {e}"
            ) from e

    def _init_detector(self):
        """Initializes the MediaPipe Vision FaceDetector instance."""
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=self.min_confidence,
        )
        self.detector = vision.FaceDetector.create_from_options(options)

    def detect_faces(self, image):
        """
        Detects faces in image (accepts BGR numpy array, RGB numpy array, or PIL Image).
        Returns list of detections with structure compatible with MTCNN:
        [
            {
                "box": (x, y, w, h),
                "confidence": float,
                "keypoints": {
                    "left_eye": (x, y),       # viewer's left
                    "right_eye": (x, y),      # viewer's right
                    "nose": (x, y),
                    "mouth_left": (x, y),
                    "mouth_right": (x, y),
                    "mouth_center": (x, y),
                    "right_ear": (x, y),
                    "left_ear": (x, y)
                }
            }
        ]
        """
        if self.detector is None:
            return []

        # Convert input to RGB uint8 numpy array
        if isinstance(image, Image.Image):
            rgb = np.array(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:
                rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 3:
                # Assume contiguous array
                rgb = image
            else:
                rgb = image[:, :, :3]
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        h, w = rgb.shape[:2]
        rgb_contiguous = np.ascontiguousarray(rgb, dtype=np.uint8)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_contiguous)
        detection_result = self.detector.detect(mp_image)

        if not detection_result or not detection_result.detections:
            return []

        results = []
        for det in detection_result.detections:
            bb = det.bounding_box
            bx = max(0, int(bb.origin_x))
            by = max(0, int(bb.origin_y))
            bw = min(w - bx, int(bb.width))
            bh = min(h - by, int(bb.height))

            conf = float(det.categories[0].score) if det.categories else 0.0

            keypoints = {}
            if det.keypoints and len(det.keypoints) >= 4:
                # BlazeFace keypoints:
                # 0: Right eye (subject's right, viewer's left)
                # 1: Left eye (subject's left, viewer's right)
                # 2: Nose tip
                # 3: Mouth center
                # 4: Right ear tragion (viewer's left)
                # 5: Left ear tragion (viewer's right)
                lx, ly = int(det.keypoints[0].x * w), int(det.keypoints[0].y * h)
                rx, ry = int(det.keypoints[1].x * w), int(det.keypoints[1].y * h)
                nx, ny = int(det.keypoints[2].x * w), int(det.keypoints[2].y * h)
                mcx, mcy = int(det.keypoints[3].x * w), int(det.keypoints[3].y * h)

                eye_dist = max(1, abs(rx - lx))
                mlx = max(0, mcx - int(eye_dist * 0.25))
                mrx = min(w - 1, mcx + int(eye_dist * 0.25))

                keypoints["left_eye"] = (lx, ly)
                keypoints["right_eye"] = (rx, ry)
                keypoints["nose"] = (nx, ny)
                keypoints["mouth_center"] = (mcx, mcy)
                keypoints["mouth_left"] = (mlx, mcy)
                keypoints["mouth_right"] = (mrx, mcy)

                if len(det.keypoints) >= 6:
                    keypoints["right_ear"] = (int(det.keypoints[4].x * w), int(det.keypoints[4].y * h))
                    keypoints["left_ear"] = (int(det.keypoints[5].x * w), int(det.keypoints[5].y * h))

            results.append({
                "box": (bx, by, bw, bh),
                "confidence": conf,
                "keypoints": keypoints
            })

        return results

    def detect_primary_face(self, image):
        """Returns the primary face detection (highest confidence * area) or None."""
        detections = self.detect_faces(image)
        if not detections:
            return None
        return max(detections, key=lambda d: d["confidence"] * (d["box"][2] * d["box"][3]))

    def close(self):
        """Safely closes MediaPipe detector instance."""
        det = getattr(self, "detector", None)
        if det is not None:
            self.detector = None
            try:
                det.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

"""
config.py — Central configuration for the KMU-FED pipeline.
All paths, constants, and hyperparameters live here.
"""

import os

# --- Reproducibility ---
RANDOM_SEED = 42

# --- Project root (Dynamic, works on any computer or Kaggle) ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "KMU-FED")

# --- Intermediate output directories ---
DATA_DIR            = os.path.join(PROJECT_ROOT, "data")
FRAMES_DIR          = os.path.join(DATA_DIR, "frames")
DETECTED_FACES_DIR  = os.path.join(DATA_DIR, "detected_faces")
ALIGNED_FACES_DIR   = os.path.join(DATA_DIR, "aligned_faces")
PROCESSED_DIR       = os.path.join(DATA_DIR, "processed")

# --- Metadata ---
METADATA_DIR        = os.path.join(PROJECT_ROOT, "data", "metadata")
METADATA_CSV        = os.path.join(METADATA_DIR, "kmu_fed_metadata.csv")
STATISTICS_REPORT   = os.path.join(METADATA_DIR, "dataset_statistics_report.txt")
TRAIN_CSV           = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV             = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV            = os.path.join(METADATA_DIR, "test.csv")

# --- Logs ---
LOGS_DIR                    = os.path.join(PROJECT_ROOT, "logs")
FACE_DETECTION_FAILURES_CSV = os.path.join(LOGS_DIR, "face_detection_failures.csv")
ALIGNMENT_FAILURES_CSV      = os.path.join(LOGS_DIR, "alignment_failures.csv")

# --- Features and checkpoints ---
FEATURES_DIR    = os.path.join(PROJECT_ROOT, "features")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "models_checkpoints")

# --- Face Detection (MediaPipe BlazeFace) ---
MEDIAPIPE_MODEL_PATH     = os.path.join(CHECKPOINTS_DIR, "blaze_face_short_range.tflite")
MEDIAPIPE_MIN_CONFIDENCE = 0.50

# --- Results ---
RESULTS_DIR     = os.path.join(PROJECT_ROOT, "results")
RESULTS_KMU_FED = os.path.join(RESULTS_DIR, "kmu_fed")
RESULTS_RAF_DB  = os.path.join(RESULTS_DIR, "raf_db")

# --- Filename convention: {SubjectID}_{EmotionCode}_{PersonCode}_{FrameNumber}.jpg ---
EMOTION_CODE_MAP = {
    "AN": "Anger",
    "DI": "Disgust",
    "FE": "Fear",
    "HA": "Happiness",
    "SA": "Sadness",
    "SU": "Surprise",
}

# Files starting with this prefix are byte-for-byte duplicates of 01_AN_mr_* (Kaggle artifact)
DUPLICATE_PREFIX_EXCLUDE = "1_AN_mr_"

# --- EfficientNet input ---
EFFICIENTNET_VARIANT = "efficientnet_b0"
INPUT_SIZE           = 224
IMAGE_CHANNELS       = 3
IMAGENET_MEAN        = [0.485, 0.456, 0.406]
IMAGENET_STD         = [0.229, 0.224, 0.225]

# --- Augmentation (see explainable.md for descriptions) ---
AUG_HORIZONTAL_FLIP_PROB   = 0.5
AUG_ROTATION_DEGREES       = 10
AUG_RANDOM_CROP_SCALE      = (0.85, 1.0)
AUG_BRIGHTNESS_FACTOR      = 0.2
AUG_CONTRAST_FACTOR        = 0.2
AUG_GAUSSIAN_BLUR_ENABLED  = True
AUG_GAUSSIAN_BLUR_KERNEL   = (5, 5)
AUG_GAUSSIAN_BLUR_SIGMA    = (0.1, 2.0)
AUG_GAUSSIAN_BLUR_PROB     = 0.3
AUG_GAUSSIAN_NOISE_ENABLED = True
AUG_GAUSSIAN_NOISE_STD     = 0.02
AUG_GAUSSIAN_NOISE_PROB    = 0.3

# --- Subject-wise split ratios (must sum to 1.0) ---
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# --- Contrastive learning ---
CONTRASTIVE_LOSS_TYPE   = "contrastive"   # "contrastive" or "ntxent"
CONTRASTIVE_MARGIN      = 1.0
CONTRASTIVE_TEMPERATURE = 0.07
CONTRASTIVE_BATCH_SIZE  = 32
CONTRASTIVE_EPOCHS      = 50
CONTRASTIVE_LR          = 1e-4
PROJECTION_HEAD_DIM     = 128
FREEZE_BACKBONE         = True

# --- Classifier ---
CLASSIFIER_HIDDEN_DIM   = 256
CLASSIFIER_DROPOUT      = 0.4
CLASSIFIER_BATCH_SIZE   = 32
STAGE_A_EPOCHS          = 20
STAGE_A_LR              = 1e-3
STAGE_B_EPOCHS          = 20
STAGE_B_LR              = 1e-5
STAGE_B_UNFREEZE_LAYERS = 3

# --- Evaluation and visualization ---
EVAL_METRICS        = ["accuracy", "precision", "recall", "f1"]
VIZ_TSNE_PERPLEXITY = 30
VIZ_TSNE_N_ITER     = 1000
VIZ_PCA_COMPONENTS  = 50
VIZ_UMAP_ENABLED    = False

# ==============================================================================
# UBFC CARDIOVASCULAR STRESS & HEART RATE CONFIGURATION
# ==============================================================================
KAGGLE_INPUT_DIR    = "/kaggle/input/ubfc-rppg-dataset"
KAGGLE_WORKING_DIR  = "/kaggle/working"
IS_KAGGLE           = os.path.exists("/kaggle")
UBFC_RAW_DIR        = KAGGLE_INPUT_DIR if os.path.exists(KAGGLE_INPUT_DIR) else os.path.join(DATA_DIR, "ubfc_raw")

VIDEO_FPS          = 30
WINDOW_SIZE        = 100
WINDOW_STRIDE      = 30
FACE_CROP_SIZE     = 224
FACE_MARGIN_RATIO  = 0.10

BANDPASS_LOW_HZ    = 0.70
BANDPASS_HIGH_HZ   = 3.50
BUTTERWORTH_ORDER  = 4

TOTAL_SUBJECTS = 42
TRAIN_SUBJECTS = [f"subject{i}" for i in range(1, 33)]
VAL_SUBJECTS   = [f"subject{i}" for i in range(33, 38)]
TEST_SUBJECTS  = [f"subject{i}" for i in range(38, 43)]

BATCH_SIZE         = 16
LEARNING_RATE      = 1e-3
WEIGHT_DECAY       = 1e-4
EPOCHS             = 20
EARLY_STOP_PATIENCE = 5

INPUT_DIM          = 4
INPUT_FEATURE_DIM  = 64
GRU_HIDDEN_DIM     = 64
GRU_NUM_LAYERS     = 2
DROPOUT_RATE       = 0.3

STRESS_ELEVATION_THRESHOLD = 0.15
STRESS_DECISION_THRESHOLD  = 0.48
FOCAL_LOSS_GAMMA           = 2.0
FOCAL_LOSS_ALPHA           = 0.80


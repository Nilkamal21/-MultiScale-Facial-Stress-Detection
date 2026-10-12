"""
config.py — Central Configuration for UBFC Cardiovascular Stress & Pulse System.
Contains all paths, hyperparameters, and biomedical signal settings.
Works seamlessly both on local machines and inside Kaggle notebooks.
"""

import os

# ==============================================================================
# 1. DYNAMIC PATH CONFIGURATION (No hardcoding - works on any computer or cloud)
# ==============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Local directories
DATA_DIR            = os.path.join(PROJECT_ROOT, "data")
CHECKPOINTS_DIR     = os.path.join(PROJECT_ROOT, "models_checkpoints")
RESULTS_DIR         = os.path.join(PROJECT_ROOT, "results")
LOGS_DIR            = os.path.join(PROJECT_ROOT, "logs")

# Kaggle dataset path (used when running inside a Kaggle notebook)
KAGGLE_INPUT_DIR    = "/kaggle/input/ubfc-rppg-dataset"
KAGGLE_WORKING_DIR  = "/kaggle/working"

# Check if running in Kaggle environment or locally
IS_KAGGLE = os.path.exists("/kaggle")
UBFC_RAW_DIR = KAGGLE_INPUT_DIR if os.path.exists(KAGGLE_INPUT_DIR) else os.path.join(DATA_DIR, "ubfc_raw")

# Ensure local directories exist
for folder in [DATA_DIR, CHECKPOINTS_DIR, RESULTS_DIR, LOGS_DIR]:
    os.makedirs(folder, exist_ok=True)

# ==============================================================================
# 2. MEDIAPIPE FACE DETECTION SETTINGS
# ==============================================================================
MEDIAPIPE_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, "blaze_face_short_range.tflite")
MEDIAPIPE_MIN_CONFIDENCE = 0.50

# ==============================================================================
# 3. VIDEO & TEMPORAL WINDOW SETTINGS
# ==============================================================================
VIDEO_FPS          = 30          # UBFC webcam framerate is 30 frames per second
WINDOW_SIZE        = 100         # Slices video into 100-frame windows (~3.33 seconds)
WINDOW_STRIDE      = 30          # Slides window forward by 30 frames (1.0 second stride)
FACE_CROP_SIZE     = 224         # Standardized square face crop (224x224 pixels)
FACE_MARGIN_RATIO  = 0.10        # 10% padding margin around detected face box

# ==============================================================================
# 4. BIOMEDICAL PULSE (rPPG) FILTERING SETTINGS
# ==============================================================================
# Human heart rate typically ranges between 42 BPM and 210 BPM.
# 42 BPM / 60 = 0.70 Hz
# 210 BPM / 60 = 3.50 Hz
BANDPASS_LOW_HZ    = 0.70        # Lower frequency bound (42 BPM)
BANDPASS_HIGH_HZ   = 3.50        # Upper frequency bound (210 BPM)
BUTTERWORTH_ORDER  = 4           # Filter sharpness order

# ==============================================================================
# 5. SUBJECT-INDEPENDENT DATA SPLITS (Total 42 Subjects)
# ==============================================================================
# Zero data leakage: Test subjects are complete strangers to the training set!
TOTAL_SUBJECTS = 42
TRAIN_SUBJECTS = [f"subject{i}" for i in range(1, 33)]    # 32 People (76%)
VAL_SUBJECTS   = [f"subject{i}" for i in range(33, 38)]   # 5 People (12%)
TEST_SUBJECTS  = [f"subject{i}" for i in range(38, 43)]   # 5 People (12%)

# ==============================================================================
# 6. NEURAL NETWORK & TRAINING HYPERPARAMETERS
# ==============================================================================
RANDOM_SEED        = 42
BATCH_SIZE         = 16          # Mini-batch size
LEARNING_RATE      = 1e-3        # AdamW learning rate
WEIGHT_DECAY       = 1e-4        # L2 Regularization
EPOCHS             = 20          # Maximum training epochs
EARLY_STOP_PATIENCE = 5          # Stop training if validation loss plateaus for 5 epochs

# BiGRU Architecture settings
INPUT_DIM          = 4           # Normalized R, G, B + Filtered CHROM BVP pulse wave
INPUT_FEATURE_DIM  = 64          # Spatial feature embedding dimension per frame
GRU_HIDDEN_DIM     = 64          # Hidden units per GRU direction (x2 for Bidirectional = 128)
GRU_NUM_LAYERS     = 2           # Number of stacked GRU layers
DROPOUT_RATE       = 0.3         # Dropout probability to prevent overfitting

# ==============================================================================
# 7. STRESS THRESHOLDS & FOCAL LOSS SETTINGS
# ==============================================================================
# In UBFC, Task 2 involves a time-pressured math quiz that spikes heart rate.
# A heart rate jump > 15% above resting baseline indicates acute mental stress.
STRESS_ELEVATION_THRESHOLD = 0.15
STRESS_DECISION_THRESHOLD  = 0.55 # Calibrated probability threshold for stress classification
FOCAL_LOSS_GAMMA           = 2.0  # Focusing parameter for hard boundary examples
FOCAL_LOSS_ALPHA           = 0.35 # Balanced positive class scaling

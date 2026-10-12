"""
preprocessing/rppg_extract.py — Facial Skin ROI & Pulse Signal Extractor.
Extracts Blood Volume Pulse (BVP) and heart rate from facial video frames
using Google MediaPipe, skin color normalization (CHROM), and biological bandpass filtering.
"""

import os
import sys
import numpy as np
import cv2
from scipy.signal import butter, filtfilt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ==============================================================================
# 1. FACIAL SKIN REGION OF INTEREST (ROI) EXTRACTION
# ==============================================================================
def extract_skin_rois(image_rgb, box, keypoints=None):
    """
    Isolates the forehead and cheeks (where capillaries and pulse are strongest),
    avoiding eyes (blinking), mouth (talking), hair, and glasses.

    Args:
        image_rgb: RGB image (H, W, 3) as numpy uint8 array.
        box: (x, y, w, h) bounding box of the face.
        keypoints: dictionary of facial landmarks from MediaPipe (optional).

    Returns:
        list of cropped skin patches [(H1, W1, 3), (H2, W2, 3), ...]
    """
    h_img, w_img = image_rgb.shape[:2]
    bx, by, bw, bh = box

    rois = []

    # 1. Forehead ROI: Centered above eyebrows (upper 15% to 30% of face box)
    fx = max(0, int(bx + bw * 0.25))
    fy = max(0, int(by + bh * 0.10))
    fw = int(bw * 0.50)
    fh = int(bh * 0.18)
    if fw > 5 and fh > 5 and fy + fh <= h_img and fx + fw <= w_img:
        forehead = image_rgb[fy : fy + fh, fx : fx + fw]
        rois.append(forehead)

    # 2. Left Cheek ROI: Beside nose, below eye
    lc_x = max(0, int(bx + bw * 0.15))
    lc_y = max(0, int(by + bh * 0.45))
    lc_w = int(bw * 0.25)
    lc_h = int(bh * 0.25)
    if lc_w > 5 and lc_h > 5 and lc_y + lc_h <= h_img and lc_x + lc_w <= w_img:
        left_cheek = image_rgb[lc_y : lc_y + lc_h, lc_x : lc_x + lc_w]
        rois.append(left_cheek)

    # 3. Right Cheek ROI: Beside nose, below eye
    rc_x = max(0, int(bx + bw * 0.60))
    rc_y = max(0, int(by + bh * 0.45))
    rc_w = int(bw * 0.25)
    rc_h = int(bh * 0.25)
    if rc_w > 5 and rc_h > 5 and rc_y + rc_h <= h_img and rc_x + rc_w <= w_img:
        right_cheek = image_rgb[rc_y : rc_y + rc_h, rc_x : rc_x + rc_w]
        rois.append(right_cheek)

    # Fallback: if facial sub-regions fall out of bounds, use center face crop
    if not rois:
        cx = max(0, int(bx + bw * 0.20))
        cy = max(0, int(by + bh * 0.20))
        cw = int(bw * 0.60)
        ch = int(bh * 0.60)
        rois.append(image_rgb[cy : cy + ch, cx : cx + cw])

    return rois


def extract_frame_rgb_mean(image_rgb, box, keypoints=None):
    """
    Computes the spatial mean [Red, Green, Blue] values across the skin ROIs.
    Turns an entire 224x224 face into 3 precise color values.
    """
    rois = extract_skin_rois(image_rgb, box, keypoints)
    total_pixels = 0
    rgb_sum = np.zeros(3, dtype=np.float64)

    for patch in rois:
        if patch.size > 0:
            rgb_sum += np.sum(patch, axis=(0, 1), dtype=np.float64)
            total_pixels += patch.shape[0] * patch.shape[1]

    if total_pixels == 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    return (rgb_sum / total_pixels).astype(np.float32)


# ==============================================================================
# 2. CHROMINANCE (CHROM) rPPG PULSE EXTRACTION ALGORITHM
# ==============================================================================
def chrom_rppg(rgb_signals):
    """
    Extracts the Blood Volume Pulse (BVP) wave using the CHROM algorithm
    (De Haan & Jeanne, IEEE TBME 2013).
    Removes skin tone biases and lighting variations by projecting color
    signals onto orthogonal chrominance planes (X_s and Y_s).

    Args:
        rgb_signals: Array of shape (T, 3) where columns are [R, G, B] over time.

    Returns:
        1D numpy array of shape (T,) representing the raw pulse wave.
    """
    T = rgb_signals.shape[0]
    if T < 10:
        return np.zeros(T, dtype=np.float32)

    R = rgb_signals[:, 0].astype(np.float64)
    G = rgb_signals[:, 1].astype(np.float64)
    B = rgb_signals[:, 2].astype(np.float64)

    # 1. Normalize each color channel by its temporal mean
    mean_R = np.mean(R) + 1e-6
    mean_G = np.mean(G) + 1e-6
    mean_B = np.mean(B) + 1e-6

    rn = R / mean_R
    gn = G / mean_G
    bn = B / mean_B

    # 2. Chrominance orthogonal projections (cancels motion & skin color)
    Xs = 3.0 * rn - 2.0 * gn
    Ys = 1.5 * rn + gn - 1.5 * bn

    # 3. Adaptive standard deviation weighting
    std_Xs = np.std(Xs)
    std_Ys = np.std(Ys) + 1e-6
    alpha = std_Xs / std_Ys

    # 4. Resulting blood volume pulse wave
    bvp = Xs - alpha * Ys
    return bvp.astype(np.float32)


# ==============================================================================
# 3. BIOLOGICAL BANDPASS FILTERING (0.7 Hz to 3.5 Hz)
# ==============================================================================
def butter_bandpass_filter(
    signal,
    fs=config.VIDEO_FPS,
    low_hz=config.BANDPASS_LOW_HZ,
    high_hz=config.BANDPASS_HIGH_HZ,
    order=config.BUTTERWORTH_ORDER
):
    """
    Applies zero-phase Butterworth bandpass filter.
    Preserves human pulse rhythms (42 - 210 BPM) and removes high-frequency
    camera noise and low-frequency lighting drifts.
    """
    if len(signal) <= 15:
        return signal

    nyquist = 0.5 * fs
    low = max(0.01, low_hz / nyquist)
    high = min(0.99, high_hz / nyquist)

    # Explicitly specify output='ba' to ensure returning (b, a) transfer function polynomials
    b, a = butter(order, [low, high], btype="bandpass", output="ba")  # type: ignore
    filtered = filtfilt(b, a, signal)  # type: ignore
    return np.asarray(filtered, dtype=np.float32)


# ==============================================================================
# 4. HEART RATE (BPM) CALCULATION FROM PULSE WAVE
# ==============================================================================
def estimate_heart_rate_bpm(
    bvp_signal,
    fs=config.VIDEO_FPS,
    low_bpm=42.0,
    high_bpm=210.0
):
    """
    Estimates Heart Rate in Beats Per Minute (BPM) using Fast Fourier Transform
    (FFT) spectral peak detection over the filtered BVP signal.
    """
    T = len(bvp_signal)
    if T < fs:
        return 0.0

    # Subtract mean and apply Hanning window to prevent spectral leakage
    windowed = (bvp_signal - np.mean(bvp_signal)) * np.hanning(T)

    # FFT with 4x zero-padding for fine frequency resolution
    nfft = max(256, int(2 ** np.ceil(np.log2(T * 4))))
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    fft_mag = np.abs(np.fft.rfft(windowed, n=nfft))

    # Restrict frequency search to human heart rate bounds (0.7 Hz to 3.5 Hz)
    low_hz = low_bpm / 60.0
    high_hz = high_bpm / 60.0
    valid_mask = (freqs >= low_hz) & (freqs <= high_hz)

    if not np.any(valid_mask):
        return 72.0  # Safe resting baseline fallback

    valid_freqs = freqs[valid_mask]
    valid_mag = fft_mag[valid_mask]

    peak_idx = np.argmax(valid_mag)
    peak_hz = valid_freqs[peak_idx]

    bpm = float(peak_hz * 60.0)
    return np.clip(bpm, low_bpm, high_bpm)


# ==============================================================================
# 5. DIAGNOSTIC SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Phase 2 Diagnostic — Testing rPPG Extraction Module")
    print("=" * 60)

    # 1. Test synthetic 75 BPM pulse signal (75 / 60 = 1.25 Hz)
    fs = 30
    duration_sec = 10
    num_frames = fs * duration_sec
    time_axis = np.linspace(0, duration_sec, num_frames)

    # Simulate realistic blood pulse in green channel
    true_bpm = 75.0
    pulse_freq = true_bpm / 60.0
    simulated_r = 150.0 + 1.0 * np.sin(2 * np.pi * pulse_freq * time_axis)
    simulated_g = 120.0 + 2.5 * np.sin(2 * np.pi * pulse_freq * time_axis)
    simulated_b = 90.0 + 0.5 * np.sin(2 * np.pi * pulse_freq * time_axis)

    rgb_array = np.stack([simulated_r, simulated_g, simulated_b], axis=1)

    # Extract via CHROM
    raw_bvp = chrom_rppg(rgb_array)
    clean_bvp = butter_bandpass_filter(raw_bvp, fs=fs)
    measured_bpm = estimate_heart_rate_bpm(clean_bvp, fs=fs)

    print(f"  Target Synthetic Heart Rate : {true_bpm:.1f} BPM")
    print(f"  Measured rPPG Heart Rate   : {measured_bpm:.1f} BPM")
    print(f"  Estimation Error            : {abs(measured_bpm - true_bpm):.2f} BPM")

    assert abs(measured_bpm - true_bpm) < 1.0, "rPPG calculation error too high!"
    print("  [SUCCESS] Phase 2 rPPG extraction engine verified perfectly!")
    print("=" * 60)

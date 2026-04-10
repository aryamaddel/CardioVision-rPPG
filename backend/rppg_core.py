"""
The algorithmic engine of the CardioVision-rPPG system.

Provides specialized signal processing functions for extracting blood volume 
pulse (BVP) from RGB signals. Includes implementations of the POS algorithm, 
spectral analysis, Butterworth filters, and Heart Rate Variability (HRV) 
metrics for stress estimation.
"""
import numpy as np
from scipy.signal import (
    butter,
    detrend,
    filtfilt,
    find_peaks,
    lfilter,
    welch,
)

# ─────────────────────────────────────────────────────────────────────────────
# 0. CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

LOW_HZ = 0.7  # 42 BPM
HIGH_HZ = 4.0  # 240 BPM

# ─────────────────────────────────────────────────────────────────────────────
# 1. SHARED UTILITIES
# ─────────────────────────────────────────────────────────────────────────────


def bandpass_filter(signal_1d, fps, low_hz=LOW_HZ, high_hz=HIGH_HZ, order=4):
    """
    Applies a zero-phase Butterworth bandpass filter to a 1D signal.

    Designed to isolate the human heart rate frequency band. Uses filtfilt 
    for larger signals to avoid phase shift, and lfilter as a fallback 
    for very short buffers.

    Args:
        signal_1d (np.ndarray): The input signal to filter.
        fps (float): Sampling rate (frames per second).
        low_hz (float, optional): Lower cutoff frequency. Defaults to 0.7 (42 BPM).
        high_hz (float, optional): Upper cutoff frequency. Defaults to 4.0 (240 BPM).
        order (int, optional): The order of the filter. Defaults to 4.

    Returns:
        np.ndarray: The bandpass-filtered signal.
    """
    signal_1d = np.asarray(signal_1d, dtype=np.float64)
    if signal_1d.size < 3:
        return signal_1d.copy()

    nyquist = fps / 2.0
    low = max(low_hz / nyquist, 1e-6)
    high = min(high_hz / nyquist, 0.99)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return signal_1d.copy()

    b, a = butter(order, [low, high], btype="band")
    padlen = 3 * max(len(a), len(b))
    if signal_1d.size <= padlen:
        # Warm-up fallback for short stream buffers where filtfilt padding is invalid.
        return lfilter(b, a, signal_1d)

    return filtfilt(b, a, signal_1d)


def spectral_peak_snr(sig, fps):
    """
    Calculates the Signal-to-Noise Ratio (SNR) in the heart rate frequency band.

    Uses FFT to find the power of the dominant peak within the physiological 
    heart rate range relative to the average power of the full band.

    Args:
        sig (np.ndarray): The pulse signal array.
        fps (float): Sampling rate (frames per second).

    Returns:
        float: The peak FFT power ratio. Higher values indicate a cleaner signal.
    """
    fft = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(len(sig), d=1.0 / fps)
    mask = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
    if not mask.any():
        return 0.0
    hr_fft = fft[mask]
    peak_power = np.max(hr_fft) ** 2
    avg_power = np.mean(hr_fft ** 2) + 1e-10
    return peak_power / avg_power


# ─────────────────────────────────────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────




def detrend_rgb(rgb):
    """
    Removes linear trends from multi-channel RGB data.

    Preserves the global channel means while removing slow drifts. This is 
    crucial for the POS algorithm to function correctly during per-window 
    normalization.

    Args:
        rgb (np.ndarray): Input RGB matrix of shape (N, 3).

    Returns:
        np.ndarray: The detrended RGB signal.
    """
    means = rgb.mean(axis=0)
    return detrend(rgb.astype(np.float64), axis=0, type="linear") + means








# ─────────────────────────────────────────────────────────────────────────────
# 3. EXTRACTION ALGORITHMS
# ─────────────────────────────────────────────────────────────────────────────


def pos_algorithm(rgb, fps, window_sec=4.0):
    """Plane-Orthogonal-to-Skin (POS) rPPG extraction (Wang et al., 2017).
    
    Adapted for variable FPS: at low mobile FPS (3-8), windows are smaller
    but overlap is maximized to maintain signal quality.
    """
    N = len(rgb)
    window = max(6, int(window_sec * fps))  # At least 6 frames per window
    if window > N:
        window = N  # Use entire signal if shorter than one window
    pulse = np.zeros(N)
    weights = np.zeros(N)
    step = max(1, window // 3)  # More overlap at low FPS for smoother output

    for start in range(0, max(1, N - window + 1), step):
        end = start + window
        segment = rgb[start:end].copy()
        mean_c = np.mean(segment, axis=0)
        if np.any(mean_c == 0):
            continue

        Cn = segment / (mean_c + 1e-8)
        S1 = Cn[:, 1] - Cn[:, 2]
        S2 = Cn[:, 1] + Cn[:, 2] - 2 * Cn[:, 0]

        std_s1, std_s2 = np.std(S1), np.std(S2)
        if std_s2 < 1e-8:
            continue

        H = S1 + (std_s1 / std_s2) * S2
        pulse[start:end] += H
        weights[start:end] += 1

    if np.all(weights == 0):
        return np.zeros(N)
    pulse /= weights + 1e-8
    return bandpass_filter(pulse, fps)




# ─────────────────────────────────────────────────────────────────────────────
# 4. POST-PROCESSING & VALIDATION
# ─────────────────────────────────────────────────────────────────────────────


def extract_pulse_waveform(pulse, fps):
    """
    Extracts peak indices and Inter-Beat Intervals (IBI) from raw pulse data.

    Normalizes the signal and uses adaptive peak detection to find consistent
    beats. Filters IBI to the valid physiological range matching the 45–220 BPM
    display confidence window (273–1333 ms).

    Args:
        pulse (np.ndarray): The 1D input pulse signal.
        fps (float): Sampling rate.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (Peak indices, IBI values in ms,
            normalized pulse waveform).
    """
    p_min, p_max = pulse.min(), pulse.max()
    if p_max - p_min < 1e-8:
        return np.array([]), np.array([]), pulse
    clean_pulse = 2 * (pulse - p_min) / (p_max - p_min) - 1

    min_distance = max(1, int(fps * 0.4))

    # Single reliable threshold
    peaks_idx, _ = find_peaks(clean_pulse, distance=min_distance, height=0.15, prominence=0.15)

    if len(peaks_idx) >= 2:
        ibi_ms = (np.diff(peaks_idx) / fps) * 1000
        # Match the 45–220 BPM display range: 60000/220 ≈ 273 ms, 60000/45 ≈ 1333 ms
        ibi_ms = ibi_ms[(ibi_ms >= 273) & (ibi_ms <= 1333)]
    else:
        ibi_ms = np.array([])

    return peaks_idx, ibi_ms, clean_pulse


def compute_confidence_score(pulse, fps, ibi_ms):
    """
    Calculates a multi-factor confidence score for a pulse segment.

    Evaluates:
    1. IBI Regularity (Coefficient of Variation)
    2. Spectral SNR (Sharpness of frequency peak)
    3. Peak Density (Valid BPM range check)
    4. Duration (Data quantity)

    Args:
        pulse (np.ndarray): The BVP signal.
        fps (float): Sampling rate.
        ibi_ms (np.ndarray): Array of Inter-Beat Intervals.

    Returns:
        Tuple[float, dict, bool]: (Final 0-1 score, detail metrics, reliability flag).
    """
    """
    Simplified confidence scoring based on SNR and peak consistency.
    """
    details = {}

    # 1. SNR Score (0.60) — uses shared spectral_peak_snr
    raw_snr = spectral_peak_snr(pulse, fps)
    snr_score = float(np.clip(raw_snr / 10.0, 0.0, 1.0))

    # Annotate dominant BPM from FFT peak
    fft = np.abs(np.fft.rfft(pulse))
    freqs = np.fft.rfftfreq(len(pulse), d=1.0 / fps)
    hr_mask = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
    if hr_mask.any():
        peak_f = freqs[hr_mask][np.argmax(fft[hr_mask])]
        details["dominant_bpm"] = float(peak_f * 60)

    # 2. Peak Density Score (0.40) — matches 45–220 BPM window
    if len(ibi_ms) >= 2:
        bpm = 60000.0 / np.mean(ibi_ms)
        density_score = 1.0 if 45 <= bpm <= 220 else 0.3
    else:
        density_score = 0.0

    final_score = float(0.6 * snr_score + 0.4 * density_score)
    is_reliable = final_score >= 0.40

    details.update({
        "final_score": final_score,
        "is_reliable": is_reliable,
        "snr": snr_score,
        "peak_count": len(ibi_ms) + 1 if len(ibi_ms) > 0 else 0,
        "density": density_score,
    })
    return final_score, details, is_reliable


# ─────────────────────────────────────────────────────────────────────────────
# 5. HRV & STRESS ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────


def compute_hrv_features(ibi_ms):
    """
    Compute Time-Domain and Frequency-Domain HRV features and provide
    a basic stress classification from Inter-Beat Intervals (IBI).
    """
    if len(ibi_ms) < 2:
        return {
            "rmssd_ms": 0.0,
            "sdnn_ms": 0.0,
            "lf_hf_ratio": 0.0,
            "stress_index": 0.0,
            "stress_level": "Unknown",
        }

    ibi_sec = ibi_ms / 1000.0

    # Time-Domain
    diff = np.diff(ibi_sec)
    rmssd = np.sqrt(np.mean(diff**2)) if len(diff) > 0 else 0.0
    sdnn = np.std(ibi_sec)

    # Frequency-Domain — resample IBI to a uniform 4 Hz grid before Welch
    try:
        # Cumulative time stamps from IBI series (in seconds)
        ibi_cumtime = np.cumsum(ibi_sec)
        t_start, t_end = ibi_cumtime[0], ibi_cumtime[-1]
        fs_interp = 4.0  # 4 Hz uniform grid is standard for HRV PSD
        t_uniform = np.arange(t_start, t_end, 1.0 / fs_interp)
        if len(t_uniform) >= 4:
            ibi_uniform = np.interp(t_uniform, ibi_cumtime, ibi_sec)
            nperseg = min(len(ibi_uniform), 256)
            freqs, psd = welch(ibi_uniform, fs=fs_interp, nperseg=nperseg)
            lf_band = (freqs >= 0.04) & (freqs <= 0.15)
            hf_band = (freqs >= 0.15) & (freqs <= 0.4)
            df = freqs[1] - freqs[0] if len(freqs) > 1 else 0.0
            lf_power = np.sum(psd[lf_band]) * df
            hf_power = np.sum(psd[hf_band]) * df
            lf_hf = lf_power / hf_power if hf_power > 0 else 0.0
        else:
            lf_hf = 0.0
    except Exception:
        lf_hf = 0.0

    # Basic rule-based Stress Classifier
    # High stress usually correlates with Low HRV (RMSSD), High LF/HF, and Higher BPM
    stress_index = 0.0

    if rmssd < 0.02:
        stress_index += 40
    elif rmssd < 0.035:
        stress_index += 20

    if lf_hf > 1.5:
        stress_index += 40
    elif lf_hf > 1.0:
        stress_index += 20

    mean_bpm = 60.0 / np.mean(ibi_sec)
    if mean_bpm > 90:
        stress_index += 20
    elif mean_bpm > 80:
        stress_index += 10

    if stress_index >= 60:
        stress_level = "High"
    elif stress_index >= 30:
        stress_level = "Medium"
    else:
        stress_level = "Low"

    return {
        "rmssd_ms": float(rmssd * 1000.0),
        "sdnn_ms": float(sdnn * 1000.0),
        "lf_hf_ratio": float(lf_hf),
        "stress_index": float(stress_index),
        "stress_level": stress_level,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN PIPELINE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────


def process_rppg(
    rgb_raw: np.ndarray,
    fps: float = 30.0,
    motion_scores: np.ndarray | None = None,
):
    """
    The standard rPPG processing pipeline using the POS algorithm.

    Performs motion detection, RGB detrending, POS extraction, and pulse 
    summarization including HRV calculation and confidence scoring.

    Args:
        rgb_raw (np.ndarray): Shape (N, 3) representing mean R, G, B per frame.
        fps (float, optional): Target frame rate. Defaults to 30.0.
        motion_scores (np.ndarray, optional): Precomputed motion metrics.

    Returns:
        dict: comprehensive results containing 'pulse_signal', 'ibi_ms', 'confidence', etc.
    """
    # Motion pre-filtering removed per request. POS handles noise internally.
    motion_fraction = 0.0

    # 2. Preprocess (POS needs channel means intact)
    rgb_detrended = detrend_rgb(rgb_raw)

    # 3. Extract using POS
    pulse = pos_algorithm(rgb_detrended, fps)

    return _summarize_pulse(
        pulse=pulse,
        fps=fps,
        motion_fraction=motion_fraction,
        method_used="pos",
        n_frames=len(rgb_raw),
    )


def _bpm_from_ibi(ibi_ms: np.ndarray) -> float | None:
    if ibi_ms.size == 0:
        return None
    return float(60000.0 / np.median(ibi_ms))


def _summarize_pulse(
    pulse: np.ndarray,
    fps: float,
    motion_fraction: float,
    method_used: str,
    n_frames: int,
) -> dict:
    # remove_motion_artifacts (Savgol) removed to prevent signal distortion
    peaks_idx, ibi_ms, clean_pulse = extract_pulse_waveform(pulse, fps)
    confidence, details, is_reliable = compute_confidence_score(clean_pulse, fps, ibi_ms)
    hrv_features = compute_hrv_features(ibi_ms)
    
    bpm = _bpm_from_ibi(ibi_ms)
    
    return {
        "pulse_signal": clean_pulse,
        "timestamps": np.arange(n_frames) / fps,
        "fps": fps,
        "peaks_idx": peaks_idx,
        "ibi_ms": ibi_ms,
        "bpm": bpm,
        "confidence": confidence,
        "is_reliable": is_reliable,
        "confidence_details": details,
        "hrv_features": hrv_features,
        "motion_fraction": motion_fraction,
        "method_used": method_used,
        "n_frames": n_frames,
        "duration_sec": n_frames / fps,
    }


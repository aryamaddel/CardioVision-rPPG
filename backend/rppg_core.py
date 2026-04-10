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
    medfilt,
    savgol_filter,
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
    result = np.zeros_like(rgb, dtype=np.float64)
    for i in range(3):
        channel = rgb[:, i].astype(np.float64)
        ch_mean = np.mean(channel)
        result[:, i] = detrend(channel, type="linear") + ch_mean
    return result


def detect_motion_frames(rgb_raw, threshold=2.5):
    """
    Identifies frames with excessive pixel intensity jumps.

    Computes the mean absolute difference between consecutive frames and flags 
    those that deviate significantly from the baseline.

    Args:
        rgb_raw (np.ndarray): raw RGB data.
        threshold (float, optional): standard deviation factor for rejection.

    Returns:
        np.ndarray: A boolean mask where True indicates a high-motion frame.
    """
    diff = np.abs(np.diff(rgb_raw, axis=0))
    frame_motion = diff.mean(axis=1)
    motion_threshold = np.mean(frame_motion) + threshold * np.std(frame_motion)
    bad_frames = np.concatenate([[False], frame_motion > motion_threshold])
    return bad_frames


def remove_motion_artifacts(pulse, fps, method="savgol"):
    """
    Smoothes the pulse signal to mitigate sharp motion-induced spikes.

    Supports Median filtering, Savitzky-Golay filtering, or a combination of 
    both to preserve peak shapes while removing high-frequency noise.

    Args:
        pulse (np.ndarray): The 1D BVP signal.
        fps (float): Sampling rate.
        method (str, optional): 'median', 'savgol', or 'both'. Defaults to 'savgol'.

    Returns:
        np.ndarray: The cleaned pulse signal.
    """
    cleaned = pulse.copy()
    if method in ("median", "both"):
        kernel = max(3, int(fps * 0.067) | 1)
        cleaned = medfilt(cleaned, kernel_size=kernel)
    if method in ("savgol", "both"):
        window = max(5, int(fps * 0.133) | 1)
        poly = min(3, window - 1)
        if window > poly:
            cleaned = savgol_filter(cleaned, window_length=window, polyorder=poly)
    return cleaned


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

    pulse /= weights + 1e-8
    return bandpass_filter(pulse, fps)




# ─────────────────────────────────────────────────────────────────────────────
# 4. POST-PROCESSING & VALIDATION
# ─────────────────────────────────────────────────────────────────────────────


def extract_pulse_waveform(pulse, fps):
    """
    Extracts peak indices and Inter-Beat Intervals (IBI) from raw pulse data.

    Normalizes the signal and uses adaptive peak detection with multiple 
    threshold levels to find consistent beats. Filters IBI to valid 
    physiological ranges (300ms to 1500ms).

    Args:
        pulse (np.ndarray): The 1D input pulse signal.
        fps (float): Sampling rate.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (Peak indices, IBI values in ms, 
            normalized pulse waveform).
    """
    """Extract clinical-grade features: peaks, IBI, and normalized waveform."""
    p_min, p_max = pulse.min(), pulse.max()
    if p_max - p_min < 1e-8:
        return np.array([]), np.array([]), pulse
    clean_pulse = 2 * (pulse - p_min) / (p_max - p_min) - 1

    min_distance = max(1, int(fps * 0.4))
    duration_sec = len(clean_pulse) / fps
    min_expected_peaks = max(2, int(duration_sec * 0.5))

    threshold_levels = [
        {"height": 0.20, "prominence": 0.20},
        {"height": 0.10, "prominence": 0.12},
        {"height": 0.05, "prominence": 0.06},
    ]

    peaks_idx, ibi_ms = np.array([]), np.array([])
    for params in threshold_levels:
        peaks_idx, _ = find_peaks(clean_pulse, distance=min_distance, **params)
        if len(peaks_idx) >= 2:
            ibi_ms = (np.diff(peaks_idx) / fps) * 1000
            ibi_ms = ibi_ms[(ibi_ms >= 300) & (ibi_ms <= 1500)]
        if len(peaks_idx) >= min_expected_peaks and len(ibi_ms) >= 2:
            break

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
    Computes a 0–1 confidence score based on IBI regularity, SNR,
    peak density, and data duration. Returns (score, details, is_reliable).
    """
    details = {}
    duration = len(pulse) / fps

    # 1. IBI Regularity (0.30)
    if len(ibi_ms) >= 3:
        cv = np.std(ibi_ms) / (np.mean(ibi_ms) + 1e-8)
        reg_score = float(np.clip(1.0 - (cv / 0.40), 0.0, 1.0))
    elif len(ibi_ms) >= 1:
        reg_score = (
            0.4
            if len(ibi_ms) == 1
            else float(
                np.clip(1.0 - (np.std(ibi_ms) / np.mean(ibi_ms) / 0.40), 0.0, 1.0)
            )
            * 0.7
        )
    else:
        reg_score = 0.0

    # 2. SNR (0.35)
    fft = np.abs(np.fft.rfft(pulse))
    freqs = np.fft.rfftfreq(len(pulse), d=1.0 / fps)
    hr_mask = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
    snr_score = 0.0
    if hr_mask.any() and fft[hr_mask].sum() > 0:
        hr_fft, hr_freqs = fft[hr_mask], freqs[hr_mask]
        p_idx = np.argmax(hr_fft)
        peak_pwr = hr_fft[p_idx] ** 2
        peak_f = hr_freqs[p_idx]
        noise_mask = (np.abs(hr_freqs - peak_f) > 0.15) & (
            np.abs(hr_freqs - 2 * peak_f) > 0.15
        )
        if noise_mask.any():
            snr_score = float(
                np.clip(
                    (peak_pwr / (np.mean(hr_fft[noise_mask] ** 2) + 1e-10)) / 12.0,
                    0.0,
                    1.0,
                )
            )
        details["dominant_bpm"] = float(peak_f * 60)

    # 3. Peak Density (0.15)
    if len(ibi_ms) >= 1:
        bpm = 60000.0 / np.mean(ibi_ms)
        density_score = 1.0 if 45 <= bpm <= 160 else (0.6 if 35 <= bpm <= 200 else 0.2)
    else:
        density_score = 0.1 if 40 <= details.get("dominant_bpm", 0) <= 180 else 0.0

    # 4. Data Duration (0.20)
    data_score = float(np.clip(duration / 15.0, 0.0, 1.0))

    scores = [reg_score, snr_score, density_score, data_score]
    final_score = float(np.dot(scores, [0.30, 0.35, 0.15, 0.20]))
    is_reliable = final_score >= 0.45

    details.update(
        {
            "final_score": final_score,
            "is_reliable": is_reliable,
            "ibi_regularity": reg_score,
            "snr": snr_score,
            "density": density_score,
            "duration": data_score,
        }
    )
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

    # Frequency-Domain
    try:
        from scipy.signal import welch

        fs_interp = 1.0 / np.mean(ibi_sec)
        nperseg = min(len(ibi_sec), 256)
        if nperseg > 0:
            freqs, psd = welch(ibi_sec, fs=fs_interp, nperseg=nperseg)
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

    # Dynamic 0-100 Stress Index (Continuous Mapping)
    # Higher score = More Stress.
    # 1. HRV Component (RMSSD): 0.02 (high stress) to 0.07 (very calm)
    hrv_stress = np.interp(rmssd, [0.02, 0.07], [50, 0])
    
    # 2. Autonomic Component (LF/HF): 0.5 (calm) to 2.5 (stress)
    lfhf_stress = np.interp(lf_hf, [0.5, 2.5], [0, 30])
    
    # 3. Excitement Component (BPM): 60 (resting) to 110 (high)
    mean_bpm = 60.0 / max(0.001, float(np.mean(ibi_sec)))
    bpm_stress = np.interp(mean_bpm, [65, 110], [0, 20])

    stress_index = float(hrv_stress + lfhf_stress + bpm_stress)
    stress_index = max(0, min(100, stress_index))

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
    # 1. Motion
    if motion_scores is not None:
        motion_fraction = float((motion_scores > 0.05).mean())
    else:
        bad_frames = detect_motion_frames(rgb_raw)
        motion_fraction = float(bad_frames.mean())

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


def _evaluate_pulse_candidate(
    pulse: np.ndarray,
    fps: float,
    motion_fraction: float,
    method_used: str,
    n_frames: int,
) -> dict:
    """Score one pulse candidate using the same post-processing path as POS."""
    return _summarize_pulse(
        pulse=pulse,
        fps=fps,
        motion_fraction=motion_fraction,
        method_used=method_used,
        n_frames=n_frames,
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
    pulse = remove_motion_artifacts(pulse, fps, method="savgol")
    peaks_idx, ibi_ms, clean_pulse = extract_pulse_waveform(pulse, fps)
    confidence, details, is_reliable = compute_confidence_score(clean_pulse, fps, ibi_ms)
    hrv_features = compute_hrv_features(ibi_ms)
    return {
        "pulse_signal": clean_pulse,
        "timestamps": np.arange(n_frames) / fps,
        "fps": fps,
        "peaks_idx": peaks_idx,
        "ibi_ms": ibi_ms,
        "confidence": confidence,
        "is_reliable": is_reliable,
        "confidence_details": details,
        "hrv_features": hrv_features,
        "motion_fraction": motion_fraction,
        "method_used": method_used,
        "n_frames": n_frames,
        "duration_sec": n_frames / fps,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. DEEP MODEL FUSION (POS + Neural Network)
# ─────────────────────────────────────────────────────────────────────────────


def fuse_pos_deep(
    pulse_pos: np.ndarray,
    pulse_deep: np.ndarray,
    fps: float,
    deep_available: bool,
) -> tuple[np.ndarray, str]:
    """
    Fuse POS and deep model signals using frequency-domain SNR as the selector.

    Strategy:
    - Compute spectral_peak_snr for both signals
    - If deep model SNR > POS SNR by > 20%: use deep model
    - Otherwise: weighted average (POS * 0.4 + deep * 0.6) if both reliable
    - Fallback: POS only

    Returns:
        (fused_signal, method_label)
    """
    if not deep_available or np.all(pulse_deep == 0):
        return pulse_pos, "pos_only"

    snr_pos  = spectral_peak_snr(pulse_pos,  fps)
    snr_deep = spectral_peak_snr(pulse_deep, fps)

    print(f"[fusion] SNR — POS: {snr_pos:.2f} | Deep: {snr_deep:.2f}")

    if snr_deep > snr_pos * 1.20:
        # Deep model clearly wins
        return pulse_deep, "deep_model"
    elif snr_pos > snr_deep * 1.20:
        # POS clearly wins
        return pulse_pos, "pos_only"
    else:
        # Similar quality — weighted ensemble (trust deep slightly more)
        fused = 0.40 * pulse_pos + 0.60 * pulse_deep
        return fused, "pos+deep_ensemble"


def process_rppg_with_deep(
    rgb_raw: np.ndarray,
    fps: float = 30.0,
    face_frames: np.ndarray | None = None,
    motion_scores: np.ndarray | None = None,
    selection_mode: str = "best_confidence",
    deep_max_frames: int | None = None,
) -> dict:
    """
    Advanced rPPG pipeline fusing statistical POS and deep learning models.

    Runs both POS and the deep neural model (if face frames are available and 
    capable), then selects or fuses the signals based on frequency-domain SNR 
    and guardrail consistency checks.

    Args:
        rgb_raw: (N, 3) matrix of mean RGB per frame.
        fps: The sampling rate.
        face_frames: (N, H, W, 3) BGR face crops for the deep model.
        motion_scores: Optional pre-calculated motion data.
        selection_mode: 'best_confidence' to pick best sig, or 'fuse' for ensemble.
        deep_max_frames: Max frames to send to the deep model to prevent memory issues.

    Returns:
        dict: Full result set including the 'selected_source' and 'is_reliable' status.
    """
    from deep_rppg import extract_bvp_deep, is_deep_model_available

    # POS runs first so the app can display an estimate quickly.
    pos_result = process_rppg(rgb_raw, fps=fps, motion_scores=motion_scores)
    pulse_pos = pos_result["pulse_signal"]

    deep_available = False
    pulse_deep = np.zeros_like(pulse_pos)
    deep_model_name = "none"

    if face_frames is not None and is_deep_model_available():
        try:
            pulse_deep, deep_model_name = extract_bvp_deep(
                face_frames,
                fps,
                max_frames=deep_max_frames,
            )
            pulse_deep = bandpass_filter(pulse_deep, fps)
            deep_available = not np.all(pulse_deep == 0)
        except Exception as e:
            print(f"[rppg_core] Deep model error: {e}")

    deep_result = None
    if deep_available:
        deep_result = _evaluate_pulse_candidate(
            pulse=pulse_deep,
            fps=fps,
            motion_fraction=float(pos_result["motion_fraction"]),
            method_used="deep_model",
            n_frames=len(rgb_raw),
        )

    result = pos_result.copy()
    selected = "pos"
    if deep_result is not None:
        if selection_mode == "fuse":
            fused_pulse, fusion_method = fuse_pos_deep(pulse_pos, pulse_deep, fps, True)
            fused_result = _evaluate_pulse_candidate(
                pulse=fused_pulse,
                fps=fps,
                motion_fraction=float(pos_result["motion_fraction"]),
                method_used=fusion_method,
                n_frames=len(rgb_raw),
            )
            result = fused_result
            selected = fusion_method
        else:
            if float(deep_result["confidence"]) > float(pos_result["confidence"]):
                result = deep_result
                selected = "deep_model"

    result.update(
        {
            "deep_model_used": deep_model_name,
            "selected_source": selected,
            "selection_mode": selection_mode,
            "pos_confidence": float(pos_result["confidence"]),
            "deep_confidence": float(deep_result["confidence"]) if deep_result else 0.0,
            "pos_snr": spectral_peak_snr(pulse_pos, fps),
            "deep_snr": spectral_peak_snr(pulse_deep, fps) if deep_available else 0.0,
        }
    )

    pos_bpm = _bpm_from_ibi(pos_result["ibi_ms"])
    deep_bpm = _bpm_from_ibi(deep_result["ibi_ms"]) if deep_result else None
    agreement_ok = True
    if pos_bpm is not None and deep_bpm is not None:
        agreement_ok = abs(pos_bpm - deep_bpm) <= 18.0

    confidence_ok = float(result["confidence"]) >= 0.50
    reliability_ok = bool(result["is_reliable"])
    result["is_reliable"] = bool(reliability_ok and confidence_ok and agreement_ok)
    result["guardrails"] = {
        "confidence_ok": confidence_ok,
        "agreement_ok": agreement_ok,
        "pos_bpm": pos_bpm,
        "deep_bpm": deep_bpm,
    }

    return result

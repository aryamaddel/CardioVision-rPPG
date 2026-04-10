"""
Micro-Tremor Analysis Module for CardioVision.

Performs frequency-domain analysis on raw 3-axis accelerometer samples
collected during a 30-second rPPG scan. The physiological tremor band
(8-12 Hz) is used as the primary stress/fatigue indicator.

Reference:
    Physiological tremor increases with mental fatigue and stress.
    Elble & Koller (1990). Tremor. Johns Hopkins University Press.
"""

from typing import Dict, List, Optional
import numpy as np


SAMPLE_RATE_HZ = 50.0          # Accelerometer was set to 20ms interval
PHYSIO_BAND_LOW = 8.0          # Hz – lower bound of physiological tremor
PHYSIO_BAND_HIGH = 12.0        # Hz – upper bound of physiological tremor
VOLUNTARY_BAND_HIGH = 2.0      # Hz – voluntary/coarse motion ceiling


def analyze_tremor(
    samples: List[Dict[str, float]],
    sample_rate: float = SAMPLE_RATE_HZ,
) -> Dict[str, object]:
    """
    Analyse raw accelerometer samples and return a tremor report.

    Args:
        samples: List of dicts with keys 'x', 'y', 'z' (units: g or m/s²).
        sample_rate: Recording rate in Hz. Defaults to 50 Hz.

    Returns:
        Dict with:
            tremor_score (int):   0–100, 0 = perfectly steady.
            tremor_label (str):   'Steady' | 'Mild' | 'High'.
            physio_power (float): Normalized power in 8–12 Hz band.
            rms_jitter (float):   Raw RMS jitter in signal units.
            n_samples (int):      Total samples received.
            is_reliable (bool):   True if enough samples for valid analysis.
    """
    n = len(samples)

    # Need at least 2 seconds of data at the given rate.
    min_samples = int(sample_rate * 2)
    if n < min_samples:
        return {
            "tremor_score": 0,
            "tremor_label": "Unknown",
            "physio_power": 0.0,
            "rms_jitter": 0.0,
            "n_samples": n,
            "is_reliable": False,
        }

    # Build magnitude signal from all three axes.
    magnitudes = np.array([
        np.sqrt(s["x"] ** 2 + s["y"] ** 2 + s["z"] ** 2)
        for s in samples
    ], dtype=np.float64)

    # Detrend: remove gravity and slow postural sway (high-pass at ~0.5 Hz).
    from numpy.fft import rfft, rfftfreq
    freqs = rfftfreq(len(magnitudes), d=1.0 / sample_rate)
    spectrum = rfft(magnitudes - magnitudes.mean())
    power = np.abs(spectrum) ** 2

    # RMS jitter (time domain — simple measure).
    rms_jitter = float(np.std(magnitudes))

    # Frequency-domain: power in physiological tremor band vs total.
    physio_mask = (freqs >= PHYSIO_BAND_LOW) & (freqs <= PHYSIO_BAND_HIGH)
    total_power = float(power.sum()) or 1.0
    physio_power = float(power[physio_mask].sum()) / total_power

    # Score: 60% weight on frequency-domain, 40% on time-domain RMS.
    # physio_power typically 0–0.4 range; rms_jitter 0–0.5g typical.
    freq_score = min(1.0, physio_power / 0.4)
    time_score = min(1.0, rms_jitter / 0.35)
    raw_score = 0.6 * freq_score + 0.4 * time_score
    tremor_score = int(round(raw_score * 100))
    tremor_score = max(0, min(100, tremor_score))

    if tremor_score <= 20:
        label = "Steady"
    elif tremor_score <= 50:
        label = "Mild"
    else:
        label = "High"

    return {
        "tremor_score": tremor_score,
        "tremor_label": label,
        "physio_power": round(physio_power, 4),
        "rms_jitter": round(rms_jitter, 4),
        "n_samples": n,
        "is_reliable": True,
    }

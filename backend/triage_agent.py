"""
triage_agent.py — TWIST 1: Triage Decision Agent

When rPPG confidence drops below threshold, automatically switches to
Visual Assessment Mode: analyzes facial appearance for physical distress signs
(pallor, flushing, asymmetry) and outputs a Visual Stress Score.

Dashboard displays:
  - Current mode: "BIOMETRIC" or "VISUAL ASSESSMENT"
  - Reason for mode switch
  - Active signal confidence
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional

# Thresholds
CONFIDENCE_THRESHOLD = 0.45   # Below this → switch to Visual Mode
MOTION_THRESHOLD     = 0.35   # Too much motion → trigger switch
HYSTERESIS           = 0.05   # Must exceed threshold by this to switch back


@dataclass
class TriageDecision:
    mode: str                    # "BIOMETRIC" | "VISUAL_ASSESSMENT"
    reason: str                  # Human-readable reason for mode
    confidence: float            # rPPG confidence (0-1)
    visual_stress_score: float   # Visual assessment score (0-100), 0 if not active
    visual_stress_label: str     # "Low" | "Medium" | "High" | "N/A"
    active_bpm: Optional[float]  # BPM if biometric mode
    active_hrv: Optional[dict]   # HRV features if biometric mode


class TriageAgent:
    """
    Monitors rPPG confidence and decides which mode to display.
    Uses hysteresis to avoid rapid mode toggling.
    """

    def __init__(self):
        self._current_mode = "BIOMETRIC"

    def decide(
        self,
        rppg_result: dict,
        face_frames: np.ndarray = None,
    ) -> TriageDecision:
        """
        Main entry point. Called each processing cycle.

        Args:
            rppg_result:  output dict from process_rppg() or process_rppg_with_deep()
            face_frames:  recent face crop frames for visual assessment (N, H, W, 3)

        Returns:
            TriageDecision with current mode and all outputs
        """
        confidence     = rppg_result.get("confidence", 0.0)
        is_reliable    = rppg_result.get("is_reliable", False)
        motion_frac    = rppg_result.get("motion_fraction", 0.0)
        ibi_ms         = rppg_result.get("ibi_ms", np.array([]))
        hrv_features   = rppg_result.get("hrv_features", {})

        # ── Mode switching logic (with hysteresis) ────────────────────────
        if self._current_mode == "BIOMETRIC":
            switch_to_visual = (
                confidence < CONFIDENCE_THRESHOLD or
                motion_frac > MOTION_THRESHOLD or
                not is_reliable
            )
            if switch_to_visual:
                self._current_mode = "VISUAL_ASSESSMENT"
                reason = self._build_switch_reason(confidence, motion_frac, is_reliable)
            else:
                reason = f"Signal reliable (confidence={confidence:.2f})"
        else:
            # Require confidence to exceed threshold + hysteresis to switch back
            switch_to_biometric = (
                confidence > CONFIDENCE_THRESHOLD + HYSTERESIS and
                motion_frac < MOTION_THRESHOLD and
                is_reliable
            )
            if switch_to_biometric:
                self._current_mode = "BIOMETRIC"
                reason = f"Signal recovered (confidence={confidence:.2f})"
            else:
                reason = self._build_switch_reason(confidence, motion_frac, is_reliable)

        # ── Visual Assessment (runs in Visual mode, or as secondary in Biometric) ─
        visual_score = 0.0
        visual_label = "N/A"
        if self._current_mode == "VISUAL_ASSESSMENT" and face_frames is not None:
            visual_score = compute_visual_stress_score(face_frames)
            visual_label = _score_to_label(visual_score)

        # ── BPM for display ───────────────────────────────────────────────
        active_bpm = None
        if self._current_mode == "BIOMETRIC" and len(ibi_ms) > 0:
            active_bpm = float(60000.0 / np.median(ibi_ms))

        return TriageDecision(
            mode=self._current_mode,
            reason=reason,
            confidence=confidence,
            visual_stress_score=visual_score,
            visual_stress_label=visual_label,
            active_bpm=active_bpm,
            active_hrv=hrv_features if self._current_mode == "BIOMETRIC" else None,
        )

    def _build_switch_reason(self, conf, motion, reliable):
        reasons = [
            f"Low rPPG confidence ({conf:.2f} < {CONFIDENCE_THRESHOLD})"
            if conf < CONFIDENCE_THRESHOLD else None,
            f"Excessive motion ({motion:.0%})"
            if motion > MOTION_THRESHOLD else None,
            "Signal marked unreliable" if not reliable else None,
        ]
        return " | ".join(r for r in reasons if r) or "Unknown"


# ─────────────────────────────────────────────────────────────────────────────
# VISUAL ASSESSMENT ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def compute_visual_stress_score(face_frames: np.ndarray) -> float:
    """
    Lightweight visual distress analysis from face crop frames.
    No neural network required — uses physiologically-motivated color analysis.

    Signals analyzed:
    1. Pallor index — pale skin indicates stress/shock (low red saturation)
    2. Flushing index — flushed skin (high red) indicates elevated stress response
    3. Color variability — erratic skin color changes suggest distress
    4. Brightness variance — irregular lighting may indicate movement/instability

    Returns: stress score 0-100
    """
    if face_frames is None or len(face_frames) == 0:
        return 0.0

    # Use every 5th frame for speed
    sample_frames = face_frames[::5] if len(face_frames) > 10 else face_frames

    pallor_vals, flush_vals, sat_vals = [], [], []

    for frame in sample_frames:
        if frame is None or frame.size == 0:
            continue

        # Convert to multiple color spaces
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        lab  = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Mask out black pixels (background from ROI masking)
        valid_mask = (frame[:, :, 0] > 10) | (frame[:, :, 1] > 10) | (frame[:, :, 2] > 10)
        if valid_mask.sum() < 50:
            continue

        # Pallor: low saturation in skin pixels = pale/pallid
        sat   = hsv[:, :, 1][valid_mask]
        mean_sat = float(np.mean(sat))
        sat_vals.append(mean_sat)

        # Flushing: high redness in LAB (a* channel)
        a_star = lab[:, :, 1][valid_mask]  # positive = reddish
        mean_flush = float(np.mean(a_star))
        flush_vals.append(mean_flush)

        # Brightness
        val = hsv[:, :, 2][valid_mask]
        pallor_vals.append(float(np.mean(val)))

    if not sat_vals:
        return 0.0

    # ── Score 1: Pallor (low saturation = pale) ───────────────────────────
    mean_saturation = np.mean(sat_vals)
    # Normal skin saturation ~80-150 in HSV (0-255 scale)
    # Very pale: < 50 → high stress
    pallor_score = float(np.clip((80 - mean_saturation) / 80 * 100, 0, 100))

    # ── Score 2: Flushing (high a* = red) ─────────────────────────────────
    mean_redness = np.mean(flush_vals)
    # LAB a* neutral ~128, flushed > 145
    flush_score = float(np.clip((mean_redness - 128) / 20 * 100, 0, 100))

    # ── Score 3: Color variability over time (erratic changes) ────────────
    if len(sat_vals) > 2:
        variability = float(np.std(sat_vals) / (np.mean(sat_vals) + 1e-5) * 100)
        variability_score = float(np.clip(variability * 3, 0, 100))
    else:
        variability_score = 0.0

    # ── Weighted composite ─────────────────────────────────────────────────
    visual_stress = (
        0.40 * pallor_score +
        0.35 * flush_score  +
        0.25 * variability_score
    )

    return float(np.clip(visual_stress, 0.0, 100.0))


def _score_to_label(score: float) -> str:
    if score >= 60:
        return "High"
    elif score >= 30:
        return "Medium"
    else:
        return "Low"


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD FORMATTER (for UI / terminal output)
# ─────────────────────────────────────────────────────────────────────────────

def format_dashboard(decision: TriageDecision) -> dict:
    """
    Format TriageDecision into a clean dict for the UI (Member 5).
    Keys match what the UI expects to display.
    """
    mode_color = "#00FF88" if decision.mode == "BIOMETRIC" else "#FF6600"
    badge = "🫀 BIOMETRIC MODE" if decision.mode == "BIOMETRIC" else "👁️ VISUAL ASSESSMENT MODE"

    output = {
        "mode":              decision.mode,
        "mode_badge":        badge,
        "mode_color":        mode_color,
        "reason":            decision.reason,
        "confidence":        round(decision.confidence, 3),
        "confidence_pct":    f"{decision.confidence * 100:.1f}%",
    }

    if decision.mode == "BIOMETRIC" and decision.active_bpm:
        output.update({
            "bpm":           round(decision.active_bpm, 1),
            "stress_level":  decision.active_hrv.get("stress_level", "Unknown"),
            "stress_index":  decision.active_hrv.get("stress_index", 0),
            "rmssd":         round(decision.active_hrv.get("rmssd_ms", 0), 1),
            "sdnn":          round(decision.active_hrv.get("sdnn_ms", 0), 1),
            "lf_hf":         round(decision.active_hrv.get("lf_hf_ratio", 0), 2),
        })
    else:
        output.update({
            "visual_stress_score": round(decision.visual_stress_score, 1),
            "visual_stress_label": decision.visual_stress_label,
            "bpm":                 "N/A (signal degraded)",
            "stress_level":        decision.visual_stress_label,
        })

    return output

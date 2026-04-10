"""
Eulerian Video Magnification (EVM) — Color variant, correct implementation.

Based on Wu et al., "Eulerian Video Magnification for Revealing Subtle Changes
in the World", ACM SIGGRAPH 2012.

Key algorithmic choices that produce the natural-looking MIT demo result
(and why naive BGR approaches look garish):

1.  YIQ color space (not BGR/LAB).
    Y = luminance (brightness), I and Q = chrominance (color information).
    Amplifying in this space lets us independently control how much brightness
    vs. color is boosted.  Amplifying directly in BGR couples all three channels
    and produces psychedelic artifacts.

2.  Gaussian pyramid → upsample coarsest level back to original resolution.
    The coarsest level is spatially blurred → no edge information → no edge
    artifacts when amplified.  Critically we UPSAMPLE it back to full
    resolution BEFORE filtering and adding, so it spatially matches the frame.

3.  Attenuation factor A on the chrominance channels (I, Q).
    Y is amplified by α; I and Q are amplified by α × A.
    With A < 1, colour saturation changes are damped relative to the
    brightness pulsation, keeping the output looking natural.

4.  Online IIR Butterworth bandpass filter (frame-by-frame) in place of the
    paper's batch ideal FFT filter, to support real-time processing.

Parameters used in the paper for color (heartbeat) magnification:
    l=4, α=50, A=1, ω_l=0.833 Hz, ω_h=1 Hz
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.signal import butter, sosfilt


# ── YIQ ↔ RGB conversion matrices ────────────────────────────────────────────

_RGB_TO_YIQ = np.array([
    [0.299,      0.587,      0.114     ],
    [0.59590059, -0.27455667, -0.32134392],
    [0.21153661, -0.52273617,  0.31119955],
], dtype=np.float32)

_YIQ_TO_RGB = np.linalg.inv(_RGB_TO_YIQ).astype(np.float32)


def _rgb_to_yiq(img_rgb: np.ndarray) -> np.ndarray:
    """Convert float32 RGB [0,255] image to YIQ."""
    h, w, _ = img_rgb.shape
    flat = img_rgb.reshape(-1, 3)
    yiq = flat @ _RGB_TO_YIQ.T
    return yiq.reshape(h, w, 3)


def _yiq_to_rgb(img_yiq: np.ndarray) -> np.ndarray:
    """Convert float32 YIQ image back to RGB [0,255]."""
    h, w, _ = img_yiq.shape
    flat = img_yiq.reshape(-1, 3)
    rgb = flat @ _YIQ_TO_RGB.T
    return rgb.reshape(h, w, 3)


# ── Gaussian pyramid helpers ──────────────────────────────────────────────────

def _gaussian_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    """Build a Gaussian pyramid; index 0 = original, index -1 = coarsest."""
    pyr = [img.astype(np.float32)]
    cur = img.astype(np.float32)
    for _ in range(levels - 1):
        cur = cv2.pyrDown(cur)
        pyr.append(cur)
    return pyr


def _upsample_to(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Iteratively upsample using pyrUp until we reach (target_h, target_w).
    pyrUp doubles size, so we keep going until we match or exceed target,
    then do a final bilinear resize to exactly match.
    """
    cur = img
    while cur.shape[0] < target_h or cur.shape[1] < target_w:
        cur = cv2.pyrUp(cur)
    if cur.shape[0] != target_h or cur.shape[1] != target_w:
        cur = cv2.resize(cur, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return cur


# ── IIR Butterworth bandpass ──────────────────────────────────────────────────

def _make_sos(lo_hz: float, hi_hz: float, fs: float, order: int = 1) -> np.ndarray:
    nyq = 0.5 * fs
    lo = float(np.clip(lo_hz / nyq, 1e-4, 1.0 - 1e-4))
    hi = float(np.clip(hi_hz / nyq, 1e-4, 1.0 - 1e-4))
    if lo >= hi:
        hi = min(lo + 0.01, 1.0 - 1e-4)
    return butter(order, [lo, hi], btype="band", output="sos")


# ── Main class ────────────────────────────────────────────────────────────────

class EulerianMagnifier:
    """
    Real-time Eulerian Color Magnification.

    Follows the color magnification pipeline from Wu et al. 2012 with an
    online IIR Butterworth filter in place of the batch FFT ideal filter.

    Parameters
    ----------
    fps           : Camera frame rate.
    lo_hz         : Bandpass lower cutoff in Hz (heartbeat lower bound).
    hi_hz         : Bandpass upper cutoff in Hz (heartbeat upper bound).
    alpha         : Amplification factor (α in the paper). 30–50 is typical.
    attenuation   : Chrominance attenuation A. Applied to I and Q channels
                    (alpha × A). A=1 → full colour amp; A<1 → subtler colours.
    levels        : Gaussian pyramid depth. More levels = more spatial blur.
    """

    def __init__(
        self,
        fps: float = 30.0,
        lo_hz: float = 0.8,
        hi_hz: float = 2.0,
        alpha: float = 50.0,
        attenuation: float = 1.0,
        levels: int = 4,
    ):
        self.fps = fps
        self.alpha = alpha
        self.attenuation = attenuation
        self.levels = levels

        self._sos = _make_sos(lo_hz, hi_hz, fps, order=1)

        # IIR state — shape: (n_sections, 2, H, W, 3), lazy init
        self._zi: Optional[np.ndarray] = None
        self._frame_shape: Optional[Tuple] = None
        self._warmup_frames: int = 0

    def _init_state(self, full_h: int, full_w: int) -> None:
        n_sec = self._sos.shape[0]
        self._zi = np.zeros((n_sec, 2, full_h, full_w, 3), dtype=np.float32)

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Apply one step of Eulerian color magnification.

        Args:
            frame_bgr: uint8 BGR frame from OpenCV.

        Returns:
            uint8 BGR frame with heartbeat colour pulse amplified. Looks like
            a normal video — the pulsing is visible as a natural colour shimmer
            on skin, not as artificial overlays.
        """
        h, w = frame_bgr.shape[:2]

        # ── 1. BGR → RGB → YIQ (float32) ─────────────────────────────────────
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        yiq = _rgb_to_yiq(rgb)

        # ── 2. Gaussian pyramid → coarsest level ─────────────────────────────
        pyr = _gaussian_pyramid(yiq, self.levels)
        coarse = pyr[-1]  # small, heavily blurred — no edge info

        # ── 3. Upsample coarsest level back to full resolution ────────────────
        #    (this is the step specified in the paper; amplification is added
        #     at full resolution so spatial blurring is preserved)
        upsampled = _upsample_to(coarse, h, w)

        # ── 4. Lazy init IIR state ────────────────────────────────────────────
        if self._zi is None or (h, w) != self._frame_shape:
            self._frame_shape = (h, w)
            self._init_state(h, w)
            self._warmup_frames = 0

        # ── 5. One-step IIR bandpass on the upsampled frame ──────────────────
        inp = upsampled[np.newaxis]                        # (1, H, W, 3)
        filtered_4d, self._zi = sosfilt(self._sos, inp, axis=0, zi=self._zi)
        filtered = filtered_4d[0]                          # (H, W, 3)

        self._warmup_frames += 1
        if self._warmup_frames < 20:
            # IIR cold-start transient — return original untouched
            return frame_bgr.copy()

        # ── 6. Amplify: Y × α, I and Q × α × A ──────────────────────────────
        magnified = filtered.copy()
        magnified[:, :, 0] *= self.alpha                        # Y
        magnified[:, :, 1] *= self.alpha * self.attenuation     # I
        magnified[:, :, 2] *= self.alpha * self.attenuation     # Q

        # ── 7. Add to original YIQ and reconstruct RGB → BGR ─────────────────
        out_yiq = yiq + magnified
        out_rgb = _yiq_to_rgb(out_yiq)
        out_bgr = cv2.cvtColor(
            np.clip(out_rgb, 0, 255).astype(np.uint8),
            cv2.COLOR_RGB2BGR,
        )
        return out_bgr


# ── HUD overlay ───────────────────────────────────────────────────────────────

def render_pulse_view(
    magnified_bgr: np.ndarray,
    bpm: Optional[float],
    warmup_done: bool,
) -> np.ndarray:
    """
    Adds a minimal transparent HUD to the EVM pulse frame.
    The frame itself is not altered — no false colors, no masks.
    """
    vis = magnified_bgr.copy()

    if not warmup_done:
        cv2.putText(
            vis, "Calibrating pulse magnifier...",
            (20, vis.shape[0] // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 220, 80), 2,
        )
        return vis

    # Semi-transparent banner at top
    banner_h = 56
    vis[:banner_h] = (vis[:banner_h].astype(np.float32) * 0.40).astype(np.uint8)

    bpm_text = f"BPM: {bpm:.0f}" if bpm else "BPM: --"
    cv2.putText(vis, "PULSE MAGNIFIER  (EVM)", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 255, 180), 1)
    cv2.putText(vis, bpm_text, (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 255, 0), 2)

    hint = "Heartbeat visible as rhythmic shimmer on skin"
    (hw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    xh = max(0, (vis.shape[1] - hw) // 2)
    cv2.rectangle(vis, (xh - 4, vis.shape[0] - 24),
                  (xh + hw + 4, vis.shape[0]), (0, 0, 0), -1)
    cv2.putText(vis, hint, (xh, vis.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 190, 190), 1)

    return vis

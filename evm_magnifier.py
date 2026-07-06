"""
Eulerian Video Magnification (EVM) for visualizing heartbeat as skin color shimmer.

Based on Wu et al. 2012. Uses YIQ color space + Gaussian pyramid blur + IIR
bandpass filter to amplify subtle color changes caused by blood flow.
"""
import cv2
import numpy as np
from scipy.signal import butter, sosfilt

_RGB2YIQ = np.array([
    [0.299, 0.587, 0.114],
    [0.59590059, -0.27455667, -0.32134392],
    [0.21153661, -0.52273617, 0.31119955],
], dtype=np.float32)

_YIQ2RGB = np.linalg.inv(_RGB2YIQ).astype(np.float32)


def _rgb_to_yiq(rgb):
    return rgb.reshape(-1, 3) @ _RGB2YIQ.T


def _yiq_to_rgb(yiq):
    return yiq.reshape(-1, 3) @ _YIQ2RGB.T


class PulseMagnifier:
    def __init__(self, fps=30.0, lo_hz=0.75, hi_hz=2.5, alpha=50.0, levels=4):
        self.alpha = alpha
        self.levels = levels
        sos = butter(1, [lo_hz / (fps / 2), hi_hz / (fps / 2)], btype="band", output="sos")
        self._sos = sos
        self._zi = None
        self._frame_shape = None
        self._warmup = 0

    def process(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

        yiq = _rgb_to_yiq(rgb).reshape(h, w, 3)

        cur = yiq.astype(np.float32)
        for _ in range(self.levels - 1):
            cur = cv2.pyrDown(cur)
        coarse = cv2.resize(cur, (w, h), interpolation=cv2.INTER_LINEAR)

        if self._zi is None or (h, w) != self._frame_shape:
            self._frame_shape = (h, w)
            self._zi = np.zeros((self._sos.shape[0], 2, h, w, 3), dtype=np.float32)
            self._warmup = 0

        filtered, self._zi = sosfilt(self._sos, coarse[np.newaxis], axis=0, zi=self._zi)
        self._warmup += 1

        if self._warmup < 20:
            return frame_bgr.copy()

        filtered = filtered[0]
        filtered[..., 0] *= self.alpha
        filtered[..., 1] *= self.alpha
        filtered[..., 2] *= self.alpha

        out_yiq = yiq + filtered
        out_rgb = _yiq_to_rgb(out_yiq).reshape(h, w, 3)
        out_bgr = cv2.cvtColor(np.clip(out_rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        return out_bgr

    @property
    def warm(self):
        return self._warmup >= 20

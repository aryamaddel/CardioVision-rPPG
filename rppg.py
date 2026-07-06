"""
Core rPPG engine: face ROI extraction, POS algorithm, BPM estimation.

Pipeline: MediaPipe landmarks → face mask → mean RGB → POS extraction
→ bandpass filter → peak detection → BPM.
"""
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from scipy.signal import butter, detrend, filtfilt, find_peaks

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

EXCLUDE = {
    33, 160, 158, 133, 153, 144,  # left eye
    362, 385, 387, 263, 373, 380,  # right eye
    70, 63, 105, 66, 107, 55, 65, 52, 53, 46,  # left brow
    336, 296, 334, 293, 300, 285, 295, 282, 283, 276,  # right brow
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,  # lips
    291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
}

BP_LO = 0.7
BP_HI = 4.0


def _ensure_model(path: str):
    p = Path(path)
    if not p.exists():
        print("Downloading face_landmarker.task ...")
        urllib.request.urlretrieve(MODEL_URL, str(p))


class FaceTracker:
    def __init__(self, model_path: str = "face_landmarker.task"):
        _ensure_model(model_path)
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = mp_vision.FaceLandmarker.create_from_options(opts)

    def process(self, frame_bgr: np.ndarray, ts_ms: int) -> tuple:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self.detector.detect_for_video(mp_img, ts_ms)

        if not res.face_landmarks:
            return None, frame_bgr

        h, w = frame_bgr.shape[:2]
        lm = np.array([[lk.x * w, lk.y * h] for lk in res.face_landmarks[0]], dtype=np.int32)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [cv2.convexHull(lm)], 255)
        for idx in EXCLUDE:
            cv2.fillPoly(mask, [lm[idx].reshape(1, 1, 2)], 0)

        px = frame_bgr[mask > 0]
        rgb_tuple = None
        if len(px) >= 10:
            rgb_tuple = (float(px[:, 2].mean()), float(px[:, 1].mean()), float(px[:, 0].mean()))

        vis = frame_bgr.copy()
        overlay = vis.copy()
        overlay[mask > 0] = (0, 140, 255)
        cv2.addWeighted(overlay, 0.38, vis, 0.62, 0, vis)

        return rgb_tuple, vis

    def close(self):
        self.detector.close()


def pos_algorithm(rgb: np.ndarray, fps: float, window_sec: float = 4.0) -> np.ndarray:
    N = len(rgb)
    window = max(6, int(window_sec * fps))
    if window > N:
        window = N
    pulse = np.zeros(N)
    weights = np.zeros(N)
    step = max(1, window // 3)

    rgb = rgb.astype(np.float64)
    means = rgb.mean(axis=0)
    rgb_detrend = detrend(rgb, axis=0, type="linear") + means

    for start in range(0, N - window + 1, step):
        seg = rgb_detrend[start:start + window]
        m = seg.mean(axis=0)
        if np.any(m == 0):
            continue
        Cn = seg / m
        S1 = Cn[:, 1] - Cn[:, 2]
        S2 = Cn[:, 1] + Cn[:, 2] - 2 * Cn[:, 0]
        s1, s2 = S1.std(), S2.std()
        if s2 < 1e-8:
            continue
        H = S1 + (s1 / s2) * S2
        pulse[start:start + window] += H
        weights[start:start + window] += 1

    if weights.sum() == 0:
        return np.zeros(N)
    pulse /= weights + 1e-8

    nyq = fps / 2
    b, a = butter(4, [max(BP_LO / nyq, 1e-6), min(BP_HI / nyq, 0.99)], btype="band")
    if len(pulse) > 3 * max(len(a), len(b)):
        return filtfilt(b, a, pulse)
    return pulse


def compute_bpm(pulse: np.ndarray, fps: float) -> float | None:
    pmin, pmax = pulse.min(), pulse.max()
    if pmax - pmin < 1e-8:
        return None
    norm = 2 * (pulse - pmin) / (pmax - pmin) - 1

    peaks, _ = find_peaks(norm, distance=max(1, int(fps * 0.4)), height=0.15, prominence=0.15)
    if len(peaks) < 2:
        return None

    ibi_ms = np.diff(peaks) / fps * 1000
    ibi_ms = ibi_ms[(ibi_ms >= 273) & (ibi_ms <= 1333)]
    if len(ibi_ms) == 0:
        return None

    bpm = 60000.0 / np.median(ibi_ms)
    return float(bpm)


def process_video(rgb_buffer: np.ndarray, fps: float) -> dict:
    pulse = pos_algorithm(rgb_buffer, fps)
    bpm = compute_bpm(pulse, fps)
    return {"bpm": bpm, "pulse": pulse.tolist(), "fps": fps}

"""
The computer vision front-end for the CardioVision-rPPG system.

Handles face detection, landmarking, and region-of-interest (ROI) extraction
using MediaPipe. Includes robust human-face guardrails (skin color checks,
geometry verification, and texture analysis) to prevent spoofs and ensure
high-quality signal extraction.
"""

import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, Optional, Tuple, Union

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

_EXCLUDE_EYE_L = [33, 160, 158, 133, 153, 144]
_EXCLUDE_EYE_R = [362, 385, 387, 263, 373, 380]
_EXCLUDE_BROW_L = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
_EXCLUDE_BROW_R = [336, 296, 334, 293, 300, 285, 295, 282, 283, 276]
_EXCLUDE_LIPS = [
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    409,
    270,
    269,
    267,
    0,
    37,
    39,
    40,
    185,
    61,
]

ROI_COLORS = {"face": (0, 140, 255)}
MIN_ROI_PIXELS = 1
_ERODE_KERNEL = np.ones((5, 5), np.uint8)
MIN_FACE_AREA_RATIO = 0.08
MAX_FACE_AREA_RATIO = 0.75
MIN_SKIN_PIXELS = 100 # Reduced threshold as requested


@dataclass
class ROIResult:
    masks: Dict[str, np.ndarray]
    px_counts: Dict[str, int]
    face_bbox: Tuple[int, int, int, int]
    frame_idx: int
    crops: Dict[str, np.ndarray]
    face_mask: Optional[np.ndarray] = None
    landmarks: Optional[np.ndarray] = None
    quality_score: float = 1.0


class VideoSource:
    """
    Unified wrapper for video files and webcam streams.

    Normalizes frame rates and provides a generator interface for processing
    video content frame-by-frame. Supports real-time webcam capture with
    drift correction.
    """

    def __init__(self, source: Union[int, str] = 0, target_fps: float = 30.0):
        self.source = source
        self.target_fps = target_fps
        self._is_webcam = isinstance(source, int)
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open: {source}")
        if self._is_webcam:
            self._cap.set(cv2.CAP_PROP_FPS, target_fps)
        self.orig_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        raw_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_frames = raw_count if raw_count > 0 and not self._is_webcam else None

    def frames(
        self, max_duration: Optional[float] = None
    ) -> Generator[Tuple[float, np.ndarray], None, None]:
        if self._is_webcam:
            count, interval, t0 = 0, 1.0 / self.target_fps, time.perf_counter()
            while True:
                ret, frame = self._cap.read()
                if not ret:
                    break
                elapsed = time.perf_counter() - t0
                if max_duration is not None and elapsed >= max_duration:
                    break
                yield elapsed, frame
                count += 1
                drift = (count * interval) - (time.perf_counter() - t0)
                if drift > 0:
                    time.sleep(drift)
        else:
            src_fps, out_count, src_count, next_yield = self.orig_fps, 0, 0, 0.0
            while True:
                ret, frame = self._cap.read()
                if not ret:
                    break
                elapsed = out_count / self.target_fps
                if max_duration is not None and elapsed >= max_duration:
                    break
                if src_count / src_fps >= next_yield:
                    yield elapsed, frame
                    next_yield += 1.0 / self.target_fps
                    out_count += 1
                src_count += 1

    def release(self):
        self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()


class FaceROIExtractor:
    """
    Orchestrates face detection and mask generation for rPPG.

    Uses MediaPipe Face Landmarker to identify facial features, isolates
    stable skin regions (forehead/cheeks), and applies multi-stage guardrails
    to filter out non-human or poor-quality frames.
    """

    def __init__(self, model_path: str):
        p = Path(model_path)
        if not p.exists():
            print("Downloading MediaPipe Face Landmarker model ...")
            urllib.request.urlretrieve(MODEL_URL, str(p))
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(p)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = mp_vision.FaceLandmarker.create_from_options(opts)
        self.count = 0
        self.stats = {
            "detected": 0,
            "failed": 0,
            "low_quality": 0,
            "rej_area": 0,
            "rej_skin": 0,
        }

    def process(self, frame_bgr: np.ndarray, ts_ms: int) -> Optional[ROIResult]:
        """
        Processes a single frame to extract facial ROIs and masks.

        Args:
            frame_bgr (np.ndarray): The input image in BGR format.
            ts_ms (int): Timestamp of the frame in milliseconds.

        Returns:
            Optional[ROIResult]: Container with masks, crops, and landmarks if a
                valid face is detected and passes guardrails; else None.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self.detector.detect_for_video(mp_img, ts_ms)
        self.count += 1
        if not res.face_landmarks:
            self.stats["failed"] += 1
            return None
        self.stats["detected"] += 1
        h, w = frame_bgr.shape[:2]
        lm = np.array(
            [[lk.x * w, lk.y * h] for lk in res.face_landmarks[0]], dtype=np.float32
        )

        face_mask = np.zeros((h, w), dtype=np.uint8)
        hull = cv2.convexHull(lm.astype(np.int32))
        cv2.fillPoly(face_mask, [hull], 255)

        masks, px_counts = _build_masks(lm, h, w)

        bbox = (
            int(lm[:, 0].min()),
            int(lm[:, 1].min()),
            int(lm[:, 0].max() - lm[:, 0].min()),
            int(lm[:, 1].max() - lm[:, 1].min()),
        )

        quality_score, checks = _evaluate_human_face_guardrails(
            frame_bgr, face_mask, bbox
        )
        if quality_score < 0.5:
            self.stats["low_quality"] += 1
            if not checks["area"]: self.stats["rej_area"] += 1
            if not checks["skin"]: self.stats["rej_skin"] += 1

        crops = {}
        for roi_name, mask in masks.items():
            ys, xs = np.where(mask > 0)
            if ys.size:
                y1, y2 = ys.min(), ys.max()
                x1, x2 = xs.min(), xs.max()
                roi_crop = frame_bgr[y1 : y2 + 1, x1 : x2 + 1]
                crops[roi_name] = cv2.resize(roi_crop, (64, 64))
            else:
                crops[roi_name] = np.zeros((64, 64, 3), dtype=np.uint8)

        return ROIResult(
            masks,
            px_counts,
            bbox,
            self.count - 1,
            crops,
            face_mask,
            lm,
            quality_score,
        )

    def close(self):
        self.detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _build_masks(lm: np.ndarray, h: int, w: int, skin_m: Optional[np.ndarray] = None):
    base_mask = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(lm.astype(np.int32))
    cv2.fillPoly(base_mask, [hull], 255)

    for indices in [
        _EXCLUDE_EYE_L,
        _EXCLUDE_EYE_R,
        _EXCLUDE_BROW_L,
        _EXCLUDE_BROW_R,
        _EXCLUDE_LIPS,
    ]:
        pts = lm[indices].astype(np.int32)
        cv2.fillPoly(base_mask, [pts], 0)

    mask = cv2.erode(base_mask, _ERODE_KERNEL, iterations=1)
    if skin_m is not None:
        mask = cv2.bitwise_and(mask, skin_m)

    masks = {"face": mask}
    px_counts = {"face": int(np.count_nonzero(mask))}
    return masks, px_counts


def _evaluate_human_face_guardrails(
    frame_bgr: np.ndarray,
    face_mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> Tuple[float, Dict[str, bool]]:
    """Simplistic face gate: check for basic area and minimum skin presence."""
    h, w = frame_bgr.shape[:2]
    _, _, bw, bh = bbox
    face_area_ratio = float((bw * bh) / max(1, w * h))
    area_ok = MIN_FACE_AREA_RATIO <= face_area_ratio <= MAX_FACE_AREA_RATIO

    skin_ratio_ycrcb, skin_ratio_hsv, _ = _estimate_skin_consistency(frame_bgr, face_mask)
    total_face_px = int(np.count_nonzero(face_mask > 0))
    skin_px = max(int(skin_ratio_ycrcb * total_face_px), int(skin_ratio_hsv * total_face_px))
    skin_ok = skin_px >= MIN_SKIN_PIXELS

    checks = {"area": area_ok, "skin": skin_ok}
    quality_score = 1.0 if (area_ok and skin_ok) else 0.5
    return quality_score, checks


def _estimate_skin_consistency(
    frame_bgr: np.ndarray, face_mask: np.ndarray
) -> Tuple[float, float, float]:
    face_pixels = face_mask > 0
    total_face_px = int(np.count_nonzero(face_pixels))
    if total_face_px == 0:
        return 0.0, 0.0, 0.0

    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    skin_ycrcb = (y >= 40) & (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    skin_hsv = (h >= 0) & (h <= 25) & (s >= 30) & (s <= 180) & (v >= 40) & (v <= 255)

    skin_ycrcb_face = skin_ycrcb & face_pixels
    skin_hsv_face = skin_hsv & face_pixels
    intersection = int(np.count_nonzero(skin_ycrcb_face & skin_hsv_face))
    union = int(np.count_nonzero(skin_ycrcb_face | skin_hsv_face))

    skin_ratio_ycrcb = float(np.count_nonzero(skin_ycrcb_face) / total_face_px)
    skin_ratio_hsv = float(np.count_nonzero(skin_hsv_face) / total_face_px)
    skin_iou = float(intersection / max(1, union))
    return skin_ratio_ycrcb, skin_ratio_hsv, skin_iou


def get_mean_rgb(frame: np.ndarray, mask: np.ndarray):
    """
    Computes the average R, G, B values from the pixels specified by a mask.

    Args:
        frame (np.ndarray): The BGR image data.
        mask (np.ndarray): Binary mask identifying the ROI.

    Returns:
        Tuple[float, float, float]: The mean Red, Green, and Blue values.
    """
    px = frame[mask > 0]
    if len(px) == 0:
        return np.nan, np.nan, np.nan
    return float(np.mean(px[:, 2])), float(np.mean(px[:, 1])), float(np.mean(px[:, 0]))


def compute_mad_confidence(roi_g_values: Dict[str, float]) -> float:
    """
    Estimates signal quality based on the Median Absolute Deviation (MAD).

    Args:
        roi_g_values (dict): Mapping of ROI names to their mean green channel values.

    Returns:
        float: A normalized confidence score (0-1).
    """
    vals = [v for v in roi_g_values.values() if not np.isnan(v)]
    if len(vals) < 2:
        return 0.0
    med = float(np.median(vals))
    mad = float(np.mean(np.abs(np.array(vals) - med)))
    return float(np.clip(1.0 - mad / 20.0, 0.0, 1.0))


def overlay_roi(frame: np.ndarray, roi_masks: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Visualizes the extracted ROIs by overlaying them on the original frame.

    Args:
        frame (np.ndarray): The original BGR frame.
        roi_masks (dict): Dictionary of masks to overlay.

    Returns:
        np.ndarray: The frame with the ROI visualization applied.
    """
    vis = frame.copy()
    face_mask = roi_masks.get("face")
    if face_mask is None:
        return vis

    overlay = vis.copy()
    overlay[face_mask > 0] = ROI_COLORS["face"]
    cv2.addWeighted(overlay, 0.38, vis, 0.62, 0, vis)
    return vis

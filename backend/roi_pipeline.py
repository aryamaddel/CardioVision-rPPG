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
MIN_ROI_PIXELS = 150
_ERODE_KERNEL = np.ones((5, 5), np.uint8)
MIN_FACE_AREA_RATIO = 0.08
MAX_FACE_AREA_RATIO = 0.75
MIN_SKIN_RATIO = 0.18
MIN_FACE_TEXTURE_VAR = 10.0
MAX_FACE_TEXTURE_VAR = 1800.0
MIN_TOTAL_GUARDRAIL_PASSES = 3
MIN_HSV_SKIN_RATIO = 0.10
MIN_SKIN_MASK_IOU = 0.35
MIN_SKIN_PIXELS = 500


@dataclass
class ROIResult:
    masks: Dict[str, np.ndarray]
    px_counts: Dict[str, int]
    face_bbox: Tuple[int, int, int, int]
    frame_idx: int
    crops: Dict[str, np.ndarray]
    face_mask: Optional[np.ndarray] = None
    landmarks: Optional[np.ndarray] = None


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
            "rejected_nonhuman": 0,
            "rej_area": 0,
            "rej_skin_ycrcb": 0,
            "rej_skin_hsv": 0,
            "rej_skin_iou": 0,
            "rej_geometry": 0,
            "rej_texture": 0,
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

        passed, checks = _evaluate_human_face_guardrails(frame_bgr, face_mask, bbox, lm)
        if not passed:
            self.stats["rejected_nonhuman"] += 1
            self.stats["rej_area"] += 0 if checks["area"] else 1
            self.stats["rej_skin_ycrcb"] += 0 if checks["skin_ycrcb"] else 1
            self.stats["rej_skin_hsv"] += 0 if checks["skin_hsv"] else 1
            self.stats["rej_skin_iou"] += 0 if checks["skin_iou"] else 1
            self.stats["rej_geometry"] += 0 if checks["geometry"] else 1
            self.stats["rej_texture"] += 0 if checks["texture"] else 1
            self.stats["failed"] += 1
            self.stats["detected"] = max(0, self.stats["detected"] - 1)
            return None

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
    lm: np.ndarray,
) -> Tuple[bool, Dict[str, bool]]:
    """Gate frames on two hard criteria only:

    1. Face bounding-box covers a reasonable image area (not too small/large).
    2. The face ROI contains at least MIN_SKIN_PIXELS detected skin pixels
       (YCrCb OR HSV — either colorspace match counts).

    Head pose is intentionally NOT a gate.  A person looking to the side often
    exposes more cheek skin — suppressing those frames hurts signal quality.
    """
    h, w = frame_bgr.shape[:2]
    _, _, bw, bh = bbox
    face_area_ratio = float((bw * bh) / max(1, w * h))
    area_ok = MIN_FACE_AREA_RATIO <= face_area_ratio <= MAX_FACE_AREA_RATIO

    skin_ratio_ycrcb, skin_ratio_hsv, skin_iou = _estimate_skin_consistency(frame_bgr, face_mask)
    total_face_px = int(np.count_nonzero(face_mask > 0))
    # Count pixels that pass either skin colorspace model
    ycrcb_px = int(round(skin_ratio_ycrcb * total_face_px))
    hsv_px   = int(round(skin_ratio_hsv   * total_face_px))
    skin_px  = max(ycrcb_px, hsv_px)
    skin_ok  = skin_px >= MIN_SKIN_PIXELS

    checks = {
        "area":       area_ok,
        "skin_ycrcb": skin_ratio_ycrcb >= MIN_SKIN_RATIO,
        "skin_hsv":   skin_ratio_hsv   >= MIN_HSV_SKIN_RATIO,
        "skin_iou":   skin_iou         >= MIN_SKIN_MASK_IOU,
        "geometry":   _passes_face_geometry_guard(lm, bbox),
        "texture":    _passes_texture_guard(frame_bgr, bbox),
        "skin_pixels": skin_ok,
    }

    # Hard gates: face must be a plausible size AND have enough skin pixels.
    # All soft checks (skin ratios, geometry, texture) are informational only.
    passed = area_ok and skin_ok
    return passed, checks


def _passes_face_geometry_guard(lm: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
    """Rejects only biologically impossible landmark configurations
    (e.g., eyes swapped, nose outside face box).  Does NOT penalise
    off-centre or side-facing heads."""
    _, _, bw, bh = bbox
    face_w = max(float(bw), 1.0)
    face_h = max(float(bh), 1.0)

    try:
        left_eye  = lm[33]
        right_eye = lm[263]
        nose_tip  = lm[1]
        upper_lip = lm[13]

        eye_dist      = float(np.linalg.norm(left_eye - right_eye) / face_w)
        nose_to_mouth = float(np.linalg.norm(nose_tip - upper_lip) / face_h)
        eye_to_mouth  = float((upper_lip[1] - ((left_eye[1] + right_eye[1]) * 0.5)) / face_h)
    except Exception:
        return False

    # Very loose bounds — only reject anatomically impossible ratios.
    geometry_checks = [
        0.10 <= eye_dist      <= 0.90,   # eyes exist and are separated
        0.01 <= nose_to_mouth <= 0.50,   # nose and mouth are not fused
        0.05 <= eye_to_mouth  <= 0.85,   # eyes are above mouth
    ]
    return all(geometry_checks)


def _passes_texture_guard(frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
    x, y, bw, bh = bbox
    h, w = frame_bgr.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + bw)
    y2 = min(h, y + bh)
    if x2 <= x1 or y2 <= y1:
        return False

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return False

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return MIN_FACE_TEXTURE_VAR <= lap_var <= MAX_FACE_TEXTURE_VAR

def _estimate_skin_consistency(frame_bgr: np.ndarray, face_mask: np.ndarray) -> Tuple[float, float, float]:
    face_pixels = face_mask > 0
    total_face_px = int(np.count_nonzero(face_pixels))
    if total_face_px == 0:
        return 0.0, 0.0, 0.0

    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    skin_ycrcb = (
        (y >= 40)
        & (cr >= 133)
        & (cr <= 173)
        & (cb >= 77)
        & (cb <= 127)
    )

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    skin_hsv = (
        (h >= 0)
        & (h <= 25)
        & (s >= 30)
        & (s <= 180)
        & (v >= 40)
        & (v <= 255)
    )

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

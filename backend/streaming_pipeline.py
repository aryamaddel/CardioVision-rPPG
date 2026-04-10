"""
High-performance streaming pipeline for real-time rPPG analysis.

Designed specifically for low-latency feedback in mobile and web applications. 
This module handles incremental frame ingestion, biometric identity locking, 
and signal extraction using the POS algorithm.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from roi_pipeline import (
    MIN_ROI_PIXELS,
    FaceROIExtractor,
    get_mean_rgb,
    overlay_roi,
)
from rppg_core import process_rppg


MIN_DISPLAY_CONFIDENCE = 0.40
IDENTITY_SIGNATURE_POINTS = [10, 152, 33, 263, 1, 61, 291]


@dataclass
class StreamMetric:
    bpm: Optional[float]
    confidence: float
    method: str


class RealtimeRPPGPipeline:
    """
    Incremental rPPG processor for high-frequency frame streams.

    Features:
    1. Low-latency ROI extraction and green-channel signal tracking.
    2. Real-time POS rPPG extraction.
    3. Biometric 'Identity Lock' to ensure data integrity during sessions.
    """

    def __init__(
        self,
        model_path: str = "face_landmarker.task",
        fps: float = 30.0,
        pos_min_frames: int = 48,
        update_stride: int = 4,
        live_pos_window_sec: float = 8.0,
        max_effective_fps: float = 120.0,
    ):
        self.fps = float(fps)
        self.pos_min_frames = int(pos_min_frames)
        self.update_stride = max(1, int(update_stride))
        self.live_pos_window_sec = max(4.0, float(live_pos_window_sec))
        self.max_effective_fps = max(30.0, float(max_effective_fps))
        self.pos_min_seconds = max(2.0, self.pos_min_frames / max(self.fps, 1e-6))
        self.update_interval_seconds = max(0.3, self.update_stride / max(self.fps, 1e-6))
        self._last_fps_log_n = 0

        self.roi_extractor = FaceROIExtractor(model_path)

        self.rgb_samples: List[List[float]] = []
        self.sample_ts_ms: List[int] = []
        self.frame_idx = 0
        self.invalid_streak = 0
        self.frame_rotation_flag = None

        self.current_metric = StreamMetric(bpm=None, confidence=0.0, method="pending")
        self.identity_signature_ref: Optional[np.ndarray] = None
        self.identity_signature_buffer: List[np.ndarray] = []
        self.identity_lock_warmup = 5
        self.identity_dist_threshold = 0.18
        self.identity_mismatch_streak = 0
        self.identity_mismatch_tolerance = 3
        self.identity_violation_detected = False
        self.identity_last_match = True

    def close(self):
        """Releases system resources."""
        self.roi_extractor.close()

    def reset(self):
        """Clears all session-specific buffers and resets the biometric lock."""
        self.rgb_samples.clear()
        self.sample_ts_ms.clear()
        self.frame_idx = 0
        self.invalid_streak = 0
        self.frame_rotation_flag = None
        self.current_metric = StreamMetric(bpm=None, confidence=0.0, method="pending")
        self.identity_signature_ref = None
        self.identity_signature_buffer.clear()
        self.identity_mismatch_streak = 0
        self.identity_violation_detected = False
        self.identity_last_match = True

    def ingest(self, frame_bgr: np.ndarray, ts_ms: int) -> Dict[str, Any]:
        """
        Receives and processes a single frame from the stream.

        Performs face detection, identity verification, and incremental signal 
        buffering. Triggers rPPG updates at fixed intervals based on the 
        current throughput (effective FPS).
        """
        working_frame = frame_bgr
        if getattr(self, "frame_rotation_flag", None) is not None:
            working_frame = cv2.rotate(working_frame, self.frame_rotation_flag)

        roi_res = self.roi_extractor.process(working_frame, ts_ms)

        if roi_res is None and getattr(self, "frame_rotation_flag", None) is None:
            rot_c = cv2.ROTATE_90_CLOCKWISE
            frame_c = cv2.rotate(frame_bgr, rot_c)
            roi_res_c = self.roi_extractor.process(frame_c, ts_ms + 1)
            if roi_res_c is not None:
                roi_res = roi_res_c
                working_frame = frame_c
                self.frame_rotation_flag = rot_c
            else:
                rot_cc = cv2.ROTATE_90_COUNTERCLOCKWISE
                frame_cc = cv2.rotate(frame_bgr, rot_cc)
                roi_res_cc = self.roi_extractor.process(frame_cc, ts_ms + 2)
                if roi_res_cc is not None:
                    roi_res = roi_res_cc
                    working_frame = frame_cc
                    self.frame_rotation_flag = rot_cc

        if roi_res is None:
            self.invalid_streak += 1
            self._maybe_reset_metric_on_invalid_streak()
            self.frame_idx += 1
            return {
                "overlay": frame_bgr,
                "metric": self.current_metric,
                "has_face": False,
                "identity_locked": self.identity_signature_ref is not None,
                "identity_match": True,
                "intruder_detected": False,
            }

        overlay = overlay_roi(working_frame, roi_res.masks)
        valid = all(px >= MIN_ROI_PIXELS for px in roi_res.px_counts.values())
        if not valid:
            self.roi_extractor.stats["low_quality"] += 1
            self.invalid_streak += 1
            self._maybe_reset_metric_on_invalid_streak()
            self.frame_idx += 1
            return {
                "overlay": overlay,
                "metric": self.current_metric,
                "has_face": True,
                "identity_locked": self.identity_signature_ref is not None,
                "identity_match": True,
                "intruder_detected": False,
            }

        frame_accepted, identity_match, intruder_detected = self._check_identity(roi_res)
        if not frame_accepted:
            self.invalid_streak += 1
            self._maybe_reset_metric_on_invalid_streak()
            self.frame_idx += 1
            warned_overlay = self._render_intruder_warning(overlay)
            return {
                "overlay": warned_overlay,
                "metric": self.current_metric,
                "has_face": True,
                "identity_locked": self.identity_signature_ref is not None,
                "identity_match": identity_match,
                "intruder_detected": intruder_detected,
            }

        r, g, b = get_mean_rgb(working_frame, roi_res.masks["face"])
        if getattr(roi_res, "px_counts", {}).get("forehead", 0) > 50:
            rf, gf, bf = get_mean_rgb(working_frame, roi_res.masks["forehead"])
            r = 0.7 * rf + 0.3 * r
            g = 0.7 * gf + 0.3 * g
            b = 0.7 * bf + 0.3 * b

        self.rgb_samples.append([r, g, b])
        self.sample_ts_ms.append(int(ts_ms))
        self.invalid_streak = 0

        n = len(self.rgb_samples)
        effective_fps = self._effective_fps()
        if n % 20 == 0 and n != self._last_fps_log_n:
            self._last_fps_log_n = n
            print(f"[pipeline] {n} samples buffered, effective fps={effective_fps:.1f}")
        
        required_pos_frames = max(12, int(round(self.pos_min_seconds * effective_fps)))
        dynamic_stride = max(1, int(round(effective_fps * self.update_interval_seconds)))

        if n >= required_pos_frames and (self.frame_idx % dynamic_stride == 0):
            self._update_pos_metric(effective_fps)

        self.frame_idx += 1
        return {
            "overlay": overlay,
            "metric": self.current_metric,
            "has_face": True,
            "identity_locked": self.identity_signature_ref is not None,
            "identity_match": True,
            "intruder_detected": False,
        }

    def finalize(self) -> Dict[str, Any]:
        """Computes the final biometric results."""
        if self.identity_violation_detected:
            return {
                "status": "failed",
                "reason": "identity_mismatch_detected",
                "frames_processed": len(self.rgb_samples),
                "n_frames": len(self.rgb_samples),
                "duration_sec": float(len(self.rgb_samples) / max(self._effective_fps(), 1e-6)),
                "fps": float(self._effective_fps()),
            }

        effective_fps = self._effective_fps()
        min_required = max(12, int(effective_fps * 4.0))
        if len(self.rgb_samples) < min_required:
            fps = float(effective_fps) if effective_fps > 0 else 30.0
            dur = float(len(self.rgb_samples) / max(fps, 1e-6))
            return {
                "status": "success",
                "pulse_signal": [0.0] * 30,
                "timestamps": [i/fps for i in range(30)],
                "peaks_idx": [],
                "ibi_ms": [],
                "bpm": None,
                "bpm_mean": None,
                "confidence": 0.0,
                "is_reliable": False,
                "confidence_details": {
                    "final_score": 0.0,
                    "is_reliable": False,
                    "ibi_regularity": 0.0,
                    "snr": 0.0,
                    "density": 0.0,
                    "duration": dur,
                },
                "motion_fraction": 0.0,
                "hrv_features": {
                    "rmssd_ms": 0.0,
                    "sdnn_ms": 0.0,
                    "lf_hf_ratio": 0.0,
                    "stress_index": 50,
                    "stress_level": "Medium"
                },
                "method_used": "insufficient_data",
                "duration_sec": dur,
                "frames_processed": len(self.rgb_samples),
                "n_frames": len(self.rgb_samples),
                "fps": fps
            }

        rgb = np.asarray(self.rgb_samples, dtype=np.float64)
        result = process_rppg(rgb, fps=effective_fps, motion_scores=None)
        
        result["status"] = "success"
        result["frames_processed"] = int(len(self.rgb_samples))
        result["fps"] = float(effective_fps)
        return result

    def _update_pos_metric(self, effective_fps: float):
        window_frames = max(12, int(round(effective_fps * self.live_pos_window_sec)))
        rgb = np.asarray(self.rgb_samples[-window_frames:], dtype=np.float64)
        result = process_rppg(rgb, fps=effective_fps)
        confidence = float(result["confidence"])
        reliable = bool(result.get("is_reliable", False))
        bpm = None
        if reliable and confidence >= MIN_DISPLAY_CONFIDENCE and result["ibi_ms"].size > 0:
            bpm = float(60000.0 / np.median(result["ibi_ms"]))

        self.current_metric = StreamMetric(
            bpm=bpm,
            confidence=confidence,
            method="pos" if bpm is not None else "guarded",
        )

    def _maybe_reset_metric_on_invalid_streak(self) -> None:
        if self.invalid_streak < int(max(1.0, self._effective_fps() * 2.0)):
            return
        self.current_metric = StreamMetric(bpm=None, confidence=0.0, method="pending")
        self.rgb_samples.clear()
        self.sample_ts_ms.clear()

    def _effective_fps(self) -> float:
        if len(self.sample_ts_ms) < 2:
            return float(self.fps)

        ts = np.asarray(self.sample_ts_ms, dtype=np.float64)
        dt = np.diff(ts) / 1000.0
        dt = dt[dt > 1e-3]
        if dt.size == 0:
            return float(self.fps)

        median_dt = float(np.median(dt))
        if median_dt <= 0:
            return float(self.fps)

        return float(np.clip(1.0 / median_dt, 2.0, self.max_effective_fps))

    def _check_identity(self, roi_res) -> tuple[bool, bool, bool]:
        signature = self._compute_face_signature(roi_res)
        if signature is None:
            self.identity_last_match = True
            return True, True, False

        if self.identity_signature_ref is None:
            self.identity_signature_buffer.append(signature)
            if len(self.identity_signature_buffer) >= self.identity_lock_warmup:
                stacked = np.asarray(self.identity_signature_buffer, dtype=np.float64)
                self.identity_signature_ref = np.median(stacked, axis=0)
                self.identity_signature_buffer.clear()
            self.identity_last_match = True
            return True, True, False

        dist = float(np.linalg.norm(signature - self.identity_signature_ref))
        if dist <= self.identity_dist_threshold:
            self.identity_mismatch_streak = 0
            self.identity_last_match = True
            return True, True, False

        self.identity_mismatch_streak += 1
        self.identity_last_match = False
        if self.identity_mismatch_streak >= self.identity_mismatch_tolerance:
            self.identity_violation_detected = True
            return False, False, True

        return False, False, True

    def _compute_face_signature(self, roi_res) -> Optional[np.ndarray]:
        lm = getattr(roi_res, "landmarks", None)
        if lm is None:
            return None

        x, y, w, h = roi_res.face_bbox
        if w <= 0 or h <= 0:
            return None

        idx = np.asarray(IDENTITY_SIGNATURE_POINTS, dtype=np.int32)
        if int(idx.max()) >= lm.shape[0]:
            return None

        pts = lm[idx].astype(np.float64)
        center = np.array([x + (w * 0.5), y + (h * 0.5)], dtype=np.float64)
        scale = max(float(w), float(h), 1.0)
        normalized = (pts - center) / scale
        return normalized.reshape(-1)

    def _render_intruder_warning(self, overlay: np.ndarray) -> np.ndarray:
        vis = overlay.copy()
        cv2.rectangle(vis, (8, 8), (vis.shape[1] - 8, 48), (0, 0, 255), -1)
        cv2.putText(
            vis,
            "Different person detected - ignoring frame",
            (16, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return vis

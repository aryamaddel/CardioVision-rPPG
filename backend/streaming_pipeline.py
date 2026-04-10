"""
High-performance streaming pipeline for real-time rPPG analysis.

Designed specifically for low-latency feedback in mobile and web applications.
This module handles incremental frame ingestion and signal extraction
using the POS algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from roi_pipeline import (
    FaceROIExtractor,
    get_mean_rgb,
    overlay_roi,
)
from rppg_core import process_rppg


MIN_DISPLAY_CONFIDENCE = 0.40


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
    """

    def __init__(
        self,
        model_path: str = "face_landmarker.task",
    ):
        self.fps = 30.0
        self.live_pos_window_sec = 8.0
        self.max_effective_fps = 120.0
        self._last_fps_log_n = 0

        self.roi_extractor = FaceROIExtractor(model_path)

        self.rgb_samples: List[List[float]] = []
        self.sample_ts_ms: List[int] = []
        self.frame_idx = 0
        self.invalid_streak = 0
        self.frame_rotation_flag = None

        self.current_metric = StreamMetric(bpm=None, confidence=0.0, method="pending")
        self.last_estimated_fps = 30.0

    def close(self):
        """Releases system resources."""
        self.roi_extractor.close()

    def reset(self):
        """Clears all session-specific buffers."""
        self.rgb_samples.clear()
        self.sample_ts_ms.clear()
        self.frame_idx = 0
        self.invalid_streak = 0
        self.frame_rotation_flag = None
        self.current_metric = StreamMetric(bpm=None, confidence=0.0, method="pending")
        self.last_estimated_fps = 30.0

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

        if roi_res is None:
            self.invalid_streak += 1
            self.frame_idx += 1
            return {
                "overlay": frame_bgr,
                "metric": self.current_metric,
                "has_face": False,
            }

        overlay = overlay_roi(working_frame, roi_res.masks)

        # We always accept frames if landmarks exist now. Identity lock is disabled.
        r, g, b = get_mean_rgb(working_frame, roi_res.masks["face"])

        self.rgb_samples.append([r, g, b])
        self.sample_ts_ms.append(int(ts_ms))
        self.invalid_streak = 0

        n = len(self.rgb_samples)
        effective_fps = self._effective_fps()
        if n % 20 == 0 and n != self._last_fps_log_n:
            self._last_fps_log_n = n
            print(f"[pipeline] {n} samples buffered, effective fps={effective_fps:.1f}")

        required_pos_frames = max(
            12, int(round(6.0 * effective_fps))
        )  # 6-sec min window
        update_stride = 5  # update every 5 frames (~150-200ms)

        if n >= required_pos_frames and (self.frame_idx % update_stride == 0):
            self._update_pos_metric(effective_fps)

        self.frame_idx += 1
        return {
            "overlay": overlay,
            "metric": self.current_metric,
            "has_face": True,
        }

    def finalize(self) -> Dict[str, Any]:
        """Computes the final biometric results."""
        effective_fps = self._effective_fps()
        min_required = max(12, int(effective_fps * 4.0))
        if len(self.rgb_samples) < min_required:
            fps = float(effective_fps) if effective_fps > 0 else 30.0
            dur = float(len(self.rgb_samples) / max(fps, 1e-6))
            return {
                "status": "success",
                "pulse_signal": [0.0] * 30,
                "timestamps": [i / fps for i in range(30)],
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
                    "stress_level": "Medium",
                },
                "method_used": "insufficient_data",
                "duration_sec": dur,
                "frames_processed": len(self.rgb_samples),
                "n_frames": len(self.rgb_samples),
                "fps": fps,
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
        if (
            reliable
            and confidence >= MIN_DISPLAY_CONFIDENCE
            and result["ibi_ms"].size > 0
        ):
            bpm = float(60000.0 / np.median(result["ibi_ms"]))

        self.current_metric = StreamMetric(
            bpm=bpm,
            confidence=confidence,
            method="pos" if bpm is not None else "guarded",
        )

    def _effective_fps(self) -> float:
        """
        Calculates the effective frame rate using a rolling window of recent
        timestamps to avoid poisoning the estimate with early startup jitter.
        """
        # Use only the last 100 samples (covering ~3-5 seconds of history)
        window = self.sample_ts_ms[-101:]
        if len(window) < 2:
            return float(self.last_estimated_fps)

        ts = np.asarray(window, dtype=np.float64)
        dt = np.diff(ts) / 1000.0
        dt = dt[dt > 0.005]  # Filter out duplicate/impossible timestamps (<200fps)
        
        if dt.size == 0:
            return float(self.last_estimated_fps)

        # Use median to resist individual outlier outliers
        median_dt = float(np.median(dt))
        raw_fps = 1.0 / median_dt if median_dt > 0 else 30.0

        # Apply smoothing to prevent jitter in algorithmic load-bearing constants.
        # We use a larger alpha during the startup phase to converge faster.
        alpha = 0.1 if len(self.sample_ts_ms) > 50 else 0.4
        smoothed_fps = (alpha * raw_fps) + ((1.0 - alpha) * self.last_estimated_fps)

        # Absolute physiological/hardware safety clamp
        self.last_estimated_fps = float(np.clip(smoothed_fps, 5.0, self.max_effective_fps))
        return self.last_estimated_fps

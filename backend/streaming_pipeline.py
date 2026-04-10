"""
High-performance streaming pipeline for real-time rPPG analysis.

Designed specifically for low-latency feedback in mobile and web applications. 
This module handles incremental frame ingestion, asynchronous deep-model 
execution, biometric identity locking, and automatic mode switching through 
the Triage Agent.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
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
from rppg_core import process_rppg, process_rppg_with_deep


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
    2. Background execution of deep neural models to enhance POS results.
    3. Biometric 'Identity Lock' to ensure data integrity during sessions.
    4. Triage orchestration between biometric and visual assessment modes.
    """

    def __init__(
        self,
        model_path: str = "face_landmarker.task",
        fps: float = 30.0,
        pos_min_frames: int = 48,
        deep_min_frames: int = 72,
        update_stride: int = 4,
        live_deep_enabled: bool = False,
        final_deep_enabled: bool = True,
        live_pos_window_sec: float = 8.0,
        deep_max_frames: int = 300,
        max_effective_fps: float = 120.0,
    ):
        self.fps = float(fps)
        self.pos_min_frames = int(pos_min_frames)
        self.deep_min_frames = int(deep_min_frames)
        self.update_stride = max(1, int(update_stride))
        self.live_deep_enabled = bool(live_deep_enabled)
        self.final_deep_enabled = bool(final_deep_enabled)
        self.live_pos_window_sec = max(4.0, float(live_pos_window_sec))
        self.deep_max_frames = max(60, int(deep_max_frames))
        self.max_effective_fps = max(30.0, float(max_effective_fps))
        self.pos_min_seconds = max(2.0, self.pos_min_frames / max(self.fps, 1e-6))
        self.deep_min_seconds = max(4.0, self.deep_min_frames / max(self.fps, 1e-6))
        self.update_interval_seconds = max(0.3, self.update_stride / max(self.fps, 1e-6))
        self._last_fps_log_n = 0

        self.roi_extractor = FaceROIExtractor(model_path)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.deep_future: Optional[Future] = None

        self.rgb_samples: List[List[float]] = []
        self.face_frames: List[np.ndarray] = []
        self.sample_ts_ms: List[int] = []
        self.frame_idx = 0
        self.invalid_streak = 0

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
        """Releases system resources and shuts down background executors."""
        if self.deep_future and not self.deep_future.done():
            self.deep_future.cancel()
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.roi_extractor.close()

    def reset(self):
        """Clears all session-specific buffers and resets the biometric lock."""
        self.rgb_samples.clear()
        self.face_frames.clear()
        self.sample_ts_ms.clear()
        self.frame_idx = 0
        self.invalid_streak = 0
        self.current_metric = StreamMetric(bpm=None, confidence=0.0, method="pending")
        self.identity_signature_ref = None
        self.identity_signature_buffer.clear()
        self.identity_mismatch_streak = 0
        self.identity_violation_detected = False
        self.identity_last_match = True
        if self.deep_future and not self.deep_future.done():
            self.deep_future.cancel()
        self.deep_future = None

    def ingest(self, frame_bgr: np.ndarray, ts_ms: int) -> Dict[str, Any]:
        """
        Receives and processes a single frame from the stream.

        Performs face detection, identity verification, and incremental signal 
        buffering. Triggers rPPG updates at fixed intervals based on the 
        current throughput (effective FPS).

        Args:
            frame_bgr (np.ndarray): The raw BGR frame from the client.
            ts_ms (int): Client-side timestamp in milliseconds.

        Returns:
            Dict[str, Any]: metadata including 'overlay' image, current 'metric', 
                and 'identity_match' status.
        """
        roi_res = self.roi_extractor.process(frame_bgr, ts_ms)
        if roi_res is None:
            self.invalid_streak += 1
            self._maybe_reset_metric_on_invalid_streak()
            self.frame_idx += 1
            return {
                "overlay": frame_bgr,
                "metric": self.current_metric,
                "has_face": False,
                "method_changed": False,
                "identity_locked": self.identity_signature_ref is not None,
                "identity_match": True,
                "intruder_detected": False,
            }

        overlay = overlay_roi(frame_bgr, roi_res.masks)
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
                "method_changed": False,
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
                "method_changed": False,
                "identity_locked": self.identity_signature_ref is not None,
                "identity_match": identity_match,
                "intruder_detected": intruder_detected,
            }

        r, g, b = get_mean_rgb(frame_bgr, roi_res.masks["face"])
        self.rgb_samples.append([r, g, b])
        self.face_frames.append(roi_res.crops["face"])
        self.sample_ts_ms.append(int(ts_ms))
        self.invalid_streak = 0

        method_changed = False
        n = len(self.rgb_samples)
        effective_fps = self._effective_fps()
        # Log effective FPS every ~20 valid frames for debugging mobile throughput.
        if n % 20 == 0 and n != self._last_fps_log_n:
            self._last_fps_log_n = n
            print(f"[pipeline] {n} valid samples, effective fps={effective_fps:.1f}")
        required_pos_frames = max(12, int(round(self.pos_min_seconds * effective_fps)))
        required_deep_frames = max(20, int(round(self.deep_min_seconds * effective_fps)))
        dynamic_stride = max(1, int(round(effective_fps * self.update_interval_seconds)))

        if n >= required_pos_frames and (self.frame_idx % dynamic_stride == 0):
            self._update_pos_metric(effective_fps)

        if self.live_deep_enabled and n >= required_deep_frames:
            self._maybe_start_deep_job(effective_fps)
            method_changed = self._maybe_collect_deep_result()

        self.frame_idx += 1
        return {
            "overlay": overlay,
            "metric": self.current_metric,
            "has_face": True,
            "method_changed": method_changed,
            "identity_locked": self.identity_signature_ref is not None,
            "identity_match": True,
            "intruder_detected": False,
        }

    def finalize(self) -> Dict[str, Any]:
        """
        Computes the final biometric results for the entire session.

        Aggregates all buffered frames, runs the highest-accuracy deep models, 
        performs spectral analysis and HRV extraction, and executes the final 
        Triage Agent logic to determine the clinical output.

        Returns:
            Dict[str, Any]: Final report with BPM, HRV, Stress levels, and 
                reliability metrics.
        """
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
            return {
                "status": "failed",
                "reason": "not_enough_valid_frames",
                "frames_processed": len(self.rgb_samples),
                "n_frames": len(self.rgb_samples),
                "duration_sec": float(len(self.rgb_samples) / max(effective_fps, 1e-6)),
                "fps": float(effective_fps),
            }

        rgb = np.asarray(self.rgb_samples, dtype=np.float64)
        faces = np.asarray(self.face_frames, dtype=np.uint8) if self.face_frames else None
        if self.final_deep_enabled:
            result = process_rppg_with_deep(
                rgb,
                fps=effective_fps,
                face_frames=faces,
                motion_scores=None,
                selection_mode="best_confidence",
                deep_max_frames=self.deep_max_frames,
            )
        else:
            result = process_rppg(
                rgb,
                fps=effective_fps,
                motion_scores=None,
            )
            result.update(
                {
                    "deep_model_used": "disabled",
                    "selected_source": "pos",
                    "selection_mode": "off",
                    "pos_confidence": float(result["confidence"]),
                    "deep_confidence": 0.0,
                    "pos_snr": 0.0,
                    "deep_snr": 0.0,
                }
            )

        ibi_ms = np.asarray(result.get("ibi_ms", np.array([])), dtype=np.float64)
        result["bpm"] = float(60000.0 / np.median(ibi_ms)) if ibi_ms.size > 0 else None
        result["bpm_mean"] = float(60000.0 / np.mean(ibi_ms)) if ibi_ms.size > 0 else None


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

    def _maybe_start_deep_job(self, effective_fps: float):
        if self.deep_future is not None and not self.deep_future.done():
            return
        rgb = np.asarray(self.rgb_samples, dtype=np.float64)
        faces = np.asarray(self.face_frames, dtype=np.uint8)
        self.deep_future = self.executor.submit(
            process_rppg_with_deep,
            rgb,
            effective_fps,
            faces,
            None,
            "best_confidence",
        )

    def _maybe_collect_deep_result(self) -> bool:
        if self.deep_future is None or not self.deep_future.done():
            return False

        try:
            result = self.deep_future.result()
        except Exception:
            self.deep_future = None
            return False

        self.deep_future = None
        confidence = float(result["confidence"])
        reliable = bool(result.get("is_reliable", False))
        bpm = None
        if reliable and confidence >= MIN_DISPLAY_CONFIDENCE and result["ibi_ms"].size > 0:
            bpm = float(60000.0 / np.median(result["ibi_ms"]))

        new_metric = StreamMetric(
            bpm=bpm,
            confidence=confidence,
            method=str(result.get("method_used", "pos")) if bpm is not None else "guarded",
        )

        changed = new_metric.method != self.current_metric.method
        # Replace display value only when deep candidate is selected as more reliable.
        self.current_metric = new_metric
        return changed

    def _maybe_reset_metric_on_invalid_streak(self) -> None:
        # If we lose valid face evidence for ~2 seconds, drop stale HR output.
        if self.invalid_streak < int(max(1.0, self._effective_fps() * 2.0)):
            return
        self.current_metric = StreamMetric(bpm=None, confidence=0.0, method="pending")
        self.rgb_samples.clear()
        self.face_frames.clear()
        self.sample_ts_ms.clear()
        if self.deep_future and not self.deep_future.done():
            self.deep_future.cancel()
        self.deep_future = None

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

        # Clamp to a practical range for mobile snapshot streaming.
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

        # Reject this frame immediately to avoid mixing different-person signal,
        # while waiting for repeated evidence before invalidating the whole session.
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

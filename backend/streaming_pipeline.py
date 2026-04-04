from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from roi_pipeline import (
    MIN_ROI_PIXELS,
    FaceROIExtractor,
    compute_mad_confidence,
    get_mean_rgb,
    overlay_roi,
)
from rppg_core import process_rppg, process_rppg_with_deep
from triage_agent import TriageAgent


MIN_DISPLAY_CONFIDENCE = 0.55


@dataclass
class StreamMetric:
    bpm: Optional[float]
    confidence: float
    method: str


class RealtimeRPPGPipeline:
    """Incremental rPPG pipeline for webcam/mobile frame streams.

    Pipeline notes for the upcoming mobile integration:
    1) React Native Expo app captures frames and sends them to backend.
    2) Backend receives frame stream over websocket.
    3) Backend applies ROI overlay (face mask) in near real time.
    4) Backend returns overlay frames back to mobile for display.
    5) POS runs first (fast), then deep_rPPG runs async and can replace POS if better.
    """

    def __init__(
        self,
        model_path: str = "face_landmarker.task",
        fps: float = 30.0,
        pos_min_frames: int = 150,
        deep_min_frames: int = 240,
        update_stride: int = 30,
        live_deep_enabled: bool = False,
        final_deep_enabled: bool = True,
        live_pos_window_sec: float = 12.0,
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
        self.update_interval_seconds = max(0.5, self.update_stride / max(self.fps, 1e-6))

        self.roi_extractor = FaceROIExtractor(model_path)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.deep_future: Optional[Future] = None

        self.rgb_samples: List[List[float]] = []
        self.face_frames: List[np.ndarray] = []
        self.sample_ts_ms: List[int] = []
        self.frame_idx = 0
        self.invalid_streak = 0

        self.current_metric = StreamMetric(bpm=None, confidence=0.0, method="pending")
        self.triage_agent = TriageAgent()

    def close(self):
        if self.deep_future and not self.deep_future.done():
            self.deep_future.cancel()
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.roi_extractor.close()

    def reset(self):
        self.rgb_samples.clear()
        self.face_frames.clear()
        self.sample_ts_ms.clear()
        self.frame_idx = 0
        self.invalid_streak = 0
        self.current_metric = StreamMetric(bpm=None, confidence=0.0, method="pending")
        if self.deep_future and not self.deep_future.done():
            self.deep_future.cancel()
        self.deep_future = None

    def ingest(self, frame_bgr: np.ndarray, ts_ms: int) -> Dict[str, Any]:
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
            }

        r, g, b = get_mean_rgb(frame_bgr, roi_res.masks["face"])
        _ = compute_mad_confidence({"face": g})
        self.rgb_samples.append([r, g, b])
        self.face_frames.append(roi_res.crops["face"])
        self.sample_ts_ms.append(int(ts_ms))
        self.invalid_streak = 0

        method_changed = False
        n = len(self.rgb_samples)
        effective_fps = self._effective_fps()
        required_pos_frames = max(20, int(round(self.pos_min_seconds * effective_fps)))
        required_deep_frames = max(30, int(round(self.deep_min_seconds * effective_fps)))
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
        }

    def finalize(self) -> Dict[str, Any]:
        effective_fps = self._effective_fps()
        min_required = max(24, int(effective_fps * 6.0))
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

        decision = self.triage_agent.decide(result, face_frames=faces)
        result["triage_mode"] = decision.mode
        result["triage_reason"] = decision.reason
        result["visual_stress"] = float(decision.visual_stress_score)
        result["status"] = "success"
        result["frames_processed"] = int(len(self.rgb_samples))
        result["fps"] = float(effective_fps)
        return result

    def _update_pos_metric(self, effective_fps: float):
        window_frames = max(24, int(round(effective_fps * self.live_pos_window_sec)))
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

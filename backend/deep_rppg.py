"""
deep_rppg.py
Deep neural rPPG extraction using a single pretrained open-rppg model.
Uses PhysFormer (highest accuracy option in this project setup).

Input:  face_frames — list/array of BGR face crop frames, shape (N, H, W, 3)
        fps — frames per second
Output: 1D BVP/pulse signal array of length N
"""

import numpy as np
import tempfile
import os
import cv2
from threading import Lock

_MODEL_NAME = "PhysFormer.pure"

_model_cache = {}  # Cache loaded models so we don't reload every call
_model_lock = Lock()
_rppg_module = None
_rppg_import_failed = False
_resample_fn = None
_video_fourcc = None


def _load_model(model_name: str):
    if model_name in _model_cache:
        return _model_cache[model_name]

    with _model_lock:
        if model_name not in _model_cache:
            rppg = _get_rppg_module()
            if rppg is None:
                raise ImportError("open-rppg is not installed")

            print(f"[deep_rppg] Loading {model_name}...")
            _model_cache[model_name] = rppg.Model(model_name)
            print(f"[deep_rppg] ✅ {model_name} loaded")
    return _model_cache[model_name]


def _get_rppg_module():
    global _rppg_module, _rppg_import_failed
    if _rppg_module is not None:
        return _rppg_module
    if _rppg_import_failed:
        return None
    try:
        import rppg

        _rppg_module = rppg
        return _rppg_module
    except ImportError:
        _rppg_import_failed = True
        return None


def _get_resample_fn():
    global _resample_fn
    if _resample_fn is None:
        from scipy.signal import resample

        _resample_fn = resample
    return _resample_fn


def _extract_bvp_from_result(result, model):
    bvp = None
    if isinstance(result, dict):
        bvp = result.get("bvp")
        if bvp is None:
            bvp = result.get("signal")
        if bvp is None:
            bvp = result.get("ppg")
    elif isinstance(result, np.ndarray):
        bvp = result

    if bvp is None and hasattr(model, "bvp"):
        bvp_out = model.bvp()
        bvp = bvp_out[0] if isinstance(bvp_out, tuple) and len(bvp_out) >= 1 else bvp_out

    return bvp


def frames_to_temp_video(frames_bgr: np.ndarray, fps: float) -> str:
    """Write face crop frames to a temporary video for open-rppg."""
    global _video_fourcc
    fd, tmp_path = tempfile.mkstemp(suffix=".avi")
    os.close(fd)

    h, w = frames_bgr[0].shape[:2]

    writer = None
    codes_to_try = []
    if _video_fourcc is not None:
        codes_to_try.append(_video_fourcc)
    codes_to_try.extend(["MJPG", "mp4v"])

    for code in codes_to_try:
        writer = cv2.VideoWriter(
            tmp_path,
            cv2.VideoWriter.fourcc(*code),
            fps,
            (w, h),
        )
        if writer.isOpened():
            _video_fourcc = code
            break

    if writer is None or not writer.isOpened():
        raise RuntimeError("Unable to open temporary video writer for deep rPPG")

    for frame in frames_bgr:
        writer.write(frame)
    writer.release()
    return tmp_path


def extract_bvp_deep(
    face_frames: np.ndarray,
    fps: float,
    model_name: str = "auto",
    max_frames: int | None = 300,
) -> tuple[np.ndarray, str]:
    """Extract BVP from face frames using PhysFormer only."""
    if face_frames is None or len(face_frames) == 0:
        return np.array([], dtype=np.float64), "none"

    chosen_model = _MODEL_NAME if model_name == "auto" else model_name
    if chosen_model != _MODEL_NAME:
        print(
            f"[deep_rppg] ⚠️ Requested '{chosen_model}', forcing '{_MODEL_NAME}' for accuracy"
        )
        chosen_model = _MODEL_NAME

    tmp_path = None
    original_frames = int(len(face_frames))
    frames_for_model = face_frames
    fps_for_model = float(fps)

    if max_frames is not None and original_frames > int(max_frames):
        target_frames = max(60, int(max_frames))
        idx = np.linspace(0, original_frames - 1, target_frames, dtype=np.int32)
        frames_for_model = face_frames[idx]
        fps_for_model = max(1.0, float(fps) * (target_frames / original_frames))

    if not isinstance(frames_for_model, np.ndarray):
        frames_for_model = np.asarray(frames_for_model)
    if frames_for_model.dtype != np.uint8:
        frames_for_model = frames_for_model.astype(np.uint8, copy=False)
    if not frames_for_model.flags["C_CONTIGUOUS"]:
        frames_for_model = np.ascontiguousarray(frames_for_model)

    try:
        model = _load_model(chosen_model)
        tmp_path = frames_to_temp_video(frames_for_model, fps_for_model)
        result = model.process_video(tmp_path)

        bvp = _extract_bvp_from_result(result, model)

        if bvp is None or len(bvp) <= 10:
            raise RuntimeError("Model returned empty/short BVP")

        if len(bvp) != original_frames:
            bvp = _get_resample_fn()(bvp, original_frames)

        bvp = np.asarray(bvp, dtype=np.float64)
        print(f"[deep_rppg] ✅ {chosen_model} → BVP extracted ({len(bvp)} samples)")
        return bvp, chosen_model
    except Exception as e:
        print(f"[deep_rppg] ❌ {chosen_model} failed: {e}")
        return np.zeros(original_frames, dtype=np.float64), "none"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def is_deep_model_available() -> bool:
    """Quick check if the open-rppg package is installed."""
    return _get_rppg_module() is not None

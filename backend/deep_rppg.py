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

_MODEL_NAME = "PhysFormer.pure"

_model_cache = {}  # Cache loaded models so we don't reload every call


def _load_model(model_name: str):
    if model_name not in _model_cache:
        import rppg
        print(f"[deep_rppg] Loading {model_name}...")
        _model_cache[model_name] = rppg.Model(model_name)
        print(f"[deep_rppg] ✅ {model_name} loaded")
    return _model_cache[model_name]


def frames_to_temp_video(frames_bgr: np.ndarray, fps: float) -> str:
    """Write face crop frames to a temporary MP4 for open-rppg."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp.name
    tmp.close()

    h, w = frames_bgr[0].shape[:2]
    writer = cv2.VideoWriter(
        tmp_path,
        cv2.VideoWriter.fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    for frame in frames_bgr:
        writer.write(frame)
    writer.release()
    return tmp_path


def extract_bvp_deep(
    face_frames: np.ndarray,
    fps: float,
    model_name: str = "auto",
) -> tuple[np.ndarray, str]:
    """Extract BVP from face frames using PhysFormer only."""
    chosen_model = _MODEL_NAME if model_name == "auto" else model_name
    if chosen_model != _MODEL_NAME:
        print(
            f"[deep_rppg] ⚠️ Requested '{chosen_model}', forcing '{_MODEL_NAME}' for accuracy"
        )
        chosen_model = _MODEL_NAME

    tmp_path = None

    try:
        model = _load_model(chosen_model)
        tmp_path = frames_to_temp_video(face_frames, fps)
        result = model.process_video(tmp_path)

        bvp = None
        if isinstance(result, dict):
            bvp = result.get("bvp") or result.get("signal") or result.get("ppg")
        elif isinstance(result, np.ndarray):
            bvp = result

        if bvp is None and hasattr(model, "bvp"):
            bvp_out = model.bvp()
            bvp = bvp_out[0] if isinstance(bvp_out, tuple) and len(bvp_out) >= 1 else bvp_out

        if bvp is None or len(bvp) <= 10:
            raise RuntimeError("Model returned empty/short BVP")

        if len(bvp) != len(face_frames):
            from scipy.signal import resample

            bvp = resample(bvp, len(face_frames))

        bvp = np.asarray(bvp, dtype=np.float64)
        print(f"[deep_rppg] ✅ {chosen_model} → BVP extracted ({len(bvp)} samples)")
        return bvp, chosen_model
    except Exception as e:
        print(f"[deep_rppg] ❌ {chosen_model} failed: {e}")
        return np.zeros(len(face_frames)), "none"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def is_deep_model_available() -> bool:
    """Quick check if the open-rppg package is installed."""
    try:
        import rppg  # noqa: F401
        return True
    except ImportError:
        return False

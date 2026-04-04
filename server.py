# server.py
# Flask API server — bridges the React Native app to the rPPG Python pipeline
# Run: python server.py
# The React Native app posts video to /process and gets back JSON results.

import os
import json
import tempfile
import traceback
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import cv2

# Import your existing pipeline
from main import FaceROIExtractor, get_mean_rgb, ROI_WEIGHTS
from rppg_core import process_rppg_with_deep  # or process_rppg if no deep model

app = Flask(__name__)
CORS(app)  # Allow React Native to hit this API

UPLOAD_DIR = Path("./uploads")
OUTPUT_DIR = Path("./output")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = "face_landmarker.task"


# ─── HEALTH CHECK ──────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "pipeline": "CardioVision rPPG v3"})


# ─── PROCESS VIDEO ─────────────────────────────────────────────────────────────
@app.route("/process", methods=["POST"])
def process():
    """
    Accepts a multipart/form-data POST with a 'video' file.
    Runs the full rPPG pipeline and returns JSON results.
    """
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files["video"]

    # Save uploaded video to temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", dir=UPLOAD_DIR, delete=False)
    video_path = tmp.name
    tmp.close()
    video_file.save(video_path)

    try:
        result = _run_pipeline(video_path)

        # Build JSON-serializable output
        output = _serialize_result(result)
        return jsonify(output)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "status": "failed"}), 500
    finally:
        try:
            os.unlink(video_path)
        except Exception:
            pass


def _run_pipeline(video_path: str) -> dict:
    """Run the full rPPG pipeline on a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video file")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    roi_ext = FaceROIExtractor(MODEL_PATH, 0.5, 0.5)
    rgb_data    = []
    face_frames = []
    frame_idx   = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ts_ms = int((frame_idx / fps) * 1000)
        res = roi_ext.process(frame, ts_ms)

        if res is not None:
            if res.face_mask is not None:
                frame = frame.copy()
                frame[res.face_mask == 0] = 0

            rgb = [0.0, 0.0, 0.0]
            for r, w in ROI_WEIGHTS.items():
                m_rgb = get_mean_rgb(frame, res.masks[r])
                if not np.isnan(m_rgb[1]):
                    rgb = [rgb[i] + w * m_rgb[i] for i in range(3)]
            rgb_data.append(rgb)

            if "face" in res.crops:
                face_frames.append(res.crops["face"])

        frame_idx += 1

    cap.release()
    roi_ext.close()

    if len(rgb_data) < 50:
        raise RuntimeError(f"Only {len(rgb_data)} usable frames — check video quality")

    rgb_arr    = np.array(rgb_data, dtype=np.float64)
    frames_arr = np.array(face_frames, dtype=np.uint8) if face_frames else None
    actual_fps = fps

    # Run signal processing (try deep + POS, fallback to POS only)
    try:
        result = process_rppg_with_deep(rgb_arr, fps=actual_fps, face_frames=frames_arr)
    except Exception as e:
        print(f"Deep model failed, using POS only: {e}")
        from rppg_core import process_rppg
        result = process_rppg(rgb_arr, fps=actual_fps)

    # Add BPM
    ibi_ms = result.get("ibi_ms", np.array([]))
    result["bpm"]      = float(60000.0 / np.median(ibi_ms)) if len(ibi_ms) > 0 else None
    result["bpm_mean"] = float(60000.0 / np.mean(ibi_ms))   if len(ibi_ms) > 0 else None

    # Triage info
    try:
        from triage_agent import TriageAgent
        agent    = TriageAgent()
        decision = agent.decide(result, face_frames=frames_arr)
        result["triage_mode"]   = decision.mode
        result["triage_reason"] = decision.reason
        result["visual_stress"] = float(decision.visual_stress_score)
    except Exception as e:
        print(f"Triage agent error: {e}")
        result["triage_mode"]   = "BIOMETRIC"
        result["triage_reason"] = "Triage agent unavailable"
        result["visual_stress"] = 0.0

    result["status"]           = "success"
    result["frames_processed"] = len(rgb_data)
    return result


def _serialize_result(result: dict) -> dict:
    """Convert numpy arrays to JSON-serializable Python types."""
    out = {}
    for k, v in result.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        elif isinstance(v, dict):
            out[k] = _serialize_result(v)
        elif isinstance(v, bool):
            out[k] = bool(v)
        else:
            out[k] = v
    return out


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("╔══════════════════════════════════════╗")
    print("║  CardioVision API Server v3          ║")
    print("║  POST /process   — upload video      ║")
    print("║  GET  /health    — check status      ║")
    print("╚══════════════════════════════════════╝")
    print()
    print("Find your IP: ifconfig | grep 'inet '")
    print("Update BASE_URL in src/api/rppgService.ts")
    print()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

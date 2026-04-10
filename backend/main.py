"""
The entry point for the CardioVision-rPPG streaming server.

This module provides the command-line interface to launch a WebSocket server
for real-time remote processing of physiological signals.
"""
import argparse
import asyncio
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from stream_server import run_server
from streaming_pipeline import RealtimeRPPGPipeline
from evm_magnifier import EulerianMagnifier, render_pulse_view

DEFAULT_MODEL_PATH = str(Path(__file__).with_name("face_landmarker.task"))


def main():
    """
    Main entry point for the tool.
    Starts the WebSocket server for real-time remote processing.
    """
    p = argparse.ArgumentParser(
        description="CardioVision rPPG Pipeline v3 (Stream Server)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--fps", type=float, default=60.0)
    p.add_argument("--ws-host", default="0.0.0.0")
    p.add_argument("--ws-port", type=int, default=8765)
    p.add_argument("--model", default=DEFAULT_MODEL_PATH, dest="model_path")
    p.add_argument("--jpeg-quality", type=int, default=45)
    p.add_argument("--overlay-max-side", type=int, default=320)
    p.add_argument("--overlay-stride", type=int, default=2)
    p.add_argument(
        "--mode",
        choices=["batch", "stream", "preview"],
        default="stream",
        help="stream: Start websocket server, preview: Start local webcam window.",
    )
    p.add_argument("--preview", action="store_true", help="Shortcut for --mode preview")

    args = p.parse_args()

    if args.mode == "preview" or args.preview:
        run_local_preview(args.model_path)
    else:
        asyncio.run(
            run_server(
                host=args.ws_host,
                port=args.ws_port,
                model_path=args.model_path,
            )
        )


def _probe_camera_fps(cap: cv2.VideoCapture, n_frames: int = 40) -> float:
    """
    Reads n_frames from the camera and returns the measured frame rate.
    This is important because the EVM filter must be designed at the actual
    capture fps — if it's wrong, the bandpass cutoffs will be off.
    """
    print(f"   Probing camera FPS ({n_frames} frames)...", end="", flush=True)
    t0 = time.perf_counter()
    for _ in range(n_frames):
        cap.read()
    elapsed = time.perf_counter() - t0
    fps = n_frames / elapsed
    print(f" {fps:.1f} fps")
    return fps


def run_local_preview(model_path: str):
    """
    Dual-window live preview:
      Left  – ROI overlay from the rPPG pipeline (face landmarks, BPM)
      Right – Eulerian Video Magnification pulse view (heartbeat as skin shimmer)

    Architecture
    ────────────
    The two pipelines run on SEPARATE threads so MediaPipe's latency (~100ms/
    frame) does NOT throttle the EVM path:

    • Main thread : camera read → EVM → display (30+ fps)
    • ROI thread  : camera frame copy → MediaPipe + rPPG → cached overlay

    This means EVM always gets frames at the true camera rate, which is
    critical for the temporal bandpass filter to work correctly.
    """
    print(f"📸 Starting local preview using webcam... (Model: {model_path})")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    # ── Step 1: probe actual camera fps ──────────────────────────────────────
    actual_fps = _probe_camera_fps(cap)
    # Clamp to a sane range (old cameras may report garbage)
    actual_fps = float(np.clip(actual_fps, 10.0, 120.0))

    # ── Step 2: EVM magnifier at TRUE camera fps ──────────────────────────────
    # The filter cutoffs are in Hz, and scipy designs them relative to the
    # sample rate (fps). Wrong fps → wrong cutoffs → wrong result.
    magnifier = EulerianMagnifier(
        fps=actual_fps,
        lo_hz=0.75,        # 45 BPM  (safe low bound for resting HR)
        hi_hz=2.5,         # 150 BPM (safe high bound)
        alpha=50.0,        # amplification α (paper uses 50 for colour EVM)
        attenuation=1.0,   # chrominance attenuation A
        levels=6,          # more levels = more spatial blur = cleaner signal
    )
    print(f"   EVM filter designed for {actual_fps:.1f} fps  "
          f"(bandpass: 0.75–2.5 Hz = 45–150 BPM)")

    # ── Step 3: ROI pipeline on a background thread ───────────────────────────
    # Shared state between threads — protected by a lock.
    _roi_state: dict = {
        "overlay": None,     # last ROI-annotated frame (or None)
        "bpm": None,
        "method": "pending",
    }
    _roi_lock = threading.Lock()
    _latest_frame: dict = {"frame": None, "ts_ms": 0}
    _latest_lock = threading.Lock()
    _stop = threading.Event()

    def roi_worker():
        pipeline = RealtimeRPPGPipeline(model_path=model_path)
        last_ts_seen = -1
        try:
            while not _stop.is_set():
                with _latest_lock:
                    f = _latest_frame["frame"]
                    ts = _latest_frame["ts_ms"]

                # Skip if this is the same frame we already processed —
                # MediaPipe VIDEO mode requires strictly increasing timestamps.
                if f is None or ts <= last_ts_seen:
                    time.sleep(0.005)
                    continue

                last_ts_seen = ts
                res = pipeline.ingest(f, ts)
                ov = res["overlay"].copy()
                m = res["metric"]

                # Annotate the cached overlay
                bpm_text = f"BPM: {m.bpm:.1f}" if m.bpm else "BPM: --"
                cv2.putText(ov, bpm_text, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(ov, f"Method: {m.method}", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                cv2.putText(ov, "ROI PREVIEW", (20, ov.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

                with _roi_lock:
                    _roi_state["overlay"] = ov
                    _roi_state["bpm"] = m.bpm
                    _roi_state["method"] = m.method
        finally:
            pipeline.close()

    roi_thread = threading.Thread(target=roi_worker, daemon=True)
    roi_thread.start()

    print("✅ Preview active. Press 'q' to quit.")
    print("   Left  → ROI overlay (MediaPipe landmarks + BPM)")
    print("   Right → Pulse magnifier (EVM heartbeat visualisation)")

    # ── Step 4: Main display loop at full camera fps ──────────────────────────
    fps_samples: list[float] = []
    last_ts = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            ts_ms = int(time.time() * 1000)

            # Push latest raw frame for the ROI thread to consume
            with _latest_lock:
                _latest_frame["frame"] = frame.copy()
                _latest_frame["ts_ms"] = ts_ms

            # ── EVM: runs on EVERY frame at true camera fps ───────────────────
            magnified = magnifier.process(frame)
            warmup_done = magnifier._warmup_frames >= int(actual_fps * 1.5)

            # Read cached ROI overlay (updated asynchronously by ROI thread)
            with _roi_lock:
                roi_vis = _roi_state["overlay"]
                current_bpm = _roi_state["bpm"]

            # Fall back to raw frame if ROI not ready yet
            if roi_vis is None:
                roi_vis = frame.copy()
                cv2.putText(roi_vis, "Detecting face...", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 0), 2)

            evm_vis = render_pulse_view(magnified, current_bpm, warmup_done)

            # ── Composite side-by-side ────────────────────────────────────────
            h1, w1 = roi_vis.shape[:2]
            h2, w2 = evm_vis.shape[:2]
            if h1 != h2:
                evm_vis = cv2.resize(evm_vis, (int(w2 * h1 / h2), h1))

            combined = np.hstack([roi_vis, evm_vis])

            # Live fps counter
            now = time.perf_counter()
            dt = now - last_ts
            last_ts = now
            if dt > 0:
                fps_samples.append(1.0 / dt)
                fps_samples = fps_samples[-30:]
            live_fps = np.mean(fps_samples) if fps_samples else 0.0
            cv2.putText(combined, f"fps: {live_fps:.1f}",
                        (combined.shape[1] - 110, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)

            cv2.imshow("CardioVision Preview  [Q to quit]", combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        _stop.set()
        cap.release()
        cv2.destroyAllWindows()
        roi_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()

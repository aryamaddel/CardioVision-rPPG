"""
The entry point for the CardioVision-rPPG streaming server.

This module provides the command-line interface to launch a WebSocket server 
for real-time remote processing of physiological signals.
"""
import argparse
import asyncio
from pathlib import Path

import cv2
import time
import numpy as np

from stream_server import run_server
from streaming_pipeline import RealtimeRPPGPipeline

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
    # Maintain mode argument for backward compatibility
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

def run_local_preview(model_path: str):
    """
    Launches a local OpenCV window using the laptop webcam to visualize ROI tracking.
    """
    print(f"📸 Starting local preview using webcam... (Model: {model_path})")
    pipeline = RealtimeRPPGPipeline(model_path=model_path)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open laptop webcam.")
        return

    print("✅ Preview active. Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # mirror for natural feel
            frame = cv2.flip(frame, 1)
            ts_ms = int(time.time() * 1000)
            
            # Run through pipeline to get overlay
            res = pipeline.ingest(frame, ts_ms)
            vis = res["overlay"]
            
            # Overlay some data
            m = res["metric"]
            bpm_text = f"BPM: {m.bpm:.1f}" if m.bpm else "BPM: --"
            cv2.putText(vis, f"{bpm_text}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(vis, f"Method: {m.method}", (20, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            cv2.imshow("CardioVision ROI Preview", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pipeline.close()

if __name__ == "__main__":
    main()

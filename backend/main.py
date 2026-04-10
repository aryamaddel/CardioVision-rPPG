"""
The entry point for the CardioVision-rPPG streaming server.

This module provides the command-line interface to launch a WebSocket server 
for real-time remote processing of physiological signals.
"""
import argparse
import asyncio
from pathlib import Path

from stream_server import run_server

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
        "--live-deep-mode",
        choices=["off", "final-only", "live+final"],
        default="final-only",
        help="off: POS only, final-only: run deep model only at stop, live+final: run deep in live updates and final",
    )
    # Maintain mode argument for backward compatibility
    p.add_argument(
        "--mode",
        choices=["batch", "stream"],
        default="stream",
        help="Legacy argument, now forces stream mode.",
    )

    args = p.parse_args()
    
    asyncio.run(
        run_server(
            host=args.ws_host,
            port=args.ws_port,
            fps=args.fps,
            model_path=args.model_path,
            jpeg_quality=args.jpeg_quality,
            overlay_max_side=args.overlay_max_side,
            overlay_stride=args.overlay_stride,
            live_deep_mode=args.live_deep_mode,
        )
    )

if __name__ == "__main__":
    main()

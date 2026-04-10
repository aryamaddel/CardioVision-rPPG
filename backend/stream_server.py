"""
WebSocket server implementation for remote rPPG streaming.
"""
import asyncio
import base64
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import websockets

from streaming_pipeline import RealtimeRPPGPipeline

DEFAULT_MODEL_PATH = str(Path(__file__).with_name("face_landmarker.task"))

def _encode_jpeg(frame_bgr: np.ndarray, quality: int = 45, max_side: int = 0) -> Optional[str]:
    if max_side > 0:
        h, w = frame_bgr.shape[:2]
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            frame_bgr = cv2.resize(frame_bgr, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return base64.b64encode(buf).decode("ascii") if ok else None

async def _handle_client(websocket, pipeline: RealtimeRPPGPipeline, config: Dict[str, Any]):
    """Streamlined client handler."""
    frame_count = 0
    start_time = time.monotonic()

    async for message in websocket:
        # 1. Binary Frame Handling (Optimized Path)
        if isinstance(message, (bytes, bytearray)):
            frame = cv2.imdecode(np.frombuffer(message, np.uint8), cv2.IMREAD_COLOR)
            if frame is None: continue
            
            res = pipeline.ingest(frame, int(time.time() * 1000))
            await websocket.send(json.dumps({
                "type": "frame_result",
                "bpm": res["metric"].bpm,
                "confidence": res["metric"].confidence,
                "has_face": res["has_face"],
                "identity_match": res.get("identity_match", True),
                "overlay": _encode_jpeg(res["overlay"], config["quality"], config["max_side"]) if frame_count % config["stride"] == 0 else None
            }))
            frame_count += 1
            continue

        # 2. JSON Message Handling
        try:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "start":
                config.update({
                    "quality": data.get("overlay_quality", config["quality"]),
                    "max_side": data.get("overlay_max_side", config["max_side"]),
                    "stride": data.get("overlay_stride", config["stride"]),
                })
                await websocket.send(json.dumps({"type": "ack", "status": "ready"}))
            
            elif msg_type == "stop":
                result = pipeline.finalize()
                await websocket.send(json.dumps({"type": "final_result", "result": _to_json(result)}))
                pipeline.reset()
            
            elif msg_type == "ping":
                await websocket.send(json.dumps({"type": "pong"}))

        except Exception as e:
            await websocket.send(json.dumps({"type": "error", "message": str(e)}))

    print(f"[stream] Session ended. Processed {frame_count} frames in {time.monotonic()-start_time:.1f}s")

def _to_json(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray, np.generic)): return obj.tolist()
    if isinstance(obj, dict): return {k: _to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_to_json(x) for x in obj]
    return obj

async def run_server(host, port, **kwargs):
    async def handler(ws):
        pipeline = RealtimeRPPGPipeline(
            model_path=kwargs.get("model_path", DEFAULT_MODEL_PATH)
        )
        config = {
            "quality": kwargs.get("jpeg_quality", 45),
            "max_side": kwargs.get("overlay_max_side", 320),
            "stride": kwargs.get("overlay_stride", 2)
        }
        try:
            await _handle_client(ws, pipeline, config)
        finally:
            pipeline.close()

    async with websockets.serve(handler, host, port):
        print(f"🚀 CardioVision stream server running on ws://{host}:{port}")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(run_server("0.0.0.0", 8765))

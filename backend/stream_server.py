import argparse
import asyncio
import base64
import json
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from streaming_pipeline import RealtimeRPPGPipeline

DEFAULT_MODEL_PATH = str(Path(__file__).with_name("face_landmarker.task"))

try:
    import websockets  # type: ignore[import-not-found]
except ImportError as exc:
    raise ImportError(
        "websockets is required for stream mode. Install with: pip install websockets"
    ) from exc


def _encode_jpeg(frame_bgr: np.ndarray, quality: int = 80) -> Optional[str]:
    ok, buf = cv2.imencode(
        ".jpg",
        frame_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(quality, 1, 100))],
    )
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


async def _handle_client(websocket, pipeline: RealtimeRPPGPipeline, jpeg_quality: int):
    async for payload in websocket:
        if isinstance(payload, (bytes, bytearray)):
            frame = _decode_binary_frame(payload)
            if frame is None:
                continue
            await _send_frame_result(websocket, pipeline, frame, jpeg_quality)
            continue

        if not isinstance(payload, str):
            continue

        msg = _parse_json_message(payload)
        msg_type = msg.get("type")

        if msg_type == "start":
            await websocket.send(json.dumps({"type": "ack", "status": "ready"}))
            continue

        if msg_type == "frame":
            frame_b64 = msg.get("frame_jpeg_b64")
            if not isinstance(frame_b64, str):
                await websocket.send(json.dumps({"type": "error", "message": "missing frame_jpeg_b64"}))
                continue
            frame = _decode_b64_frame(frame_b64)
            if frame is None:
                await websocket.send(json.dumps({"type": "error", "message": "invalid frame payload"}))
                continue
            client_ts_ms = msg.get("ts_ms")
            ts_ms_override = int(client_ts_ms) if isinstance(client_ts_ms, (int, float)) else None
            await _send_frame_result(websocket, pipeline, frame, jpeg_quality, ts_ms_override)
            continue

        if msg_type == "stop":
            final_result = pipeline.finalize()
            await websocket.send(
                json.dumps(
                    {
                        "type": "final_result",
                        "result": _json_ready(final_result),
                    }
                )
            )
            pipeline.reset()
            continue

        if msg_type == "ping":
            await websocket.send(json.dumps({"type": "pong"}))
            continue

        await websocket.send(json.dumps({"type": "error", "message": f"unsupported message type: {msg_type}"}))


def _parse_json_message(payload: str) -> Dict[str, Any]:
    try:
        msg = json.loads(payload)
        if isinstance(msg, dict):
            return msg
    except Exception:
        pass
    return {}


def _decode_binary_frame(payload: bytes | bytearray) -> Optional[np.ndarray]:
    frame_arr = np.frombuffer(payload, dtype=np.uint8)
    return cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)


def _decode_b64_frame(frame_b64: str) -> Optional[np.ndarray]:
    try:
        raw = base64.b64decode(frame_b64)
    except Exception:
        return None
    return _decode_binary_frame(raw)


async def _send_frame_result(
    websocket,
    pipeline: RealtimeRPPGPipeline,
    frame: np.ndarray,
    jpeg_quality: int,
    ts_ms_override: Optional[int] = None,
):
    ts_ms = int(ts_ms_override) if ts_ms_override is not None else int(asyncio.get_running_loop().time() * 1000)
    out = pipeline.ingest(frame, ts_ms)
    jpeg_b64 = _encode_jpeg(out["overlay"], quality=jpeg_quality)

    response = {
        "type": "frame_result",
        "metric": {
            "bpm": out["metric"].bpm,
            "confidence": out["metric"].confidence,
            "method": out["metric"].method,
        },
        "has_face": out["has_face"],
        "method_changed": out["method_changed"],
        "overlay_jpeg_b64": jpeg_b64,
    }
    await websocket.send(json.dumps(response))


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


async def run_server(host: str, port: int, fps: float, model_path: str, jpeg_quality: int):
    async def handler(websocket):
        pipeline = RealtimeRPPGPipeline(model_path=model_path, fps=fps)
        try:
            await _handle_client(websocket, pipeline, jpeg_quality)
        finally:
            pipeline.close()

    async with websockets.serve(handler, host, port, max_size=4 * 1024 * 1024):
        print(f"WebSocket server listening on ws://{host}:{port}")
        await asyncio.Future()


def main():
    parser = argparse.ArgumentParser(description="CardioVision streaming backend")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, dest="model_path")
    parser.add_argument("--jpeg-quality", type=int, default=75)
    args = parser.parse_args()

    try:
        asyncio.run(
            run_server(
                host=args.host,
                port=args.port,
                fps=args.fps,
                model_path=args.model_path,
                jpeg_quality=args.jpeg_quality,
            )
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

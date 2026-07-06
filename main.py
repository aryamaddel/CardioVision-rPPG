"""
CLI entry point for rPPG research pipeline.

Usage:
  python main.py                     # webcam (device 0)
  python main.py --source 1          # webcam (device 1)
  python main.py --source video.mp4  # video file
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from rppg import FaceTracker, process_video
from evm_magnifier import PulseMagnifier

DEFAULT_MODEL = str(Path(__file__).with_name("face_landmarker.task"))


def _parse_source(value: str):
    try:
        return int(value)
    except ValueError:
        return value


def main():
    p = argparse.ArgumentParser(description="rPPG from camera or video")
    p.add_argument("--source", default="0", help="Camera index or video file path")
    p.add_argument("--model", default=DEFAULT_MODEL, dest="model_path")
    p.add_argument("--window", type=float, default=6.0, help="Seconds of data for each BPM estimate")
    args = p.parse_args()

    source = _parse_source(args.source)
    is_camera = isinstance(source, int)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: cannot open {source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if is_camera:
        fps = 30.0
    min_frames = int(fps * args.window)

    tracker = FaceTracker(args.model_path)
    magnifier = PulseMagnifier(fps=fps)

    rgb_buffer = []
    result = {"bpm": None}
    print(f"Source: {source}  |  Window: {args.window}s ({min_frames} frames @ {fps:.1f} fps)")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if is_camera:
            frame = cv2.flip(frame, 1)

        ts_ms = int(time.time() * 1000)

        rgb, roi_vis = tracker.process(frame, ts_ms)
        if rgb is not None:
            rgb_buffer.append(rgb)
            if len(rgb_buffer) >= min_frames:
                arr = np.array(rgb_buffer[-min_frames:])
                result = process_video(arr, fps)
                bpm = result["bpm"]
                if bpm:
                    print(f"\rBPM: {bpm:.1f}  ({len(rgb_buffer)} frames)", end="")

        bpm_text = "BPM: --"
        if len(rgb_buffer) >= min_frames and result.get("bpm"):
            bpm_text = f"BPM: {result['bpm']:.1f}"
        cv2.putText(roi_vis, bpm_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(roi_vis, f"Frames: {len(rgb_buffer)}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        evm_vis = magnifier.process(frame)
        if magnifier.warm:
            h, w = evm_vis.shape[:2]
            overlay_bgr = evm_vis[:56]
            evm_vis[:56] = (overlay_bgr * 0.4).astype(np.uint8)
            cv2.putText(evm_vis, "PULSE MAGNIFIER (EVM)", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1)
            cv2.putText(evm_vis, bpm_text, (10, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        h1, w1 = roi_vis.shape[:2]
        h2, w2 = evm_vis.shape[:2]
        if h1 != h2:
            evm_vis = cv2.resize(evm_vis, (int(w2 * h1 / h2), h1))
        combined = np.hstack([roi_vis, evm_vis])

        cv2.imshow("rPPG  [Q to quit]", combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()

    if len(rgb_buffer) >= min_frames:
        arr = np.array(rgb_buffer)
        result = process_video(arr, fps)
        bpm = result.get("bpm")
        print(f"\nFinal BPM: {bpm:.1f}" if bpm else "\nFinal: could not estimate BPM")
    else:
        print(f"\nNot enough frames captured ({len(rgb_buffer)} < {min_frames})")


if __name__ == "__main__":
    main()

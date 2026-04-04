import argparse
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple, Union, cast
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from tqdm import tqdm
from rppg_core import process_rppg_with_deep

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
DEFAULT_MODEL_PATH = str(Path(__file__).with_name("face_landmarker.task"))

# Exclusion indices for inverting the ROI selection methodology
_EXCLUDE_EYE_L = [33, 160, 158, 133, 153, 144]
_EXCLUDE_EYE_R = [362, 385, 387, 263, 373, 380]
_EXCLUDE_BROW_L = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
_EXCLUDE_BROW_R = [336, 296, 334, 293, 300, 285, 295, 282, 283, 276]
_EXCLUDE_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61]

ROI_COLORS = {"face": (0, 140, 255)}

MIN_ROI_PIXELS = 150
_ERODE_KERNEL = np.ones((5, 5), np.uint8)
MIN_FACE_AREA_RATIO = 0.08
MAX_FACE_AREA_RATIO = 0.75
MIN_SKIN_RATIO = 0.18
MIN_FACE_TEXTURE_VAR = 10.0
MAX_FACE_TEXTURE_VAR = 1800.0
MIN_TOTAL_GUARDRAIL_PASSES = 3
MIN_HSV_SKIN_RATIO = 0.10
MIN_SKIN_MASK_IOU = 0.35


@dataclass
class ROIResult:
    masks: Dict[str, np.ndarray]
    px_counts: Dict[str, int]
    face_bbox: Tuple[int, int, int, int]
    frame_idx: int
    crops: Dict[str, np.ndarray]
    face_mask: Optional[np.ndarray] = None


class VideoSource:
    def __init__(self, source: Union[int, str] = 0, target_fps: float = 30.0):
        self.source = source
        self.target_fps = target_fps
        self._is_webcam = isinstance(source, int)
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open: {source}")
        if self._is_webcam:
            self._cap.set(cv2.CAP_PROP_FPS, target_fps)
        self.orig_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        raw_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_frames = raw_count if raw_count > 0 and not self._is_webcam else None

    def frames(
        self, max_duration: Optional[float] = None
    ) -> Generator[Tuple[float, np.ndarray], None, None]:
        if self._is_webcam:
            count, interval, t0 = 0, 1.0 / self.target_fps, time.perf_counter()
            while True:
                ret, frame = self._cap.read()
                if not ret:
                    break
                elapsed = time.perf_counter() - t0
                if max_duration is not None and elapsed >= max_duration:
                    break
                yield elapsed, frame
                count += 1
                drift = (count * interval) - (time.perf_counter() - t0)
                if drift > 0:
                    time.sleep(drift)
        else:
            src_fps, out_count, src_count, next_yield = self.orig_fps, 0, 0, 0.0
            while True:
                ret, frame = self._cap.read()
                if not ret:
                    break
                elapsed = out_count / self.target_fps
                if max_duration is not None and elapsed >= max_duration:
                    break
                if src_count / src_fps >= next_yield:
                    yield elapsed, frame
                    next_yield += 1.0 / self.target_fps
                    out_count += 1
                src_count += 1

    def release(self):
        self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()


class FaceROIExtractor:
    def __init__(self, model_path: str):
        p = Path(model_path)
        if not p.exists():
            print("Downloading MediaPipe Face Landmarker model …")
            urllib.request.urlretrieve(MODEL_URL, str(p))
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(p)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = mp_vision.FaceLandmarker.create_from_options(opts)
        self.count = 0
        self.stats = {
            "detected": 0,
            "failed": 0,
            "low_quality": 0,
            "rejected_nonhuman": 0,
            "rej_area": 0,
            "rej_skin_ycrcb": 0,
            "rej_skin_hsv": 0,
            "rej_skin_iou": 0,
            "rej_geometry": 0,
            "rej_texture": 0,
        }

    def process(self, frame_bgr: np.ndarray, ts_ms: int) -> Optional[ROIResult]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self.detector.detect_for_video(mp_img, ts_ms)
        self.count += 1
        if not res.face_landmarks:
            self.stats["failed"] += 1
            return None
        self.stats["detected"] += 1
        h, w = frame_bgr.shape[:2]
        lm = np.array(
            [[lk.x * w, lk.y * h] for lk in res.face_landmarks[0]], dtype=np.float32
        )

        face_mask = np.zeros((h, w), dtype=np.uint8)
        hull = cv2.convexHull(lm.astype(np.int32))
        cv2.fillPoly(face_mask, [hull], 255)

        masks, px_counts = _build_masks(lm, h, w)

        bbox = (
            int(lm[:, 0].min()),
            int(lm[:, 1].min()),
            int(lm[:, 0].max() - lm[:, 0].min()),
            int(lm[:, 1].max() - lm[:, 1].min()),
        )

        passed, checks = _evaluate_human_face_guardrails(frame_bgr, face_mask, bbox, lm)
        if not passed:
            self.stats["rejected_nonhuman"] += 1
            self.stats["rej_area"] += 0 if checks["area"] else 1
            self.stats["rej_skin_ycrcb"] += 0 if checks["skin_ycrcb"] else 1
            self.stats["rej_skin_hsv"] += 0 if checks["skin_hsv"] else 1
            self.stats["rej_skin_iou"] += 0 if checks["skin_iou"] else 1
            self.stats["rej_geometry"] += 0 if checks["geometry"] else 1
            self.stats["rej_texture"] += 0 if checks["texture"] else 1
            self.stats["failed"] += 1
            self.stats["detected"] = max(0, self.stats["detected"] - 1)
            return None

        crops = {}
        for roi_name, m in masks.items():
            ys, xs = np.where(m > 0)
            if ys.size:
                y1, y2 = ys.min(), ys.max()
                x1, x2 = xs.min(), xs.max()
                roi_crop = frame_bgr[y1 : y2 + 1, x1 : x2 + 1]
                crops[roi_name] = cv2.resize(roi_crop, (64, 64))
            else:
                crops[roi_name] = np.zeros((64, 64, 3), dtype=np.uint8)

        return ROIResult(masks, px_counts, bbox, self.count - 1, crops, face_mask)

    def close(self):
        self.detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _build_masks(lm: np.ndarray, h: int, w: int, skin_m: Optional[np.ndarray] = None):
    base_mask = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(lm.astype(np.int32))
    cv2.fillPoly(base_mask, [hull], 255)

    for indices in [
        _EXCLUDE_EYE_L,
        _EXCLUDE_EYE_R,
        _EXCLUDE_BROW_L,
        _EXCLUDE_BROW_R,
        _EXCLUDE_LIPS,
    ]:
        pts = lm[indices].astype(np.int32)
        cv2.fillPoly(base_mask, [pts], 0)

    mask = cv2.erode(base_mask, _ERODE_KERNEL, iterations=1)
    if skin_m is not None:
        mask = cv2.bitwise_and(mask, skin_m)

    masks = {"face": mask}
    px_counts = {"face": int(np.count_nonzero(mask))}
    return masks, px_counts


def _evaluate_human_face_guardrails(
    frame_bgr: np.ndarray,
    face_mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    lm: np.ndarray,
) -> Tuple[bool, Dict[str, bool]]:
    h, w = frame_bgr.shape[:2]
    _, _, bw, bh = bbox
    face_area_ratio = float((bw * bh) / max(1, w * h))
    skin_ratio_ycrcb, skin_ratio_hsv, skin_iou = _estimate_skin_consistency(frame_bgr, face_mask)

    checks = {
        "area": MIN_FACE_AREA_RATIO <= face_area_ratio <= MAX_FACE_AREA_RATIO,
        "skin_ycrcb": skin_ratio_ycrcb >= MIN_SKIN_RATIO,
        "skin_hsv": skin_ratio_hsv >= MIN_HSV_SKIN_RATIO,
        "skin_iou": skin_iou >= MIN_SKIN_MASK_IOU,
        "geometry": _passes_face_geometry_guard(lm, bbox),
        "texture": _passes_texture_guard(frame_bgr, bbox),
    }

    mandatory_ok = checks["area"] and checks["skin_ycrcb"] and checks["skin_hsv"]
    # Multiple independent checks must pass to avoid non-human/screen spoofs.
    passed = mandatory_ok and (sum(bool(v) for v in checks.values()) >= (MIN_TOTAL_GUARDRAIL_PASSES + 2))
    return passed, checks


def _passes_face_geometry_guard(lm: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
    _, _, bw, bh = bbox
    face_w = max(float(bw), 1.0)
    face_h = max(float(bh), 1.0)

    try:
        left_eye = lm[33]
        right_eye = lm[263]
        nose_tip = lm[1]
        upper_lip = lm[13]

        eye_dist = float(np.linalg.norm(left_eye - right_eye) / face_w)
        nose_to_mouth = float(np.linalg.norm(nose_tip - upper_lip) / face_h)
        eye_y = float((left_eye[1] + right_eye[1]) * 0.5)
        eye_to_mouth = float((upper_lip[1] - eye_y) / face_h)
        nose_center_offset = float(
            abs(nose_tip[0] - ((left_eye[0] + right_eye[0]) * 0.5)) / face_w
        )
    except Exception:
        return False

    geometry_checks = [
        0.22 <= eye_dist <= 0.70,
        0.03 <= nose_to_mouth <= 0.35,
        0.12 <= eye_to_mouth <= 0.70,
        nose_center_offset <= 0.20,
    ]
    return sum(bool(v) for v in geometry_checks) >= 3


def _passes_texture_guard(frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
    x, y, bw, bh = bbox
    h, w = frame_bgr.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + bw)
    y2 = min(h, y + bh)
    if x2 <= x1 or y2 <= y1:
        return False

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return False

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return MIN_FACE_TEXTURE_VAR <= lap_var <= MAX_FACE_TEXTURE_VAR

def _estimate_skin_consistency(frame_bgr: np.ndarray, face_mask: np.ndarray) -> Tuple[float, float, float]:
    face_pixels = face_mask > 0
    total_face_px = int(np.count_nonzero(face_pixels))
    if total_face_px == 0:
        return 0.0, 0.0, 0.0

    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    skin_ycrcb = (
        (y >= 40)
        & (cr >= 133)
        & (cr <= 173)
        & (cb >= 77)
        & (cb <= 127)
    )

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    skin_hsv = (
        (h >= 0)
        & (h <= 25)
        & (s >= 30)
        & (s <= 180)
        & (v >= 40)
        & (v <= 255)
    )

    skin_ycrcb_face = skin_ycrcb & face_pixels
    skin_hsv_face = skin_hsv & face_pixels
    intersection = int(np.count_nonzero(skin_ycrcb_face & skin_hsv_face))
    union = int(np.count_nonzero(skin_ycrcb_face | skin_hsv_face))

    skin_ratio_ycrcb = float(np.count_nonzero(skin_ycrcb_face) / total_face_px)
    skin_ratio_hsv = float(np.count_nonzero(skin_hsv_face) / total_face_px)
    skin_iou = float(intersection / max(1, union))
    return skin_ratio_ycrcb, skin_ratio_hsv, skin_iou


def get_mean_rgb(frame: np.ndarray, mask: np.ndarray):
    px = frame[mask > 0]
    if len(px) == 0:
        return np.nan, np.nan, np.nan
    return float(np.mean(px[:, 2])), float(np.mean(px[:, 1])), float(np.mean(px[:, 0]))





def compute_mad_confidence(roi_g_values: Dict[str, float]) -> float:
    vals = [v for v in roi_g_values.values() if not np.isnan(v)]
    if len(vals) < 2:
        return 0.0
    med = float(np.median(vals))
    mad = float(np.mean(np.abs(np.array(vals) - med)))
    return float(np.clip(1.0 - mad / 20.0, 0.0, 1.0))


def run_extraction(args):
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    with VideoSource(args.source, args.fps) as vid, FaceROIExtractor(args.model_path) as roi_ext:
        print(
            f"Source : {args.source}\n"
            f"Resolution : {vid.width}×{vid.height} @ {vid.orig_fps:.1f} fps\n"
        )

        data = {"ts": [], "idx": [], "r_face": [], "g_face": [], "b_face": [], "conf": []}
        face_crops = []
        writer = None
        if args.save_preview:
            writer = cv2.VideoWriter(
                str(out_dir / f"preview_{ts_tag}.mp4"),
                cv2.VideoWriter.fourcc(*"mp4v"),
                args.fps,
                (vid.width, vid.height),
            )

        total_frames = int(args.duration * args.fps) if args.duration else None
        pbar = tqdm(vid.frames(args.duration), total=total_frames, desc="Processing", unit="fr")

        for idx, (ts, frame) in enumerate(pbar):
            roi_res = roi_ext.process(frame, int(ts * 1000))
            if roi_res is None:
                continue

            valid_quality = all(px >= MIN_ROI_PIXELS for px in roi_res.px_counts.values())
            if not valid_quality:
                roi_ext.stats["low_quality"] += 1
                if not (args.preview or writer):
                    continue

            if valid_quality:
                r, g, b = get_mean_rgb(frame, roi_res.masks["face"])
                data["ts"].append(ts)
                data["idx"].append(idx)
                data["r_face"].append(r)
                data["g_face"].append(g)
                data["b_face"].append(b)
                data["conf"].append(compute_mad_confidence({"face": g}))
                face_crops.append(roi_res.crops["face"])

            if args.preview or writer:
                vis = frame.copy()
                for roi, color in ROI_COLORS.items():
                    overlay = vis.copy()
                    overlay[roi_res.masks[roi] > 0] = color
                    cv2.addWeighted(overlay, 0.38, vis, 0.62, 0, vis)
                if writer:
                    writer.write(vis)
                if args.preview:
                    cv2.imshow("CardioVision v3", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            pbar.set_postfix(
                det=roi_ext.stats["detected"],
                miss=roi_ext.stats["failed"],
                low_q=roi_ext.stats["low_quality"],
                conf=f"{data['conf'][-1]:.2f}" if valid_quality else "---",
            )

        if writer:
            writer.release()
        cv2.destroyAllWindows()

        if not data["ts"]:
            print("No usable frames — check lighting and source.")
            return

        df = pd.DataFrame(data)
        for ch in ("r", "g", "b"):
            df[f"{ch}_combined"] = df[f"{ch}_face"]

        csv_path = out_dir / f"signals_{ts_tag}.csv"
        npz_path = out_dir / f"signals_{ts_tag}.npz"
        df.to_csv(csv_path, index=False)

        npz_payload: Dict[str, Any] = {
            str(c): np.asarray(df[c].to_numpy(copy=False)) for c in df.columns
        }
        np.savez_compressed(
            npz_path,
            **cast(dict[str, Any], npz_payload),
        )

        print(f"\n✓ Saved CSV : {csv_path}")
        print(f"✓ Saved NPZ : {npz_path}")
        print(f" Detected : {roi_ext.stats['detected']}/{roi_ext.stats['detected'] + roi_ext.stats['failed']} frames")
        print(f" Low-quality: {roi_ext.stats['low_quality']} frames discarded (< {MIN_ROI_PIXELS}px)")
        print(f" Non-human rejects: {roi_ext.stats['rejected_nonhuman']}")
        print(
            " Guardrail rejects -> "
            f"area:{roi_ext.stats['rej_area']} "
            f"skin_ycrcb:{roi_ext.stats['rej_skin_ycrcb']} "
            f"skin_hsv:{roi_ext.stats['rej_skin_hsv']} "
            f"skin_iou:{roi_ext.stats['rej_skin_iou']} "
            f"geometry:{roi_ext.stats['rej_geometry']} "
            f"texture:{roi_ext.stats['rej_texture']}"
        )
        print(f" Usable : {len(df)} frames written to disk")
        print(f" Mean conf : {df['conf'].mean():.3f}")

        ts_values = np.asarray(df["ts"].to_numpy(copy=False), dtype=np.float64)
        actual_fps = 1.0 / np.mean(np.diff(ts_values)) if len(ts_values) > 1 else args.fps

        r_interp = np.asarray(
            df["r_combined"].interpolate().to_numpy(copy=False), dtype=np.float64
        )
        g_interp = np.asarray(
            df["g_combined"].interpolate().to_numpy(copy=False), dtype=np.float64
        )
        b_interp = np.asarray(
            df["b_combined"].interpolate().to_numpy(copy=False), dtype=np.float64
        )
        rgb = np.column_stack(
            [
                r_interp,
                g_interp,
                b_interp,
            ]
        )
        fps_for_processing = int(round(actual_fps)) if actual_fps > 0 else int(args.fps)
        face_frames = np.asarray(face_crops, dtype=np.uint8) if face_crops else None
        result = process_rppg_with_deep(
            rgb_raw=rgb,
            fps=fps_for_processing,
            face_frames=face_frames,
            selection_mode="best_confidence",
        )

        proc_path = out_dir / f"processed_{ts_tag}.npz"
        np.savez_compressed(
            proc_path,
            pulse_signal=result["pulse_signal"],
            timestamps=result["timestamps"],
            fps=result["fps"],
            peaks_idx=result["peaks_idx"],
            ibi_ms=result["ibi_ms"],
            confidence=result["confidence"],
            is_reliable=result["is_reliable"],
            motion_fraction=result["motion_fraction"],
            method_used=result["method_used"],
            n_frames=result["n_frames"],
            duration_sec=result["duration_sec"],
        )
        print(f"\n✓ Saved processed : {proc_path}")

        if result["is_reliable"] and len(result["ibi_ms"]) > 0:
            bpm = 60000.0 / result["ibi_ms"].mean()
            print(f" ♥ Estimated HR : {bpm:.1f} BPM")
            print(f" ♥ Mean IBI : {result['ibi_ms'].mean():.1f} ms")

            hrv = result.get("hrv_features", {})
            if hrv:
                print(f" 📊 RMSSD : {hrv.get('rmssd_ms', 0):.1f} ms")
                print(f" 📊 SDNN : {hrv.get('sdnn_ms', 0):.1f} ms")
                print(f" 📈 LF/HF Ratio : {hrv.get('lf_hf_ratio', 0):.2f}")
                print(f" 🧠 Stress Level : {hrv.get('stress_level', 'Unknown')} (Index: {hrv.get('stress_index', 0):.1f})")

            print(f" Confidence : {result['confidence']:.3f}")
            print(f" Reliable : {result['is_reliable']}")
            print(f" Motion fraction : {result['motion_fraction']:.1%}")
            print(f" Method selected : {result.get('selected_source', result.get('method_used', 'pos'))}")
            print(f" POS confidence : {result.get('pos_confidence', 0.0):.3f}")
            print(f" Deep confidence : {result.get('deep_confidence', 0.0):.3f}")


def main():
    p = argparse.ArgumentParser(
        description="CardioVision rPPG Pipeline v3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--source", default="0", help="Webcam index (int) or path to video file"
    )
    p.add_argument("--fps", type=float, default=60.0)
    p.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Seconds to record (None = until 'q' or EOF)",
    )
    p.add_argument("--output", default="./output")
    p.add_argument("--preview", action="store_true", help="Show live overlay window")
    p.add_argument("--save-preview", action="store_true")
    p.add_argument("--model", default=DEFAULT_MODEL_PATH, dest="model_path")
    p.add_argument(
        "--mode",
        choices=["batch", "stream"],
        default="batch",
        help="batch: process a source file/webcam, stream: start websocket backend",
    )
    p.add_argument("--ws-host", default="0.0.0.0")
    p.add_argument("--ws-port", type=int, default=8765)
    p.add_argument("--jpeg-quality", type=int, default=45)
    p.add_argument("--overlay-max-side", type=int, default=320)
    p.add_argument("--overlay-stride", type=int, default=2)
    p.add_argument(
        "--live-deep-mode",
        choices=["off", "final-only", "live+final"],
        default="final-only",
        help="off: POS only, final-only: run deep model only at stop, live+final: run deep in live updates and final",
    )

    args = p.parse_args()
    if args.mode == "stream":
        import asyncio
        from stream_server import run_server

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
        return

    try:
        args.source = int(args.source)
    except ValueError:
        pass
    run_extraction(args)


if __name__ == "__main__":
    main()

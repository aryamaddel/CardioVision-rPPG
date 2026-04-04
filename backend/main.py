import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, cast

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from roi_pipeline import (
    FaceROIExtractor,
    MIN_ROI_PIXELS,
    VideoSource,
    compute_mad_confidence,
    get_mean_rgb,
    overlay_roi,
)
from rppg_core import process_rppg_with_deep

DEFAULT_MODEL_PATH = str(Path(__file__).with_name("face_landmarker.task"))


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
                vis = overlay_roi(frame, roi_res.masks)
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

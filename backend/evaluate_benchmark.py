import json
import argparse
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from main import FaceROIExtractor, get_mean_rgb, ROI_WEIGHTS, ROI_COLORS
from rppg_core import process_rppg


def load_gt_bpm(gt_path):
    with open(gt_path, "r") as f:
        lines = [line_str.strip() for line_str in f.readlines() if line_str.strip()]
    return (
        float(np.mean([float(v) for v in lines[1].split()]))
        if len(lines) >= 2
        else None
    )


def evaluate(subject_dir, preview=False, verbose=True):
    video_path, gt_path = subject_dir / "vid.avi", subject_dir / "ground_truth.txt"
    gt_bpm = load_gt_bpm(gt_path)
    if not video_path.exists() or gt_bpm is None:
        return None

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    roi_ext = FaceROIExtractor("face_landmarker.task", 0.5, 0.5)
    
    idx, frame_data = 0, []
    pbar = None
    if verbose:
        pbar = tqdm(total=total_frames, desc=f"  {subject_dir.name}", leave=False)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        res = roi_ext.process(frame, int((idx / fps) * 1000))
        
        if res:
            # Cut out everything except the face (use copy to avoid corrupting original)
            if res.face_mask is not None:
                frame = frame.copy()
                frame[res.face_mask == 0] = 0

            rgb = [0.0, 0.0, 0.0]
            for r, w in ROI_WEIGHTS.items():
                m_rgb = get_mean_rgb(frame, res.masks[r])
                if not np.isnan(m_rgb[1]):
                    rgb = [rgb[i] + w * m_rgb[i] for i in range(3)]
            
            frame_data.append({"idx": idx, "ts": idx / fps, "r": rgb[0], "g": rgb[1], "b": rgb[2]})

            if preview:
                vis = frame.copy()
                for roi, color in ROI_COLORS.items():
                    overlay = vis.copy()
                    if roi in res.masks:
                        overlay[res.masks[roi] > 0] = color
                        cv2.addWeighted(overlay, 0.38, vis, 0.62, 0, vis)

                y_off = 24
                for roi, cnt in res.px_counts.items():
                    cv2.putText(
                        vis,
                        f"{roi}: {cnt}px",
                        (8, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    y_off += 18
                
                cv2.imshow("Benchmark Preview", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    preview = False # Stop previewing but keep processing
                    cv2.destroyWindow("Benchmark Preview")

        idx += 1
        if pbar:
            pbar.update(1)
        
    if pbar:
        pbar.close()
    cap.release()
    roi_ext.close()
    if preview:
        cv2.destroyAllWindows()

    if len(frame_data) < 100:
        return None

    df = pd.DataFrame(frame_data)
    rgb_arr = df[["r", "g", "b"]].values
    actual_fps = 1.0 / np.mean(np.diff(df["ts"].values)) if len(df) > 1 else fps
    res_rppg = process_rppg(rgb_arr, fps=actual_fps)
    est_bpm = 60000.0 / np.median(res_rppg["ibi_ms"]) if res_rppg["ibi_ms"].size > 0 else None
    
    return (
        {
            "subject": subject_dir.name,
            "gt": round(gt_bpm, 2),
            "est": round(est_bpm, 2) if est_bpm else None,
            "err": round(abs(est_bpm - gt_bpm), 2) if est_bpm else None,
        }
        if est_bpm
        else None
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate CardioVision on Benchmark")
    parser.add_argument("--preview", action="store_true", help="Show live preview (disables multi-processing)")
    parser.add_argument("--jobs", type=int, default=1, help="Number of parallel processes (default: 1)")
    args = parser.parse_args()

    subjects = sorted([d for d in Path("test_video").iterdir() if d.is_dir()])
    report = {}
    
    print("--- Running Benchmark (POS) ---")
    results = []
    
    # Force single process if preview is enabled
    use_parallel = args.jobs > 1 and not args.preview
    if args.jobs > 1 and args.preview:
        print("Warning: Live preview enabled. Forcing sequential processing (jobs=1).")
        use_parallel = False

    if use_parallel:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {executor.submit(evaluate, s, preview=False, verbose=False): s for s in subjects}
            # Overall progress bar for parallel execution
            for future in tqdm(as_completed(futures), total=len(subjects), desc="Processing"):
                r = future.result()
                if r:
                    results.append(r)
                    print(f"    {r['subject']}: GT={r['gt']}, Est={r['est']}, Err={r['err']}")
    else:
        # Sequential execution with detailed progress bar
        for s in subjects:
            r = evaluate(s, preview=args.preview, verbose=True)
            if r:
                results.append(r)
                print(f"  {r['subject']}: GT={r['gt']}, Est={r['est']}, Err={r['err']}")
    
    mae = np.mean([r["err"] for r in results if r["err"] is not None]) if results else 0
    report = {"mae": round(float(mae), 2), "results": results}
    print(f"  MAE: {mae:.2f}\n")

    with open("benchmark_results.json", "w") as f:
        json.dump(report, f, indent=4)
    print("Done! Results saved to benchmark_results.json")


if __name__ == "__main__":
    # MediaPipe and OpenCV on macOS require 'spawn' for safe multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()



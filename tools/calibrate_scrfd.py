"""Calibrate SCRFD decode: anchor offset, prediction scaling, kps point order.

Compares decoded 5-point keypoints to MediaPipe's landmark anatomy
(iris centres 468/473, nose tip 1, mouth corners 61/291) on real frames.
"""
from __future__ import annotations
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gpu_core import SCRFD  # noqa: E402
from deepfake_detect import RegionExtractor  # noqa: E402

REF_IDX = [468, 473, 1, 61, 291]
FRAMES = [10, 50, 300, 700, 1100, 1400]
_TS = [0]  # MediaPipe VIDEO mode needs strictly increasing timestamps


def read_frame(video: Path, idx: int):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, f = cap.read()
    cap.release()
    return f if ok else None


def main():
    video = ROOT / "Dataset" / "subject_1" / "vid.avi"
    ref = RegionExtractor()
    results = {}
    for offset in (0.0, 0.5):
        for scale in (True, False):
            det = SCRFD(gpu=False, center_offset=offset, scale_preds=scale)
            errs_swap, errs_keep, box_ok = [], [], 0
            for fi in FRAMES:
                frame = read_frame(video, fi)
                if frame is None:
                    continue
                _TS[0] += 33
                ref_lm = ref.landmarks(frame, _TS[0])
                if ref_lm is None:
                    continue
                boxes, kps, scores = det.detect(frame)
                if len(boxes) == 0:
                    continue
                i = int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))
                k = kps[i]
                ref_pts = ref_lm[REF_IDX]
                e_keep = np.linalg.norm(k - ref_pts, axis=1)
                k_swap = k[[1, 0, 2, 4, 3]]
                e_swap = np.linalg.norm(k_swap - ref_pts, axis=1)
                errs_keep.append(e_keep.mean())
                errs_swap.append(e_swap.mean())
                # box should roughly contain the MediaPipe face
                mnp = ref_lm[:468]
                inside = ((boxes[i, 0] <= mnp[:, 0].min()) and (boxes[i, 1] <= mnp[:, 1].min())
                          and (boxes[i, 2] >= mnp[:, 0].max()) and (boxes[i, 3] >= mnp[:, 1].max()))
                box_ok += int(inside)
            key = (offset, scale)
            results[key] = (float(np.mean(errs_keep)) if errs_keep else 1e9,
                            float(np.mean(errs_swap)) if errs_swap else 1e9,
                            box_ok, len(errs_keep))
    print(f"\n{'offset':>7} {'scale':>6} {'err_keep':>9} {'err_swap':>9} {'box_contains_face':>18}")
    best = None
    for (offset, scale), (ek, es, ok, n) in results.items():
        print(f"{offset:>7.1f} {str(scale):>6} {ek:>9.2f} {es:>9.2f} {f'{ok}/{n}':>18}")
        cand = min(ek, es)
        if best is None or cand < best[0]:
            best = (cand, offset, scale, ek, es)
    order = "swap(kps[1],kps[0],kps[2],kps[4],kps[3])" if best[4] > best[3] else "keep(as-is)"
    print(f"\n-> best: center_offset={best[1]} scale_preds={best[2]} "
          f"({best[0]:.2f} px mean kps error; order: {order})")
    ref.close()


if __name__ == "__main__":
    main()

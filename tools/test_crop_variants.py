"""Test crop-rect variants for the GPU mesh path against MediaPipe.

Variants:
  A current  : cw=1.61*bw, ch=1.33*bh (joint ratio fit), fraction centres
  B absolute : fixed 191x229 px crop (mean optimal side from fit_crop),
               fraction centres
  C abs+aff  : fixed 191x229 px crop + affine centres dx=-0.598*bw+83.4,
               dy=-0.721*bh+138.4 (fit_crop.py 1 2)
  D hybrid   : absolute crop clamped to >= 1.2*box (safe for small faces)

Reports mean/p95 landmark error vs MediaPipe per variant, per subject.
"""
from __future__ import annotations
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from deepfake_detect import RegionExtractor  # noqa: E402
from gpu_core import GPUMeshLandmarker, SCRFD  # noqa: E402

ABS_W, ABS_H = 191.0, 229.0


def crop_rect(variant, box, h, w):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if variant == "A":
        cw, ch = 1.61 * bw, 1.33 * bh
        cx += 0.11 * bw
        cy += 0.07 * bh
    elif variant == "B":
        cw, ch = ABS_W, ABS_H
        cx += 0.11 * bw
        cy += 0.07 * bh
    elif variant == "C":
        cw, ch = ABS_W, ABS_H
        cx += -0.598 * bw + 83.4
        cy += -0.721 * bh + 138.4
    elif variant == "D":
        cw, ch = max(ABS_W, 1.2 * bw), max(ABS_H, 1.2 * bh)
        cx += -0.598 * bw + 83.4
        cy += -0.721 * bh + 138.4
    return cx - cw / 2, cy - ch / 2, cw, ch


def run(video: Path, mesh, ref: RegionExtractor, n=100, ts0=0):
    """ts0: running timestamp offset (MediaPipe VIDEO mode needs strictly
    increasing stamps for the life of one landmarker instance)."""
    cap = cv2.VideoCapture(str(video))
    ts = ts0
    errs = {v: [] for v in "ABCD"}
    for i in range(n):
        ok, fr = cap.read()
        if not ok:
            break
        ts += 33
        ref_lm = ref.landmarks(fr, ts)
        if ref_lm is None:
            continue
        d = mesh.detector.largest_face(fr)
        if d is None:
            continue
        h, w = fr.shape[:2]
        for v in "ABCD":
            x0, y0, cw, ch = crop_rect(v, d[0], h, w)
            mat = np.array([[cw / 256, 0, x0], [0, ch / 256, y0]], np.float32)
            crop = cv2.warpAffine(fr, mat, (256, 256), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
            out = mesh.sess.run(None, {mesh.in_name: crop.astype(np.float32)[None] / 255.0})
            pts = out[0].reshape(-1, 3)[:478, :2]
            lm = pts @ mat[:, :2].T + mat[:, 2]
            e = np.linalg.norm(lm - ref_lm, axis=1)
            errs[v].append((float(e.mean()), float(np.percentile(e, 95))))
    cap.release()
    return errs, ts


def main():
    ref = RegionExtractor()
    mesh = GPUMeshLandmarker()
    ts_run = 0
    for s in (1, 2):
        video = ROOT / "Dataset" / f"subject_{s}" / "vid.avi"
        errs, ts_run = run(video, mesh, ref, ts0=ts_run)
        print(f"=== subject_{s} ===")
        for v in "ABCD":
            if not errs[v]:
                continue
            m = np.array(errs[v])
            print(f"  {v}: mean {m[:, 0].mean():5.2f} px   p95 {m[:, 1].mean():5.2f} px")
    ref.close()


if __name__ == "__main__":
    main()

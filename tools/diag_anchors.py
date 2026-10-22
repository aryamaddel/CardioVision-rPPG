"""Landmark error at the ROI-anchor indices specifically (vs 478-point mean).

The rPPG ROIs only use a handful of interior points (forehead/cheek anchors +
face hull). If those agree with MediaPipe better than the full-mesh mean
(contour/eyebrow points are the worst), the effective trace quality hit is
smaller than the headline number suggests.

Usage: python tools/diag_anchors.py [--subject 1|2] [--frames 120]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from deepfake_detect import REGIONS, RegionExtractor  # noqa: E402
from gpu_core import GPURegionExtractor  # noqa: E402
from rppg import EXCLUDE  # noqa: E402

ANCHORS = sorted({i for ids in REGIONS.values() for i in ids})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=2, choices=[1, 2])
    ap.add_argument("--frames", type=int, default=120)
    args = ap.parse_args()

    video = ROOT / "Dataset" / f"subject_{args.subject}" / "vid.avi"
    mp_ex, gpu_ex = RegionExtractor(), GPURegionExtractor(smooth=False)
    cap = cv2.VideoCapture(str(video))
    errs_full, errs_anch = [], []
    ts = 0
    for i in range(args.frames):
        ok, fr = cap.read()
        if not ok:
            break
        ts += 33
        a = mp_ex.landmarks(fr, ts)
        b = gpu_ex.landmarks(fr, ts)
        if a is None or b is None:
            continue
        e = np.linalg.norm(a - b, axis=1)
        errs_full.append(e)
        errs_anch.append(e[ANCHORS])
    cap.release()
    mp_ex.close(); gpu_ex.close()
    f = np.concatenate(errs_full)
    an = np.concatenate(errs_anch)
    print(f"subject_{args.subject}: full 478 pts mean {f.mean():.2f} "
          f"p95 {np.percentile(f, 95):.2f} px")
    print(f"subject_{args.subject}: ROI anchors   mean {an.mean():.2f} "
          f"p95 {np.percentile(an, 95):.2f} px (indices {ANCHORS})")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()

"""Generate 4 additional out-of-sample test clips (all CPU).

  fake_smooth_k2_s1    mild temporal average (k=2) on subject_1 - a *harder*
                       smooth fake (less attenuation, still not generated)
  fake_smooth_k15_s2   heavy temporal average (k=15) on subject_2
  fake_nnsmooth_s2toS1 neural swap -> temporal average chain (S2->S1)
  fake_nnsmooth_s1toS2 neural swap -> temporal average chain (S1->S2)

The chains answer "what if a generator applies temporal smoothing on top of a
neural swap?" - the strongest fakes in the set. Smoothing is pure CPU; the
nnswap source clips already exist in tmp/.

Run:  uv run python tools/make_more_fakes.py
"""
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepfake_detect import ROOT, TMP

FRAMES = 360


def smooth_clip(src: Path, dst: Path, k: int, n_frames: int = FRAMES):
    if dst.exists():
        print(f"  exists: {dst.name}")
        return
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wr = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    buf = deque(maxlen=2 * k + 1)
    for _ in range(n_frames):
        ok, f = cap.read()
        if not ok:
            break
        buf.append(f.astype(np.float32))
        wr.write(np.mean(buf, axis=0).astype(np.uint8))
    wr.release(); cap.release()
    print(f"  wrote {dst.name} (k={k})")


def main():
    TMP.mkdir(exist_ok=True)
    v1 = ROOT / "Dataset" / "subject_1" / "vid.avi"
    v2 = ROOT / "Dataset" / "subject_2" / "vid.avi"
    print("== mild/heavy smooth fakes ==")
    smooth_clip(v1, TMP / "fake_smooth_k2_s1_360.avi", k=2)
    smooth_clip(v2, TMP / "fake_smooth_k15_s2_360.avi", k=15)
    print("== neural-swap + smooth chains ==")
    smooth_clip(TMP / "fake_nnswap_s2toS1_360.avi",
                TMP / "fake_nnsmooth_s2toS1_360.avi", k=9)
    smooth_clip(TMP / "fake_nnswap_s1toS2_360.avi",
                TMP / "fake_nnsmooth_s1toS2_360.avi", k=9)


if __name__ == "__main__":
    main()

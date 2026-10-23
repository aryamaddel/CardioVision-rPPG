"""Find the mesh.onnx input convention: normalization, channel order, layout.

Metric: after the model runs on a fixed crop, fit an affine map from its raw
output to MediaPipe's 468 reference landmarks; the residual is invariant to
crop geometry, so it isolates prediction quality. Best convention = lowest
residual across frames.
"""
from __future__ import annotations
import itertools
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gpu_core import GPUMeshLandmarker  # noqa: E402
from deepfake_detect import RegionExtractor  # noqa: E402

FRAMES = [100, 500, 900]
NORMS = {
    "x_0_255": (lambda c: c.astype(np.float32), 1.0),
    "x_0_1": (lambda c: c.astype(np.float32) / 255.0, 1.0),
    "pm1_from255": (lambda c: c.astype(np.float32) / 127.5 - 1.0, 1.0),
    "pm1_from128": (lambda c: c.astype(np.float32) / 128.0 - 1.0, 1.0),
}
SHIFTS = [0, 10]   # in case iris points come first


def fit_resid(raw, ref):
    ones = np.ones((len(raw), 1))
    Ax = np.hstack([ones, raw[:, [0]]])
    Ay = np.hstack([ones, raw[:, [1]]])
    sx, *_ = np.linalg.lstsq(Ax, ref[:, 0], rcond=None)
    sy, *_ = np.linalg.lstsq(Ay, ref[:, 1], rcond=None)
    pred = np.stack([Ax @ sx, Ay @ sy], axis=1)
    return float(np.linalg.norm(pred - ref, axis=1).mean())


def main():
    video = ROOT / "Dataset" / "subject_1" / "vid.avi"
    ref = RegionExtractor()
    mesh = GPUMeshLandmarker()

    frames = []
    cap = cv2.VideoCapture(str(video))
    want = set(FRAMES)
    idx = ts = 0
    while want:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in want:
            want.discard(idx)
            ts += 33
            rlm = ref.landmarks(frame, ts)
            d = mesh.detector.largest_face(frame)
            if rlm is None or d is None:
                idx += 1
                continue
            h, w = frame.shape[:2]
            x0, y0, cw, ch = mesh._crop_rect(d[0], h, w)
            mat = np.array([[cw / mesh.SIZE, 0, x0], [0, ch / mesh.SIZE, y0]], np.float32)
            crop = cv2.warpAffine(frame, mat, (mesh.SIZE, mesh.SIZE),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            frames.append((crop, rlm[:468]))
        idx += 1
    cap.release()
    print(f"loaded {len(frames)} frames for input-convention sweep\n")

    results = []
    for (norm_name, (fn, _)), chg_first, shift in itertools.product(
            NORMS.items(), (False, True), SHIFTS):
        errs = []
        for crop, ref_pts in frames:
            c = fn(crop[..., ::-1] if chg_first else crop)   # RGB variant first-swap
            out = mesh.sess.run(None, {mesh.in_name: c[None]})[0]
            raw = out.reshape(-1, 3)[shift:shift + 468, :2]
            errs.append(fit_resid(raw, ref_pts))
        results.append((float(np.mean(errs)), norm_name, "RGB" if chg_first else "BGR", shift))
        print(f"{norm_name:<13} {'RGB' if chg_first else 'BGR'} shift={shift:>2} "
              f"resid={results[-1][0]:6.2f} px")

    results.sort()
    best = results[0]
    print(f"\n-> best: norm={best[1]} channels={best[2]} shift={best[3]} "
          f"(residual {best[0]:.2f} px)")


if __name__ == "__main__":
    main()

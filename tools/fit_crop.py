"""Fit the exact crop->frame geometry the mesh model wants.

For each frame: run mesh on a nominal crop, then regress MediaPipe's reference
landmarks on the model's raw crop-space output:
    ref_x = x0 + raw_x * kx   (crop origin + scale, k = side/256)
gives the *optimal* crop rect for that frame. We then express it relative to
the SCRFD box (margin ratio, centre delta) and check the shape residual
(after alignment) to see how good the mesh model actually is.

Also fits a rotation (similarity) variant to detect rotation sensitivity.
"""
from __future__ import annotations
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gpu_core import GPUMeshLandmarker, SCRFD  # noqa: E402
from deepfake_detect import RegionExtractor  # noqa: E402

FRAMES = list(range(50, 1450, 100))
_START_MARGIN = 1.3


def fit_one(video: Path, ref: RegionExtractor, mesh: GPUMeshLandmarker,
            ts0: int = 0):
    """Fit crop geometry on one video. ts0: starting timestamp offset —
    MediaPipe VIDEO mode needs strictly increasing stamps across the whole
    life of a landmarker, so callers pass a running offset per video."""
    det = mesh.detector
    rows = []
    cap = cv2.VideoCapture(str(video))
    want = set(FRAMES)
    idx = 0
    ts = ts0
    while want:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in want:
            want.discard(idx)
            ts += 33
            ref_lm = ref.landmarks(frame, ts)
            if ref_lm is None:
                idx += 1
                continue
            d = det.largest_face(frame)
            if d is None:
                idx += 1
                continue
            box = d[0]
            h, w = frame.shape[:2]
            x0, y0, cw, ch = mesh._crop_rect(box, h, w)
            mat = np.array([[cw / mesh.SIZE, 0, x0], [0, ch / mesh.SIZE, y0]], np.float32)
            crop = cv2.warpAffine(frame, mat, (mesh.SIZE, mesh.SIZE),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            out = mesh.sess.run(None, {mesh.in_name: crop.astype(np.float32)[None] / 255.0})
            raw = out[0].reshape(-1, 3)[:468, :2]              # crop-space pts
            ref_pts = ref_lm[:468]                              # frame-space ref

            # affine (axis-aligned) fit: ref = a + raw * k  (per axis)
            ones = np.ones((len(raw), 1))
            Ax = np.hstack([ones, raw[:, [0]]])
            Ay = np.hstack([ones, raw[:, [1]]])
            sol_x, *_ = np.linalg.lstsq(Ax, ref_pts[:, 0], rcond=None)
            sol_y, *_ = np.linalg.lstsq(Ay, ref_pts[:, 1], rcond=None)
            pred = np.stack([Ax @ sol_x, Ay @ sol_y], axis=1)
            resid = float(np.linalg.norm(pred - ref_pts, axis=1).mean())

            # similarity fit (scale + rot + trans)
            M, _ = cv2.estimateAffinePartial2D(raw.astype(np.float32), ref_pts.astype(np.float32),
                                               method=cv2.LMEDS)
            pred_sim = cv2.transform(raw.reshape(1, -1, 2).astype(np.float32), M)[0]
            resid_sim = float(np.linalg.norm(pred_sim - ref_pts, axis=1).mean())

            fit_x0, fit_kx = sol_x
            fit_y0, fit_ky = sol_y
            side_x, side_y = fit_kx * mesh.SIZE, fit_ky * mesh.SIZE
            cx, cy = fit_x0 + fit_kx * 128, fit_y0 + fit_ky * 128
            bw, bh = box[2] - box[0], box[3] - box[1]
            bcx, bcy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            rows.append(dict(idx=idx, side_x=side_x, side_y=side_y,
                             ratio_w=side_x / bw, ratio_h=side_y / bh,
                             fdx=(cx - bcx) / bw, fdy=(cy - bcy) / bh,
                             dx=cx - bcx, dy=cy - bcy,
                             bw=bw, bh=bh, aspect=bw / bh,
                             resid_affine=resid, resid_sim=resid_sim))
            print(f"frame {idx:>4}: side_x={side_x:6.1f} side_y={side_y:6.1f} "
                  f"ratio_w={side_x / bw:.3f}*bw ratio_h={side_y / bh:.3f}*bh "
                  f"center_delta=({cx - bcx:+6.1f},{cy - bcy:+6.1f}) "
                  f"resid_affine={resid:5.2f} resid_sim={resid_sim:5.2f} px")
        idx += 1
    cap.release()
    return rows, ts


def main():
    subjects = [int(s) for s in (sys.argv[1:] or ["1"])]
    ref = RegionExtractor()
    mesh = GPUMeshLandmarker()
    print("backends ok; fitting crop geometry\n")

    all_rows = []
    ts_run = 0
    for s in subjects:
        video = ROOT / "Dataset" / f"subject_{s}" / "vid.avi"
        print(f"=== {video.name} ===")
        rows, ts_run = fit_one(video, ref, mesh, ts0=ts_run)
        for r in rows:
            r["subject"] = s
        all_rows += rows
    ref.close()

    if not all_rows:
        print("no frames fitted")
        return
    print("\n=== aggregate ===")
    for k in ("side_x", "side_y", "ratio_w", "ratio_h", "fdx", "fdy", "dx", "dy",
              "resid_affine", "resid_sim"):
        v = np.array([r[k] for r in all_rows])
        print(f"{k:>13}: mean {v.mean():7.2f}  std {v.std():6.2f}")
    for s in subjects:
        v = {k: np.array([r[k] for r in all_rows if r["subject"] == s])
             for k in ("ratio_w", "ratio_h", "fdx", "fdy", "resid_affine")}
        print(f"  subject_{s}: ratio_w={v['ratio_w'].mean():.3f} "
              f"ratio_h={v['ratio_h'].mean():.3f} fdx={v['fdx'].mean():+.3f} "
              f"fdy={v['fdy'].mean():+.3f} resid={v['resid_affine'].mean():.2f}px")

    # can SCRFD box geometry predict the per-frame optimal crop? (adaptive crop)
    print("\n=== crop predictability from box geometry ===")
    for tgt in ("ratio_w", "ratio_h", "fdx", "fdy"):
        y = np.array([r[tgt] for r in all_rows])
        for feat in ("aspect", "bw", "bh"):
            x = np.array([r[feat] for r in all_rows])
            c = float(np.corrcoef(x, y)[0, 1])
            print(f"  corr({feat:<6}, {tgt}) = {c:+.3f}")

    # affine crop model: side = k*box + pad  (fixed padding + proportional)
    print("\n=== affine crop model: side = k*box + pad ===")
    bw = np.array([r["bw"] for r in all_rows])
    bh = np.array([r["bh"] for r in all_rows])
    for name, boxv, sidename in (("cw", bw, "side_x"), ("ch", bh, "side_y")):
        side = np.array([r[sidename] for r in all_rows])
        A = np.hstack([boxv[:, None], np.ones((len(boxv), 1))])
        (k, pad), *_ = np.linalg.lstsq(A, side, rcond=None)
        pred = A @ np.array([k, pad])
        rms = float(np.sqrt(((pred - side) ** 2).mean()))
        print(f"  {sidename} = {k:.3f}*{name.replace('c', 'b')} + {pad:+.1f} px "
              f"(resid rms {rms:.2f}px)")
    # same for centre offsets in px
    for name, boxv, dname in (("dx", bw, "dx"), ("dy", bh, "dy")):
        d = np.array([r[dname] for r in all_rows])
        A = np.hstack([boxv[:, None], np.ones((len(boxv), 1))])
        (k, pad), *_ = np.linalg.lstsq(A, d, rcond=None)
        pred = A @ np.array([k, pad])
        rms = float(np.sqrt(((pred - d) ** 2).mean()))
        print(f"  {dname} = {k:.3f}*{name.replace('d', 'b')} + {pad:+.1f} px "
              f"(resid rms {rms:.2f}px)")
    print("\nIf resid_* small -> mesh shape is fine, only crop rect was wrong.")
    print("rect crop (side_x != side_y) confirms anisotropic MediaPipe rect.")


if __name__ == "__main__":
    main()

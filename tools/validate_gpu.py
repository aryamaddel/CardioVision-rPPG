"""
Validate the GPU pipeline against the MediaPipe CPU reference.

1. SCRFD detection sanity + kps vs MediaPipe anatomy.
2. Mesh landmarks: mean/p95 pixel error vs MediaPipe 478-landmark reference
   (crop geometry + input normalisation were fitted by tools/fit_crop.py and
   tools/fit_input.py; this script verifies the final numbers).
3. inswapper normalisation visual montage.
4. Saves PNGs to output/ for eyeballing.

Usage: python tools/validate_gpu.py [--frames 50 300 600]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gpu_core import (ArcFace, GPUMeshLandmarker, Inswapper, SCRFD,  # noqa: E402
                      backend_report)
from deepfake_detect import RegionExtractor  # noqa: E402 (MediaPipe reference)

OUT = ROOT / "output"

_REF_TS = [0]


def mp_reference(frame, ref: RegionExtractor):
    """MediaPipe VIDEO mode: strictly increasing timestamps, so auto-count."""
    _REF_TS[0] += 33
    return ref.landmarks(frame, _REF_TS[0])


def read_frame(video: Path, idx: int):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {idx} of {video}")
    return frame


def scrfd_sanity(det: SCRFD, frame, ref_lm):
    boxes, kps, scores = det.detect(frame)
    print(f"[scrfd] detections={len(boxes)} scores={np.round(scores, 3).tolist()}")
    if len(boxes) == 0:
        print("  !! no face detected")
        return False
    i = int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))
    b = boxes[i]
    ref_pts = ref_lm[[468, 473, 1, 61, 291]]
    err = float(np.linalg.norm(kps[i] - ref_pts, axis=1).mean())
    print(f"  largest box: {np.round(b, 1).tolist()}  kps error vs MediaPipe: {err:.2f} px")
    return True


def inswapper_montage(det, arc, swap, frame, name="swap"):
    """Visual check: target alignment + 4 in/out normalisation variants."""
    detres = det.largest_face(frame)
    if detres is None:
        print("  !! no face for swap test")
        return
    _box, kps, _s = detres
    aimg, M = swap.align_target(frame, kps)
    if aimg is None:
        print("  !! alignment failed")
        return
    emb = arc.embed(frame, kps)  # donor = same frame: identity round trip
    variants = {
        "A_raw0_255": (1.0, 0.0, 1.0, 0.0),
        "B_pm1": (1 / 127.5, -1.0, 127.5, 127.5),
        "C_0_1_default": (1 / 255.0, 0.0, 255.0, 0.0),
        "D_0_1_out1": (1 / 255.0, 0.0, 1.0, 0.0),
    }
    tiles = [cv2.resize(aimg, (256, 256))]
    labels = ["target align"]
    for key, (s_in, b_in, s_out, b_out) in variants.items():
        swap.in_scale, swap.in_bias, swap.out_scale, swap.out_bias = s_in, b_in, s_out, b_out
        sw = swap.swap_face(aimg, emb)
        pasted = swap.paste_back(frame, sw, M)
        tiles.append(cv2.resize(pasted, (256, 256)))
        labels.append(key)
        print(f"  variant {key}: swapped patch stats "
              f"mean={sw.mean():.1f} std={sw.std():.1f} min={sw.min()} max={sw.max()}")
    swap.in_scale, swap.in_bias = 1 / 255.0, 0.0
    swap.out_scale, swap.out_bias = 255.0, 0.0
    top = np.full((26 + 256 + 6, len(tiles) * 260, 3), 40, np.uint8)
    for i, t in enumerate(tiles):
        x = i * 260
        top[26:282, x + 2:x + 258] = t
        cv2.putText(top, labels[i], (x + 6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(OUT / f"validate_{name}_variants.png"), top)
    print(f"  wrote output/validate_{name}_variants.png")


def overlay(frame, lm_a, lm_b, path, labels=("A", "B")):
    img = frame.copy()
    if lm_a is not None:
        for p in lm_a.astype(int):
            cv2.circle(img, tuple(p), 1, (0, 255, 0), -1, lineType=cv2.LINE_AA)
    if lm_b is not None:
        for p in lm_b.astype(int):
            cv2.circle(img, tuple(p), 1, (0, 0, 255), -1, lineType=cv2.LINE_AA)
    cv2.putText(img, f"green={labels[0]} red={labels[1]} n="
                     f"{len(lm_a) if lm_a is not None else 0}"
                     f"/{len(lm_b) if lm_b is not None else 0}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, nargs="*", default=[50, 300, 600])
    ap.add_argument("--subject", type=str, default=str(ROOT / "Dataset" / "subject_1" / "vid.avi"))
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    video = Path(args.subject)
    ref = RegionExtractor()                       # MediaPipe CPU reference
    det = SCRFD()
    mesh = GPUMeshLandmarker()
    arc = ArcFace()
    swap = Inswapper()
    print("backends:", backend_report())

    all_err = []
    for fi, idx in enumerate(args.frames):
        frame = read_frame(video, idx)
        ref_lm = mp_reference(frame, ref)
        if ref_lm is None:
            print(f"frame {idx}: MediaPipe found no face, skipping")
            continue
        print(f"\n=== frame {idx} ===")
        scrfd_sanity(det, frame, ref_lm)
        gpu_lm = mesh.landmarks(frame, idx)
        if gpu_lm is None:
            print("  !! GPU mesh found no face")
            continue
        err = np.linalg.norm(gpu_lm - ref_lm, axis=1)
        all_err.append(err)
        print(f"[mesh] GPU vs MediaPipe landmarks: mean {err.mean():.2f} px, "
              f"p95 {np.percentile(err, 95):.2f} px, max {err.max():.2f} px")
        if fi == 0:
            inswapper_montage(det, arc, swap, frame, name=f"frame{idx}")
            overlay(frame, ref_lm, gpu_lm, OUT / "validate_landmarks_overlay.png",
                    labels=("mediapipe", "gpu"))

    if all_err:
        m = np.concatenate(all_err)
        print(f"\n=== overall: {len(all_err)} frames, mean {m.mean():.2f} px, "
              f"p95 {np.percentile(m, 95):.2f} px ===")
    ref.close()
    print("backends:", backend_report())


if __name__ == "__main__":
    main()

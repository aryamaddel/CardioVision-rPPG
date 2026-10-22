"""
GPU deepfake generation: modern one-shot neural face swap (InsightFace
inswapper_128) instead of the naive landmark-warp + Poisson blend in
deepfake_detect.make_swap_fake.

Everything runs on CUDA via gpu_core (SCRFD detection -> arcface identity
embedding -> inswapper generator -> feathered paste-back).

Usage:
  python make_deepfakes.py                 # both directions, 360 frames
  python make_deepfakes.py --frames 120
  python make_deepfakes.py --target Dataset/subject_1/vid.avi \
                           --donor  Dataset/subject_2/vid.avi \
                           --out    tmp/myswap.avi
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from gpu_core import ArcFace, Inswapper, SCRFD, GpuMonitor, backend_report

ROOT = Path(__file__).resolve().parent
TMP = ROOT / "tmp"

# module-level singletons: the 554 MB inswapper graph is loaded once per process
_COMP: dict = {}


def _components(gpu: bool = True) -> dict:
    if not _COMP:
        _COMP["det"] = SCRFD(gpu=gpu)
        _COMP["arc"] = ArcFace(gpu=gpu)
        _COMP["swap"] = Inswapper(gpu=gpu)
    return _COMP


def donor_embedding(donor: Path, comp: dict, max_probe: int = 40) -> np.ndarray:
    """ArcFace identity of the donor from its first detectable frame."""
    cap = cv2.VideoCapture(str(donor))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {donor}")
    for _ in range(max_probe):
        ok, frame = cap.read()
        if not ok:
            break
        det = comp["det"].largest_face(frame)
        if det is None:
            continue
        emb = comp["arc"].embed(frame, det[1])
        if emb is not None:
            cap.release()
            print(f"  donor identity from {donor.name} ({_ + 1}th frame)")
            return emb
    cap.release()
    raise RuntimeError(f"no detectable face in donor {donor}")


def make_nnswap_fake(target: Path, donor: Path, dst: Path, n_frames: int,
                     gpu: bool = True) -> Path:
    """inswapper_128 identity swap: donor identity -> every frame of target."""
    if dst.exists():
        return dst
    TMP.mkdir(exist_ok=True)
    comp = _components(gpu=gpu)
    src_emb = donor_embedding(donor, comp)

    cap_t, cap_d = cv2.VideoCapture(str(target)), cv2.VideoCapture(str(donor))
    if not cap_t.isOpened():
        raise RuntimeError(f"cannot open {target}")
    fps = cap_t.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap_t.get(cv2.CAP_PROP_FRAME_WIDTH))
    hh = int(cap_t.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wr = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, hh))

    # 3-stage pipeline: decoder thread -> GPU worker (detect/align/swap) ->
    # ordered paste+write on this thread. Keeps the GPU fed (decoder and
    # paste-back run concurrently with inference) and preserves frame order
    # via the pending dict even if stages finish out of order.
    import queue as _queue
    from threading import Thread

    q_raw: _queue.Queue = _queue.Queue(maxsize=6)
    q_gpu: _queue.Queue = _queue.Queue(maxsize=6)
    err: list = []

    def _decode() -> None:
        try:
            i = 0
            while i < n_frames:
                ok_t, ft = cap_t.read()
                if not ok_t:
                    break
                q_raw.put((i, ft))
                i += 1
        except Exception as e:  # noqa: BLE001
            err.append(e)
        finally:
            q_raw.put(None)
            cap_t.release()

    def _gpu() -> None:
        try:
            while True:
                item = q_raw.get()
                if item is None:
                    break
                i, ft = item
                det = comp["det"].largest_face(ft)
                swapped = M = None
                if det is not None:
                    _box, kps, _score = det
                    aimg, M = comp["swap"].align_target(ft, kps)
                    if aimg is not None:
                        swapped = comp["swap"].swap_face(aimg, src_emb)
                    else:
                        M = None
                q_gpu.put((i, ft, swapped, M))
        except Exception as e:  # noqa: BLE001
            err.append(e)
        finally:
            q_gpu.put(None)

    Thread(target=_decode, daemon=True).start()
    Thread(target=_gpu, daemon=True).start()

    n_written = n_ok = 0
    pending: dict = {}
    nxt = 0
    while n_written < n_frames:
        if err:
            break
        item = q_gpu.get()
        if item is None:
            break
        i, ft, swapped, M = item
        pending[i] = (ft, swapped, M)
        while nxt in pending:
            ft, swapped, M = pending.pop(nxt)
            if swapped is None:
                wr.write(ft)
            else:
                wr.write(comp["swap"].paste_back(ft, swapped, M))
                n_ok += 1
            n_written += 1
            nxt += 1

    wr.release()
    cap_d.release()
    if err:
        raise RuntimeError(f"nnswap pipeline failed: {err[0]!r}")
    print(f"  wrote {dst.name} ({n_ok}/{n_written} frames swapped, "
          f"backends: {backend_report()})")
    return dst


def main():
    ap = argparse.ArgumentParser(description="GPU neural face-swap deepfake generator")
    ap.add_argument("--frames", type=int, default=360)
    ap.add_argument("--target", type=str, default=None)
    ap.add_argument("--donor", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--no-gpu", action="store_true", help="force CPU onnxruntime")
    args = ap.parse_args()
    gpu = not args.no_gpu

    mon = GpuMonitor(interval=0.4).start(label="make_deepfakes")
    if args.target and args.donor and args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        make_nnswap_fake(Path(args.target), Path(args.donor), Path(args.out),
                         args.frames, gpu=gpu)
    else:
        videos = {s: ROOT / "Dataset" / f"subject_{s}" / "vid.avi" for s in (1, 2)}
        for name, tgt, don in (("fake_nnswap_s2toS1", videos[1], videos[2]),
                               ("fake_nnswap_s1toS2", videos[2], videos[1])):
            make_nnswap_fake(tgt, don, TMP / f"{name}_{args.frames}.avi",
                             args.frames, gpu=gpu)
    summary = mon.stop()
    mon.save_csv(ROOT / "output" / "gpu_make_deepfakes.csv")
    print("summary:", summary)


if __name__ == "__main__":
    main()

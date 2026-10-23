"""Reconstruct per-subject videos + ground truth from UBFC-rPPG-Faces.

The open Hugging Face mirror (Horusprg/UBFC-rPPG-Faces, 316 MB) stores
UBFC-rPPG DATASET_2 as ~1,500-2,050 mpeg4 clip files per subject, each a
5-frame stride-1 sliding window (128x128, ~29.26 fps) taken from the
original minute-long recording: clip N's first frame == original frame N.
One PPG sample per clip sits in metadata.csv.

This tool rebuilds the classic per-subject layout the rest of the repo
expected before Drive quota killed the raw download:

    UBFC_FACES/subject<N>/vid.avi          # original frames, 128x128 MJPG
    UBFC_FACES/subject<N>/ground_truth.txt # 2 rows x N cols like UBFC

Usage:
  python tools/build_ubfc_faces.py                 # all 42 subjects
  python tools/build_ubfc_faces.py --subjects 1 3 5
"""
import argparse
import struct
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / "tmp"
OUT_DEFAULT = ROOT / "UBFC_FACES"
META_URL = ("https://huggingface.co/datasets/Horusprg/UBFC-rPPG-Faces/"
            "resolve/main/metadata.csv")


def read_metadata(path: Path):
    """-> {subject_id: {clip_index: ppg}} parsed from metadata.csv."""
    import csv
    meta = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = row["clip_filename"]            # clip_subject3_000123.avi
            stem = name[len("clip_"):-len(".avi")]  # subject3_000123
            sid, idx = stem.split("_", 1)
            meta.setdefault(sid, {})[int(idx)] = float(row["ppg_signal"])
    return meta


def unpack_clip(zf, member: str):
    """Read one 5-frame clip archive member -> (fps, frames) via a temp file."""
    data = zf.read(member)
    tmp = TMP / "clip_tmp.avi"
    tmp.write_bytes(data)
    cap = cv2.VideoCapture(str(tmp))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open clip {member}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    tmp.unlink(missing_ok=True)
    if len(frames) != 5:
        raise RuntimeError(f"{member}: expected 5 frames, got {len(frames)}")
    return fps, frames


def clip_members_for(zf, subject: str):
    """Zip member names clip_subjectS_000000.avi.. in clip-index order."""
    pre = f"UBFC-rPPG-Faces/clip_{subject}_"
    found = {m: m for m in zf.namelist() if m.startswith(pre) and m.endswith(".avi")}
    return [found[i] for i in sorted(found, key=lambda m: int(m[len(pre):-len(".avi")]))]


def verify_stride(zf, members, probe: int = 30) -> float:
    """MAD between frame4 of clip i and frame3 of clip i+1 (should be ~0)."""
    mads = []
    for i in range(min(probe, len(members) - 1)):
        _, fr_i = unpack_clip(zf, members[i])
        _, fr_j = unpack_clip(zf, members[i + 1])
        mads.append(float(np.abs(
            fr_i[4].astype(float) - fr_j[3].astype(float)).mean()))
    return float(np.mean(mads))


def hr_from_ppg(ppg: np.ndarray, fps: float, bp=(0.75, 3.0), win_s: float = 10.0) -> np.ndarray:
    """Per-frame BPM from band-limited Welch on 10 s windows."""
    from scipy.signal import welch
    hr = np.full(len(ppg), np.nan)
    w = int(win_s * fps)
    for i in range(len(ppg)):
        seg = ppg[max(0, i - w):i + w]
        if len(seg) < w // 2:
            continue
        f, P = welch(seg, fs=fps, nperseg=min(256, len(seg)))
        band = (f >= bp[0]) & (f <= bp[1])
        if band.any():
            hr[i] = f[band][int(np.argmax(P[band]))] * 60.0
    return hr


def build_subject(zf, member_template_meta, subject: str, out_dir: Path,
                  crop_probe: int = 30):
    ppg_by_clip = member_template_meta  # {clip_index: ppg}
    members = clip_members_for(zf, subject)
    n_clips = len(members)
    stride_mad = verify_stride(zf, members, crop_probe)
    print(f"  {subject}: {n_clips} clips, stride check MAD {stride_mad:.3f}px"
          + ("  [WARN: not stride-1?]" if stride_mad > 4 else ""))

    fps, first = unpack_clip(zf, members[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    wr = cv2.VideoWriter(str(out_dir / "vid.avi"),
                         cv2.VideoWriter_fourcc(*"MJPG"), fps,
                         (first[0].shape[1], first[0].shape[0]))
    ppg = [ppg_by_clip[i] for i in sorted(ppg_by_clip)]
    for m in members:
        idx = int(m.split("_")[-1].split(".")[0])
        if idx not in ppg_by_clip:
            raise RuntimeError(f"{m} missing from metadata")
        _, frames = unpack_clip(zf, m)
        wr.write(frames[0])
    wr.release()
    ppg = np.asarray(ppg, dtype=float)
    hr = hr_from_ppg(ppg, fps)
    np.savetxt(out_dir / "ground_truth.txt", np.vstack([ppg, hr]))
    print(f"    wrote {out_dir/'vid.avi'} ({n_clips} frames @ {fps:.2f} fps), "
          f"GT hr mean {np.nanmean(hr):.1f} bpm")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", type=str, default=str(TMP / "UBFC-rPPG-Faces.zip"))
    ap.add_argument("--meta", type=str, default=None,
                    help="metadata.csv (downloaded automatically if missing)")
    ap.add_argument("--out", type=str, default=str(OUT_DEFAULT))
    ap.add_argument("--subjects", type=int, nargs="*", default=None,
                    help="subject numbers to build (default: all)")
    args = ap.parse_args()

    import zipfile
    import requests

    zip_path = Path(args.zip)
    if not zip_path.exists():
        raise SystemExit(f"missing {zip_path}; run tools/fetch_ubfc_faces.py first")
    meta_path = Path(args.meta) if args.meta else TMP / "ubfc_faces_metadata.csv"
    if not meta_path.exists():
        print(f"downloading {META_URL}")
        meta_path.write_bytes(requests.get(META_URL).content)
    meta = read_metadata(meta_path)

    subjects = args.subjects if args.subjects else \
        sorted({int(s[len("subject"):]) for s in meta})
    print(f"building {len(subjects)} subjects -> {args.out}")
    with zipfile.ZipFile(zip_path) as zf:
        for s in subjects:
            sid = f"subject{s}"
            if sid not in meta:
                print(f"  [skip] {sid} not in mirror")
                continue
            build_subject(zf, meta[sid], sid, Path(args.out) / sid)


if __name__ == "__main__":
    main()

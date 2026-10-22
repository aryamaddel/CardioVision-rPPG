"""
Diagnose GPU-landmark trace quality vs MediaPipe on a real clip.

Per backend (MediaPipe CPU reference, GPU raw, GPU One-Euro-smoothed):
  - landmark agreement with MediaPipe (mean/p95 px)
  - frame-to-frame jitter: RMS of the 2nd temporal difference (removes
    constant velocity head motion, leaves high-frequency noise)
  - POS pulse quality: |r| vs contact-PPG ground truth, SPR dB, coherence

Usage: python tools/diag_jitter.py [--frames 120] [--subject 1|2]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import welch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from deepfake_detect import (REGIONS, ROI_R, RegionExtractor, load_gt,  # noqa: E402
                             zscore, BP_LO, BP_HI)
from gpu_core import GPURegionExtractor, BaseRegionExtractor  # noqa: E402
from rppg import pos_algorithm  # noqa: E402


def jitter_rms(lms: np.ndarray) -> float:
    """RMS of second differences (px): 0 for constant-velocity motion."""
    d2 = lms[2:] - 2 * lms[1:-1] + lms[:-2]
    return float(np.sqrt((d2 ** 2).sum(axis=2)).mean())


def pulse_stats(traces, fps, gt_ppg):
    pulse = pos_algorithm(traces["full"], fps)
    n = min(len(pulse), len(gt_ppg))
    r = float(abs(np.corrcoef(zscore(pulse[:n]), zscore(gt_ppg[:n]))[0, 1]))
    f, P = welch(pulse, fs=fps, nperseg=min(256, len(pulse)))
    band = (f >= BP_LO) & (f <= BP_HI)
    Pb = P[band] + 1e-15
    spr = float(10 * np.log10(Pb.max() / np.median(Pb)))
    bpm = float(f[band][np.argmax(Pb)] * 60)
    reg = {k: pos_algorithm(traces[k], fps) for k in REGIONS}
    keys = list(reg)
    coh = [float(np.corrcoef(zscore(reg[keys[a]]), zscore(reg[keys[b]]))[0, 1])
           for a in range(len(keys)) for b in range(a + 1, len(keys))]
    return r, spr, bpm, float(np.mean(coh))


def collect(make_ex, video: Path, n_frames: int, gt_ppg):
    ex = make_ex()
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    lms, frames = [], []
    for i in range(n_frames):
        ok, fr = cap.read()
        if not ok:
            break
        lm = ex.landmarks(fr, int(i * 1000.0 / fps))
        if lm is not None:
            lms.append(lm)
            frames.append(fr)
    cap.release()
    ex.close()
    lms_a = np.stack(lms)
    # region traces from the collected (frame, landmark) pairs; region_rgb is
    # pure NumPy, so no landmarker model needed here
    ex2 = BaseRegionExtractor(regions=REGIONS, roi_r=ROI_R)
    traces = {k: [] for k in ["full", *REGIONS]}
    for fr, lm in zip(frames, lms_a):
        rr = ex2.region_rgb(fr, lm)
        for k in traces:
            traces[k].append(rr[k])
    traces = {k: np.array(v, dtype=np.float64) for k, v in traces.items()}
    r, spr, bpm, coh = pulse_stats(traces, fps, gt_ppg)
    return lms_a, r, spr, bpm, coh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--subject", type=int, default=2, choices=[1, 2])
    args = ap.parse_args()

    video = ROOT / "Dataset" / f"subject_{args.subject}" / "vid.avi"
    gt_ppg, _gt_hr = load_gt(ROOT / "Dataset" / f"subject_{args.subject}")
    n = args.frames
    print(f"== {video.name}, {n} frames, GT |r| reference ==")

    mp_lms, mp_r, mp_spr, mp_bpm, mp_coh = collect(
        lambda: RegionExtractor(), video, n, gt_ppg)
    gpu_raw_lms, raw_r, raw_spr, raw_bpm, raw_coh = collect(
        lambda: GPURegionExtractor(smooth=False), video, n, gt_ppg)
    gpu_1e_lms, s_r, s_spr, s_bpm, s_coh = collect(
        lambda: GPURegionExtractor(smooth=True), video, n, gt_ppg)

    def row(name, lms, r, spr, bpm, coh, ref=None):
        err = ""
        if ref is not None:
            k = min(len(lms), len(ref))
            e = np.linalg.norm(lms[:k] - ref[:k], axis=2)
            err = f"  err_vs_mp {e.mean():5.2f} (p95 {np.percentile(e, 95):5.2f}) px"
        print(f"{name:<12} jitter {jitter_rms(lms):5.3f}px{err}   "
              f"|r|GT {r:.2f}  SPR {spr:5.2f}dB  BPM {bpm:6.1f}  coh {coh:+.3f}")

    row("mediapipe", mp_lms, mp_r, mp_spr, mp_bpm, mp_coh)
    row("gpu_raw", gpu_raw_lms, raw_r, raw_spr, raw_bpm, raw_coh, ref=mp_lms)
    row("gpu_1e", gpu_1e_lms, s_r, s_spr, s_bpm, s_coh, ref=mp_lms)


if __name__ == "__main__":
    main()

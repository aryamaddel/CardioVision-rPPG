"""
rPPG-based deepfake detection experiment.

Idea (FakeCatcher, Ciftci & Demir TPAMI 2020; DeepFakesON-Phys,
Hernandez-Ortega et al. 2021): the blood-volume pulse leaves a tiny,
spatially coherent, periodic colour trace on real skin at the heart
rate. Generated/face-swapped faces have no blood flow, so the rPPG
signal is absent, non-periodic, or incoherent across facial regions.

Pipeline per clip:
  frames -> per-region ROI RGB traces (forehead, cheeks, full face)
  -> POS pulse per region
  -> features: spectral peak ratio (SPR), inter-region coherence,
     BPM estimation, correlation with ground-truth PPG (when available)
  -> verdict: REAL if periodic + coherent + physiological, else FAKE

Clips: the 2 real UBFC-rPPG subjects in Dataset/, plus synthetic fakes
we generate from them:
  - "smooth": temporal box-car over frames (mimics the pulsation-free,
     temporally incoherent texture of generated faces)
  - "swap":  donor face warped onto the target via landmarks + Poisson
     blending (classic naive face-swap deepfake)

Usage:
  python deepfake_detect.py                 # full run (360 frames)
  python deepfake_detect.py --frames 200    # shorter run
  python deepfake_detect.py --tau-spr 4 --tau-coh 0.35
"""
import argparse
import json
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from scipy.signal import welch

from rppg import _ensure_model, pos_algorithm, compute_bpm, BP_LO, BP_HI

ROOT = Path(__file__).resolve().parent
MODEL = str(ROOT / "face_landmarker.task")
TMP = ROOT / "tmp"
CACHE = ROOT / ".rppg_cache"
OUT = ROOT / "output"

# Small elliptical ROIs anchored on stable MediaPipe Face Mesh indices.
REGIONS = {
    "forehead": [10, 151, 108, 69, 337, 299],
    "right_cheek": [205, 50, 118, 101],
    "left_cheek": [425, 280, 348, 330],
}
ROI_R = 0.055  # ellipse semi-axis, fraction of face scale
SMOOTH_K = 9   # temporal box-car half-length for the "smooth" fake
SWAP_ANCHORS = [33, 263, 1, 61, 291]  # outer eyes, nose tip, mouth corners

# Verdict thresholds (calibrated on the clips in this repo, see output).
DEFAULT_TAU_SPR_DB = 4.0
DEFAULT_TAU_COH = 0.35
TAU_SPR_DB = DEFAULT_TAU_SPR_DB
TAU_COH = DEFAULT_TAU_COH
BPM_RANGE = (45.0, 140.0)


# ---------------------------------------------------------------- extraction
class RegionExtractor:
    """MediaPipe face landmarker -> mean RGB for small ROIs + full face."""

    def __init__(self, model_path: str = MODEL):
        _ensure_model(model_path)
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = mp_vision.FaceLandmarker.create_from_options(opts)

    def close(self):
        self.detector.close()

    def landmarks(self, frame_bgr, ts_ms):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self.detector.detect_for_video(mp_img, ts_ms)
        if not res.face_landmarks:
            return None
        h, w = frame_bgr.shape[:2]
        return np.array([[lk.x * w, lk.y * h] for lk in res.face_landmarks[0]],
                        dtype=np.float32)

    @staticmethod
    def _mean_bgr(frame, mask):
        px = frame[mask > 0]
        if len(px) < 10:
            return (np.nan, np.nan, np.nan)
        return (float(px[:, 2].mean()), float(px[:, 1].mean()), float(px[:, 0].mean()))  # R,G,B

    def region_rgb(self, frame, lm):
        from rppg import EXCLUDE
        h, w = frame.shape[:2]
        lm_i = lm.astype(np.int32)
        hull = cv2.convexHull(lm_i)
        scale = np.sqrt(max(cv2.contourArea(hull), 1.0))
        out = {}
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [hull], 255)
        for idx in EXCLUDE:
            cv2.fillPoly(mask, [lm_i[idx].reshape(1, 1, 2)], 0)
        out["full"] = self._mean_bgr(frame, mask)
        ax = max(4, int(ROI_R * scale))
        for name, idxs in REGIONS.items():
            c = tuple(lm[idxs].mean(axis=0).astype(int))
            m = np.zeros((h, w), np.uint8)
            cv2.ellipse(m, c, (ax, ax), 0, 0, 360, 255, -1)
            out[name] = self._mean_bgr(frame, m)
        return out


def extract_traces(video: Path, n_frames: int, fps_hint: float):
    """Return dict region -> (N,3) RGB arrays + fps. NaNs interpolated."""
    key = f"traces_{video.stem}_{video.stat().st_size}_{n_frames}"
    cache_file = CACHE / f"{key}.npz"
    if cache_file.exists():
        z = np.load(cache_file)
        return {k: z[k] for k in z.files if k != "fps"}, float(z["fps"])

    ex = RegionExtractor()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or fps_hint
    rows = {k: [] for k in ["full", *REGIONS]}
    t0 = cv2.getTickCount()
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        ts = int(i * 1000.0 / fps)
        lm = ex.landmarks(frame, ts)
        rr = ex.region_rgb(frame, lm) if lm is not None else None
        for k in rows:
            rows[k].append(rr[k] if rr else (np.nan,) * 3)
    cap.release()
    ex.close()
    n = len(rows["full"])
    dt = (cv2.getTickCount() - t0) / cv2.getTickFrequency()
    print(f"  extracted {n} frames from {video.name} in {dt:.1f}s ({n/max(dt,1e-6):.1f} fps)")

    traces = {}
    for k, v in rows.items():
        a = np.array(v, dtype=np.float64)
        for c in range(3):
            col = a[:, c]
            m = ~np.isnan(col)
            if m.sum() < 4:
                a[:, c] = 0.0
            elif not m.all():
                a[:, c] = np.interp(np.arange(len(col)), np.flatnonzero(m), col[m])
        traces[k] = a
    CACHE.mkdir(exist_ok=True)
    np.savez(cache_file, fps=fps, **traces)
    return traces, fps


# ------------------------------------------------------------------- fakes
def make_smooth_fake(src: Path, dst: Path, n_frames: int):
    if dst.exists():
        return dst
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); hh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wr = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, hh))
    buf = deque(maxlen=2 * SMOOTH_K + 1)
    for i in range(n_frames):
        ok, f = cap.read()
        if not ok:
            break
        buf.append(f.astype(np.float32))
        out = np.mean(buf, axis=0)  # box-car length == 2*SMOOTH_K+1 once warm
        wr.write(out.astype(np.uint8))
    wr.release(); cap.release()
    print(f"  wrote {dst}")
    return dst


def make_swap_fake(target: Path, donor: Path, dst: Path, n_frames: int):
    """Warp donor face texture onto target (landmark align + Poisson clone)."""
    if dst.exists():
        return dst
    cap_t, cap_d = cv2.VideoCapture(str(target)), cv2.VideoCapture(str(donor))
    fps = cap_t.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap_t.get(cv2.CAP_PROP_FRAME_WIDTH)); hh = int(cap_t.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wr = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, hh))
    ex_t, ex_d = RegionExtractor(), RegionExtractor()
    kern = np.ones((3, 3), np.uint8)
    for i in range(n_frames):
        ok_t, ft = cap_t.read(); ok_d, fd = cap_d.read()
        if not (ok_t and ok_d):
            break
        ts = int(i * 1000.0 / fps)
        lmt, lmd = ex_t.landmarks(ft, ts), ex_d.landmarks(fd, ts)
        if lmt is None or lmd is None:
            wr.write(ft); continue
        M, _ = cv2.estimateAffinePartial2D(lmd[SWAP_ANCHORS], lmt[SWAP_ANCHORS])
        if M is None:
            wr.write(ft); continue
        warped = cv2.warpAffine(fd, M, (w, hh))
        mask = np.zeros((hh, w), np.uint8)
        cv2.fillPoly(mask, [cv2.convexHull(lmt.astype(np.int32))], 255)
        mask = cv2.erode(mask, kern, iterations=3)
        center = tuple(np.column_stack(np.nonzero(mask)).mean(axis=0)[::-1].astype(int))
        try:
            out = cv2.seamlessClone(warped, ft, mask, center, cv2.NORMAL_CLONE)
        except cv2.error:
            out = ft
        wr.write(out)
    wr.release(); cap_t.release(); cap_d.release()
    ex_t.close(); ex_d.close()
    print(f"  wrote {dst}")
    return dst


# ---------------------------------------------------------------- features
def zscore(x):
    s = x.std()
    return (x - x.mean()) / (s if s > 1e-12 else 1.0)


def clip_features(traces, fps, gt_ppg=None):
    pulse = pos_algorithm(traces["full"], fps)
    n = len(pulse)
    if n < 60:
        return {"pulse": pulse, "freqs": np.array([]), "psd": np.array([]),
                "spr_db": float("nan"), "coherence": float("nan"), "bpm": float("nan"),
                "bpm_spec": float("nan"), "corr_gt": None,
                "verdict": "INSUFFICIENT_FRAMES"}
    f, P = welch(pulse, fs=fps, nperseg=min(256, len(pulse)))
    band = (f >= BP_LO) & (f <= BP_HI)
    Pb = P[band] + 1e-15
    i_pk = int(np.argmax(Pb))
    spr_db = float(10 * np.log10(Pb[i_pk] / np.median(Pb)))
    bpm_spec = float(f[band][i_pk] * 60.0)
    bpm = compute_bpm(pulse, fps) or bpm_spec

    reg_pulse = {k: pos_algorithm(traces[k], fps) for k in REGIONS}
    keys = list(reg_pulse)
    coh = []
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            xa, xb = zscore(reg_pulse[keys[a]]), zscore(reg_pulse[keys[b]])
            coh.append(float(np.corrcoef(xa, xb)[0, 1]))
    coherence = float(np.mean(coh))

    corr_gt = None
    if gt_ppg is not None and len(gt_ppg) >= len(pulse):
        g = zscore(gt_ppg[:len(pulse)])
        corr_gt = float(abs(np.corrcoef(zscore(pulse), g)[0, 1]))

    plausible = BPM_RANGE[0] <= bpm <= BPM_RANGE[1]
    real = (spr_db >= TAU_SPR_DB) and (coherence >= TAU_COH) and plausible
    return {
        "pulse": pulse, "freqs": f, "psd": P,
        "spr_db": spr_db, "coherence": coherence, "bpm": bpm,
        "bpm_spec": bpm_spec, "corr_gt": corr_gt,
        "verdict": "REAL" if real else "FAKE",
    }


def load_gt(subject_dir: Path):
    g = np.loadtxt(subject_dir / "ground_truth.txt")
    return g[0], g[1]  # ppg waveform, hr per frame


# --------------------------------------------------------------------- main
def main():
    global TAU_SPR_DB, TAU_COH
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=360)
    ap.add_argument("--tau-spr", type=float, default=DEFAULT_TAU_SPR_DB)
    ap.add_argument("--tau-coh", type=float, default=DEFAULT_TAU_COH)
    ap.add_argument("--ai-glob", type=str, default=None,
                    help="extra AI-generated clips to evaluate, e.g. 'tmp/ai/*.mp4'")
    args = ap.parse_args()
    TAU_SPR_DB, TAU_COH = args.tau_spr, args.tau_coh

    TMP.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    videos = {s: ROOT / "Dataset" / f"subject_{s}" / "vid.avi" for s in (1, 2)}

    print("== generating fake clips (if missing) ==")
    fakes = {
        "fake_smooth_s1": make_smooth_fake(videos[1], TMP / f"fake_smooth_s1_{args.frames}.avi", args.frames),
        "fake_smooth_s2": make_smooth_fake(videos[2], TMP / f"fake_smooth_s2_{args.frames}.avi", args.frames),
        "fake_swap_s2toS1": make_swap_fake(videos[1], videos[2], TMP / f"fake_swap_s2toS1_{args.frames}.avi", args.frames),
        "fake_swap_s1toS2": make_swap_fake(videos[2], videos[1], TMP / f"fake_swap_s1toS2_{args.frames}.avi", args.frames),
    }

    clips = [
        ("real_s1", videos[1], 1), ("real_s2", videos[2], 2),
        *[ (k, v, None) for k, v in fakes.items() ],
    ]
    if args.ai_glob:
        import glob as _glob
        for p in sorted(_glob.glob(str(ROOT / args.ai_glob))):
            clips.append((f"ai_{Path(p).stem}", Path(p), None))

    print("== extracting rPPG traces ==")
    results = {}
    for name, path, subj in clips:
        traces, fps = extract_traces(path, args.frames, 30.0)
        gt = None
        if subj is not None:
            gt_ppg, gt_hr = load_gt(ROOT / "Dataset" / f"subject_{subj}")
            gt = gt_ppg
        else:
            gt_ppg = gt_hr = None
        results[name] = clip_features(traces, fps, gt)
        results[name]["fps"] = fps

    print("\n== results ==")
    hdr = f"{'clip':<18} {'SPR(dB)':>8} {'coher.':>7} {'BPM':>7} {'GT BPM':>7} {'|r| GT':>7}  verdict"
    print(hdr); print("-" * len(hdr))
    for name, r in results.items():
        gt_hr_txt = "--"
        if name.startswith("real_s"):
            s = name[-1]
            _, gt_hr = load_gt(ROOT / "Dataset" / f"subject_{s}")
            gt_hr_txt = f"{np.mean(gt_hr[:args.frames]):.1f}"
        rgt = "--" if r["corr_gt"] is None else f"{r['corr_gt']:.2f}"
        print(f"{name:<18} {r['spr_db']:>8.2f} {r['coherence']:>7.3f} {r['bpm']:>7.1f} {gt_hr_txt:>7} {rgt:>7}  {r['verdict']}")

    (OUT / "deepfake_ppg_results.json").write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if kk not in ("pulse", "freqs", "psd")}
         for k, v in results.items()}, indent=2))

    # ---- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))

    r1 = results["real_s1"]
    t = np.arange(len(r1["pulse"])) / r1["fps"]
    gt_ppg, _ = load_gt(ROOT / "Dataset" / "subject_1")
    ax[0, 0].plot(t, zscore(r1["pulse"]), label="rPPG (POS), real_s1")
    ax[0, 0].plot(t, zscore(gt_ppg[:len(t)]), alpha=0.7, label="contact PPG (GT)")
    ax[0, 0].set_title("Real subject_1: rPPG pulse vs ground truth")
    ax[0, 0].legend(); ax[0, 0].set_xlabel("s")

    for name, style in [("real_s1", "-"), ("fake_smooth_s1", "--"), ("fake_swap_s2toS1", ":")]:
        r = results[name]
        b = (r["freqs"] >= BP_LO) & (r["freqs"] <= BP_HI)
        ax[0, 1].plot(r["freqs"][b] * 60, zscore(r["psd"][b]), style, label=name)
    ax[0, 1].set_title("Pulse PSD (HR band)"); ax[0, 1].legend(); ax[0, 1].set_xlabel("BPM")

    names = list(results)
    sprs = [results[n]["spr_db"] for n in names]
    cohs = [results[n]["coherence"] for n in names]
    def _col(n):
        if n.startswith("real"):
            return "tab:blue"
        if n.startswith("ai_"):
            return "tab:orange"
        return "tab:red"
    cols = [_col(n) for n in names]
    ax[1, 0].bar(range(len(names)), sprs, color=cols)
    ax[1, 0].axhline(TAU_SPR_DB, color="k", ls="--", lw=1, label=f"tau={TAU_SPR_DB}")
    ax[1, 0].set_xticks(range(len(names))); ax[1, 0].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax[1, 0].set_title("Spectral peak ratio (dB)"); ax[1, 0].legend()
    ax[1, 1].bar(range(len(names)), cohs, color=cols)
    ax[1, 1].axhline(TAU_COH, color="k", ls="--", lw=1, label=f"tau={TAU_COH}")
    ax[1, 1].set_xticks(range(len(names))); ax[1, 1].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax[1, 1].set_title("Inter-region pulse coherence"); ax[1, 1].legend()

    fig.suptitle("rPPG deepfake detection: real skin pulsates coherently, fakes do not")
    fig.tight_layout()
    fig.savefig(OUT / "deepfake_ppg_demo.png", dpi=110)
    print(f"\nwrote {OUT / 'deepfake_ppg_results.json'} and {OUT / 'deepfake_ppg_demo.png'}")


if __name__ == "__main__":
    main()

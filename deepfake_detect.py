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

Clips: every UBFC-rPPG subject found in Dataset/ and UBFC/DATASET_2/
(the official full 42-subject benchmark), plus synthetic fakes we
generate from Dataset/subject_1 + subject_2:
  - "smooth": temporal box-car over frames (mimics the pulsation-free,
     temporally incoherent texture of generated faces)
  - "swap":  donor face warped onto the target via landmarks + Poisson
     blending (classic naive face-swap deepfake)
  - "nnswap": inswapper_128 neural identity swap generated on the GPU
    (modern high-quality deepfake: SCRFD align -> arcface identity ->
    generator -> feathered paste-back; see make_deepfakes.py)

Extraction backends (gpu_core.py):
  - gpu:       SCRFD + face-mesh onnx on onnxruntime-gpu CUDA, MediaPipe-
    compatible landmarks (~6-9 px agreement + One-Euro temporal smoothing;
    MediaPipe's Windows wheel ships with GPU disabled, so this is the GPU
    path)
  - mediapipe: original CPU pipeline (reference / fallback)

Usage:
  python deepfake_detect.py                 # all real subjects, 12 s each, GPU auto
  python deepfake_detect.py --extractor mediapipe   # CPU reference run
  python deepfake_detect.py --duration 20   # longer equal-duration windows
  python deepfake_detect.py --frames 200    # legacy: fixed frame count instead
  python deepfake_detect.py --tau-spr 4 --tau-coh 0.35
  python deepfake_detect.py --ai-glob "tmp/ai/*.mp4"

Equal-duration protocol: every clip (real and fake) is evaluated on the
first `--duration` seconds of video (round(duration * fps) frames at the
clip's own fps), so all features come from identical time windows even
though the source videos have slightly different frame rates.
"""
import argparse
import json
import queue
import threading
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from scipy.signal import welch

from rppg import _ensure_model, pos_algorithm, compute_bpm, BP_LO, BP_HI
from gpu_core import (GPURegionExtractor, GpuMonitor, BaseRegionExtractor,
                      backend_report, cuda_available, MODELS, BUFFALO)

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
# Ellipse semi-axis, fraction of face scale. 0.08 (not the original 0.055) was
# chosen by the ROI ablation in tools/exp_gpu_params.py: bigger patches average
# over more skin, which tolerates the GPU path's ~5 px landmark error. At the
# shipped thresholds this lifts GPU separation from +0.03 to +0.22 (12/12),
# and lifts MediaPipe too (+0.08 -> +0.15) -- the change is backend-neutral,
# re-validated out-of-sample on 4 held-out clips (tools/exp_final.py).
ROI_R = 0.08
SMOOTH_K = 9   # temporal box-car half-length for the "smooth" fake
SWAP_ANCHORS = [33, 263, 1, 61, 291]  # outer eyes, nose tip, mouth corners

# Verdict thresholds (calibrated on the clips in this repo, see output).
DEFAULT_TAU_SPR_DB = 4.0
DEFAULT_TAU_COH = 0.35
TAU_SPR_DB = DEFAULT_TAU_SPR_DB
TAU_COH = DEFAULT_TAU_COH
BPM_RANGE = (45.0, 140.0)


# ---------------------------------------------------------------- extraction
def discover_real_subjects(max_subjects: int | None = None):
    """Find every real UBFC-rPPG subject available locally.

    Scans Dataset/ (the repo's own two subjects), UBFC/DATASET_2/ (the
    full 42-subject benchmark, when present) and UBFC_FACES/ (videos
    rebuilt by tools/build_ubfc_faces.py). Returns {label: subject_dir},
    where label is e.g. "real_s3" and each subject_dir holds vid.avi +
    ground_truth.txt. Duplicates across roots (same normalized name and
    same byte size) are dropped, so copied subjects are not evaluated
    twice.
    """
    roots = [ROOT / "Dataset", ROOT / "UBFC" / "DATASET_2", ROOT / "UBFC_FACES"]
    found: dict[str, Path] = {}
    seen_sizes: set[tuple[str, int]] = set()
    for root in roots:
        if not root.exists():
            continue
        for d in sorted(root.glob("subject*")):
            if not (d / "vid.avi").exists() or not (d / "ground_truth.txt").exists():
                continue
            num = "".join(ch for ch in d.name if ch.isdigit())
            vid_size = (d / "vid.avi").stat().st_size
            key = (f"subject{num}", vid_size)
            if key in seen_sizes:
                continue  # same recording already discovered in an earlier root
            seen_sizes.add(key)
            label = f"real_s{num}"
            if label in found:  # two distinct recordings share a subject number
                i = 2
                while f"{label}_{i}" in found:
                    i += 1
                label = f"{label}_{i}"
            found[label] = d
            if max_subjects and len(found) >= max_subjects:
                return found
    return found


class RegionExtractor(BaseRegionExtractor):
    """MediaPipe face landmarker (CPU) -> mean RGB for small ROIs + full face."""

    def __init__(self, model_path: str = MODEL):
        super().__init__(regions=REGIONS, roi_r=ROI_R)
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


def extract_traces(video: Path, duration: float, fps_hint: float,
                   tag: str = "mediapipe", make_extractor=None,
                   max_frames: int | None = None):
    """Return dict region -> (N,3) RGB arrays + fps. NaNs interpolated.

    Equal-duration protocol: reads round(duration*fps) frames from frame 0
    (or `max_frames` frames when set), so every clip contributes the same
    number of seconds regardless of its native frame rate.

    `tag` keys the cache per extractor backend (GPU landmarks track MediaPipe
    to ~5 px but are not bit-identical, so caches must not mix).
    """
    key = f"traces_{video.stem}_{video.stat().st_size}_d{duration:.2f}_f{max_frames or 0}_{tag}"
    cache_file = CACHE / f"{key}.npz"
    if cache_file.exists():
        z = np.load(cache_file)
        return {k: z[k] for k in z.files if k != "fps"}, float(z["fps"])

    ex = (make_extractor or (lambda: RegionExtractor()))()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or fps_hint
    n_frames = int(round(duration * fps)) if max_frames is None else max_frames
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    if total and n_frames > total:
        n_frames = int(total)
        print(f"  [warn] {video.name}: shorter than {duration}s "
              f"({n_frames} frames) - using the whole clip")
    rows = {k: [] for k in ["full", *REGIONS]}
    t0 = cv2.getTickCount()

    # 3-stage pipeline: decode (CPU) -> landmarks (GPU/CPU) -> ROI means (CPU),
    # so the landmark backend never waits on video decode (keeps GPU fed).
    q1: queue.Queue = queue.Queue(maxsize=8)
    q2: queue.Queue = queue.Queue(maxsize=8)
    err: list = []

    def _decode():
        try:
            for i in range(n_frames):
                ok, frame = cap.read()
                if not ok:
                    break
                q1.put((i, frame))
            q1.put(None)
        except BaseException as e:  # noqa: BLE001
            err.append(e)
            q1.put(None)
            q2.put(None)

    def _infer():
        try:
            while True:
                item = q1.get()
                if item is None:
                    q2.put(None)
                    return
                i, frame = item
                lm = ex.landmarks(frame, int(i * 1000.0 / fps))
                q2.put((frame, lm))
        except BaseException as e:  # noqa: BLE001
            err.append(e)
            q2.put(None)

    th_d = threading.Thread(target=_decode, daemon=True)
    th_i = threading.Thread(target=_infer, daemon=True)
    th_d.start()
    th_i.start()
    while True:
        item = q2.get()
        if item is None:
            break
        frame, lm = item
        rr = ex.region_rgb(frame, lm) if lm is not None else None
        for k in rows:
            rows[k].append(rr[k] if rr else (np.nan,) * 3)
    th_d.join()
    th_i.join()
    if err:
        raise err[0]
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
def frames_for_duration(video: Path, duration: float) -> int:
    """Frame count covering `duration` seconds of `video` at its own fps."""
    cap = cv2.VideoCapture(str(video))
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
    else:
        fps = 30.0
    return int(round(duration * fps))


def make_smooth_fake(src: Path, dst: Path, n_frames: int, k: int = SMOOTH_K):
    """Temporal box-car average (half-length `k`): kills pulse without any
    generative model, so it doubles as a controlled fake-strength dial."""
    if dst.exists():
        return dst
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); hh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wr = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, hh))
    buf = deque(maxlen=2 * k + 1)
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
DEFAULT_DURATION = 12.0   # seconds per clip in the equal-duration protocol


def main():
    global TAU_SPR_DB, TAU_COH
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=DEFAULT_DURATION,
                    help="seconds read from frame 0 of every clip (equal-duration protocol)")
    ap.add_argument("--frames", type=int, default=None,
                    help="legacy fixed frame count per clip (overrides --duration)")
    ap.add_argument("--max-real", type=int, default=None,
                    help="cap the number of real subjects (default: all found)")
    ap.add_argument("--tau-spr", type=float, default=DEFAULT_TAU_SPR_DB)
    ap.add_argument("--tau-coh", type=float, default=DEFAULT_TAU_COH)
    ap.add_argument("--ai-glob", type=str, default=None,
                    help="extra AI-generated clips to evaluate, e.g. 'tmp/ai/*.mp4'")
    ap.add_argument("--extractor", choices=["auto", "gpu", "mediapipe"], default="mediapipe",
                    help="landmark backend: GPU onnx (curated on full-frame clips) "
                         "or MediaPipe CPU (reference; also the only validated "
                         "option on 128x128 UBFC-Faces crops)")
    args = ap.parse_args()
    TAU_SPR_DB, TAU_COH = args.tau_spr, args.tau_coh

    TMP.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    reals = discover_real_subjects(args.max_real)
    if not reals:
        raise SystemExit("no real subjects found (expected Dataset/subject_* "
                         "with vid.avi + ground_truth.txt)")
    print(f"== real subjects: {len(reals)} ==", ", ".join(reals))
    # Fakes must be generated at the same resolution as the reals they are
    # compared against: prefer donors from the UBFC-Faces crops (the mirror)
    # and fall back to the repo's two full-frame subjects.
    crop_v1 = ROOT / "UBFC_FACES" / "subject1" / "vid.avi"
    crop_v3 = ROOT / "UBFC_FACES" / "subject3" / "vid.avi"
    if crop_v1.exists() and crop_v3.exists():
        videos = {1: crop_v1, 2: crop_v3}   # donors 's1'/'s2' = subjects 1 and 3
    else:
        videos = {1: ROOT / "Dataset" / "subject_1" / "vid.avi",
                  2: ROOT / "Dataset" / "subject_2" / "vid.avi"}
    print(f"== fake donors: {videos[1].parent.name}, {videos[2].parent.name} ==")
    tag_dur = f"{args.duration:.0f}s"  # fake filenames key on the time window

    gpu_ok = (cuda_available() and (MODELS / "mesh.onnx").exists()
              and (BUFFALO / "det_10g.onnx").exists())
    if args.extractor == "gpu" or (args.extractor == "auto" and gpu_ok):
        kind = "gpu"
        make_ex = lambda: GPURegionExtractor(regions=REGIONS, roi_r=ROI_R)  # noqa: E731
    else:
        kind = "mediapipe"
        make_ex = lambda: RegionExtractor()  # noqa: E731
    print(f"== extractor: {kind} ==")
    if kind == "gpu" and (ROOT / "UBFC_FACES" / "subject1").exists():
        print("  [warn] UBFC_FACES crops found: the GPU mesh crop fit was "
              "calibrated on full-frame clips and is ~51 px off on 128x128 "
              "face-fill crops - use --extractor mediapipe for those.")

    print("== generating fake clips (if missing) ==")
    gen_mon = GpuMonitor(interval=0.4).start(label="generate_fakes")
    # equal duration: each generated fake covers the same seconds of its
    # source video (frames = round(duration * source fps))
    fakes = {
        "fake_smooth_s1": make_smooth_fake(videos[1], TMP / f"fake_smooth_s1_{tag_dur}.avi",
                                           frames_for_duration(videos[1], args.duration)),
        "fake_smooth_s2": make_smooth_fake(videos[2], TMP / f"fake_smooth_s2_{tag_dur}.avi",
                                           frames_for_duration(videos[2], args.duration)),
        "fake_smooth_k2_s1": make_smooth_fake(videos[1], TMP / f"fake_smooth_k2_s1_{tag_dur}.avi",
                                              frames_for_duration(videos[1], args.duration), k=2),
        "fake_smooth_k15_s2": make_smooth_fake(videos[2], TMP / f"fake_smooth_k15_s2_{tag_dur}.avi",
                                               frames_for_duration(videos[2], args.duration), k=15),
        "fake_swap_s2toS1": make_swap_fake(videos[1], videos[2], TMP / f"fake_swap_s2toS1_{tag_dur}.avi",
                                           frames_for_duration(videos[1], args.duration)),
        "fake_swap_s1toS2": make_swap_fake(videos[2], videos[1], TMP / f"fake_swap_s1toS2_{tag_dur}.avi",
                                           frames_for_duration(videos[2], args.duration)),
    }
    try:
        from make_deepfakes import make_nnswap_fake
        # generation always prefers CUDA (independent of the extraction backend)
        fakes["fake_nnswap_s2toS1"] = make_nnswap_fake(
            videos[1], videos[2], TMP / f"fake_nnswap_s2toS1_{tag_dur}.avi",
            frames_for_duration(videos[1], args.duration), gpu=True)
        fakes["fake_nnswap_s1toS2"] = make_nnswap_fake(
            videos[2], videos[1], TMP / f"fake_nnswap_s1toS2_{tag_dur}.avi",
            frames_for_duration(videos[2], args.duration), gpu=True)
        # neural swap -> smooth chains: the strongest fakes in the set, i.e.
        # "what if a generator stacks temporal smoothing on the identity swap"
        for d, s in (("s2toS1", "fake_nnswap_s2toS1"), ("s1toS2", "fake_nnswap_s1toS2")):
            if s in fakes:
                fakes[f"fake_nnsmooth_{d}"] = make_smooth_fake(
                    fakes[s], TMP / f"fake_nnsmooth_{d}_{tag_dur}.avi",
                    frames_for_duration(fakes[s], args.duration))
    except Exception as exc:  # noqa: BLE001 - classic fakes still testable
        print(f"  [warn] neural swap clips skipped: {exc}")
    gen_stats = gen_mon.stop()
    gen_mon.save_csv(OUT / "gpu_generation.csv")

    clips = [(name, d / "vid.avi", name) for name, d in reals.items()]
    clips += [(k, v, None) for k, v in fakes.items()]
    if args.ai_glob:
        import glob as _glob
        for p in sorted(_glob.glob(str(ROOT / args.ai_glob))):
            clips.append((f"ai_{Path(p).stem}", Path(p), None))

    print("== extracting rPPG traces ==")
    ex_mon = GpuMonitor(interval=0.5).start(label=f"extraction[{kind}]")
    # cache tag: gpu landmarks are One-Euro-smoothed with the absolute-crop
    # joint fit (v3); r08 = the shipped ROI_R=0.08. Both changed every trace,
    # so older caches (gpu_1e3 / plain mediapipe at r=0.055) must not mix.
    tag = f"{kind}_1e3r08d" if kind == "gpu" else f"{kind}_r08d"
    results = {}
    for name, path, label in clips:
        traces, fps = extract_traces(path, args.duration, 30.0, tag=tag,
                                     make_extractor=make_ex,
                                     max_frames=args.frames)
        gt = None
        if label is not None and label.startswith("real_"):
            gt_ppg, _ = load_gt(reals[label])
            gt = gt_ppg
        results[name] = clip_features(traces, fps, gt)
        results[name]["fps"] = fps
    ext_stats = ex_mon.stop()
    ex_mon.save_csv(OUT / f"gpu_extraction_{kind}.csv")

    print("\n== results ==")
    hdr = f"{'clip':<12} {'SPR(dB)':>8} {'coher.':>7} {'BPM':>7} {'GT BPM':>7} {'|r| GT':>7}  verdict"
    print(hdr); print("-" * len(hdr))
    for name, r in results.items():
        gt_hr_txt = "--"
        if name in reals:
            n = int(round(args.duration * r["fps"]))
            _, gt_hr = load_gt(reals[name])
            gt_hr_txt = f"{np.mean(gt_hr[:n]):.1f}"
        rgt = "--" if r["corr_gt"] is None else f"{r['corr_gt']:.2f}"
        print(f"{name:<12} {r['spr_db']:>8.2f} {r['coherence']:>7.3f} {r['bpm']:>7.1f} {gt_hr_txt:>7} {rgt:>7}  {r['verdict']}")

    n_real = sum(1 for n in results if n in reals)
    real_spr = [results[n]["spr_db"] for n in results if n in reals]
    fake_spr = [results[n]["spr_db"] for n in results if n not in reals]
    print(f"\nreals: {n_real}  mean SPR {np.mean(real_spr):.2f} dB (min {np.min(real_spr):.2f})"
          if real_spr else "\nno reals")
    if fake_spr:
        print(f"fakes: {len(fake_spr)}  mean SPR {np.mean(fake_spr):.2f} dB (max {np.max(fake_spr):.2f})")
        margin = np.min(real_spr) - np.max(fake_spr) if real_spr else float("nan")
        print(f"separation margin (min real - max fake): {margin:+.2f} dB")

    def plausible(r):
        bpm = r.get("bpm", float("nan"))
        return BPM_RANGE[0] <= bpm <= BPM_RANGE[1]

    ok_r = [n for n in results if n in reals]
    ok_f = [n for n in results if n not in reals]
    if ok_r and ok_f:
        print(f"\n== real/fake confusion vs tau-coh (tau-spr {TAU_SPR_DB:g} dB) ==")
        print("  tau-coh   acc   TPR(real)  FPR(fake)")
        for tc in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
            pr = [n for n in ok_r if results[n]["spr_db"] >= TAU_SPR_DB
                  and results[n]["coherence"] >= tc and plausible(results[n])]
            pf = [n for n in ok_f if results[n]["spr_db"] >= TAU_SPR_DB
                  and results[n]["coherence"] >= tc and plausible(results[n])]
            tpr = len(pr) / len(ok_r)
            fpr = len([n for n in ok_f if n in pf]) / len(ok_f)
            acc = (len(pr) + (len(ok_f) - len(pf))) / (len(ok_r) + len(ok_f))
            print(f"   {tc:.2f}   {acc:.2f}    {tpr:.2f}       {fpr:.2f}")

    payload = {k: {kk: vv for kk, vv in v.items() if kk not in ("pulse", "freqs", "psd")}
               for k, v in results.items()}
    payload["_meta"] = {
        "extractor": kind,
        "real_subjects": {k: str(v) for k, v in reals.items()},
        "n_real": n_real,
        "gpu_backends": backend_report(),
        "gpu_stages": [gen_stats, ext_stats],
        "duration_s": args.duration,
        "frames": args.frames,
        "tau_spr_db": TAU_SPR_DB,
        "tau_coh": TAU_COH,
    }
    dur_tag = f"_{args.duration:.0f}s" if args.frames is None else \
        f"_{args.frames}f"
    (OUT / f"deepfake_ppg_results_{kind}{dur_tag}.json").write_text(
        json.dumps(payload, indent=2))

    # ---- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))

    first_real_name = next((n for n in results if n in reals), None)
    if first_real_name is None:
        raise SystemExit("no real clip evaluated - cannot draw demo figure")
    r1 = results[first_real_name]
    t = np.arange(len(r1["pulse"])) / r1["fps"]
    gt_ppg, _ = load_gt(reals[first_real_name])
    ax[0, 0].plot(t, zscore(r1["pulse"]), label=f"rPPG (POS), {first_real_name}")
    ax[0, 0].plot(t, zscore(gt_ppg[:len(t)]), alpha=0.7, label="contact PPG (GT)")
    ax[0, 0].set_title(f"{first_real_name}: rPPG pulse vs ground truth")
    ax[0, 0].legend(); ax[0, 0].set_xlabel("s")

    for name, style in [(first_real_name, "-"), ("fake_smooth_s1", "--"),
                        ("fake_swap_s2toS1", ":"), ("fake_nnswap_s2toS1", "-.")]:
        if name not in results or not results[name]["freqs"].size:
            continue
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
    fig.savefig(OUT / f"deepfake_ppg_demo_{kind}{dur_tag}.png", dpi=110)
    print(f"\nwrote {OUT / f'deepfake_ppg_results_{kind}{dur_tag}.json'} "
          f"and {OUT / f'deepfake_ppg_demo_{kind}{dur_tag}.png'}")
    print(f"extractor={kind} | onnx backends: {backend_report()}")


if __name__ == "__main__":
    main()

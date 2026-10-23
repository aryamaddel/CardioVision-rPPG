"""Feature-side experiments: can a better coherence estimator close the GPU gap?

The GPU path's ONLY failure is coherence on real_s1 (0.228 < tau 0.35 while the
MediaPipe reference sits at 0.372). Everything SPR/BPM-side already agrees.
This script reuses the cached traces (no GPU re-run) and sweeps:

  A1 bp        bandpass region pulses (BP_LO..BP_HI) before Pearson
  A2 msc       Welch magnitude-squared spectral coherence, band-averaged
  A3 medpair   median of the 3 region-pair correlations (drops a bad pair)
  A4 best2     mean of the best 2 of 3 pairs
  A5 sm5       5-frame box smoothing of the raw RGB traces before POS
  A6 sm5_best2 A5 + A4

plus C1: windowed re-evaluation (90 / 60-frame windows) -> n=32 / n=48 tests,
with Wilson 95% CIs and majority-vote per clip.

Verdicts reuse baseline SPR/BPM (>= tau_spr, plausible BPM) with the variant
coherence at tau_coh = 0.35, so the comparison isolates the coherence estimator.

Run:  uv run python tools/exp_variants.py
"""
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, welch as swelch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepfake_detect import (CACHE, REGIONS, ROOT, clip_features, load_gt,
                             TAU_COH, TAU_SPR_DB, BPM_RANGE)
from rppg import pos_algorithm, BP_LO, BP_HI

TAGS = {"gpu": "gpu_1e3", "mediapipe": "mediapipe"}
WINS = (90, 60)


def boxcar(a, half):
    if half <= 0:
        return a
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(a, size=2 * half + 1, axis=0, mode="nearest")


def bandpass(x, fps):
    nyq = fps / 2.0
    b, a = butter(4, [max(BP_LO / nyq, 1e-6), min(BP_HI / nyq, 0.99)],
                  btype="band")
    return filtfilt(b, a, x)


def pair_stats(reg_pulse):
    keys = list(reg_pulse)
    rs = [float(np.corrcoef(z(reg_pulse[x]), z(reg_pulse[y]))[0, 1])
          for x, y in combinations(keys, 2)]
    return rs


def z(x):
    s = x.std()
    return (x - x.mean()) / (s if s > 1e-12 else 1.0)


def coh_msc(reg_pulse, fps):
    keys = list(reg_pulse)
    vals = []
    for x, y in combinations(keys, 2):
        f, C = msc(reg_pulse[x], reg_pulse[y], fps)
        band = (f >= BP_LO) & (f <= BP_HI)
        vals.append(float(np.mean(C[band])))
    return float(np.mean(vals))


def msc(x, y, fps):
    from scipy.signal import csd
    nper = min(128, len(x))
    f, Pxy = csd(x, y, fs=fps, nperseg=nper)
    _, Pxx = csd(x, x, fs=fps, nperseg=nper)
    _, Pyy = csd(y, y, fs=fps, nperseg=nper)
    C = np.abs(Pxy) ** 2 / (Pxx * Pyy + 1e-30)
    return f, C


def variants(traces, fps, base):
    """name -> coherence value for one clip."""
    out = {}
    rp = {k: pos_algorithm(traces[k], fps) for k in REGIONS}

    rs = pair_stats(rp)
    out["baseline"] = float(np.mean(rs))
    out["medpair"] = float(np.median(rs))
    out["best2"] = float(np.mean(sorted(rs)[1:]))

    rpb = {k: bandpass(v, fps) for k, v in rp.items()}
    out["bp"] = float(np.mean(pair_stats(rpb)))
    out["bp_best2"] = float(np.mean(sorted(pair_stats(rpb))[1:]))

    out["msc"] = coh_msc(rp, fps)

    rp5 = {k: pos_algorithm(boxcar(traces[k], 2), fps) for k in REGIONS}
    rs5 = pair_stats(rp5)
    out["sm5"] = float(np.mean(rs5))
    out["sm5_best2"] = float(np.mean(sorted(rs5)[1:]))
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def enumerate_clips(tag):
    """name -> npz path for one backend tag."""
    size2subj = {}
    for s in (1, 2):
        p = ROOT / "Dataset" / f"subject_{s}" / "vid.avi"
        size2subj[p.stat().st_size] = f"real_s{s}"
    out = {}
    for f in sorted(CACHE.glob(f"traces_*_{tag}.npz")):
        core = f.name[len("traces_"):-len(f"_{tag}.npz")]
        m = re.match(r"^(.*)_(\d+)_(\d+)$", core)
        if not m:
            continue
        stem, size, _frames = m.group(1), int(m.group(2)), m.group(3)
        stem = re.sub(r"_\d+$", "", stem)   # tmp files are named *_360.avi
        name = size2subj.get(size, stem)
        out[name] = f
    return out


def main():
    clips = {t: enumerate_clips(tg) for t, tg in TAGS.items()}
    print("clips:", {t: sorted(v) for t, v in clips.items()})
    assert all(len(v) == 8 for v in clips.values()), "expected 8 clips/backend"

    data = {}   # backend -> clip -> dict(feats, traces, fps)
    for kind, tag in TAGS.items():
        for name, path in clips[kind].items():
            z_ = np.load(path)
            tr = {k: z_[k] for k in z_.files if k != "fps"}
            fps = float(z_["fps"])
            gt = None
            if name.startswith("real_s"):
                gt, _ = load_gt(ROOT / "Dataset" / f"subject_{name[-1]}")
            base = clip_features(tr, fps, gt)
            data.setdefault(kind, {})[name] = {
                "traces": tr, "fps": fps, "base": base,
                "vs": variants(tr, fps, base)}

    # ---------------- set A: estimator sweep (full clip) ----------------
    vnames = list(next(iter(data["gpu"].values()))["vs"])
    print("\n== A. coherence per clip x variant ==")
    order = ["real_s1", "real_s2", "fake_smooth_s1", "fake_smooth_s2",
             "fake_swap_s2toS1", "fake_swap_s1toS2",
             "fake_nnswap_s2toS1", "fake_nnswap_s1toS2"]
    for kind in TAGS:
        print(f"\n-- {kind} --")
        hdr = f"{'clip':<19}" + "".join(f"{v:>10}" for v in vnames)
        print(hdr)
        for name in order:
            row = f"{name:<19}"
            for v in vnames:
                row += f"{data[kind][name]['vs'][v]:>10.3f}"
            print(row)

    print("\n== A. summary: separation = min(real) - max(fake), accuracy @tau=%.2f ==" % TAU_COH)
    hdr = (f"{'variant':<12}" + "".join(f"{k:>22}" for k in TAGS))
    print(hdr)
    summary = {}
    for v in vnames:
        row = f"{v:<12}"
        summary[v] = {}
        for kind in TAGS:
            reals = [data[kind][n]["vs"][v] for n in ("real_s1", "real_s2")]
            fakes = [data[kind][n]["vs"][v] for n in order[2:]]
            sep = min(reals) - max(fakes)
            correct = 0
            for name in order:
                b = data[kind][name]["base"]
                ok_spr = b["spr_db"] >= TAU_SPR_DB
                ok_bpm = BPM_RANGE[0] <= b["bpm"] <= BPM_RANGE[1]
                pred_real = ok_spr and ok_bpm and data[kind][name]["vs"][v] >= TAU_COH
                truth_real = name.startswith("real_s")
                correct += int(pred_real == truth_real)
            summary[v][kind] = {"sep": sep, "correct": correct,
                                "real": reals, "fake_max": max(fakes)}
            row += f"{correct}/8 sep={sep:+.3f}"
        print(row)

    # ---------------- set C1: windowed evaluation ----------------
    print("\n== C1. windowed evaluation (SPR+coherence recomputed per window) ==")
    for w in WINS:
        print(f"\n-- window = {w} frames --")
        for kind in TAGS:
            for v in ("baseline", "best2"):
                n_ok = n_all = 0
                per_clip = {}
                for name in order:
                    tr, fps = data[kind][name]["traces"], data[kind][name]["fps"]
                    n = len(next(iter(tr.values())))
                    votes = []
                    for s in range(0, n - w + 1, w):
                        sl = {k: a[s:s + w] for k, a in tr.items()}
                        base = clip_features(sl, fps, None)
                        coh = variants(sl, fps, base)[v]
                        pred = (base["spr_db"] >= TAU_SPR_DB
                                and BPM_RANGE[0] <= base["bpm"] <= BPM_RANGE[1]
                                and coh >= TAU_COH)
                        truth = name.startswith("real_s")
                        n_ok += int(pred == truth)
                        n_all += 1
                        votes.append(pred)
                    maj = (sum(votes) * 2 > len(votes)) == name.startswith("real_s")
                    per_clip[name] = maj
                ci = wilson(n_ok, n_all)
                maj_ok = sum(per_clip.values())
                print(f"  {kind:<10} variant={v:<9} windows {n_ok}/{n_all}"
                      f" ({100*n_ok/n_all:.1f}%, Wilson95 [{100*ci[0]:.1f},{100*ci[1]:.1f}])"
                      f"  majority-vote clips {maj_ok}/8")

    out = {"variants": {k: {n: d["vs"] for n, d in v.items()}
                        for k, v in data.items()},
           "summary": summary}
    Path("output/exp_variants.json").write_text(json.dumps(out, indent=2))
    print("\nwrote output/exp_variants.json")


if __name__ == "__main__":
    main()

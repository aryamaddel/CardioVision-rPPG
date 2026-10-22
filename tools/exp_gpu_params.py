"""Re-extraction experiments: does a better GPU landmark/ROI config beat MediaPipe?

The estimator sweep (tools/exp_variants.py) showed best-2-of-3 pair pooling
lifts real_s1 above tau on the GPU path. This script attacks the OTHER lever:
the geometry/signal side of extraction itself.

Configs (all GPU / CUDA, fresh cache tags so gpu_1e3 stays pristine):
  p05        One-Euro min_cutoff 0.5 (stronger landmark smoothing)
  p025       One-Euro min_cutoff 0.25 (stronger still)
  roi08      ROI semi-axis 0.055 -> 0.08 (bigger skin patches, higher SNR)
  p05roi08   both

Each config is scored with BOTH coherence estimators (baseline mean-of-3-pairs
and best-2-of-3), against the shipped baselines (gpu_1e3, mediapipe).

Run:  uv run python tools/exp_gpu_params.py
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepfake_detect import (CACHE, REGIONS, ROOT, ROI_R, TMP, clip_features,
                             extract_traces, load_gt, TAU_COH, TAU_SPR_DB,
                             BPM_RANGE)
from gpu_core import GPURegionExtractor, _OneEuro

FRAMES = 360
ROI_R_SHIPPED = 0.055  # ROI_R at the time of this experiment (pre-ablation)
CONFIGS = [
    # (tag,           min_cutoff, roi_r)
    ("p05",           0.5,  ROI_R_SHIPPED),
    ("p025",          0.25, ROI_R_SHIPPED),
    ("roi08",         1.0,  0.08),
    ("p05roi08",      0.5,  0.08),
]
VARIANTS = ("baseline", "best2")


def pair_rs(traces, fps):
    from rppg import pos_algorithm
    rp = {k: pos_algorithm(traces[k], fps) for k in REGIONS}
    keys = list(rp)
    out = {}
    for x, y in combinations(keys, 2):
        a, b = rp[x], rp[y]
        s = lambda t: (t - t.mean()) / (t.std() if t.std() > 1e-12 else 1.0)  # noqa: E731
        out[f"{x}|{y}"] = float(np.corrcoef(s(a), s(b))[0, 1])
    return out


def clip_matrix():
    """name -> (path, gt) for the 8 clips."""
    vids = {s: ROOT / "Dataset" / f"subject_{s}" / "vid.avi" for s in (1, 2)}
    fakes = {p.stem: p for p in sorted(TMP.glob("fake_*_360.avi"))}
    clips = {"real_s1": (vids[1], 1), "real_s2": (vids[2], 2)}
    clips.update({k: (v, None) for k, v in fakes.items()})
    assert len(clips) == 8, f"expected 8 clips, got {sorted(clips)}"
    return clips


def score(coh_map, clips, tag, variant):
    reals = [coh_map["real_s1"][variant], coh_map["real_s2"][variant]]
    fakes = [coh_map[k][variant] for k in clips if k.startswith("fake_")]
    correct = 0
    for name, (path, subj) in clips.items():
        b = coh_map[name]["_base"]
        pred = (b["spr_db"] >= TAU_SPR_DB
                and BPM_RANGE[0] <= b["bpm"] <= BPM_RANGE[1]
                and coh_map[name][variant] >= TAU_COH)
        correct += int(pred == name.startswith("real_s"))
    return correct, min(reals) - max(fakes), min(reals), max(fakes)


def main():
    clips = clip_matrix()

    # -------- baselines from existing caches --------
    def load_cached(tag_suffix):
        out = {}
        for name, (path, subj) in clips.items():
            key = CACHE / f"traces_{path.stem}_{path.stat().st_size}_{FRAMES}_{tag_suffix}.npz"
            if not key.exists():
                continue
            z = np.load(key)
            tr = {k: z[k] for k in z.files if k != "fps"}
            fps = float(z["fps"])
            gt = load_gt(ROOT / "Dataset" / f"subject_{subj}")[0] if subj else None
            base = clip_features(tr, fps, gt)
            rs = pair_rs(tr, fps)
            vals = sorted(rs.values())
            out[name] = {"_base": base, "baseline": base["coherence"],
                         "best2": float(np.mean(vals[1:]))}
        return out

    rows = []
    for label, tag_suffix in (("gpu_v3 (shipped)", "gpu_1e3"),
                              ("mediapipe (shipped)", "mediapipe")):
        cm = load_cached(tag_suffix)
        for v in VARIANTS:
            c, sep, rmin, fmax = score(cm, clips, tag_suffix, v)
            rows.append((f"{label} [{v}]", c, sep, rmin, fmax, cm))

    # -------- re-extraction configs --------
    for tag, min_cutoff, roi_r in CONFIGS:
        def factory(mc=min_cutoff, rr=roi_r):
            ex = GPURegionExtractor(regions=REGIONS, roi_r=rr)
            if mc != 1.0:
                ex._filt = _OneEuro(min_cutoff=mc, beta=0.05, d_cutoff=1.0)
            return ex
        cm = {}
        print(f"\n=== config {tag}: min_cutoff={min_cutoff} roi_r={roi_r} ===")
        for name, (path, subj) in clips.items():
            tr, fps = extract_traces(path, FRAMES, 30.0, tag=f"gpu_{tag}",
                                     make_extractor=factory)
            gt = load_gt(ROOT / "Dataset" / f"subject_{subj}")[0] if subj else None
            base = clip_features(tr, fps, gt)
            rs = pair_rs(tr, fps)
            vals = sorted(rs.values())
            cm[name] = {"_base": base, "baseline": base["coherence"],
                        "best2": float(np.mean(vals[1:]))}
        for v in VARIANTS:
            c, sep, rmin, fmax = score(cm, clips, tag, v)
            rows.append((f"gpu_{tag} [{v}]", c, sep, rmin, fmax, cm))

    # -------- report --------
    print("\n== extraction x estimator results ==")
    print(f"{'config':<34} {'acc':>5} {'sep':>7} {'min(real)':>10} {'max(fake)':>10}")
    for label, c, sep, rmin, fmax, _cm in rows:
        print(f"{label:<34} {c}/8   {sep:+.3f}  {rmin:>10.3f} {fmax:>10.3f}")

    print("\n== per-clip coherence, best configs vs baselines ==")
    pick = [r for r in rows if r[0].endswith("[best2]")]
    hdr = f"{'clip':<19}" + "".join(f"{r[0][:24]:>26}" for r in pick)
    print(hdr)
    for name in ["real_s1", "real_s2"] + sorted(k for k in clips if k.startswith("fake_")):
        line = f"{name:<19}"
        for _l, _c, _s, _r, _f, cm in pick:
            line += f"{cm[name]['best2']:>26.3f}"
        print(line)


if __name__ == "__main__":
    main()

"""Final experiment: 12 clips x 4 extraction configs x 2 coherence estimators.

New since exp_gpu_params.py:
  * 4 out-of-sample fake clips (tools/make_more_fakes.py): mild smooth k=2,
    heavy smooth k=15, and two neural-swap -> smooth chains
  * FAIR comparison: MediaPipe extracted with the SAME enlarged ROI (r=0.08)
    - if the ROI change helps both backends, the winner must be measured on
      matched features, not against the shipped small-ROI baseline

Configs:
  gpu_shipped        r=0.055, One-Euro 1.0  (shipped gpu_1e3 cache)
  mp_shipped         r=0.055                 (shipped mediapipe cache)
  gpu_roi08          r=0.08,  One-Euro 1.0  (from exp_gpu_params)
  mp_roi08           r=0.08                  (extracted here)

Each config scored with baseline (mean of 3 pairs) and best2 estimators at the
SHIPPED thresholds (tau_spr=4 dB, tau_coh=0.35, BPM 45..140) - thresholds are
never re-fit on the new clips. Windowed (60-frame) evaluation, n=72/config.

Run:  uv run python tools/exp_final.py
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepfake_detect import (CACHE, REGIONS, ROOT, ROI_R, TMP, RegionExtractor,
                             clip_features, extract_traces, load_gt,
                             TAU_COH, TAU_SPR_DB, BPM_RANGE)
from gpu_core import GPURegionExtractor
from rppg import pos_algorithm

FRAMES = 360
WIN = 60
ROI_R_SHIPPED = 0.055  # ROI_R at the time of this experiment (pre-ablation)
NEW_FAKES = ["fake_smooth_k2_s1", "fake_smooth_k15_s2",
             "fake_nnsmooth_s2toS1", "fake_nnsmooth_s1toS2"]


def all_clips():
    v = {s: ROOT / "Dataset" / f"subject_{s}" / "vid.avi" for s in (1, 2)}
    clips = {"real_s1": (v[1], 1), "real_s2": (v[2], 2)}
    for p in sorted(TMP.glob("fake_*_360.avi")):
        clips[p.stem] = (p, None)
    assert len(clips) == 12, f"expected 12 clips, got {len(clips)}: {sorted(clips)}"
    return clips


def factory_gpu(roi_r):
    return lambda: GPURegionExtractor(regions=REGIONS, roi_r=roi_r)


def factory_mp(roi_r):
    def f():
        ex = RegionExtractor()
        ex.roi_r = roi_r
        return ex
    return f


def feats(tr, fps, gt):
    base = clip_features(tr, fps, gt)
    rp = {k: pos_algorithm(tr[k], fps) for k in REGIONS}
    keys = list(rp)
    s = lambda t: (t - t.mean()) / (t.std() if t.std() > 1e-12 else 1.0)  # noqa: E731
    rs = sorted(float(np.corrcoef(s(rp[x]), s(rp[y]))[0, 1])
                for x, y in combinations(keys, 2))
    return {"base": base, "baseline": base["coherence"],
            "best2": float(np.mean(rs[1:]))}


def verdict(f, coh):
    b = f["base"]
    return (b["spr_db"] >= TAU_SPR_DB
            and BPM_RANGE[0] <= b["bpm"] <= BPM_RANGE[1]
            and coh >= TAU_COH)


def main():
    clips = all_clips()
    configs = [
        ("gpu_shipped", "gpu_1e3", factory_gpu(ROI_R_SHIPPED)),
        ("mp_shipped", "mediapipe", factory_mp(ROI_R_SHIPPED)),
        ("gpu_roi08", "gpu_roi08", factory_gpu(0.08)),
        ("mp_roi08", "mediapipe_r08", factory_mp(0.08)),
    ]
    order = ["real_s1", "real_s2"] + sorted(k for k in clips if k.startswith("fake_"))

    data = {}
    for cfg, tag, fac in configs:
        print(f"\n=== {cfg} (tag={tag}) ===")
        cm = {}
        for name, (path, subj) in clips.items():
            tr, fps = extract_traces(path, FRAMES, 30.0, tag=tag, make_extractor=fac)
            gt = load_gt(ROOT / "Dataset" / f"subject_{subj}")[0] if subj else None
            cm[name] = feats(tr, fps, gt)
        data[cfg] = cm

    # ---- full-clip accuracy + separation ----
    print("\n== full-clip results (12 clips, shipped thresholds) ==")
    print(f"{'config':<14} {'estimator':<10} {'acc':>6} {'sep':>7} {'min(real)':>10} {'max(fake)':>10}")
    summary = {}
    for cfg, _t, _f in configs:
        cm = data[cfg]
        for est in ("baseline", "best2"):
            ok = sum(int(verdict(cm[n], cm[n][est]) == n.startswith("real_s"))
                     for n in order)
            reals = [cm[n][est] for n in ("real_s1", "real_s2")]
            fakes = [cm[n][est] for n in order[2:]]
            sep = min(reals) - max(fakes)
            summary[f"{cfg}|{est}"] = {"acc": ok, "sep": sep,
                                       "min_real": min(reals),
                                       "max_fake": max(fakes)}
            print(f"{cfg:<14} {est:<10} {ok:>4}/12 {sep:>+7.3f} {min(reals):>10.3f} {max(fakes):>10.3f}")

    # ---- per-clip coherence table (12 clips) ----
    print("\n== per-clip coherence ==")
    hdr = f"{'clip':<23}" + "".join(f"{c:>14}" for c, _t, _f in configs)
    print(hdr + "   (baseline | best2)")
    for name in order:
        line = f"{name:<23}"
        for cfg, _t, _f in configs:
            cm = data[cfg]
            line += f"{cm[name]['baseline']:>7.3f}|{cm[name]['best2']:<6.3f}"
        print(line)

    # ---- wrong clips detail ----
    print("\n== errors at shipped thresholds ==")
    for cfg, _t, _f in configs:
        for est in ("baseline", "best2"):
            errs = [n for n in order
                    if verdict(data[cfg][n], data[cfg][n][est]) != n.startswith("real_s")]
            if errs:
                print(f"  {cfg} [{est}]: {errs}")

    # ---- windowed evaluation ----
    print(f"\n== windowed ({WIN} frames -> n=72 per config, baseline estimator) ==")
    for cfg, tag, fac in configs:
        n_ok = n_all = 0
        maj = 0
        for name, (path, subj) in clips.items():
            key = CACHE / f"traces_{path.stem}_{path.stat().st_size}_{FRAMES}_{tag}.npz"
            z = np.load(key)
            tr = {k: z[k] for k in z.files if k != "fps"}
            fps = float(z["fps"])
            votes = []
            n = len(next(iter(tr.values())))
            truth = name.startswith("real_s")
            for s0 in range(0, n - WIN + 1, WIN):
                sl = {k: a[s0:s0 + WIN] for k, a in tr.items()}
                f = feats(sl, fps, None)
                pred = verdict(f, f["baseline"])
                n_ok += int(pred == truth)
                n_all += 1
                votes.append(pred)
            maj += int((sum(votes) * 2 > len(votes)) == truth)
        p = 100 * n_ok / n_all
        print(f"  {cfg:<14} windows {n_ok}/{n_all} ({p:.1f}%)   majority-vote clips {maj}/12")

    Path("output/exp_final.json").write_text(json.dumps(summary, indent=2))
    print("\nwrote output/exp_final.json")


if __name__ == "__main__":
    main()

"""Windowed failure diagnosis: WHY do short windows fail, per config?

exp_final showed gpu_roi08 wins full-clip (12/12, sep +0.222) but has the
worst 60-frame window accuracy (76.4%). This breaks every failing window down
by cause (SPR / coherence / BPM) and by clip, and sweeps window sizes
60 / 90 / 120 frames to see whether the GPU gap is a 2-second-estimator
artifact or a real stability problem.

Run:  uv run python tools/exp_window_diag.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepfake_detect import (CACHE, ROOT, TMP, clip_features,
                             TAU_COH, TAU_SPR_DB, BPM_RANGE)

FRAMES = 360
WINS = (60, 90, 120)

CONFIGS = [
    ("gpu_shipped", "gpu_1e3"),
    ("mp_shipped", "mediapipe"),
    ("gpu_roi08", "gpu_roi08"),
    ("mp_roi08", "mediapipe_r08"),
]


def all_clips():
    v = {s: ROOT / "Dataset" / f"subject_{s}" / "vid.avi" for s in (1, 2)}
    clips = {"real_s1": (v[1], 1), "real_s2": (v[2], 2)}
    for p in sorted(TMP.glob("fake_*_360.avi")):
        clips[p.stem] = (p, None)
    return clips


def main():
    clips = all_clips()
    order = ["real_s1", "real_s2"] + sorted(k for k in clips if k.startswith("fake_"))

    for cfg, tag in CONFIGS:
        print(f"\n=== {cfg} (tag={tag}) ===")
        for win in WINS:
            n_ok = n_all = 0
            fails = []          # (clip, window_start, reason)
            for name, (path, subj) in clips.items():
                key = CACHE / f"traces_{path.stem}_{path.stat().st_size}_{FRAMES}_{tag}.npz"
                z = np.load(key)
                tr = {k: z[k] for k in z.files if k != "fps"}
                fps = float(z["fps"])
                truth = name.startswith("real_s")
                n = len(next(iter(tr.values())))
                for s0 in range(0, n - win + 1, win):
                    sl = {k: a[s0:s0 + win] for k, a in tr.items()}
                    base = clip_features(sl, fps, None)
                    coh = base["coherence"]
                    reasons = []
                    if not (base["spr_db"] >= TAU_SPR_DB):
                        reasons.append("spr")
                    if not (coh >= TAU_COH):
                        reasons.append("coh")
                    if not (BPM_RANGE[0] <= base["bpm"] <= BPM_RANGE[1]):
                        reasons.append("bpm")
                    pred = not reasons
                    n_ok += int(pred == truth)
                    n_all += 1
                    if pred != truth:
                        want = "fake" if truth else "real"
                        fails.append((name, s0, want + "_miss(" + "+".join(reasons) + ")"))
            p = 100 * n_ok / n_all if n_all else 0
            print(f"  win={win:>3}  {n_ok}/{n_all} ({p:.1f}%)")
            if win == 60 and fails:
                for name, s0, why in fails:
                    print(f"      miss {name} @f{s0:<4} {why}")


if __name__ == "__main__":
    main()

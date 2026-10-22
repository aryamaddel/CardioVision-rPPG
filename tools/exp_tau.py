"""Threshold sensitivity: is the (tau_spr, tau_coh) verdict robust?

Sweeps tau_coh over 0.10..0.60 (and tau_spr over 2..8 dB at fixed tau_coh)
for both backends x both coherence estimators (baseline / best2), using the
shipped cached traces. Reports accuracy grids and the largest tau_coh that
still gets 8/8 - i.e. how much headroom the thresholds have.

Run:  uv run python tools/exp_tau.py
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepfake_detect import (CACHE, REGIONS, ROOT, clip_features, load_gt,
                             BPM_RANGE)
from rppg import pos_algorithm

TAUS_COH = np.arange(0.10, 0.61, 0.05)
TAUS_SPR = np.arange(2.0, 8.1, 1.0)


def clips_for(tag_suffix):
    size2subj = { (ROOT / "Dataset" / f"subject_{s}" / "vid.avi").stat().st_size:
                  f"real_s{s}" for s in (1, 2) }
    out = {}
    import re
    for f in sorted(CACHE.glob(f"traces_*_{tag_suffix}.npz")):
        core = f.name[len("traces_"):-len(f"_{tag_suffix}.npz")]
        m = re.match(r"^(.*)_(\d+)_(\d+)$", core)
        if not m:
            continue
        stem = re.sub(r"_\d+$", "", m.group(1))
        name = size2subj.get(int(m.group(2)), stem)
        out[name] = f
    return out


def main():
    for kind, tag in (("gpu", "gpu_1e3"), ("mediapipe", "mediapipe")):
        rows = []
        for name, path in sorted(clips_for(tag).items()):
            z = np.load(path)
            tr = {k: z[k] for k in z.files if k != "fps"}
            fps = float(z["fps"])
            gt = load_gt(ROOT / "Dataset" / f"subject_{name[-1]}")[0] \
                if name.startswith("real_s") else None
            base = clip_features(tr, fps, gt)
            rp = {k: pos_algorithm(tr[k], fps) for k in REGIONS}
            keys = list(rp)
            s = lambda t: (t - t.mean()) / (t.std() if t.std() > 1e-12 else 1.0)  # noqa: E731
            rs = sorted(float(np.corrcoef(s(rp[x]), s(rp[y]))[0, 1])
                        for x, y in combinations(keys, 2))
            rows.append({"name": name, "spr": base["spr_db"],
                         "bpm": base["bpm"], "b_coh": base["coherence"],
                         "b2_coh": float(np.mean(rs[1:])),
                         "real": name.startswith("real_s")})

        print(f"\n== {kind} : accuracy vs tau_coh (tau_spr=4) ==")
        print("tau_coh: " + "".join(f"{t:>7.2f}" for t in TAUS_COH))
        for est, key in (("baseline", "b_coh"), ("best2", "b2_coh")):
            line = f"{est:<9}"
            for tc in TAUS_COH:
                ok = 0
                for r in rows:
                    pred = (r["spr"] >= 4.0
                            and BPM_RANGE[0] <= r["bpm"] <= BPM_RANGE[1]
                            and r[key] >= tc)
                    ok += int(bool(pred) == r["real"])
                line += f"{ok:>6}/8"
            print(line)

        best = max(TAUS_COH, key=lambda tc: sum(
            int(((r["spr"] >= 4.0 and BPM_RANGE[0] <= r["bpm"] <= BPM_RANGE[1]
                  and r["b2_coh"] >= tc) == r["real"])) for r in rows))
        print(f"  best2: largest tau_coh with max accuracy = {best:.2f}")

        print(f"== {kind} : accuracy vs tau_spr at tau_coh=0.35 (best2) ==")
        line = "tau_spr: "
        for ts in TAUS_SPR:
            ok = 0
            for r in rows:
                pred = (r["spr"] >= ts
                        and BPM_RANGE[0] <= r["bpm"] <= BPM_RANGE[1]
                        and r["b2_coh"] >= 0.35)
                ok += int(bool(pred) == r["real"])
            line += f"{ts:>7.1f}"
            _ = ok
        print(line)
        line = "acc:     "
        for ts in TAUS_SPR:
            ok = sum(int(((r["spr"] >= ts
                           and BPM_RANGE[0] <= r["bpm"] <= BPM_RANGE[1]
                           and r["b2_coh"] >= 0.35) == r["real"]))
                     for r in rows)
            line += f"{ok:>6}/8 "
        print(line)


if __name__ == "__main__":
    main()

"""Train a small rPPG deepfake detector on TRUE AI-generated videos.

Dataset (50/50 real:AI by window count, group-aware split by identity):
  real : every subject discovered by deepfake_detect.discover_real_subjects()
         (44 subjects: 2 full-frame originals + 42 UBFC-Faces mirror crops),
         raw videos, NO alteration - windows are cut by decoding only.
  ai   : fully generated talking-face clips from tmp/ai_pool/ (EMO / EMO2
         diffusion outputs whose SOURCE PORTRAITS are also AI-generated).
         See tmp/ai_pool/manifest.json for per-file provenance.

Per 12 s window (stride 6 s) the same trace path as deepfake_detect.py runs:
landmarks -> per-region RGB means -> POS pulse -> features:
  SPR per region + full, 3 pairwise coherences + mean, bpm, bpm_spec,
  24 log-PSD shape bins (full pulse), 4 band power ratios.

Model: numpy MLP (1 hidden layer), deployed as INT8: per-output-neuron
symmetric weight quant + per-tensor activation quant calibrated on the
training fold (fake-quant eval; FP32 comparison verified INT8 matches
before the ladder was reduced to INT8-only).
Evaluation: 5-fold cross-validation, folds never share an identity between
train and test, so a face seen in training is never in the test fold.

Run:  uv run python tools/train_rppg_detector.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deepfake_detect import (  # noqa: E402
    BP_HI, BP_LO, CACHE, REGIONS, RegionExtractor, discover_real_subjects,
)
from rppg import pos_algorithm  # noqa: E402

WIN_S = 12.0          # seconds per window (matches the repo's protocol)
STRIDE_S = 6.0        # 50% overlap; splits are identity-aware so overlap
                      # never leaks across train/test
AI_POOL = ROOT / "tmp" / "ai_pool"
OUT_JSON = ROOT / "output" / "train_rppg_results.json"
MODEL_DIR = ROOT / "models" / "rppg_mlp"
SEED = 7

# ------------------------------------------------------------------ features
PSD_BINS = 24                     # log-PSD shape bins across the HR band
BANDS = [(0.7, 1.2), (1.2, 2.0), (2.0, 2.8), (2.8, 4.0)]  # Hz


def _z(x):
    m, s = np.mean(x), np.std(x)
    return (x - m) / (s if s > 1e-9 else 1.0)


def window_features(traces: dict, fps: float) -> np.ndarray:
    """Feature vector for one window of region traces (same path as
    deepfake_detect.clip_features, plus PSD shape - non-finite -> -1)."""
    feats = []
    pulses = {}
    for k in ["full", *REGIONS]:
        p = pos_algorithm(traces[k], fps)
        pulses[k] = p
        n = len(p)
        if n < 60:
            continue
        f, P = welch(p, fs=fps, nperseg=min(256, n))
        band = (f >= BP_LO) & (f <= BP_HI)
        if band.sum() < 3:
            continue
        Pb = P[band] + 1e-15
        spr = float(10 * np.log10(Pb.max() / np.median(Pb)))
        if k == "full":
            shape = 10 * np.log10(Pb / np.median(Pb))
            fx = f[band]
            shape_i = np.interp(np.linspace(fx[0], fx[-1], PSD_BINS), fx, shape)
            feats.extend(shape_i.tolist())
            for lo, hi in BANDS:
                m = (fx >= lo) & (fx < hi)
                feats.append(float(np.mean(Pb[m]) / np.median(Pb)) if m.any() else 0.0)
        feats.append(spr)
    # coherence block
    keys = list(REGIONS)
    coh = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = pulses[keys[i]], pulses[keys[j]]
            if len(a) < 60 or len(b) < 60:
                coh.append(-1.0)
                continue
            xa, xb = _z(a), _z(b)
            c = float(np.corrcoef(xa, xb)[0, 1])
            coh.append(c if np.isfinite(c) else -1.0)
    feats.extend(coh)
    feats.append(float(np.mean([c for c in coh if c > -1]) if any(c > -1 for c in coh) else -1.0))
    v = np.asarray(feats, dtype=np.float64)
    v[~np.isfinite(v)] = -1.0
    return v


# ------------------------------------------------------- windowed extraction
def extract_windows(video: Path, win_s: float, stride_s: float) -> tuple[list[dict], float]:
    """Decode the whole video once, cut region traces into windows."""
    key = f"wintraces_{video.stem}_{video.stat().st_size}_w{win_s:.1f}_s{stride_s:.1f}_mp"
    cache_file = CACHE / f"{key}.npz"
    if cache_file.exists():
        z = np.load(cache_file)
        n = int(z["n"])
        fps = float(z["fps"])
        out = [{k: z[f"{k}_{i}"] for k in ["full", *REGIONS]} for i in range(n)]
        return out, fps

    ex = RegionExtractor()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    win_n = int(round(win_s * fps))
    stride_n = max(1, int(round(stride_s * fps)))
    rows = {k: [] for k in ["full", *REGIONS]}
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        lm = ex.landmarks(frame, int(i * 1000.0 / fps))
        rr = ex.region_rgb(frame, lm) if lm is not None else None
        for k in rows:
            rows[k].append(rr[k] if rr else (np.nan,) * 3)
        i += 1
    cap.release()
    ex.close()

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

    out = []
    total = len(rows["full"])
    start = 0
    while start + win_n <= total:
        out.append({k: traces[k][start:start + win_n] for k in traces})
        start += stride_n
    CACHE.mkdir(exist_ok=True)
    blob = {"n": len(out), "fps": fps}
    for wi, w in enumerate(out):
        for k in ["full", *REGIONS]:
            blob[f"{k}_{wi}"] = w[k]
    np.savez(cache_file, **blob)
    return out, fps


def build_dataset():
    X, y, gid = [], [], []          # y: 1 = REAL, 0 = AI
    rng = np.random.default_rng(SEED)

    print("== AI windows (fully generated, tmp/ai_pool) ==")
    ai_ids = []
    for vid in sorted(AI_POOL.glob("*.mp4")):
        wins, afps = extract_windows(vid, WIN_S, STRIDE_S)
        for w in wins:
            f = window_features(w, afps)
            X.append(f); y.append(0); gid.append(f"ai:{vid.stem}")
        ai_ids.append(vid.stem)
        print(f"  {vid.stem}: {len(wins)} windows")
    n_ai = len(y)
    print(f"AI total: {n_ai} windows from {len(ai_ids)} identities")

    print("== real windows (raw UBFC subjects, no alteration) ==")
    reals = discover_real_subjects()
    per_id = []
    for label, d in sorted(reals.items()):
        wins, fps = extract_windows(d / "vid.avi", WIN_S, STRIDE_S)
        per_id.append((label, d, wins, fps))
    # cap real windows to the AI count for the 50/50 protocol:
    # round-robin one window per identity until the budget is used.
    budget = n_ai
    order = list(range(len(per_id)))
    rng.shuffle(order)
    taken = {i: 0 for i in order}
    while budget > 0:
        progressed = False
        for i in order:
            label, d, wins, fps = per_id[i]
            if taken[i] < len(wins):
                w = wins[taken[i]]
                f = window_features(w, fps)
                X.append(f); y.append(1); gid.append(f"real:{label}")
                taken[i] += 1; budget -= 1; progressed = True
                if budget == 0:
                    break
        if not progressed:
            break
    used_ids = [l for i in order if taken[i] for (l, _, _, _) in [per_id[i]]]
    n_real = len(y) - n_ai
    print(f"real total: {n_real} windows from {sum(1 for i in taken if taken[i])} identities")
    return (np.array(X, dtype=np.float64), np.array(y), np.array(gid),
            n_ai, n_real, used_ids)


# ------------------------------------------------------------------- model
def mlp_train(Xtr, ytr, hidden=24, epochs=400, lr=3e-3, l2=1e-4, seed=SEED):
    rng = np.random.default_rng(seed)
    n, d = Xtr.shape
    W1 = rng.normal(0, np.sqrt(2 / d), (d, hidden))
    b1 = np.zeros(hidden)
    W2 = rng.normal(0, np.sqrt(2 / hidden), (hidden, 2))
    b2 = np.zeros(2)
    Y = np.eye(2)[ytr]
    mW1 = np.zeros_like(W1); vW1 = np.zeros_like(W1)
    mW2 = np.zeros_like(W2); vW2 = np.zeros_like(W2)
    mb1 = np.zeros_like(b1); vb1 = np.zeros_like(b1)
    mb2 = np.zeros_like(b2); vb2 = np.zeros_like(b2)
    beta, eps, t = 0.9, 1e-8, 0
    for _ep in range(epochs):
        h = np.maximum(Xtr @ W1 + b1, 0)             # ReLU
        logits = h @ W2 + b2
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits); p /= p.sum(axis=1, keepdims=True)
        g = (p - Y) / n
        gW2 = h.T @ g + l2 * W2
        gb2 = g.sum(0)
        gh = (g @ W2.T) * (h > 0)
        gW1 = Xtr.T @ gh + l2 * W1
        gb1 = gh.sum(0)

        def adam(p_, g_, m_, v_, t_, beta=beta, eps=eps, lr=lr):
            t_ += 1
            m_ *= beta; m_ += (1 - beta) * g_
            v_ *= beta; v_ += (1 - beta) * g_ * g_
            mh = m_ / (1 - beta ** t_); vh = v_ / (1 - beta ** t_)
            p_ -= lr * mh / (np.sqrt(vh) + eps)
            return t_
        t = adam(W1, gW1, mW1, vW1, t); t = adam(b1, gb1, mb1, vb1, t)
        t = adam(W2, gW2, mW2, vW2, t); t = adam(b2, gb2, mb2, vb2, t)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def calibrate_ranges(model, Xcal):
    """Per-tensor activation ranges for INT8 quantization:
    a1 = max|X| (input range), a2 = max|h| (hidden range). The logits are
    never quantized (only their softmax/argmax is used). Using max|o| (the
    logit range) for the hidden activations instead would let hidden outputs
    round to zero once logits grow beyond 2*max|h| - which collapsed the
    INT8 model to chance (0.52) once the pool made the margins large."""
    h = Xcal @ model["W1"] + model["b1"]
    return {"a1": float(np.abs(Xcal).max()), "a2": float(np.abs(h).max())}


def quantize_int8(model, X, ranges):
    """Fake-quant INT8: per-output-neuron symmetric weights, per-tensor
    activations. The calibrated range maps to the full +/-127 span, so the
    step is range/127 (NOT the range itself - rounding at step=range is a
    1-level quantizer that destroys the signal). Logits stay fp32: only
    their softmax/argmax is consumed."""
    def qw(W):
        s = np.max(np.abs(W), axis=0, keepdims=True) / 127.0
        s[s == 0] = 1e-12
        return np.round(W / s) * s
    W1q, W2q = qw(model["W1"]), qw(model["W2"])
    s1 = max(ranges["a1"], 1e-12) / 127.0
    s2 = max(ranges["a2"], 1e-12) / 127.0

    def qa(a, s):
        return np.clip(np.round(a / s), -127, 127) * s
    h = qa(X, s1)
    h = np.maximum(h @ W1q + model["b1"], 0)
    h = qa(h, s2)
    logits = h @ W2q + model["b2"]
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits); p /= p.sum(axis=1, keepdims=True)
    return p[:, 1]


def auc_score(y, s):
    o = np.argsort(s)
    y_, s_ = y[o], s[o]
    ranks = np.arange(1, len(y_) + 1, dtype=np.float64)
    # average ranks for ties
    i = 0
    while i < len(s_):
        j = i
        while j + 1 < len(s_) and s_[j + 1] == s_[i]:
            j += 1
        if j > i:
            ranks[i:j + 1] = (i + j + 2) / 2.0
        i = j + 1
    n1 = y_.sum()
    n0 = len(y_) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y_ == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ----------------------------------------------------------------------- cv
def main():
    t0 = time.time()
    X, y, gid, n_ai, n_real, used_real_ids = build_dataset()
    d = X.shape[1]
    print(f"\ndataset: {len(y)} windows x {d} features "
          f"(AI {n_ai} / real {n_real}, 50/50 = {n_real / max(len(y),1):.0%} real)")

    # group folds: identities of each class are distributed round-robin over
    # 5 folds, so no identity is in both train and test.
    K = 5
    rng = np.random.default_rng(SEED)
    fold_of = {}
    for cls in (0, 1):
        ids = sorted(set(gid[(y == cls)]))
        rng.shuffle(ids)
        for k, ident in enumerate(ids):
            fold_of[ident] = k % K

    results = {"int8": {"win_acc": [], "id_acc": [], "auc": []}}
    confusion = {}
    for k in range(K):
        te = np.array([fold_of[g] == k for g in gid])
        tr = ~te
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd < 1e-9] = 1.0
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        model = mlp_train(Xtr, y[tr])
        cal = calibrate_ranges(model, Xtr)
        preds = {"int8": quantize_int8(model, Xte, cal)}
        yte, gte = y[te], gid[te]
        for q, s in preds.items():
            hard = (s >= 0.5).astype(int)
            results[q]["win_acc"].append(float((hard == yte).mean()))
            results[q]["auc"].append(auc_score(yte, s))
            idv = []
            for ident in sorted(set(gte)):
                m = gte == ident
                idv.append(int((hard[m] == yte[m]).mean() > 0.5))
            results[q]["id_acc"].append(float(np.mean(idv)))
        confusion[k] = {"n_test": int(te.sum()),
                        "int8_hard": ((preds["int8"] >= 0.5).astype(int)).tolist(),
                        "y": yte.tolist()}
        print(f"fold {k}: test={te.sum():3d}  " +
              "  ".join(f"{q}_acc={results[q]['win_acc'][-1]:.2f}" for q in results))

    print(f"\n== INT8 (5-fold identity-disjoint CV, "
          f"{len(y)} windows, 50/50 real:AI) ==")
    hdr = f"{'quant':<6} {'window acc':>11} {'identity acc':>13} {'AUC':>7}"
    print(hdr); print("-" * len(hdr))
    summary = {}
    for q in ("int8",):
        w = results[q]["win_acc"]; a = results[q]["id_acc"]; z = results[q]["auc"]
        print(f"{q:<6} {np.mean(w):>7.3f}±{np.std(w):.3f} "
              f"{np.mean(a):>9.3f}±{np.std(a):.3f} {np.mean(z):>7.3f}")
        summary[q] = {"window_acc": float(np.mean(w)), "window_std": float(np.std(w)),
                      "identity_acc": float(np.mean(a)), "identity_std": float(np.std(a)),
                      "auc": float(np.mean(z))}

    # final model on ALL data, exported as the INT8 artifact
    mu, sd = X.mean(0), X.std(0); sd[sd < 1e-9] = 1.0
    Xs = (X - mu) / sd
    model = mlp_train(Xs, y)
    cal = calibrate_ranges(model, Xs)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    qw = {}
    for k_ in ("W1", "W2"):
        s = np.max(np.abs(model[k_]), axis=0, keepdims=True) / 127.0
        s[s == 0] = 1e-12
        qw[k_] = s
    np.savez(MODEL_DIR / "int8_model.npz",
             W1_q=np.round(model["W1"] / qw["W1"]).astype(np.int8),
             W1_scale=qw["W1"][0], b1=model["b1"],
             W2_q=np.round(model["W2"] / qw["W2"]).astype(np.int8),
             W2_scale=qw["W2"][0], b2=model["b2"],
             mu=mu, sd=sd,
             # activation ranges: step = a1/127 (input), a2/127 (hidden);
             # logits stay fp32 (softmax/argmax only)
             a1=cal["a1"], a2=cal["a2"])
    print(f"\nwrote {MODEL_DIR / 'int8_model.npz'}")

    payload = {"window_s": WIN_S, "stride_s": STRIDE_S, "features": d,
               "n_windows": int(len(y)), "n_ai": int(n_ai), "n_real": int(n_real),
               "n_ai_identities": int(len(set(g for g in gid if g.startswith('ai:')))),
               "n_real_identities": int(len(set(g for g in gid if g.startswith('real:')))),
               "real_ids_used": used_real_ids, "folds": confusion, "summary": summary,
               "seconds": time.time() - t0}
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()

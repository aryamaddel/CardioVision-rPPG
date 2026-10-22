# CardioVision-rPPG

Remote Photoplethysmography (rPPG) pipeline for non-contact heart rate estimation from facial video. Uses POS algorithm (Wang et al. 2017) with MediaPipe face tracking.

```
python main.py                              # webcam
python main.py --source video.mp4           # video file
python main.py --source 1 --window 8        # camera 1, 8s window
```

Displays face ROI with BPM overlay.

**Results website:** open `docs/index.html` in a browser — tabbed results for both
backends, confusion matrices with precision/recall/F1, the visual checks (landmark
overlay, swap calibration), and the GPU/landmark calibration story.

---

# Deepfake detection from rPPG (`deepfake_detect.py`)

## How people have used PPG for deepfake detection

The core idea across the literature: **real skin carries a heartbeat**. Blood-volume
pulsation modulates light absorption, producing a tiny but periodic, *spatially
coherent* colour signal across the face at the heart rate (~0.7–4 Hz). GAN /
face-swap generators synthesise texture per frame with no underlying physiology,
so this signal is absent, non-periodic, or incoherent — and it is very hard for a
generator to fake without explicitly modelling it.

Key works:

- **FakeCatcher** (Ciftci & Demir, TPAMI 2020, arXiv:1901.02212) — extracts PPG
  signals from multiple facial ROIs, builds pairwise signal-difference maps and a
  classifier. ~91–99% accuracy on FaceForensics, FF++, Celeb-DF, and their own
  Deep Fakes Dataset. The central observation: *spatial coherence* of the pulse
  across face regions breaks down in fakes.
- **DeepFakesON-Phys** (Hernandez-Ortega et al., AAAI 2021, arXiv:2010.00400) —
  feeds rPPG-related features into a Convolutional Attention Network; >98% AUC
  on Celeb-DF and DFDC.
- **DeepRhythm** (Qi et al., ACM MM 2020) — learns visual heartbeat rhythm
  features with temporal attention to expose deepfakes.
- **Fernandes et al.** (2020) — combine appearance cues with physiological
  (heartbeat) features and a neural-ODE to flag videos lacking a plausible pulse.

Practical caveats noted in the literature and observed here: heavy compression
and low light attenuate rPPG for *everyone*, simple temporal smoothing alone
reduces but may not kill the HR spectral peak, and newer diffusion/generative
models can partially preserve the colour pulsation — so coherence-style spatial
features are more robust than "does a peak exist" spectral features.

## What we do in this repo

`deepfake_detect.py` is a minimal, dependency-light version of that pipeline on
the 2 UBFC-rPPG subjects in `Dataset/` (real videos + contact-PPG ground truth):

1. **Clips** (N frames each, default 360 ≈ 12 s):
   - `real_s1`, `real_s2` — the original videos.
   - `fake_smooth_*` — temporal box-car over frames (mimics pulsation-free
     generated texture).
   - `fake_swap_*` — donor face aligned by landmarks (eyes/nose/mouth) and
     Poisson-blended onto the target — a classic naive face-swap deepfake.
   - `fake_nnswap_*` — **modern neural identity swap** generated on the GPU with
     InsightFace `inswapper_128` (SCRFD detect → ArcFace 512-D identity → 554 MB
     generator → feathered paste-back; `make_deepfakes.py`). This is the current
     state of the art — visually near-flawless, and the primary adversarial test.
2. **Traces**: landmarks → mean RGB per region (forehead, both cheeks, full
   face) → POS pulse per region (`rppg.py` reused as-is). Two backends:
   - `--extractor mediapipe` — original CPU pipeline (reference).
   - `--extractor gpu` (default `auto`) — SCRFD + 478-pt face-mesh ONNX on
     `onnxruntime-gpu` CUDA (`gpu_core.py`), One-Euro-smoothed landmarks.
3. **Features** per clip:
   - **SPR (dB)** — Welch PSD peak-to-median ratio inside the HR band
     (is there a dominant periodic signal?).
   - **Coherence** — mean pairwise correlation of cheek/forehead pulses
     (is the same pulse present everywhere on the face?).
   - **BPM** — peak-detection estimate; physiology check 45–140 bpm.
   - **|r| GT** — correlation with contact PPG ground truth (real clips only;
     validates the extractor itself).
4. **Verdict**: REAL iff `SPR >= 4 dB` and `coherence >= 0.35` and BPM plausible.

## Results (360 frames ≈ 12 s @ 29.3 fps)

The shipped configuration uses ROI semi-axis r = 0.08 (chosen by the ablation in
the tuning section below) and unchanged thresholds (SPR ≥ 4 dB, coherence ≥ 0.35,
BPM 45–140). The four `k2 / k15 / nnsmooth` clips are held-out: generated after
the tuning was frozen, never used to pick anything.

### MediaPipe CPU reference — 12/12 correct

| clip                 | SPR (dB) | coherence |  BPM  | GT BPM | \|r\| GT | verdict |
|----------------------|---------:|----------:|------:|-------:|:-------:|:-------:|
| real_s1              |    19.97 |     0.468 | 100.3 |   98.5 |  0.65   | REAL    |
| real_s2              |    17.89 |     0.709 |  80.6 |   97.4 |  0.56   | REAL    |
| fake_smooth_s1       |     7.81 |     0.026 |  83.6 |     -- |   --    | FAKE    |
| fake_smooth_s2       |    11.92 |     0.059 |  48.5 |     -- |   --    | FAKE    |
| fake_smooth_k2_s1 †  |    19.78 |     0.135 | 103.3 |     -- |   --    | FAKE    |
| fake_smooth_k15_s2 † |     7.52 |     0.093 |  90.0 |     -- |   --    | FAKE    |
| fake_swap_s2toS1     |    11.82 |     0.114 |  92.4 |     -- |   --    | FAKE    |
| fake_swap_s1toS2     |    10.36 |     0.321 |  75.4 |     -- |   --    | FAKE    |
| fake_nnswap_s2toS1   |    19.81 |     0.226 |  97.5 |     -- |   --    | FAKE    |
| fake_nnswap_s1toS2   |    16.15 |     0.256 |  84.4 |     -- |   --    | FAKE    |
| fake_nnsmooth_s2toS1 †|   13.70 |     0.129 |  60.5 |     -- |   --    | FAKE    |
| fake_nnsmooth_s1toS2 †|    8.85 |     0.014 |  80.6 |     -- |   --    | FAKE    |

### GPU landmark path (ONNX + CUDA, One-Euro) — 12/12 correct

| clip                 | SPR (dB) | coherence |  BPM  | GT BPM | \|r\| GT | verdict |
|----------------------|---------:|----------:|------:|-------:|:-------:|:-------:|
| real_s1              |    18.43 |     0.468 | 100.3 |   98.5 |  0.44   | REAL    |
| real_s2              |    15.28 |     0.565 |  88.6 |   97.4 |  0.39   | REAL    |
| fake_smooth_s1       |     9.49 |     0.171 |  87.8 |     -- |   --    | FAKE    |
| fake_smooth_s2       |    10.21 |     0.237 |  80.6 |     -- |   --    | FAKE    |
| fake_smooth_k2_s1 †  |    15.16 |     0.205 |  87.8 |     -- |   --    | FAKE    |
| fake_smooth_k15_s2 † |    10.62 |     0.188 | 110.8 |     -- |   --    | FAKE    |
| fake_swap_s2toS1     |     7.44 |     0.121 |  61.7 |     -- |   --    | FAKE    |
| fake_swap_s1toS2     |     7.52 |     0.246 |  55.4 |     -- |   --    | FAKE    |
| fake_nnswap_s2toS1   |    16.09 |     0.227 | 103.3 |     -- |   --    | FAKE    |
| fake_nnswap_s1toS2   |    11.02 |     0.174 |  88.6 |     -- |   --    | FAKE    |
| fake_nnsmooth_s2toS1 †|   17.59 |     0.124 |  70.2 |     -- |   --    | FAKE    |
| fake_nnsmooth_s1toS2 †|   14.13 |    -0.022 |  77.1 |     -- |   --    | FAKE    |

† = held-out clip (generated after tuning was frozen).

### Accuracy summary (positive class = FAKE)

| metric            | MediaPipe CPU | GPU ONNX + CUDA |
|-------------------|--------------:|----------------:|
| Accuracy          |   100% (12/12)|   100% (12/12)  |
| Precision (FAKE)  |   100% (10/10)|   100% (10/10)  |
| Recall (FAKE)     |   100% (10/10)|   100% (10/10)  |
| F1 (FAKE)         |          100% |          100%   |
| Specificity (REAL)|     100% (2/2)|     100% (2/2)  |

Confusion matrices are identical: TP=10 / FP=0 / FN=0 / TN=2. Caveat: the test
set is 12 clips from only 2 real subjects — one real-side miss moves accuracy
8 points and specificity still rests on two clips, so this is a feasibility
result, not a benchmark score. What distinguishes the backends is *margin*, not
accuracy: separation (weakest real − strongest fake) is **+0.22 on GPU vs +0.15
on MediaPipe** (GPU real min 0.468 vs fake max 0.246; MediaPipe 0.468 vs 0.321).

Stress test — re-judging every clip in 60-frame (2 s) windows (n = 72 per
backend, same thresholds, no re-fitting):

| backend   | windows correct | window acc | majority-vote clips |
|-----------|----------------:|-----------:|--------------------:|
| MediaPipe |         65 / 72 |     90.3%  |            12 / 12  |
| GPU       |         55 / 72 |     76.4%  |            11 / 12  |

This is where MediaPipe's temporal tracker still wins: 2-second windows are
noisier on the per-frame CUDA mesh. The GPU gap shrinks with window length
(85.4% at 3 s, 86.1% at 4 s vs 95.8% / 91.7%). Honest scoreboard: **GPU wins
full-clip separation, MediaPipe wins short-window stability.**

Observations:

- HR recovery works on both real subjects vs GT; |r| against contact PPG
  0.56–0.65 (MediaPipe) / 0.39–0.44 (GPU).
- **All 10 fakes caught on both backends**, including both neural swaps and both
  held-out swap→smooth chains. Coherence separates the groups cleanly
  (real 0.47–0.71 vs fake −0.02…0.32), matching FakeCatcher's finding.
- **The neural swaps are the hardest fakes — and they attack SPR, not coherence.**
  inswapper's feathered paste-back leaves the target's real skin (and its real
  pulse) around the face margins, so SPR reaches 19.8 dB — *higher* than any
  naive-swap or smoothed clip, and above the real clips' own SPR in the GPU run.
  Even stacking temporal smoothing on top (nnsmooth) leaves a 17.6 dB peak.
  Spectral-only detection would misclassify them; the inter-region coherence test
  (≤ 0.23) is what exposes them. This reproduces FakeCatcher's central claim
  against a modern one-shot neural generator.
- Mild smoothing (k=2) is a genuinely hard fake: it keeps a 19.8 dB pulse peak
  (MediaPipe) and near-real BPM, and still fails only the coherence test.
- Smoothing leaves a spectral remnant (SPR up to 11.9 dB) — SPR alone is a
  weaker detector than coherence, and the two features fail independently.

## GPU execution (RTX 5070, onnxruntime-gpu 1.26)

Every inference stage runs on CUDA: SCRFD detection, 478-pt face mesh, ArcFace
identity embedding, the 554 MB inswapper generator, and region extraction.
Windows-specific detail: the pip NVIDIA wheels put DLLs in
`site-packages/nvidia/<pkg>/bin`, which Python never searches —
`gpu_core.ensure_cuda_dlls()` registers them via `os.add_dll_directory`.

Measured (nvidia-smi, 0.4 s sampling, 12-clip runs):

| stage                           | util mean | util peak | memory peak | power mean |
|---------------------------------|----------:|----------:|------------:|-----------:|
| neural-swap generation (2×360)  |     23.1% |     74.0% |     3875 MB |      56.6 W |
| GPU trace extraction (12 clips) |     31.3% |     39.0% |     1800 MB |      70.0 W |
| MediaPipe CPU extraction        |      0.0% |      0.0% |     1300 MB |      23.2 W |

Generation peaks at 74% because the decoder → GPU → ordered-paste 3-stage
pipeline keeps the generator fed. Extraction runs ~60 fps end-to-end at a steady
~31%. Honest caveat: MediaPipe's *CPU* build (XNNPACK + internal tracking)
extracts at ~167 fps — faster than the per-frame CUDA mesh path, since it
amortises detection across frames. The GPU path exists because MediaPipe's
Windows wheel ships with GPU disabled ("GPU processing is disabled in build
flags"), so it is the way to keep the *whole* pipeline on the card — not a claim
that CUDA is always faster for 256-px landmark models.

## GPU landmark calibration (`gpu_core.py`, `tools/`)

The GPU mesh must land landmarks in MediaPipe's index space or every ROI/EXCLUDE
mask inherited from `rppg.py` drifts. Fitted empirically against the MediaPipe
CPU reference (`tools/calibrate_scrfd.py`, `fit_crop.py`, `fit_input.py`,
validated by `validate_gpu.py`, measured by `diag_jitter.py`/`diag_anchors.py`):

| quantity                | before          | after            | how |
|-------------------------|-----------------|------------------|-----|
| SCRFD keypoints         | —               | 1.4–2.3 px       | 2 anchors/cell decode, offset 0, stride-scaled preds |
| Mesh landmarks (s1 / s2)| 12.6 px (s2)    | **6.4 / 7.4 px** | absolute 191×229 crop + affine centre offsets, fitted on both subjects |
| ROI-anchor points       | —               | 4.0 / 5.5 px     | the 14 points the ROIs use beat the 478-pt mean |
| Frame-to-frame jitter   | 8–10 px         | **1.7–2.2 px**   | vectorised One-Euro filter (adaptive cutoff); MediaPipe tracker: 0.04–0.06 px |
| inswapper I/O           | garbage output  | correct faces    | float 0..1 in / 0..1 out, verified over 4 normalisation variants |

Key findings from the calibration:

- **A constant-ratio crop is the wrong model.** The optimal crop is near-constant
  in *absolute pixels* across both subjects' face sizes; the final geometry is a
  fixed 191×229 px window with affine centre offsets. This cut mean landmark
  error ~35–45% versus a proportional fit and rescued subject_2 from 12.6 px.
  Constants are calibrated for ~110–140 px SCRFD boxes; other resolutions need a
  refit (`tools/fit_crop.py 1 2`).
- **Temporal smoothing matters more than raw accuracy.** Per-frame ONNX inference
  jitters 8–10 px (150× MediaPipe's tracked output), randomising the tiny ROI
  positions. One-Euro cuts this 5× with negligible lag.
- **The residual gap is the model, not the plumbing.** Intrinsic mesh residual
  vs MediaPipe is ~5–6 px even with the optimal crop — this bounds the GPU path's
  raw landmark accuracy. As the tuning section below shows, ROI sizing removes
  its effect on the detection verdicts.

## Tuning experiments: making the GPU path the best performer

The first GPU run scored 7/8: real_s1's coherence (0.228) sat below τ=0.35 while
MediaPipe cleared it (0.372). Three levers were swept on the original 8-clip
development set (`tools/exp_variants.py`, `exp_gpu_params.py`, `exp_tau.py`;
full 12-clip replay: `tools/exp_final.py`), thresholds untouched:

| change vs shipped baseline             | GPU acc | GPU sep | MP acc | MP sep |
|----------------------------------------|--------:|--------:|-------:|-------:|
| baseline — ROI r=0.055, mean of 3 pairs|     7/8 |   +0.03 |    8/8 |  +0.08 |
| best-2-of-3 region-pair pooling        |     8/8 |   +0.06 |    8/8 |  +0.13 |
| stronger One-Euro (min_cutoff 0.5/0.25)|     7/8 |   +0.02 |     —  |    —   |
| **bigger ROIs — r 0.055 → 0.08 (kept)**| **8/8** | **+0.22**| **8/8**| **+0.15**|

Findings:

- **Estimator variants** (bandpass before Pearson, Welch magnitude-squared
  coherence, median / best-2-of-3 pair pooling, trace smoothing) all lift
  real_s1, but with r=0.08 the plain mean-of-3-pairs already wins — best-2
  pooling pushes fake_swap_s1toS2 over τ. Shipped estimator kept.
- **Stronger smoothing hurts.** One-Euro min_cutoff 0.5 and 0.25 both score 7/8:
  over-smoothed landmarks lag head motion and drag ROIs off skin.
- **ROI size is the decisive lever.** r=0.08 lifts GPU separation seven-fold
  (+0.03 → +0.22) and MediaPipe two-fold (+0.08 → +0.15). Bigger patches average
  over more skin, so the GPU's ~5 px residual jitter costs less signal — the
  change is backend-neutral but compensates precisely the error the GPU has.
  real_s1 coherence: 0.228 → 0.468.
- **Out-of-sample validation.** r=0.08 was picked on the original 8 clips, then
  frozen. Four new clips (`fake_smooth_k2_s1`, `fake_smooth_k15_s2`, and two
  nnswap→smooth chains via `tools/make_more_fakes.py`) were generated with the
  configuration frozen — all four are caught, and both backends score 12/12 at
  the unchanged thresholds.
- **Threshold sensitivity** (`tools/exp_tau.py`): the GPU path has a wide safe
  band — any τ_coh in (0.25, 0.46] gives 12/12 (CPU: (0.33, 0.46]). Verdicts are
  carried by the signal, not by a lucky cut-off.

## Real AI-generated videos (downloaded)

Four demo clips from **SadTalker** (Zhang et al., CVPR 2023; audio-driven
single-image talking-face generation) were downloaded from the project's public
GitHub Pages assets into `tmp/ai/` and run through the identical pipeline
(`--ai-glob "tmp/ai/*.mp4"`). These are grid-format demo videos (multiple
faces side-by-side, small per-face resolution) — a harder setting than clean
portrait video because rPPG SNR drops with face size.

| clip                    | SPR (dB) | coherence |  BPM  | verdict |
|-------------------------|---------:|----------:|------:|:-------:|
| ai_sadtalker_chinese    |     6.67 |    -0.046 |  65.2 | FAKE    |
| ai_sadtalker_eyeblinking|     0.00 |       nan |  42.6 | FAKE    |
| ai_sadtalker_sing       |     7.02 |     0.795 |  56.8 | **REAL** (wrong) |
| ai_sadtalker_talking    |     9.13 |     0.163 |  90.9 | FAKE    |

Findings:

- Two AI clips show *no* periodic signal at all (SPR ≤ 7 dB, flat/incoherent
  regions) — exactly the "generated faces have no pulse" literature claim.
- `sadtalker_sing` is a **false negative with an instructive cause**: the head
  bobs on the music's beat (~57 bpm), and that strong periodic *motion* modulates
  the mean RGB of every ROI identically — periodic (SPR 7 dB) and coherent
  (0.80), so it passes both tests. This is the classic rPPG motion-artifact
  failure mode noted in the literature (motion contamination is separable from
  blood flow only with spatial-difference/stacked-signal features as in
  FakeCatcher, or learned models as in DeepFakesON-Phys/DeepRhythm — beyond the
  scope of this minimal demo).
- `eyeblinking`: faces are very small in the 2160-px-wide grid → the signal is
  degenerate (SPR 0 dB, coherence undefined). Verdict is correct but the
  evidence is weak; small faces are a practical limit of any rPPG detector.

Outputs: `output/deepfake_ppg_results_{gpu,mediapipe}.json` (+ `_meta` block:
extractor, backends, GPU telemetry), `output/deepfake_ppg_demo_{gpu,mediapipe}.png`,
`gpu_*.csv` traces.
Intermediate fake clips land in `tmp/`; extracted traces cached in
`.rppg_cache/` keyed per landmark backend and ROI configuration.

```
uv run python deepfake_detect.py                     # full run (360 frames, GPU auto)
uv run python deepfake_detect.py --extractor mediapipe  # CPU reference run
uv run python make_deepfakes.py                      # neural-swap clips only
uv run python tools/validate_gpu.py                  # landmark/inswapper validation
uv run python deepfake_detect.py --ai-glob "tmp/ai/*.mp4"   # include downloaded AI clips
```

**Model weights (716 MB):** published as the
[`model-weights-v1`](https://github.com/ratnajeet00/CardioVision-rPPG/releases/tag/model-weights-v1)
GitHub release — `inswapper_128.onnx` and `mesh.onnx` go into `models/`,
`det_10g.onnx` and `w600k_r50.onnx` into `models/buffalo_l/` (SHA-256 prefixes
and upstream sources are listed on the release page). MediaPipe's
`face_landmarker.task` downloads itself on first run.

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

### MediaPipe CPU reference — 8/8 correct

| clip              | SPR (dB) | coherence |  BPM  | GT BPM | \|r\| GT | verdict |
|-------------------|---------:|----------:|------:|-------:|:-------:|:-------:|
| real_s1           |    19.97 |     0.372 | 100.3 |   98.5 |  0.65   | REAL    |
| real_s2           |    17.89 |     0.562 |  80.6 |   97.4 |  0.56   | REAL    |
| fake_smooth_s1    |     7.81 |    -0.076 |  83.6 |     -- |   --    | FAKE    |
| fake_smooth_s2    |    11.92 |     0.058 |  48.5 |     -- |   --    | FAKE    |
| fake_swap_s2toS1  |    11.82 |     0.118 |  92.4 |     -- |   --    | FAKE    |
| fake_swap_s1toS2  |    10.36 |     0.296 |  75.4 |     -- |   --    | FAKE    |
| fake_nnswap_s2toS1|    19.81 |     0.210 |  97.5 |     -- |   --    | FAKE    |
| fake_nnswap_s1toS2|    16.15 |     0.069 |  84.4 |     -- |   --    | FAKE    |

### GPU landmark path (ONNX + CUDA, One-Euro) — 7/8 correct

| clip              | SPR (dB) | coherence |  BPM  | GT BPM | \|r\| GT | verdict |
|-------------------|---------:|----------:|------:|-------:|:-------:|:-------:|
| real_s1           |    18.43 |     0.228 | 100.3 |   98.5 |  0.44   | FAKE ⚠  |
| real_s2           |    15.28 |     0.389 |  88.6 |   97.4 |  0.39   | REAL    |
| fake_smooth_s1    |     9.49 |     0.099 |  87.8 |     -- |   --    | FAKE    |
| fake_smooth_s2    |    10.21 |     0.200 |  80.6 |     -- |   --    | FAKE    |
| fake_swap_s2toS1  |     7.44 |     0.041 |  61.7 |     -- |   --    | FAKE    |
| fake_swap_s1toS2  |     7.52 |     0.174 |  55.4 |     -- |   --    | FAKE    |
| fake_nnswap_s2toS1|    16.09 |     0.162 | 103.3 |     -- |   --    | FAKE    |
| fake_nnswap_s1toS2|    11.02 |     0.056 |  88.6 |     -- |   --    | FAKE    |

⚠ = false negative: real_s1's coherence 0.228 falls below τ=0.35 (the reference
backend clears it by only 0.02). Landmark accuracy — not the physiology — is the
binding constraint on the GPU path; see the calibration section below.

### Accuracy summary (positive class = FAKE)

| metric            | MediaPipe CPU | GPU ONNX + CUDA |
|-------------------|--------------:|----------------:|
| Accuracy          |    100% (8/8) |    87.5% (7/8)  |
| Precision (FAKE)  |    100% (6/6) |    85.7% (6/7)  |
| Recall (FAKE)     |    100% (6/6) |    100% (6/6)   |
| F1 (FAKE)         |    100%       |    92.3%        |
| Specificity (REAL)|    100% (2/2) |    50% (1/2)    |

Confusion matrices: MediaPipe TP=6 / FP=0 / FN=0 / TN=2; GPU
TP=6 / FP=1 / FN=0 / TN=1 (the FP is real_s1). Caveat: the test set is only
8 clips (2 real / 6 fake) — one miss moves accuracy 12.5 points and specificity
rests on two clips, so this is a feasibility result, not a benchmark score.

Observations:

- HR recovery works on both real subjects vs GT; |r| against contact PPG
  0.56–0.65 (MediaPipe) / 0.39–0.44 (GPU).
- **All 6 fakes caught on both backends, including both neural swaps.** Coherence
  separates the groups cleanly (real 0.37–0.56 vs fake -0.08…0.30 on MediaPipe),
  matching FakeCatcher's finding.
- **The neural swaps are the hardest fakes — and they attack SPR, not coherence.**
  inswapper's feathered paste-back leaves the target's real skin (and its real
  pulse) around the face margins, so SPR reaches 19.8 dB — *higher* than any
  naive-swap or smoothed clip, and above the real clips' own SPR in the GPU run.
  Spectral-only detection would misclassify them; the inter-region coherence test
  (0.07–0.21) is what exposes them. This reproduces FakeCatcher's central claim
  against a modern one-shot neural generator.
- Smoothing leaves a spectral remnant (SPR up to 11.9 dB) — SPR alone is a
  weaker detector than coherence, and the two features fail independently.

## GPU execution (RTX 5070, onnxruntime-gpu 1.26)

Every inference stage runs on CUDA: SCRFD detection, 478-pt face mesh, ArcFace
identity embedding, the 554 MB inswapper generator, and region extraction.
Windows-specific detail: the pip NVIDIA wheels put DLLs in
`site-packages/nvidia/<pkg>/bin`, which Python never searches —
`gpu_core.ensure_cuda_dlls()` registers them via `os.add_dll_directory`.

Measured (nvidia-smi, 0.4 s sampling):

| stage                          | util mean | util peak | memory peak | power mean |
|--------------------------------|----------:|----------:|------------:|-----------:|
| neural-swap generation (2×360) |     23.1% |     74.0% |     3875 MB |      56.6 W |
| GPU trace extraction (8 clips) |     31.5% |     39.0% |     1878 MB |      68.9 W |
| MediaPipe CPU extraction       |      2.6% |      4.0% |     1827 MB |      14.6 W |

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
  vs MediaPipe is ~5–6 px even with the optimal crop — that is what pushes
  real_s1's coherence below threshold on the GPU path. A production system would
  fine-tune the mesh or keep MediaPipe's tracker and GPU-offload only heavy stages.

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

Outputs: `output/deepfake_ppg_results.json` (+ `_meta` block: extractor,
backends, GPU telemetry), `output/deepfake_ppg_demo.png`, `gpu_*.csv` traces.
Intermediate fake clips land in `tmp/`; extracted traces cached in
`.rppg_cache/` keyed per landmark backend.

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

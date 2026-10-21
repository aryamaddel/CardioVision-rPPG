# CardioVision-rPPG

Remote Photoplethysmography (rPPG) pipeline for non-contact heart rate estimation from facial video. Uses POS algorithm (Wang et al. 2017) with MediaPipe face tracking.

```
python main.py                              # webcam
python main.py --source video.mp4           # video file
python main.py --source 1 --window 8        # camera 1, 8s window
```

Displays face ROI with BPM overlay.

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
2. **Traces**: MediaPipe landmarks → mean RGB per region (forehead, both
   cheeks, full face) → POS pulse per region (`rppg.py` reused as-is).
3. **Features** per clip:
   - **SPR (dB)** — Welch PSD peak-to-median ratio inside the HR band
     (is there a dominant periodic signal?).
   - **Coherence** — mean pairwise correlation of cheek/forehead pulses
     (is the same pulse present everywhere on the face?).
   - **BPM** — peak-detection estimate; physiology check 45–140 bpm.
   - **|r| GT** — correlation with contact PPG ground truth (real clips only;
     validates the extractor itself).
4. **Verdict**: REAL iff `SPR >= 4 dB` and `coherence >= 0.35` and BPM plausible.

## Results (360 frames ≈ 12 s @ 29.3 fps, 6/6 correct)

| clip              | SPR (dB) | coherence |  BPM  | GT BPM | \|r\| GT | verdict |
|-------------------|---------:|----------:|------:|-------:|:-------:|:-------:|
| real_s1           |    19.97 |     0.372 | 100.3 |   98.5 |  0.65   | REAL    |
| real_s2           |    19.56 |     0.562 |  94.3 |   96.0 |  0.75   | REAL    |
| fake_smooth_s1    |     7.81 |    -0.076 |  83.6 |     -- |   --    | FAKE    |
| fake_smooth_s2    |    12.02 |    -0.192 |  66.4 |     -- |   --    | FAKE    |
| fake_swap_s2toS1  |    10.17 |    -0.109 |  97.5 |     -- |   --    | FAKE    |
| fake_swap_s1toS2  |     5.40 |     0.145 |  81.5 |     -- |   --    | FAKE    |

Observations:

- HR recovery works on both real subjects (94–100 bpm vs GT 96–99 bpm;
  |r| against contact PPG 0.65–0.75).
- Fake clips may still produce *a* BPM number — the demand is not "any pulse"
  but a *periodic and spatially coherent* one. Coherence separates the groups
  cleanly (real 0.37–0.56 vs fake -0.19…0.15), matching FakeCatcher's finding.
- Smoothing leaves a spectral remnant (SPR up to 12 dB) stronger than the face
  swap leaves (5.4 dB) — i.e. SPR alone is a weaker detector than coherence.

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

- 9/10 clips overall classified correctly (6/6 local + 3/4 AI).
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

Outputs: `output/deepfake_ppg_results.json`, `output/deepfake_ppg_demo.png`.
Intermediate fake clips land in `tmp/`; extracted traces cached in `.rppg_cache/`.

```
python deepfake_detect.py                    # full run (360 frames)
python deepfake_detect.py --frames 90        # quick run
python deepfake_detect.py --tau-spr 4 --tau-coh 0.35
python deepfake_detect.py --ai-glob "tmp/ai/*.mp4"   # include downloaded AI clips
```

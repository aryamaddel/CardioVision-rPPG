# Model weights

The ONNX weights are **not** stored in git (716 MB total). They are published as the
[`model-weights-v1` release](https://github.com/ratnajeet00/CardioVision-rPPG/releases/tag/model-weights-v1).

## Files

| file | size | place at | SHA-256 (first 16) | source |
|---|---|---|---|---|
| `inswapper_128.onnx` | 528.6 MB | `models/inswapper_128.onnx` | `E4A3F08C753CB72D` | [ashleykleynhans/inswapper](https://huggingface.co/ashleykleynhans/inswapper) mirror |
| `w600k_r50.onnx` | 166.3 MB | `models/buffalo_l/w600k_r50.onnx` | `4C06341C33C2CA1F` | InsightFace `buffalo_l` |
| `det_10g.onnx` | 16.1 MB | `models/buffalo_l/det_10g.onnx` | `5838F7FE053675B1` | InsightFace `buffalo_l` |
| `mesh.onnx` | 4.9 MB | `models/mesh.onnx` | `C334C26B128DFC70` | MediaPipe-compatible 478-pt face mesh (see `research.md`) |

`mesh.onnx` + `det_10g.onnx` are the minimum for `--extractor gpu`;
`w600k_r50.onnx` and `inswapper_128.onnx` are additionally required for
`make_deepfakes.py` (neural identity swaps).

## Download

```powershell
gh release download model-weights-v1 --repo ratnajeet00/CardioVision-rPPG
Move-Item inswapper_128.onnx, mesh.onnx models/
Move-Item det_10g.onnx, w600k_r50.onnx models/buffalo_l/
```

MediaPipe's `face_landmarker.task` (CPU path) downloads automatically on first run.

InsightFace models are released for non-commercial research use only.

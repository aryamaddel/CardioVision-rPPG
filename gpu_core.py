"""
GPU inference core: everything in this repo that can run on the RTX 5070 does.

- CUDA bootstrap: registers the pip-installed NVIDIA cuDNN/cuBLAS/cuFFT DLL
  directories on Windows so onnxruntime-gpu's CUDAExecutionProvider loads
  (the wheels put DLLs in `nvidia/<pkg>/bin`, which Python never searches).
- SCRFD face detector (buffalo_l/det_10g.onnx) on CUDA.
- MediaPipe-compatible face mesh (`models/mesh.onnx`, 478 landmarks: 468 face
  mesh + 10 iris, exact MediaPipe FaceMesh index space) on CUDA, so all ROI
  definitions/EXCLUDE indices from rppg.py + deepfake_detect.py work unchanged.
- ArcFace (w600k_r50) identity embedder + inswapper_128 face-swap generator.
- GPURegionExtractor: same interface as deepfake_detect.RegionExtractor
  (.landmarks() / .region_rgb() / .close()), but inference on CUDA.
- GpuMonitor: samples nvidia-smi while a stage runs.

MediaPipe's Windows wheel is built without GPU support ("GPU processing is
disabled in build flags"), so the CPU fallback lives in deepfake_detect.py and
this module is the GPU path.
"""
from __future__ import annotations

import glob
import os
import site
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
BUFFALO = MODELS / "buffalo_l"

DET_MODEL = BUFFALO / "det_10g.onnx"
ARCFACE_MODEL = BUFFALO / "w600k_r50.onnx"
MESH_MODEL = MODELS / "mesh.onnx"
INSWAPPER_MODEL = MODELS / "inswapper_128.onnx"

# MediaPipe-compatible 112px arcface template (image-left eye first).
ARCFACE_TEMPLATE_112 = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)

# backend report: model path -> "CUDA" | "CPU"
_BACKENDS: dict[str, str] = {}
_DLL_DIRS: list[str] = []


# ------------------------------------------------------------------ CUDA boot
def ensure_cuda_dlls() -> list[str]:
    """Make pip-installed CUDA/cuDNN DLLs resolvable (Windows only)."""
    global _DLL_DIRS
    if _DLL_DIRS:
        return _DLL_DIRS
    dirs: list[str] = []
    for sp in site.getsitepackages():
        base = Path(sp) / "nvidia"
        if not base.is_dir():
            continue
        for sub in ("bin", "lib"):
            dirs += [str(p) for p in sorted(base.glob(f"*/{sub}")) if p.is_dir()]
    dirs = list(dict.fromkeys(dirs))  # dedupe, keep order
    for d in dirs:
        try:
            os.add_dll_directory(d)
        except OSError:
            pass
    os.environ["PATH"] = os.pathsep.join(dirs) + os.pathsep + os.environ.get("PATH", "")
    _DLL_DIRS = dirs
    return dirs


def cuda_available() -> bool:
    """True when onnxruntime-gpu is importable and its CUDA EP loads."""
    try:
        import onnxruntime as ort
    except ImportError:
        return False
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return False
    ensure_cuda_dlls()
    return True


def create_session(model_path: Path | str, prefer_gpu: bool = True):
    """ONNX session on CUDA (falls back to CPU, records which was used)."""
    ensure_cuda_dlls()
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3  # hide info noise, keep warnings/errors
    key = str(model_path)
    if not prefer_gpu or not cuda_available():
        _BACKENDS[key] = "CPU"
        return ort.InferenceSession(key, so, providers=["CPUExecutionProvider"])

    providers = [
        ("CUDAExecutionProvider", {
            "device_id": 0,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "HEURISTIC",  # DEFAULT benchmarks at startup
            "do_copy_in_default_stream": True,
        }),
        "CPUExecutionProvider",
    ]
    try:
        sess = ort.InferenceSession(key, so, providers=providers)
        used = sess.get_providers()[0]
        _BACKENDS[key] = "CUDA" if used == "CUDAExecutionProvider" else "CPU"
        return sess
    except Exception as exc:  # noqa: BLE001 - fall back rather than die
        print(f"  [gpu_core] CUDA session failed for {Path(key).name}: {exc}")
        _BACKENDS[key] = "CPU"
        return ort.InferenceSession(key, so, providers=["CPUExecutionProvider"])


def backend_report() -> dict[str, str]:
    return {Path(k).name: v for k, v in _BACKENDS.items()}


def report_str() -> str:
    b = backend_report()
    return ", ".join(f"{k}={v}" for k, v in b.items()) if b else "no models loaded"


# --------------------------------------------------------------- region base
class BaseRegionExtractor:
    """Shared ROI logic: mean R,G,B per region given landmarks (MediaPipe ids)."""

    def __init__(self, regions: dict | None = None, roi_r: float | None = None):
        if regions is None or roi_r is None:
            from deepfake_detect import REGIONS, ROI_R  # lazy: avoid import cycle
            regions = regions or REGIONS
            roi_r = ROI_R if roi_r is None else roi_r
        self.regions = regions
        self.roi_r = roi_r

    @staticmethod
    def _mean_bgr(frame, mask):
        px = frame[mask > 0]
        if len(px) < 10:
            return (np.nan, np.nan, np.nan)
        return (float(px[:, 2].mean()), float(px[:, 1].mean()), float(px[:, 0].mean()))  # R,G,B

    def region_rgb(self, frame, lm):
        from rppg import EXCLUDE

        h, w = frame.shape[:2]
        lm_i = lm.astype(np.int32)
        hull = cv2.convexHull(lm_i)
        scale = np.sqrt(max(cv2.contourArea(hull), 1.0))
        out = {}
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [hull], 255)
        for idx in EXCLUDE:
            cv2.fillPoly(mask, [lm_i[idx].reshape(1, 1, 2)], 0)
        out["full"] = self._mean_bgr(frame, mask)
        ax = max(4, int(self.roi_r * scale))
        for name, idxs in self.regions.items():
            c = tuple(lm[idxs].mean(axis=0).astype(int))
            m = np.zeros((h, w), np.uint8)
            cv2.ellipse(m, c, (ax, ax), 0, 0, 360, 255, -1)
            out[name] = self._mean_bgr(frame, m)
        return out


# ------------------------------------------------------------------- SCRFD
class SCRFD:
    """SCRFD-10G face detector on CUDA (buffalo_l). Decoded per insightface."""

    def __init__(self, model: Path = DET_MODEL, input_size: int = 640,
                 conf_thr: float = 0.5, nms_thr: float = 0.4, gpu: bool = True,
                 center_offset: float = 0.0, scale_preds: bool = True):
        self.sess = create_session(model, prefer_gpu=gpu)
        self.backend = _BACKENDS[str(model)]
        self.input_size = input_size
        self.conf_thr = conf_thr
        self.nms_thr = nms_thr
        self.in_name = self.sess.get_inputs()[0].name
        # this det_10g export: 2 anchors per FPN cell (NHWC-flattened heads)
        self.center_offset = center_offset   # fraction of stride (0.0 or 0.5)
        self.scale_preds = scale_preds       # raw preds are in stride units
        self.strides = (8, 16, 32)

    def detect(self, frame_bgr):
        """-> (boxes (N,4) x1y1x2y2 float, kps (N,5), scores (N,)) in input px."""
        h, w = frame_bgr.shape[:2]
        scale = self.input_size / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_size, self.input_size, 3), 114, np.float32)
        canvas[:nh, :nw] = resized
        blob = (canvas - 127.5) / 128.0
        blob = blob.transpose(2, 0, 1)[None]

        outs = self.sess.run(None, {self.in_name: blob})
        fmc = 3
        all_boxes, all_kps, all_scores = [], [], []
        for i, stride in enumerate(self.strides[:fmc]):
            fh = fw = self.input_size // stride
            cells = fh * fw
            sc = outs[i]
            bp = outs[i + fmc]
            kp = outs[i + 2 * fmc]
            if len(sc) % cells:            # unexpected layout; skip level
                continue
            a = len(sc) // cells           # anchors per cell (2 for det_10g)
            scores = sc.reshape(cells, a)
            bpreds = bp.reshape(cells, a, 4)
            kpreds = kp.reshape(cells, a, 10)
            ys, xs = np.mgrid[0:fh, 0:fw]
            centers = np.stack([xs, ys], axis=-1).reshape(-1, 2).astype(np.float32)
            centers = centers * float(stride) + self.center_offset * float(stride)

            hit = np.argwhere(scores > self.conf_thr)
            if hit.size == 0:
                continue
            cell_i, anc_i = hit[:, 0], hit[:, 1]
            c = centers[cell_i]
            d = bpreds[cell_i, anc_i]
            if self.scale_preds:
                d = d * float(stride)
            boxes = np.stack([c[:, 0] - d[:, 0], c[:, 1] - d[:, 1],
                              c[:, 0] + d[:, 2], c[:, 1] + d[:, 3]], axis=1)
            k = kpreds[cell_i, anc_i]
            if self.scale_preds:
                k = k * float(stride)
            kps = np.empty((c.shape[0], 5, 2), np.float32)
            kps[:, :, 0] = c[:, None, 0] + k[:, 0::2]
            kps[:, :, 1] = c[:, None, 1] + k[:, 1::2]
            all_boxes.append(boxes)
            all_kps.append(kps)
            all_scores.append(scores[cell_i, anc_i])

        if not all_boxes:
            return (np.zeros((0, 4), np.float32), np.zeros((0, 5, 2), np.float32),
                    np.zeros((0,), np.float32))

        boxes = np.concatenate(all_boxes)
        kps = np.concatenate(all_kps)
        scores = np.concatenate(all_scores)
        idx = self._nms(boxes, scores, self.nms_thr)
        boxes, kps, scores = boxes[idx], kps[idx], scores[idx]

        # undo letterbox: scale back + drop padding
        boxes /= scale
        kps /= scale
        return boxes, kps, scores

    @staticmethod
    def _nms(boxes, scores, thr):
        order = scores.argsort()[::-1]
        keep = []
        while order.size:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
            yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
            yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
            iou = inter / (area_i + area_r - inter + 1e-9)
            order = rest[iou <= thr]
        return np.array(keep, np.int64)

    def largest_face(self, frame_bgr):
        """-> (box, kps, score) of the biggest face or None."""
        boxes, kps, scores = self.detect(frame_bgr)
        if len(boxes) == 0:
            return None
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        i = int(np.argmax(areas))
        return boxes[i], kps[i], float(scores[i])


# ------------------------------------------------------------- face mesh (GPU)
class GPUMeshLandmarker:
    """478 MediaPipe-compatible landmarks from a face crop, on CUDA.

    Crop geometry + input normalisation fitted against the MediaPipe CPU
    reference over BOTH dataset subjects (tools/fit_crop.py 1 2,
    tools/test_crop_variants.py):
      - absolute crop 191x229 px (clamped to >= 1.2*box so small faces
        never get a crop smaller than their box),
      - centre offset dx = -0.598*bw + 83.4, dy = -0.721*bh + 138.4 (px).
    This beats a purely proportional crop (1.61*bw, 1.33*bh, +0.11/+0.07):
    6.3/9.1 px mean landmark error on subjects 1/2 vs 8.0/11.8 px. The
    optimal crop is nearly constant in *absolute* pixels across both
    subjects' face sizes, which a ratio fit can't express. Intrinsic mesh
    residual vs MediaPipe is ~5-6 px, so this is near the model floor.
    Constants are calibrated on these subjects' face size range
    (~110-140 px SCRFD boxes); not guaranteed at other resolutions.
    Input normalisation: float BGR / 255 (tools/fit_input.py).
    """

    ABS_W = 191.0     # preferred crop width in px (calibration-set mean)
    ABS_H = 229.0     # preferred crop height in px
    MIN_CROP = 1.2    # never smaller than this multiple of the det box
    DX_K, DX_B = -0.598, 83.4    # centre x offset = DX_K*bw + DX_B (px)
    DY_K, DY_B = -0.721, 138.4   # centre y offset = DY_K*bh + DY_B (px)
    SIZE = 256

    def __init__(self, model: Path = MESH_MODEL, gpu: bool = True):
        self.sess = create_session(model, prefer_gpu=gpu)
        self.backend = _BACKENDS[str(model)]
        self.in_name = self.sess.get_inputs()[0].name
        self.detector = SCRFD(gpu=gpu)
        self._last_box = None
        self.layout = "xyz"  # validated vs MediaPipe by tools/validate_gpu.py

    def _crop_rect(self, box, h, w):
        """-> (x0, y0, cw, ch) of the fitted landmark-model crop."""
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        cw = max(self.ABS_W, self.MIN_CROP * bw)
        ch = max(self.ABS_H, self.MIN_CROP * bh)
        cx = (x1 + x2) / 2.0 + self.DX_K * bw + self.DX_B
        cy = (y1 + y2) / 2.0 + self.DY_K * bh + self.DY_B
        return cx - cw / 2.0, cy - ch / 2.0, cw, ch

    def landmarks(self, frame_bgr, ts_ms=None):
        """Return (478,2) pixel landmarks (MediaPipe index space) or None."""
        det = self.detector.largest_face(frame_bgr)
        if det is None:
            det_box = self._last_box
            if det_box is None:
                return None
        else:
            det_box = det[0]
            self._last_box = det_box

        h, w = frame_bgr.shape[:2]
        x0, y0, cw, ch = self._crop_rect(det_box, h, w)
        mat = np.array([[cw / self.SIZE, 0, x0], [0, ch / self.SIZE, y0]], np.float32)
        crop = cv2.warpAffine(frame_bgr, mat, (self.SIZE, self.SIZE),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        blob = crop.astype(np.float32) / 255.0  # verified: 0..1 BGR, tools/fit_input.py
        out = self.sess.run(None, {self.in_name: blob[None]})
        if self.layout == "xyz":
            pts = out[0].reshape(-1, 3)[:478, :2]       # (478,2) crop-pixel coords
        else:  # blocked: [all x][all y][all z]
            flat = out[0].reshape(-1)
            pts = np.stack([flat[:478], flat[478:956]], axis=1)
        orig = pts @ mat[:, :2].T + mat[:, 2]           # crop -> frame coords
        return orig.astype(np.float32)

    def close(self):
        self.detector = None
        self.sess = None


# ----------------------------------------------------------------- ArcFace
class ArcFace:
    """w600k_r50 identity embedder (112px input), on CUDA."""

    def __init__(self, model: Path = ARCFACE_MODEL, gpu: bool = True):
        self.sess = create_session(model, prefer_gpu=gpu)
        self.backend = _BACKENDS[str(model)]
        self.in_name = self.sess.get_inputs()[0].name

    @staticmethod
    def align(frame_bgr, kps5):
        """Similarity-warp face to 112x112 arcface crop (kps: 5x2)."""
        M, _ = cv2.estimateAffinePartial2D(kps5.astype(np.float32), ARCFACE_TEMPLATE_112,
                                           method=cv2.LMEDS)
        if M is None:
            return None, None
        crop = cv2.warpAffine(frame_bgr, M, (112, 112), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
        return crop, M

    def embed(self, frame_bgr, kps5) -> np.ndarray | None:
        crop, _ = self.align(frame_bgr, kps5)
        if crop is None:
            return None
        blob = (crop.astype(np.float32) - 127.5) / 127.5
        blob = blob.transpose(2, 0, 1)[None][:, ::-1]  # BGR -> RGB, NCHW
        emb = self.sess.run(None, {self.in_name: blob})[0][0]
        n = float(np.linalg.norm(emb))
        return (emb / n).astype(np.float32) if n > 1e-8 else None


# --------------------------------------------------------------- inswapper
class Inswapper:
    """inswapper_128 face-swap generator on CUDA.

    This ONNX takes float32 NCHW 'target' (1,3,128,128) + 'source' (1,512).
    Normalisation validated by tools/validate_gpu.py: input 0..1, output 0..1.
    """

    def __init__(self, model: Path = INSWAPPER_MODEL, gpu: bool = True,
                 in_scale: float = 1.0 / 255.0, in_bias: float = 0.0,
                 out_scale: float = 255.0, out_bias: float = 0.0):
        self.sess = create_session(model, prefer_gpu=gpu)
        self.backend = _BACKENDS[str(model)]
        self.in_target = self.sess.get_inputs()[0].name
        self.in_source = self.sess.get_inputs()[1].name
        self.in_scale = in_scale
        self.in_bias = in_bias
        self.out_scale = out_scale
        self.out_bias = out_bias
        self.template_128 = ARCFACE_TEMPLATE_112 * (128.0 / 112.0)

    def align_target(self, frame_bgr, kps5):
        """-> (aimg 128x128 BGR, M affine frame->crop) or (None, None)."""
        M, _ = cv2.estimateAffinePartial2D(kps5.astype(np.float32), self.template_128,
                                           method=cv2.LMEDS)
        if M is None:
            return None, None
        aimg = cv2.warpAffine(frame_bgr, M, (128, 128), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
        return aimg, M

    def swap_face(self, aimg_bgr: np.ndarray, source_emb: np.ndarray) -> np.ndarray:
        """128x128 swapped face (BGR uint8)."""
        x = aimg_bgr.astype(np.float32) * self.in_scale + self.in_bias
        x = x.transpose(2, 0, 1)[None]
        out = self.sess.run(None, {self.in_target: x,
                                   self.in_source: source_emb[None].astype(np.float32)})[0]
        out = out[0].transpose(1, 2, 0) * self.out_scale + self.out_bias
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def paste_back(frame_bgr, swapped_128, M) -> np.ndarray:
        """Warp swapped patch back into the frame with a feathered seamless blend."""
        h, w = frame_bgr.shape[:2]
        Minv = cv2.invertAffineTransform(M)
        full = cv2.warpAffine(swapped_128, Minv, (w, h), flags=cv2.INTER_LINEAR,
                              borderValue=0)
        m = cv2.warpAffine(np.full((128, 128), 255, np.uint8), Minv, (w, h),
                           flags=cv2.INTER_NEAREST)
        # shrink blend region away from the face outline, then feather
        k = max(3, int(0.05 * m.mean() ** 0.5 + 3))
        m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        soft = cv2.GaussianBlur(m, (0, 0), sigmaX=3)
        soft = (soft.astype(np.float32) / 255.0)[..., None]
        blended = frame_bgr.astype(np.float32) * (1 - soft) + full.astype(np.float32) * soft
        out = np.clip(blended, 0, 255).astype(np.uint8)
        return out


# ------------------------------------------------------- GPU region extractor
class _OneEuro:
    """Vectorised One Euro filter for (N,2) landmarks (Casiez et al. 2012).

    MediaPipe VIDEO mode smooths landmarks via internal temporal tracking; the
    ONNX mesh is per-frame independent, so its ~6-9 px error includes high-
    frequency jitter that displaces the small rPPG ROIs frame to frame. One
    Euro filters that jitter while adapting to head motion (cutoff rises with
    speed, so tracking lag stays small when it matters). Cuts frame-to-frame
    jitter ~5x (8-10 px -> ~1.7-2.2 px RMS of second differences).
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.05,
                 d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x: np.ndarray | None = None
        self.dx: np.ndarray | None = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * np.asarray(cutoff, dtype=np.float64))
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, dt: float) -> np.ndarray:
        dt = float(min(max(dt, 1e-3), 1.0))
        if self.x is None or self.x.shape != x.shape:
            self.x = x.astype(np.float64).copy()
            self.dx = np.zeros_like(self.x)
            return x.copy()
        dx_raw = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx_raw + (1.0 - a_d) * self.dx
        a = self._alpha(self.min_cutoff + self.beta * np.abs(self.dx), dt)
        self.x = a * x + (1.0 - a) * self.x
        return self.x.astype(np.float32).copy()


class GPURegionExtractor(BaseRegionExtractor):
    """drop-in replacement for deepfake_detect.RegionExtractor (runs on CUDA).

    Landmarks are One-Euro-smoothed by default (smooth=True) to match
    MediaPipe's internal temporal tracking; without it, per-frame jitter
    degrades the rPPG traces (see research.md GPU-vs-MediaPipe section).
    """

    def __init__(self, regions: dict | None = None, roi_r: float | None = None,
                 gpu: bool = True, mesh_model: Path = MESH_MODEL,
                 smooth: bool = True):
        super().__init__(regions, roi_r)
        self.mesh = GPUMeshLandmarker(model=mesh_model, gpu=gpu)
        self.n_infer = 0
        self._filt = _OneEuro() if smooth else None
        self._last_ts: float | None = None

    def landmarks(self, frame_bgr, ts_ms=None):
        lm = self.mesh.landmarks(frame_bgr, ts_ms)
        if lm is not None:
            self.n_infer += 1
            if self._filt is not None:
                if ts_ms is not None and self._last_ts is not None:
                    dt = (ts_ms - self._last_ts) / 1000.0
                else:
                    dt = 1.0 / 30.0
                if ts_ms is not None:
                    self._last_ts = float(ts_ms)
                lm = self._filt(lm, dt)
        return lm

    def close(self):
        self.mesh.close()


# ---------------------------------------------------------------- GPU monitor
class GpuMonitor:
    """Samples nvidia-smi (util %, memory, power) in a background thread."""

    QUERY = ("name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw")

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.samples: list[tuple] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.label = ""

    def _loop(self):
        while not self._stop.is_set():
            try:
                raw = subprocess.run(
                    ["nvidia-smi", f"--query-gpu={self.QUERY}",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10, check=True).stdout.strip()
                parts = [p.strip() for p in raw.splitlines()[0].split(",")]
                if len(parts) == 6:
                    self.samples.append(tuple(parts))
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(self.interval)

    def start(self, label: str = ""):
        self.label = label
        self.samples.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict:
        """Stop sampling -> summary dict (also prints a one-liner)."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=15)
        if not self.samples:
            return {"label": self.label, "n": 0}
        util = [float(s[1]) for s in self.samples]
        mem = [float(s[3]) for s in self.samples]   # memory.used (MB)
        pow_ = [float(s[5]) for s in self.samples if s[5] != "[N/A]"]
        summary = {
            "label": self.label, "n": len(self.samples),
            "gpu_util_mean": round(float(np.mean(util)), 1),
            "gpu_util_max": round(float(np.max(util)), 1),
            "mem_mean_mb": round(float(np.mean(mem)), 0),
            "mem_max_mb": round(float(np.max(mem)), 0),
            "power_mean_w": round(float(np.mean(pow_)), 1) if pow_ else None,
            "power_max_w": round(float(np.max(pow_)), 1) if pow_ else None,
            "name": self.samples[0][0],
        }
        print(f"  [gpu] {summary['label'] or 'stage'}: util mean {summary['gpu_util_mean']}% "
              f"max {summary['gpu_util_max']}%, mem max {summary['mem_max_mb']:.0f} MB, "
              f"power mean {summary['power_mean_w']} W")
        return summary

    def save_csv(self, path: Path):
        path.parent.mkdir(exist_ok=True)
        with path.open("w") as f:
            f.write("name,util_gpu,util_mem,mem_used_mb,mem_total_mb,power_w\n")
            for s in self.samples:
                f.write(",".join(str(x) for x in s) + "\n")

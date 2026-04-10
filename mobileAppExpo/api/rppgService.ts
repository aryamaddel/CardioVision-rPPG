// src/api/rppgService.ts
import axios from "axios";
import Constants from "expo-constants";

function detectExpoHost(): string | null {
  const c = Constants as any;
  const hostUri: string | undefined =
    c?.expoConfig?.hostUri ??
    c?.manifest2?.extra?.expoClient?.hostUri ??
    c?.manifest?.debuggerHost;

  if (!hostUri || typeof hostUri !== "string") return null;
  return hostUri.split(":")[0] ?? null;
}

const DEFAULT_HOST = detectExpoHost() ?? "127.0.0.1";
const BASE_URL =
  process.env.EXPO_PUBLIC_BASE_URL ?? `http://${DEFAULT_HOST}:5000`;
const BACKEND_HOST =
  process.env.EXPO_PUBLIC_BACKEND_HOST ??
  BASE_URL.replace(/^https?:\/\//, "").split(":")[0];
const WS_URL = process.env.EXPO_PUBLIC_WS_URL ?? `ws://${BACKEND_HOST}:8765`;

const api = axios.create({ baseURL: BASE_URL, timeout: 120_000 });

export interface HRVFeatures {
  rmssd_ms: number;
  sdnn_ms: number;
  lf_hf_ratio: number;
  stress_index: number;
  stress_level: "Low" | "Medium" | "High" | "Unknown";
}

export interface ConfidenceDetails {
  final_score: number;
  is_reliable: boolean;
  ibi_regularity: number;
  snr: number;
  density: number;
  duration: number;
  dominant_bpm?: number;
}

export interface RPPGResult {
  pulse_signal: number[];
  timestamps: number[];
  peaks_idx: number[];
  ibi_ms: number[];
  fps: number;
  bpm: number | null;
  bpm_mean: number | null;
  confidence: number;
  is_reliable: boolean;
  confidence_details: ConfidenceDetails;
  motion_fraction: number;
  hrv_features: HRVFeatures;
  method_used: string;
  deep_model_used?: string;
  pos_snr?: number;
  deep_snr?: number;

  duration_sec: number;
  frames_processed: number;
  n_frames: number;
  status: string;
}

interface LiveMetric {
  bpm: number | null;
  confidence: number;
  method: string;
}

interface LiveFrameResult {
  type: "frame_result";
  metric?: LiveMetric;
  bpm?: number | null;
  confidence?: number;
  method?: string;
  has_face: boolean;
  method_changed?: boolean;
  identity_locked?: boolean;
  identity_match?: boolean;
  intruder_detected?: boolean;
  overlay_jpeg_b64?: string | null;
  overlay?: string | null;
}

type LiveSocketMessage =
  | LiveFrameResult
  | { type: "ack"; status: string }
  | { type: "final_result"; result: RPPGResult }
  | { type: "error"; message: string }
  | { type: "pong" };

interface LiveClientOptions {
  onFrame?: (frame: LiveFrameResult) => void;
  onFinal?: (result: RPPGResult) => void;
  onError?: (message: string) => void;
  onClose?: () => void;
  overlayQuality?: number;
  overlayMaxSide?: number;
  overlayStride?: number;
}

export class LiveRPPGClient {
  private ws: WebSocket | null = null;
  private readonly options: LiveClientOptions;
  private isOpen = false;
  private finalResolver: ((result: RPPGResult) => void) | null = null;
  private finalRejecter: ((reason?: unknown) => void) | null = null;

  constructor(options: LiveClientOptions = {}) {
    this.options = options;
  }

  connect(timeoutMs: number = 4000): Promise<void> {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(WS_URL);
      this.ws = ws;
      let settled = false;
      const timer = setTimeout(() => {
        if (settled) return;
        settled = true;
        try {
          ws.close();
        } catch {}
        reject(new Error("Live stream connection timed out"));
      }, timeoutMs);

      ws.onopen = () => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        this.isOpen = true;
        ws.send(
          JSON.stringify({
            type: "start",
            client: "expo_mobile",
            overlay_quality: Math.max(
              1,
              Math.min(100, Math.round(this.options.overlayQuality ?? 45)),
            ),
            overlay_max_side: Math.max(
              0,
              Math.round(this.options.overlayMaxSide ?? 320),
            ),
            overlay_stride: Math.max(
              1,
              Math.round(this.options.overlayStride ?? 2),
            ),
          }),
        );
        resolve();
      };

      ws.onerror = () => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        const msg = `Live stream connection failed (${WS_URL})`;
        this.options.onError?.(msg);
        reject(new Error(msg));
      };

      ws.onclose = () => {
        clearTimeout(timer);
        this.isOpen = false;
        this.options.onClose?.();
      };

      ws.onmessage = (event) => {
        const parsed = this.parseMessage(event.data);
        if (!parsed) return;

        if (parsed.type === "frame_result") {
          this.options.onFrame?.(parsed);
          return;
        }

        if (parsed.type === "final_result") {
          this.options.onFinal?.(parsed.result);
          this.finalResolver?.(parsed.result);
          this.finalResolver = null;
          this.finalRejecter = null;
          return;
        }

        if (parsed.type === "error") {
          this.options.onError?.(parsed.message);
          this.finalRejecter?.(new Error(parsed.message));
          this.finalResolver = null;
          this.finalRejecter = null;
        }
      };
    });
  }

  sendFrameBinary(frameJpegB64: string, tsMs?: number): boolean {
    if (!this.ws || !this.isOpen) return false;
    try {
      // Decode base64 to raw bytes and send as binary WebSocket message.
      // The server handles binary payloads directly — no JSON parse overhead.
      const binaryStr = atob(frameJpegB64);
      const bytes = new Uint8Array(binaryStr.length);
      for (let i = 0; i < binaryStr.length; i++) {
        bytes[i] = binaryStr.charCodeAt(i);
      }
      this.ws.send(bytes.buffer);
    } catch {
      // Fallback to JSON if binary encoding fails
      return this.sendFrameBase64(frameJpegB64, tsMs);
    }
    return true;
  }

  sendFrameBase64(frameJpegB64: string, tsMs?: number): boolean {
    if (!this.ws || !this.isOpen) return false;
    this.ws.send(
      JSON.stringify({
        type: "frame",
        frame_jpeg_b64: frameJpegB64,
        ts_ms: tsMs ?? Date.now(),
      }),
    );
    return true;
  }

  stopAndGetFinalResult(timeoutMs: number = 15000): Promise<RPPGResult> {
    if (!this.ws || !this.isOpen) {
      return Promise.reject(new Error("Live socket is not connected"));
    }

    return new Promise<RPPGResult>((resolve, reject) => {
      this.finalResolver = resolve;
      this.finalRejecter = reject;

      this.ws?.send(JSON.stringify({ type: "stop" }));
      setTimeout(() => {
        if (!this.finalResolver) return;
        this.finalResolver = null;
        this.finalRejecter = null;
        reject(new Error("Timed out waiting for final stream result"));
      }, timeoutMs);
    });
  }

  disconnect() {
    this.isOpen = false;
    this.ws?.close();
    this.ws = null;
  }

  private parseMessage(data: string): LiveSocketMessage | null {
    try {
      return JSON.parse(data) as LiveSocketMessage;
    } catch {
      return null;
    }
  }
}

export async function processVideo(
  videoUri: string,
  onProgress?: (pct: number) => void,
): Promise<RPPGResult> {
  const formData = new FormData();
  formData.append("video", {
    uri: videoUri,
    type: "video/mp4",
    name: "recording.mp4",
  } as any);
  const response = await api.post<RPPGResult>("/process", formData, {
    headers: { "Content-Type": "multipart/form-data" },
    onUploadProgress: (evt) => {
      if (evt.total && onProgress)
        onProgress(Math.round((evt.loaded / evt.total) * 50));
    },
  });
  return response.data;
}

export function getMockResult(): RPPGResult {
  const fps = 30,
    duration = 30,
    n = fps * duration,
    bpm = 72,
    freq = bpm / 60;
  const pulseSignal: number[] = [],
    timestamps: number[] = [];
  for (let i = 0; i < n; i++) {
    const t = i / fps;
    timestamps.push(t);
    pulseSignal.push(
      Math.sin(2 * Math.PI * freq * t) +
        0.3 * Math.sin(4 * Math.PI * freq * t) +
        0.05 * (Math.random() - 0.5),
    );
  }
  const ibi_ms: number[] = [];
  const avgIBI = 60000 / bpm;
  let accTime = avgIBI;
  while (accTime < duration * 1000) {
    const jitter = (Math.random() - 0.5) * 40;
    ibi_ms.push(avgIBI + jitter);
    accTime += avgIBI + jitter;
  }
  const rmssd = Math.sqrt(
    ibi_ms.slice(1).reduce((s, v, i) => s + Math.pow(v - ibi_ms[i], 2), 0) /
      (ibi_ms.length - 1),
  );
  return {
    pulse_signal: pulseSignal,
    timestamps,
    peaks_idx: Array.from(
      { length: Math.floor((duration * bpm) / 60) },
      (_, i) => Math.round((i * fps * 60) / bpm),
    ),
    ibi_ms,
    fps,
    bpm,
    bpm_mean: bpm + 0.4,
    confidence: 0.81,
    is_reliable: true,
    confidence_details: {
      final_score: 0.81,
      is_reliable: true,
      ibi_regularity: 0.88,
      snr: 0.79,
      density: 0.92,
      duration: 1.0,
      dominant_bpm: bpm,
    },
    motion_fraction: 0.04,
    hrv_features: {
      rmssd_ms: rmssd,
      sdnn_ms: 42.3,
      lf_hf_ratio: 0.87,
      stress_index: 22,
      stress_level: "Low",
    },
    method_used: "pos+deep_ensemble",
    deep_model_used: "PhysFormer.pure",
    pos_snr: 8.4,
    deep_snr: 11.2,
    duration_sec: 30,
    frames_processed: n,
    n_frames: n,
    status: "success",
  };
}

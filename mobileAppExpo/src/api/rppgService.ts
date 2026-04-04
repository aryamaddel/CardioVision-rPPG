// src/api/rppgService.ts
import axios from 'axios';

const BASE_URL = 'http://192.168.1.100:5000'; // ← UPDATE to your IP

const api = axios.create({ baseURL: BASE_URL, timeout: 120_000 });

export interface HRVFeatures {
  rmssd_ms: number;
  sdnn_ms: number;
  lf_hf_ratio: number;
  stress_index: number;
  stress_level: 'Low' | 'Medium' | 'High' | 'Unknown';
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
  triage_mode: string;
  triage_reason: string;
  visual_stress: number;
  duration_sec: number;
  frames_processed: number;
  n_frames: number;
  status: string;
}

export async function processVideo(
  videoUri: string,
  onProgress?: (pct: number) => void,
): Promise<RPPGResult> {
  const formData = new FormData();
  formData.append('video', { uri: videoUri, type: 'video/mp4', name: 'recording.mp4' } as any);
  const response = await api.post<RPPGResult>('/process', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (evt) => {
      if (evt.total && onProgress) onProgress(Math.round((evt.loaded / evt.total) * 50));
    },
  });
  return response.data;
}

export async function checkHealth(): Promise<boolean> {
  try { await api.get('/health', { timeout: 3000 }); return true; } catch { return false; }
}

export function getMockResult(): RPPGResult {
  const fps = 30, duration = 30, n = fps * duration, bpm = 72, freq = bpm / 60;
  const pulseSignal: number[] = [], timestamps: number[] = [];
  for (let i = 0; i < n; i++) {
    const t = i / fps; timestamps.push(t);
    pulseSignal.push(Math.sin(2 * Math.PI * freq * t) + 0.3 * Math.sin(4 * Math.PI * freq * t) + 0.05 * (Math.random() - 0.5));
  }
  const ibi_ms: number[] = [];
  const avgIBI = 60000 / bpm;
  let accTime = avgIBI;
  while (accTime < duration * 1000) {
    const jitter = (Math.random() - 0.5) * 40;
    ibi_ms.push(avgIBI + jitter); accTime += avgIBI + jitter;
  }
  const rmssd = Math.sqrt(ibi_ms.slice(1).reduce((s, v, i) => s + Math.pow(v - ibi_ms[i], 2), 0) / (ibi_ms.length - 1));
  return {
    pulse_signal: pulseSignal, timestamps,
    peaks_idx: Array.from({ length: Math.floor(duration * bpm / 60) }, (_, i) => Math.round((i * fps * 60) / bpm)),
    ibi_ms, fps, bpm, bpm_mean: bpm + 0.4,
    confidence: 0.81, is_reliable: true,
    confidence_details: { final_score: 0.81, is_reliable: true, ibi_regularity: 0.88, snr: 0.79, density: 0.92, duration: 1.0, dominant_bpm: bpm },
    motion_fraction: 0.04,
    hrv_features: { rmssd_ms: rmssd, sdnn_ms: 42.3, lf_hf_ratio: 0.87, stress_index: 22, stress_level: 'Low' },
    method_used: 'pos+deep_ensemble', deep_model_used: 'PhysFormer.pure',
    pos_snr: 8.4, deep_snr: 11.2,
    triage_mode: 'BIOMETRIC', triage_reason: 'Signal reliable (confidence=0.81)', visual_stress: 0,
    duration_sec: 30, frames_processed: n, n_frames: n, status: 'success',
  };
}

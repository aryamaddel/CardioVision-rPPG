// src/api/rppgService.ts
// Communicates with the Python CardioVision backend

import axios from 'axios';

// ─── CONFIG ──────────────────────────────────────────────────────────────────
// Change this to your laptop's local IP when testing on device
// Run: `ifconfig | grep "inet " | grep -v 127.0.0.1` to find your IP
const BASE_URL = 'http://192.168.1.100:5000';  // ← UPDATE THIS

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 120_000, // 2 min timeout for video processing
});

// ─── TYPES ────────────────────────────────────────────────────────────────────

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

export interface TriageInfo {
  mode: 'BIOMETRIC' | 'VISUAL_ASSESSMENT';
  reason: string;
  visual_stress_score: number;
  visual_stress_label: string;
}

export interface RPPGResult {
  // Signal
  pulse_signal: number[];
  timestamps: number[];
  peaks_idx: number[];
  ibi_ms: number[];
  fps: number;

  // Core vitals
  bpm: number | null;
  bpm_mean: number | null;

  // Quality
  confidence: number;
  is_reliable: boolean;
  confidence_details: ConfidenceDetails;
  motion_fraction: number;

  // HRV & Stress
  hrv_features: HRVFeatures;

  // Method
  method_used: string;
  deep_model_used?: string;
  pos_snr?: number;
  deep_snr?: number;

  // Triage
  triage_mode: string;
  triage_reason: string;
  visual_stress: number;

  // Meta
  duration_sec: number;
  frames_processed: number;
  n_frames: number;
  status: string;
}

// ─── UPLOAD & PROCESS VIDEO ──────────────────────────────────────────────────

export async function processVideo(
  videoUri: string,
  onProgress?: (pct: number) => void,
): Promise<RPPGResult> {
  const formData = new FormData();

  // React Native FormData file append
  formData.append('video', {
    uri: videoUri,
    type: 'video/mp4',
    name: 'recording.mp4',
  } as any);

  const response = await api.post<RPPGResult>('/process', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (evt) => {
      if (evt.total && onProgress) {
        onProgress(Math.round((evt.loaded / evt.total) * 50));
      }
    },
  });

  return response.data;
}

// ─── GET PROCESSED VIDEO (green channel viz) ─────────────────────────────────

export function getProcessedVideoUrl(jobId: string): string {
  return `${BASE_URL}/processed-video/${jobId}`;
}

// ─── HEALTH CHECK ─────────────────────────────────────────────────────────────

export async function checkHealth(): Promise<boolean> {
  try {
    await api.get('/health', { timeout: 3000 });
    return true;
  } catch {
    return false;
  }
}

// ─── MOCK RESULT (for UI development without backend) ─────────────────────────
// Remove this and use real API in production

export function getMockResult(): RPPGResult {
  // Generate realistic pulse waveform (30s @ 30fps = 900 points)
  const fps = 30;
  const duration = 30;
  const n = fps * duration;
  const bpm = 68;
  const freq = bpm / 60;

  const pulseSignal: number[] = [];
  const timestamps: number[] = [];
  for (let i = 0; i < n; i++) {
    const t = i / fps;
    timestamps.push(t);
    const base = Math.sin(2 * Math.PI * freq * t);
    const harmonic = 0.3 * Math.sin(4 * Math.PI * freq * t);
    const noise = 0.05 * (Math.random() - 0.5);
    pulseSignal.push(base + harmonic + noise);
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
    peaks_idx: Array.from({ length: Math.floor(duration * bpm / 60) }, (_, i) =>
      Math.round((i * fps * 60) / bpm),
    ),
    ibi_ms,
    fps,
    bpm: bpm,
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
      stress_level: 'Low',
    },
    method_used: 'pos+deep_ensemble',
    deep_model_used: 'PhysFormer.pure',
    pos_snr: 8.4,
    deep_snr: 11.2,
    triage_mode: 'BIOMETRIC',
    triage_reason: 'Signal reliable (confidence=0.81)',
    visual_stress: 0,
    duration_sec: 30,
    frames_processed: n,
    n_frames: n,
    status: 'success',
  };
}

# CardioVision — React Native App

Contact-free cardiac monitoring via rPPG. Monochromatic "Clinical Noir" design.

## Architecture

```
React Native App (Expo)
       │
       │  POST /process (multipart video)
       ▼
Flask API Server (server.py)
       │
       ├── FaceROIExtractor (main.py)
       ├── POS Algorithm (rppg_core.py)
       ├── PhysFormer Deep Model (deep_rppg.py)
       ├── HRV + Stress (rppg_core.py)
       └── Triage Agent (triage_agent.py)
```

## Screens

| Screen | Description |
|--------|-------------|
| **HomeScreen** | Hero with live BPM preview, ECG animation, pipeline breakdown |
| **RecordScreen** | 30s camera capture with face guide oval, countdown, signal quality meter |
| **ProcessingScreen** | Animated 8-step pipeline progress |
| **ResultsScreen** | Full biometric dashboard: BPM, waveform, IBI chart, HRV, stress, health tips |
| **VideoPlaybackScreen** | Playback with green channel overlay + real pulse markers on face |

## Setup

### 1. Backend (Python)

```bash
cd your_rppg_project/

pip install flask flask-cors

# Update your rppg_core.py to include process_rppg_with_deep()
# (see previous conversation for that code)

python server.py
# → Running on http://0.0.0.0:5000
```

Find your laptop IP:
```bash
# macOS/Linux
ifconfig | grep "inet " | grep -v 127.0.0.1
# Windows
ipconfig | findstr "IPv4"
```

### 2. React Native App

```bash
cd cardio_app/

npm install

# Update the IP in src/api/rppgService.ts:
# const BASE_URL = 'http://YOUR_LAPTOP_IP:5000';
```

**Run on device:**
```bash
npx expo start
# Scan QR code with Expo Go app
```

**Run on simulator:**
```bash
npx expo start --ios
npx expo start --android
```

### 3. Test without backend

The app uses `getMockResult()` as automatic fallback when the backend
is unavailable. You'll see realistic fake data for UI testing.

## Key Files

```
cardio_app/
├── App.tsx                          # Navigation root
├── server.py                        # Flask API (copy to your rPPG folder)
├── src/
│   ├── api/rppgService.ts           # API calls + mock data
│   ├── theme/index.ts               # Design system + health tips
│   ├── screens/
│   │   ├── HomeScreen.tsx           # Landing with ECG animation
│   │   ├── RecordScreen.tsx         # Camera + 30s countdown
│   │   ├── ProcessingScreen.tsx     # Animated pipeline steps
│   │   ├── ResultsScreen.tsx        # Full biometric dashboard
│   │   └── VideoPlaybackScreen.tsx  # Green channel replay
```

## Connecting to Your Pipeline

The app sends video to `POST /process` and expects this JSON shape:

```json
{
  "pulse_signal": [0.1, 0.2, ...],   // 1D normalized array
  "timestamps":   [0.0, 0.033, ...], // seconds
  "peaks_idx":    [27, 56, 89, ...], // frame indices of heartbeats
  "ibi_ms":       [865, 870, ...],   // inter-beat intervals
  "fps":          30,
  "bpm":          72.1,
  "confidence":   0.81,
  "is_reliable":  true,
  "confidence_details": {
    "ibi_regularity": 0.88,
    "snr": 0.79,
    "density": 0.92,
    "duration": 1.0
  },
  "hrv_features": {
    "rmssd_ms":    34.2,
    "sdnn_ms":     42.3,
    "lf_hf_ratio": 0.87,
    "stress_index": 22,
    "stress_level": "Low"
  },
  "method_used":     "pos+deep_ensemble",
  "deep_model_used": "PhysFormer.pure",
  "triage_mode":     "BIOMETRIC",
  "triage_reason":   "Signal reliable (confidence=0.81)",
  "visual_stress":   0,
  "duration_sec":    30,
  "frames_processed": 900,
  "status":          "success"
}
```

## Design System

**Clinical Noir** — pure monochromatic palette, surgical precision.

- Background: `#080808` (near-black)  
- Surface: `#111111`
- Accent: pure `#FFFFFF` with opacity variants
- Green channel overlay: `rgba(0, 220, 80, ...)` on video playback only
- Font: Space Grotesk (tabular numerals for vitals)
- Animations: React Native Animated API (no Reanimated dependency)

## Health Tips Logic

Tips are selected based on the result:
- `stress_level === 'High'`   → breathing exercises, hydration, cold water
- `stress_level === 'Medium'` → music, outdoor walk
- `stress_level === 'Low'`    → Zone 2 cardio, sleep optimization  
- `bpm > 90`                  → stimulant avoidance, cooling
- Always: measurement consistency tip

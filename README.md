# CardioVision

Contact-free cardiac monitoring with live rPPG streaming.

## Current Project Layout

- `backend/`: Python realtime streaming backend (ROI, POS, deep_rPPG, triage)
- `mobileAppExpo/`: Expo React Native mobile app

## Realtime Streaming Pipeline

1. Mobile app captures front-camera frames.
2. Frames stream to backend over WebSocket.
3. Backend applies face ROI pipeline and returns overlay frames.
4. Backend computes POS/deep_rPPG incrementally.
5. On stop, backend returns final result payload (BPM, HRV, confidence, triage).

## Run Backend

```bash
cd backend
uv sync
uv run main.py --mode stream --ws-host 0.0.0.0 --ws-port 8765
```

Expected startup log:

```text
WebSocket server listening on ws://0.0.0.0:8765
```

## Run Mobile App

```bash
cd mobileAppExpo
bun install
EXPO_PUBLIC_BACKEND_HOST=<YOUR_LAPTOP_IP> bunx expo start --lan
```

Notes:

- Phone and laptop must be on the same Wi-Fi.
- Use LAN mode for local WebSocket connectivity.

## Main Runtime Files

Backend:

- `backend/main.py`
- `backend/stream_server.py`
- `backend/streaming_pipeline.py`
- `backend/roi_pipeline.py`
- `backend/rppg_core.py`
- `backend/deep_rppg.py`
- `backend/triage_agent.py`

Frontend:

- `mobileAppExpo/app/record.tsx`
- `mobileAppExpo/app/processing.tsx`
- `mobileAppExpo/app/results.tsx`
- `mobileAppExpo/src/api/rppgService.ts`
- `mobileAppExpo/src/state/scanSession.ts`

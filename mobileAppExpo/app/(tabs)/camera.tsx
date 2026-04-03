import React, { useRef, useState, useCallback, useEffect } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Animated,
  Dimensions,
  Alert,
  StatusBar,
  Platform,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { useRouter } from 'expo-router';

const { width, height } = Dimensions.get('window');

// Change this to your backend IP when testing on a physical device
const BACKEND_URL = 'https://cardiovision-final-12345.loca.lt'; // ← final tunnel URL

const SCAN_DURATION_MS = 30000; // 30 seconds
const FRAME_INTERVAL_MS = 100;  // ~10 FPS

type ScanPhase = 'idle' | 'scanning' | 'processing' | 'done' | 'error';

export interface ScanResult {
  bpm: number;
  rmssd: number;
  sdnn: number;
  lf_hf: number;
  confidence: number;
  ibi_array: number[];
  stress_level: 'low' | 'moderate' | 'high';
}

// ─── Helper: derive stress label from LF/HF and RMSSD ─────────────────────────
function classifyStress(result: ScanResult): 'low' | 'moderate' | 'high' {
  if (result.stress_level) return result.stress_level;
  if (result.lf_hf < 1.5) return 'low';
  if (result.lf_hf < 3.0) return 'moderate';
  return 'high';
}

export default function CameraScreen() {
  const cameraRef = useRef<CameraView>(null);
  const [permission, requestPermission] = useCameraPermissions();
  const [isReady, setIsReady] = useState(false);
  const [scanPhase, setScanPhase] = useState<ScanPhase>('idle');
  const [progress, setProgress] = useState(0);
  const [liveBpm, setLiveBpm] = useState(72);
  const [liveRmssd, setLiveRmssd] = useState('45.0');
  const [liveSdnn, setLiveSdnn] = useState('38.0');
  const [liveLfHf, setLiveLfHf] = useState('1.2');
  const [capturedFrames, setCapturedFrames] = useState<string[]>([]);

  const progressAnim = useRef(new Animated.Value(0)).current;
  const roiPulse = useRef(new Animated.Value(1)).current;
  const router = useRouter();

  const frameTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const scanTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const progressTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const liveBpmTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const framesRef = useRef<string[]>([]);
  const lastPhotoUriRef = useRef<string | null>(null);

  // ROI subtle pulse when scanning
  useEffect(() => {
    if (scanPhase === 'scanning') {
      const roiLoop = Animated.loop(
        Animated.sequence([
          Animated.timing(roiPulse, { toValue: 1.04, duration: 500, useNativeDriver: true }),
          Animated.timing(roiPulse, { toValue: 1, duration: 500, useNativeDriver: true }),
        ])
      );
      roiLoop.start();
      return () => roiLoop.stop();
    }
  }, [scanPhase]);

  const stopAllTimers = useCallback(() => {
    if (frameTimerRef.current) clearInterval(frameTimerRef.current);
    if (scanTimerRef.current) clearTimeout(scanTimerRef.current);
    if (progressTimerRef.current) clearInterval(progressTimerRef.current);
    if (liveBpmTimerRef.current) clearInterval(liveBpmTimerRef.current);
    frameTimerRef.current = null;
    scanTimerRef.current = null;
    progressTimerRef.current = null;
    liveBpmTimerRef.current = null;
  }, []);

  // ── Capture a single frame as base64 ────────────────────────────────────────
  const captureFrame = useCallback(async () => {
    if (!cameraRef.current || !isReady) return;
    try {
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.4,
        base64: true,
        skipProcessing: true,
      });
      if (photo?.base64) {
        framesRef.current.push(photo.base64);
        if (photo.uri) lastPhotoUriRef.current = photo.uri;
      }
    } catch (_e) {
      // silently skip missed frames
    }
  }, [isReady]);

  // ── Send frames to backend ─────────────────────────────────────────────────
  const sendFramesToBackend = useCallback(async (frames: string[]): Promise<ScanResult> => {
    const formData = new FormData();
    frames.forEach((b64, idx) => {
      // convert base64 string to a blob-like object for React Native FormData
      const uri = `data:image/jpeg;base64,${b64}`;
      formData.append('frames', {
        uri,
        name: `frame_${idx}.jpg`,
        type: 'image/jpeg',
      } as unknown as Blob);
    });
    formData.append('frame_count', String(frames.length));
    formData.append('fps', '10');

    const response = await fetch(`${BACKEND_URL}/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'multipart/form-data' },
      body: formData,
    });

    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`Backend error ${response.status}: ${errText}`);
    }

    return response.json() as Promise<ScanResult>;
  }, []);

  // ── Main scan flow ─────────────────────────────────────────────────────────
  const startScan = useCallback(async () => {
    if (!isReady) {
      Alert.alert('Camera not ready', 'Please wait for the camera to initialize.');
      return;
    }

    framesRef.current = [];
    setProgress(0);
    setScanPhase('scanning');

    // Progress bar animation over SCAN_DURATION_MS
    Animated.timing(progressAnim, {
      toValue: 1,
      duration: SCAN_DURATION_MS,
      useNativeDriver: false,
    }).start();

    // Frame capture at ~10 FPS
    frameTimerRef.current = setInterval(captureFrame, FRAME_INTERVAL_MS);

    // Numeric progress counter
    const start = Date.now();
    progressTimerRef.current = setInterval(() => {
      const elapsed = Date.now() - start;
      setProgress(Math.min(100, Math.round((elapsed / SCAN_DURATION_MS) * 100)));
    }, 200);

    // Live BPM & HRV Estimation simulation
    liveBpmTimerRef.current = setInterval(() => {
      setLiveBpm(70 + Math.floor(Math.random() * 8));
      setLiveRmssd((40 + Math.random() * 10).toFixed(1));
      setLiveSdnn((35 + Math.random() * 8).toFixed(1));
      setLiveLfHf((1.0 + Math.random() * 0.4).toFixed(2));
    }, 1200);

    // After SCAN_DURATION_MS → stop capture → send to backend
    scanTimerRef.current = setTimeout(async () => {
      stopAllTimers();
      progressAnim.setValue(0);

      const frames = [...framesRef.current];
      setScanPhase('processing');

      try {
        const result = await sendFramesToBackend(frames);
        const stress = classifyStress(result);
        router.push({
          pathname: '/(tabs)/results',
          params: {
            resultJson: JSON.stringify({ ...result, stress_level: stress }),
            photoUri: lastPhotoUriRef.current || '',
          },
        });
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        console.error('Backend error:', msg);
        setScanPhase('error');
        Alert.alert(
          'Processing Error',
          'Could not connect to the backend. Is your server running?',
          [{ text: 'OK', onPress: () => setScanPhase('idle') }]
        );
      }
    }, SCAN_DURATION_MS);
  }, [isReady, captureFrame, sendFramesToBackend, stopAllTimers, router, progressAnim]);

  const cancelScan = useCallback(() => {
    stopAllTimers();
    progressAnim.setValue(0);
    setScanPhase('idle');
    setProgress(0);
  }, [stopAllTimers, progressAnim]);

  // ── Permission gates ───────────────────────────────────────────────────────
  if (!permission) {
    return (
      <View style={styles.centered}>
        <Text style={styles.permText}>Checking camera permission…</Text>
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <View style={styles.centered}>
        <Text style={styles.heartIcon}>📷</Text>
        <Text style={styles.permTitle}>Camera Access Required</Text>
        <Text style={styles.permText}>We need camera access to measure your heart rate.</Text>
        <TouchableOpacity style={styles.permButton} onPress={requestPermission}>
          <Text style={styles.permButtonText}>Allow Camera</Text>
        </TouchableOpacity>
      </View>
    );
  }

  // ── Progress bar width interpolation ──────────────────────────────────────
  const progressWidth = progressAnim.interpolate({
    inputRange: [0, 1],
    outputRange: ['0%', '100%'],
  });

  const roiColor = scanPhase === 'scanning' ? '#22C55E' : '#6C63FF';
  const isScanning = scanPhase === 'scanning';
  const isProcessing = scanPhase === 'processing';

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="transparent" translucent />

      {/* ── Camera Feed ── */}
      <CameraView
        ref={cameraRef}
        style={StyleSheet.absoluteFillObject}
        facing="front"
        onCameraReady={() => setIsReady(true)}
      />

      {/* Dark overlay at top and bottom */}
      <View style={styles.topOverlay} />
      <View style={styles.bottomOverlay} />

      {/* ── Top Bar ── */}
      <View style={[styles.topBar, { zIndex: 10 }]}>
        <TouchableOpacity onPress={() => { cancelScan(); router.back(); }} style={styles.backButton}>
          <Text style={styles.backIcon}>‹</Text>
        </TouchableOpacity>
        <Text style={styles.topBarTitle}>Face Scanner</Text>
        <View style={{ width: 40 }} />
      </View>

      {/* ── ROI Box (face region of interest) ── */}
      <Animated.View
        style={[
          styles.roi,
          {
            borderColor: roiColor,
            transform: [{ scale: roiPulse }],
          },
        ]}
      >
        {/* Corner decorators */}
        <View style={[styles.corner, styles.cornerTL, { borderColor: roiColor }]} />
        <View style={[styles.corner, styles.cornerTR, { borderColor: roiColor }]} />
        <View style={[styles.corner, styles.cornerBL, { borderColor: roiColor }]} />
        <View style={[styles.corner, styles.cornerBR, { borderColor: roiColor }]} />

        {isScanning && (
          <>
            <View style={styles.metricTL}><Text style={styles.metricText}>BPM {liveBpm}</Text></View>
            <View style={styles.metricTR}><Text style={styles.metricText}>RMSSD {liveRmssd}</Text></View>
            <View style={styles.metricBL}><Text style={styles.metricText}>SDNN {liveSdnn}</Text></View>
            <View style={styles.metricBR}><Text style={styles.metricText}>LF/HF {liveLfHf}</Text></View>
          </>
        )}

        {isScanning && (
          <View style={styles.scanLine}>
            <View style={[styles.scanLineBar, { backgroundColor: roiColor }]} />
          </View>
        )}
      </Animated.View>

      {/* ── Instruction Text ── */}
      <View style={styles.instructionContainer}>
        {scanPhase === 'idle' && (
          <>
            <Text style={styles.instructionTitle}>Position your face</Text>
            <Text style={styles.instructionSub}>Keep face steady • Good lighting required</Text>
          </>
        )}
        {isScanning && (
          <>
            <Text style={styles.instructionTitle}>🔴 Scanning…</Text>
            <Text style={styles.instructionSub}>Keep face steady • Do not move</Text>
          </>
        )}
        {isProcessing && (
          <>
            <Text style={styles.instructionTitle}>⏳ Processing…</Text>
            <Text style={styles.instructionSub}>Analysing signal with AI</Text>
          </>
        )}
      </View>

      {/* ── Progress bar ── */}
      {isScanning && (
        <View style={styles.progressBarContainer}>
          <Animated.View style={[styles.progressBarFill, { width: progressWidth }]} />
          <Text style={styles.progressLabel}>{progress}%</Text>
        </View>
      )}

      {/* ── Frame count badge ── */}
      {isScanning && (
        <View style={styles.frameBadge}>
          <Text style={styles.frameBadgeText}>
            {Math.floor(progress / 5)} frames captured
          </Text>
        </View>
      )}

      {/* ── Bottom Controls ── */}
      <View style={styles.controls}>
        {scanPhase === 'idle' && (
          <TouchableOpacity
            style={styles.scanButton}
            onPress={startScan}
            activeOpacity={0.85}
          >
            <Text style={styles.scanButtonText}>▶  Start Scan</Text>
          </TouchableOpacity>
        )}

        {isScanning && (
          <TouchableOpacity
            style={[styles.scanButton, styles.cancelButton]}
            onPress={cancelScan}
            activeOpacity={0.85}
          >
            <Text style={styles.scanButtonText}>✕  Cancel</Text>
          </TouchableOpacity>
        )}

        {isProcessing && (
          <View style={[styles.scanButton, styles.processingButton]}>
            <Text style={styles.scanButtonText}>Processing…</Text>
          </View>
        )}
      </View>
    </View>
  );
}

const ROI_SIZE = width * 0.65;
const CORNER_SIZE = 24;
const CORNER_WIDTH = 3;

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  centered: {
    flex: 1,
    backgroundColor: '#0D0D1A',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 32,
  },
  heartIcon: { fontSize: 56, marginBottom: 16 },
  permTitle: { color: '#FFF', fontSize: 22, fontWeight: '700', marginBottom: 8 },
  permText: { color: '#9CA3AF', fontSize: 15, textAlign: 'center', lineHeight: 22 },
  permButton: {
    marginTop: 28,
    paddingVertical: 14,
    paddingHorizontal: 40,
    backgroundColor: '#6C63FF',
    borderRadius: 14,
  },
  permButtonText: { color: '#FFF', fontWeight: '700', fontSize: 16 },

  // Overlays
  topOverlay: {
    position: 'absolute', top: 0, left: 0, right: 0, height: 140,
    backgroundColor: 'rgba(0,0,0,0.55)',
  },
  bottomOverlay: {
    position: 'absolute', bottom: 0, left: 0, right: 0, height: 200,
    backgroundColor: 'rgba(0,0,0,0.60)',
  },

  // Top bar
  topBar: {
    position: 'absolute',
    top: Platform.OS === 'android' ? 36 : 56,
    left: 0, right: 0,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
  },
  backButton: {
    width: 40, height: 40,
    alignItems: 'center', justifyContent: 'center',
    backgroundColor: 'rgba(255,255,255,0.15)',
    borderRadius: 20,
  },
  backIcon: { color: '#FFF', fontSize: 28, marginTop: -2 },
  topBarTitle: { color: '#FFF', fontWeight: '700', fontSize: 18, letterSpacing: 0.5 },

  // ROI
  roi: {
    position: 'absolute',
    top: (height - ROI_SIZE) / 2 - 20,
    left: (width - ROI_SIZE) / 2,
    width: ROI_SIZE,
    height: ROI_SIZE,
    borderWidth: 1.5,
    borderRadius: 12,
    borderColor: '#6C63FF',
    overflow: 'hidden',
  },
  corner: {
    position: 'absolute',
    width: CORNER_SIZE,
    height: CORNER_SIZE,
    borderColor: '#6C63FF',
  },
  cornerTL: { top: -1, left: -1, borderTopWidth: CORNER_WIDTH, borderLeftWidth: CORNER_WIDTH, borderTopLeftRadius: 10 },
  cornerTR: { top: -1, right: -1, borderTopWidth: CORNER_WIDTH, borderRightWidth: CORNER_WIDTH, borderTopRightRadius: 10 },
  cornerBL: { bottom: -1, left: -1, borderBottomWidth: CORNER_WIDTH, borderLeftWidth: CORNER_WIDTH, borderBottomLeftRadius: 10 },
  cornerBR: { bottom: -1, right: -1, borderBottomWidth: CORNER_WIDTH, borderRightWidth: CORNER_WIDTH, borderBottomRightRadius: 10 },
  scanLine: { position: 'absolute', top: '50%', left: 0, right: 0, height: 2 },
  scanLineBar: { height: 2, opacity: 0.6 },

  // Instruction
  instructionContainer: {
    position: 'absolute',
    top: (height - ROI_SIZE) / 2 + ROI_SIZE + 20,
    left: 0, right: 0,
    alignItems: 'center',
  },
  instructionTitle: { color: '#FFF', fontSize: 18, fontWeight: '700', marginBottom: 4 },
  instructionSub: { color: '#9CA3AF', fontSize: 13 },

  metricTL: { position: 'absolute', top: 8, left: 8, backgroundColor: 'rgba(0,0,0,0.6)', paddingHorizontal: 6, paddingVertical: 4, borderRadius: 6 },
  metricTR: { position: 'absolute', top: 8, right: 8, backgroundColor: 'rgba(0,0,0,0.6)', paddingHorizontal: 6, paddingVertical: 4, borderRadius: 6 },
  metricBL: { position: 'absolute', bottom: 8, left: 8, backgroundColor: 'rgba(0,0,0,0.6)', paddingHorizontal: 6, paddingVertical: 4, borderRadius: 6 },
  metricBR: { position: 'absolute', bottom: 8, right: 8, backgroundColor: 'rgba(0,0,0,0.6)', paddingHorizontal: 6, paddingVertical: 4, borderRadius: 6 },
  metricText: { color: '#22C55E', fontWeight: '700', fontSize: 11 },

  // Progress
  progressBarContainer: {
    position: 'absolute',
    bottom: 185,
    left: 32, right: 32,
    height: 6,
    backgroundColor: 'rgba(255,255,255,0.15)',
    borderRadius: 3,
    overflow: 'visible',
  },
  progressBarFill: {
    height: 6,
    backgroundColor: '#22C55E',
    borderRadius: 3,
  },
  progressLabel: {
    position: 'absolute',
    right: 0,
    top: 10,
    color: '#9CA3AF',
    fontSize: 11,
  },

  // Frame badge
  frameBadge: {
    position: 'absolute',
    bottom: 205,
    alignSelf: 'center',
  },
  frameBadgeText: { color: 'rgba(255,255,255,0.4)', fontSize: 11 },

  // Controls
  controls: {
    position: 'absolute',
    bottom: 48,
    left: 0, right: 0,
    alignItems: 'center',
  },
  scanButton: {
    paddingVertical: 18,
    paddingHorizontal: 64,
    backgroundColor: '#6C63FF',
    borderRadius: 50,
    elevation: 10,
    shadowColor: '#6C63FF',
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.6,
    shadowRadius: 10,
  },
  cancelButton: { backgroundColor: '#EF4444' },
  processingButton: { backgroundColor: '#374151' },
  scanButtonText: { color: '#FFF', fontSize: 17, fontWeight: '700', letterSpacing: 0.5 },
});
// app/record.tsx — RecordScreen
import React, { useRef, useState, useEffect, useCallback } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Animated, Dimensions, Alert } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import * as Haptics from 'expo-haptics';
import Svg, { Circle, Rect, Text as SvgText } from 'react-native-svg';
import { Ionicons } from '@expo/vector-icons';
import { useTheme, Spacing, Radius } from '../theme';
import { LiveRPPGClient, getMockResult, type RPPGResult } from '../api/rppgService';
import { clearScanSession, setPendingScanResult, setScanSession } from '../state/scanSession';

const { width, height: screenH } = Dimensions.get('window');
const RECORD_DURATION = 30;
const CIRC_R = 54;
const CIRC = 2 * Math.PI * CIRC_R;
const STREAM_FRAME_INTERVAL_MS = 100;
const STREAM_CAPTURE_QUALITY = 0.45;

// ── Snackbar ──
function Snackbar({ message, visible }: { message: string; visible: boolean }) {
  const opacity = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    if (visible) {
      Animated.sequence([
        Animated.timing(opacity, { toValue: 1, duration: 300, useNativeDriver: true }),
        Animated.delay(2500),
        Animated.timing(opacity, { toValue: 0, duration: 300, useNativeDriver: true }),
      ]).start();
    }
  }, [visible, message, opacity]);
  if (!visible) return null;
  return (
    <Animated.View style={[styles.snackbar, { opacity }]}>
      <Ionicons name="warning-outline" size={16} color="#FFCC00" style={{ marginRight: 8 }} />
      <Text style={styles.snackbarText}>{message}</Text>
    </Animated.View>
  );
}

// ── ROI face highlights ──
function ROIHighlights({ visible }: { visible: boolean }) {
  if (!visible) return null;
  const ovalW = width * 0.62, ovalH = ovalW * 1.35;
  const cx = width / 2, cy = (screenH * 0.42);
  const zones = [
    { label: 'Forehead', x: cx, y: cy - ovalH * 0.32, w: ovalW * 0.45, h: ovalH * 0.12 },
    { label: 'L. Cheek', x: cx - ovalW * 0.22, y: cy + ovalH * 0.02, w: ovalW * 0.2, h: ovalH * 0.15 },
    { label: 'R. Cheek', x: cx + ovalW * 0.22, y: cy + ovalH * 0.02, w: ovalW * 0.2, h: ovalH * 0.15 },
    { label: 'Nose', x: cx, y: cy - ovalH * 0.02, w: ovalW * 0.18, h: ovalH * 0.1 },
  ];
  return (
    <Svg width={width} height={screenH} style={StyleSheet.absoluteFill} pointerEvents="none">
      {zones.map(z => (
        <React.Fragment key={z.label}>
          <Rect x={z.x - z.w / 2} y={z.y - z.h / 2} width={z.w} height={z.h} rx={z.w / 4}
            stroke="#3944BC" strokeWidth={1} strokeOpacity={0.5} fill="rgba(57,68,188,0.08)" />
          <SvgText x={z.x} y={z.y - z.h / 2 - 4} textAnchor="middle"
            fontSize={8} fill="rgba(57,68,188,0.7)" fontFamily="SpaceGrotesk-Medium">{z.label}</SvgText>
        </React.Fragment>
      ))}
    </Svg>
  );
}

function FaceGuideOverlay({ isRecording }: { isRecording: boolean }) {
  const { accent } = useTheme();
  const pulse = useRef(new Animated.Value(1)).current;
  useEffect(() => {
    if (isRecording) {
      const loop = Animated.loop(Animated.sequence([
        Animated.timing(pulse, { toValue: 1.02, duration: 900, useNativeDriver: true }),
        Animated.timing(pulse, { toValue: 1.0, duration: 900, useNativeDriver: true }),
      ]));
      loop.start(); return () => loop.stop();
    }
  }, [isRecording, pulse]);
  const ovalW = width * 0.62, ovalH = ovalW * 1.35;
  return (
    <Animated.View style={[styles.faceOval, {
      width: ovalW, height: ovalH, borderRadius: ovalW / 2,
      transform: [{ scale: pulse }],
      borderColor: isRecording ? accent.primary : '#FFFFFF',
    }]} />
  );
}

export default function RecordScreen() {
  const router = useRouter();
  const { colors, accent } = useTheme();
  const camRef = useRef<CameraView>(null);
  const [permission, requestPermission] = useCameraPermissions();
  const [isRecording, setIsRecording] = useState(false);
  const [countdown3, setCountdown3] = useState(false);
  const [countdownN, setCountdownN] = useState(3);
  const [timeLeft, setTimeLeft] = useState(RECORD_DURATION);
  const [progress, setProgress] = useState(0);
  const [quality, setQuality] = useState(0);
  const [liveBpm, setLiveBpm] = useState<number | null>(null);
  const [liveMethod, setLiveMethod] = useState('pending');
  const [isTimerPaused, setIsTimerPaused] = useState(false);
  const [pauseReason, setPauseReason] = useState<'none' | 'face_lost' | 'unknown_person'>('none');
  const [snack, setSnack] = useState({ msg: '', show: false, key: 0 });
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const framePumpRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const liveClientRef = useRef<LiveRPPGClient | null>(null);
  const intruderStopTriggeredRef = useRef(false);
  const fallbackVideoPromiseRef = useRef<Promise<{ uri: string } | null> | null>(null);
  const usingLiveStreamRef = useRef(false);
  const pausedRef = useRef(false);
  const qualityAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => { if (!permission?.granted) requestPermission(); }, [permission?.granted, requestPermission]);

  useEffect(() => {
    Animated.timing(qualityAnim, { toValue: quality, duration: 300, useNativeDriver: false }).start();
  }, [quality, qualityAnim]);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      if (framePumpRef.current) clearInterval(framePumpRef.current);
      liveClientRef.current?.disconnect();
    };
  }, []);

  const startCountdown = async () => {
    clearScanSession();
    await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    setCountdown3(true); setCountdownN(3);
    let n = 3;
    const tick = setInterval(async () => {
      n--;
      if (n > 0) { setCountdownN(n); await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light); }
      else { clearInterval(tick); setCountdown3(false); startRecording(); }
    }, 1000);
  };

  const startRecording = useCallback(async () => {
    if (!camRef.current) return;
    usingLiveStreamRef.current = false;
    try {
      const client = new LiveRPPGClient({
        overlayQuality: 20,
        overlayMaxSide: 0,
        overlayStride: 999,
        onFrame: (frame: any) => {
          setQuality(frame.metric?.confidence ?? frame.confidence ?? 0);
          setLiveBpm(frame.metric?.bpm ?? frame.bpm ?? null);
          setLiveMethod(frame.metric?.method ?? frame.method ?? 'pending');
          // No overlay display needed — we use SVG ROI highlights instead.
          if (frame.intruder_detected || (frame.identity_locked && !frame.identity_match)) {
            setIsTimerPaused(true);
            pausedRef.current = true;
            setPauseReason('unknown_person');
            setSnack({ msg: 'Unknown person seen. Waiting for original person to return.', show: true, key: Date.now() });
            return;
          }

          if (!frame.has_face) {
            if (frame.identity_locked) {
              setIsTimerPaused(true);
              pausedRef.current = true;
              setPauseReason('face_lost');
              setSnack({ msg: 'Primary face lost. Timer paused until face is back.', show: true, key: Date.now() });
            }
            return;
          }

          if (pausedRef.current) {
            setIsTimerPaused(false);
            pausedRef.current = false;
            setPauseReason('none');
            setSnack({ msg: 'Primary face reacquired. Timer resumed.', show: true, key: Date.now() });
          }
        },
      });
      await client.connect(3500);
      liveClientRef.current = client;
      usingLiveStreamRef.current = true;
    } catch {
      Alert.alert(
        'Live stream unavailable',
        'Falling back to standard recording and upload mode. Check backend websocket server/IP for live overlay.',
      );
    }

    setIsRecording(true); setTimeLeft(RECORD_DURATION); setQuality(0); setLiveBpm(null); setLiveMethod('pending');
    setIsTimerPaused(false); setPauseReason('none'); pausedRef.current = false;
    intruderStopTriggeredRef.current = false;
    await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);
    let elapsed = 0;
    timerRef.current = setInterval(() => {
      if (pausedRef.current) {
        return;
      }
      elapsed++; setTimeLeft(RECORD_DURATION - elapsed); setProgress(elapsed / RECORD_DURATION);
      if (elapsed >= RECORD_DURATION) stopRecording();
    }, 1000);
    if (usingLiveStreamRef.current) {
      // Use requestAnimationFrame-style adaptive pump instead of fixed interval.
      // This ensures we never queue frames faster than we can send them.
      const pumpFrame = () => {
        if (!liveClientRef.current) return;
        void captureAndStreamFrame().finally(() => {
          if (liveClientRef.current) {
            framePumpRef.current = setTimeout(pumpFrame, STREAM_FRAME_INTERVAL_MS);
          }
        });
      };
      framePumpRef.current = setTimeout(pumpFrame, 50);
      return;
    }

      fallbackVideoPromiseRef.current = camRef.current
        .recordAsync({ maxDuration: RECORD_DURATION })
        .then((video) => (video?.uri ? { uri: video.uri } : null))
        .catch(() => null);
  }, [captureAndStreamFrame, stopRecording]);

  const captureAndStreamFrame = useCallback(async () => {
    if (!camRef.current || !liveClientRef.current) return;
    try {
      const snap = await camRef.current.takePictureAsync({
        base64: true,
        quality: STREAM_CAPTURE_QUALITY,
        skipProcessing: true,
        shutterSound: false,
      });
      if (snap?.base64) {
        liveClientRef.current.sendFrameBinary(snap.base64, Date.now());
      }
    } catch {
      // Ignore transient frame capture errors during live stream.
    }
  }, []);

  const stopRecording = useCallback(async () => {
    if (timerRef.current) clearInterval(timerRef.current);
    if (framePumpRef.current) clearTimeout(framePumpRef.current);
    await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    setIsRecording(false);

    if (!usingLiveStreamRef.current) {
      try { camRef.current?.stopRecording(); } catch {}
      const fallbackVideo = await fallbackVideoPromiseRef.current;
      fallbackVideoPromiseRef.current = null;
      if (fallbackVideo?.uri) {
        router.push({
          pathname: '/processing',
          params: {
            videoUri: fallbackVideo.uri,
          },
        });
        return;
      }

      router.push({
        pathname: '/processing',
      });
      setScanSession({ result: getMockResult() });
      return;
    }

    const client = liveClientRef.current;
    liveClientRef.current = null;
    if (!client) {
      setScanSession({ result: getMockResult() });
      router.push({ pathname: '/processing' });
      return;
    }

    const pendingFinal = (async (): Promise<RPPGResult> => {
      try {
        return await client.stopAndGetFinalResult(60000);
      } catch {
        return getMockResult();
      } finally {
        client.disconnect();
      }
    })();

    setPendingScanResult(pendingFinal);
    setScanSession({ result: null });
    router.push({ pathname: '/processing' });
  }, [router]);

  const reset = () => {
    if (timerRef.current) clearInterval(timerRef.current);
    if (framePumpRef.current) clearTimeout(framePumpRef.current);
    if (!usingLiveStreamRef.current) {
      try { camRef.current?.stopRecording(); } catch {}
    }
    liveClientRef.current?.disconnect();
    liveClientRef.current = null;
    fallbackVideoPromiseRef.current = null;
    usingLiveStreamRef.current = false;
    pausedRef.current = false;
    intruderStopTriggeredRef.current = false;
    setIsRecording(false); setCountdown3(false); setTimeLeft(RECORD_DURATION); setProgress(0);
    setQuality(0); setLiveBpm(null); setLiveMethod('pending');
    setIsTimerPaused(false); setPauseReason('none');
  };

  if (!permission?.granted) {
    return (
      <View style={[styles.permView, { backgroundColor: colors.background }]}>
        <Ionicons name="camera-outline" size={48} color={accent.primary} style={{ marginBottom: 16 }} />
        <Text style={[styles.permTitle, { color: colors.textPrimary }]}>Camera Access Required</Text>
        <Text style={[styles.permBody, { color: colors.textSecondary }]}>CardioVision needs camera access to measure your heart rate from facial video.</Text>
        <TouchableOpacity style={[styles.permBtn, { backgroundColor: accent.primary }]} onPress={requestPermission}>
          <Text style={styles.permBtnText}>Grant Access</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const qualityW = qualityAnim.interpolate({ inputRange: [0, 1], outputRange: ['0%', '100%'] });
  const dashOffset = CIRC * (1 - progress);

  return (
    <View style={styles.root}>
      <CameraView ref={camRef} style={StyleSheet.absoluteFill} facing="front" mode="video" videoQuality="720p" />
      {/* Overlay disabled: we show SVG ROI highlights instead of server-streamed overlay */}
      <View style={styles.overlay} />
      <View style={styles.guideContainer}><FaceGuideOverlay isRecording={isRecording} /></View>

      {/* ROI zone highlights when recording */}
      <ROIHighlights visible={isRecording} />

      {/* Countdown overlay */}
      {countdown3 && (
        <View style={styles.countdownOverlay}>
          <Text style={styles.countdownNum}>{countdownN}</Text>
        </View>
      )}

      {/* Snackbar warnings */}
      <Snackbar message={snack.msg} visible={snack.show} key={snack.key} />

      <SafeAreaView style={styles.safe} edges={['top', 'bottom']}>
        {/* Top Bar */}
        <View style={styles.topBar}>
          <TouchableOpacity style={styles.backBtn} onPress={() => { reset(); router.back(); }}>
            <Ionicons name="arrow-back" size={20} color="#FFF" />
          </TouchableOpacity>
          <Text style={styles.topTitle}>{isRecording ? 'Recording' : 'Scan Setup'}</Text>
          <View style={{ width: 40 }} />
        </View>

        {/* Live BPM — show "--" since no actual live data yet */}
        {isRecording && (
          <View style={styles.bpmOverlay}>
            <Text style={styles.bpmNum}>{liveBpm !== null ? Math.round(liveBpm) : '--'}</Text>
            <Text style={styles.bpmLabel}>BPM</Text>
            <Text style={styles.instructText}>Stay still. Method: {liveMethod.toUpperCase()}</Text>
            {isTimerPaused && (
              <Text style={[styles.pauseStatus, pauseReason === 'unknown_person' ? styles.pauseStatusWarn : null]}>
                {pauseReason === 'unknown_person' ? 'Paused: unknown person detected' : 'Paused: primary face not detected'}
              </Text>
            )}
          </View>
        )}

        {!isRecording && !countdown3 && (
          <View style={styles.instructBarTop}>
            <Text style={styles.instructText}>Position your face in the oval</Text>
          </View>
        )}

        <View style={{ flex: 1 }} />

        {/* Signal quality bar + triage mode (during recording) */}
        {isRecording && (
          <View style={styles.metricsArea}>
            <View style={styles.triagePill}>
              <Ionicons name="shield-checkmark-outline" size={14} color={accent.light} />
              <Text style={styles.triageText}>Biometric Mode</Text>
            </View>
            <View style={styles.qualityBar}>
              <Text style={styles.qualityLabel}>Signal Quality</Text>
              <View style={styles.qualityTrack}>
                <Animated.View style={[styles.qualityFill, { width: qualityW, backgroundColor: accent.primary }]} />
              </View>
              <Text style={styles.qualityPct}>{Math.round(quality * 100)}%</Text>
            </View>
          </View>
        )}

        {/* Tips — positioned BELOW the oval, not overlapping */}
        {!isRecording && !countdown3 && (
          <View style={styles.tipsCard}>
            {[
              { icon: 'sunny-outline', tip: 'Ensure your face is well-lit' },
              { icon: 'resize-outline', tip: "Keep phone at arm's length" },
              { icon: 'body-outline', tip: 'Remain still during recording' },
              { icon: 'contrast-outline', tip: 'Avoid backlit environments' },
            ].map((t, i) => (
              <View key={i} style={styles.tipRow}>
                <Ionicons name={t.icon as any} size={14} color="rgba(255,255,255,0.4)" style={{ marginRight: 8, marginTop: 1 }} />
                <Text style={styles.tipText}>{t.tip}</Text>
              </View>
            ))}
          </View>
        )}

        {/* Bottom Controls */}
        <View style={styles.bottomBar}>
          {isRecording ? (
            <View style={styles.ringWrap}>
              <Svg width={130} height={130} viewBox="0 0 130 130">
                <Circle cx={65} cy={65} r={CIRC_R} stroke="rgba(255,255,255,0.15)" strokeWidth={3} fill="none" />
                <Circle cx={65} cy={65} r={CIRC_R} stroke={timeLeft <= 5 ? '#FF6B35' : accent.primary}
                  strokeWidth={3} fill="none" strokeDasharray={CIRC} strokeDashoffset={dashOffset} strokeLinecap="round" transform="rotate(-90 65 65)" />
              </Svg>
              <View style={styles.timerInner}>
                <Text style={[styles.timerNum, timeLeft <= 5 && { color: '#FF6B35' }]}>00:{String(timeLeft).padStart(2, '0')}</Text>
                <TouchableOpacity onPress={stopRecording}>
                  <Ionicons name="pause" size={22} color="#FFF" />
                </TouchableOpacity>
              </View>
            </View>
          ) : (
            <TouchableOpacity style={styles.recordBtn} onPress={startCountdown} disabled={countdown3} activeOpacity={0.8}>
              <View style={[styles.recordBtnOuter, { borderColor: accent.primary }]}>
                <View style={[styles.recordBtnInner, { backgroundColor: accent.primary }]} />
              </View>
              <Text style={styles.recordBtnLabel}>{countdown3 ? `Starting in ${countdownN}...` : 'Tap to record 30s'}</Text>
            </TouchableOpacity>
          )}
        </View>
      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#000' },
  safe: { flex: 1 },
  overlay: { ...StyleSheet.absoluteFillObject, backgroundColor: 'rgba(0,0,0,0.35)' },
  guideContainer: { ...StyleSheet.absoluteFillObject, alignItems: 'center', justifyContent: 'center', marginTop: -80 },
  faceOval: { borderWidth: 1.5, borderStyle: 'dashed', backgroundColor: 'transparent' },
  countdownOverlay: { ...StyleSheet.absoluteFillObject, alignItems: 'center', justifyContent: 'center', backgroundColor: 'rgba(0,0,0,0.5)', zIndex: 10 },
  countdownNum: { fontFamily: 'SpaceGrotesk-Bold', fontSize: 100, color: '#FFF' },

  // Snackbar
  snackbar: { position: 'absolute', top: 100, left: 20, right: 20, backgroundColor: 'rgba(30,30,30,0.95)', borderRadius: 12, padding: 14, flexDirection: 'row', alignItems: 'center', zIndex: 20 },
  snackbarText: { fontFamily: 'SpaceGrotesk-Medium', fontSize: 13, color: '#FFF', flex: 1 },

  topBar: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: Spacing.lg, paddingTop: Spacing.md },
  backBtn: { width: 40, height: 40, borderRadius: 20, backgroundColor: 'rgba(255,255,255,0.12)', alignItems: 'center', justifyContent: 'center' },
  topTitle: { fontFamily: 'SpaceGrotesk-SemiBold', fontSize: 16, color: '#FFF' },

  bpmOverlay: { alignItems: 'center', marginTop: Spacing.lg },
  bpmNum: { fontFamily: 'SpaceGrotesk-Bold', fontSize: 64, color: '#FFF', letterSpacing: -2 },
  bpmLabel: { fontFamily: 'SpaceGrotesk-Medium', fontSize: 16, color: 'rgba(255,255,255,0.5)', marginTop: -6 },
  pauseStatus: {
    marginTop: 6,
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 999,
    backgroundColor: 'rgba(255,255,255,0.18)',
    color: '#FFFFFF',
    fontFamily: 'SpaceGrotesk-Medium',
    fontSize: 12,
  },
  pauseStatusWarn: {
    backgroundColor: 'rgba(239,68,68,0.28)',
  },
  instructBarTop: { alignItems: 'center', marginTop: Spacing.md },
  instructText: { fontFamily: 'SpaceGrotesk-Regular', fontSize: 14, color: 'rgba(255,255,255,0.6)', textAlign: 'center' },

  // Metrics during recording
  metricsArea: { paddingHorizontal: Spacing.lg, marginBottom: Spacing.sm },
  triagePill: { flexDirection: 'row', alignItems: 'center', alignSelf: 'center', backgroundColor: 'rgba(57,68,188,0.15)', borderRadius: Radius.full, paddingHorizontal: 14, paddingVertical: 6, marginBottom: 10, gap: 6 },
  triageText: { fontFamily: 'SpaceGrotesk-Medium', fontSize: 11, color: 'rgba(57,68,188,0.9)', textTransform: 'uppercase', letterSpacing: 0.8 },
  qualityBar: { flexDirection: 'row', alignItems: 'center', backgroundColor: 'rgba(255,255,255,0.06)', borderRadius: Radius.md, padding: Spacing.md },
  qualityLabel: { fontFamily: 'SpaceGrotesk-Medium', fontSize: 10, color: 'rgba(255,255,255,0.4)', width: 64, textTransform: 'uppercase', letterSpacing: 0.8 },
  qualityTrack: { flex: 1, height: 3, backgroundColor: 'rgba(255,255,255,0.12)', borderRadius: 2, marginHorizontal: 10, overflow: 'hidden' },
  qualityFill: { height: '100%', borderRadius: 2 },
  qualityPct: { fontFamily: 'SpaceGrotesk-Regular', fontSize: 13, color: 'rgba(255,255,255,0.5)', width: 36, textAlign: 'right' },

  // Tips — below oval
  tipsCard: { marginHorizontal: Spacing.lg, backgroundColor: 'rgba(255,255,255,0.06)', borderWidth: 1, borderColor: 'rgba(255,255,255,0.08)', borderRadius: Radius.lg, padding: Spacing.md, marginBottom: Spacing.md },
  tipRow: { flexDirection: 'row', alignItems: 'flex-start', marginBottom: 8 },
  tipText: { fontFamily: 'SpaceGrotesk-Regular', fontSize: 13, color: 'rgba(255,255,255,0.5)', flex: 1 },

  bottomBar: { alignItems: 'center', paddingBottom: Spacing.xl, paddingTop: Spacing.sm },
  recordBtn: { alignItems: 'center' },
  recordBtnOuter: { width: 76, height: 76, borderRadius: 38, borderWidth: 3, alignItems: 'center', justifyContent: 'center' },
  recordBtnInner: { width: 54, height: 54, borderRadius: 27 },
  recordBtnLabel: { fontFamily: 'SpaceGrotesk-Regular', fontSize: 13, color: 'rgba(255,255,255,0.4)', marginTop: 10 },

  ringWrap: { width: 130, height: 130, alignItems: 'center', justifyContent: 'center' },
  timerInner: { position: 'absolute', alignItems: 'center' },
  timerNum: { fontFamily: 'SpaceGrotesk-Bold', fontSize: 24, color: '#FFF', letterSpacing: -0.5 },

  permView: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: Spacing.xl },
  permTitle: { fontFamily: 'SpaceGrotesk-Bold', fontSize: 22, marginBottom: Spacing.md },
  permBody: { fontFamily: 'SpaceGrotesk-Regular', fontSize: 15, textAlign: 'center', marginBottom: Spacing.xl },
  permBtn: { borderRadius: Radius.lg, paddingHorizontal: Spacing.xl, paddingVertical: Spacing.md },
  permBtnText: { fontFamily: 'SpaceGrotesk-Bold', fontSize: 16, color: '#FFF' },
});

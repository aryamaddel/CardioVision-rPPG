// app/record.tsx — RecordScreen
import React, { useRef, useState, useEffect, useCallback } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Animated, Dimensions, Alert } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import * as Haptics from 'expo-haptics';
import Svg, { Circle, Ellipse, Rect, Text as SvgText } from 'react-native-svg';
import { Ionicons } from '@expo/vector-icons';
import { useTheme, Typography, Spacing, Radius } from '../src/theme';

const { width, height: screenH } = Dimensions.get('window');
const RECORD_DURATION = 30;
const CIRC_R = 54;
const CIRC = 2 * Math.PI * CIRC_R;

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
  }, [visible, message]);
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
  }, [isRecording]);
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
  const [snack, setSnack] = useState({ msg: '', show: false, key: 0 });
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const qualityAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => { if (!permission?.granted) requestPermission(); }, []);

  // Simulate signal quality + snackbar warnings
  useEffect(() => {
    if (isRecording) {
      const q = setInterval(() => {
        const v = 0.55 + Math.random() * 0.4;
        setQuality(v);
        Animated.timing(qualityAnim, { toValue: v, duration: 400, useNativeDriver: false }).start();
        // Random annotation warnings
        const rand = Math.random();
        if (rand < 0.05) setSnack({ msg: 'Camera is too shaky — hold steady', show: true, key: Date.now() });
        else if (rand < 0.08) setSnack({ msg: 'Low lighting detected — move to brighter area', show: true, key: Date.now() });
        else if (rand < 0.10) setSnack({ msg: 'Face not fully in oval boundary', show: true, key: Date.now() });
      }, 800);
      return () => clearInterval(q);
    }
  }, [isRecording]);

  const startCountdown = async () => {
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
    setIsRecording(true); setTimeLeft(RECORD_DURATION);
    await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);
    let elapsed = 0;
    timerRef.current = setInterval(() => {
      elapsed++; setTimeLeft(RECORD_DURATION - elapsed); setProgress(elapsed / RECORD_DURATION);
      if (elapsed >= RECORD_DURATION) stopRecording();
    }, 1000);
    try {
      const video = await camRef.current.recordAsync({ maxDuration: RECORD_DURATION });
      if (video?.uri) router.push({ pathname: '/processing', params: { videoUri: video.uri } });
    } catch (e) { Alert.alert('Recording failed', 'Please try again.'); reset(); }
  }, [router]);

  const stopRecording = useCallback(async () => {
    if (timerRef.current) clearInterval(timerRef.current);
    await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    camRef.current?.stopRecording(); setIsRecording(false);
  }, []);

  const reset = () => {
    if (timerRef.current) clearInterval(timerRef.current);
    setIsRecording(false); setCountdown3(false); setTimeLeft(RECORD_DURATION); setProgress(0);
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
            <Text style={styles.bpmNum}>--</Text>
            <Text style={styles.bpmLabel}>BPM</Text>
            <Text style={styles.instructText}>Stay still. Look at the camera.</Text>
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

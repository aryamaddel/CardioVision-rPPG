// src/screens/RecordScreen.tsx
import React, { useRef, useState, useEffect, useCallback } from 'react';
import {
  View, Text, TouchableOpacity, StyleSheet,
  Animated, Dimensions, Alert,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useNavigation } from '@react-navigation/native';
import * as Haptics from 'expo-haptics';
import Svg, { Circle, Path } from 'react-native-svg';
import { Colors, Typography, Spacing, Radius } from '../theme';

const { width, height } = Dimensions.get('window');
const RECORD_DURATION = 30; // seconds
const COUNTDOWN_RADIUS = 54;
const CIRC = 2 * Math.PI * COUNTDOWN_RADIUS;

function CountdownRing({ progress, timeLeft }: { progress: number; timeLeft: number }) {
  const dashOffset = CIRC * (1 - progress);
  const urgent = timeLeft <= 5;
  return (
    <View style={styles.ringWrap}>
      <Svg width={130} height={130} viewBox="0 0 130 130">
        {/* Track */}
        <Circle cx={65} cy={65} r={COUNTDOWN_RADIUS} stroke={Colors.border}
          strokeWidth={3} fill="none" />
        {/* Progress */}
        <Circle cx={65} cy={65} r={COUNTDOWN_RADIUS}
          stroke={urgent ? Colors.fog : Colors.white}
          strokeWidth={3} fill="none"
          strokeDasharray={CIRC}
          strokeDashoffset={dashOffset}
          strokeLinecap="round"
          transform="rotate(-90 65 65)"
        />
      </Svg>
      <View style={styles.timerInner}>
        <Text style={[styles.timerNum, urgent && { color: Colors.fog }]}>
          {String(timeLeft).padStart(2, '0')}
        </Text>
        <Text style={styles.timerSec}>sec</Text>
      </View>
    </View>
  );
}

function FaceGuideOverlay({ isRecording }: { isRecording: boolean }) {
  const pulse = useRef(new Animated.Value(1)).current;
  const alpha  = useRef(new Animated.Value(0.4)).current;

  useEffect(() => {
    if (isRecording) {
      const loop = Animated.loop(
        Animated.sequence([
          Animated.parallel([
            Animated.timing(pulse, { toValue: 1.02, duration: 900, useNativeDriver: true }),
            Animated.timing(alpha, { toValue: 0.9, duration: 900, useNativeDriver: true }),
          ]),
          Animated.parallel([
            Animated.timing(pulse, { toValue: 1.0,  duration: 900, useNativeDriver: true }),
            Animated.timing(alpha, { toValue: 0.4, duration: 900, useNativeDriver: true }),
          ]),
        ]),
      );
      loop.start();
      return () => loop.stop();
    }
  }, [isRecording]);

  const ovalW = width * 0.62;
  const ovalH = ovalW * 1.35;

  return (
    <Animated.View
      style={[
        styles.faceOval,
        {
          width: ovalW, height: ovalH,
          borderRadius: ovalW / 2,
          transform: [{ scale: pulse }],
          borderColor: Colors.white,
          opacity: alpha,
        },
      ]}
    />
  );
}

function InstructionBar({ recording, countdown }: { recording: boolean; countdown: boolean }) {
  const fadeAnim = useRef(new Animated.Value(1)).current;
  const prevMsg  = useRef('');

  const msg = !recording && !countdown
    ? 'Position your face in the oval'
    : countdown
    ? 'Hold still — recording starts soon'
    : 'Stay still. Look at the camera.';

  useEffect(() => {
    if (msg !== prevMsg.current) {
      Animated.sequence([
        Animated.timing(fadeAnim, { toValue: 0, duration: 150, useNativeDriver: true }),
        Animated.timing(fadeAnim, { toValue: 1, duration: 300, useNativeDriver: true }),
      ]).start();
      prevMsg.current = msg;
    }
  }, [msg]);

  return (
    <Animated.View style={[styles.instructBar, { opacity: fadeAnim }]}>
      <Text style={styles.instructText}>{msg}</Text>
    </Animated.View>
  );
}

export default function RecordScreen() {
  const nav = useNavigation<any>();
  const camRef = useRef<CameraView>(null);
  const [permission, requestPermission] = useCameraPermissions();

  const [isRecording, setIsRecording]   = useState(false);
  const [countdown3, setCountdown3]     = useState(false);
  const [countdownN, setCountdownN]     = useState(3);
  const [timeLeft, setTimeLeft]         = useState(RECORD_DURATION);
  const [progress, setProgress]         = useState(0);
  const [quality, setQuality]           = useState(0); // simulated signal quality

  const timerRef   = useRef<NodeJS.Timeout | null>(null);
  const qualityRef = useRef<NodeJS.Timeout | null>(null);
  const btnScale   = useRef(new Animated.Value(1)).current;
  const qualityAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    if (!permission?.granted) requestPermission();
  }, []);

  // Simulate signal quality meter while recording
  useEffect(() => {
    if (isRecording) {
      qualityRef.current = setInterval(() => {
        const q = 0.6 + Math.random() * 0.35;
        setQuality(q);
        Animated.timing(qualityAnim, { toValue: q, duration: 400, useNativeDriver: false }).start();
      }, 600);
    }
    return () => { if (qualityRef.current) clearInterval(qualityRef.current); };
  }, [isRecording]);

  const startCountdown = async () => {
    await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    setCountdown3(true);
    setCountdownN(3);

    let n = 3;
    const tick = setInterval(async () => {
      n--;
      if (n > 0) {
        setCountdownN(n);
        await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
      } else {
        clearInterval(tick);
        setCountdown3(false);
        startRecording();
      }
    }, 1000);
  };

  const startRecording = useCallback(async () => {
    if (!camRef.current) return;
    setIsRecording(true);
    setTimeLeft(RECORD_DURATION);
    await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);

    let elapsed = 0;
    timerRef.current = setInterval(() => {
      elapsed++;
      const left = RECORD_DURATION - elapsed;
      setTimeLeft(left);
      setProgress(elapsed / RECORD_DURATION);

      if (elapsed >= RECORD_DURATION) {
        stopRecording();
      }
    }, 1000);

    try {
      const video = await camRef.current.recordAsync({ maxDuration: RECORD_DURATION });
      if (video?.uri) {
        nav.navigate('Processing', { videoUri: video.uri });
      }
    } catch (e) {
      console.error('Recording error:', e);
      Alert.alert('Recording failed', 'Please try again.');
      reset();
    }
  }, [nav]);

  const stopRecording = useCallback(async () => {
    if (timerRef.current) clearInterval(timerRef.current);
    await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    camRef.current?.stopRecording();
    setIsRecording(false);
  }, []);

  const reset = () => {
    if (timerRef.current) clearInterval(timerRef.current);
    setIsRecording(false);
    setCountdown3(false);
    setTimeLeft(RECORD_DURATION);
    setProgress(0);
  };

  if (!permission?.granted) {
    return (
      <View style={styles.permView}>
        <Text style={styles.permTitle}>Camera Access Required</Text>
        <Text style={styles.permBody}>CardioVision needs camera access to measure your heart rate from facial video.</Text>
        <TouchableOpacity style={styles.permBtn} onPress={requestPermission}>
          <Text style={styles.permBtnText}>Grant Access</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const qualityW = qualityAnim.interpolate({ inputRange: [0, 1], outputRange: ['0%', '100%'] });

  return (
    <View style={styles.root}>
      {/* Camera */}
      <CameraView
        ref={camRef}
        style={StyleSheet.absoluteFill}
        facing="front"
        mode="video"
        videoQuality="720p"
      />

      {/* Dark overlay */}
      <View style={styles.overlay} />

      {/* Face guide */}
      <View style={styles.guideContainer}>
        <FaceGuideOverlay isRecording={isRecording} />
      </View>

      {/* Countdown */}
      {countdown3 && (
        <View style={styles.countdownOverlay}>
          <Animated.Text style={styles.countdownNum}>{countdownN}</Animated.Text>
        </View>
      )}

      <SafeAreaView style={styles.safe} edges={['top', 'bottom']}>
        {/* ── TOP BAR ── */}
        <View style={styles.topBar}>
          <TouchableOpacity style={styles.backBtn} onPress={() => { reset(); nav.goBack(); }}>
            <Text style={styles.backText}>✕</Text>
          </TouchableOpacity>
          <Text style={styles.topTitle}>
            {isRecording ? '● RECORDING' : 'SCAN SETUP'}
          </Text>
          <View style={{ width: 40 }} />
        </View>

        {/* ── INSTRUCTION BAR ── */}
        <InstructionBar recording={isRecording} countdown={countdown3} />

        <View style={{ flex: 1 }} />

        {/* ── SIGNAL QUALITY (only when recording) ── */}
        {isRecording && (
          <View style={styles.qualityBar}>
            <Text style={styles.qualityLabel}>Signal Quality</Text>
            <View style={styles.qualityTrack}>
              <Animated.View style={[styles.qualityFill, { width: qualityW }]} />
            </View>
            <Text style={styles.qualityPct}>{Math.round(quality * 100)}%</Text>
          </View>
        )}

        {/* ── TIPS ── */}
        {!isRecording && !countdown3 && (
          <View style={styles.tipsCard}>
            {['Ensure your face is well-lit', 'Keep phone at arm\'s length', 'Remain still during recording', 'Avoid backlit or dark environments'].map((tip, i) => (
              <View key={i} style={styles.tipRow}>
                <Text style={styles.tipDot}>○</Text>
                <Text style={styles.tipText}>{tip}</Text>
              </View>
            ))}
          </View>
        )}

        {/* ── BOTTOM CONTROLS ── */}
        <View style={styles.bottomBar}>
          {isRecording ? (
            <>
              <CountdownRing progress={progress} timeLeft={timeLeft} />
              <TouchableOpacity style={styles.stopBtn} onPress={stopRecording}>
                <View style={styles.stopSquare} />
              </TouchableOpacity>
            </>
          ) : (
            <TouchableOpacity
              style={styles.recordBtn}
              onPress={startCountdown}
              disabled={countdown3}
              activeOpacity={0.8}
            >
              <View style={styles.recordBtnOuter}>
                <View style={styles.recordBtnInner} />
              </View>
              <Text style={styles.recordBtnLabel}>
                {countdown3 ? `Starting in ${countdownN}...` : 'Tap to record 30s'}
              </Text>
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
  overlay: { ...StyleSheet.absoluteFillObject, backgroundColor: 'rgba(0,0,0,0.45)' },

  guideContainer: {
    ...StyleSheet.absoluteFillObject,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: -60,
  },
  faceOval: {
    borderWidth: 1.5,
    borderStyle: 'dashed',
    backgroundColor: 'transparent',
  },

  countdownOverlay: {
    ...StyleSheet.absoluteFillObject,
    alignItems: 'center', justifyContent: 'center',
    backgroundColor: 'rgba(0,0,0,0.4)',
  },
  countdownNum: {
    fontFamily: 'SpaceGrotesk-Bold',
    fontSize: 120, color: Colors.white, opacity: 0.9,
  },

  topBar: {
    flexDirection: 'row', alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: Spacing.lg, paddingTop: Spacing.md,
  },
  backBtn: {
    width: 40, height: 40, borderRadius: 20,
    backgroundColor: Colors.glass,
    borderWidth: 1, borderColor: Colors.glassBorder,
    alignItems: 'center', justifyContent: 'center',
  },
  backText: { color: Colors.white, fontSize: 16 },
  topTitle: {
    ...Typography.label,
    color: Colors.textSecondary,
    letterSpacing: 2,
  },

  instructBar: {
    alignItems: 'center', marginTop: Spacing.sm,
    paddingHorizontal: Spacing.xl,
  },
  instructText: {
    ...Typography.body, color: Colors.fog,
    textAlign: 'center',
  },

  tipsCard: {
    marginHorizontal: Spacing.lg,
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderWidth: 1, borderColor: Colors.glassBorder,
    borderRadius: Radius.lg,
    padding: Spacing.md, marginBottom: Spacing.md,
  },
  tipRow: { flexDirection: 'row', alignItems: 'flex-start', marginBottom: 8 },
  tipDot: { color: Colors.textMuted, marginRight: 8, marginTop: 1 },
  tipText: { ...Typography.bodySmall, color: Colors.textSecondary, flex: 1 },

  qualityBar: {
    flexDirection: 'row', alignItems: 'center',
    marginHorizontal: Spacing.lg, marginBottom: Spacing.md,
    backgroundColor: Colors.glass, borderRadius: Radius.md,
    padding: Spacing.md, borderWidth: 1, borderColor: Colors.glassBorder,
  },
  qualityLabel: { ...Typography.label, width: 80 },
  qualityTrack: {
    flex: 1, height: 3, backgroundColor: Colors.border,
    borderRadius: 2, marginHorizontal: 12, overflow: 'hidden',
  },
  qualityFill: {
    height: '100%', backgroundColor: Colors.white,
    borderRadius: 2,
  },
  qualityPct: { ...Typography.mono, width: 36, textAlign: 'right' },

  bottomBar: {
    alignItems: 'center', paddingBottom: Spacing.xl,
    paddingTop: Spacing.md,
  },

  // Record button
  recordBtn: { alignItems: 'center' },
  recordBtnOuter: {
    width: 80, height: 80, borderRadius: 40,
    borderWidth: 3, borderColor: Colors.white,
    alignItems: 'center', justifyContent: 'center',
  },
  recordBtnInner: {
    width: 56, height: 56, borderRadius: 28,
    backgroundColor: Colors.white,
  },
  recordBtnLabel: {
    ...Typography.bodySmall,
    color: Colors.textSecondary, marginTop: 12,
  },

  // Countdown ring
  ringWrap: {
    width: 130, height: 130,
    alignItems: 'center', justifyContent: 'center',
  },
  timerInner: {
    position: 'absolute', alignItems: 'center',
  },
  timerNum: {
    fontFamily: 'SpaceGrotesk-Bold',
    fontSize: 36, color: Colors.white, letterSpacing: -1,
  },
  timerSec: { ...Typography.label, color: Colors.textMuted, marginTop: -2 },

  // Stop button
  stopBtn: {
    width: 56, height: 56, borderRadius: 28,
    borderWidth: 2, borderColor: Colors.white,
    alignItems: 'center', justifyContent: 'center',
    marginTop: 12,
  },
  stopSquare: { width: 22, height: 22, backgroundColor: Colors.white, borderRadius: 3 },

  // Permission
  permView: {
    flex: 1, backgroundColor: Colors.background,
    alignItems: 'center', justifyContent: 'center',
    padding: Spacing.xl,
  },
  permTitle: { ...Typography.h1, marginBottom: Spacing.md },
  permBody: { ...Typography.body, textAlign: 'center', marginBottom: Spacing.xl },
  permBtn: {
    backgroundColor: Colors.white, borderRadius: Radius.lg,
    paddingHorizontal: Spacing.xl, paddingVertical: Spacing.md,
  },
  permBtnText: {
    fontFamily: 'SpaceGrotesk-Bold', fontSize: 16, color: Colors.black,
  },
});

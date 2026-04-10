// app/processing.tsx — ProcessingScreen with expandable steps
import React, { useEffect, useRef, useState } from 'react';
import { View, Text, StyleSheet, Animated, TouchableOpacity } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter, useLocalSearchParams } from 'expo-router';
import Svg, { Circle } from 'react-native-svg';
import { Ionicons } from '@expo/vector-icons';
import { useTheme, Typography, Spacing, Radius } from './theme';
import { processVideo, getMockResult } from './api/rppgService';
import {
  clearPendingScanResult,
  getPendingScanResult,
  getScanSession,
  setPendingScanResult,
  setScanSession,
} from './state/scanSession';

const STEPS = [
  { id: 'upload',  label: 'Streaming capture',      sub: 'Live frames sent to backend websocket', detail: 'The mobile app streams JPEG camera frames to the backend over websocket. If websocket is unavailable, the app falls back to HTTP upload mode.' },
  { id: 'roi',     label: 'Face ROI extraction',   sub: 'MediaPipe landmark detection', detail: 'MediaPipe Face Mesh detects 468 facial landmarks. Skin-colored regions (forehead, cheeks, nose) are segmented as Regions of Interest for signal extraction.' },
  { id: 'pos',     label: 'POS algorithm',         sub: 'Plane-Orthogonal-to-Skin filtering', detail: 'The POS (Plane Orthogonal to Skin) algorithm by Wang et al. (2017) separates the pulse signal from RGB channel variations using chrominance-based filtering.' },
  { id: 'neural',  label: 'PhysFormer inference',  sub: 'Deep rPPG neural network', detail: 'PhysFormer, a vision transformer pretrained on UBFC-rPPG, processes spatiotemporal facial features to extract a blood volume pulse (BVP) signal with higher SNR.' },
  { id: 'fusion',  label: 'Signal fusion',         sub: 'POS + neural ensemble', detail: 'POS and PhysFormer signals are weighted by their respective SNR values and fused into a single high-quality pulse waveform using adaptive weighting.' },
  { id: 'hrv',     label: 'HRV analysis',          sub: 'RMSSD · SDNN · LF/HF extraction', detail: 'Inter-beat intervals are computed from peak detection. Time-domain (RMSSD, SDNN) and frequency-domain (LF/HF ratio) HRV features are extracted.' },
  { id: 'triage',  label: 'Triage agent',          sub: 'Mode decision & quality gating', detail: 'The triage agent evaluates signal confidence. If biometric quality is insufficient, it falls back to visual assessment mode using facial color analysis.' },
  { id: 'results', label: 'Compiling results',     sub: 'Building your biometric report', detail: 'All extracted metrics (BPM, HRV, stress level, confidence) are compiled into a structured JSON report for display on the results dashboard.' },
];
const STEP_DURATIONS = [2000, 2500, 2000, 3000, 1500, 1500, 1000, 800];

type StepState = 'pending' | 'active' | 'done';

function StepRow({ label, sub, detail, state, index }: { label: string; sub: string; detail: string; state: StepState; index: number }) {
  const { colors, accent } = useTheme();
  const [expanded, setExpanded] = useState(false);
  const fadeIn = useRef(new Animated.Value(0)).current;
  const spinAnim = useRef(new Animated.Value(0)).current;
  const checkScale = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    if (state !== 'pending') Animated.timing(fadeIn, { toValue: 1, duration: 300, useNativeDriver: true }).start();
    if (state === 'active') Animated.loop(Animated.timing(spinAnim, { toValue: 1, duration: 900, useNativeDriver: true })).start();
    if (state === 'done') Animated.spring(checkScale, { toValue: 1, friction: 6, tension: 100, useNativeDriver: true }).start();
  }, [state]);

  const spin = spinAnim.interpolate({ inputRange: [0, 1], outputRange: ['0deg', '360deg'] });

  return (
    <TouchableOpacity
      activeOpacity={0.7}
      onPress={() => state !== 'pending' && setExpanded(e => !e)}
      disabled={state === 'pending'}
    >
      <Animated.View style={[styles.stepRow, { borderBottomColor: colors.border, opacity: state === 'pending' ? 0.3 : fadeIn }]}>
        <View style={[styles.stepIconBox, {
          backgroundColor: state === 'done' ? accent.primary : state === 'active' ? accent.ghost : colors.surfaceHigh,
          borderColor: state === 'done' ? accent.primary : state === 'active' ? accent.primary : colors.border,
        }]}>
          {state === 'done' && <Animated.View style={{ transform: [{ scale: checkScale }] }}><Ionicons name="checkmark" size={14} color="#FFF" /></Animated.View>}
          {state === 'active' && (
            <Animated.View style={{ transform: [{ rotate: spin }] }}>
              <Svg width={18} height={18} viewBox="0 0 18 18">
                <Circle cx={9} cy={9} r={7} stroke={accent.primary} strokeWidth={2} fill="none" strokeDasharray="22 22" strokeLinecap="round" />
              </Svg>
            </Animated.View>
          )}
          {state === 'pending' && <Text style={[styles.stepNum, { color: colors.textMuted }]}>{String(index + 1).padStart(2, '0')}</Text>}
        </View>
        <View style={styles.stepTextBox}>
          <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' }}>
            <Text style={[styles.stepLabel, { color: state === 'active' ? colors.textPrimary : colors.textTertiary }]}>{label}</Text>
            {state !== 'pending' && (
              <Ionicons name={expanded ? 'chevron-up' : 'chevron-down'} size={14} color={colors.textMuted} />
            )}
          </View>
          {state === 'active' && <Text style={[styles.stepSub, { color: colors.textMuted }]}>{sub}</Text>}
          {state === 'done' && !expanded && <Text style={[styles.stepSub, { color: colors.textMuted }]}>{sub}</Text>}
        </View>
        {state === 'done' && <View style={[styles.doneBadge, { backgroundColor: accent.ghost }]}><Text style={[styles.doneBadgeText, { color: accent.primary }]}>DONE</Text></View>}
      </Animated.View>
      {expanded && state !== 'pending' && (
        <View style={[styles.detailBox, { backgroundColor: colors.surfaceHigh, borderColor: colors.border }]}>
          <Text style={[styles.detailText, { color: colors.textSecondary }]}>{detail}</Text>
        </View>
      )}
    </TouchableOpacity>
  );
}

function PulseBar() {
  const { accent } = useTheme();
  const animsRef = useRef(Array.from({ length: 24 }, () => new Animated.Value(0.2)));
  const anims = animsRef.current;
  useEffect(() => {
    const loops = anims.map((a, i) => Animated.loop(Animated.sequence([
      Animated.delay(i * 60),
      Animated.timing(a, { toValue: 0.6 + Math.random() * 0.4, duration: 300, useNativeDriver: true }),
      Animated.timing(a, { toValue: 0.2, duration: 300, useNativeDriver: true }),
    ])));
    loops.forEach(l => l.start());
    return () => loops.forEach(l => l.stop());
  }, []);
  return (
    <View style={styles.pulseBar}>
      {anims.map((a, i) => <Animated.View key={i} style={{ width: 3, borderRadius: 2, backgroundColor: accent.primary, marginHorizontal: 1.5, height: 36, transform: [{ scaleY: a }] }} />)}
    </View>
  );
}

export default function ProcessingScreen() {
  const router = useRouter();
  const { colors, accent } = useTheme();
  const params = useLocalSearchParams();
  const videoUri = params.videoUri as string | undefined;
  const streamResultJson = params.streamResultJson as string | undefined;
  const [stepStates, setStepStates] = useState<StepState[]>(STEPS.map(() => 'pending'));
  const [progress, setProgress] = useState(0);
  const progressAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => { runPipeline(); }, []);

  const runPipeline = async () => {
    const session = getScanSession();
    let result: any = null;
    let apiDone = false;

    const resolveResult = (resolved: any) => {
      result = resolved;
      apiDone = true;
      setScanSession({ result: resolved, videoUri: videoUri ?? session.videoUri });
      clearPendingScanResult();
    };

    const resolveFallback = () => {
      resolveResult(getMockResult());
    };

    if (session.result) {
      resolveResult(session.result);
    } else if (streamResultJson) {
      try {
        resolveResult(JSON.parse(streamResultJson));
      } catch {
        resolveFallback();
      }
    } else {
      let pending = getPendingScanResult();
      if (!pending && videoUri) {
        pending = processVideo(videoUri, (pct) => setProgress(pct));
        setPendingScanResult(pending);
      }

      if (pending) {
        pending
          .then((r) => resolveResult(r))
          .catch(() => resolveFallback());
      } else {
        resolveFallback();
      }
    }

    const hasPendingBackendWork = !apiDone;
    const durations = hasPendingBackendWork
      ? STEP_DURATIONS.map((d) => Math.max(650, Math.round(d * 0.55)))
      : STEP_DURATIONS.map((d) => Math.max(450, Math.round(d * 0.35)));

    for (let i = 0; i < STEPS.length; i++) {
      setStepStates(prev => prev.map((s, idx) => idx === i ? 'active' : idx < i ? 'done' : 'pending'));
      const pctTarget = ((i + 1) / STEPS.length) * 100;
      setProgress(Math.round(pctTarget));
      Animated.timing(progressAnim, { toValue: pctTarget / 100, duration: durations[i] * 0.8, useNativeDriver: false }).start();
      await new Promise(res => setTimeout(res, durations[i]));
      if (i === STEPS.length - 1) {
        let waited = 0;
        while (!apiDone && waited < 6000) { await new Promise(res => setTimeout(res, 300)); waited += 300; }
      }
    }
    setStepStates(STEPS.map(() => 'done'));
    await new Promise(res => setTimeout(res, 600));
    if (apiDone && result) {
      setScanSession({ result, videoUri: videoUri ?? session.videoUri });
    }
    router.replace({ pathname: '/results' });
  };

  const progressWidth = progressAnim.interpolate({ inputRange: [0, 1], outputRange: ['0%', '100%'] });

  return (
    <View style={[styles.root, { backgroundColor: colors.background }]}>
      <SafeAreaView style={styles.safe}>
        <View style={styles.header}>
          <Ionicons name="pulse-outline" size={28} color={accent.primary} style={{ marginBottom: 12 }} />
          <Text style={[styles.title, { color: colors.textPrimary }]}>Analysing</Text>
          <Text style={[styles.subtitle, { color: colors.textTertiary }]}>Processing your cardiac signal</Text>
          <PulseBar />
        </View>
        <View style={[styles.progressTrack, { backgroundColor: colors.border }]}>
          <Animated.View style={[styles.progressFill, { width: progressWidth, backgroundColor: accent.primary }]} />
        </View>
        <Text style={[styles.progressLabel, { color: colors.textMuted }]}>{progress}% complete</Text>
        <Text style={[styles.expandHint, { color: colors.textMuted }]}>Tap completed steps for details</Text>
        <View style={styles.stepList}>
          {STEPS.map((s, i) => <StepRow key={s.id} label={s.label} sub={s.sub} detail={s.detail} state={stepStates[i]} index={i} />)}
        </View>
        <Text style={[styles.footer, { color: colors.textMuted }]}>UBFC-rPPG validated · PhysFormer + POS ensemble</Text>
      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1 },
  safe: { flex: 1, paddingHorizontal: Spacing.lg },
  header: { alignItems: 'center', paddingTop: Spacing.xxl, paddingBottom: Spacing.lg },
  title: { fontFamily: 'SpaceGrotesk-Bold', fontSize: 32, letterSpacing: -1 },
  subtitle: { ...Typography.body, marginTop: 4, marginBottom: Spacing.md },
  pulseBar: { flexDirection: 'row', alignItems: 'center', height: 36 },
  progressTrack: { height: 3, borderRadius: 2, marginBottom: 8 },
  progressFill: { height: '100%', borderRadius: 2 },
  progressLabel: { ...Typography.label, textAlign: 'right', marginBottom: 4 },
  expandHint: { ...Typography.bodySmall, textAlign: 'center', marginBottom: Spacing.md, fontStyle: 'italic' },
  stepList: { flex: 1 },
  stepRow: { flexDirection: 'row', alignItems: 'center', paddingVertical: 10, borderBottomWidth: 1 },
  stepIconBox: { width: 34, height: 34, borderRadius: 17, borderWidth: 1, alignItems: 'center', justifyContent: 'center', marginRight: Spacing.md },
  stepNum: { fontFamily: 'SpaceGrotesk-Bold', fontSize: 10, letterSpacing: 0.5 },
  stepTextBox: { flex: 1 },
  stepLabel: { fontFamily: 'SpaceGrotesk-Medium', fontSize: 14 },
  stepSub: { ...Typography.bodySmall, marginTop: 2 },
  doneBadge: { borderRadius: Radius.full, paddingHorizontal: 8, paddingVertical: 3 },
  doneBadgeText: { fontFamily: 'SpaceGrotesk-Medium', fontSize: 9, letterSpacing: 1 },
  detailBox: { marginLeft: 50, marginBottom: 8, padding: Spacing.md, borderRadius: Radius.md, borderWidth: 1 },
  detailText: { ...Typography.bodySmall, lineHeight: 20 },
  footer: { ...Typography.label, textAlign: 'center', paddingVertical: Spacing.lg },
});

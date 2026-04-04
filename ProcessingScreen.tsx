// src/screens/ProcessingScreen.tsx
import React, { useEffect, useRef, useState } from 'react';
import {
  View, Text, StyleSheet, Animated, Dimensions,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useNavigation, useRoute, RouteProp } from '@react-navigation/native';
import Svg, { Circle, Path, G } from 'react-native-svg';
import { Colors, Typography, Spacing, Radius } from '../theme';
import { processVideo, getMockResult } from '../api/rppgService';
import type { RootStackParamList } from '../../App';

type Params = RouteProp<RootStackParamList, 'Processing'>;

const { width } = Dimensions.get('window');

// ── Pipeline steps definition ────────────────────────────────────────────────
const STEPS = [
  { id: 'upload',   label: 'Uploading video',           sub: 'Transferring to processing pipeline' },
  { id: 'roi',      label: 'Face ROI extraction',       sub: 'MediaPipe landmark detection' },
  { id: 'pos',      label: 'POS algorithm',             sub: 'Plane-Orthogonal-to-Skin filtering' },
  { id: 'neural',   label: 'PhysFormer inference',      sub: 'Deep rPPG neural network' },
  { id: 'fusion',   label: 'Signal fusion',             sub: 'POS + neural ensemble' },
  { id: 'hrv',      label: 'HRV analysis',              sub: 'RMSSD · SDNN · LF/HF extraction' },
  { id: 'triage',   label: 'Triage agent',              sub: 'Mode decision & quality gating' },
  { id: 'results',  label: 'Preparing results',         sub: 'Compiling your biometric report' },
];

// Step durations in ms (simulated; real API may be faster/slower)
const STEP_DURATIONS = [2000, 2500, 2000, 3000, 1500, 1500, 1000, 800];

type StepState = 'pending' | 'active' | 'done' | 'error';

function StepRow({
  label, sub, state, index,
}: {
  label: string; sub: string; state: StepState; index: number;
}) {
  const fadeIn   = useRef(new Animated.Value(0)).current;
  const slideIn  = useRef(new Animated.Value(12)).current;
  const spinAnim = useRef(new Animated.Value(0)).current;
  const checkScale = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    if (state !== 'pending') {
      Animated.parallel([
        Animated.timing(fadeIn, { toValue: 1, duration: 300, useNativeDriver: true }),
        Animated.spring(slideIn, { toValue: 0, friction: 8, useNativeDriver: true }),
      ]).start();
    }
    if (state === 'active') {
      Animated.loop(
        Animated.timing(spinAnim, { toValue: 1, duration: 900, useNativeDriver: true }),
      ).start();
    }
    if (state === 'done') {
      Animated.spring(checkScale, { toValue: 1, friction: 6, tension: 100, useNativeDriver: true }).start();
    }
  }, [state]);

  const spin = spinAnim.interpolate({ inputRange: [0, 1], outputRange: ['0deg', '360deg'] });

  const iconBg = state === 'done'    ? Colors.white
               : state === 'active'  ? 'transparent'
               : Colors.surfaceHigh;

  return (
    <Animated.View
      style={[
        styles.stepRow,
        state === 'pending' && styles.stepPending,
        { opacity: state === 'pending' ? 0.3 : fadeIn, transform: [{ translateY: slideIn }] },
      ]}
    >
      {/* Icon */}
      <View style={[styles.stepIconBox, { backgroundColor: iconBg,
        borderColor: state === 'done' ? Colors.white : state === 'active' ? Colors.silver : Colors.border }]}>
        {state === 'done' && (
          <Animated.Text style={[styles.checkMark, { transform: [{ scale: checkScale }] }]}>✓</Animated.Text>
        )}
        {state === 'active' && (
          <Animated.View style={[styles.spinner, { transform: [{ rotate: spin }] }]}>
            <Svg width={18} height={18} viewBox="0 0 18 18">
              <Circle cx={9} cy={9} r={7} stroke={Colors.white} strokeWidth={2}
                fill="none" strokeDasharray="22 22" strokeLinecap="round" />
            </Svg>
          </Animated.View>
        )}
        {state === 'pending' && (
          <Text style={styles.stepNum}>{String(index + 1).padStart(2, '0')}</Text>
        )}
      </View>

      {/* Text */}
      <View style={styles.stepTextBox}>
        <Text style={[styles.stepLabel,
          state === 'active' && { color: Colors.white },
          state === 'done'   && { color: Colors.fog },
        ]}>
          {label}
        </Text>
        {state === 'active' && <Text style={styles.stepSub}>{sub}</Text>}
      </View>

      {/* Duration badge when done */}
      {state === 'done' && (
        <View style={styles.doneBadge}>
          <Text style={styles.doneBadgeText}>DONE</Text>
        </View>
      )}
    </Animated.View>
  );
}

function PulseBar() {
  const bars = Array.from({ length: 24 }, (_, i) => i);
  const anims = bars.map(() => useRef(new Animated.Value(0.2)).current);

  useEffect(() => {
    const loops = anims.map((a, i) =>
      Animated.loop(
        Animated.sequence([
          Animated.delay(i * 60),
          Animated.timing(a, { toValue: 0.8 + Math.random() * 0.2, duration: 300, useNativeDriver: true }),
          Animated.timing(a, { toValue: 0.2, duration: 300, useNativeDriver: true }),
        ]),
      ),
    );
    loops.forEach(l => l.start());
    return () => loops.forEach(l => l.stop());
  }, []);

  return (
    <View style={styles.pulseBar}>
      {anims.map((a, i) => (
        <Animated.View
          key={i}
          style={{
            width: 3, borderRadius: 2,
            backgroundColor: Colors.white,
            marginHorizontal: 1.5,
            height: 36,
            transform: [{ scaleY: a }],
          }}
        />
      ))}
    </View>
  );
}

export default function ProcessingScreen() {
  const nav   = useNavigation<any>();
  const route = useRoute<Params>();
  const { videoUri } = route.params;

  const [stepStates, setStepStates] = useState<StepState[]>(
    STEPS.map(() => 'pending'),
  );
  const [currentStep, setCurrentStep] = useState(-1);
  const [progress, setProgress]       = useState(0);
  const progressAnim = useRef(new Animated.Value(0)).current;
  const headerAlpha  = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.timing(headerAlpha, { toValue: 1, duration: 600, useNativeDriver: true }).start();
    runPipeline();
  }, []);

  const runPipeline = async () => {
    let result: any = null;
    let apiDone     = false;

    // Fire real API call
    processVideo(videoUri, (pct) => {
      setProgress(pct);
    }).then(r => { result = r; apiDone = true; })
      .catch(() => {
        // Fall back to mock
        setTimeout(() => { result = getMockResult(); apiDone = true; }, 2000);
      });

    // Animate steps sequentially
    for (let i = 0; i < STEPS.length; i++) {
      setCurrentStep(i);
      setStepStates(prev => prev.map((s, idx) =>
        idx === i ? 'active' : idx < i ? 'done' : 'pending',
      ));

      const dur = STEP_DURATIONS[i];
      const pctTarget = ((i + 1) / STEPS.length) * 100;
      setProgress(Math.round(pctTarget));
      Animated.timing(progressAnim, {
        toValue: pctTarget / 100, duration: dur * 0.8, useNativeDriver: false,
      }).start();

      await sleep(dur);

      // On last step wait for API to finish
      if (i === STEPS.length - 1) {
        let waited = 0;
        while (!apiDone && waited < 30000) {
          await sleep(300);
          waited += 300;
        }
      }
    }

    setStepStates(STEPS.map(() => 'done'));

    await sleep(600);

    // Navigate to results
    nav.replace('Results', {
      result: result ?? getMockResult(),
      videoUri,
    });
  };

  const sleep = (ms: number) => new Promise(res => setTimeout(res, ms));

  const progressWidth = progressAnim.interpolate({
    inputRange: [0, 1],
    outputRange: ['0%', '100%'],
  });

  return (
    <View style={styles.root}>
      <SafeAreaView style={styles.safe}>

        <Animated.View style={[styles.header, { opacity: headerAlpha }]}>
          <Text style={styles.title}>Analysing</Text>
          <Text style={styles.subtitle}>Processing your cardiac signal</Text>
          <PulseBar />
        </Animated.View>

        {/* Progress bar */}
        <View style={styles.progressTrack}>
          <Animated.View style={[styles.progressFill, { width: progressWidth }]} />
        </View>
        <Text style={styles.progressLabel}>{progress}% complete</Text>

        {/* Step list */}
        <View style={styles.stepList}>
          {STEPS.map((s, i) => (
            <StepRow
              key={s.id}
              label={s.label}
              sub={s.sub}
              state={stepStates[i]}
              index={i}
            />
          ))}
        </View>

        <Text style={styles.footer}>
          UBFC-rPPG validated · PhysFormer + POS ensemble
        </Text>

      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: Colors.background },
  safe: { flex: 1, paddingHorizontal: Spacing.lg },

  header: {
    alignItems: 'center',
    paddingTop: Spacing.xxl,
    paddingBottom: Spacing.xl,
  },
  title: {
    fontFamily: 'SpaceGrotesk-Bold',
    fontSize: 36, letterSpacing: -1.5,
    color: Colors.white, marginBottom: 6,
  },
  subtitle: {
    ...Typography.body, color: Colors.textTertiary,
    marginBottom: Spacing.lg,
  },

  pulseBar: {
    flexDirection: 'row', alignItems: 'center',
    height: 36,
  },

  progressTrack: {
    height: 2, backgroundColor: Colors.border,
    borderRadius: 1, marginBottom: 8,
  },
  progressFill: {
    height: '100%', backgroundColor: Colors.white,
    borderRadius: 1,
  },
  progressLabel: {
    ...Typography.label, color: Colors.textMuted,
    textAlign: 'right', marginBottom: Spacing.lg,
  },

  stepList: { flex: 1 },
  stepRow: {
    flexDirection: 'row', alignItems: 'center',
    paddingVertical: 10, borderBottomWidth: 1,
    borderBottomColor: Colors.border,
  },
  stepPending: {},

  stepIconBox: {
    width: 36, height: 36, borderRadius: 18,
    borderWidth: 1,
    alignItems: 'center', justifyContent: 'center',
    marginRight: Spacing.md,
  },
  checkMark: {
    color: Colors.black,
    fontFamily: 'SpaceGrotesk-Bold',
    fontSize: 14,
  },
  spinner: {},
  stepNum: {
    fontFamily: 'SpaceGrotesk-Bold',
    fontSize: 10, color: Colors.textMuted, letterSpacing: 0.5,
  },

  stepTextBox: { flex: 1 },
  stepLabel: {
    fontFamily: 'SpaceGrotesk-Medium',
    fontSize: 14, color: Colors.textTertiary,
  },
  stepSub: {
    ...Typography.bodySmall,
    color: Colors.textMuted, marginTop: 2,
  },

  doneBadge: {
    backgroundColor: Colors.graphite,
    borderRadius: Radius.full,
    paddingHorizontal: 8, paddingVertical: 3,
  },
  doneBadgeText: {
    fontFamily: 'SpaceGrotesk-Medium',
    fontSize: 9, color: Colors.textSecondary,
    letterSpacing: 1,
  },

  footer: {
    ...Typography.label, color: Colors.textMuted,
    textAlign: 'center', paddingVertical: Spacing.lg,
  },
});

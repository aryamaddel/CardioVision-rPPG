// src/screens/HomeScreen.tsx
import React, { useEffect, useRef, useState } from 'react';
import {
  View, Text, TouchableOpacity, StyleSheet,
  Animated, Dimensions, ScrollView,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useNavigation } from '@react-navigation/native';
import { LinearGradient } from 'expo-linear-gradient';
import * as Haptics from 'expo-haptics';
import Svg, { Path, Circle, Line, Defs, RadialGradient, Stop } from 'react-native-svg';
import { Colors, Typography, Spacing, Radius } from '../theme';

const { width, height } = Dimensions.get('window');
const W = width;

// ── Animated heartbeat ECG path ───────────────────────────────────────────────
const ECG_PATH = `M0,50 L${W*0.1},50 L${W*0.15},50 L${W*0.17},20 L${W*0.19},80 L${W*0.22},15 L${W*0.25},50 L${W*0.3},50 L${W*0.45},50 L${W*0.47},20 L${W*0.49},80 L${W*0.52},15 L${W*0.55},50 L${W*0.7},50 L${W*0.72},20 L${W*0.74},80 L${W*0.77},15 L${W*0.8},50 L${W},50`;

function PulseRing({ size, delay, opacity }: { size: number; delay: number; opacity: number }) {
  const scale = useRef(new Animated.Value(0.6)).current;
  const alpha  = useRef(new Animated.Value(opacity)).current;

  useEffect(() => {
    const loop = Animated.loop(
      Animated.sequence([
        Animated.delay(delay),
        Animated.parallel([
          Animated.timing(scale, { toValue: 1, duration: 2000, useNativeDriver: true }),
          Animated.timing(alpha, { toValue: 0, duration: 2000, useNativeDriver: true }),
        ]),
        Animated.parallel([
          Animated.timing(scale, { toValue: 0.6, duration: 0, useNativeDriver: true }),
          Animated.timing(alpha, { toValue: opacity, duration: 0, useNativeDriver: true }),
        ]),
      ]),
    );
    loop.start();
    return () => loop.stop();
  }, []);

  return (
    <Animated.View
      style={{
        position: 'absolute',
        width: size, height: size,
        borderRadius: size / 2,
        borderWidth: 1,
        borderColor: Colors.white,
        transform: [{ scale }],
        opacity: alpha,
      }}
    />
  );
}

function ECGLine() {
  const offset = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    const loop = Animated.loop(
      Animated.timing(offset, {
        toValue: -W,
        duration: 3000,
        useNativeDriver: true,
      }),
    );
    loop.start();
    return () => loop.stop();
  }, []);

  return (
    <View style={styles.ecgContainer} pointerEvents="none">
      <Animated.View style={{ transform: [{ translateX: offset }], width: W * 2 }}>
        <Svg width={W * 2} height={100} viewBox={`0 0 ${W * 2} 100`}>
          {/* First pass */}
          <Path d={ECG_PATH} stroke={Colors.white} strokeWidth={1.5}
            fill="none" strokeLinecap="round" strokeLinejoin="round" opacity={0.18} />
          {/* Second pass (offset) */}
          <Path d={`M${W},50 ` + ECG_PATH.substring(ECG_PATH.indexOf(' '))}
            stroke={Colors.white} strokeWidth={1.5}
            fill="none" strokeLinecap="round" strokeLinejoin="round" opacity={0.18} />
        </Svg>
      </Animated.View>
    </View>
  );
}

function StatBadge({ label, value, sub }: { label: string; value: string; sub: string }) {
  const fadeIn = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    Animated.timing(fadeIn, { toValue: 1, duration: 800, delay: 600, useNativeDriver: true }).start();
  }, []);
  return (
    <Animated.View style={[styles.statBadge, { opacity: fadeIn }]}>
      <Text style={styles.statLabel}>{label}</Text>
      <Text style={styles.statValue}>{value}</Text>
      <Text style={styles.statSub}>{sub}</Text>
    </Animated.View>
  );
}

export default function HomeScreen() {
  const nav = useNavigation<any>();
  const heartScale = useRef(new Animated.Value(1)).current;
  const heroY      = useRef(new Animated.Value(40)).current;
  const heroAlpha  = useRef(new Animated.Value(0)).current;
  const btnScale   = useRef(new Animated.Value(0.9)).current;
  const [bpmTick, setBpmTick] = useState(72);

  useEffect(() => {
    // Entry animation
    Animated.parallel([
      Animated.spring(heroY, { toValue: 0, friction: 8, tension: 50, useNativeDriver: true }),
      Animated.timing(heroAlpha, { toValue: 1, duration: 1000, useNativeDriver: true }),
      Animated.spring(btnScale, { toValue: 1, delay: 400, friction: 7, useNativeDriver: true }),
    ]).start();

    // Heart pulse
    const heartLoop = Animated.loop(
      Animated.sequence([
        Animated.timing(heartScale, { toValue: 1.12, duration: 300, useNativeDriver: true }),
        Animated.timing(heartScale, { toValue: 1.0,  duration: 300, useNativeDriver: true }),
        Animated.timing(heartScale, { toValue: 1.08, duration: 200, useNativeDriver: true }),
        Animated.timing(heartScale, { toValue: 1.0,  duration: 200, useNativeDriver: true }),
        Animated.delay(600),
      ]),
    );
    heartLoop.start();

    // Fake BPM ticker
    const bpmInterval = setInterval(() => {
      setBpmTick(prev => 68 + Math.floor(Math.random() * 8));
    }, 900);

    return () => { heartLoop.stop(); clearInterval(bpmInterval); };
  }, []);

  const handleStart = async () => {
    await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    Animated.sequence([
      Animated.timing(btnScale, { toValue: 0.95, duration: 80, useNativeDriver: true }),
      Animated.timing(btnScale, { toValue: 1,    duration: 80, useNativeDriver: true }),
    ]).start(() => nav.navigate('Record'));
  };

  return (
    <View style={styles.root}>
      <LinearGradient
        colors={['#000000', '#111111', '#0A0A0A']}
        style={StyleSheet.absoluteFill}
      />

      {/* Ambient glow */}
      <View style={styles.glowTop} />

      <ECGLine />

      <SafeAreaView style={styles.safe}>
        <ScrollView contentContainerStyle={styles.scroll} showsVerticalScrollIndicator={false}>

          {/* ── HEADER ── */}
          <Animated.View style={[styles.header, { opacity: heroAlpha, transform: [{ translateY: heroY }] }]}>
            <Text style={styles.badge}>CardioVision · rPPG</Text>
            <Text style={styles.tagline}>Contact-Free{'\n'}Cardiac Monitor</Text>
            <Text style={styles.subTagline}>
              Extracts your heart rate from facial video.{'\n'}
              No wearables. No hardware.
            </Text>
          </Animated.View>

          {/* ── LIVE DEMO RING ── */}
          <Animated.View style={[styles.ringContainer, { opacity: heroAlpha }]}>
            <PulseRing size={260} delay={0}    opacity={0.06} />
            <PulseRing size={210} delay={400}  opacity={0.10} />
            <PulseRing size={160} delay={800}  opacity={0.15} />

            <Animated.View style={[styles.centerRing, { transform: [{ scale: heartScale }] }]}>
              <View style={styles.bpmRing}>
                <Text style={styles.bpmNumber}>{bpmTick}</Text>
                <Text style={styles.bpmUnit}>BPM</Text>
                <Text style={styles.bpmSub}>live preview</Text>
              </View>
            </Animated.View>
          </Animated.View>

          {/* ── STAT STRIP ── */}
          <View style={styles.statStrip}>
            <StatBadge label="Method"    value="rPPG"    sub="Photoplethysmographic" />
            <View style={styles.statDivider} />
            <StatBadge label="Duration"  value="30s"     sub="Selfie video" />
            <View style={styles.statDivider} />
            <StatBadge label="Outputs"   value="4+"      sub="Biometric signals" />
          </View>

          {/* ── FEATURE PILLS ── */}
          <Animated.View style={[styles.pillRow, { opacity: heroAlpha }]}>
            {['Heart Rate', 'HRV Analysis', 'Stress Level', 'Signal Quality', 'Deep Neural', 'Triage AI'].map(p => (
              <View key={p} style={styles.pill}>
                <Text style={styles.pillText}>{p}</Text>
              </View>
            ))}
          </Animated.View>

          {/* ── PIPELINE VISUAL ── */}
          <View style={styles.pipelineCard}>
            <Text style={styles.pipelineTitle}>Signal Pipeline</Text>
            {[
              ['01', 'Face ROI Extraction', 'MediaPipe landmarks + skin mask'],
              ['02', 'POS Algorithm',        'Plane-Orthogonal-to-Skin (Wang 2017)'],
              ['03', 'PhysFormer Neural',    'Pretrained deep rPPG transformer'],
              ['04', 'HRV + Stress',         'RMSSD · SDNN · LF/HF classifier'],
              ['05', 'Triage Agent',         'Dynamic biometric ↔ visual mode'],
            ].map(([num, title, desc]) => (
              <View key={num} style={styles.pipelineRow}>
                <Text style={styles.pipelineNum}>{num}</Text>
                <View style={styles.pipelineContent}>
                  <Text style={styles.pipelineStep}>{title}</Text>
                  <Text style={styles.pipelineDesc}>{desc}</Text>
                </View>
              </View>
            ))}
          </View>

          {/* ── CTA BUTTON ── */}
          <Animated.View style={{ transform: [{ scale: btnScale }] }}>
            <TouchableOpacity onPress={handleStart} activeOpacity={0.9} style={styles.ctaButton}>
              <LinearGradient
                colors={['#FFFFFF', '#E0E0E0']}
                style={styles.ctaGradient}
              >
                <Text style={styles.ctaText}>Begin Scan</Text>
                <Text style={styles.ctaSubtext}>30-second facial recording</Text>
              </LinearGradient>
            </TouchableOpacity>
          </Animated.View>

          <Text style={styles.footer}>
            UBFC-rPPG validated · Clinical-grade confidence scoring
          </Text>

        </ScrollView>
      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: Colors.background },
  safe: { flex: 1 },
  scroll: { paddingHorizontal: Spacing.lg, paddingBottom: 60 },

  glowTop: {
    position: 'absolute',
    top: -100, left: W / 2 - 150,
    width: 300, height: 300,
    borderRadius: 150,
    backgroundColor: 'rgba(255,255,255,0.03)',
  },

  ecgContainer: {
    position: 'absolute',
    bottom: 160, left: 0, right: 0,
    height: 100, overflow: 'hidden',
  },

  header: { marginTop: Spacing.xl, alignItems: 'center' },

  badge: {
    ...Typography.label,
    color: Colors.textTertiary,
    borderWidth: 1, borderColor: Colors.border,
    paddingHorizontal: 12, paddingVertical: 4,
    borderRadius: Radius.full,
    marginBottom: Spacing.lg,
  },

  tagline: {
    fontFamily: 'SpaceGrotesk-Bold',
    fontSize: 40, lineHeight: 44,
    letterSpacing: -1.5,
    color: Colors.white,
    textAlign: 'center',
    marginBottom: Spacing.md,
  },

  subTagline: {
    ...Typography.body,
    textAlign: 'center',
    color: Colors.textTertiary,
    lineHeight: 22,
  },

  // Rings
  ringContainer: {
    alignItems: 'center', justifyContent: 'center',
    height: 280, marginTop: Spacing.xl,
  },
  centerRing: {
    width: 140, height: 140, borderRadius: 70,
    backgroundColor: Colors.surfaceHigh,
    borderWidth: 1, borderColor: Colors.borderLight,
    alignItems: 'center', justifyContent: 'center',
  },
  bpmRing: { alignItems: 'center' },
  bpmNumber: {
    fontFamily: 'SpaceGrotesk-Bold',
    fontSize: 42, letterSpacing: -2,
    color: Colors.white,
  },
  bpmUnit: {
    ...Typography.label,
    color: Colors.textSecondary,
    marginTop: -4,
  },
  bpmSub: {
    fontFamily: 'SpaceGrotesk-Regular',
    fontSize: 10, letterSpacing: 0.5,
    color: Colors.textMuted,
    marginTop: 2,
  },

  // Stats
  statStrip: {
    flexDirection: 'row',
    backgroundColor: Colors.surface,
    borderWidth: 1, borderColor: Colors.border,
    borderRadius: Radius.lg,
    padding: Spacing.md,
    marginTop: Spacing.lg,
  },
  statBadge: { flex: 1, alignItems: 'center' },
  statDivider: { width: 1, backgroundColor: Colors.border },
  statLabel: { ...Typography.label, marginBottom: 4 },
  statValue: {
    fontFamily: 'SpaceGrotesk-Bold',
    fontSize: 20, letterSpacing: -0.5,
    color: Colors.white,
  },
  statSub: {
    fontFamily: 'SpaceGrotesk-Regular',
    fontSize: 10, color: Colors.textMuted,
    marginTop: 2, textAlign: 'center',
  },

  // Pills
  pillRow: {
    flexDirection: 'row', flexWrap: 'wrap',
    gap: 8, marginTop: Spacing.lg,
    justifyContent: 'center',
  },
  pill: {
    backgroundColor: Colors.surfaceHigh,
    borderWidth: 1, borderColor: Colors.border,
    borderRadius: Radius.full,
    paddingHorizontal: 12, paddingVertical: 6,
  },
  pillText: { ...Typography.bodySmall, color: Colors.textSecondary },

  // Pipeline
  pipelineCard: {
    backgroundColor: Colors.surface,
    borderWidth: 1, borderColor: Colors.border,
    borderRadius: Radius.lg,
    padding: Spacing.lg,
    marginTop: Spacing.lg,
  },
  pipelineTitle: { ...Typography.label, marginBottom: Spacing.md },
  pipelineRow: {
    flexDirection: 'row', alignItems: 'flex-start',
    marginBottom: Spacing.md,
  },
  pipelineNum: {
    fontFamily: 'SpaceGrotesk-Bold',
    fontSize: 11, color: Colors.textMuted,
    letterSpacing: 1, marginRight: Spacing.md,
    marginTop: 2, width: 22,
  },
  pipelineContent: { flex: 1 },
  pipelineStep: { ...Typography.h3, fontSize: 14, marginBottom: 2 },
  pipelineDesc: { ...Typography.bodySmall, color: Colors.textTertiary },

  // CTA
  ctaButton: {
    borderRadius: Radius.lg, overflow: 'hidden',
    marginTop: Spacing.xl,
  },
  ctaGradient: {
    paddingVertical: 20, paddingHorizontal: Spacing.xl,
    alignItems: 'center',
  },
  ctaText: {
    fontFamily: 'SpaceGrotesk-Bold',
    fontSize: 18, letterSpacing: -0.3,
    color: Colors.black,
  },
  ctaSubtext: {
    fontFamily: 'SpaceGrotesk-Regular',
    fontSize: 12, color: '#555',
    marginTop: 3,
  },

  footer: {
    ...Typography.label,
    color: Colors.textMuted,
    textAlign: 'center',
    marginTop: Spacing.xl,
  },
});

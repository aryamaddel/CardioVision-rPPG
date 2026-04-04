// src/screens/VideoPlaybackScreen.tsx
// Shows the recorded video with:
//   1. Green-channel intensity overlay (tinted face)
//   2. Pulse beat markers synced to detected IBI timing
//   3. Live BPM counter matched to detected peaks
import React, { useRef, useState, useEffect, useCallback } from 'react';
import {
  View, Text, TouchableOpacity, StyleSheet,
  Animated, Dimensions, PanResponder,
} from 'react-native';
import { Video, ResizeMode, AVPlaybackStatus } from 'expo-av';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useNavigation, useRoute, RouteProp } from '@react-navigation/native';
import Svg, { Circle, Rect, Path, Defs, RadialGradient, Stop } from 'react-native-svg';
import * as Haptics from 'expo-haptics';
import { Colors, Typography, Spacing, Radius } from '../theme';
import type { RootStackParamList } from '../../App';

type Params = RouteProp<RootStackParamList, 'VideoPlayback'>;
const { width, height } = Dimensions.get('window');

// Face oval geometry (matches ROI region — forehead, cheeks, nose bridge)
const OVAL_W = width * 0.60;
const OVAL_H = OVAL_W * 1.30;
const OVAL_X = (width - OVAL_W) / 2;
const OVAL_Y = (height * 0.5 - OVAL_H) / 2 + 40;

// ROI sub-zones as approximate face proportions
const ROI_ZONES = [
  { id: 'forehead', label: 'Forehead',  x: 0.25, y: 0.08, w: 0.50, h: 0.18 },
  { id: 'leftCheek', label: 'L. Cheek', x: 0.08, y: 0.38, w: 0.25, h: 0.25 },
  { id: 'rightCheek',label: 'R. Cheek', x: 0.67, y: 0.38, w: 0.25, h: 0.25 },
  { id: 'nose',     label: 'Nose',      x: 0.35, y: 0.32, w: 0.30, h: 0.20 },
];

function PulseOverlay({
  pulsing, intensity, peakDetected,
}: { pulsing: boolean; intensity: number; peakDetected: boolean }) {
  const glowAnim   = useRef(new Animated.Value(0)).current;
  const beatScale  = useRef(new Animated.Value(1)).current;

  useEffect(() => {
    if (peakDetected) {
      // Sharp beat flash
      Animated.sequence([
        Animated.timing(glowAnim,  { toValue: 1, duration: 60,  useNativeDriver: true }),
        Animated.timing(glowAnim,  { toValue: 0, duration: 400, useNativeDriver: true }),
        Animated.timing(beatScale, { toValue: 1.04, duration: 80,  useNativeDriver: true }),
        Animated.timing(beatScale, { toValue: 1.0,  duration: 300, useNativeDriver: true }),
      ]).start();
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    }
  }, [peakDetected]);

  const greenOpacity = glowAnim.interpolate({
    inputRange: [0, 1],
    outputRange: [intensity * 0.12, intensity * 0.35],
  });

  return (
    <>
      {/* Green channel tint over face oval */}
      <Animated.View
        pointerEvents="none"
        style={[
          styles.greenOverlay,
          {
            left: OVAL_X, top: OVAL_Y,
            width: OVAL_W, height: OVAL_H,
            borderRadius: OVAL_W / 2,
            backgroundColor: `rgba(0, 220, 80, ${intensity * 0.08})`,
          },
        ]}
      />

      {/* ROI zone pulses */}
      <Animated.View
        pointerEvents="none"
        style={{ ...StyleSheet.absoluteFillObject, transform: [{ scale: beatScale }] }}
      >
        <Svg width={width} height={height} style={StyleSheet.absoluteFill}>
          <Defs>
            <RadialGradient id="pulseGrad" cx="50%" cy="50%" r="50%">
              <Stop offset="0%" stopColor="#00FF55" stopOpacity={0.4} />
              <Stop offset="100%" stopColor="#00FF55" stopOpacity={0} />
            </RadialGradient>
          </Defs>
          {ROI_ZONES.map(z => {
            const rx = OVAL_X + z.x * OVAL_W;
            const ry = OVAL_Y + z.y * OVAL_H;
            const rw = z.w * OVAL_W;
            const rh = z.h * OVAL_H;
            return (
              <Rect
                key={z.id}
                x={rx} y={ry} width={rw} height={rh}
                rx={rw / 2}
                fill={peakDetected ? "url(#pulseGrad)" : "transparent"}
                stroke="#00FF55"
                strokeWidth={peakDetected ? 1.5 : 0.5}
                strokeOpacity={peakDetected ? 0.6 : 0.15}
                fillOpacity={peakDetected ? 0.3 : 0}
              />
            );
          })}
        </Svg>
      </Animated.View>
    </>
  );
}

function GreenChannelStrip({ signal, playheadPct }: { signal: number[]; playheadPct: number }) {
  if (!signal || signal.length < 2) return null;

  const H = 60;
  const W = width - Spacing.lg * 2;
  const step = Math.max(1, Math.floor(signal.length / 200));
  const data = signal.filter((_, i) => i % step === 0);
  const min  = Math.min(...data);
  const max  = Math.max(...data);
  const range = max - min || 1;

  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1)) * W;
    const y = H - ((v - min) / range) * (H - 10) - 5;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });

  const playheadX = playheadPct * W;

  return (
    <View style={styles.signalStripContainer}>
      <Text style={styles.signalStripLabel}>rPPG Signal</Text>
      <Svg width={W} height={H}>
        <Path d={'M' + pts.join(' L')} stroke="#00FF55" strokeWidth={1.5}
          fill="none" strokeLinecap="round" opacity={0.8} />
        {/* Playhead */}
        <Rect x={playheadX - 1} y={0} width={2} height={H}
          fill={Colors.white} opacity={0.7} />
      </Svg>
    </View>
  );
}

export default function VideoPlaybackScreen() {
  const nav   = useNavigation<any>();
  const route = useRoute<Params>();
  const { videoUri, result } = route.params;

  const videoRef     = useRef<Video>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [position,  setPosition]  = useState(0); // ms
  const [duration,  setDuration]  = useState(result.duration_sec * 1000 || 30000);
  const [peakNow,   setPeakNow]   = useState(false);
  const [currentBPM, setCurrentBPM] = useState(0);
  const [pulseIntensity, setPulseIntensity] = useState(0.5);
  const [showGreenCh, setShowGreenCh] = useState(true);

  const ibi_ms   = result.ibi_ms ?? [];
  const signal   = result.pulse_signal ?? [];
  const fps      = result.fps ?? 30;
  const peaks    = result.peaks_idx ?? [];
  const peakTimesMs = peaks.map(p => (p / fps) * 1000);

  const progressAnim = useRef(new Animated.Value(0)).current;
  const progressRef  = useRef(0);
  const lastPeakRef  = useRef(-1);

  // Detect heartbeat peaks in real-time during playback
  useEffect(() => {
    if (!isPlaying) return;
    const positionSec = position / 1000;

    // Find which peak we're near
    const nearPeak = peakTimesMs.findIndex(
      t => Math.abs(t - position) < 120 // within 120ms
    );

    if (nearPeak >= 0 && nearPeak !== lastPeakRef.current) {
      lastPeakRef.current = nearPeak;
      setPeakNow(true);
      setTimeout(() => setPeakNow(false), 250);

      // Compute local BPM from surrounding IBIs
      const localIBIs = ibi_ms.slice(Math.max(0, nearPeak - 4), nearPeak + 1);
      if (localIBIs.length > 0) {
        const avgIBI = localIBIs.reduce((a, b) => a + b, 0) / localIBIs.length;
        setCurrentBPM(Math.round(60000 / avgIBI));
      }
    }

    // Signal intensity from pulse waveform
    const frameIdx = Math.floor((position / 1000) * fps);
    if (signal[frameIdx] !== undefined) {
      const v = signal[frameIdx];
      const norm = (v + 1) / 2; // pulse is normalized to [-1,1]
      setPulseIntensity(0.3 + norm * 0.7);
    }
  }, [position, isPlaying]);

  const onPlaybackStatusUpdate = (status: AVPlaybackStatus) => {
    if (!status.isLoaded) return;
    setIsPlaying(status.isPlaying);
    const pos = status.positionMillis ?? 0;
    setPosition(pos);
    const dur = status.durationMillis ?? 30000;
    setDuration(dur);
    progressRef.current = pos / dur;
    Animated.timing(progressAnim, {
      toValue: pos / dur, duration: 200, useNativeDriver: false,
    }).start();
  };

  const togglePlay = async () => {
    if (!videoRef.current) return;
    await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    if (isPlaying) {
      await videoRef.current.pauseAsync();
    } else {
      await videoRef.current.playAsync();
    }
  };

  const seek = async (pct: number) => {
    if (!videoRef.current) return;
    const ms = pct * duration;
    await videoRef.current.setPositionAsync(ms);
  };

  const progressWidth = progressAnim.interpolate({
    inputRange: [0, 1], outputRange: ['0%', '100%'],
  });

  const formatTime = (ms: number) => {
    const s = Math.floor(ms / 1000);
    return `${Math.floor(s / 60).toString().padStart(2, '0')}:${(s % 60).toString().padStart(2, '0')}`;
  };

  return (
    <View style={styles.root}>

      {/* ── VIDEO ── */}
      <Video
        ref={videoRef}
        source={{ uri: videoUri }}
        style={styles.video}
        resizeMode={ResizeMode.COVER}
        isLooping
        onPlaybackStatusUpdate={onPlaybackStatusUpdate}
      />

      {/* ── Dark gradient overlay ── */}
      <View style={StyleSheet.absoluteFill} pointerEvents="none">
        <View style={styles.gradTop} />
        <View style={{ flex: 1 }} />
        <View style={styles.gradBottom} />
      </View>

      {/* ── PULSE VISUAL OVERLAY ── */}
      {showGreenCh && (
        <PulseOverlay
          pulsing={isPlaying}
          intensity={pulseIntensity}
          peakDetected={peakNow}
        />
      )}

      {/* ── ROI OVAL GUIDE ── */}
      <View
        pointerEvents="none"
        style={[styles.ovalBorder, {
          left: OVAL_X, top: OVAL_Y,
          width: OVAL_W, height: OVAL_H,
          borderRadius: OVAL_W / 2,
          borderColor: showGreenCh ? 'rgba(0,255,80,0.3)' : 'rgba(255,255,255,0.15)',
        }]}
      />

      {/* ── ROI ZONE LABELS ── */}
      {showGreenCh && ROI_ZONES.map(z => (
        <View
          key={z.id}
          pointerEvents="none"
          style={{
            position: 'absolute',
            left: OVAL_X + z.x * OVAL_W + (z.w * OVAL_W) / 2 - 25,
            top: OVAL_Y + z.y * OVAL_H + (z.h * OVAL_H) / 2 - 8,
          }}
        >
          <Text style={styles.roiLabel}>{z.label}</Text>
        </View>
      ))}

      <SafeAreaView style={styles.safe}>

        {/* ── TOP BAR ── */}
        <View style={styles.topBar}>
          <TouchableOpacity style={styles.iconBtn} onPress={() => nav.goBack()}>
            <Text style={styles.iconBtnText}>←</Text>
          </TouchableOpacity>
          <Text style={styles.topTitle}>Pulse Replay</Text>
          <TouchableOpacity
            style={[styles.iconBtn, showGreenCh && styles.iconBtnActive]}
            onPress={() => setShowGreenCh(g => !g)}
          >
            <Text style={styles.iconBtnText}>🫀</Text>
          </TouchableOpacity>
        </View>

        <View style={{ flex: 1 }} />

        {/* ── LIVE BPM COUNTER ── */}
        {isPlaying && currentBPM > 0 && (
          <Animated.View style={styles.liveBPM}>
            <Text style={styles.liveBPMNum}>{currentBPM}</Text>
            <Text style={styles.liveBPMUnit}>BPM</Text>
            {peakNow && <View style={styles.peakFlash} />}
          </Animated.View>
        )}

        {/* ── SIGNAL STRIP ── */}
        <GreenChannelStrip
          signal={signal}
          playheadPct={progressRef.current}
        />

        {/* ── CONTROLS ── */}
        <View style={styles.controls}>

          {/* Progress bar */}
          <View style={styles.progressRow}>
            <Text style={styles.timeText}>{formatTime(position)}</Text>
            <TouchableOpacity
              style={styles.progressTrack}
              onPress={(e) => seek(e.nativeEvent.locationX / (width - 80 - Spacing.lg * 2))}
              activeOpacity={1}
            >
              <Animated.View style={[styles.progressFill, { width: progressWidth }]} />
            </TouchableOpacity>
            <Text style={styles.timeText}>{formatTime(duration)}</Text>
          </View>

          {/* Buttons row */}
          <View style={styles.btnRow}>
            <TouchableOpacity style={styles.ctrlBtn}
              onPress={() => videoRef.current?.setPositionAsync(0)}>
              <Text style={styles.ctrlIcon}>⏮</Text>
            </TouchableOpacity>

            <TouchableOpacity style={styles.playBtn} onPress={togglePlay}>
              <Text style={styles.playIcon}>{isPlaying ? '⏸' : '▶'}</Text>
            </TouchableOpacity>

            <TouchableOpacity style={styles.ctrlBtn}
              onPress={() => videoRef.current?.setPositionAsync(duration)}>
              <Text style={styles.ctrlIcon}>⏭</Text>
            </TouchableOpacity>
          </View>

          {/* Info strip */}
          <View style={styles.infoStrip}>
            <Text style={styles.infoText}>
              {peaks.length} peaks detected
            </Text>
            <Text style={styles.infoText}>·</Text>
            <Text style={styles.infoText}>
              {showGreenCh ? 'Green channel active' : 'Overlay hidden'}
            </Text>
            <Text style={styles.infoText}>·</Text>
            <Text style={styles.infoText}>
              {result.method_used?.replace('_', '+') ?? 'POS'}
            </Text>
          </View>
        </View>

      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#000' },
  safe: { ...StyleSheet.absoluteFillObject },
  video: { ...StyleSheet.absoluteFillObject },

  gradTop: {
    height: 160,
    backgroundColor: 'rgba(0,0,0,0.65)',
  },
  gradBottom: {
    height: 280,
    backgroundColor: 'rgba(0,0,0,0.80)',
  },

  greenOverlay: { position: 'absolute' },
  ovalBorder: {
    position: 'absolute',
    borderWidth: 1, borderStyle: 'dashed',
    backgroundColor: 'transparent',
  },
  roiLabel: {
    fontFamily: 'SpaceGrotesk-Regular',
    fontSize: 9, color: 'rgba(0,255,80,0.6)',
    letterSpacing: 0.5,
  },

  topBar: {
    flexDirection: 'row', alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: Spacing.lg,
    paddingTop: Spacing.md,
  },
  iconBtn: {
    width: 40, height: 40, borderRadius: 20,
    backgroundColor: 'rgba(255,255,255,0.1)',
    alignItems: 'center', justifyContent: 'center',
  },
  iconBtnActive: { backgroundColor: 'rgba(0,255,80,0.2)' },
  iconBtnText: { fontSize: 16, color: Colors.white },
  topTitle: {
    fontFamily: 'SpaceGrotesk-SemiBold',
    fontSize: 16, color: Colors.white,
    letterSpacing: -0.3,
  },

  liveBPM: {
    position: 'absolute',
    right: Spacing.lg, top: '35%',
    backgroundColor: 'rgba(0,0,0,0.6)',
    borderWidth: 1, borderColor: 'rgba(0,255,80,0.4)',
    borderRadius: Radius.lg,
    padding: Spacing.md,
    alignItems: 'center', minWidth: 72,
  },
  liveBPMNum: {
    fontFamily: 'SpaceGrotesk-Bold',
    fontSize: 36, color: Colors.white,
    letterSpacing: -2,
  },
  liveBPMUnit: {
    ...Typography.label, color: 'rgba(0,255,80,0.8)',
    marginTop: -4,
  },
  peakFlash: {
    position: 'absolute',
    inset: 0, borderRadius: Radius.lg,
    backgroundColor: 'rgba(0,255,80,0.2)',
  },

  // Signal strip
  signalStripContainer: {
    marginHorizontal: Spacing.lg,
    marginBottom: 8,
    backgroundColor: 'rgba(0,0,0,0.6)',
    borderWidth: 1, borderColor: 'rgba(0,255,80,0.2)',
    borderRadius: Radius.sm,
    padding: 8,
  },
  signalStripLabel: {
    ...Typography.label, color: 'rgba(0,255,80,0.5)',
    marginBottom: 4,
  },

  // Controls
  controls: {
    paddingHorizontal: Spacing.lg,
    paddingBottom: Spacing.lg,
    backgroundColor: 'rgba(0,0,0,0.7)',
    borderTopWidth: 1, borderTopColor: 'rgba(255,255,255,0.08)',
    paddingTop: Spacing.md,
  },

  progressRow: {
    flexDirection: 'row', alignItems: 'center',
    gap: 10, marginBottom: Spacing.md,
  },
  timeText: { ...Typography.mono, fontSize: 11 },
  progressTrack: {
    flex: 1, height: 3,
    backgroundColor: Colors.border,
    borderRadius: 2, overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    backgroundColor: 'rgba(0,255,80,0.8)',
    borderRadius: 2,
  },

  btnRow: {
    flexDirection: 'row', justifyContent: 'center',
    alignItems: 'center', gap: Spacing.lg,
    marginBottom: Spacing.md,
  },
  ctrlBtn: {
    width: 44, height: 44, borderRadius: 22,
    backgroundColor: 'rgba(255,255,255,0.08)',
    alignItems: 'center', justifyContent: 'center',
  },
  ctrlIcon: { color: Colors.textSecondary, fontSize: 18 },
  playBtn: {
    width: 60, height: 60, borderRadius: 30,
    backgroundColor: Colors.white,
    alignItems: 'center', justifyContent: 'center',
  },
  playIcon: { color: Colors.black, fontSize: 22 },

  infoStrip: {
    flexDirection: 'row', justifyContent: 'center',
    gap: 8, flexWrap: 'wrap',
  },
  infoText: { ...Typography.label, color: Colors.textMuted },
});

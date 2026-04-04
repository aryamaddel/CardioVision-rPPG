// src/screens/ResultsScreen.tsx
import React, { useEffect, useRef, useState } from 'react';
import {
  View, Text, ScrollView, TouchableOpacity,
  StyleSheet, Animated, Dimensions,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useNavigation, useRoute, RouteProp } from '@react-navigation/native';
import Svg, { Path, Circle, Line, G, Rect, Text as SvgText } from 'react-native-svg';
import * as Haptics from 'expo-haptics';
import { Colors, Typography, Spacing, Radius, HealthTipsData } from '../theme';
import type { RootStackParamList } from '../../App';
import type { RPPGResult } from '../api/rppgService';

type Params = RouteProp<RootStackParamList, 'Results'>;
const { width } = Dimensions.get('window');
const CHART_W = width - Spacing.lg * 2;
const CHART_H = 120;

// ── Waveform chart ────────────────────────────────────────────────────────────
function WaveformChart({ signal, peaks }: { signal: number[]; peaks: number[] }) {
  if (!signal || signal.length === 0) return null;

  const MAX_POINTS = 300;
  const step = Math.max(1, Math.floor(signal.length / MAX_POINTS));
  const data = signal.filter((_, i) => i % step === 0);
  const min  = Math.min(...data);
  const max  = Math.max(...data);
  const range = max - min || 1;

  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1)) * CHART_W;
    const y = CHART_H - ((v - min) / range) * (CHART_H - 20) - 10;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });

  const pathD = 'M' + pts.join(' L');

  // Mark peaks on downsampled scale
  const peakMarkers = peaks
    .filter(p => p % step === 0)
    .map(p => Math.floor(p / step))
    .filter(p => p < data.length)
    .map(p => {
      const x = (p / (data.length - 1)) * CHART_W;
      const y = CHART_H - ((data[p] - min) / range) * (CHART_H - 20) - 10;
      return { x, y };
    });

  return (
    <View style={styles.chartContainer}>
      <Svg width={CHART_W} height={CHART_H} viewBox={`0 0 ${CHART_W} ${CHART_H}`}>
        {/* Grid lines */}
        {[0.25, 0.5, 0.75].map(frac => (
          <Line key={frac}
            x1={0} y1={CHART_H * (1 - frac)}
            x2={CHART_W} y2={CHART_H * (1 - frac)}
            stroke={Colors.border} strokeWidth={1}
          />
        ))}
        {/* Fill area */}
        <Path
          d={pathD + ` L${CHART_W},${CHART_H} L0,${CHART_H} Z`}
          fill="rgba(255,255,255,0.04)"
        />
        {/* Line */}
        <Path d={pathD} stroke={Colors.white} strokeWidth={1.5}
          fill="none" strokeLinecap="round" strokeLinejoin="round" />
        {/* Peak markers */}
        {peakMarkers.map(({ x, y }, i) => (
          <Circle key={i} cx={x} cy={y} r={3}
            fill={Colors.white} opacity={0.7} />
        ))}
      </Svg>
    </View>
  );
}

// ── IBI variability chart ─────────────────────────────────────────────────────
function IBIChart({ ibi }: { ibi: number[] }) {
  if (!ibi || ibi.length < 2) return null;

  const min  = Math.min(...ibi);
  const max  = Math.max(...ibi);
  const range = max - min || 1;
  const H = 80;
  const barW = Math.max(4, (CHART_W / ibi.length) - 2);

  return (
    <View style={styles.chartContainer}>
      <Svg width={CHART_W} height={H}>
        {ibi.map((v, i) => {
          const x = i * (CHART_W / ibi.length);
          const barH = ((v - min) / range) * (H - 16) + 8;
          const y = H - barH;
          const intensity = (v - min) / range;
          return (
            <Rect key={i} x={x + 1} y={y} width={barW} height={barH}
              rx={2} fill={Colors.white}
              opacity={0.25 + intensity * 0.55}
            />
          );
        })}
      </Svg>
    </View>
  );
}

// ── Confidence arc ────────────────────────────────────────────────────────────
function ConfidenceArc({ score }: { score: number }) {
  const R = 44, cx = 52, cy = 52;
  const angle = score * 270; // 270deg sweep
  const startAngle = -225 * (Math.PI / 180);
  const endAngle   = startAngle + (angle * Math.PI) / 180;
  const x1 = cx + R * Math.cos(startAngle), y1 = cy + R * Math.sin(startAngle);
  const x2 = cx + R * Math.cos(endAngle),   y2 = cy + R * Math.sin(endAngle);
  const largeArc = angle > 180 ? 1 : 0;
  const trackEnd  = cx + R * Math.cos(-225 * (Math.PI/180) + 270 * Math.PI/180);
  const trackEndY = cy + R * Math.sin(-225 * (Math.PI/180) + 270 * Math.PI/180);

  return (
    <View style={styles.arcContainer}>
      <Svg width={104} height={104}>
        {/* Track */}
        <Path
          d={`M ${x1.toFixed(1)} ${y1.toFixed(1)} A ${R} ${R} 0 1 1 ${trackEnd.toFixed(1)} ${trackEndY.toFixed(1)}`}
          stroke={Colors.border} strokeWidth={6} fill="none" strokeLinecap="round"
        />
        {/* Fill */}
        {score > 0.01 && (
          <Path
            d={`M ${x1.toFixed(1)} ${y1.toFixed(1)} A ${R} ${R} 0 ${largeArc} 1 ${x2.toFixed(1)} ${y2.toFixed(1)}`}
            stroke={Colors.white} strokeWidth={6} fill="none" strokeLinecap="round"
          />
        )}
      </Svg>
      <View style={styles.arcInner}>
        <Text style={styles.arcScore}>{Math.round(score * 100)}</Text>
        <Text style={styles.arcLabel}>%</Text>
      </View>
    </View>
  );
}

// ── Large metric display ──────────────────────────────────────────────────────
function VitalCard({ label, value, unit, sub, animDelay = 0 }: {
  label: string; value: string; unit: string; sub?: string; animDelay?: number;
}) {
  const countAnim = useRef(new Animated.Value(0)).current;
  const fadeAnim  = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(fadeAnim, { toValue: 1, duration: 600, delay: animDelay, useNativeDriver: true }),
    ]).start();
  }, []);

  return (
    <Animated.View style={[styles.vitalCard, { opacity: fadeAnim }]}>
      <Text style={styles.vitalLabel}>{label}</Text>
      <View style={styles.vitalValueRow}>
        <Text style={styles.vitalValue}>{value}</Text>
        <Text style={styles.vitalUnit}>{unit}</Text>
      </View>
      {sub && <Text style={styles.vitalSub}>{sub}</Text>}
    </Animated.View>
  );
}

// ── Triage mode badge ─────────────────────────────────────────────────────────
function TriageBadge({ mode, reason }: { mode: string; reason: string }) {
  const isBiometric = mode === 'BIOMETRIC';
  const pulseAnim   = useRef(new Animated.Value(1)).current;

  useEffect(() => {
    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, { toValue: 1.05, duration: 800, useNativeDriver: true }),
        Animated.timing(pulseAnim, { toValue: 1.0,  duration: 800, useNativeDriver: true }),
      ]),
    );
    loop.start();
    return () => loop.stop();
  }, []);

  return (
    <Animated.View style={[styles.triageBadge, { transform: [{ scale: pulseAnim }] }]}>
      <View style={styles.triageDot} />
      <View style={{ flex: 1 }}>
        <Text style={styles.triageMode}>
          {isBiometric ? '🫀 BIOMETRIC MODE' : '👁️ VISUAL ASSESSMENT MODE'}
        </Text>
        <Text style={styles.triageReason} numberOfLines={2}>{reason}</Text>
      </View>
    </Animated.View>
  );
}

// ── Health tip card ───────────────────────────────────────────────────────────
function TipCard({ tip, index }: { tip: any; index: number }) {
  const [expanded, setExpanded] = useState(false);
  const slideAnim = useRef(new Animated.Value(0)).current;
  const fadeAnim  = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.timing(fadeAnim, {
      toValue: 1, duration: 400, delay: index * 100, useNativeDriver: true,
    }).start();
  }, []);

  const toggle = () => {
    Haptics.selectionAsync();
    setExpanded(e => !e);
  };

  const urgencyColor = tip.urgency === 'high' ? Colors.white
    : tip.urgency === 'med' ? Colors.fog
    : Colors.smoke;

  return (
    <Animated.View style={[styles.tipCard, { opacity: fadeAnim }]}>
      <TouchableOpacity onPress={toggle} activeOpacity={0.7}>
        <View style={styles.tipHeader}>
          <Text style={styles.tipIcon}>{tip.icon}</Text>
          <View style={{ flex: 1 }}>
            <Text style={[styles.tipTitle, { color: urgencyColor }]}>{tip.title}</Text>
            <Text style={styles.tipSubtitle}>{tip.subtitle}</Text>
          </View>
          <View style={styles.tipMeta}>
            <Text style={styles.tipDuration}>{tip.duration}</Text>
            <Text style={styles.tipChevron}>{expanded ? '∧' : '∨'}</Text>
          </View>
        </View>
        {expanded && (
          <Text style={styles.tipDetail}>{tip.detail}</Text>
        )}
      </TouchableOpacity>
    </Animated.View>
  );
}

// ── Main Results Screen ───────────────────────────────────────────────────────
export default function ResultsScreen() {
  const nav   = useNavigation<any>();
  const route = useRoute<Params>();
  const { result, videoUri } = route.params;

  const hrv   = result.hrv_features ?? {};
  const bpm   = result.bpm ?? (result.ibi_ms?.length ? 60000 / (result.ibi_ms.reduce((a,b)=>a+b,0)/result.ibi_ms.length) : 0);
  const ibi   = result.ibi_ms ?? [];
  const conf  = result.confidence ?? 0;

  const scrollAlpha = useRef(new Animated.Value(0)).current;
  const headerY     = useRef(new Animated.Value(-20)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(scrollAlpha, { toValue: 1, duration: 500, useNativeDriver: true }),
      Animated.spring(headerY, { toValue: 0, friction: 8, useNativeDriver: true }),
    ]).start();
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
  }, []);

  // Pick tips based on stress level + BPM
  const stress = hrv.stress_level ?? 'Low';
  const tips = [
    ...(stress === 'High'   ? HealthTipsData.highStress : []),
    ...(stress === 'Medium' ? HealthTipsData.medStress  : []),
    ...(stress === 'Low'    ? HealthTipsData.lowStress  : []),
    ...(bpm > 90            ? HealthTipsData.highBPM    : []),
    ...HealthTipsData.general,
  ].slice(0, 5);

  const stressColor = stress === 'High' ? Colors.ash
    : stress === 'Medium' ? Colors.silver
    : Colors.mist;

  const methodLabel = (result.method_used ?? 'pos').replace('_', ' ').toUpperCase();

  return (
    <View style={styles.root}>
      <SafeAreaView style={styles.safe} edges={['top']}>

        {/* ── TOP BAR ── */}
        <Animated.View style={[styles.topBar, { opacity: scrollAlpha, transform: [{ translateY: headerY }] }]}>
          <TouchableOpacity style={styles.backBtn} onPress={() => nav.navigate('Home')}>
            <Text style={styles.backText}>↩ Home</Text>
          </TouchableOpacity>
          <Text style={styles.topTitle}>Biometric Report</Text>
          <TouchableOpacity style={styles.videoBtn}
            onPress={() => nav.navigate('VideoPlayback', { videoUri, result })}>
            <Text style={styles.videoBtnText}>▶ Video</Text>
          </TouchableOpacity>
        </Animated.View>

        <ScrollView
          style={{ flex: 1 }}
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
        >

          {/* ── TRIAGE MODE ── */}
          <TriageBadge
            mode={result.triage_mode ?? 'BIOMETRIC'}
            reason={result.triage_reason ?? ''}
          />

          {/* ── HERO BPM ── */}
          {result.is_reliable ? (
            <View style={styles.heroBPM}>
              <Text style={styles.heroBPMLabel}>Heart Rate</Text>
              <View style={styles.heroBPMRow}>
                <Text style={styles.heroBPMValue}>{Math.round(bpm)}</Text>
                <View style={styles.heroBPMRight}>
                  <Text style={styles.heroBPMUnit}>BPM</Text>
                  <Text style={styles.heroBPMRange}>
                    {bpm < 60 ? 'Below resting' : bpm < 100 ? 'Normal range' : 'Elevated'}
                  </Text>
                </View>
              </View>
            </View>
          ) : (
            <View style={styles.unreliableCard}>
              <Text style={styles.unreliableIcon}>⚠</Text>
              <Text style={styles.unreliableTitle}>Signal Unreliable</Text>
              <Text style={styles.unreliableBody}>
                Confidence too low for clinical output. Visual assessment active.
              </Text>
            </View>
          )}

          {/* ── PULSE WAVEFORM ── */}
          <View style={styles.section}>
            <View style={styles.sectionHeader}>
              <Text style={styles.sectionTitle}>rPPG Pulse Waveform</Text>
              <Text style={styles.sectionBadge}>{methodLabel}</Text>
            </View>
            <WaveformChart signal={result.pulse_signal ?? []} peaks={result.peaks_idx ?? []} />
            <Text style={styles.chartCaption}>
              Green channel photoplethysmographic signal · {(result.duration_sec ?? 30).toFixed(0)}s recording
            </Text>
          </View>

          {/* ── IBI VARIABILITY ── */}
          <View style={styles.section}>
            <View style={styles.sectionHeader}>
              <Text style={styles.sectionTitle}>IBI Variability</Text>
              <Text style={styles.sectionBadge}>{ibi.length} beats</Text>
            </View>
            <IBIChart ibi={ibi} />
            <Text style={styles.chartCaption}>
              Inter-Beat Intervals · Beat-to-beat autonomic variation
            </Text>
          </View>

          {/* ── HRV METRICS GRID ── */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>HRV Metrics</Text>
            <View style={styles.metricGrid}>
              <VitalCard label="RMSSD"    value={(hrv.rmssd_ms ?? 0).toFixed(1)} unit="ms"  sub="Parasympathetic tone"  animDelay={100} />
              <VitalCard label="SDNN"     value={(hrv.sdnn_ms  ?? 0).toFixed(1)} unit="ms"  sub="Overall variability"   animDelay={200} />
              <VitalCard label="LF/HF"    value={(hrv.lf_hf_ratio ?? 0).toFixed(2)} unit="" sub="Autonomic balance"     animDelay={300} />
              <VitalCard label="Mean IBI" value={ibi.length ? (ibi.reduce((a,b)=>a+b,0)/ibi.length).toFixed(0) : '—'} unit="ms" sub="Avg inter-beat" animDelay={400} />
            </View>
          </View>

          {/* ── STRESS + CONFIDENCE ── */}
          <View style={[styles.section, styles.stressRow]}>

            {/* Stress */}
            <View style={[styles.stressCard, { flex: 1.4 }]}>
              <Text style={styles.sectionTitle}>Stress Level</Text>
              <Text style={[styles.stressValue, { color: stressColor }]}>{stress}</Text>
              <Text style={styles.stressSub}>
                Stress Index: {hrv.stress_index?.toFixed(0) ?? '—'}/100
              </Text>
              <View style={styles.stressBar}>
                <View style={[styles.stressFill, {
                  width: `${hrv.stress_index ?? 0}%`,
                  backgroundColor: stressColor,
                }]} />
              </View>
            </View>

            {/* Confidence */}
            <View style={[styles.stressCard, { flex: 1, alignItems: 'center' }]}>
              <Text style={styles.sectionTitle}>Confidence</Text>
              <ConfidenceArc score={conf} />
              <Text style={styles.confLabel}>
                {conf >= 0.7 ? 'High' : conf >= 0.45 ? 'Medium' : 'Low'}
              </Text>
            </View>
          </View>

          {/* ── SIGNAL QUALITY BREAKDOWN ── */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Quality Breakdown</Text>
            <View style={styles.qualGrid}>
              {[
                ['IBI Regularity',  result.confidence_details?.ibi_regularity ?? 0],
                ['Spectral SNR',    result.confidence_details?.snr ?? 0],
                ['Peak Density',    result.confidence_details?.density ?? 0],
                ['Data Completeness', result.confidence_details?.duration ?? 0],
              ].map(([label, val]) => (
                <View key={label as string} style={styles.qualRow}>
                  <Text style={styles.qualLabel}>{label as string}</Text>
                  <View style={styles.qualTrack}>
                    <View style={[styles.qualFill, { width: `${(val as number) * 100}%` }]} />
                  </View>
                  <Text style={styles.qualVal}>{Math.round((val as number) * 100)}%</Text>
                </View>
              ))}
            </View>
            <View style={styles.qualMetaRow}>
              <Text style={styles.qualMeta}>
                Method: <Text style={{ color: Colors.textSecondary }}>{methodLabel}</Text>
              </Text>
              {result.deep_model_used && result.deep_model_used !== 'none' && (
                <Text style={styles.qualMeta}>
                  Neural: <Text style={{ color: Colors.textSecondary }}>{result.deep_model_used}</Text>
                </Text>
              )}
            </View>
          </View>

          {/* ── HEALTH TIPS ── */}
          <View style={styles.section}>
            <View style={styles.sectionHeader}>
              <Text style={styles.sectionTitle}>Recommendations</Text>
              <Text style={styles.sectionBadge}>
                {stress} Stress
              </Text>
            </View>
            {tips.map((tip, i) => (
              <TipCard key={i} tip={tip} index={i} />
            ))}
          </View>

          {/* ── SCAN AGAIN ── */}
          <TouchableOpacity
            style={styles.rescanBtn}
            onPress={() => nav.navigate('Record')}
            activeOpacity={0.8}
          >
            <Text style={styles.rescanText}>Scan Again</Text>
          </TouchableOpacity>

          <Text style={styles.disclaimer}>
            For informational purposes only. Not a medical device.{'\n'}
            Consult a physician for clinical decisions.
          </Text>

        </ScrollView>
      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: Colors.background },
  safe: { flex: 1 },

  topBar: {
    flexDirection: 'row', alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: Spacing.lg,
    paddingTop: Spacing.sm, paddingBottom: Spacing.md,
    borderBottomWidth: 1, borderBottomColor: Colors.border,
  },
  backBtn: {},
  backText: { ...Typography.body, color: Colors.textSecondary },
  topTitle: { ...Typography.h3 },
  videoBtn: {
    backgroundColor: Colors.surfaceHigh,
    borderWidth: 1, borderColor: Colors.border,
    borderRadius: Radius.full,
    paddingHorizontal: 12, paddingVertical: 5,
  },
  videoBtnText: { ...Typography.bodySmall, color: Colors.textSecondary },

  scrollContent: { paddingHorizontal: Spacing.lg, paddingBottom: 80 },

  // Triage badge
  triageBadge: {
    flexDirection: 'row', alignItems: 'center',
    backgroundColor: Colors.surface,
    borderWidth: 1, borderColor: Colors.border,
    borderRadius: Radius.md,
    padding: Spacing.md,
    marginTop: Spacing.md, marginBottom: Spacing.md,
    gap: 10,
  },
  triageDot: {
    width: 8, height: 8, borderRadius: 4,
    backgroundColor: Colors.white,
  },
  triageMode: {
    fontFamily: 'SpaceGrotesk-SemiBold',
    fontSize: 12, color: Colors.white,
    letterSpacing: 0.5,
  },
  triageReason: { ...Typography.bodySmall, color: Colors.textTertiary, marginTop: 2 },

  // Hero BPM
  heroBPM: {
    backgroundColor: Colors.surface,
    borderWidth: 1, borderColor: Colors.border,
    borderRadius: Radius.lg,
    padding: Spacing.lg,
    marginBottom: Spacing.md,
  },
  heroBPMLabel: { ...Typography.label, marginBottom: 6 },
  heroBPMRow: { flexDirection: 'row', alignItems: 'flex-end' },
  heroBPMValue: {
    fontFamily: 'SpaceGrotesk-Bold',
    fontSize: 80, letterSpacing: -4,
    color: Colors.white, lineHeight: 80,
  },
  heroBPMRight: { marginLeft: 12, marginBottom: 8 },
  heroBPMUnit: {
    fontFamily: 'SpaceGrotesk-Bold',
    fontSize: 22, color: Colors.textSecondary,
  },
  heroBPMRange: {
    ...Typography.bodySmall,
    color: Colors.textTertiary, marginTop: 4,
  },

  unreliableCard: {
    backgroundColor: Colors.surface,
    borderWidth: 1, borderColor: Colors.border,
    borderRadius: Radius.lg, padding: Spacing.lg,
    alignItems: 'center', marginBottom: Spacing.md,
  },
  unreliableIcon: { fontSize: 32, marginBottom: Spacing.sm },
  unreliableTitle: { ...Typography.h2, marginBottom: 6 },
  unreliableBody: { ...Typography.body, textAlign: 'center', color: Colors.textTertiary },

  // Section
  section: {
    backgroundColor: Colors.surface,
    borderWidth: 1, borderColor: Colors.border,
    borderRadius: Radius.lg,
    padding: Spacing.lg,
    marginBottom: Spacing.md,
  },
  sectionHeader: {
    flexDirection: 'row', justifyContent: 'space-between',
    alignItems: 'center', marginBottom: Spacing.md,
  },
  sectionTitle: { ...Typography.label, color: Colors.textSecondary },
  sectionBadge: {
    ...Typography.label,
    backgroundColor: Colors.surfaceHigh,
    borderWidth: 1, borderColor: Colors.border,
    borderRadius: Radius.full,
    paddingHorizontal: 8, paddingVertical: 3,
    color: Colors.textTertiary,
  },

  // Charts
  chartContainer: { overflow: 'hidden', borderRadius: Radius.sm },
  chartCaption: {
    ...Typography.label,
    color: Colors.textMuted,
    marginTop: 8, textAlign: 'center',
  },

  // Metric grid
  metricGrid: {
    flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginTop: 4,
  },
  vitalCard: {
    flex: 1, minWidth: '45%',
    backgroundColor: Colors.surfaceHigh,
    borderRadius: Radius.md,
    padding: Spacing.md,
    borderWidth: 1, borderColor: Colors.border,
  },
  vitalLabel: { ...Typography.label, marginBottom: 4 },
  vitalValueRow: { flexDirection: 'row', alignItems: 'baseline', gap: 4 },
  vitalValue: {
    fontFamily: 'SpaceGrotesk-Bold',
    fontSize: 28, letterSpacing: -1, color: Colors.white,
  },
  vitalUnit: {
    fontFamily: 'SpaceGrotesk-Regular',
    fontSize: 13, color: Colors.textSecondary,
  },
  vitalSub: {
    ...Typography.bodySmall,
    color: Colors.textMuted, marginTop: 3,
  },

  // Stress + confidence row
  stressRow: { flexDirection: 'row', gap: 8, padding: 0, backgroundColor: 'transparent', borderWidth: 0 },
  stressCard: {
    backgroundColor: Colors.surface,
    borderWidth: 1, borderColor: Colors.border,
    borderRadius: Radius.lg,
    padding: Spacing.lg,
  },
  stressValue: {
    fontFamily: 'SpaceGrotesk-Bold',
    fontSize: 36, letterSpacing: -1,
    marginTop: 8, marginBottom: 6,
  },
  stressSub: { ...Typography.bodySmall, color: Colors.textTertiary, marginBottom: 10 },
  stressBar: {
    height: 3, backgroundColor: Colors.border,
    borderRadius: 2, overflow: 'hidden',
  },
  stressFill: { height: '100%', borderRadius: 2 },

  // Confidence arc
  arcContainer: {
    width: 104, height: 104,
    alignItems: 'center', justifyContent: 'center',
    marginTop: 8,
  },
  arcInner: {
    position: 'absolute',
    alignItems: 'center', justifyContent: 'center',
    flexDirection: 'row', alignItems: 'baseline',
  },
  arcScore: {
    fontFamily: 'SpaceGrotesk-Bold',
    fontSize: 26, color: Colors.white, letterSpacing: -1,
  },
  arcLabel: {
    fontFamily: 'SpaceGrotesk-Regular',
    fontSize: 13, color: Colors.textSecondary,
    marginLeft: 1,
  },
  confLabel: { ...Typography.label, color: Colors.textTertiary, marginTop: 6 },

  // Quality breakdown
  qualGrid: { marginTop: 8, gap: 10 },
  qualRow: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  qualLabel: { ...Typography.bodySmall, color: Colors.textTertiary, width: 130 },
  qualTrack: {
    flex: 1, height: 3, backgroundColor: Colors.border,
    borderRadius: 2, overflow: 'hidden',
  },
  qualFill: { height: '100%', backgroundColor: Colors.white, borderRadius: 2 },
  qualVal: { ...Typography.mono, width: 36, textAlign: 'right' },
  qualMetaRow: {
    flexDirection: 'row', justifyContent: 'space-between',
    marginTop: 12, flexWrap: 'wrap', gap: 4,
  },
  qualMeta: { ...Typography.label, color: Colors.textMuted },

  // Health tips
  tipCard: {
    backgroundColor: Colors.surfaceHigh,
    borderWidth: 1, borderColor: Colors.border,
    borderRadius: Radius.md,
    padding: Spacing.md,
    marginTop: 8,
  },
  tipHeader: { flexDirection: 'row', alignItems: 'flex-start', gap: 12 },
  tipIcon: { fontSize: 22, marginTop: 1 },
  tipTitle: {
    fontFamily: 'SpaceGrotesk-SemiBold',
    fontSize: 14, color: Colors.white,
  },
  tipSubtitle: { ...Typography.bodySmall, color: Colors.textTertiary, marginTop: 2 },
  tipMeta: { alignItems: 'flex-end' },
  tipDuration: { ...Typography.label, color: Colors.textMuted },
  tipChevron: { color: Colors.textMuted, marginTop: 4, fontSize: 12 },
  tipDetail: {
    ...Typography.bodySmall, color: Colors.textTertiary,
    marginTop: 10, lineHeight: 18,
    borderTopWidth: 1, borderTopColor: Colors.border,
    paddingTop: 10,
  },

  // Rescan
  rescanBtn: {
    borderWidth: 1, borderColor: Colors.border,
    borderRadius: Radius.lg,
    padding: Spacing.md,
    alignItems: 'center',
    marginTop: Spacing.sm,
    marginBottom: Spacing.md,
  },
  rescanText: {
    fontFamily: 'SpaceGrotesk-SemiBold',
    fontSize: 16, color: Colors.textSecondary,
  },

  disclaimer: {
    ...Typography.label, color: Colors.textMuted,
    textAlign: 'center', lineHeight: 18,
    paddingBottom: Spacing.xl,
  },
});

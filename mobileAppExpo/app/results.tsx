// app/results.tsx — Full Dashboard ResultsScreen (real data only)
import React, { useEffect, useRef, useState } from 'react';
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  StyleSheet,
  Animated,
  Dimensions,
  Alert,
  ActivityIndicator,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter, useLocalSearchParams } from 'expo-router';
import Svg, { Path, Circle, Line, Rect } from 'react-native-svg';
import * as Haptics from 'expo-haptics';
import { Ionicons } from '@expo/vector-icons';
import { useTheme, Typography, Spacing, Radius, HealthTipsData } from '../theme';
import type { RPPGResult } from '../api/rppgService';
import { generateAndShareReport } from '../api/reportGenerator';
import {
  clearPendingScanResult,
  clearScanSession,
  getPendingScanResult,
  getScanSession,
  setScanSession,
} from '../state/scanSession';

const { width } = Dimensions.get('window');

// ── Waveform Chart ──
function WaveformChart({ signal, peaks, colors, accent }: { signal: number[]; peaks: number[]; colors: any; accent: any }) {
  if (!signal || signal.length === 0) return <Text style={[styles.noData, { color: colors.textMuted }]}>No waveform data</Text>;
  const CW = width - Spacing.lg * 4, CH = 120;
  const step = Math.max(1, Math.floor(signal.length / 250));
  const data = signal.filter((_, i) => i % step === 0);
  const min = Math.min(...data), max = Math.max(...data), range = max - min || 1;
  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1)) * CW;
    const y = CH - ((v - min) / range) * (CH - 20) - 10;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  const peakMarkers = peaks.filter(p => p % step === 0).map(p => Math.floor(p / step)).filter(p => p < data.length).map(p => ({
    x: (p / (data.length - 1)) * CW, y: CH - ((data[p] - min) / range) * (CH - 20) - 10,
  }));
  return (
    <View style={styles.chartBox}>
      <Svg width={CW} height={CH}>
        {[0.25, 0.5, 0.75].map(f => <Line key={f} x1={0} y1={CH * (1 - f)} x2={CW} y2={CH * (1 - f)} stroke={colors.border} strokeWidth={0.5} />)}
        <Path d={'M' + pts.join(' L') + ` L${CW},${CH} L0,${CH} Z`} fill={accent.ghost} />
        <Path d={'M' + pts.join(' L')} stroke={accent.primary} strokeWidth={1.5} fill="none" strokeLinecap="round" />
        {peakMarkers.map(({ x, y }, i) => <Circle key={i} cx={x} cy={y} r={3} fill={accent.primary} opacity={0.8} />)}
      </Svg>
    </View>
  );
}

// ── IBI Bar Chart ──
function IBIChart({ ibi, colors, accent }: { ibi: number[]; colors: any; accent: any }) {
  if (!ibi || ibi.length < 2) return <Text style={[styles.noData, { color: colors.textMuted }]}>No IBI data</Text>;
  const CW = width - Spacing.lg * 4, CH = 80;
  const min = Math.min(...ibi), max = Math.max(...ibi), range = max - min || 1;
  const barW = Math.max(3, (CW / ibi.length) - 2);
  return (
    <View style={styles.chartBox}>
      <Svg width={CW} height={CH}>
        {ibi.map((v, i) => {
          const x = i * (CW / ibi.length);
          const barH = ((v - min) / range) * (CH - 16) + 8;
          return <Rect key={i} x={x + 1} y={CH - barH} width={barW} height={barH} rx={2} fill={accent.primary} opacity={0.25 + ((v - min) / range) * 0.55} />;
        })}
      </Svg>
    </View>
  );
}

// ── Confidence Arc ──
function ConfidenceArc({ score, colors, accent }: { score: number; colors: any; accent: any }) {
  const R = 42, cx = 50, cy = 50;
  const angle = score * 270;
  const start = -225 * (Math.PI / 180);
  const end = start + (angle * Math.PI) / 180;
  const x1 = cx + R * Math.cos(start), y1 = cy + R * Math.sin(start);
  const x2 = cx + R * Math.cos(end), y2 = cy + R * Math.sin(end);
  const largeArc = angle > 180 ? 1 : 0;
  const tEnd = cx + R * Math.cos(-225 * Math.PI / 180 + 270 * Math.PI / 180);
  const tEndY = cy + R * Math.sin(-225 * Math.PI / 180 + 270 * Math.PI / 180);
  return (
    <View style={styles.arcContainer}>
      <Svg width={100} height={100}>
        <Path d={`M ${x1.toFixed(1)} ${y1.toFixed(1)} A ${R} ${R} 0 1 1 ${tEnd.toFixed(1)} ${tEndY.toFixed(1)}`} stroke={colors.border} strokeWidth={5} fill="none" strokeLinecap="round" />
        {score > 0.01 && <Path d={`M ${x1.toFixed(1)} ${y1.toFixed(1)} A ${R} ${R} 0 ${largeArc} 1 ${x2.toFixed(1)} ${y2.toFixed(1)}`} stroke={accent.primary} strokeWidth={5} fill="none" strokeLinecap="round" />}
      </Svg>
      <View style={styles.arcInner}>
        <Text style={[styles.arcScore, { color: colors.textPrimary }]}>{Math.round(score * 100)}</Text>
        <Text style={[styles.arcLabel, { color: colors.textTertiary }]}>%</Text>
      </View>
    </View>
  );
}

// ── Vital Metric Card ──
function VitalCard({ label, value, unit, sub, delay = 0, colors }: { label: string; value: string; unit: string; sub?: string; delay?: number; colors: any }) {
  const anim = useRef(new Animated.Value(0)).current;
  useEffect(() => { Animated.timing(anim, { toValue: 1, duration: 500, delay, useNativeDriver: true }).start(); }, [anim, delay]);
  return (
    <Animated.View style={[styles.vitalCard, { backgroundColor: colors.surfaceHigh, borderColor: colors.border, opacity: anim }]}>
      <Text style={[styles.vitalLabel, { color: colors.textTertiary }]}>{label}</Text>
      <View style={{ flexDirection: 'row', alignItems: 'baseline', gap: 3 }}>
        <Text style={[styles.vitalValue, { color: colors.textPrimary }]}>{value}</Text>
        {unit ? <Text style={[styles.vitalUnit, { color: colors.textTertiary }]}>{unit}</Text> : null}
      </View>
      {sub && <Text style={[styles.vitalSub, { color: colors.textMuted }]}>{sub}</Text>}
    </Animated.View>
  );
}

// ── Health Tip ──
function TipCard({ tip, index, colors, accent }: { tip: any; index: number; colors: any; accent: any }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <TouchableOpacity activeOpacity={0.7} onPress={() => { Haptics.selectionAsync(); setExpanded(e => !e); }}
      style={[styles.tipCard, { backgroundColor: colors.surfaceHigh, borderColor: colors.border }]}>
      <View style={styles.tipHeader}>
        <Ionicons name={tip.icon as any} size={20} color={accent.primary} style={{ marginRight: 12 }} />
        <View style={{ flex: 1 }}>
          <Text style={[styles.tipTitle, { color: colors.textPrimary }]}>{tip.title}</Text>
          <Text style={[styles.tipSubtitle, { color: colors.textTertiary }]}>{tip.subtitle}</Text>
        </View>
        <Text style={[styles.tipDuration, { color: colors.textMuted }]}>{tip.duration}</Text>
      </View>
      {expanded && <Text style={[styles.tipDetail, { color: colors.textSecondary, borderTopColor: colors.border }]}>{tip.detail}</Text>}
    </TouchableOpacity>
  );
}

function LoadingBlock({ width = '100%', height = 14, colors }: { width?: any; height?: number; colors: any }) {
  const alpha = useRef(new Animated.Value(0.45)).current;
  useEffect(() => {
    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(alpha, { toValue: 0.85, duration: 700, useNativeDriver: true }),
        Animated.timing(alpha, { toValue: 0.45, duration: 700, useNativeDriver: true }),
      ]),
    );
    loop.start();
    return () => loop.stop();
  }, [alpha]);

  return (
    <Animated.View
      style={{
        width,
        height,
        borderRadius: 8,
        backgroundColor: colors.border,
        opacity: alpha,
      }}
    />
  );
}

// ── Main Screen ──
export default function ResultsScreen() {
  const router = useRouter();
  const { colors, accent, isDark, toggle } = useTheme();
  const params = useLocalSearchParams();
  const session = getScanSession();

  const initialResult = (() => {
    try {
      if (typeof params.resultJson === 'string' && params.resultJson.length > 0) {
        return JSON.parse(params.resultJson as string) as RPPGResult;
      }
      return session.result;
    } catch {
      return session.result;
    }
  })();

  const [resolvedResult, setResolvedResult] = useState<RPPGResult | null>(initialResult ?? null);
  const [isLoadingResult, setIsLoadingResult] = useState<boolean>(!initialResult);

  useEffect(() => {
    if (resolvedResult) {
      setIsLoadingResult(false);
      return;
    }

    const pending = getPendingScanResult();
    if (!pending) {
      setIsLoadingResult(false);
      return;
    }

    let cancelled = false;
    setIsLoadingResult(true);
    pending
      .then((result) => {
        if (cancelled) return;
        setResolvedResult(result);
        setScanSession({ result, videoUri: session.videoUri });
        clearPendingScanResult();
      })
      .catch(() => {
        if (cancelled) return;
        clearPendingScanResult();
      })
      .finally(() => {
        if (cancelled) return;
        setIsLoadingResult(false);
      });

    return () => {
      cancelled = true;
    };
  }, [resolvedResult, session.videoUri]);

  const result = (resolvedResult ?? {}) as RPPGResult;
  const hrv = (result.hrv_features ?? {}) as any;
  const ibi = result.ibi_ms ?? [];
  const bpm = result.bpm ?? (ibi.length ? Math.round(60000 / (ibi.reduce((a: number, b: number) => a + b, 0) / ibi.length)) : null);
  const conf = result.confidence ?? 0;
  const isFailed = result.status === 'failed';
  const failReason = (result as any).reason as string | undefined;
  const failReasonText =
    failReason === 'identity_mismatch_detected'
      ? 'Session interrupted: a different person was detected. Restart and keep the same person in frame for the full scan.'
      : (failReason ?? 'Insufficient valid face frames. Keep face centered and well-lit.');
  const stressScore = hrv.stress_index !== undefined && hrv.stress_index !== null ? Math.round(hrv.stress_index) : null;
  const stress = stressScore !== null ? (stressScore >= 60 ? 'High' : stressScore >= 30 ? 'Medium' : 'Low') : 'Unknown';
  const sdnn = hrv.sdnn_ms ?? null;
  const rmssd = hrv.rmssd_ms ?? null;
  const lfhf = hrv.lf_hf_ratio ?? null;
  const meanIBI = ibi.length ? (ibi.reduce((a: number, b: number) => a + b, 0) / ibi.length) : null;
  const tremorScore = result.tremor_score !== undefined ? result.tremor_score : null;
  const tremorLabel = tremorScore === null ? '--' : tremorScore <= 15 ? 'Steady' : tremorScore <= 40 ? 'Mild' : 'High';

  const tips = [
    ...(stress === 'High' ? HealthTipsData.highStress : []),
    ...(stress === 'Medium' ? HealthTipsData.medStress : []),
    ...(stress === 'Low' ? HealthTipsData.lowStress : []),
    ...(bpm && bpm > 90 ? HealthTipsData.highBPM : []),
    ...HealthTipsData.general,
  ].slice(0, 5);

  useEffect(() => {
    if (resolvedResult) {
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    }
  }, [resolvedResult]);

  const methodLabel = isLoadingResult ? 'ANALYZING' : (result.method_used ?? 'pos').replace('_', '+').toUpperCase();
  const fmt = (v: number | null, d: number = 1) => v !== null && v !== undefined ? v.toFixed(d) : '--';

  const [isDownloading, setIsDownloading] = useState(false);

  const handleDownloadReport = async () => {
    if (!resolvedResult || isDownloading) return;
    setIsDownloading(true);
    try {
      await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
      await generateAndShareReport(resolvedResult);
    } catch (e: any) {
      Alert.alert('Report Error', e?.message ?? 'Failed to generate report. Please try again.');
    } finally {
      setIsDownloading(false);
    }
  };

  return (
    <View style={[styles.root, { backgroundColor: colors.background }]}>
      <SafeAreaView style={styles.safe} edges={['top']}>
        {/* Top Bar */}
        <View style={[styles.topBar, { borderBottomColor: colors.border }]}>
          <TouchableOpacity onPress={() => router.replace('/')} style={{ flexDirection: 'row', alignItems: 'center' }}>
            <Ionicons name="arrow-back" size={18} color={colors.textSecondary} />
            <Text style={[styles.backText, { color: colors.textSecondary }]}> Back</Text>
          </TouchableOpacity>
          <Text style={[styles.topTitle, { color: colors.textPrimary }]}>Results</Text>
          <TouchableOpacity onPress={toggle} style={[styles.toggleBtn, { backgroundColor: colors.surfaceHigh, borderColor: colors.border }]}>
            <Ionicons name={isDark ? 'sunny-outline' : 'moon-outline'} size={16} color={colors.textSecondary} />
          </TouchableOpacity>
        </View>

        <ScrollView style={{ flex: 1 }} contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>

          {isLoadingResult && (
            <View style={[styles.loadingCard, { backgroundColor: colors.card, borderColor: colors.cardBorder }]}> 
              <View style={styles.loadingHeaderRow}>
                <ActivityIndicator size="small" color={accent.primary} />
                <Text style={[styles.loadingTitle, { color: colors.textPrimary }]}>Finalizing Deep rPPG Analysis</Text>
              </View>
              <Text style={[styles.loadingBody, { color: colors.textTertiary }]}>POS metrics are ready. PhysFormer + fusion outputs are still being computed.</Text>
              <View style={{ gap: 10, marginTop: 12 }}>
                <LoadingBlock colors={colors} width="62%" height={14} />
                <LoadingBlock colors={colors} width="100%" height={10} />
                <LoadingBlock colors={colors} width="86%" height={10} />
                <LoadingBlock colors={colors} width="72%" height={10} />
              </View>
            </View>
          )}

          {isFailed && (
            <View style={[styles.failBanner, { backgroundColor: '#7F1D1D', borderColor: '#FCA5A5' }]}> 
              <Text style={styles.failTitle}>Signal Could Not Be Computed</Text>
              <Text style={styles.failBody}>{failReasonText}</Text>
            </View>
          )}

          {/* Hero BPM */}
          <View style={styles.heroBPM}>
            <Text style={[styles.heroBPMValue, { color: colors.textPrimary }]}>{bpm !== null ? Math.round(bpm) : '--'}</Text>
            <Text style={[styles.heroBPMUnit, { color: colors.textTertiary }]}>BPM</Text>
          </View>

          {/* HRV Section */}
          <View style={[styles.section, { backgroundColor: colors.card, borderColor: colors.cardBorder }]}>
            <View style={styles.sectionHeader}>
              <Text style={[styles.sectionTitle, { color: colors.textTertiary }]}>Heart Rate Variability</Text>
              <Text style={[styles.hrvVal, { color: colors.textPrimary }]}>{fmt(rmssd, 0)} <Text style={[styles.hrvU, { color: colors.textTertiary }]}>ms</Text></Text>
            </View>
            <IBIChart ibi={ibi} colors={colors} accent={accent} />
          </View>

          {/* Stress + Avg. Variability + Tremor */}
          <View style={styles.metricsRow}>
            <View style={[styles.metricBox, { backgroundColor: colors.card, borderColor: colors.cardBorder }]}>
              <Text style={[styles.metricBoxLabel, { color: colors.textTertiary }]}>Stress Score</Text>
              <Text style={[styles.metricBoxValue, { color: stress === 'High' ? '#EF4444' : stress === 'Medium' ? '#F59E0B' : colors.textPrimary }]}>{stressScore !== null ? stressScore : '--'} <Text style={[styles.metricBoxUnit, { color: colors.textTertiary }]}>/100</Text></Text>
            </View>
            <View style={[styles.metricBox, { backgroundColor: colors.card, borderColor: colors.cardBorder }]}>
              <Text style={[styles.metricBoxLabel, { color: colors.textTertiary }]}>Avg. Variability</Text>
              <Text style={[styles.metricBoxValue, { color: colors.textPrimary }]}>{fmt(sdnn, 0)} <Text style={[styles.metricBoxUnit, { color: colors.textTertiary }]}>ms</Text></Text>
            </View>
            <View style={[styles.metricBox, { backgroundColor: colors.card, borderColor: colors.cardBorder }]}>
              <Text style={[styles.metricBoxLabel, { color: colors.textTertiary }]}>Hand Tremor</Text>
              <Text style={[styles.metricBoxValue, { color: tremorScore !== null && tremorScore > 40 ? '#EF4444' : tremorScore !== null && tremorScore > 15 ? '#F59E0B' : colors.textPrimary }]}>{tremorScore !== null ? tremorScore : '--'} <Text style={[styles.metricBoxUnit, { color: colors.textTertiary }]}>/100</Text></Text>
            </View>
          </View>

          {/* rPPG Pulse Waveform */}
          <View style={[styles.section, { backgroundColor: colors.card, borderColor: colors.cardBorder }]}>
            <View style={styles.sectionHeader}>
              <Text style={[styles.sectionTitle, { color: colors.textTertiary }]}>rPPG Pulse Waveform</Text>
              <View style={[styles.badge, { backgroundColor: accent.ghost }]}><Text style={[styles.badgeText, { color: accent.primary }]}>{methodLabel}</Text></View>
            </View>
            <WaveformChart signal={result.pulse_signal ?? []} peaks={result.peaks_idx ?? []} colors={colors} accent={accent} />
          </View>

          {/* HRV Metrics Grid */}
          <View style={[styles.section, { backgroundColor: colors.card, borderColor: colors.cardBorder }]}>
            <Text style={[styles.sectionTitle, { color: colors.textTertiary }]}>HRV Metrics</Text>
            <View style={styles.metricGrid}>
              <VitalCard label="RMSSD" value={fmt(rmssd)} unit="ms" sub="Parasympathetic tone" delay={100} colors={colors} />
              <VitalCard label="SDNN" value={fmt(sdnn)} unit="ms" sub="Overall variability" delay={200} colors={colors} />
              <VitalCard label="LF/HF" value={fmt(lfhf, 2)} unit="" sub="Autonomic balance" delay={300} colors={colors} />
              <VitalCard label="Mean IBI" value={meanIBI !== null ? meanIBI.toFixed(0) : '--'} unit="ms" sub="Avg. inter-beat" delay={400} colors={colors} />
              <VitalCard label="Stability" value={tremorLabel} unit="" sub={tremorScore !== null ? `Tremor: ${tremorScore}/100` : 'No data'} delay={500} colors={colors} />
            </View>
          </View>

          {/* Confidence + Stress Index */}
          <View style={styles.stressConfRow}>
            <View style={[styles.stressConfCard, { backgroundColor: colors.card, borderColor: colors.cardBorder, flex: 1.4 }]}>
              <Text style={[styles.sectionTitle, { color: colors.textTertiary }]}>Stress Score</Text>
              <Text style={[styles.stressValue, { color: stress === 'High' ? '#EF4444' : stress === 'Low' ? '#22C55E' : colors.textPrimary }]}>{stressScore !== null ? stressScore : '--'}</Text>
              <Text style={[styles.stressSub, { color: colors.textTertiary }]}>Level: {stress}</Text>
              <View style={[styles.stressBar, { backgroundColor: colors.border }]}>
                <View style={[styles.stressFill, { width: `${hrv.stress_index ?? 0}%`, backgroundColor: accent.primary }]} />
              </View>
            </View>
            <View style={[styles.stressConfCard, { backgroundColor: colors.card, borderColor: colors.cardBorder, flex: 1, alignItems: 'center' }]}>
              <Text style={[styles.sectionTitle, { color: colors.textTertiary }]}>Confidence</Text>
              <ConfidenceArc score={conf} colors={colors} accent={accent} />
              <Text style={[styles.confLabel, { color: colors.textTertiary }]}>{conf >= 0.7 ? 'High' : conf >= 0.45 ? 'Medium' : 'Low'}</Text>
            </View>
          </View>

          {/* Quality Breakdown */}
          <View style={[styles.section, { backgroundColor: colors.card, borderColor: colors.cardBorder }]}>
            <Text style={[styles.sectionTitle, { color: colors.textTertiary }]}>Quality Breakdown</Text>
            <View style={styles.qualGrid}>
              {[
                ['IBI Regularity', result.confidence_details?.ibi_regularity],
                ['Spectral SNR', result.confidence_details?.snr],
                ['Peak Density', result.confidence_details?.density],
                ['Data Completeness', result.confidence_details?.duration],
              ].map(([label, val]) => (
                <View key={label as string} style={styles.qualRow}>
                  <Text style={[styles.qualLabel, { color: colors.textTertiary }]}>{label as string}</Text>
                  <View style={[styles.qualTrack, { backgroundColor: colors.border }]}>
                    <View style={[styles.qualFill, { width: `${((val as number) ?? 0) * 100}%`, backgroundColor: accent.primary }]} />
                  </View>
                  <Text style={[styles.qualVal, { color: colors.textSecondary }]}>{val !== undefined && val !== null ? Math.round((val as number) * 100) + '%' : '--'}</Text>
                </View>
              ))}
            </View>
          </View>

          {/* Recommendations */}
          <View style={[styles.section, { backgroundColor: colors.card, borderColor: colors.cardBorder }]}>
            <View style={styles.sectionHeader}>
              <Text style={[styles.sectionTitle, { color: colors.textTertiary }]}>Recommendations</Text>
              <Ionicons name="bulb-outline" size={16} color={accent.primary} />
            </View>
            {tips.map((tip, i) => <TipCard key={i} tip={tip} index={i} colors={colors} accent={accent} />)}
          </View>

          {/* Download Report */}
          <TouchableOpacity
            style={[styles.downloadBtn, { backgroundColor: accent.primary, opacity: (!resolvedResult || isDownloading) ? 0.55 : 1 }]}
            activeOpacity={0.85}
            onPress={handleDownloadReport}
            disabled={!resolvedResult || isDownloading}
          >
            {isDownloading ? (
              <ActivityIndicator size="small" color="#FFF" style={{ marginRight: 10 }} />
            ) : (
              <Ionicons name="download-outline" size={20} color="#FFF" style={{ marginRight: 10 }} />
            )}
            <Text style={styles.downloadText}>{isDownloading ? 'Generating...' : 'Download Report'}</Text>
          </TouchableOpacity>

          {/* Scan Again */}
          <TouchableOpacity
            style={[styles.rescanBtn, { borderColor: accent.primary }]}
            onPress={() => {
              clearScanSession();
              router.push('/record');
            }}
            activeOpacity={0.8}
          >
            <Ionicons name="refresh-outline" size={18} color={accent.primary} style={{ marginRight: 8 }} />
            <Text style={[styles.rescanText, { color: accent.primary }]}>Scan Again</Text>
          </TouchableOpacity>

          <Text style={[styles.disclaimer, { color: colors.textMuted }]}>
            For informational purposes only. Not a medical device.{'\n'}Consult a physician for clinical decisions.
          </Text>

        </ScrollView>
      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1 },
  safe: { flex: 1 },
  topBar: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: Spacing.lg, paddingTop: Spacing.sm, paddingBottom: Spacing.md, borderBottomWidth: 1 },
  backText: { ...Typography.body },
  topTitle: { fontFamily: 'SpaceGrotesk-SemiBold', fontSize: 18 },
  toggleBtn: { width: 34, height: 34, borderRadius: 17, borderWidth: 1, alignItems: 'center', justifyContent: 'center' },
  scrollContent: { paddingHorizontal: Spacing.lg, paddingBottom: 80 },
  loadingCard: { borderWidth: 1, borderRadius: Radius.lg, padding: Spacing.lg, marginTop: Spacing.md, marginBottom: Spacing.md },
  loadingHeaderRow: { flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: 8 },
  loadingTitle: { fontFamily: 'SpaceGrotesk-SemiBold', fontSize: 15 },
  loadingBody: { ...Typography.bodySmall, lineHeight: 18 },

  // Hero
  heroBPM: { flexDirection: 'row', alignItems: 'baseline', marginTop: Spacing.xl, marginBottom: Spacing.lg },
  heroBPMValue: { fontFamily: 'SpaceGrotesk-Bold', fontSize: 80, letterSpacing: -4, lineHeight: 80 },
  heroBPMUnit: { fontFamily: 'SpaceGrotesk-SemiBold', fontSize: 22, marginLeft: 8, marginBottom: 8 },

  // Sections
  section: { borderWidth: 1, borderRadius: Radius.lg, padding: Spacing.lg, marginBottom: Spacing.md },
  sectionHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: Spacing.md },
  sectionTitle: { ...Typography.label },
  badge: { borderRadius: Radius.full, paddingHorizontal: 8, paddingVertical: 3 },
  badgeText: { fontFamily: 'SpaceGrotesk-Medium', fontSize: 10, letterSpacing: 0.5 },
  hrvVal: { fontFamily: 'SpaceGrotesk-Bold', fontSize: 20 },
  hrvU: { fontFamily: 'SpaceGrotesk-Regular', fontSize: 14 },
  chartBox: { overflow: 'hidden', borderRadius: Radius.sm },
  noData: { ...Typography.bodySmall, textAlign: 'center', paddingVertical: 20 },

  // Metrics row
  metricsRow: { flexDirection: 'row', gap: 10, marginBottom: Spacing.md },
  metricBox: { flex: 1, borderWidth: 1, borderRadius: Radius.lg, padding: Spacing.md },
  metricBoxLabel: { ...Typography.label, marginBottom: 6 },
  metricBoxValue: { fontFamily: 'SpaceGrotesk-Bold', fontSize: 26, letterSpacing: -1 },
  metricBoxUnit: { fontFamily: 'SpaceGrotesk-Regular', fontSize: 14 },

  // Metric grid
  metricGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginTop: 4 },
  vitalCard: { flex: 1, minWidth: '45%', borderRadius: Radius.md, padding: Spacing.md, borderWidth: 1 },
  vitalLabel: { ...Typography.label, marginBottom: 4 },
  vitalValue: { fontFamily: 'SpaceGrotesk-Bold', fontSize: 26, letterSpacing: -1 },
  vitalUnit: { fontFamily: 'SpaceGrotesk-Regular', fontSize: 13 },
  vitalSub: { ...Typography.bodySmall, marginTop: 3 },

  // Stress + Confidence
  stressConfRow: { flexDirection: 'row', gap: 8, marginBottom: Spacing.md },
  stressConfCard: { borderWidth: 1, borderRadius: Radius.lg, padding: Spacing.lg },
  stressValue: { fontFamily: 'SpaceGrotesk-Bold', fontSize: 34, letterSpacing: -1, marginTop: 8, marginBottom: 6 },
  stressSub: { ...Typography.bodySmall, marginBottom: 10 },
  stressBar: { height: 3, borderRadius: 2, overflow: 'hidden' },
  stressFill: { height: '100%', borderRadius: 2 },
  arcContainer: { width: 100, height: 100, alignItems: 'center', justifyContent: 'center', marginTop: 8 },
  arcInner: { position: 'absolute', alignItems: 'center', justifyContent: 'center', flexDirection: 'row' },
  arcScore: { fontFamily: 'SpaceGrotesk-Bold', fontSize: 24, letterSpacing: -1 },
  arcLabel: { fontFamily: 'SpaceGrotesk-Regular', fontSize: 13, marginLeft: 1 },
  confLabel: { ...Typography.label, marginTop: 6 },

  // Quality
  qualGrid: { marginTop: 8, gap: 10 },
  qualRow: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  qualLabel: { ...Typography.bodySmall, width: 120 },
  qualTrack: { flex: 1, height: 3, borderRadius: 2, overflow: 'hidden' },
  qualFill: { height: '100%', borderRadius: 2 },
  qualVal: { ...Typography.mono, width: 36, textAlign: 'right' },

  // Tips
  tipCard: { borderWidth: 1, borderRadius: Radius.md, padding: Spacing.md, marginTop: 8 },
  tipHeader: { flexDirection: 'row', alignItems: 'flex-start' },
  tipTitle: { fontFamily: 'SpaceGrotesk-SemiBold', fontSize: 14 },
  tipSubtitle: { ...Typography.bodySmall, marginTop: 2 },
  tipDuration: { ...Typography.label },
  tipDetail: { ...Typography.bodySmall, marginTop: 10, lineHeight: 18, borderTopWidth: 1, paddingTop: 10 },

  // Buttons
  downloadBtn: { borderRadius: Radius.lg, padding: Spacing.md, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', marginTop: Spacing.sm, marginBottom: Spacing.md },
  downloadText: { fontFamily: 'SpaceGrotesk-SemiBold', fontSize: 16, color: '#FFF' },
  rescanBtn: { borderRadius: Radius.lg, borderWidth: 1.5, padding: Spacing.md, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', marginBottom: Spacing.md },
  rescanText: { fontFamily: 'SpaceGrotesk-SemiBold', fontSize: 16 },
  disclaimer: { ...Typography.label, textAlign: 'center', lineHeight: 18, paddingBottom: Spacing.xl },
  failBanner: { borderWidth: 1, borderRadius: Radius.md, padding: Spacing.md, marginTop: Spacing.md, marginBottom: Spacing.sm },
  failTitle: { fontFamily: 'SpaceGrotesk-Bold', fontSize: 14, color: '#FEE2E2', marginBottom: 4 },
  failBody: { ...Typography.bodySmall, color: '#FECACA' },
});

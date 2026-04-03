import React, { useRef, useEffect, useState } from 'react';
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  StyleSheet,
  Dimensions,
  Animated,
  StatusBar,
  Platform,
  Image,
} from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { LineChart } from 'react-native-chart-kit';

const { width } = Dimensions.get('window');
const CHART_WIDTH = width - 48;

type StressLevel = 'low' | 'moderate' | 'high';

interface ScanResult {
  bpm: number;
  rmssd: number;
  sdnn: number;
  lf_hf: number;
  confidence: number;
  ibi_array: number[];
  stress_level: StressLevel;
}

// ─── Color helpers ─────────────────────────────────────────────────────────────
const STRESS_COLORS: Record<StressLevel, string> = {
  low: '#22C55E',
  moderate: '#F59E0B',
  high: '#EF4444',
};
const STRESS_LABELS: Record<StressLevel, string> = {
  low: '😌  Low Stress',
  moderate: '😐  Moderate Stress',
  high: '😰  High Stress',
};
const STRESS_BG: Record<StressLevel, string> = {
  low: 'rgba(34,197,94,0.12)',
  moderate: 'rgba(245,158,11,0.12)',
  high: 'rgba(239,68,68,0.12)',
};

// Low-pass smooth IBI for charting
function smoothIBI(arr: number[], k = 3): number[] {
  if (arr.length <= k) return arr;
  return arr.map((_, i) => {
    const slice = arr.slice(Math.max(0, i - k), i + k + 1);
    return slice.reduce((a, b) => a + b, 0) / slice.length;
  });
}

function AnimatedMetricCard({
  icon,
  label,
  value,
  unit,
  color,
  delay = 0,
}: {
  icon: string;
  label: string;
  value: string;
  unit: string;
  color: string;
  delay?: number;
}) {
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const slideAnim = useRef(new Animated.Value(20)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(fadeAnim, { toValue: 1, duration: 500, delay, useNativeDriver: true }),
      Animated.timing(slideAnim, { toValue: 0, duration: 500, delay, useNativeDriver: true }),
    ]).start();
  }, []);

  return (
    <Animated.View
      style={[
        styles.metricCard,
        { opacity: fadeAnim, transform: [{ translateY: slideAnim }] },
      ]}
    >
      <Text style={styles.metricIcon}>{icon}</Text>
      <Text style={[styles.metricValue, { color }]}>{value}</Text>
      <Text style={styles.metricUnit}>{unit}</Text>
      <Text style={styles.metricLabel}>{label}</Text>
    </Animated.View>
  );
}

export default function ResultsScreen() {
  const params = useLocalSearchParams();
  const router = useRouter();
  const [chartReady, setChartReady] = useState(false);

  const headerFade = useRef(new Animated.Value(0)).current;
  const stressFade = useRef(new Animated.Value(0)).current;
  const chartFade = useRef(new Animated.Value(0)).current;

  // Parse result from navigation params
  const result: ScanResult | null = (() => {
    try {
      const raw = params.resultJson as string;
      return raw ? (JSON.parse(raw) as ScanResult) : null;
    } catch {
      return null;
    }
  })();

  useEffect(() => {
    Animated.stagger(200, [
      Animated.timing(headerFade, { toValue: 1, duration: 600, useNativeDriver: true }),
      Animated.timing(stressFade, { toValue: 1, duration: 600, useNativeDriver: true }),
    ]).start();

    setTimeout(() => {
      setChartReady(true);
      Animated.timing(chartFade, { toValue: 1, duration: 700, useNativeDriver: true }).start();
    }, 500);
  }, []);

  // ── Low confidence screen ──────────────────────────────────────────────────
  if (!result) {
    return (
      <View style={styles.centeredScreen}>
        <Text style={styles.errorIcon}>⚠️</Text>
        <Text style={styles.errorTitle}>No Result Data</Text>
        <Text style={styles.errorSub}>Something went wrong while processing.</Text>
        <TouchableOpacity style={styles.retryButton} onPress={() => router.replace('/(tabs)/camera')}>
          <Text style={styles.retryText}>Try Again</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const CONFIDENCE_THRESHOLD = 0.5;
  if (result.confidence < CONFIDENCE_THRESHOLD) {
    return (
      <View style={styles.centeredScreen}>
        <Text style={styles.errorIcon}>📶</Text>
        <Text style={styles.errorTitle}>Low Signal Quality</Text>
        <Text style={styles.errorSub}>
          Confidence: {Math.round(result.confidence * 100)}%{'\n'}
          Please ensure good lighting and keep face steady.
        </Text>
        <TouchableOpacity
          style={styles.retryButton}
          onPress={() => router.replace('/(tabs)/camera')}
        >
          <Text style={styles.retryText}>🔄  Retry Scan</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const stress = result.stress_level || 'low';
  const stressColor = STRESS_COLORS[stress];
  const confidencePct = Math.round(result.confidence * 100);
  const photoUri = params.photoUri as string;

  // Prepare IBI chart data
  const rawIBI = result.ibi_array?.length > 2 ? result.ibi_array : [800, 820, 810, 790, 830, 815, 800, 825];
  const chartIBI = smoothIBI(rawIBI).slice(-30); // last 30 points
  const chartLabels = chartIBI.map((_, i) => (i % 6 === 0 ? `${i}` : ''));

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#0D0D1A" />

      {/* ── Fixed Header ── */}
      <Animated.View style={[styles.header, { opacity: headerFade }]}>
        <TouchableOpacity onPress={() => router.replace('/(tabs)/camera')} style={styles.backBtn}>
          <Text style={styles.backBtnText}>‹</Text>
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Scan Results</Text>
        <View style={{ width: 40 }} />
      </Animated.View>

      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {/* ── Captured Frame ── */}
        {photoUri ? (
          <View style={styles.photoContainer}>
            <Image source={{ uri: photoUri }} style={styles.capturedPhoto} resizeMode="cover" />
            <Text style={styles.photoLabel}>Analyzed Frame</Text>
          </View>
        ) : null}

        {/* ── Stress Banner ── */}
        <Animated.View
          style={[
            styles.stressBanner,
            { backgroundColor: STRESS_BG[stress], borderColor: stressColor, opacity: stressFade },
          ]}
        >
          <Text style={[styles.stressBannerText, { color: stressColor }]}>
            {STRESS_LABELS[stress]}
          </Text>
          <Text style={styles.stressBannerSub}>Based on HRV analysis</Text>
        </Animated.View>

        {/* ── BPM Hero ── */}
        <View style={styles.bpmHero}>
          <Text style={styles.bpmValue}>{Math.round(result.bpm)}</Text>
          <Text style={styles.bpmUnit}>BPM</Text>
          <Text style={styles.bpmLabel}>Heart Rate</Text>
        </View>

        {/* ── Confidence badge ── */}
        <View
          style={[
            styles.confidenceBadge,
            {
              backgroundColor:
                confidencePct >= 80
                  ? 'rgba(34,197,94,0.15)'
                  : confidencePct >= 60
                  ? 'rgba(245,158,11,0.15)'
                  : 'rgba(239,68,68,0.15)',
              borderColor:
                confidencePct >= 80
                  ? '#22C55E'
                  : confidencePct >= 60
                  ? '#F59E0B'
                  : '#EF4444',
            },
          ]}
        >
          <Text
            style={[
              styles.confidenceText,
              {
                color:
                  confidencePct >= 80
                    ? '#22C55E'
                    : confidencePct >= 60
                    ? '#F59E0B'
                    : '#EF4444',
              },
            ]}
          >
            Signal Confidence: {confidencePct}%
          </Text>
        </View>

        {/* ── HRV Metric Cards ── */}
        <Text style={styles.sectionTitle}>HRV Metrics</Text>
        <View style={styles.metricsGrid}>
          <AnimatedMetricCard
            icon="⚡"
            label="RMSSD"
            value={result.rmssd?.toFixed(1) ?? '--'}
            unit="ms"
            color="#A78BFA"
            delay={0}
          />
          <AnimatedMetricCard
            icon="📊"
            label="SDNN"
            value={result.sdnn?.toFixed(1) ?? '--'}
            unit="ms"
            color="#60A5FA"
            delay={100}
          />
          <AnimatedMetricCard
            icon="〰️"
            label="LF/HF Ratio"
            value={result.lf_hf?.toFixed(2) ?? '--'}
            unit=""
            color={stressColor}
            delay={200}
          />
          <AnimatedMetricCard
            icon="💓"
            label="IBI Samples"
            value={String(rawIBI.length)}
            unit="beats"
            color="#F472B6"
            delay={300}
          />
        </View>

        {/* ── IBI / Pulse Wave Chart ── */}
        <Text style={styles.sectionTitle}>IBI Signal (Inter-Beat Interval)</Text>
        <Animated.View style={[styles.chartContainer, { opacity: chartFade }]}>
          {chartReady && chartIBI.length > 1 && (
            <LineChart
              data={{
                labels: chartLabels,
                datasets: [
                  {
                    data: chartIBI,
                    strokeWidth: 2,
                    color: (opacity = 1) => `rgba(108, 99, 255, ${opacity})`,
                  },
                ],
              }}
              width={CHART_WIDTH}
              height={200}
              chartConfig={{
                backgroundColor: 'transparent',
                backgroundGradientFrom: '#13131F',
                backgroundGradientTo: '#1A1A2E',
                decimalPlaces: 0,
                color: (opacity = 1) => `rgba(108, 99, 255, ${opacity})`,
                labelColor: (opacity = 1) => `rgba(156, 163, 175, ${opacity})`,
                propsForDots: { r: '2', strokeWidth: '1', stroke: '#6C63FF' },
                propsForBackgroundLines: { strokeDasharray: '', stroke: 'rgba(255,255,255,0.05)' },
              }}
              bezier
              withDots={false}
              withShadow
              style={styles.chart}
            />
          )}
          {(!chartReady || chartIBI.length <= 1) && (
            <View style={styles.chartPlaceholder}>
              <Text style={styles.chartPlaceholderText}>Loading chart…</Text>
            </View>
          )}
        </Animated.View>

        {/* ── Stress Explanation ── */}
        <View style={[styles.explanationCard, { borderColor: stressColor }]}>
          <Text style={[styles.explanationTitle, { color: stressColor }]}>
            {stress === 'low' && '✅ Good HRV — Relaxed state'}
            {stress === 'moderate' && '⚠️ Moderate Tension Detected'}
            {stress === 'high' && '🔴 High Stress Detected'}
          </Text>
          <Text style={styles.explanationBody}>
            {stress === 'low' &&
              'Your LF/HF ratio indicates a balanced autonomic nervous system. Great job staying calm!'}
            {stress === 'moderate' &&
              'Slight sympathetic dominance observed. Consider deep breathing or a short break.'}
            {stress === 'high' &&
              'Elevated sympathetic activity. Try Box Breathing: inhale 4s → hold 4s → exhale 4s → hold 4s.'}
          </Text>
        </View>

        {/* ── Retry Button ── */}
        <TouchableOpacity
          style={styles.retryButton}
          onPress={() => router.replace('/(tabs)/camera')}
          activeOpacity={0.85}
        >
          <Text style={styles.retryText}>🔄  New Scan</Text>
        </TouchableOpacity>

        <View style={{ height: 40 }} />
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0D0D1A' },
  centeredScreen: {
    flex: 1,
    backgroundColor: '#0D0D1A',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 32,
  },
  errorIcon: { fontSize: 56, marginBottom: 16 },
  errorTitle: { color: '#FFF', fontSize: 22, fontWeight: '700', marginBottom: 8 },
  errorSub: { color: '#9CA3AF', fontSize: 15, textAlign: 'center', lineHeight: 24, marginBottom: 32 },

  // Header
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingTop: Platform.OS === 'android' ? 44 : 60,
    paddingBottom: 16,
    backgroundColor: '#0D0D1A',
  },
  backBtn: {
    width: 40, height: 40,
    alignItems: 'center', justifyContent: 'center',
    backgroundColor: 'rgba(255,255,255,0.08)',
    borderRadius: 20,
  },
  backBtnText: { color: '#FFF', fontSize: 28, marginTop: -2 },
  headerTitle: { color: '#FFF', fontWeight: '700', fontSize: 18 },

  // Photo
  photoContainer: {
    alignItems: 'center',
    marginBottom: 24,
  },
  capturedPhoto: {
    width: 120,
    height: 120,
    borderRadius: 60,
    borderWidth: 2,
    borderColor: 'rgba(108,99,255,0.4)',
  },
  photoLabel: {
    color: '#9CA3AF',
    fontSize: 12,
    marginTop: 8,
  },

  // Scroll
  scroll: { flex: 1 },
  scrollContent: { paddingHorizontal: 24, paddingTop: 8, paddingBottom: 20 },

  // Stress banner
  stressBanner: {
    borderRadius: 16,
    borderWidth: 1.5,
    padding: 18,
    alignItems: 'center',
    marginBottom: 20,
  },
  stressBannerText: { fontSize: 22, fontWeight: '800', marginBottom: 4 },
  stressBannerSub: { color: '#9CA3AF', fontSize: 12 },

  // BPM hero
  bpmHero: {
    alignItems: 'center',
    marginBottom: 12,
  },
  bpmValue: {
    fontSize: 80,
    fontWeight: '900',
    color: '#FFFFFF',
    lineHeight: 90,
  },
  bpmUnit: { fontSize: 22, color: '#A78BFA', fontWeight: '700', marginTop: -8 },
  bpmLabel: { color: '#6B7280', fontSize: 13, marginTop: 4 },

  // Confidence
  confidenceBadge: {
    alignSelf: 'center',
    paddingVertical: 7,
    paddingHorizontal: 20,
    borderRadius: 20,
    borderWidth: 1,
    marginBottom: 24,
  },
  confidenceText: { fontWeight: '600', fontSize: 13 },

  // Section title
  sectionTitle: {
    color: '#E5E7EB',
    fontSize: 15,
    fontWeight: '700',
    marginBottom: 12,
    letterSpacing: 0.3,
  },

  // Metric cards
  metricsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
    marginBottom: 28,
  },
  metricCard: {
    width: (width - 60) / 2,
    backgroundColor: '#13131F',
    borderRadius: 16,
    padding: 16,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.07)',
  },
  metricIcon: { fontSize: 22, marginBottom: 6 },
  metricValue: { fontSize: 26, fontWeight: '800' },
  metricUnit: { color: '#6B7280', fontSize: 12, marginTop: 2 },
  metricLabel: { color: '#9CA3AF', fontSize: 12, marginTop: 4 },

  // Chart
  chartContainer: {
    backgroundColor: '#13131F',
    borderRadius: 20,
    overflow: 'hidden',
    marginBottom: 24,
    borderWidth: 1,
    borderColor: 'rgba(108,99,255,0.2)',
  },
  chart: { borderRadius: 20 },
  chartPlaceholder: {
    height: 200,
    alignItems: 'center',
    justifyContent: 'center',
  },
  chartPlaceholderText: { color: '#6B7280', fontSize: 14 },

  // Explanation
  explanationCard: {
    borderRadius: 16,
    borderWidth: 1.5,
    padding: 18,
    backgroundColor: 'rgba(255,255,255,0.03)',
    marginBottom: 28,
  },
  explanationTitle: { fontSize: 15, fontWeight: '700', marginBottom: 8 },
  explanationBody: { color: '#9CA3AF', fontSize: 13, lineHeight: 22 },

  // Retry
  retryButton: {
    paddingVertical: 16,
    alignItems: 'center',
    backgroundColor: '#6C63FF',
    borderRadius: 50,
    marginBottom: 8,
    elevation: 8,
    shadowColor: '#6C63FF',
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.5,
    shadowRadius: 10,
  },
  retryText: { color: '#FFF', fontSize: 16, fontWeight: '700' },
});

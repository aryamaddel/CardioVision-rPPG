// app/index.tsx — HomeScreen
import React, { useEffect, useRef, useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Animated, Dimensions, ScrollView, Image, Alert } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';
import * as Haptics from 'expo-haptics';
import * as DocumentPicker from 'expo-document-picker';
import Svg, { Path, Line, Circle as SvgCircle, Defs, LinearGradient as SvgLinGrad, Stop } from 'react-native-svg';
import { Ionicons } from '@expo/vector-icons';
import { useTheme, Typography, Spacing, Radius, Shadows } from '../theme';

const { width } = Dimensions.get('window');

// ── Animated ECG line behind the heart ──
function ECGBackground() {
  const { accent } = useTheme();
  const offset = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    Animated.loop(Animated.timing(offset, { toValue: -width, duration: 3000, useNativeDriver: true })).start();
    return () => {};
  }, []);
  const ecg = `M0,40 L${width*0.08},40 L${width*0.12},40 L${width*0.14},15 L${width*0.16},65 L${width*0.19},8 L${width*0.22},40 L${width*0.35},40 L${width*0.38},15 L${width*0.40},65 L${width*0.43},8 L${width*0.46},40 L${width*0.6},40 L${width*0.63},15 L${width*0.65},65 L${width*0.68},8 L${width*0.71},40 L${width},40`;
  return (
    <Animated.View style={{ position: 'absolute', top: 0, left: 0, width: width * 2, transform: [{ translateX: offset }] }} pointerEvents="none">
      <Svg width={width * 2} height={80}>
        <Path d={ecg} stroke={accent.primary} strokeWidth={1.5} fill="none" opacity={0.18} />
        <Path d={`M${width},40 ` + ecg.substring(ecg.indexOf(' '))} stroke={accent.primary} strokeWidth={1.5} fill="none" opacity={0.18} />
      </Svg>
    </Animated.View>
  );
}

// ── n8n-style process flow node ──
function FlowNode({ icon, label, index, total }: { icon: string; label: string; index: number; total: number }) {
  const { colors, accent } = useTheme();
  const fadeIn = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    Animated.timing(fadeIn, { toValue: 1, duration: 500, delay: index * 120, useNativeDriver: true }).start();
  }, []);
  return (
    <Animated.View style={{ opacity: fadeIn, alignItems: 'center' }}>
      <View style={[styles.flowNode, { backgroundColor: colors.surface, borderColor: colors.border }]}>
        <Ionicons name={icon as any} size={20} color={accent.primary} />
      </View>
      <Text style={[styles.flowLabel, { color: colors.textSecondary }]}>{label}</Text>
      {index < total - 1 && (
        <View style={styles.flowConnector}>
          <Ionicons name="chevron-down" size={14} color={colors.textMuted} />
        </View>
      )}
    </Animated.View>
  );
}

export default function HomeScreen() {
  const router = useRouter();
  const { colors, accent, isDark, toggle } = useTheme();
  const [isPickingVideo, setIsPickingVideo] = useState(false);
  const heroAlpha = useRef(new Animated.Value(0)).current;
  const heroY = useRef(new Animated.Value(30)).current;
  const btnScale = useRef(new Animated.Value(0.95)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.spring(heroY, { toValue: 0, friction: 8, tension: 50, useNativeDriver: true }),
      Animated.timing(heroAlpha, { toValue: 1, duration: 800, useNativeDriver: true }),
      Animated.spring(btnScale, { toValue: 1, delay: 300, friction: 7, useNativeDriver: true }),
    ]).start();
  }, []);

  const handleScan = async () => {
    await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    router.push('/record');
  };

  const handleUploadVideo = async () => {
    if (isPickingVideo) {
      return;
    }

    setIsPickingVideo(true);
    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: ['video/*'],
      });

      if (result.canceled || !result.assets?.[0]?.uri) {
        return;
      }

      await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
      router.push({
        pathname: '/processing',
        params: { videoUri: result.assets[0].uri },
      });
    } catch {
      Alert.alert('Unable to pick video', 'Please try again.');
    } finally {
      setIsPickingVideo(false);
    }
  };

  const flowSteps = [
    { icon: 'camera-outline', label: 'Face Capture' },
    { icon: 'scan-outline', label: 'ROI Detection' },
    { icon: 'pulse-outline', label: 'Signal Extract' },
    { icon: 'git-merge-outline', label: 'POS + Neural' },
    { icon: 'analytics-outline', label: 'HRV Analysis' },
    { icon: 'shield-checkmark-outline', label: 'Triage Agent' },
  ];

  return (
    <View style={[styles.root, { backgroundColor: colors.background }]}>
      <SafeAreaView style={styles.safe}>
        {/* Dark mode toggle */}
        <View style={styles.topRow}>
          <View />
          <TouchableOpacity onPress={toggle} style={[styles.toggleBtn, { backgroundColor: colors.surfaceHigh, borderColor: colors.border }]}>
            <Ionicons name={isDark ? 'sunny-outline' : 'moon-outline'} size={18} color={colors.textSecondary} />
          </TouchableOpacity>
        </View>

        <ScrollView contentContainerStyle={styles.scroll} showsVerticalScrollIndicator={false}>
          {/* Header */}
          <Animated.View style={[styles.header, { opacity: heroAlpha, transform: [{ translateY: heroY }] }]}>
            <Text style={[styles.tagline, { color: colors.textPrimary }]}>CardioVision</Text>
            <Text style={[styles.subTagline, { color: colors.textTertiary }]}>
              Contact-free cardiac monitoring{'\n'}using remote photoplethysmography
            </Text>
          </Animated.View>

          {/* Heart with gradient + ECG behind */}
          <Animated.View style={[styles.heroContainer, { opacity: heroAlpha }]}>
            {/* ECG behind heart */}
            <View style={styles.ecgBehind}>
              <ECGBackground />
            </View>
            {/* Gradient overlay */}
            <LinearGradient
              colors={isDark
                ? ['transparent', 'rgba(10,10,10,0.7)', 'rgba(10,10,10,0.95)']
                : ['transparent', 'rgba(247,247,248,0.7)', 'rgba(247,247,248,0.95)']
              }
              style={styles.heartGradient}
            />
            <Image source={require('../assets/images/heart_3d.png')} style={styles.heartImage} resizeMode="contain" />
          </Animated.View>

          {/* n8n-style process flow */}
          <View style={styles.flowContainer}>
            <Text style={[styles.flowTitle, { color: colors.textTertiary }]}>HOW IT WORKS</Text>
            <View style={styles.flowGrid}>
              {/* Left column */}
              <View style={styles.flowColumn}>
                {flowSteps.slice(0, 3).map((s, i) => (
                  <FlowNode key={s.label} icon={s.icon} label={s.label} index={i} total={3} />
                ))}
              </View>
              {/* Arrow between columns */}
              <View style={styles.flowArrow}>
                <Ionicons name="arrow-forward" size={20} color={accent.primary} />
              </View>
              {/* Right column */}
              <View style={styles.flowColumn}>
                {flowSteps.slice(3).map((s, i) => (
                  <FlowNode key={s.label} icon={s.icon} label={s.label} index={i + 3} total={6} />
                ))}
              </View>
            </View>
          </View>

          {/* CTA Buttons */}
          <Animated.View style={[styles.ctaContainer, { transform: [{ scale: btnScale }] }]}>
            <TouchableOpacity onPress={handleScan} activeOpacity={0.9} style={[styles.ctaButton, Shadows.cardMd]}>
              <LinearGradient colors={[accent.primary, accent.dark]} style={styles.ctaGradient}>
                <Ionicons name="videocam-outline" size={22} color="#FFF" style={{ marginRight: 10 }} />
                <View>
                  <Text style={styles.ctaText}>Begin Scan</Text>
                  <Text style={styles.ctaSubtext}>30-second facial recording</Text>
                </View>
              </LinearGradient>
            </TouchableOpacity>

            <TouchableOpacity
              onPress={handleUploadVideo}
              disabled={isPickingVideo}
              activeOpacity={0.9}
              style={[
                styles.uploadButton,
                { borderColor: accent.primary },
                isPickingVideo && styles.uploadButtonDisabled,
              ]}
            >
              <Ionicons name="cloud-upload-outline" size={20} color={accent.primary} style={{ marginRight: 10 }} />
              <Text style={[styles.uploadText, { color: accent.primary }]}>Upload a Video</Text>
            </TouchableOpacity>
          </Animated.View>

          <Text style={[styles.footer, { color: colors.textMuted }]}>
            UBFC-rPPG validated · Clinical-grade confidence scoring
          </Text>

        </ScrollView>
      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1 },
  safe: { flex: 1 },
  scroll: { paddingHorizontal: Spacing.lg, paddingBottom: 60 },
  topRow: { flexDirection: 'row', justifyContent: 'flex-end', paddingHorizontal: Spacing.lg, paddingTop: Spacing.sm },
  toggleBtn: { width: 38, height: 38, borderRadius: 19, borderWidth: 1, alignItems: 'center', justifyContent: 'center' },

  header: { marginTop: Spacing.lg, alignItems: 'center' },
  tagline: { fontFamily: 'SpaceGrotesk-Bold', fontSize: 36, letterSpacing: -1.5, textAlign: 'center' },
  subTagline: { ...Typography.body, textAlign: 'center', marginTop: 8, lineHeight: 22 },

  // Hero
  heroContainer: { alignItems: 'center', marginTop: Spacing.md, height: 260, justifyContent: 'center', overflow: 'hidden' },
  ecgBehind: { position: 'absolute', top: 90, left: 0, right: 0, height: 80, overflow: 'hidden' },
  heartGradient: { position: 'absolute', bottom: 0, left: 0, right: 0, height: 120 },
  heartImage: { width: width * 0.65, height: width * 0.55 },

  // Flow
  flowContainer: { marginTop: Spacing.sm },
  flowTitle: { ...Typography.label, textAlign: 'center', marginBottom: Spacing.md },
  flowGrid: { flexDirection: 'row', justifyContent: 'center', alignItems: 'flex-start', gap: 12 },
  flowColumn: { alignItems: 'center', gap: 4 },
  flowArrow: { justifyContent: 'center', paddingTop: 20 },
  flowNode: { width: 50, height: 50, borderRadius: 14, borderWidth: 1.5, alignItems: 'center', justifyContent: 'center', marginBottom: 4 },
  flowLabel: { fontFamily: 'SpaceGrotesk-Medium', fontSize: 10, textAlign: 'center', width: 70 },
  flowConnector: { marginVertical: 2 },

  // CTA
  ctaContainer: { marginTop: Spacing.xl },
  ctaButton: { borderRadius: Radius.lg, overflow: 'hidden', marginBottom: Spacing.md },
  ctaGradient: { paddingVertical: 18, paddingHorizontal: Spacing.xl, flexDirection: 'row', alignItems: 'center', justifyContent: 'center' },
  ctaText: { fontFamily: 'SpaceGrotesk-Bold', fontSize: 18, color: '#FFF' },
  ctaSubtext: { fontFamily: 'SpaceGrotesk-Regular', fontSize: 11, color: 'rgba(255,255,255,0.7)' },
  uploadButton: { borderRadius: Radius.lg, borderWidth: 1.5, paddingVertical: 16, flexDirection: 'row', alignItems: 'center', justifyContent: 'center' },
  uploadButtonDisabled: { opacity: 0.6 },
  uploadText: { fontFamily: 'SpaceGrotesk-SemiBold', fontSize: 16 },

  footer: { ...Typography.label, textAlign: 'center', marginTop: Spacing.xl },
});

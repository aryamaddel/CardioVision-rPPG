import React, { useEffect, useRef } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Animated,
  StatusBar,
  Dimensions,
} from 'react-native';
import { useRouter } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';

const { width, height } = Dimensions.get('window');

export default function HomeScreen() {
  const router = useRouter();
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const slideAnim = useRef(new Animated.Value(40)).current;

  useEffect(() => {
    // Fade in on mount
    Animated.parallel([
      Animated.timing(fadeAnim, { toValue: 1, duration: 800, useNativeDriver: true }),
      Animated.timing(slideAnim, { toValue: 0, duration: 800, useNativeDriver: true }),
    ]).start();

    // Pulse animation for heart icon
    const pulse = Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, { toValue: 1.25, duration: 600, useNativeDriver: true }),
        Animated.timing(pulseAnim, { toValue: 1, duration: 600, useNativeDriver: true }),
      ])
    );
    pulse.start();
    return () => pulse.stop();
  }, []);

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#0D0D1A" />

      {/* Background gradient circles */}
      <View style={styles.bgCircle1} />
      <View style={styles.bgCircle2} />

      <Animated.View
        style={[
          styles.content,
          { opacity: fadeAnim, transform: [{ translateY: slideAnim }] },
        ]}
      >
        {/* Logo / Pulse Icon */}
        <Animated.View style={[styles.iconWrapper, { transform: [{ scale: pulseAnim }] }]}>
          <Text style={styles.heartIcon}>❤️</Text>
        </Animated.View>

        {/* App Name */}
        <Text style={styles.appName}>CardioVision</Text>
        <Text style={styles.tagline}>rPPG Heart Rate & Stress Monitor</Text>

        {/* Feature pills */}
        <View style={styles.pillRow}>
          {['BPM', 'HRV', 'Stress'].map((label) => (
            <View key={label} style={styles.pill}>
              <Text style={styles.pillText}>{label}</Text>
            </View>
          ))}
        </View>

        {/* Description */}
        <Text style={styles.description}>
          Place your face in front of the camera.{'\n'}
          Our AI measures your pulse using light reflection.
        </Text>

        {/* CTA Button */}
        <TouchableOpacity
          style={styles.startButton}
          onPress={() => router.push('/(tabs)/camera')}
          activeOpacity={0.85}
        >
          <LinearGradient
            colors={['#7C3AED', '#6C63FF', '#4F46E5']}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
            style={styles.startButtonGradient}
          >
            <Text style={styles.startButtonText}>🔬  Start Scan</Text>
          </LinearGradient>
        </TouchableOpacity>

        <Text style={styles.disclaimer}>Ensure good lighting for best results</Text>
      </Animated.View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0D0D1A',
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
  },
  bgCircle1: {
    position: 'absolute',
    width: 350,
    height: 350,
    borderRadius: 175,
    backgroundColor: 'rgba(108,99,255,0.12)',
    top: -80,
    right: -80,
  },
  bgCircle2: {
    position: 'absolute',
    width: 280,
    height: 280,
    borderRadius: 140,
    backgroundColor: 'rgba(124,58,237,0.10)',
    bottom: -60,
    left: -80,
  },
  content: {
    alignItems: 'center',
    paddingHorizontal: 32,
  },
  iconWrapper: {
    width: 100,
    height: 100,
    borderRadius: 50,
    backgroundColor: 'rgba(108,99,255,0.18)',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 24,
    borderWidth: 2,
    borderColor: 'rgba(108,99,255,0.4)',
  },
  heartIcon: {
    fontSize: 44,
  },
  appName: {
    fontSize: 36,
    fontWeight: '800',
    color: '#FFFFFF',
    letterSpacing: 1,
    marginBottom: 6,
  },
  tagline: {
    fontSize: 14,
    color: '#A78BFA',
    letterSpacing: 0.5,
    marginBottom: 24,
  },
  pillRow: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 28,
  },
  pill: {
    paddingVertical: 6,
    paddingHorizontal: 18,
    borderRadius: 20,
    backgroundColor: 'rgba(108,99,255,0.2)',
    borderWidth: 1,
    borderColor: 'rgba(108,99,255,0.45)',
  },
  pillText: {
    color: '#A78BFA',
    fontWeight: '600',
    fontSize: 13,
  },
  description: {
    textAlign: 'center',
    color: '#9CA3AF',
    fontSize: 14,
    lineHeight: 22,
    marginBottom: 40,
  },
  startButton: {
    width: width * 0.75,
    borderRadius: 16,
    overflow: 'hidden',
    elevation: 8,
    shadowColor: '#6C63FF',
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.5,
    shadowRadius: 10,
    marginBottom: 20,
  },
  startButtonGradient: {
    paddingVertical: 18,
    alignItems: 'center',
    justifyContent: 'center',
  },
  startButtonText: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: '700',
    letterSpacing: 0.5,
  },
  disclaimer: {
    color: '#6B7280',
    fontSize: 12,
  },
});
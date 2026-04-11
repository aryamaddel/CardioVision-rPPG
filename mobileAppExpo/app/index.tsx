// app/index.tsx — HomeScreen
import React, { useEffect, useRef, useState } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Animated,
  Dimensions,
  ScrollView,
  Alert,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useRouter } from "expo-router";
import { LinearGradient } from "expo-linear-gradient";
import * as Haptics from "expo-haptics";
import * as DocumentPicker from "expo-document-picker";
import { Ionicons } from "@expo/vector-icons";
import { useTheme, Typography, Spacing, Radius, Shadows } from "../theme";

const { width } = Dimensions.get("window");

export default function HomeScreen() {
  const router = useRouter();
  const { colors, accent, isDark, toggle } = useTheme();
  const [isPickingVideo, setIsPickingVideo] = useState(false);
  const heroAlpha = useRef(new Animated.Value(0)).current;
  const heroY = useRef(new Animated.Value(20)).current;
  const btnScale = useRef(new Animated.Value(0.95)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.spring(heroY, {
        toValue: 0,
        friction: 8,
        tension: 40,
        useNativeDriver: true,
      }),
      Animated.timing(heroAlpha, {
        toValue: 1,
        duration: 1000,
        useNativeDriver: true,
      }),
      Animated.spring(btnScale, {
        toValue: 1,
        delay: 200,
        friction: 7,
        useNativeDriver: true,
      }),
    ]).start();
  }, [heroY, heroAlpha, btnScale]);

  const handleScan = async () => {
    await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    router.push("/record");
  };

  const handleUploadVideo = async () => {
    if (isPickingVideo) return;
    setIsPickingVideo(true);
    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: ["video/*"],
      });
      if (result.canceled || !result.assets?.[0]?.uri) return;
      await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
      router.push({
        pathname: "/processing",
        params: { videoUri: result.assets[0].uri },
      });
    } catch {
      Alert.alert("Unable to pick video", "Please try again.");
    } finally {
      setIsPickingVideo(false);
    }
  };

  return (
    <View style={[styles.root, { backgroundColor: colors.background }]}>
      <SafeAreaView style={styles.safe}>
        {/* Top Header Row */}
        <View style={styles.topRow}>
          <Text style={[styles.logoText, { color: colors.textPrimary }]}>
            CardioVision
          </Text>
          <TouchableOpacity
            onPress={toggle}
            style={[
              styles.toggleBtn,
              {
                backgroundColor: colors.surfaceHigh,
                borderColor: colors.border,
              },
            ]}
          >
            <Ionicons
              name={isDark ? "sunny-outline" : "moon-outline"}
              size={18}
              color={colors.textSecondary}
            />
          </TouchableOpacity>
        </View>

        <ScrollView
          contentContainerStyle={styles.scroll}
          showsVerticalScrollIndicator={false}
        >
          {/* Hero Section */}
          <Animated.View
            style={[
              styles.hero,
              { opacity: heroAlpha, transform: [{ translateY: heroY }] },
            ]}
          >
            <View
              style={[
                styles.badge,
                {
                  backgroundColor: accent.ghost,
                  borderColor: accent.light + "20",
                },
              ]}
            >
              <Text style={[styles.badgeText, { color: accent.primary }]}>
                AI-POWERED VITALS
              </Text>
            </View>
            <Text style={[styles.title, { color: colors.textPrimary }]}>
              Seamless cardiac insights
            </Text>
            <Text style={[styles.subtitle, { color: colors.textSecondary }]}>
              Contact-free monitoring powered by remote photoplethysmography
              (rPPG).
            </Text>
          </Animated.View>

          {/* Action Cards */}
          <Animated.View
            style={[styles.ctaContainer, { transform: [{ scale: btnScale }] }]}
          >
            <TouchableOpacity
              onPress={handleScan}
              activeOpacity={0.9}
              style={[styles.mainCard, Shadows.cardMd]}
            >
              <LinearGradient
                colors={[accent.primary, accent.dark]}
                style={styles.cardGradient}
              >
                <View style={styles.cardIconWrapper}>
                  <Ionicons name="scan-outline" size={32} color="#FFF" />
                </View>
                <View style={styles.cardContent}>
                  <Text style={styles.cardTitle}>Instant Scan</Text>
                  <Text style={styles.cardDescription}>
                    Point the camera at your face for 30s
                  </Text>
                </View>
                <Ionicons
                  name="chevron-forward"
                  size={20}
                  color="rgba(255,255,255,0.6)"
                />
              </LinearGradient>
            </TouchableOpacity>

            <TouchableOpacity
              onPress={handleUploadVideo}
              disabled={isPickingVideo}
              activeOpacity={0.7}
              style={[
                styles.secondaryCard,
                { backgroundColor: colors.surface, borderColor: colors.border },
              ]}
            >
              <View
                style={[
                  styles.secondaryIconWrapper,
                  {
                    backgroundColor: isDark
                      ? colors.surfaceHigh
                      : colors.background,
                  },
                ]}
              >
                <Ionicons
                  name="cloud-upload-outline"
                  size={24}
                  color={accent.primary}
                />
              </View>
              <View style={styles.cardContent}>
                <Text
                  style={[
                    styles.secondaryCardTitle,
                    { color: colors.textPrimary },
                  ]}
                >
                  Upload Recording
                </Text>
                <Text
                  style={[
                    styles.secondaryCardDescription,
                    { color: colors.textSecondary },
                  ]}
                >
                  Analyze a pre-recorded video
                </Text>
              </View>
              <Ionicons
                name="arrow-forward-outline"
                size={18}
                color={colors.textMuted}
              />
            </TouchableOpacity>
          </Animated.View>

          <View style={styles.spacer} />

          {/* Footer Info */}
          <View
            style={[
              styles.footer,
              {
                backgroundColor: isDark
                  ? "rgba(255,255,255,0.03)"
                  : "rgba(0,0,0,0.02)",
              },
            ]}
          >
            <Ionicons
              name="shield-checkmark-outline"
              size={16}
              color={colors.textSecondary}
              style={{ marginBottom: 8 }}
            />
            <Text style={[styles.footerText, { color: colors.textSecondary }]}>
              Clinically validated · Privacy focused
            </Text>
            <Text style={[styles.footerSubtext, { color: colors.textMuted }]}>
              Data is processed locally on your device
            </Text>
          </View>
        </ScrollView>
      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1 },
  safe: { flex: 1 },
  topRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingHorizontal: Spacing.xl,
    paddingTop: Spacing.md,
    paddingBottom: Spacing.sm,
  },
  logoText: {
    fontFamily: "SpaceGrotesk-Bold",
    fontSize: 18,
    letterSpacing: -0.5,
  },
  toggleBtn: {
    width: 40,
    height: 40,
    borderRadius: 20,
    borderWidth: 1,
    alignItems: "center",
    justifyContent: "center",
  },

  scroll: {
    paddingHorizontal: Spacing.xl,
    paddingTop: Spacing.xxl,
    paddingBottom: 40,
  },

  hero: { marginBottom: Spacing.xxxl },
  badge: {
    alignSelf: "flex-start",
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: Radius.full,
    borderWidth: 1,
    marginBottom: Spacing.md,
  },
  badgeText: {
    fontFamily: "SpaceGrotesk-Bold",
    fontSize: 10,
    letterSpacing: 1,
  },
  title: { ...Typography.displayM, marginBottom: Spacing.sm },
  subtitle: { ...Typography.body, lineHeight: 24, fontSize: 16 },

  ctaContainer: { gap: Spacing.md },

  mainCard: { borderRadius: Radius.lg, overflow: "hidden" },
  cardGradient: {
    padding: Spacing.xl,
    flexDirection: "row",
    alignItems: "center",
  },
  cardIconWrapper: {
    width: 56,
    height: 56,
    borderRadius: 16,
    backgroundColor: "rgba(255,255,255,0.15)",
    alignItems: "center",
    justifyContent: "center",
    marginRight: Spacing.lg,
  },
  cardContent: { flex: 1 },
  cardTitle: {
    fontFamily: "SpaceGrotesk-Bold",
    fontSize: 20,
    color: "#FFF",
    marginBottom: 2,
  },
  cardDescription: {
    fontFamily: "SpaceGrotesk-Regular",
    fontSize: 13,
    color: "rgba(255,255,255,0.7)",
  },

  secondaryCard: {
    borderRadius: Radius.lg,
    borderWidth: 1,
    paddingHorizontal: Spacing.lg,
    paddingVertical: Spacing.lg,
    flexDirection: "row",
    alignItems: "center",
  },
  secondaryIconWrapper: {
    width: 48,
    height: 48,
    borderRadius: 14,
    alignItems: "center",
    justifyContent: "center",
    marginRight: Spacing.md,
  },
  secondaryCardTitle: {
    fontFamily: "SpaceGrotesk-SemiBold",
    fontSize: 16,
    marginBottom: 2,
  },
  secondaryCardDescription: {
    fontFamily: "SpaceGrotesk-Regular",
    fontSize: 13,
  },

  spacer: { height: Spacing.xxxl },

  footer: {
    padding: Spacing.xl,
    borderRadius: Radius.lg,
    alignItems: "center",
    marginTop: Spacing.xl,
  },
  footerText: {
    fontFamily: "SpaceGrotesk-SemiBold",
    fontSize: 14,
    marginBottom: 4,
  },
  footerSubtext: {
    fontFamily: "SpaceGrotesk-Regular",
    fontSize: 12,
    textAlign: "center",
  },
});

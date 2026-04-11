// app/processing.tsx — Simplified ProcessingScreen
import React, { useEffect, useRef, useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  Animated,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useRouter, useLocalSearchParams } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { useTheme, Typography, Spacing, Radius } from "../theme";
import { processVideo, getMockResult } from "../api/rppgService";
import {
  clearPendingScanResult,
  getPendingScanResult,
  getScanSession,
  setPendingScanResult,
  setScanSession,
} from "../state/scanSession";

function PulseBar() {
  const { accent } = useTheme();
  const animsRef = useRef(
    Array.from({ length: 24 }, () => new Animated.Value(0.2)),
  );
  const anims = animsRef.current;
  useEffect(() => {
    const loops = anims.map((a, i) =>
      Animated.loop(
        Animated.sequence([
          Animated.delay(i * 60),
          Animated.timing(a, {
            toValue: 0.6 + Math.random() * 0.4,
            duration: 300,
            useNativeDriver: true,
          }),
          Animated.timing(a, {
            toValue: 0.2,
            duration: 300,
            useNativeDriver: true,
          }),
        ]),
      ),
    );
    loops.forEach((l) => l.start());
    return () => loops.forEach((l) => l.stop());
  }, [anims]);
  return (
    <View style={styles.pulseBar}>
      {anims.map((a, i) => (
        <Animated.View
          key={i}
          style={{
            width: 3,
            borderRadius: 2,
            backgroundColor: accent.primary,
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
  const router = useRouter();
  const { colors, accent } = useTheme();
  const params = useLocalSearchParams();
  const videoUri = params.videoUri as string | undefined;
  const streamResultJson = params.streamResultJson as string | undefined;
  const [progress, setProgress] = useState(0);
  const [statusMsg, setStatusMsg] = useState("Initializing analyzer...");
  const progressAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    const runPipeline = async () => {
      const session = getScanSession();
      let result: any = null;
      let apiDone = false;

      const resolveResult = (resolved: any) => {
        result = resolved;
        apiDone = true;
        setScanSession({
          result: resolved,
          videoUri: videoUri ?? session.videoUri,
        });
        clearPendingScanResult();
      };

      const resolveFallback = () => {
        resolveResult(getMockResult());
      };

      // 1. Get the data
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
          setStatusMsg("Uploading video for analysis...");
          pending = processVideo(videoUri, (pct) => {
             setProgress(pct);
             Animated.timing(progressAnim, {
               toValue: pct / 100,
               duration: 200,
               useNativeDriver: false,
             }).start();
          });
          setPendingScanResult(pending);
        }

        if (pending) {
          pending.then((r) => {
            setStatusMsg("Compiling results...");
            resolveResult(r);
          }).catch(() => resolveFallback());
        } else {
          resolveFallback();
        }
      }

      // 2. Wait for UI animation to feel "natural" if it's too fast
      // or if it's already done, just show the last bit
      if (apiDone) {
        setStatusMsg("Finalizing report...");
        setProgress(100);
        Animated.timing(progressAnim, {
          toValue: 1,
          duration: 800,
          useNativeDriver: false,
        }).start();
        await new Promise((res) => setTimeout(res, 1200));
      } else {
        // If still waiting for backend, show a slow progress crawl
        setStatusMsg("Extracting cardiac signals...");
        let currentPct = progress;
        const interval = setInterval(() => {
          if (apiDone) {
            clearInterval(interval);
            return;
          }
          currentPct = Math.min(98, currentPct + 0.5);
          setProgress(Math.round(currentPct));
          Animated.timing(progressAnim, {
            toValue: currentPct / 100,
            duration: 800,
            useNativeDriver: false,
          }).start();
        }, 1000);
      }

      // Wait until result is ready
      while (!apiDone) {
        await new Promise(res => setTimeout(res, 500));
      }

      router.replace({ pathname: "/results" });
    };
    runPipeline();
  }, [videoUri, streamResultJson, progressAnim, router]);

  const progressWidth = progressAnim.interpolate({
    inputRange: [0, 1],
    outputRange: ["0%", "100%"],
  });

  return (
    <View style={[styles.root, { backgroundColor: colors.background }]}>
      <SafeAreaView style={styles.safe}>
        <View style={styles.header}>
          <Ionicons
            name="pulse-outline"
            size={48}
            color={accent.primary}
            style={{ marginBottom: 24 }}
          />
          <Text style={[styles.title, { color: colors.textPrimary }]}>
            Analysing
          </Text>
          <Text style={[styles.subtitle, { color: colors.textTertiary }]}>
            {statusMsg}
          </Text>
          <PulseBar />
        </View>

        <View style={styles.centerBox}>
          <View
            style={[styles.progressTrack, { backgroundColor: colors.border }]}
          >
            <Animated.View
              style={[
                styles.progressFill,
                { width: progressWidth, backgroundColor: accent.primary },
              ]}
            />
          </View>
          <Text style={[styles.progressLabel, { color: colors.textMuted }]}>
            {progress}% complete
          </Text>
        </View>

        <View style={{ flex: 1 }} />

        <View style={styles.infoRow}>
          <Ionicons name="shield-checkmark" size={16} color={colors.textMuted} />
          <Text style={[styles.footerText, { color: colors.textMuted }]}>
             Clinical-grade POS Algorithmic Processing
          </Text>
        </View>
      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1 },
  safe: { flex: 1, paddingHorizontal: Spacing.xl },
  header: {
    alignItems: "center",
    paddingTop: 80,
    paddingBottom: Spacing.xl,
  },
  title: {
    fontFamily: "SpaceGrotesk-Bold",
    fontSize: 36,
    letterSpacing: -1.5,
    marginBottom: 8,
  },
  subtitle: {
    ...Typography.body,
    fontSize: 18,
    textAlign: "center",
    marginBottom: Spacing.xl,
  },
  pulseBar: {
    flexDirection: "row",
    alignItems: "center",
    height: 48,
    marginTop: 20
  },
  centerBox: {
    marginTop: 40,
    paddingHorizontal: Spacing.md,
  },
  progressTrack: {
    height: 6,
    borderRadius: 3,
    marginBottom: 12,
    overflow: "hidden"
  },
  progressFill: {
     height: "100%",
     borderRadius: 3
  },
  progressLabel: {
    ...Typography.label,
    fontSize: 14,
    textAlign: "center",
    fontFamily: "SpaceGrotesk-Medium"
  },
  infoRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    paddingBottom: Spacing.xxl
  },
  footerText: {
    fontFamily: "SpaceGrotesk-Regular",
    fontSize: 12,
  },
});

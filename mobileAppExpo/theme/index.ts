// src/theme/index.ts
// CardioVision — Monochrome Design System with Blue Accents
// Supports Light + Dark mode via ThemeProvider

import React, { createContext, useContext, useState, useMemo } from "react";

// ── Blue accent palette ──
export const Accent = {
  primary: "#3944BC",
  dark: "#2C36A0",
  light: "#5B64D4",
  subtle: "#E8E9F8",
  ghost: "#F0F1FA",
  text: "#3944BC",
};

// ── Light mode palette ──
interface ThemeColorSet {
  background: string;
  surface: string;
  surfaceHigh: string;
  surfaceDark: string;
  border: string;
  borderLight: string;
  borderDark: string;
  textPrimary: string;
  textSecondary: string;
  textTertiary: string;
  textMuted: string;
  card: string;
  cardBorder: string;
  statusBar: "dark" | "light";
  isDark: boolean;
  glass: string;
  glassBorder: string;
}

const LightColors: ThemeColorSet = {
  background: "#F7F7F8",
  surface: "#FFFFFF",
  surfaceHigh: "#F0F0F2",
  surfaceDark: "#E8E8EC",
  border: "#E5E7EB",
  borderLight: "#F0F0F2",
  borderDark: "#D1D5DB",
  textPrimary: "#111111",
  textSecondary: "#6B7280",
  textTertiary: "#9CA3AF",
  textMuted: "#C5C9D0",
  card: "#FFFFFF",
  cardBorder: "#E5E7EB",
  statusBar: "dark",
  isDark: false,
  glass: "rgba(255,255,255,0.8)",
  glassBorder: "rgba(0,0,0,0.06)",
};

// ── Dark mode palette ──
const DarkColors: ThemeColorSet = {
  background: "#0A0A0A",
  surface: "#161616",
  surfaceHigh: "#1E1E1E",
  surfaceDark: "#121212",
  border: "#2A2A2A",
  borderLight: "#333333",
  borderDark: "#222222",
  textPrimary: "#F5F5F5",
  textSecondary: "#A0A0A0",
  textTertiary: "#666666",
  textMuted: "#444444",
  card: "#161616",
  cardBorder: "#2A2A2A",
  statusBar: "light",
  isDark: true,
  glass: "rgba(0,0,0,0.6)",
  glassBorder: "rgba(255,255,255,0.08)",
};

type ThemeColors = ThemeColorSet;

// ── Theme Context ──
interface ThemeContextType {
  colors: ThemeColors;
  accent: typeof Accent;
  isDark: boolean;
  toggle: () => void;
}

const ThemeContext = createContext<ThemeContextType>({
  colors: LightColors,
  accent: Accent,
  isDark: false,
  toggle: () => {},
});

export const useTheme = () => useContext(ThemeContext);

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [isDark, setIsDark] = useState(false);
  const value = useMemo(
    () => ({
      colors: isDark ? DarkColors : LightColors,
      accent: Accent,
      isDark,
      toggle: () => setIsDark((d) => !d),
    }),
    [isDark],
  );
  return React.createElement(ThemeContext.Provider, { value }, children);
}

// ── Typography ──
export const Typography = {
  displayXL: {
    fontFamily: "SpaceGrotesk-Bold",
    fontSize: 80,
    lineHeight: 80,
    letterSpacing: -3,
  },
  displayL: {
    fontFamily: "SpaceGrotesk-Bold",
    fontSize: 56,
    lineHeight: 56,
    letterSpacing: -2,
  },
  displayM: {
    fontFamily: "SpaceGrotesk-SemiBold",
    fontSize: 36,
    lineHeight: 40,
    letterSpacing: -1,
  },
  h1: {
    fontFamily: "SpaceGrotesk-Bold",
    fontSize: 28,
    lineHeight: 34,
    letterSpacing: -0.5,
  },
  h2: {
    fontFamily: "SpaceGrotesk-SemiBold",
    fontSize: 22,
    lineHeight: 28,
    letterSpacing: -0.3,
  },
  h3: { fontFamily: "SpaceGrotesk-Medium", fontSize: 17, lineHeight: 22 },
  body: { fontFamily: "SpaceGrotesk-Regular", fontSize: 15, lineHeight: 22 },
  bodySmall: {
    fontFamily: "SpaceGrotesk-Regular",
    fontSize: 13,
    lineHeight: 18,
  },
  label: {
    fontFamily: "SpaceGrotesk-Medium",
    fontSize: 11,
    lineHeight: 14,
    letterSpacing: 1.2,
    textTransform: "uppercase" as const,
  },
  mono: {
    fontFamily: "SpaceGrotesk-Regular",
    fontSize: 13,
    letterSpacing: 0.5,
  },
};

export const Spacing = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  xxl: 48,
  xxxl: 64,
};
export const Radius = { sm: 8, md: 12, lg: 20, xl: 28, full: 999 };

export const Shadows = {
  card: {
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.06,
    shadowRadius: 12,
    elevation: 3,
  },
  cardMd: {
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.08,
    shadowRadius: 16,
    elevation: 5,
  },
  soft: {
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.04,
    shadowRadius: 6,
    elevation: 2,
  },
};

// ── Health tips (hardcoded as discussed) ──
export const HealthTipsData = {
  highStress: [
    {
      icon: "fitness-outline",
      title: "Box Breathing",
      subtitle: "Inhale 4s · Hold 4s · Exhale 4s · Hold 4s",
      detail:
        "Box breathing activates your parasympathetic nervous system, lowering cortisol and heart rate within 2 minutes.",
      duration: "4 min",
      urgency: "high",
    },
    {
      icon: "water-outline",
      title: "Hydrate Now",
      subtitle: "Drink 250–400ml of water",
      detail: "Dehydration increases heart rate by 2–8 BPM.",
      duration: "Now",
      urgency: "high",
    },
    {
      icon: "walk-outline",
      title: "5-Minute Walk",
      subtitle: "Step away from screens",
      detail:
        "Brief ambulation reduces sympathetic nervous system activation by up to 23%.",
      duration: "5 min",
      urgency: "high",
    },
  ],
  medStress: [
    {
      icon: "snow-outline",
      title: "Cold Water Splash",
      subtitle: "Splash cold water on face & wrists",
      detail:
        "Activates the dive reflex — your heart rate drops within 30 seconds.",
      duration: "1 min",
      urgency: "med",
    },
    {
      icon: "musical-notes-outline",
      title: "Slow Music",
      subtitle: "60 BPM music entrains your heart",
      detail:
        "Listening to music at 60 BPM causes your heart rate to synchronize downward.",
      duration: "10 min",
      urgency: "med",
    },
  ],
  lowStress: [
    {
      icon: "checkmark-circle-outline",
      title: "Your vitals look good",
      subtitle: "Keep your current rhythm",
      detail:
        "Low LF/HF ratio and strong RMSSD indicate healthy autonomic balance.",
      duration: "Ongoing",
      urgency: "low",
    },
    {
      icon: "bicycle-outline",
      title: "Zone 2 Cardio",
      subtitle: "Sustain elevated HRV long-term",
      detail:
        "Consistent moderate-intensity cardio increases RMSSD by 15–25% over 4 weeks.",
      duration: "30 min/day",
      urgency: "low",
    },
  ],
  highBPM: [
    {
      icon: "thermometer-outline",
      title: "Cool the Body",
      subtitle: "Lower ambient temperature reduces HR",
      detail:
        "A room at 18–20°C naturally decreases resting heart rate by 3–5 BPM.",
      duration: "Ongoing",
      urgency: "high",
    },
  ],
  general: [
    {
      icon: "heart-outline",
      title: "Measure Again in 10 min",
      subtitle: "Confirm your baseline",
      detail:
        "Taking 3 measurements across 30 minutes gives a clinically meaningful average.",
      duration: "30 min",
      urgency: "low",
    },
    {
      icon: "stats-chart-outline",
      title: "Track Over Time",
      subtitle: "HRV trends matter more than values",
      detail: "Daily morning measurements reveal true HRV trends.",
      duration: "Daily",
      urgency: "low",
    },
  ],
};

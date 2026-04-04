// src/theme/index.ts
// CardioVision — Clinical Noir Design System
// Monochromatic palette with surgical precision

export const Colors = {
  // Core monochromatic scale
  black:      '#000000',
  void:       '#080808',
  obsidian:   '#111111',
  charcoal:   '#1C1C1C',
  graphite:   '#2A2A2A',
  iron:       '#3D3D3D',
  ash:        '#555555',
  smoke:      '#777777',
  silver:     '#999999',
  fog:        '#BBBBBB',
  mist:       '#D4D4D4',
  ghost:      '#E8E8E8',
  snow:       '#F5F5F5',
  white:      '#FFFFFF',

  // Semantic
  background:   '#080808',
  surface:      '#111111',
  surfaceHigh:  '#1C1C1C',
  border:       '#2A2A2A',
  borderLight:  '#3D3D3D',

  // Text
  textPrimary:   '#FFFFFF',
  textSecondary: '#999999',
  textTertiary:  '#555555',
  textMuted:     '#3D3D3D',

  // Status accents (minimal, surgical)
  pulse:         '#FFFFFF',       // pure white for BPM
  reliable:      '#E8E8E8',
  warning:       '#BBBBBB',
  danger:        '#777777',

  // Stress levels (monochromatic gradient)
  stressLow:    '#D4D4D4',
  stressMed:    '#888888',
  stressHigh:   '#444444',

  // Mode badges
  biometricBg:  '#1C1C1C',
  visualBg:     '#2A2A2A',

  // Waveform
  wave:         '#FFFFFF',
  waveGlow:     'rgba(255,255,255,0.08)',
  waveSubtle:   'rgba(255,255,255,0.15)',

  // Glass
  glass:        'rgba(255,255,255,0.04)',
  glassBorder:  'rgba(255,255,255,0.08)',
  glassMed:     'rgba(255,255,255,0.07)',
};

export const Typography = {
  // Display — massive numerics for vitals
  displayXL: {
    fontFamily: 'SpaceGrotesk-Bold',
    fontSize: 88,
    lineHeight: 88,
    letterSpacing: -4,
    color: Colors.textPrimary,
  },
  displayL: {
    fontFamily: 'SpaceGrotesk-Bold',
    fontSize: 56,
    lineHeight: 56,
    letterSpacing: -2,
    color: Colors.textPrimary,
  },
  displayM: {
    fontFamily: 'SpaceGrotesk-SemiBold',
    fontSize: 36,
    lineHeight: 40,
    letterSpacing: -1,
    color: Colors.textPrimary,
  },

  // Headings
  h1: {
    fontFamily: 'SpaceGrotesk-Bold',
    fontSize: 28,
    lineHeight: 34,
    letterSpacing: -0.5,
    color: Colors.textPrimary,
  },
  h2: {
    fontFamily: 'SpaceGrotesk-SemiBold',
    fontSize: 22,
    lineHeight: 28,
    letterSpacing: -0.3,
    color: Colors.textPrimary,
  },
  h3: {
    fontFamily: 'SpaceGrotesk-Medium',
    fontSize: 17,
    lineHeight: 22,
    color: Colors.textPrimary,
  },

  // Body
  body: {
    fontFamily: 'SpaceGrotesk-Regular',
    fontSize: 15,
    lineHeight: 22,
    color: Colors.textSecondary,
  },
  bodySmall: {
    fontFamily: 'SpaceGrotesk-Regular',
    fontSize: 13,
    lineHeight: 18,
    color: Colors.textSecondary,
  },

  // Label
  label: {
    fontFamily: 'SpaceGrotesk-Medium',
    fontSize: 11,
    lineHeight: 14,
    letterSpacing: 1.2,
    textTransform: 'uppercase' as const,
    color: Colors.textTertiary,
  },
  labelBright: {
    fontFamily: 'SpaceGrotesk-Medium',
    fontSize: 11,
    lineHeight: 14,
    letterSpacing: 1.2,
    textTransform: 'uppercase' as const,
    color: Colors.textSecondary,
  },

  // Mono — for data/timestamps
  mono: {
    fontFamily: 'SpaceGrotesk-Regular',
    fontSize: 13,
    letterSpacing: 0.5,
    color: Colors.textSecondary,
  },
};

export const Spacing = {
  xs:  4,
  sm:  8,
  md:  16,
  lg:  24,
  xl:  32,
  xxl: 48,
  xxxl: 64,
};

export const Radius = {
  sm:  8,
  md:  12,
  lg:  20,
  xl:  28,
  full: 999,
};

export const Shadows = {
  glow: {
    shadowColor: '#FFFFFF',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.12,
    shadowRadius: 20,
    elevation: 8,
  },
  soft: {
    shadowColor: '#000000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.4,
    shadowRadius: 12,
    elevation: 6,
  },
};

// Health tips data — contextual based on stress/BPM
export const HealthTipsData = {
  highStress: [
    {
      icon: '🫁',
      title: 'Box Breathing',
      subtitle: 'Inhale 4s · Hold 4s · Exhale 4s · Hold 4s',
      detail: 'Box breathing activates your parasympathetic nervous system, lowering cortisol and heart rate within 2 minutes.',
      duration: '4 min',
      urgency: 'high',
    },
    {
      icon: '💧',
      title: 'Hydrate Now',
      subtitle: 'Drink 250–400ml of water',
      detail: 'Dehydration increases heart rate by 2–8 BPM and elevates perceived stress. Your blood volume drops, forcing the heart to work harder.',
      duration: 'Immediate',
      urgency: 'high',
    },
    {
      icon: '🚶',
      title: '5-Minute Walk',
      subtitle: 'Step away from screens',
      detail: 'Brief ambulation reduces sympathetic nervous system activation by up to 23%. Even slow walking resets cortisol rhythms.',
      duration: '5 min',
      urgency: 'high',
    },
    {
      icon: '🧘',
      title: 'Progressive Relaxation',
      subtitle: 'Tense and release muscle groups',
      detail: 'Starting from your toes upward, tense each muscle group for 5 seconds then release. Reduces systolic blood pressure measurably.',
      duration: '8 min',
      urgency: 'med',
    },
  ],
  medStress: [
    {
      icon: '☀️',
      title: 'Cold Water Splash',
      subtitle: 'Splash cold water on face & wrists',
      detail: 'Activates the dive reflex — your heart rate drops within 30 seconds. Effective reset for elevated HRV imbalance.',
      duration: '1 min',
      urgency: 'med',
    },
    {
      icon: '🎵',
      title: 'Slow Music',
      subtitle: '60 BPM music entrains your heart',
      detail: 'Listening to music at 60 BPM causes your heart rate to synchronize downward via cardiac entrainment — measurable on rPPG.',
      duration: '10 min',
      urgency: 'med',
    },
    {
      icon: '🌿',
      title: 'Step Outside',
      subtitle: 'Natural light & fresh air',
      detail: '5 minutes of natural light exposure resets your circadian HRV rhythm and drops cortisol by up to 12%.',
      duration: '5 min',
      urgency: 'low',
    },
  ],
  lowStress: [
    {
      icon: '✅',
      title: 'Your vitals look good',
      subtitle: 'Keep your current rhythm',
      detail: 'Low LF/HF ratio and strong RMSSD indicate healthy autonomic balance. Maintain your current sleep and exercise habits.',
      duration: 'Ongoing',
      urgency: 'low',
    },
    {
      icon: '🏃',
      title: 'Zone 2 Cardio',
      subtitle: 'Sustain elevated HRV long-term',
      detail: 'Consistent moderate-intensity cardio (60–70% max HR) increases RMSSD by 15–25% over 4 weeks.',
      duration: '30 min/day',
      urgency: 'low',
    },
    {
      icon: '😴',
      title: 'Sleep Optimization',
      subtitle: 'Protect your HRV overnight',
      detail: 'HRV peaks during deep sleep stages. Consistent sleep timing (±30 min) increases nocturnal RMSSD by up to 18%.',
      duration: '7–9 hrs',
      urgency: 'low',
    },
  ],
  highBPM: [
    {
      icon: '🧊',
      title: 'Cool the Body',
      subtitle: 'Lower ambient temperature reduces HR',
      detail: 'A room at 18–20°C naturally decreases resting heart rate by 3–5 BPM vs 25°C environments.',
      duration: 'Ongoing',
      urgency: 'high',
    },
    {
      icon: '🚫',
      title: 'Avoid Stimulants',
      subtitle: 'No caffeine for 4 hours',
      detail: 'Caffeine blocks adenosine receptors, increasing HR by 5–15 BPM for 4–6 hours. Switch to herbal tea or water.',
      duration: 'Now',
      urgency: 'high',
    },
  ],
  general: [
    {
      icon: '🫀',
      title: 'Measure Again in 10 min',
      subtitle: 'Confirm your baseline',
      detail: 'Single-session rPPG readings can vary. Taking 3 measurements across 30 minutes gives a clinically meaningful average.',
      duration: '30 min',
      urgency: 'low',
    },
    {
      icon: '📊',
      title: 'Track Over Time',
      subtitle: 'HRV trends matter more than values',
      detail: 'Daily morning measurements before eating or exercise reveal true HRV trends. Week-over-week changes are more meaningful than absolutes.',
      duration: 'Daily',
      urgency: 'low',
    },
  ],
};

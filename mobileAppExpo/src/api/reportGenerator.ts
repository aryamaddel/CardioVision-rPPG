// src/api/reportGenerator.ts — PDF Report Generator (mirrors LaTeX template)
import * as Print from 'expo-print';
import * as Sharing from 'expo-sharing';
import type { RPPGResult } from './rppgService';

// ────────────────────────────────────────────────────────────────────
// Helper Utils
// ────────────────────────────────────────────────────────────────────
const fmt = (v: number | null | undefined, d: number = 1): string =>
  v !== null && v !== undefined && !isNaN(v) ? v.toFixed(d) : '--';

function generateReportId(): string {
  const now = new Date();
  const y = now.getFullYear();
  const m = String(now.getMonth() + 1).padStart(2, '0');
  const d = String(now.getDate()).padStart(2, '0');
  const seq = String(Math.floor(Math.random() * 9000) + 1000);
  return `RPPG-${y}-${m}${d}-${seq}`;
}

function generateSubjectId(): string {
  return `SUB-${String(Math.floor(Math.random() * 9000) + 1000)}`;
}

function formatDate(): string {
  return new Date().toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });
}

function getStressColor(level: string): string {
  switch (level) {
    case 'High': return '#DC2626';
    case 'Medium': return '#D97706';
    case 'Low': return '#16A34A';
    default: return '#6B7280';
  }
}

function confidenceLabel(score: number): string {
  if (score >= 0.7) return 'High';
  if (score >= 0.45) return 'Medium';
  return 'Low';
}

// ────────────────────────────────────────────────────────────────────
// SVG Waveform Generator (for the pulse chart in the report)
// ────────────────────────────────────────────────────────────────────
function generateWaveformSVG(signal: number[], peaks: number[], width = 560, height = 120): string {
  if (!signal || signal.length === 0) return '<p style="text-align:center;color:#999;">No waveform data available</p>';

  const step = Math.max(1, Math.floor(signal.length / 280));
  const data = signal.filter((_, i) => i % step === 0);
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;

  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1)) * width;
    const y = height - ((v - min) / range) * (height - 20) - 10;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });

  const peakDots = peaks
    .filter(p => p % step === 0)
    .map(p => Math.floor(p / step))
    .filter(p => p < data.length)
    .map(p => {
      const x = (p / (data.length - 1)) * width;
      const y = height - ((data[p] - min) / range) * (height - 20) - 10;
      return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="3" fill="#19507D" opacity="0.7"/>`;
    })
    .join('\n');

  return `
    <svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">
      <!-- Grid lines -->
      ${[0.25, 0.5, 0.75].map(f => `<line x1="0" y1="${height * (1 - f)}" x2="${width}" y2="${height * (1 - f)}" stroke="#E5E7EB" stroke-width="0.5"/>`).join('\n')}
      <!-- Filled area -->
      <path d="M${pts.join(' L')} L${width},${height} L0,${height} Z" fill="rgba(25,80,125,0.08)"/>
      <!-- Line -->
      <path d="M${pts.join(' L')}" stroke="#19507D" stroke-width="1.5" fill="none" stroke-linecap="round"/>
      <!-- Peak markers -->
      ${peakDots}
      <!-- Axes -->
      <line x1="0" y1="${height}" x2="${width}" y2="${height}" stroke="#9CA3AF" stroke-width="0.5"/>
      <text x="${width - 30}" y="${height - 4}" font-size="9" fill="#9CA3AF">Time (s)</text>
      <text x="4" y="12" font-size="9" fill="#9CA3AF">Amplitude</text>
    </svg>`;
}

// ────────────────────────────────────────────────────────────────────
// IBI Histogram SVG
// ────────────────────────────────────────────────────────────────────
function generateIBIHistogramSVG(ibi: number[], width = 560, height = 80): string {
  if (!ibi || ibi.length < 2) return '<p style="text-align:center;color:#999;">No IBI data available</p>';

  const min = Math.min(...ibi);
  const max = Math.max(...ibi);
  const range = max - min || 1;
  const barW = Math.max(3, (width / ibi.length) - 2);

  const bars = ibi.map((v, i) => {
    const x = i * (width / ibi.length);
    const barH = ((v - min) / range) * (height - 16) + 8;
    const opacity = (0.25 + ((v - min) / range) * 0.55).toFixed(2);
    return `<rect x="${(x + 1).toFixed(1)}" y="${(height - barH).toFixed(1)}" width="${barW.toFixed(1)}" height="${barH.toFixed(1)}" rx="2" fill="#19507D" opacity="${opacity}"/>`;
  }).join('\n');

  return `
    <svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">
      ${bars}
    </svg>`;
}

// ────────────────────────────────────────────────────────────────────
// Clinical Recommendation Engine
// ────────────────────────────────────────────────────────────────────
function generateRecommendations(result: RPPGResult): string[] {
  const hrv = result.hrv_features ?? ({} as any);
  const recs: string[] = [];

  // Autonomic balance
  const lfhf = hrv.lf_hf_ratio ?? 0;
  if (lfhf > 1.5) {
    recs.push('<b>Autonomic Balance:</b> The LF/HF ratio (' + fmt(lfhf, 2) + ') suggests sympathetic dominance. Deep breathing exercises (4-7-8 pattern) or guided meditation may help restore parasympathetic tone.');
  } else if (lfhf > 1.0) {
    recs.push('<b>Autonomic Balance:</b> The LF/HF ratio (' + fmt(lfhf, 2) + ') is within a mildly elevated range. Regular relaxation practices are recommended.');
  } else {
    recs.push('<b>Autonomic Balance:</b> The LF/HF ratio (' + fmt(lfhf, 2) + ') indicates healthy parasympathetic-sympathetic balance.');
  }

  // Signal quality
  const conf = result.confidence ?? 0;
  if (conf >= 0.7) {
    recs.push('<b>Signal Quality:</b> Signal integrity was optimal (' + Math.round(conf * 100) + '%). Results are considered highly reliable.');
  } else if (conf >= 0.45) {
    recs.push('<b>Signal Quality:</b> Signal quality was moderate (' + Math.round(conf * 100) + '%). Results should be interpreted with some caution. Re-scan in well-lit conditions for improved accuracy.');
  } else {
    recs.push('<b>Signal Quality:</b> Signal quality was below clinical threshold (' + Math.round(conf * 100) + '%). A repeat scan under controlled lighting with minimal motion is strongly recommended.');
  }

  // RMSSD + recovery
  const rmssd = hrv.rmssd_ms ?? 0;
  if (rmssd < 30) {
    recs.push('<b>Recovery:</b> RMSSD (' + fmt(rmssd) + ' ms) is below 30 ms, suggesting limited cardiac recovery capacity. Consider reviewing sleep hygiene, hydration, and physical recovery markers.');
  } else if (rmssd < 50) {
    recs.push('<b>Recovery:</b> RMSSD (' + fmt(rmssd) + ' ms) is in the moderate range. Maintaining regular sleep/wake cycles and moderate exercise is recommended.');
  }

  // Stress
  const stress = hrv.stress_level ?? 'Unknown';
  if (stress === 'High') {
    recs.push('<b>Stress Management:</b> Elevated stress detected. Consider incorporating structured breaks, mindfulness practices, or physical activity into your daily routine.');
  }

  // BPM
  const bpm = result.bpm;
  if (bpm !== null && bpm !== undefined && bpm > 100) {
    recs.push('<b>Heart Rate:</b> Resting heart rate (' + Math.round(bpm) + ' BPM) is elevated. If consistently above 100 BPM at rest, medical consultation is advised.');
  }

  return recs;
}

// ────────────────────────────────────────────────────────────────────
// HTML Report Builder
// ────────────────────────────────────────────────────────────────────
function buildReportHTML(result: RPPGResult): string {
  const hrv = result.hrv_features ?? ({} as any);
  const ibi = result.ibi_ms ?? [];
  const bpm = result.bpm ?? (ibi.length ? Math.round(60000 / (ibi.reduce((a, b) => a + b, 0) / ibi.length)) : null);
  const conf = result.confidence ?? 0;
  const stress = hrv.stress_level ?? 'Unknown';
  const sdnn = hrv.sdnn_ms ?? null;
  const rmssd = hrv.rmssd_ms ?? null;
  const lfhf = hrv.lf_hf_ratio ?? null;
  const meanIBI = ibi.length ? (ibi.reduce((a, b) => a + b, 0) / ibi.length) : null;
  const stressIndex = hrv.stress_index ?? null;

  const reportId = generateReportId();
  const subjectId = generateSubjectId();
  const date = formatDate();
  const stressColor = getStressColor(stress);
  const methodLabel = (result.method_used ?? 'pos').replace('_', '+').toUpperCase();
  const waveformSVG = generateWaveformSVG(result.pulse_signal ?? [], result.peaks_idx ?? []);
  const ibiSVG = generateIBIHistogramSVG(ibi);
  const recommendations = generateRecommendations(result);

  const confDetails = result.confidence_details ?? ({} as any);

  return `<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<style>
  @page { margin: 20mm 18mm; size: A4; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 10pt;
    color: #1F2937;
    line-height: 1.5;
    background: #fff;
  }

  /* ── Header ── */
  .header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 24px; }
  .header-left h1 { font-size: 22pt; color: #19507D; margin-bottom: 2px; font-weight: 700; letter-spacing: -0.5px; }
  .header-left .subtitle { font-size: 11pt; color: #6B7280; font-style: italic; }
  .header-right { text-align: right; font-size: 9pt; color: #6B7280; line-height: 1.8; }
  .header-right b { color: #374151; }

  /* ── Vitals Box ── */
  .vitals-box {
    background: #F0F4F8;
    border: 1.5px solid #19507D;
    border-radius: 8px;
    padding: 18px 24px;
    margin-bottom: 20px;
  }
  .vitals-box .box-title {
    font-size: 10pt;
    font-weight: 700;
    color: #19507D;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 14px;
    padding-bottom: 6px;
    border-bottom: 1px solid rgba(25,80,125,0.15);
  }
  .vitals-grid { display: flex; justify-content: space-between; gap: 12px; }
  .vital-cell { flex: 1; }
  .vital-cell .label { font-size: 9pt; color: #6B7280; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
  .vital-cell .value { font-size: 28pt; font-weight: 700; color: #1F2937; letter-spacing: -1px; }
  .vital-cell .value .unit { font-size: 11pt; font-weight: 400; color: #6B7280; margin-left: 2px; }
  .vital-cell .range { font-size: 8pt; color: #9CA3AF; margin-top: 2px; }

  /* ── Sections ── */
  .section-heading {
    font-size: 12pt;
    font-weight: 700;
    color: #19507D;
    margin: 20px 0 10px 0;
    padding-bottom: 6px;
    border-bottom: 2px solid #19507D;
  }

  /* ── Charts ── */
  .chart-container {
    background: #FAFBFC;
    border: 1px solid #E5E7EB;
    border-radius: 6px;
    padding: 14px;
    margin-bottom: 6px;
    text-align: center;
    overflow: hidden;
  }
  .chart-caption { font-size: 8pt; color: #9CA3AF; font-style: italic; text-align: center; margin-top: 6px; }

  /* ── HRV Table ── */
  .hrv-table { width: 100%; border-collapse: collapse; margin-bottom: 6px; }
  .hrv-table th {
    background: #F5F5F5;
    font-size: 9pt;
    font-weight: 700;
    color: #374151;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 8px 12px;
    text-align: left;
    border-bottom: 1.5px solid #D1D5DB;
  }
  .hrv-table td {
    font-size: 10pt;
    padding: 7px 12px;
    border-bottom: 1px solid #E5E7EB;
    color: #4B5563;
  }
  .hrv-table td:nth-child(2), .hrv-table td:nth-child(4) { font-weight: 600; color: #1F2937; }
  .hrv-table tr:last-child td { border-bottom: none; }

  /* ── Quality List ── */
  .quality-list { list-style: none; padding: 0; }
  .quality-list li {
    padding: 6px 0;
    font-size: 9.5pt;
    color: #4B5563;
    border-bottom: 1px solid #F3F4F6;
  }
  .quality-list li:last-child { border-bottom: none; }
  .quality-list b { color: #1F2937; }
  .quality-bar-track { display: inline-block; width: 120px; height: 5px; background: #E5E7EB; border-radius: 3px; vertical-align: middle; margin: 0 8px; }
  .quality-bar-fill { height: 100%; background: #19507D; border-radius: 3px; }

  /* ── Recommendations ── */
  .recs-box {
    background: #FFF7ED;
    border: 1.5px solid #F59E0B;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 20px;
  }
  .recs-box .box-title {
    font-size: 10pt;
    font-weight: 700;
    color: #B45309;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid rgba(245,158,11,0.25);
  }
  .recs-box ul { padding-left: 16px; }
  .recs-box li { font-size: 9.5pt; color: #4B5563; margin-bottom: 8px; line-height: 1.6; }
  .recs-box li b { color: #92400E; }

  /* ── Footer ── */
  .footer {
    text-align: center;
    font-size: 8pt;
    color: #9CA3AF;
    margin-top: 30px;
    padding-top: 12px;
    border-top: 1px solid #E5E7EB;
  }

  /* ── Scan Meta ── */
  .meta-row { display: flex; gap: 16px; font-size: 8.5pt; color: #9CA3AF; margin-bottom: 16px; justify-content: center; }
  .meta-row span b { color: #6B7280; }
</style>
</head>
<body>

<!-- ═══════════════ HEADER ═══════════════ -->
<div class="header">
  <div class="header-left">
    <h1>Physiological Assessment</h1>
    <span class="subtitle">Remote PPG Analysis Report</span>
  </div>
  <div class="header-right">
    <b>Date:</b> ${date}<br/>
    <b>Report ID:</b> ${reportId}<br/>
    <b>Subject ID:</b> ${subjectId}
  </div>
</div>

<!-- ═══════════════ CORE VITALS ═══════════════ -->
<div class="vitals-box">
  <div class="box-title">Core Vital Metrics</div>
  <div class="vitals-grid">
    <div class="vital-cell">
      <div class="label">Heart Rate (BPM)</div>
      <div class="value">${bpm !== null ? Math.round(bpm) : '--'}<span class="unit">bpm</span></div>
      <div class="range">${bpm !== null ? `Range: ${Math.max(0, Math.round(bpm! - 4))}-${Math.round(bpm! + 4)}` : 'N/A'}</div>
    </div>
    <div class="vital-cell">
      <div class="label">HRV (Overall)</div>
      <div class="value">${fmt(sdnn, 0)}<span class="unit">ms</span></div>
      <div class="range">Avg. Variability: ${fmt(rmssd, 0)}ms</div>
    </div>
    <div class="vital-cell">
      <div class="label">Stress Level</div>
      <div class="value" style="color:${stressColor}">${stress}</div>
      <div class="range">Index: ${stressIndex !== null ? fmt(stressIndex, 0) : '--'}/100</div>
    </div>
  </div>
</div>

<!-- ═══════════════ SCAN META ═══════════════ -->
<div class="meta-row">
  <span><b>Method:</b> ${methodLabel}</span>
  <span><b>Duration:</b> ${fmt(result.duration_sec, 1)}s</span>
  <span><b>Frames:</b> ${result.frames_processed ?? result.n_frames ?? '--'}</span>
  <span><b>FPS:</b> ${fmt(result.fps, 1)}</span>
  <span><b>Confidence:</b> ${Math.round(conf * 100)}% (${confidenceLabel(conf)})</span>
</div>

<!-- ═══════════════ PULSE WAVEFORM ═══════════════ -->
<div class="section-heading">rPPG Pulse Waveform Analysis</div>
<div class="chart-container">
  ${waveformSVG}
</div>
<div class="chart-caption">Figure 1: Filtered rPPG signal extracted via facial blood volume pulse (BVP) analysis using ${methodLabel} method.</div>

<!-- ═══════════════ IBI HISTOGRAM ═══════════════ -->
<div class="section-heading">Inter-Beat Interval Distribution</div>
<div class="chart-container">
  ${ibiSVG}
</div>
<div class="chart-caption">Figure 2: Beat-to-beat interval distribution showing cardiac rhythm regularity across ${ibi.length} detected intervals.</div>

<!-- ═══════════════ HRV TABLE ═══════════════ -->
<div class="section-heading">HRV Matrix (Heart Rate Variability)</div>
<table class="hrv-table">
  <tr>
    <th>Metric</th><th>Value</th><th>Metric</th><th>Value</th>
  </tr>
  <tr>
    <td>RMSSD</td><td>${fmt(rmssd)} ms</td>
    <td>Mean IBI</td><td>${meanIBI !== null ? fmt(meanIBI, 0) : '--'} ms</td>
  </tr>
  <tr>
    <td>SDNN</td><td>${fmt(sdnn)} ms</td>
    <td>LF/HF Ratio</td><td>${fmt(lfhf, 2)}</td>
  </tr>
  <tr>
    <td>Stress Level Index</td><td>${stressIndex !== null ? fmt(stressIndex, 0) : '--'} (${stress})</td>
    <td>Motion Fraction</td><td>${fmt(result.motion_fraction !== undefined ? result.motion_fraction * 100 : null, 1)}%</td>
  </tr>
</table>

<!-- ═══════════════ SIGNAL QUALITY ═══════════════ -->
<div class="section-heading">Signal Integrity & Quality Breakdown</div>
<ul class="quality-list">
  <li>
    <b>IBI Regularity:</b>
    <span class="quality-bar-track"><span class="quality-bar-fill" style="width:${Math.round((confDetails.ibi_regularity ?? 0) * 100)}%"></span></span>
    ${Math.round((confDetails.ibi_regularity ?? 0) * 100)}% — ${(confDetails.ibi_regularity ?? 0) >= 0.8 ? 'High stability in inter-beat intervals' : 'Moderate inter-beat interval consistency'}
  </li>
  <li>
    <b>Spectral SNR:</b>
    <span class="quality-bar-track"><span class="quality-bar-fill" style="width:${Math.round((confDetails.snr ?? 0) * 100)}%"></span></span>
    ${Math.round((confDetails.snr ?? 0) * 100)}% — ${(confDetails.snr ?? 0) >= 0.6 ? 'Signal-to-noise ratio within clinical threshold' : 'Moderate spectral signal quality'}
  </li>
  <li>
    <b>Peak Density:</b>
    <span class="quality-bar-track"><span class="quality-bar-fill" style="width:${Math.round((confDetails.density ?? 0) * 100)}%"></span></span>
    ${Math.round((confDetails.density ?? 0) * 100)}% — ${(confDetails.density ?? 0) >= 0.8 ? 'Consistent peak detection' : 'Adequate peak detection rate'}
  </li>
  <li>
    <b>Data Completeness:</b>
    <span class="quality-bar-track"><span class="quality-bar-fill" style="width:${Math.round((confDetails.duration ?? 0) * 100)}%"></span></span>
    ${Math.round((confDetails.duration ?? 0) * 100)}% — ${(confDetails.duration ?? 0) >= 0.9 ? 'Minimal packet loss or facial occlusion' : 'Some data gaps detected'}
  </li>
</ul>

<!-- ═══════════════ RECOMMENDATIONS ═══════════════ -->
<div class="recs-box">
  <div class="box-title">Clinical Recommendations</div>
  <ul>
    ${recommendations.map(r => `<li>${r}</li>`).join('\n    ')}
  </ul>
</div>

<!-- ═══════════════ FOOTER ═══════════════ -->
<div class="footer">
  This report is generated via automated rPPG analysis using CardioVision.<br/>
  For clinical diagnosis, please consult a certified medical professional.<br/>
  UBFC-rPPG validated · PhysFormer + POS ensemble · Report generated ${date}
</div>

</body>
</html>`;
}

// ────────────────────────────────────────────────────────────────────
// Public API
// ────────────────────────────────────────────────────────────────────

/**
 * Generate a PDF from the rPPG result and share it.
 * Uses expo-print to render HTML → PDF and expo-sharing to present the share sheet.
 */
export async function generateAndShareReport(result: RPPGResult): Promise<void> {
  const html = buildReportHTML(result);

  const { uri } = await Print.printToFileAsync({
    html,
    width: 595,   // A4 width in points
    height: 842,  // A4 height in points
  });

  // Rename to a descriptive filename
  const newUri = uri.replace(/\/([^/]+)\.pdf$/i, '/CardioVision_Report.pdf');
  try {
    // Try to rename the file for a nicer filename in share sheet
    const FileSystem = require('expo-file-system');
    await FileSystem.moveAsync({ from: uri, to: newUri });
    await Sharing.shareAsync(newUri, {
      mimeType: 'application/pdf',
      dialogTitle: 'Share CardioVision Report',
      UTI: 'com.adobe.pdf',
    });
  } catch {
    // Fallback: share with original filename
    await Sharing.shareAsync(uri, {
      mimeType: 'application/pdf',
      dialogTitle: 'Share CardioVision Report',
      UTI: 'com.adobe.pdf',
    });
  }
}

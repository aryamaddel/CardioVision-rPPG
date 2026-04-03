import { View, Text, TouchableOpacity, StyleSheet, ScrollView } from 'react-native';
import { useRouter, useLocalSearchParams } from 'expo-router';

export default function ResultsScreen() {
  const router = useRouter();
  const { resultJson } = useLocalSearchParams();
  
  let data;
  try {
    data = JSON.parse((resultJson as string) || "{}");
  } catch (e) {
    data = {};
  }

  const { bpm = 0, rmssd = 0, sdnn = 0, lf_hf = 0, confidence = 0, stress_level = 'Unknown' } = data;

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.header}>Analysis Complete</Text>
      
      <View style={styles.card}>
        <Metric title="Heart Rate" value={`${bpm} BPM`} />
        <Metric title="RMSSD (HRV)" value={`${rmssd} ms`} />
        <Metric title="SDNN" value={`${sdnn} ms`} />
        <Metric title="LF/HF Ratio" value={`${lf_hf}`} />
        <Metric title="Confidence" value={`${(confidence * 100).toFixed(0)}%`} />
        <Metric title="Stress Classification" value={typeof stress_level === 'string' ? stress_level.toUpperCase() : String(stress_level)} />
      </View>
      
      <TouchableOpacity style={styles.button} onPress={() => router.replace('/')}>
        <Text style={styles.buttonText}>New Scan</Text>
      </TouchableOpacity>
    </ScrollView>
  );
}

function Metric({ title, value }: { title: string, value: string }) {
  return (
    <View style={styles.metricRow}>
      <Text style={styles.metricTitle}>{title}</Text>
      <Text style={styles.metricValue}>{value}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    backgroundColor: '#111827',
    alignItems: 'center',
    padding: 24,
    paddingTop: 80,
  },
  header: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#FFF',
    marginBottom: 32,
  },
  card: {
    backgroundColor: '#1F2937',
    borderRadius: 16,
    width: '100%',
    padding: 16,
    marginBottom: 32,
  },
  metricRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#374151',
  },
  metricTitle: {
    color: '#9CA3AF',
    fontSize: 16,
  },
  metricValue: {
    color: '#F9FAFB',
    fontSize: 16,
    fontWeight: 'bold',
  },
  button: {
    backgroundColor: '#3B82F6',
    paddingVertical: 16,
    paddingHorizontal: 32,
    borderRadius: 8,
    width: '100%',
    alignItems: 'center',
  },
  buttonText: {
    color: '#FFF',
    fontSize: 18,
    fontWeight: 'bold',
  },
});

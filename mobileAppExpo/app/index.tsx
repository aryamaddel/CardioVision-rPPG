import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { useRouter } from 'expo-router';

export default function HomeScreen() {
  const router = useRouter();

  return (
    <View style={styles.container}>
      <Text style={styles.title}>CardioVision</Text>
      <Text style={styles.subtitle}>Remote Photoplethysmography</Text>
      
      <View style={styles.card}>
        <Text style={styles.instruction}>
          We will record a 30-second video of your face to measure your heart rate and heart rate variability without any physical sensors.
        </Text>
        
        <TouchableOpacity style={styles.button} onPress={() => router.push('/camera')}>
          <Text style={styles.buttonText}>Start Scan</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#111827',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 24,
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#F9FAFB',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    color: '#9CA3AF',
    marginBottom: 48,
  },
  card: {
    backgroundColor: '#1F2937',
    padding: 24,
    borderRadius: 16,
    width: '100%',
    alignItems: 'center',
  },
  instruction: {
    color: '#D1D5DB',
    textAlign: 'center',
    fontSize: 16,
    lineHeight: 24,
    marginBottom: 32,
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
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: '600',
  },
});

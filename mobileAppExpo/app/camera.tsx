import React, { useRef, useState, useCallback } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ActivityIndicator } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { useRouter } from 'expo-router';

// Connect to local backend instead of localtunnel for Expo Go
const BACKEND_URL = 'http://10.23.44.193:8000';

export default function CameraScreen() {
  const cameraRef = useRef<CameraView>(null);
  const [permission, requestPermission] = useCameraPermissions();
  const [status, setStatus] = useState<'idle' | 'recording' | 'processing'>('idle');
  const router = useRouter();

  const startRecord = useCallback(async () => {
    if (!cameraRef.current) return;
    setStatus('recording');
    try {
      // Record exactly 30s as per problem statement
      const video = await cameraRef.current.recordAsync({ maxDuration: 30 });
      if (video && video.uri) {
        setStatus('processing');
        await sendToBackend(video.uri);
      } else {
        setStatus('idle');
      }
    } catch (err) {
      console.error(err);
      setStatus('idle');
    }
  }, []);

  const sendToBackend = async (uri: string) => {
    const formData = new FormData();
    formData.append('video', {
      uri,
      name: 'recording.mp4',
      type: 'video/mp4',
    } as any);

    try {
      const response = await fetch(`${BACKEND_URL}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'multipart/form-data' },
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}`);
      }
      
      const payload = await response.json();
      router.push({
        pathname: '/results',
        params: { resultJson: JSON.stringify(payload) }
      });
    } catch (err) {
      console.error(err);
      alert('Error processing video');
      setStatus('idle');
    }
  };

  if (!permission) return <View style={styles.center}><ActivityIndicator color="#fff" /></View>;
  if (!permission.granted) {
    return (
      <View style={styles.center}>
        <Text style={styles.text}>Camera access required.</Text>
        <TouchableOpacity style={styles.btn} onPress={requestPermission}>
          <Text style={styles.btnText}>Allow</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <CameraView 
        ref={cameraRef} 
        mode="video" 
        style={StyleSheet.absoluteFillObject} 
        facing="front"
      />
      
      {/* Simple overlay logic */}
      <View style={styles.overlay}>
        {status === 'idle' && (
          <TouchableOpacity style={styles.recordBtn} onPress={startRecord}>
            <Text style={styles.recordText}>Start 30s Record</Text>
          </TouchableOpacity>
        )}
        
        {status === 'recording' && (
          <View style={styles.statusBox}>
            <Text style={styles.statusText}>🔴 Recording 30s...</Text>
            <Text style={styles.statusSub}>Please remain still.</Text>
          </View>
        )}

        {status === 'processing' && (
          <View style={styles.statusBox}>
            <ActivityIndicator color="#fff" style={{ marginBottom: 8 }} />
            <Text style={styles.statusText}>Processing Video...</Text>
            <Text style={styles.statusSub}>Running rPPG pipeline...</Text>
          </View>
        )}
        
        {status === 'idle' && (
           <TouchableOpacity style={{marginTop: 20}} onPress={() => router.back()}>
             <Text style={{color: '#9CA3AF'}}>Cancel</Text>
           </TouchableOpacity>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  center: { flex: 1, backgroundColor: '#111827', alignItems: 'center', justifyContent: 'center' },
  text: { color: '#FFF', fontSize: 16, marginBottom: 16 },
  btn: { backgroundColor: '#3B82F6', padding: 12, borderRadius: 8 },
  btnText: { color: '#FFF', fontWeight: 'bold' },
  
  overlay: {
    position: 'absolute',
    bottom: 0, left: 0, right: 0,
    backgroundColor: 'rgba(0,0,0,0.6)',
    padding: 32,
    alignItems: 'center',
    borderTopLeftRadius: 16,
    borderTopRightRadius: 16,
  },
  recordBtn: {
    backgroundColor: '#EF4444',
    paddingVertical: 16,
    paddingHorizontal: 32,
    borderRadius: 32,
    width: '100%',
    alignItems: 'center'
  },
  recordText: { color: '#FFF', fontSize: 18, fontWeight: 'bold' },
  
  statusBox: { alignItems: 'center' },
  statusText: { color: '#FFF', fontSize: 18, fontWeight: 'bold', marginBottom: 4 },
  statusSub: { color: '#9CA3AF', fontSize: 14 }
});

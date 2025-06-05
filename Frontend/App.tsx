import { CameraView, useCameraPermissions, CameraCapturedPicture } from 'expo-camera';
import { useEffect, useRef, useState } from 'react';
import {
  SafeAreaView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
  Image,
  ActivityIndicator,
  Dimensions,
} from 'react-native';

export default function App() {
  const [facing, setFacing] = useState<'front' | 'back'>('back');
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraView | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [detections, setDetections] = useState<any[]>([]);
  const [renderedImage, setRenderedImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const API_URL = 'http://192.168.38.130:8000/predict'; // 🔁 Replace with your backend IP

  useEffect(() => {
    if (isStreaming) {
      const interval = setInterval(() => {
        captureAndSendFrame();
      }, 7000); // ⏱️ 7 seconds between frames
      return () => clearInterval(interval);
    }
  }, [isStreaming]);

  if (!permission) return <View />;
  if (!permission.granted) {
    return (
      <SafeAreaView style={styles.container}>
        <Text style={styles.message}>We need your permission to show the camera</Text>
        <TouchableOpacity onPress={requestPermission} style={styles.permissionButton}>
          <Text style={styles.text}>Grant Permission</Text>
        </TouchableOpacity>
      </SafeAreaView>
    );
  }

  const toggleCameraFacing = () => {
    setFacing(prev => (prev === 'back' ? 'front' : 'back'));
  };

  const captureAndSendFrame = async () => {
    if (!cameraRef.current) return;
    try {
      setLoading(true);
      const photo: CameraCapturedPicture = await cameraRef.current.takePictureAsync({
        base64: true,
        quality: 0.5,
        skipMetadata: true,
      });

      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_base64: photo.base64,  // ✅ send correct key
          return_image: true,          // ✅ ask for annotated image
        }),
      });

      const result = await response.json();

      // ✅ Handle new backend response format
      setDetections(result.boxes || []);
      if (result.image_base64) {
        setRenderedImage(`data:image/jpeg;base64,${result.image_base64}`);
      } else {
        setRenderedImage(null);
      }

    } catch (err) {
      console.error('Streaming error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.cameraWrapper}>
        <CameraView
          style={styles.camera}
          facing={facing}
          ref={cameraRef}
          photo={true}
        >
          <View style={styles.overlay}>
            <TouchableOpacity style={styles.topRightButton} onPress={() => setIsStreaming(!isStreaming)}>
              <Text style={styles.text}>{isStreaming ? 'Stop' : 'Start'}</Text>
            </TouchableOpacity>

            <TouchableOpacity style={styles.topLeftButton} onPress={toggleCameraFacing}>
              <Text style={styles.text}>Flip</Text>
            </TouchableOpacity>
          </View>
        </CameraView>
      </View>

      {loading && (
        <View style={styles.loadingRow}>
          <ActivityIndicator />
          <Text>Processing...</Text>
        </View>
      )}

      {renderedImage && (
        <Image source={{ uri: renderedImage }} style={styles.previewImage} />
      )}

      {detections.length > 0 && (
        <View style={styles.results}>
          <Text style={styles.resultHeader}>Detections:</Text>
          {detections.map((d, idx) => (
            <Text key={idx}>{`${d.label}: ${(d.confidence * 100).toFixed(1)}%`}</Text>
          ))}
        </View>
      )}
    </SafeAreaView>
  );
}

const { width } = Dimensions.get('window');

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  cameraWrapper: {
    flex: 1,
    overflow: 'hidden',
  },
  camera: {
    flex: 1,
    borderBottomLeftRadius: 16,
    borderBottomRightRadius: 16,
  },
  overlay: {
    flex: 1,
    position: 'absolute',
    top: 20,
    left: 0,
    right: 0,
    paddingHorizontal: 16,
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  topRightButton: {
    backgroundColor: '#ff4081',
    padding: 10,
    borderRadius: 8,
  },
  topLeftButton: {
    backgroundColor: '#2196F3',
    padding: 10,
    borderRadius: 8,
  },
  text: {
    color: '#fff',
    fontWeight: 'bold',
  },
  message: {
    textAlign: 'center',
    padding: 20,
    fontSize: 16,
  },
  permissionButton: {
    backgroundColor: '#2196F3',
    padding: 10,
    marginHorizontal: 50,
    borderRadius: 8,
    alignItems: 'center',
  },
  loadingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
    marginVertical: 10,
  },
  results: {
    padding: 10,
  },
  resultHeader: {
    fontWeight: 'bold',
    marginBottom: 5,
  },
  previewImage: {
    width: width - 20,
    height: width * 0.75,
    marginHorizontal: 10,
    marginBottom: 10,
    borderRadius: 10,
  },
});

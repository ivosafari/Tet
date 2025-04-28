import cv2
import mediapipe as mp

# Inisialisasi MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def deteksi_wajah(image_path):
    # Membaca gambar
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Inisialisasi deteksi wajah
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(image_rgb)

        # Gambar hasil deteksi wajah
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)
            return image
        else:
            print("Tidak ada wajah terdeteksi.")
            return None
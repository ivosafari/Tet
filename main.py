import cv2
import numpy as np
from deteksi_wajah import deteksi_wajah
from model_qi import buat_model_qi

# Fungsi untuk preprocessing wajah
def preprocess_wajah(image_path):
    # Membaca dan mendeteksi wajah
    detected_face = deteksi_wajah(image_path)

    if detected_face is None:
        return None

    # Resize wajah menjadi ukuran 100x100
    face_resized = cv2.resize(detected_face, (100, 100))
    face_flattened = face_resized.flatten()  # Ubah menjadi array datar
    return face_flattened / 255.0  # Normalisasi nilai pixel

# Main program untuk prediksi "qi"
if __name__ == "__main__":
    # Membuat model
    input_shape = (100 * 100 * 3,)  # Input shape untuk gambar berwarna 100x100
    model = buat_model_qi(input_shape)

    # Path gambar
    image_path = "path_to_your_image.jpg"

    # Preprocess dan prediksi
    face_data = preprocess_wajah(image_path)

    if face_data is not None:
        # Prediksi "qi"
        qi_prediction = model.predict(np.array([face_data]))
        print(f"Prediksi 'qi': {qi_prediction[0][0]}")
    else:
        print("Wajah tidak terdeteksi, tidak dapat memprediksi 'qi'.")
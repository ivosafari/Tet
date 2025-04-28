from flask import Flask, request, jsonify
import numpy as np
from main import preprocess_wajah
from model_qi import buat_model_qi

app = Flask(__name__)

# Inisialisasi model
input_shape = (100 * 100 * 3,)
model = buat_model_qi(input_shape)

@app.route('/predict_qi', methods=['POST'])
def predict_qi():
    # Ambil gambar dari permintaan
    image_file = request.files['image']
    image_path = f"./{image_file.filename}"
    image_file.save(image_path)

    # Preprocess gambar
    face_data = preprocess_wajah(image_path)

    if face_data is not None:
        # Prediksi "qi"
        qi_prediction = model.predict(np.array([face_data]))
        return jsonify({'qi': float(qi_prediction[0][0])})
    else:
        return jsonify({'error': 'Wajah tidak terdeteksi'}), 400

if __name__ == "__main__":
    app.run(debug=True)
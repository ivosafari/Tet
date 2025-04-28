import tensorflow as tf
from tensorflow.keras import layers, models

# Membuat model neural network untuk menghasilkan "qi"
def buat_model_qi(input_shape):
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=input_shape),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Output untuk "qi"
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
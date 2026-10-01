import numpy as np
import pandas as pd
from tensorflow.keras import layers, models

# facial recognition model
df = pd.read_csv('fer2013.csv')

X_train_face, y_train_face, X_test_face, y_test_face = [], [], [], []
for index, row in df.iterrows():
    pixels = np.asarray(list(row['pixels'].split(' ')), dtype=np.uint8)
    if row['Usage'] == 'Training':
        X_train_face.append(pixels)
        y_train_face.append(row['emotion'])
    elif row['Usage'] in ['PublicTest', 'PrivateTest']:
        X_test_face.append(pixels)
        y_test_face.append(row['emotion'])

X_train_face = np.array(X_train_face).reshape(-1, 48, 48, 1) / 255.0
X_test_face = np.array(X_test_face).reshape(-1, 48, 48, 1) / 255.0
y_train_face = np.array(y_train_face)
y_test_face = np.array(y_test_face)

face_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(7, activation='softmax')
])

face_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

face_model.fit(X_train_face, y_train_face, epochs=10, batch_size=64, validation_data=(X_test_face, y_test_face))
face_model.save('emotion_detection_model.h5')


# Voice emotion model architecture

def build_voice_emotion_model(input_shape=(40,)):
    """Creates a Dense NN architecture for classifying 40-feature MFCC voice vectors into 7 emotions."""
    model = models.Sequential([
        layers.Dense(256, activation='relu', input_shape=input_shape),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(7, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model



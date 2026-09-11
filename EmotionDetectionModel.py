import numpy as np
import pandas as pd
from tensorflow.keras import layers, models

# Load FER2013 dataset
df = pd.read_csv('fer2013.csv')

# Preprocess the data
X_train, y_train, X_test, y_test = [], [], [], []
for index, row in df.iterrows():
    pixels = np.asarray(list(row['pixels'].split(' ')), dtype=np.uint8)
    if row['Usage'] == 'Training':
        X_train.append(pixels)
        y_train.append(row['emotion'])
    elif row['Usage'] == 'PublicTest' or row['Usage'] == 'PrivateTest':
        X_test.append(pixels)
        y_test.append(row['emotion'])

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Reshape and normalize the data
X_train = X_train.reshape(-1, 48, 48, 1) / 255.0
X_test = X_test.reshape(-1, 48, 48, 1) / 255.0

# Define the model architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(7, activation='softmax')  # 7 classes for 7 emotions
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# Save the model as .h5 file
model.save('emotion_detection_model.h5')
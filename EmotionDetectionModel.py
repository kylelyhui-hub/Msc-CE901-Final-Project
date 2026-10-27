import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

print("--- Training Facial Emotion Recognition Model ---", flush=True)

# load FER2013 Dataset
csv_file = 'fer2013.csv'
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"Could not find '{csv_file}'. Ensure it is in your project root directory.")

df = pd.read_csv(csv_file)
print(f"[DATASET] Loaded '{csv_file}' with {len(df)} total rows.", flush=True)

X_train_face, y_train_face, X_test_face, y_test_face = [], [], [], []

for index, row in df.iterrows():
    pixels = np.asarray(list(row['pixels'].split(' ')), dtype=np.uint8)
    if row['Usage'] == 'Training':
        X_train_face.append(pixels)
        y_train_face.append(row['emotion'])
    elif row['Usage'] in ['PublicTest', 'PrivateTest']:
        X_test_face.append(pixels)
        y_test_face.append(row['emotion'])

X_train_face = np.array(X_train_face, dtype=np.float32).reshape(-1, 48, 48, 1) / 255.0
X_test_face = np.array(X_test_face, dtype=np.float32).reshape(-1, 48, 48, 1) / 255.0
y_train_face = np.array(y_train_face, dtype=np.int32)
y_test_face = np.array(y_test_face, dtype=np.int32)

print(f"[SPLIT] Training samples: {len(X_train_face)}, Testing samples: {len(X_test_face)}", flush=True)

# High-performance tf.data Pipeline
BATCH_SIZE = 256

train_dataset = tf.data.Dataset.from_tensor_slices((X_train_face, y_train_face))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test_face, y_test_face))
test_dataset = test_dataset.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)


#  Define 4 different CNN Architectures
def build_model_1_standard():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(7, activation='softmax')
    ])
    return model


def build_model_2_deep_bn():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(48, 48, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(7, activation='softmax')
    ])
    return model


def build_model_3_wide():
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(7, activation='softmax')
    ])
    return model


def build_model_4_compact():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(48, 48, 1)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(7, activation='softmax')
    ])
    return model


architectures = {
    "Standard CNN": build_model_1_standard(),
    "Deep CNN with BatchNorm": build_model_2_deep_bn(),
    "Wide CNN": build_model_3_wide(),
    "Compact Dual-Block CNN": build_model_4_compact()
}

best_model_name = None
best_model = None
best_accuracy = 0.0
comparison_results = {}


print("--- Training & Comparing 4 CNN Models ---")


EPOCHS = 8  # Reduced maximum epochs
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=2,  # Stops early if validation loss doesn't improve for 2 epochs
    restore_best_weights=True
)

for name, model in architectures.items():
    print(f"\n[TRAINING] Compiling and training {name}...", flush=True)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=test_dataset,
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluate on test set
    loss, acc = model.evaluate(test_dataset, verbose=0)
    comparison_results[name] = acc
    print(f"[{name}] Test Accuracy: {acc * 100:.2f}%", flush=True)

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name

print("--- FACIAL MODEL COMPARISON ---")

for name, acc in comparison_results.items():
    marker = " (best classifier)" if name == best_model_name else ""
    print(f"  • {name}: {acc * 100:.2f}%{marker}")


print(f"\n[WINNER] Selected best performing facial model: {best_model_name} with {best_accuracy * 100:.2f}% accuracy.",
      flush=True)

# Save the best model
best_model.save('emotion_detection_model.h5')
print("[SUCCESS] Exported best facial model as 'emotion_detection_model.h5' successfully!", flush=True)

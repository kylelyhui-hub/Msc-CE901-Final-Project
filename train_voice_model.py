import ast
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras import layers, models

print("--- Training Voice Emotion Model ---", flush=True)

# 1. Load Dataset
df_voice = pd.read_csv('mood_detection_data_realistic.csv')

def parse_voice_features(row):
    mfcc_list = ast.literal_eval(row['MFCCs'])
    acoustic_feats = [
        float(row['Pitch']),
        float(row['Speech Rate']),
        float(row['Jitter']),
        float(row['Shimmer'])
    ]
    return mfcc_list + acoustic_feats

X_voice = np.array([parse_voice_features(row) for _, row in df_voice.iterrows()])

# 2. Target Label Encoding
voice_label_encoder = LabelEncoder()
y_voice = voice_label_encoder.fit_transform(df_voice['Mood'])
num_classes = len(voice_label_encoder.classes_)

# 3. Scale Features (Crucial to prevent 'Angry' dominance)
scaler = StandardScaler()
X_voice_scaled = scaler.fit_transform(X_voice)

# Save the fitted scaler
joblib.dump(scaler, 'voice_scaler.pkl')

# 4. Train / Test Split
X_train_voice, X_test_voice, y_train_voice, y_test_voice = train_test_split(
    X_voice_scaled, y_voice, test_size=0.2, random_state=42, stratify=y_voice
)

# 5. Build Neural Network Architecture
voice_model = models.Sequential([
    layers.Dense(256, activation='relu', input_shape=(X_voice.shape[1],)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

voice_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Fit Model
voice_model.fit(
    X_train_voice, y_train_voice,
    epochs=25,
    batch_size=32,
    validation_data=(X_test_voice, y_test_voice)
)

# 7. Save Model File
voice_model.save('voice_emotion_model.h5')
print("[SUCCESS] Exported 'voice_emotion_model.h5' and 'voice_scaler.pkl' successfully!", flush=True)
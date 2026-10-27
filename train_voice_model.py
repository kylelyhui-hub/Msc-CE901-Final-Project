import os
import joblib
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

print("--- Training Voice Emotion Model with MELD Dataset ---", flush=True)

# 1. Load MELD Metadata CSV
csv_file = 'train_sent_emo.csv'  # Or path to your MELD split CSV
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"Could not find '{csv_file}'. Ensure it is in your project root directory.")

df = pd.read_csv(csv_file)
print(f"[DATASET] Loaded '{csv_file}' with {len(df)} utterances.", flush=True)


# 2. Normalize MELD Emotions for Driver Safety Monitoring
def normalize_meld_emotion(raw_label):
    e = str(raw_label).lower().strip()
    if e in ['anger', 'angry', 'frustration']:
        return 'Angry'
    elif e in ['disgust']:
        return 'Disgust'
    elif e in ['fear']:
        return 'Fear'
    elif e in ['joy', 'excitement']:
        return 'Happy'
    elif e in ['sadness', 'sad']:
        return 'Sad'
    elif e in ['surprise']:
        return 'Surprise'
    else:
        return 'Neutral'


df['normalized_emotion'] = df['Emotion'].apply(normalize_meld_emotion)


# 3. Rich Acoustic Feature Extraction
def extract_rich_audio_features(file_path):
    try:
        if not os.path.exists(file_path):
            return None
        audio, sample_rate = librosa.load(file_path, duration=2.5, sr=22050)

        # MFCCs (Mean & Std -> 80 features)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfcc_mean = np.mean(mfccs.T, axis=0)
        mfcc_std = np.std(mfccs.T, axis=0)

        # Spectral Contrast & Chroma
        contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sample_rate).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T, axis=0)
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
        rms = np.mean(librosa.feature.rms(y=audio))

        feature_vector = np.concatenate([mfcc_mean, mfcc_std, contrast, chroma, np.array([zcr, rms])])
        return np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as e:
        return None


# 4. Loop Through MELD Utterances to Extract Features
X_list, y_list = [], []
print("[PROCESSING] Extracting acoustic features from MELD audio clips...", flush=True)

for idx, row in df.iterrows():
    # Adjust this path format to match where your MELD audio clips are stored
    # MELD naming convention typically maps to dialogue and utterance IDs
    audio_path = f"audio/dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.wav"

    features = extract_rich_audio_features(audio_path)
    if features is not None:
        X_list.append(features)
        y_list.append(row['normalized_emotion'])

X_voice = np.array(X_list, dtype=np.float32)
y_voice_raw = np.array(y_list)

print(f"[DATASET] Extracted feature matrix shape: {X_voice.shape}", flush=True)

# 5. Encoding, Scaling & Exporting Artifacts
voice_label_encoder = LabelEncoder()
y_voice = voice_label_encoder.fit_transform(y_voice_raw)

scaler = StandardScaler()
X_voice_scaled = scaler.fit_transform(X_voice)

joblib.dump(scaler, 'voice_scaler.pkl')
joblib.dump(voice_label_encoder, 'voice_label_encoder.pkl')

# 6. Train and Compare Classifiers (Random Forest, SVC, Gradient Boosting, MLP)
X_train, X_test, y_train, y_test = train_test_split(
    X_voice_scaled, y_voice, test_size=0.2, random_state=42, stratify=y_voice
)

classifiers = {
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "SVC": SVC(kernel='rbf', probability=True, C=5.0, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
    "MLP Neural Network": MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42)
}

best_model, best_accuracy = None, 0.0
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"[{name}] Accuracy: {acc * 100:.2f}%", flush=True)
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = clf

joblib.dump(best_model, 'voice_emotion_model.pkl')
print(f"[SUCCESS] Saved best model to 'voice_emotion_model.pkl' with {best_accuracy * 100:.2f}% accuracy.", flush=True)

import os
import joblib
import threading
import tkinter as tk
from tkinter import messagebox
import cv2
import librosa
import numpy as np
import requests
import sounddevice as sd
from tensorflow.keras.models import load_model

# Suppress TensorFlow GPU/CPU logging noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


class DriverMonitor:
    def __init__(self):
        # Initialize hidden Tkinter root window
        self.root = tk.Tk()
        self.root.withdraw()

        # Display initial reminder popup
        messagebox.showinfo(
            "Driver Safety Reminder",
            "Please drive safely, follow traffic laws, and fasten your seatbelt at all times."
        )
        self.root.update()

        # Load facial recognition model
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.face_model = load_model('emotion_detection_model.h5')

        # Facial emotion classes
        self.face_emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

        # Voice/Mood emotion classes

        self.voice_emotion_labels = [
            'Angry', 'Bored', 'Calm', 'Confused', 'Excited',
            'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised'
        ]

        # Load voice model trained on mood_detection_data_realistic.csv
        voice_model_path = 'voice_emotion_model.h5'
        if os.path.exists('voice_emotion_model.h5') and os.path.exists('voice_scaler.pkl'):
            self.voice_model = load_model('voice_emotion_model.h5')
            self.voice_scaler = joblib.load('voice_scaler.pkl')
            print("[MODEL SETUP] Voice model & scaler loaded successfully.", flush=True)
        else:
            self.voice_model = None
            self.voice_scaler = None

        # Shared state tracking
        self.current_face_emotion = "Neutral"
        self.current_voice_emotion = "Calibrating Audio..."

        # Vehicle telemetry config
        self.driving_speed = 60  # mph
        self.road_speed_limit = 60  # mph
        self.normal_heart_rate = 75  # BPM
        self.current_heart_rate = 75  # BPM

        # API setup
        self.google_maps_api_key = os.getenv('GOOGLE_MAPS_API_KEY', 'YOUR_FALLBACK_KEY_HERE')
        self.cached_rest_area = None

        # Camera & Audio setup (44100 Hz for standard Windows MME compatibility)
        self.cap = cv2.VideoCapture(0)
        self.sample_rate = 44100
        self.audio_duration = 1.0  # 1-second audio chunks
        self.noise_threshold = 0.015
        self.is_running = True

        # Start background audio loop
        self.audio_thread = threading.Thread(target=self._audio_listener_loop, daemon=True)
        self.audio_thread.start()

    def calibrate_ambient_noise(self, mic_index):
        """Measures background noise level for 1 second at startup."""
        print("[AUDIO SETUP] Calibrating ambient noise floor...", flush=True)
        try:
            recording = sd.rec(
                int(1.0 * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                device=mic_index
            )
            sd.wait()
            rms_noise = np.mean(librosa.feature.rms(y=recording.flatten())[0])
            self.noise_threshold = max(float(rms_noise) * 2.5, 0.01)
            print(f"[AUDIO SETUP] Noise floor calibrated to: {self.noise_threshold:.4f}", flush=True)
        except Exception as e:
            self.noise_threshold = 0.015
            print(f"[AUDIO SETUP] Calibration error ({e}), default threshold set to 0.015", flush=True)

    def extract_voice_features(self, audio_data):
        mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
        mfcc_mean = np.mean(mfccs.T, axis=0)

        pitches, magnitudes = librosa.piptrack(y=audio_data, sr=self.sample_rate)
        pitch_vals = pitches[magnitudes > np.median(magnitudes)]
        pitch_vals = pitch_vals[(pitch_vals > 80) & (pitch_vals < 350)]
        avg_pitch = float(np.mean(pitch_vals)) if len(pitch_vals) > 0 else 150.0

        try:
            onset_env = librosa.onset.onset_strength(y=audio_data, sr=self.sample_rate)
            speech_rate = float(np.sum(onset_env > np.mean(onset_env)))
        except Exception:
            speech_rate = 100.0

        jitter = float(np.std(pitch_vals) / (avg_pitch + 1e-6)) if len(pitch_vals) > 1 else 0.05
        rms_frames = librosa.feature.rms(y=audio_data)[0]
        shimmer = float(np.std(rms_frames) / (np.mean(rms_frames) + 1e-6)) if len(rms_frames) > 1 else 0.05

        feature_vector = np.hstack([mfcc_mean, [avg_pitch, speech_rate, jitter, shimmer]])

        # Standardize live features using the trained scaler
        if self.voice_scaler is not None:
            feature_vector = self.voice_scaler.transform([feature_vector])
        else:
            feature_vector = np.expand_dims(feature_vector, axis=0)

        return feature_vector


    def _audio_listener_loop(self):
        """Continuously records audio in 1-second chunks and runs normalized neural network inference."""
        mic_device_index = 1  # Physical audio jack index

        self.calibrate_ambient_noise(mic_device_index)
        self.current_voice_emotion = "Listening..."

        while self.is_running:
            try:
                audio_data = sd.rec(
                    int(self.audio_duration * self.sample_rate),
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype='float32',
                    device=mic_device_index
                )
                sd.wait()
                audio_data = audio_data.flatten()

                rms_val = np.mean(librosa.feature.rms(y=audio_data)[0])

                if rms_val > self.noise_threshold:
                    if self.voice_model is not None:
                        feats = self.extract_voice_features(audio_data)
                        predictions = self.voice_model.predict(feats, verbose=0)[0]

                        top_confidence = np.max(predictions)
                        predicted_idx = int(np.argmax(predictions))

                        # If prediction confidence is low, fall back to "Neutral"
                        if top_confidence > 0.25:
                            self.current_voice_emotion = self.voice_emotion_labels[predicted_idx]
                        else:
                            self.current_voice_emotion = "Neutral"

                        print(
                            f"[VOICE MODEL DETECTED] Emotion: {self.current_voice_emotion} (Confidence: {top_confidence:.2f})",
                            flush=True)
                    else:
                        self.current_voice_emotion = "Model Missing"
                else:
                    self.current_voice_emotion = "Silent"
            except Exception as e:
                self.current_voice_emotion = "Silent"

    def get_nearest_rest_area(self, lat=50.86285, lng=-0.08785446):
        """Fetch nearest rest area via Google Places API."""
        if self.cached_rest_area is not None:
            return self.cached_rest_area

        try:
            url = f'https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={lat},{lng}&radius=50000&type=rest_area&key={self.google_maps_api_key}'
            response = requests.get(url, timeout=3)
            data = response.json()
            if 'results' in data and data['results']:
                self.cached_rest_area = data['results'][0]['name']
            else:
                self.cached_rest_area = 'No resting areas found nearby'
        except Exception:
            self.cached_rest_area = 'Error fetching rest area'

        return self.cached_rest_area

    def run(self):
        """Main OpenCV webcam loop execution."""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("[CAMERA ERROR] Unable to read video stream.", flush=True)
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    roi_gray = gray[y:y + h, x:x + w]
                    roi_gray_resized = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
                    roi_gray_resized = roi_gray_resized / 255.0

                    roi_input = np.expand_dims(roi_gray_resized, axis=0)
                    roi_input = np.expand_dims(roi_input, axis=-1)

                    predicted_emotion = self.face_model.predict(roi_input, verbose=0)
                    emotion_index = np.argmax(predicted_emotion)
                    self.current_face_emotion = self.face_emotion_labels[emotion_index]

                    print(f"[FACE DETECTED] Facial Emotion: {self.current_face_emotion}", flush=True)

                    cv2.putText(
                        frame, f'Face Emotion: {self.current_face_emotion}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2
                    )

                    if self.current_face_emotion in ['Angry', 'Disgust', 'Fear', 'Sad']:
                        cv2.putText(
                            frame, 'WARNING: Extreme Emotion Detected, consider taking a break.',
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                        )

                        nearest_rest_area = self.get_nearest_rest_area()
                        cv2.putText(
                            frame, f'Nearest Rest Area: {nearest_rest_area}',
                            (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                        )

                # Render Voice Emotion on feed
                cv2.putText(
                    frame, f'Voice Emotion: {self.current_voice_emotion}', (50, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )

                if self.driving_speed > self.road_speed_limit:
                    cv2.putText(
                        frame, f'Speed Limit Exceeded! ({self.driving_speed} mph > {self.road_speed_limit} mph)',
                        (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                    )

                if self.current_heart_rate > self.normal_heart_rate + 20:
                    cv2.putText(
                        frame, 'WARNING: High Heart Rate Detected, consider taking a break.',
                        (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                    )

                cv2.imshow('Driver Multimodal Emotion Detector', frame)

                try:
                    self.root.update()
                except Exception:
                    pass

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.is_running = False
            self.cap.release()
            cv2.destroyAllWindows()
            try:
                self.root.destroy()
            except Exception:
                pass



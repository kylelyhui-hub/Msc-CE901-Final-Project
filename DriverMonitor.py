import os

# Prevent OpenMP/BLAS thread collisions between TensorFlow and Scikit-learn
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import queue
import threading
import tkinter as tk
from tkinter import messagebox
import cv2
import joblib
import librosa
import numpy as np
import requests
import sounddevice as sd
from tensorflow.keras.models import load_model


class DriverMonitor:
    def __init__(self):
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(
            "Driver Safety Reminder",
            "Please drive safely, follow traffic laws, and fasten your seatbelt at all times."
        )
        root.destroy()

        # Load facial recognition model cleanly without ABSL compile warnings
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        if os.path.exists('emotion_detection_model.h5'):
            self.face_model = load_model('emotion_detection_model.h5', compile=False)
        else:
            self.face_model = None
            print("[MODEL WARNING] 'emotion_detection_model.h5' not found.", flush=True)

        # Facial emotion classes
        self.face_emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

        # Default Voice emotion classes fallback to match the 4 target categories
        self.voice_emotion_labels = ['euphoric', 'joy', 'sad', 'surprised']

        # Load scikit-learn voice emotion model
        self.voice_model = None
        self.voice_scaler = None

        if os.path.exists('voice_emotion_model.pkl') and os.path.exists('voice_scaler.pkl'):
            self.voice_model = joblib.load('voice_emotion_model.pkl')
            self.voice_scaler = joblib.load('voice_scaler.pkl')

            if os.path.exists('voice_label_encoder.pkl'):
                encoder = joblib.load('voice_label_encoder.pkl')
                self.voice_emotion_labels = [str(cls) for cls in encoder.classes_]
            print(f"[MODEL SETUP] Loaded Voice Model with classes: {self.voice_emotion_labels}", flush=True)
        else:
            print("[MODEL WARNING] 'voice_emotion_model.pkl' or 'voice_scaler.pkl' not found.", flush=True)

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

        # Camera setup
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("\n[CRITICAL ERROR] Could not open webcam (Index 0).", flush=True)
            print(
                "macOS Privacy Notice: Go to System Settings > Privacy & Security > Camera and ensure your IDE/Terminal has permission.\n",
                flush=True)

        # Automatically detect microphone index, native sample rate, and supported channels
        self.mic_device_index, self.hardware_sample_rate, self.mic_channels = self._get_mac_mic_info()
        self.model_sample_rate = 22050  # Sample rate model was trained on
        self.audio_duration = 1.0  # 1-second evaluation window
        self.noise_threshold = 0.02
        self.is_running = True

        # Thread-safe queue for zero-lag audio streaming
        self.audio_queue = queue.Queue()

        # Start background audio listener thread
        self.audio_thread = threading.Thread(target=self._audio_listener_loop, daemon=True)
        self.audio_thread.start()

    def _get_mac_mic_info(self):
        """Automatically detects active input device, native sample rate, and supported channels."""
        try:
            default_in = sd.default.device[0]
            if default_in is not None and default_in >= 0:
                info = sd.query_devices(default_in, 'input')
                max_ch = info.get('max_input_channels', 0)
                if max_ch > 0:
                    sr = int(info.get('default_samplerate', 44100))
                    channels = 2 if max_ch >= 2 else 1
                    print(
                        f"[AUDIO SETUP] Using default microphone index {default_in} ('{info.get('name')}') at {sr}Hz with {channels} channel(s)",
                        flush=True)
                    return default_in, sr, channels
        except Exception as e:
            print(f"[AUDIO SETUP] Default device query notice: {e}", flush=True)

        for idx, dev in enumerate(sd.query_devices()):
            max_ch = dev.get('max_input_channels', 0)
            if max_ch > 0:
                sr = int(dev.get('default_samplerate', 44100))
                channels = 2 if max_ch >= 2 else 1
                print(
                    f"[AUDIO SETUP] Fallback microphone found index {idx}: '{dev['name']}' at {sr}Hz with {channels} channel(s)",
                    flush=True)
                return idx, sr, channels

        print("[AUDIO ERROR] No valid input microphone found. Defaulting to index 0.", flush=True)
        return 0, 44100, 1

    def calibrate_ambient_noise(self, mic_index):
        """Measures background noise level briefly during startup."""
        print("[AUDIO SETUP] Calibrating ambient noise floor...", flush=True)
        try:
            recording = sd.rec(
                int(1.0 * self.hardware_sample_rate),
                samplerate=self.hardware_sample_rate,
                channels=self.mic_channels,
                dtype='float32',
                device=mic_index
            )
            sd.wait()

            if recording.ndim > 1 and recording.shape[1] > 1:
                mono_recording = np.mean(recording, axis=1)
            else:
                mono_recording = recording.flatten()

            rms_noise = np.mean(librosa.feature.rms(y=mono_recording)[0])
            self.noise_threshold = max(float(rms_noise) * 3.0, 0.015)
            print(f"[AUDIO SETUP] Noise floor calibrated to: {self.noise_threshold:.4f}", flush=True)
        except Exception as e:
            self.noise_threshold = 0.02
            print(f"[AUDIO SETUP] Calibration error ({e}), default threshold set to 0.02", flush=True)

    def extract_voice_features(self, audio_data):
        """Resamples hardware audio and extracts 101 rich acoustic features to match the scaler."""
        if len(audio_data) == 0 or np.all(audio_data == 0):
            return None

        try:
            resampled_audio = librosa.resample(
                y=audio_data,
                orig_sr=self.hardware_sample_rate,
                target_sr=self.model_sample_rate
            )

            # MFCCs (40 coefficients -> Mean & Std -> 80 features)
            mfccs = librosa.feature.mfcc(y=resampled_audio, sr=self.model_sample_rate, n_mfcc=40)
            mfcc_mean = np.mean(mfccs.T, axis=0)
            mfcc_std = np.std(mfccs.T, axis=0)

            # Spectral Contrast (7 features)
            contrast = librosa.feature.spectral_contrast(y=resampled_audio, sr=self.model_sample_rate)
            contrast_mean = np.mean(contrast.T, axis=0)

            # Chroma Stft (12 features)
            chroma = librosa.feature.chroma_stft(y=resampled_audio, sr=self.model_sample_rate)
            chroma_mean = np.mean(chroma.T, axis=0)

            # ZCR and RMS (2 features)
            zcr = np.mean(librosa.feature.zero_crossing_rate(resampled_audio))
            rms = np.mean(librosa.feature.rms(y=resampled_audio))

            # Concatenate into a unified 101-feature vector
            feature_vector = np.concatenate([
                mfcc_mean, mfcc_std, contrast_mean, chroma_mean, np.array([zcr, rms])
            ])

            return np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:
            print(f"[FEATURE EXTRACTION ERROR] {e}", flush=True)
            return None

    def _audio_callback(self, indata, frames, time_info, status):
        """Thread-safe PortAudio callback."""
        if status:
            pass
        if self.is_running:
            try:
                self.audio_queue.put_nowait(indata.copy())
            except queue.Full:
                pass

    def _audio_listener_loop(self):
        """Continuously consumes audio chunks using native PortAudio buffer negotiation."""
        if self.mic_device_index is None:
            print("[AUDIO ERROR] No valid microphone device found.", flush=True)
            self.current_voice_emotion = "No Mic Detected"
            return

        self.calibrate_ambient_noise(self.mic_device_index)
        self.current_voice_emotion = "Listening..."

        chunk_samples = int(self.audio_duration * self.hardware_sample_rate)
        buffer_list = []
        collected_samples = 0

        try:
            with sd.InputStream(
                    device=self.mic_device_index,
                    channels=self.mic_channels,
                    samplerate=self.hardware_sample_rate,
                    callback=self._audio_callback,
                    latency='high',
                    dtype='float32'
            ):
                while self.is_running:
                    try:
                        block = self.audio_queue.get(timeout=0.2)
                    except queue.Empty:
                        continue

                    buffer_list.append(block)
                    collected_samples += len(block)

                    if collected_samples >= chunk_samples:
                        audio_data = np.concatenate(buffer_list, axis=0)[:chunk_samples]

                        buffer_list = []
                        collected_samples = 0

                        if audio_data.ndim > 1 and audio_data.shape[1] > 1:
                            audio_chunk = np.mean(audio_data, axis=1)
                        else:
                            audio_chunk = audio_data.flatten()

                        rms_val = np.mean(librosa.feature.rms(y=audio_chunk)[0])

                        if rms_val > self.noise_threshold:
                            if self.voice_model is not None:
                                raw_feats = self.extract_voice_features(audio_chunk)

                                if raw_feats is not None:
                                    if self.voice_scaler is not None:
                                        scaled_feats = self.voice_scaler.transform([raw_feats])
                                    else:
                                        scaled_feats = np.expand_dims(raw_feats, axis=0)

                                    if hasattr(self.voice_model, "predict_proba"):
                                        predictions = self.voice_model.predict_proba(scaled_feats)[0]

                                        temperature = 1.0
                                        logits = predictions - np.max(predictions)
                                        exp_preds = np.exp(logits / temperature)
                                        smoothed_preds = exp_preds / np.sum(exp_preds)

                                        predicted_idx = int(np.argmax(smoothed_preds))
                                        max_prob = float(smoothed_preds[predicted_idx])

                                        prob_dict = {
                                            str(label): round(float(prob), 3)
                                            for label, prob in zip(self.voice_emotion_labels, smoothed_preds)
                                        }
                                        print(f"[VOICE DEBUG] Probabilities: {prob_dict} | Max Prob: {max_prob:.3f}", flush=True)

                                        # Apply a confidence threshold to filter out random guessing on ambient noise
                                        CONFIDENCE_THRESHOLD = 0.19
                                        if max_prob >= CONFIDENCE_THRESHOLD:
                                            self.current_voice_emotion = self.voice_emotion_labels[predicted_idx]
                                        else:
                                            self.current_voice_emotion = "Neutral"
                                    else:
                                        pred_idx = int(self.voice_model.predict(scaled_feats)[0])
                                        self.current_voice_emotion = self.voice_emotion_labels[pred_idx]
                            else:
                                self.current_voice_emotion = "Model Missing"
                        else:
                            self.current_voice_emotion = "Silent"
        except Exception as e:
            if self.is_running:
                print(f"[AUDIO ERROR] Stream error: {e}", flush=True)
                self.current_voice_emotion = "Audio Error"
    # method to fetch the location for the nearest rest area for the driver if extreme emotion is detected
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
        if not self.cap.isOpened():
            print("[CAMERA ERROR] Cannot start main loop because camera is closed.", flush=True)
            self.is_running = False
            return

        try:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    print("[CAMERA ERROR] Unable to read video stream from camera.", flush=True)
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    if self.face_model is not None:
                        roi_gray = gray[y:y + h, x:x + w]
                        roi_gray_resized = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
                        roi_gray_resized = roi_gray_resized / 255.0

                        roi_input = np.expand_dims(roi_gray_resized, axis=0)
                        roi_input = np.expand_dims(roi_input, axis=-1)

                        predicted_emotion = self.face_model.predict(roi_input, verbose=0)
                        emotion_index = np.argmax(predicted_emotion)
                        self.current_face_emotion = self.face_emotion_labels[emotion_index]

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

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.is_running = False
            if self.cap.isOpened():
                self.cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    monitor = DriverMonitor()
    monitor.run()


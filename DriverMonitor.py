import os
import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import requests
from tensorflow.keras.models import load_model


class DriverMonitor:
    def __init__(self):
        # Initialize Tkinter root window (hidden)
        self.root = tk.Tk()
        self.root.withdraw()

        # Display initial warning popup
        messagebox.showinfo(
            "Driver Safety Reminder",
            "Please drive safely, follow traffic laws, and fasten your seatbelt at all times."
        )

        # Load pre-trained models
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.model = load_model('emotion_detection_model.h5')

        # Configuration and labels
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.driving_speed = 70  # mph
        self.road_speed_limit = 60  # mph
        self.normal_heart_rate = 75  # BPM
        self.current_heart_rate = 80  # BPM

        # API setup
        self.google_maps_api_key = os.getenv('GOOGLE_MAPS_API_KEY', 'YOUR_FALLBACK_KEY_HERE')
        self.cached_rest_area = None

        # Camera setup
        self.cap = cv2.VideoCapture(0)

    def get_nearest_rest_area(self, lat=50.86285, lng=-0.08785446):
        """Fetch rest area details with caching to avoid lag in video feed."""
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
        except Exception as e:
            self.cached_rest_area = 'Error fetching rest area'

        return self.cached_rest_area

    def show_warning(self):
        """Display an auxiliary Tkinter warning window."""
        warning_window = tk.Toplevel(self.root)
        warning_window.title("Driver Safety Warning")
        warning_label = tk.Label(
            warning_window,
            text="Please drive Safely! Follow Traffic Laws and Fasten Seatbelt at all times."
        )
        warning_label.pack(padx=20, pady=10)
        close_button = tk.Button(warning_window, text="Close", command=warning_window.destroy)
        close_button.pack(pady=5)

    def run(self):
        """Main webcam execution loop."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Process ROI for emotion classification
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray_resized = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
                roi_gray_resized = roi_gray_resized / 255.0

                roi_input = np.expand_dims(roi_gray_resized, axis=0)
                roi_input = np.expand_dims(roi_input, axis=-1)

                predicted_emotion = self.model.predict(roi_input, verbose=0)
                emotion_index = np.argmax(predicted_emotion)
                emotion = self.emotion_labels[emotion_index]

                cv2.putText(
                    frame, f'Emotion: {emotion}', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2
                )

                if emotion in ['Angry', 'Disgust', 'Fear', 'Sad']:
                    cv2.putText(
                        frame, 'WARNING: Extreme Emotion Detected, consider taking a break.',
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                    )

                    nearest_rest_area = self.get_nearest_rest_area()
                    cv2.putText(
                        frame, f'Nearest Rest Area: {nearest_rest_area}',
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                    )

            if self.driving_speed > self.road_speed_limit:
                cv2.putText(
                    frame, f'Speed Limit Exceeded! ({self.driving_speed} mph > {self.road_speed_limit} mph)',
                    (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                )

            if self.current_heart_rate > self.normal_heart_rate + 20:
                cv2.putText(
                    frame, 'WARNING: High Heart Rate Detected, consider taking a break.',
                    (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                )

            cv2.imshow('Driver Emotion Detector', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()



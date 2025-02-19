import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt, welch

# Function to apply Butterworth Bandpass Filter
def butter_bandpass_filter(data, lowcut=0.7, highcut=3.0, fs=30, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Streamlit UI Setup
st.title("Real-Time Pulse Rate Detection & Stopwatch")

# Stopwatch UI
if "stopwatch_running" not in st.session_state:
    st.session_state.stopwatch_running = False
    st.session_state.start_time = None
    st.session_state.elapsed_time = 0.0
    st.session_state.pulse_records = []

col1, col2 = st.columns(2)
with col1:
    if st.button("Start Stopwatch"):
        st.session_state.stopwatch_running = True
        st.session_state.start_time = time.time() - st.session_state.elapsed_time

with col2:
    if st.button("Stop Stopwatch"):
        st.session_state.stopwatch_running = False

if st.button("Reset Stopwatch"):
    st.session_state.stopwatch_running = False
    st.session_state.start_time = None
    st.session_state.elapsed_time = 0.0

if st.session_state.stopwatch_running:
    st.session_state.elapsed_time = time.time() - st.session_state.start_time

st.subheader(f"Elapsed Time: {st.session_state.elapsed_time:.2f} seconds")

# Webcam Pulse Detection
st.subheader("Webcam Pulse Detection")
run_camera = st.checkbox("Start Camera")

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

if run_camera:
    cap = cv2.VideoCapture(0)
    pulse_values = []
    frame_window = st.image([])

    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to capture video")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, c = frame.shape
                    x, y, w, h = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                    roi = frame[y:y+h//5, x:x+w]
                    mean_intensity = np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
                    pulse_values.append(mean_intensity)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            frame_window.image(frame, channels="RGB")

            if len(pulse_values) > 100:
                filtered_signal = butter_bandpass_filter(pulse_values)
                peaks, _ = find_peaks(filtered_signal, distance=15, prominence=0.07 * np.std(filtered_signal))
                peak_based_pulse = (len(peaks) / (len(pulse_values) / 30)) * 60
                f_welch, Pxx_welch = welch(filtered_signal, fs=30, nperseg=len(filtered_signal) // 2)
                valid_freqs = (f_welch > 0.7) & (f_welch < 3.0)
                fft_based_pulse = (f_welch[np.argmax(Pxx_welch[valid_freqs])] * 60) if np.any(valid_freqs) else 0
                final_pulse_rate = round((peak_based_pulse * 0.35) + (fft_based_pulse * 0.80), 2)
                st.write(f"**Estimated Pulse Rate: {final_pulse_rate} BPM**")
                st.session_state.pulse_records.append({"Time (s)": round(st.session_state.elapsed_time, 2), "Pulse Rate (BPM)": final_pulse_rate})
                pulse_values = []
            time.sleep(0.1)

    cap.release()

if st.session_state.pulse_records:
    df = pd.DataFrame(st.session_state.pulse_records)
    st.subheader("Recorded Pulse Rates")
    st.dataframe(df)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Pulse Data", csv, "pulse_data.csv", "text/csv")


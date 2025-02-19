# -*- coding: utf-8 -*-
"""stopwatchforpulse.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1muSMFFnaN5j_YGwFsXO-ioMQ2rpNErpH

For the running the code and streamlit app using command prompt
"""

#pip install streamlit opencv-python numpy mediapipe

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
import base64

# Title
st.title("Real-Time Pulse Rate Detection with Stopwatch")

# Stopwatch UI
st.subheader("Stopwatch for Pulse Measurement")

stopwatch_js = """
<script>
var startTime, updatedTime, difference, tInterval;
var running = false;

function startStopwatch() {
    if (!running) {
        startTime = new Date().getTime() - (difference || 0);
        tInterval = setInterval(updateDisplay, 10);
        running = true;
    }
}

function stopStopwatch() {
    if (running) {
        clearInterval(tInterval);
        difference = new Date().getTime() - startTime;
        running = false;
    }
}

function resetStopwatch() {
    clearInterval(tInterval);
    startTime = null;
    difference = 0;
    running = false;
    document.getElementById("display").innerHTML = "00:00:00";
}

function updateDisplay() {
    updatedTime = new Date().getTime();
    difference = updatedTime - startTime;

    let milliseconds = Math.floor((difference % 1000) / 10);
    let seconds = Math.floor((difference / 1000) % 60);
    let minutes = Math.floor((difference / (1000 * 60)) % 60);

    document.getElementById("display").innerHTML =
        (minutes < 10 ? "0" : "") + minutes + ":" +
        (seconds < 10 ? "0" : "") + seconds + ":" +
        (milliseconds < 10 ? "0" : "") + milliseconds;
}
</script>

<div>
    <p id="display" style="font-size: 24px;">00:00:00</p>
    <button onclick="startStopwatch()">Start</button>
    <button onclick="stopStopwatch()">Stop</button>
    <button onclick="resetStopwatch()">Reset</button>
</div>
"""

st.components.v1.html(stopwatch_js, height=150)

# Webcam-based Pulse Detection
st.subheader("Webcam Pulse Detection")
run_camera = st.checkbox("Start Camera")

# Initialize Mediapipe Face Detection
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

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect face
            results = face_detection.process(frame_rgb)
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, c = frame.shape
                    x, y, w, h = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

                    # Extract Forehead ROI (for rPPG)
                    roi = frame[y:y+h//5, x:x+w]
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                    # Calculate Average Pixel Intensity (rPPG Proxy)
                    avg_pixel_value = np.mean(roi_gray)
                    pulse_values.append(avg_pixel_value)

                    # Draw Face Bounding Box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Show processed video in Streamlit
            frame_window.image(frame, channels="RGB")

            # Estimate Pulse Rate using FFT
            if len(pulse_values) > 100:
                fft_values = np.fft.fft(pulse_values)
                freqs = np.fft.fftfreq(len(fft_values), d=1/30)  # Assuming 30 FPS

                # Find dominant frequency in human pulse range (0.7 - 3 Hz)
                valid_freqs = (freqs > 0.7) & (freqs < 3)
                dominant_freq = freqs[np.argmax(np.abs(fft_values[valid_freqs]))]
                pulse_rate = int(dominant_freq * 60)

                st.write(f"**Estimated Pulse Rate: {pulse_rate} BPM**")
                pulse_values = []  # Reset

            time.sleep(0.1)

    cap.release()

streamlit run pulse_stopwatch.py


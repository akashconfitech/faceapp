import os
import fitz  # PyMuPDF
from openai import AzureOpenAI
import json
from flask_cors import CORS
from flask import Flask, request, Response, jsonify
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
import pytesseract
from PIL import Image
from werkzeug.utils import secure_filename
import re
import pycountry

from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
import tempfile
import mediapipe as mp
from flask_cors import CORS
# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Handle CORS
# Azure OpenAI Client Initialization

@app.route("/")
def hello():
    return "Hello, World from Flask in Azure!"


# Initialize mediapipe face detection module (optional, for dynamic face detection)
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def estimate_spo2_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    red_signal = []
    green_signal = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Use MediaPipe to detect the face in the frame
        with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            if results.detections:
                for detection in results.detections:
                    # Extract the bounding box for the detected face
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                    # Crop the face region from the frame (ROI)
                    roi = frame[y:y+h, x:x+w]
                    if roi.size == 0:
                        continue

                    # Calculate the average red and green pixel values (for simplicity, using average intensity)
                    r = np.mean(roi[:, :, 2])  # Red channel
                    g = np.mean(roi[:, :, 1])  # Green channel

                    red_signal.append(r)
                    green_signal.append(g)

        # Display the video frame (optional)
        # cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Break on 'q' key press (optional)
            break

    cap.release()
    cv2.destroyAllWindows()

    red_signal = np.array(red_signal)
    green_signal = np.array(green_signal)

    # Remove DC component
    ac_red = red_signal - np.mean(red_signal)
    ac_green = green_signal - np.mean(green_signal)

    # Ratio of AC/DC
    r_ratio = (np.std(ac_red) / np.mean(red_signal)) / (np.std(ac_green) / np.mean(green_signal))

    # Simplified SpO2 estimate using the ratio
    spo2 = 100 - 5 * r_ratio  # Constants A=100, B=5 can be tuned

    return round(spo2, 2)

@app.route('/analyze', methods=['POST'])
def analyze():
    heart_rate =88
    stress="Relaxed"
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    file = request.files['video']
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        file.save(temp_video.name)

        # Estimate SpO2 from the uploaded video file
        spo2_level = estimate_spo2_from_video(temp_video.name)
        if np.isnan(spo2_level):
            spo2_level = 0
            heart_rate =0
            stress="None"
        
        result = {
            "spo2_estimate": spo2_level,
            "heart_rate":heart_rate,
            "stress":stress,
            "note": "SpO2 is estimated using RGB signal and simplified algorithm"
        }

        os.unlink(temp_video.name)

    return jsonify({"message":result}), 200

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Set background image
page_bg_img = '''
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://source.unsplash.com/1600x900/?road,traffic");
    background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Sidebar with functionalities
st.sidebar.title("🚗 Vehicle Speed Detection App")
st.sidebar.subheader("App Functionalities")
st.sidebar.markdown("- Upload Video for Speed Detection")
st.sidebar.markdown("- Use Webcam for Live Detection")
st.sidebar.markdown("- View Detected Vehicle Speeds")

# Main UI
st.title("🚦 Vehicle Speed Detection System")
option = st.radio("Choose Input Method:", ("Upload Video", "Live Camera"))

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    vehicle_positions = {}
    speed_estimates = {}
    scale_factor = 0.005
    
    stframe = st.empty()
    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        current_time = time.time()
        elapsed_time = current_time - prev_time
        expected_time = 1.0 / fps
        if elapsed_time < expected_time:
            time.sleep(expected_time - elapsed_time)
        prev_time = time.time()
        
        results = model(frame)
        new_positions = {}

        for result in results:
            for i, box in enumerate(result.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                centroid_x = (x1 + x2) // 2
                centroid_y = (y1 + y2) // 2
                
                if i in vehicle_positions:
                    prev_x, prev_y, prev_time = vehicle_positions[i]
                    distance_pixels = np.sqrt((centroid_x - prev_x) ** 2 + (centroid_y - prev_y) ** 2)
                    time_diff = current_time - prev_time if prev_time else expected_time
                    speed = (distance_pixels * scale_factor / time_diff) * 3.6
                    if 5 < speed < 150:
                        speed_estimates[i] = int(round(speed))
                
                new_positions[i] = (centroid_x, centroid_y, current_time)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                speed_text = f"Speed: {speed_estimates.get(i, 0)} km/h"
                cv2.putText(frame, speed_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        vehicle_positions = new_positions
        stframe.image(frame, channels="BGR")
    
    cap.release()

def process_camera():
    cap = cv2.VideoCapture(0)  # Open webcam
    vehicle_positions = {}
    speed_estimates = {}
    scale_factor = 0.005
    
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        current_time = time.time()
        results = model(frame)
        new_positions = {}

        for result in results:
            for i, box in enumerate(result.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                centroid_x = (x1 + x2) // 2
                centroid_y = (y1 + y2) // 2
                
                if i in vehicle_positions:
                    prev_x, prev_y, prev_time = vehicle_positions[i]
                    distance_pixels = np.sqrt((centroid_x - prev_x) ** 2 + (centroid_y - prev_y) ** 2)
                    time_diff = current_time - prev_time if prev_time else 1/30
                    speed = (distance_pixels * scale_factor / time_diff) * 3.6
                    if 5 < speed < 150:
                        speed_estimates[i] = int(round(speed))
                
                new_positions[i] = (centroid_x, centroid_y, current_time)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                speed_text = f"Speed: {speed_estimates.get(i, 0)} km/h"
                cv2.putText(frame, speed_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        vehicle_positions = new_positions
        stframe.image(frame, channels="BGR")
    
    cap.release()

if option == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        process_video(tfile.name)

elif option == "Live Camera":
    process_camera()

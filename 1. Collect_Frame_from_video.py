import cv2
import os

# Create a folder to store frames
output_folder = "video_frames"
os.makedirs(output_folder, exist_ok=True)

# Load the video
video_path = r"C:\Users\sunil\DK\VSCODE\YOLO\video\Trafic_video.mp4"
cap = cv2.VideoCapture(video_path)
frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Get frame rate

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
    cv2.imwrite(frame_path, frame)
    frame_count += 1

cap.release()
print(f"Extracted {frame_count} frames at {frame_rate} FPS")

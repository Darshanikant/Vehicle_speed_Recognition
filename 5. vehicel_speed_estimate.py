import cv2
import numpy as np
import time
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open video
video_path = r"C:\Users\sunil\DK\VSCODE\YOLO\video\1900-151662242_small.mp4"
cap = cv2.VideoCapture(video_path)

# Video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second

# Output video
output_path = "vehicle_speed_output.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Vehicle tracking storage
vehicle_positions = {}  # {ID: (x, y, time)}
speed_estimates = {}  # {ID: speed}
scale_factor = 0.005  # Adjust based on real-world distance per pixel

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop when the video ends

    current_time = time.time()  # Get timestamp
    results = model(frame)  # Perform vehicle detection
    
    new_positions = {}

    for result in results:
        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            centroid_x = (x1 + x2) // 2
            centroid_y = (y1 + y2) // 2

            # Check if vehicle is already tracked
            if i in vehicle_positions:
                prev_x, prev_y, prev_time = vehicle_positions[i]
                distance_pixels = np.sqrt((centroid_x - prev_x) ** 2 + (centroid_y - prev_y) ** 2)

                # Time difference to avoid FPS dependency
                time_diff = current_time - prev_time if prev_time else 1/fps

                # Convert pixel distance to real-world speed (km/h)
                speed = (distance_pixels * scale_factor / time_diff) * 3.6  # Convert m/s to km/h
                
                # Filter unrealistic speeds
                if 5 < speed < 150:
                    speed_estimates[i] = int(round(speed))  # Round speed to whole number
            
            new_positions[i] = (centroid_x, centroid_y, current_time)

            # Draw bounding box & speed label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            speed_text = f"Speed: {speed_estimates.get(i, 0)} km/h"
            cv2.putText(frame, speed_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    vehicle_positions = new_positions  # Update positions

    # Display video
    cv2.imshow("Vehicle Speed Estimation", frame)
    out.write(frame)  # Save video

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key press
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved at: {output_path}")

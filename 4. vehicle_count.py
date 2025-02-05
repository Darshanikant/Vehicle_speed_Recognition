import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open video
video_path = r"C:\Users\sunil\DK\VSCODE\YOLO\video\1900-151662242_small.mp4"
cap = cv2.VideoCapture(video_path)

# Video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define output video
output_path = "vehicle_count_output.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Vehicle tracking
vehicle_count = 0
vehicles = {}  # Store tracked vehicle centroids

# Define counting line (ROI - Region of Interest)
line_y = frame_height // 2  # Mid-point of frame
offset = 10  # Error margin

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop when the video ends

    # Perform vehicle detection
    results = model(frame)
    
    new_vehicles = {}

    # Loop through detections
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            centroid_x = (x1 + x2) // 2
            centroid_y = (y1 + y2) // 2
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 0, 255), -1)  # Draw centroid
            
            # Check if vehicle crosses the counting line
            if line_y - offset < centroid_y < line_y + offset:
                if all(abs(centroid_x - cx) > 30 for cx in vehicles.keys()):  # Avoid duplicate counts
                    vehicle_count += 1
                new_vehicles[centroid_x] = centroid_y  # Store centroid positions

    vehicles = new_vehicles  # Update tracked vehicles
    
    # Draw counting line
    #cv2.line(frame, (0, line_y), (frame_width, line_y), (255, 0, 0), 2)
    cv2.putText(frame, f"Vehicles Counted: {vehicle_count}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Vehicle Counting", frame)

    # Save video output
    out.write(frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved at: {output_path}")
print(f"Total Vehicles Counted: {vehicle_count}")

from ultralytics import YOLO
import cv2
# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Perform detection on an extracted frame
results = model("video_frames/frame_100.jpg", show=True)
cv2.waitKey(0)
cv2.destroyAllWindows()
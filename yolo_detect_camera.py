# yolo_detect_camera.py
from ultralytics import YOLO
import cv2

# Load YOLOv8 pretrained model (small version)
model = YOLO('yolov8n.pt')

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Cannot open camera")
    exit()

print("✅ YOLOv8 Webcam Detection Started - Press 'Q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Run YOLO detection
    results = model(frame)

    # Annotate and display
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv8 Webcam Detection", annotated_frame)

    # Quit with 'Q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# yolo_detect_video.py
from ultralytics import YOLO
import cv2
import time
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import csv
from datetime import datetime

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Use yolov8s.pt for higher accuracy

# Video path
video_path = "test_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"❌ Error: Cannot open video file: {video_path}")
    exit()

# Optional: save output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_detected.mp4', fourcc, 20.0,
                      (int(cap.get(3)), int(cap.get(4))))

# ✅ CSV logging setup
csv_file = open('risk_log.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Frame", "Timestamp", "Vehicle_Count", "People_Count", "Risk_Level"])

# --- Dashboard Setup ---
plt.ion()
fig = plt.figure(figsize=(10, 5))

# 📈 Line chart
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_title("📈 Vehicle Count Trend")
ax1.set_xlabel("Frames")
ax1.set_ylabel("Vehicle Count")
ax1.set_ylim(0, 50)
line, = ax1.plot([], [], 'r-', lw=2)
x_data = deque(maxlen=100)
y_data = deque(maxlen=100)
frame_idx = 0

# 🕹 Gauge chart
ax2 = fig.add_subplot(1, 2, 2, polar=True)
ax2.set_ylim(0, 1)
ax2.set_yticklabels([])
ax2.set_xticklabels([])
ax2.set_title("🕹 Traffic Risk Gauge", va='bottom')

def draw_gauge(value):
    ax2.clear()
    ax2.set_ylim(0, 1)
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_theta_zero_location('N')
    ax2.set_theta_direction(-1)
    angles = np.linspace(0, np.pi, 100)
    ax2.fill_between(angles, 0, 1, color='lightgray')
    if value < 10:
        color = 'green'
        risk = 'LOW'
    elif value < 20:
        color = 'orange'
        risk = 'MEDIUM'
    else:
        color = 'red'
        risk = 'HIGH'
    angle = np.pi * min(value / 30, 1)
    ax2.fill_between(angles[:int(len(angles)*(value/30))], 0, 1, color=color)
    ax2.plot([angle, angle], [0, 1], color='black', lw=2)
    ax2.set_title(f"🚦 Risk Level: {risk}", va='bottom', fontsize=12)
    return risk

vehicle_classes = ['car', 'bus', 'truck', 'motorbike']
print("✅ YOLOv8 Traffic Dashboard Started - Press 'Q' to quit")

# --------------------------
# Main Loop
# --------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ Video completed")
        break

    start_time = time.time()
    results = model(frame, verbose=False)

    # 🚘 Count vehicles & people
    vehicle_count = 0
    people_count = 0
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        if cls_name in vehicle_classes:
            vehicle_count += 1
        elif cls_name == 'person':
            people_count += 1

    # ⚠ Risk text & color
    risk_level = "LOW"
    color = (0, 255, 0)
    if vehicle_count > 20:
        risk_level = "⚠ HIGH"
        color = (0, 0, 255)
    elif vehicle_count > 10:
        risk_level = "⚠ MEDIUM"
        color = (0, 165, 255)

    # 📝 Overlay info
    cv2.putText(frame, f"Vehicles: {vehicle_count} | People: {people_count}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"RISK: {risk_level}",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {int(fps)}",
                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # 🖼️ Annotated frame
    annotated = results[0].plot()
    combined = cv2.addWeighted(annotated, 0.7, frame, 0.3, 0)
    cv2.imshow("🚦 Traffic Accident Detection Dashboard", combined)
    out.write(combined)

    # 📊 Update line chart
    frame_idx += 1
    x_data.append(frame_idx)
    y_data.append(vehicle_count)
    line.set_xdata(x_data)
    line.set_ydata(y_data)
    ax1.set_xlim(max(0, frame_idx - 100), frame_idx + 10)

    # 🕹 Update gauge
    risk_gauge = draw_gauge(vehicle_count)

    # ✍ Log to CSV
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    csv_writer.writerow([frame_idx, timestamp, vehicle_count, people_count, risk_gauge])

    # 🪄 Update dashboard
    fig.canvas.draw()
    fig.canvas.flush_events()

    # 🔴 Quit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
out.release()
csv_file.close()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
print("📄 Data logged to risk_log.csv ✅")

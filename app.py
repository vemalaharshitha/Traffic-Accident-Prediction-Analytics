import os
import cv2
import math
import numpy as np
from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
from collections import defaultdict

app = Flask(__name__)

# -----------------------
# CONFIG
# -----------------------
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

model = YOLO("yolov8n.pt")
VEHICLE_NAMES = {"car", "bus", "truck", "motorbike"}

# Global analytics data
analytics = {
    "risk": {"probability": 0.0, "level": "LOW"},
    "counts": {},
    "status": "IDLE",
    "output_file": None
}

# -----------------------
# Helpers
# -----------------------
def calculate_risk(vehicle_count):
    prob = min(1.0, vehicle_count * 0.05)
    if prob < 0.33:
        return prob, "LOW"
    elif prob < 0.66:
        return prob, "MEDIUM"
    else:
        return prob, "HIGH"

def process_video(input_path, output_path):
    """Process video and compute analytics frame by frame."""
    global analytics
    analytics["status"] = "PROCESSING"
    analytics["output_file"] = None

    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"🚀 Processing {frame_count} frames...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        counts = defaultdict(int)

        # YOLO detections
        if results and len(results[0].boxes) > 0:
            for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
                x1, y1, x2, y2 = map(int, box.tolist())
                label = model.names[int(cls)]
                if label in VEHICLE_NAMES:
                    counts[label] += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        total = sum(counts.values())
        prob, level = calculate_risk(total)
        analytics["risk"] = {"probability": prob, "level": level}
        analytics["counts"] = counts

        color = (0, 255, 0) if level == "LOW" else (0, 255, 255) if level == "MEDIUM" else (0, 0, 255)
        cv2.putText(frame, f"RISK: {level} ({prob*100:.1f}%)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        cv2.putText(frame, f"Vehicles: {total}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"✅ Video saved to: {output_path}")

    analytics["status"] = "DONE"
    analytics["output_file"] = os.path.basename(output_path)


# -----------------------
# API
# -----------------------

@app.route("/api/upload", methods=["POST"])
def upload_and_start():
    """Upload file and start background processing."""
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "No file uploaded"}), 400

    filename = f.filename
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    output_path = os.path.join(PROCESSED_FOLDER, f"processed_{filename}")
    f.save(input_path)

    print(f"📥 Uploaded: {input_path}")
    process_video(input_path, output_path)
    return jsonify({"message": "Video processing started"})


@app.route("/api/results", methods=["GET"])
def get_results():
    """Return current analytics for dashboard."""
    return jsonify(analytics)


@app.route("/api/download", methods=["GET"])
def download_processed():
    """Download processed file."""
    if not analytics["output_file"]:
        return jsonify({"error": "No file yet"}), 404
    path = os.path.join(PROCESSED_FOLDER, analytics["output_file"])
    return send_file(path, as_attachment=True)


if __name__ == "__main__":
    print("🚀 Flask backend running on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)

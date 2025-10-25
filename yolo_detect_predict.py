# 🚦 yolo_detect_predict.py (Dashboard Enhanced Version)

import os
import cv2
import csv
import math
import numpy as np
from collections import deque
from ultralytics import YOLO
from datetime import datetime

# =========================
# CONFIG
# =========================
MODEL_WEIGHTS   = "yolov8n.pt"     
VIDEO_PATH      = "test_accident.mp4"
TRACKER_CFG     = "bytetrack.yaml"

PIXEL_TO_KMH    = 0.15             
TTC_ALERT_S     = 2.0              
TTC_WARN_S      = 4.0              
MIN_REL_SPEED   = 5.0              
TTC_SMOOTHING   = 10              
ALERT_MEMORY    = 45              
LOG_FILE        = "risk_events.csv"

VEHICLE_NAMES   = {"car", "bus", "truck", "motorbike"}
DANGER_BLINK_FRAMES = 10

# =========================
# HELPERS
# =========================
def get_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)

def draw_arrow(frame, p, v, color, length=50):
    n = np.linalg.norm(v)
    if n < 1e-3:
        return
    d = v / (n + 1e-9)
    end = (int(p[0] + d[0] * length), int(p[1] + d[1] * length))
    cv2.arrowedLine(frame, (int(p[0]), int(p[1])), end, color, 2, tipLength=0.35)

def ttc_seconds(p_i, v_i, p_j, v_j, fps):
    r = p_j - p_i
    v = v_j - v_i
    v2 = float(np.dot(v, v))
    if v2 < 1e-6:
        return None
    closing = -float(np.dot(r, v))
    if closing <= 0:
        return None
    t_frames = closing / v2
    t_seconds = t_frames / max(fps, 1e-6)
    if t_seconds < 0 or t_seconds > 20:
        return None
    return t_seconds

def approach_angle_deg(p_i, v_i, p_j):
    r = p_j - p_i
    if np.linalg.norm(v_i) < 1e-6 or np.linalg.norm(r) < 1e-6:
        return 180.0
    a = np.dot(v_i, r) / (np.linalg.norm(v_i) * np.linalg.norm(r))
    a = np.clip(a, -1.0, 1.0)
    return math.degrees(math.acos(a))

def risk_score(ttc, rel_speed, angle_deg):
    if ttc is None:
        return 0.0
    ttc_term   = max(0.0, (1.0 / ttc))          
    speed_term = max(0.0, (rel_speed * 0.01))   
    angle_term = max(0.0, (max(0.0, 45.0 - angle_deg) / 45.0)) 
    return 0.6 * ttc_term + 0.3 * speed_term + 0.1 * angle_term

def draw_bar(panel, x, y, width, height, value, max_value, label, color):
    pct = min(1.0, value / max_value)
    filled = int(pct * width)
    cv2.rectangle(panel, (x, y), (x + width, y + height), (80,80,80), -1)
    cv2.rectangle(panel, (x, y), (x + filled, y + height), color, -1)
    cv2.rectangle(panel, (x, y), (x + width, y + height), (200,200,200), 1)
    cv2.putText(panel, f"{label}: {int(pct*100)}%", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

def draw_dashboard(frame, total_vehicles, avg_speed, min_ttc, risk_text, blink_on, vehicle_counts, risk_level):
    panel = np.zeros((frame.shape[0], 330, 3), dtype=np.uint8)
    panel[:] = (25, 25, 25)
    cv2.putText(panel, "TRAFFIC DASHBOARD", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.line(panel, (15, 58), (310, 58), (70,70,70), 1)

    # Total vehicles
    cv2.putText(panel, f"Vehicles: {total_vehicles}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(panel, f"Avg speed: {avg_speed:.1f} km/h", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # TTC
    if min_ttc is not None and math.isfinite(min_ttc):
        cv2.putText(panel, f"Min TTC: {min_ttc:.2f}s", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)
    else:
        cv2.putText(panel, f"Min TTC: --", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,180,180), 2)

    # Time
    now = datetime.now().strftime("%H:%M:%S")
    cv2.putText(panel, f"Time: {now}", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)

    # Vehicle type counts
    y_start = 220
    for k, v in vehicle_counts.items():
        cv2.putText(panel, f"{k}: {v}", (20, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)
        y_start += 25

    # Speed gauge bar
    draw_bar(panel, 20, y_start + 10, 250, 18, avg_speed, 120, "Avg Speed", (0, 200, 0))

    # Risk level bar
    risk_color = (0, 200, 0) if risk_level < 0.4 else (0, 200, 255) if risk_level < 0.7 else (0, 0, 255)
    draw_bar(panel, 20, y_start + 50, 250, 18, risk_level, 1.0, "Risk Level", risk_color)

    # Bottom danger banner
    color = (0,255,0)
    if "WARNING" in risk_text: color = (0,255,255)
    if "DANGER"  in risk_text: color = (0,0,255)
    if "DANGER" in risk_text and blink_on:
        cv2.rectangle(panel, (20, y_start + 100), (290, y_start + 135), (0,0,180), -1)
    else:
        cv2.rectangle(panel, (20, y_start + 100), (290, y_start + 135), color, -1)

    cv2.putText(panel, risk_text, (35, y_start + 127), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    return np.hstack((panel, frame))

def translucent_box(frame, x1, y1, x2, y2, color=(0,0,255), alpha=0.25, border=2):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, border)

def log_event_csv(fname, data_dict):
    new_file = not os.path.exists(fname)
    with open(fname, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(data_dict.keys()))
        if new_file:
            w.writeheader()
        w.writerow(data_dict)

# =========================
# MAIN
# =========================
def main():
    if not os.path.exists(MODEL_WEIGHTS):
        print(f"❌ Model file not found: {MODEL_WEIGHTS}")
        return
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Video not found: {VIDEO_PATH}")
        return

    model = YOLO(MODEL_WEIGHTS)
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    prev_centers = {}
    velocities   = {}
    alert_memory = {}
    ttc_hist     = deque(maxlen=TTC_SMOOTHING)
    frame_idx    = 0

    print("✅ Accident prediction started. Press 'Q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("✅ Video completed.")
            break

        results = model.track(frame, persist=True, tracker=TRACKER_CFG, verbose=False)
        if not results:
            cv2.imshow("Accident Prediction", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            continue

        res = results[0]
        if res.boxes is None or res.boxes.xyxy is None:
            cv2.imshow("Accident Prediction", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            continue

        xyxy = res.boxes.xyxy.cpu().numpy()
        cls  = res.boxes.cls.cpu().numpy()
        ids  = res.boxes.id
        ids  = ids.cpu().numpy().astype(int) if ids is not None else np.arange(len(xyxy))

        centers, speeds = {}, []
        vehicle_counts = {"Car": 0, "Bus": 0, "Truck": 0, "Bike": 0}

        for box, c, vid in zip(xyxy, cls, ids):
            name = model.names[int(c)]
            if name not in VEHICLE_NAMES:
                continue
            if name == "car": vehicle_counts["Car"] += 1
            if name == "bus": vehicle_counts["Bus"] += 1
            if name == "truck": vehicle_counts["Truck"] += 1
            if name == "motorbike": vehicle_counts["Bike"] += 1

            ctr = get_center(box)
            centers[vid] = ctr
            if vid in prev_centers:
                v = ctr - prev_centers[vid]
                velocities[vid] = v
                spd_kmh = np.linalg.norm(v) * PIXEL_TO_KMH * fps
                speeds.append(spd_kmh)
            prev_centers[vid] = ctr

        # TTC calculations
        min_ttc = None
        best_score = 0.0
        best_pair = None

        vids = list(centers.keys())
        for i in range(len(vids)):
            for j in range(i+1, len(vids)):
                vi = velocities.get(vids[i], np.zeros(2))
                vj = velocities.get(vids[j], np.zeros(2))
                if np.linalg.norm(vi) < MIN_REL_SPEED and np.linalg.norm(vj) < MIN_REL_SPEED:
                    continue
                t = ttc_seconds(centers[vids[i]], vi, centers[vids[j]], vj, fps)
                if t is None:
                    continue
                rel_v = vj - vi
                rel_speed = float(np.linalg.norm(rel_v))
                ai = approach_angle_deg(centers[vids[i]], vi, centers[vids[j]])
                aj = approach_angle_deg(centers[vids[j]], vj, centers[vids[i]])
                angle = min(ai, aj)
                score = risk_score(t, rel_speed, angle)
                if min_ttc is None or t < min_ttc:
                    min_ttc = t
                if score > best_score:
                    best_score = score
                    best_pair = (vids[i], vids[j])

        smoothed_ttc = None
        if min_ttc is not None:
            ttc_hist.append(min_ttc)
            smoothed_ttc = float(np.mean(ttc_hist))

        alerted_id = None
        if best_pair is not None and (smoothed_ttc is not None and smoothed_ttc < TTC_ALERT_S or best_score > 0.5):
            i, j = best_pair
            vi = velocities.get(i, np.zeros(2))
            vj = velocities.get(j, np.zeros(2))
            alerted_id = i if np.linalg.norm(vi) >= np.linalg.norm(vj) else j
            alert_memory[alerted_id] = ALERT_MEMORY
            ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            log_event_csv(LOG_FILE, {
                "timestamp_ms": int(ms),
                "frame_idx": frame_idx,
                "alert_vehicle_id": int(alerted_id),
                "ttc_seconds": round(smoothed_ttc if smoothed_ttc else -1, 3),
                "risk_score": round(best_score, 3)
            })

        # Memory fade
        for vid in list(alert_memory.keys()):
            alert_memory[vid] -= 1
            if alert_memory[vid] <= 0:
                del alert_memory[vid]

        risk_val = f"{smoothed_ttc:.1f}s" if smoothed_ttc is not None else "--"
        risk_text = "SAFE"
        if smoothed_ttc is not None and smoothed_ttc < TTC_WARN_S or len(alert_memory) > 0:
            if smoothed_ttc is not None and smoothed_ttc < TTC_ALERT_S or len(alert_memory) > 0:
                risk_text = f"DANGER  TTC ~ {risk_val}"
            else:
                risk_text = f"WARNING  TTC ~ {risk_val}"

        total_vehicles = len(centers)
        avg_speed = float(np.mean(speeds)) if speeds else 0.0

        for box, c, vid in zip(xyxy, cls, ids):
            name = model.names[int(c)]
            if name not in VEHICLE_NAMES:
                continue
            x1, y1, x2, y2 = map(int, box)
            ctr = centers.get(vid, None)
            v   = velocities.get(vid, np.zeros(2))
            spd_kmh = np.linalg.norm(v) * PIXEL_TO_KMH * fps
            is_alerted = vid in alert_memory

            if is_alerted:
                translucent_box(frame, x1, y1, x2, y2, color=(0, 0, 255), alpha=0.30, border=3)
                cv2.putText(frame, f"{name} {spd_kmh:.0f}km/h DANGER", (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                cv2.circle(frame, (int(ctr[0]), int(ctr[1])), 60, (0, 0, 255), 3)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(frame, f"{name} {spd_kmh:.0f}km/h", (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,0), 2)

            draw_arrow(frame, ctr, v, (0, 0, 255) if is_alerted else (0, 200, 0), length=55)

        blink_on = (frame_idx // DANGER_BLINK_FRAMES) % 2 == 0
        if "DANGER" in risk_text and blink_on:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 70), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            cv2.putText(frame, "COLLISION ALERT", (40, 48),
                        cv2.FONT_HERSHEY_DUPLEX, 1.1, (255,255,255), 2)

        composed = draw_dashboard(frame, total_vehicles, avg_speed, smoothed_ttc, risk_text, blink_on, vehicle_counts, best_score)
        cv2.imshow("Accident Prediction Dashboard", composed)

        frame_idx += 1
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"📄 Events logged to {LOG_FILE}")

if __name__ == "__main__":
    main()

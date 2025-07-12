import cv2
import time
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLO model
model = YOLO("models/yolo_soccer.pt")

# Load DeepSORT tracker
tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)

# Open video
cap = cv2.VideoCapture("videos/15sec_input_720p.mp4")
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Store player paths for trails
player_paths = {}
# Store last positions for speed estimation
last_positions = {}

while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 360))
    frame_h, frame_w, _ = frame.shape

    # YOLO detection
    results = model.predict(frame, conf=0.5, verbose=False)
    detections = []

    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            width = x2 - x1
            height = y2 - y1
            conf = 0.99
            detections.append(([x1, y1, width, height], conf, 'player'))

    # DeepSORT tracking
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, w, h = track.to_ltrb()
        l, t, w, h = int(l), int(t), int(w), int(h)

        # --- Adaptive box resizing based on height ---
        relative_height = h / frame_h
        shrink_factor = 0.1 + 0.2 * (1 - relative_height)
        shrink_factor = max(0.1, min(shrink_factor, 0.3))

        shrink_w = int(w * shrink_factor / 2)
        shrink_h = int(h * shrink_factor / 2)

        l += shrink_w
        t += shrink_h
        w -= 2 * shrink_w
        h -= 2 * shrink_h

        # --- Speed estimation ---
        center_x = l + w // 2
        center_y = t + h // 2

        speed_text = ""
        if track_id in last_positions:
            lx, ly = last_positions[track_id]
            pixel_dist = np.sqrt((center_x - lx)**2 + (center_y - ly)**2)
            fps = 1 / (time.time() - start_time)
            pixel_speed = pixel_dist * fps
            # Approx conversion: assume 1 pixel ~ 5 cm
            speed_kph = pixel_speed * 0.05 * 3.6
            speed_text = f"{speed_kph:.1f} km/h"
        last_positions[track_id] = (center_x, center_y)

        # --- Store and draw trail path ---
        if track_id not in player_paths:
            player_paths[track_id] = []
        player_paths[track_id].append((center_x, center_y))
        # Keep only last 30 points
        player_paths[track_id] = player_paths[track_id][-30:]

        for pt in player_paths[track_id]:
            cv2.circle(frame, pt, 1, (0, 255, 255), -1)

        # --- Draw final rectangle and text ---
        cv2.rectangle(frame, (l, t), (l + w, t + h), (0, 255, 0), 1)
        cv2.putText(frame, f"ID:{track_id}", (l, t - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        if speed_text:
            cv2.putText(frame, speed_text, (l, t - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    # --- FPS overlay ---
    fps_text = f"FPS: {1 / (time.time() - start_time):.2f}"
    cv2.putText(frame, fps_text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("Advanced Soccer Analytics", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # lightweight model

cap = cv2.VideoCapture("result.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(3))
height = int(cap.get(4))

out = cv2.VideoWriter("final_output.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps*2, (width, height))

car_count = 0
bike_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    for box in results.boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        label = model.names[cls]

        if label == "car":
            car_count += 1
        elif label == "motorcycle":
            bike_count += 1
        else:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # Bottom-left UI
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, height-80), (220, height-10), (0,0,0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    cv2.putText(frame, f"Cars: {car_count}", (20, height-50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.putText(frame, f"Bikes: {bike_count}", (20, height-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    out.write(frame)

cap.release()
out.release()

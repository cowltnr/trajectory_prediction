import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# 모델 로드 (fine-tuned 모델 또는 yolov8n/s/m/l/x.pt 가능)
model = YOLO("runs/detect/train4/weights/best.pt")

# DeepSORT 초기화 (ID 유지 설정)
tracker = DeepSort(max_age=60)

# 웹캠 열기 (0번 장치 사용, 필요시 1로 바꿔도 됨)
cap = cv2.VideoCapture(0)

# 실시간 프레임 처리
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 추론
    results = model(frame, conf=1)

    # 감지된 사람만 추출
    detections = []
    for box in results[0].boxes:
        if int(box.cls[0]) == 7:  # person class:14, chair class:7
            coords = box.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]
            conf = float(box.conf[0].cpu().numpy())
            detections.append((coords, conf))

    # DeepSORT 추적
    tracks = tracker.update_tracks(detections, frame=frame)

    # 추적 결과 그리기
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 화면에 출력
    cv2.imshow("YOLOv8 + DeepSORT (Webcam)", frame)

    # 'q' 키 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()

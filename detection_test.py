from ultralytics import YOLO
import cv2

# 1. 모델 로드
model = YOLO("./yolov8s.pt")  # 또는 fine-tuning한 모델 경로

# 2. 웹캠 or 비디오 파일 열기 (.webm 파일 경로)
video_path = "./input_videos/MOT17-09-SDP-raw.webm"
cap = cv2.VideoCapture(video_path)

# 3. 비디오 출력 저장 옵션 설정
output_path = "./output_video/yolo_MOT09_v2.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

# 4. 프레임 반복 처리
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 5. YOLOv8 추론
    results = model.predict(source=frame, classes=0, conf=0.8)  # person class: 0, 14

    # 6. 결과 시각화
    annotated_frame = results[0].plot()

    # 7. 프레임 저장 및 시각화
    out.write(annotated_frame)
    cv2.imshow("YOLOv8 Detection", annotated_frame)
    if cv2.waitKey(1) == ord("q"):
        break

# 8. 종료 처리
cap.release()
out.release()
cv2.destroyAllWindows()

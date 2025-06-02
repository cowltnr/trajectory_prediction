import os
import cv2
import csv
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# 모델 로드
model = YOLO("runs/detect/train4/weights/best.pt")

# 추적 실패 상태에서 ID를 최대로 유지할 프레임 수
max_age=60

# 이미지 루트 디렉토리 (PL1~PL12 폴더가 여기 안에 있어야 함)
pl_root = r"data/deepsort_input"
out_root = r"data/deepsort_output"
output_dir = os.path.join(f"{out_root}_{max_age}")
os.makedirs(output_dir, exist_ok=True)

# DeepSORT tracker 초기화
tracker = DeepSort(max_age)

# 각 PL 폴더 순회
for pl_folder in sorted(os.listdir(pl_root)):
    pl_path = os.path.join(pl_root, pl_folder)
    if not os.path.isdir(pl_path):
        continue

    # 이미지 정렬 (frame 번호 순)
    image_files = sorted(os.listdir(pl_path), key=lambda x: int(x.split('_')[1]))

    # CSV 결과 저장 준비
    csv_path = os.path.join(output_dir, f"{pl_folder}.csv")
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "track_id", "x_center", "y_center"])

        for frame_idx, image_name in enumerate(image_files):
            image_path = os.path.join(pl_path, image_name)
            frame = cv2.imread(image_path)
            results = model(frame)

            # 사람(class=14)만 필터링
            detections = []

            for box in results[0].boxes:
                if int(box.cls[0]) == 14:  # person
                    coords = box.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]
                    conf = float(box.conf[0].cpu().numpy())  # confidence
                    detections.append((coords, conf))

                    print("detections:", detections)

            # 추적기 업데이트
            tracks = tracker.update_tracks(detections, frame=frame)

            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                x1, y1, x2, y2 = track.to_ltrb()
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                writer.writerow([frame_idx, track_id, x_center, y_center])

    print(f"{pl_folder} 처리 완료 → {csv_path}")

print("PL 시퀀스 추적 및 CSV 저장 완료")

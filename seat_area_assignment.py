import cv2
import numpy as np
from ultralytics import YOLO
import torch
import torch.nn.functional as F
import os

# boxmotのReIDモデルを使用
try:
    from boxmot.appearance.reid.factory import osnet_x0_25
    print("✅ boxmot imported successfully")
    BOXMOT_AVAILABLE = True
except ImportError:
    print("⚠️ boxmot not available, using fallback")
    BOXMOT_AVAILABLE = False

# 座席エリアの4点座標（画像サイズに合わせて調整してください）
SEAT_AREAS = {
    'left':   [(372, 1212), (734, 1616), (1337, 1203), (842, 977)],
    'center': [(919, 1597), (1565, 1089), (2258, 1389), (1560, 1964)],
    'right':  [(1725, 1966), (2626, 1027), (3339, 1265), (3281, 1976)]
}

CONFIG = {
    'min_confidence': 0.5,
    'min_area_ratio': 0.005,
    'min_width': 40,
    'min_height': 80,
    'nms_iou_threshold': 0.4,
}

# 多角形内判定
def is_point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), point, False) >= 0

# 人物検出
def detect_persons(image, model, config):
    results = model(image, classes=0, conf=config['min_confidence'], iou=config['nms_iou_threshold'], verbose=False)
    detections = []
    if results[0].boxes is not None:
        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            width = x2 - x1
            height = y2 - y1
            area = width * height
            area_ratio = area / (image.shape[0] * image.shape[1])
            if conf < config['min_confidence'] or width < config['min_width'] or height < config['min_height'] or area_ratio < config['min_area_ratio']:
                continue
            detections.append({'bbox': [int(x1), int(y1), int(x2), int(y2)], 'conf': float(conf)})
    return detections

# メイン処理
def main():
    image_path = "../videos/megane04.png"  # ここをテスト画像に変更してください
    output_path = "seat_area_assignment_result.jpg"

    # 画像読み込み
    image = cv2.imread(image_path)
    if image is None:
        print(f"画像が見つかりません: {image_path}")
        return

    # YOLOv8モデルロード
    model = YOLO('yolov8m.pt')

    # 人物検出
    detections = detect_persons(image, model, CONFIG)

    # 座席エリア描画
    seat_colors = {'left': (255, 0, 0), 'center': (0, 255, 0), 'right': (0, 0, 255)}
    vis_image = image.copy()
    for seat, poly in SEAT_AREAS.items():
        cv2.polylines(vis_image, [np.array(poly, np.int32)], isClosed=True, color=seat_colors[seat], thickness=4)
        # 座席名ラベル
        label_pos = tuple(np.mean(poly, axis=0).astype(int))
        cv2.putText(vis_image, seat, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 2, seat_colors[seat], 4)

    # 各人物の中心点がどの座席エリアか判定
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        assigned_seat = None
        for seat, poly in SEAT_AREAS.items():
            if is_point_in_polygon(center, poly):
                assigned_seat = seat
                break
        # バウンディングボックスと中心点描画
        color = seat_colors.get(assigned_seat, (128, 128, 128))
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 4)
        cv2.circle(vis_image, center, 10, color, -1)
        label = f"Person {i+1}"
        if assigned_seat:
            label += f" ({assigned_seat})"
        cv2.putText(vis_image, label, (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        print(f"人物{i+1}: bbox={det['bbox']} 中心={center} → 座席: {assigned_seat if assigned_seat else '未割当'}")

    # 結果画像保存
    cv2.imwrite(output_path, vis_image)
    print(f"結果画像を保存しました: {output_path}")

if __name__ == "__main__":
    main() 
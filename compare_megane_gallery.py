import cv2
import numpy as np
from ultralytics import YOLO
import torch
import torch.nn.functional as F
import os
from pathlib import Path

# boxmotのReIDモデルを使用
try:
    from boxmot.appearance.reid.factory import osnet_x0_25
    print("✅ boxmot imported successfully")
    BOXMOT_AVAILABLE = True
except ImportError:
    print("⚠️ boxmot not available, using fallback")
    BOXMOT_AVAILABLE = False

# 設定パラメータ
CONFIG = {
    'min_confidence': 0.5,          # 最小信頼度閾値（感度向上）
    'min_area_ratio': 0.005,        # 画像全体に対する最小面積比（より小さな人物も検出）
    'min_width': 40,                # 最小幅（緩和）
    'min_height': 80,               # 最小高さ（緩和）
    'cosine_threshold': 0.85,       # コサイン類似度閾値
    'nms_iou_threshold': 0.4,       # NMS IoU閾値
    'reid_model': 'osnet_x0_25',    # OSNetモデル
}

class OSNetFeatureExtractor:
    """OSNetベースの特徴量抽出器（megane専用）"""
    
    def __init__(self, model_name='osnet_x0_25'):
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if BOXMOT_AVAILABLE:
            try:
                print(f"🧠 OSNet({model_name})モデルを初期化中...")
                
                if model_name == 'osnet_x0_25':
                    self.model = osnet_x0_25(num_classes=1000, pretrained=True)
                else:
                    from boxmot.appearance.reid.factory import MODEL_FACTORY
                    if model_name in MODEL_FACTORY:
                        model_func = MODEL_FACTORY[model_name]
                        self.model = model_func(num_classes=1000, pretrained=True)
                    else:
                        raise ValueError(f"Unknown model: {model_name}")
                
                self.model.eval()
                self.model.to(self.device)
                print(f"✅ OSNetモデル初期化完了 (device: {self.device})")
                
            except Exception as e:
                print(f"❌ OSNetモデル初期化失敗: {e}")
                print("📝 フォールバック特徴量抽出器を使用します")
                self.model = None
        else:
            print("📝 boxmot利用不可、フォールバック特徴量抽出器を使用")
    
    def extract_features(self, image, box):
        """OSNetまたはフォールバック特徴量抽出"""
        x1, y1, x2, y2 = map(int, box)
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        person_patch = image[y1:y2, x1:x2]
        if person_patch.size == 0 or person_patch.shape[0] < 20 or person_patch.shape[1] < 10:
            return None
        
        if self.model is not None and BOXMOT_AVAILABLE:
            return self._extract_osnet_features(person_patch)
        else:
            return self._extract_fallback_features(person_patch)
    
    def _extract_osnet_features(self, person_patch):
        """OSNetを使った特徴量抽出"""
        try:
            resized = cv2.resize(person_patch, (128, 256))
            rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            normalized = (rgb_image / 255.0 - mean) / std
            
            tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float()
            tensor = tensor.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model(tensor)
                features = F.normalize(features, p=2, dim=1)
                return features.cpu().numpy().flatten()
                
        except Exception as e:
            print(f"⚠️ OSNet特徴量抽出エラー: {e}")
            return self._extract_fallback_features(person_patch)
    
    def _extract_fallback_features(self, person_patch):
        """フォールバック特徴量抽出（改善版）"""
        resized = cv2.resize(person_patch, (128, 256))
        
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        mid_height = 128
        
        upper_hsv = hsv[:mid_height, :]
        lower_hsv = hsv[mid_height:, :]
        
        upper_hist_h = cv2.calcHist([upper_hsv], [0], None, [16], [0, 180])
        upper_hist_s = cv2.calcHist([upper_hsv], [1], None, [16], [0, 256])
        upper_hist_v = cv2.calcHist([upper_hsv], [2], None, [16], [0, 256])
        
        lower_hist_h = cv2.calcHist([lower_hsv], [0], None, [16], [0, 180])
        lower_hist_s = cv2.calcHist([lower_hsv], [1], None, [16], [0, 256])
        lower_hist_v = cv2.calcHist([lower_hsv], [2], None, [16], [0, 256])
        
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_hist = cv2.calcHist([edges], [0], None, [16], [0, 256])
        
        features = []
        for hist in [upper_hist_h, upper_hist_s, upper_hist_v, 
                    lower_hist_h, lower_hist_s, lower_hist_v, 
                    edge_hist]:
            normalized = cv2.normalize(hist, hist).flatten()
            features.append(normalized)
        
        combined_features = np.concatenate(features)
        norm = np.linalg.norm(combined_features)
        if norm > 0:
            combined_features = combined_features / norm
        
        return combined_features
    
    def calculate_cosine_similarity(self, feat1, feat2):
        """コサイン類似度計算"""
        if feat1 is None or feat2 is None:
            return None
        
        return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))

def filter_detections(detections, image_shape, config):
    """検出結果をフィルタリング"""
    h, w = image_shape[:2]
    image_area = h * w
    
    filtered = []
    print(f"  フィルタリング前: {len(detections)}個の検出")
    
    for i, detection in enumerate(detections):
        x1, y1, x2, y2 = detection['bbox']
        conf = detection['conf']
        
        if conf < config['min_confidence']:
            print(f"    人物{i+1}: 信頼度不足 ({conf:.3f} < {config['min_confidence']})")
            continue
        
        width = x2 - x1
        height = y2 - y1
        area = width * height
        area_ratio = area / image_area
        
        if (width < config['min_width'] or 
            height < config['min_height'] or 
            area_ratio < config['min_area_ratio']):
            print(f"    人物{i+1}: サイズ不足 (幅{width:.0f}, 高{height:.0f}, 面積比{area_ratio:.4f})")
            continue
        
        filtered.append(detection)
        print(f"    人物{i+1}: フィルタ通過 ✓")
    
    print(f"  フィルタリング後: {len(filtered)}個の検出")
    return filtered

def detect_and_extract_features_megane(image_path, model, extractor, config, output_dir="detection_results_megane"):
    """megane専用検出・特徴量抽出"""
    print(f"\n=== {image_path} の処理（megane専用） ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"エラー: {image_path} を読み込めませんでした")
        return None, None
    
    print(f"画像サイズ: {image.shape}")
    
    # YOLOv8で人物検出
    results = model(image, classes=0, conf=config['min_confidence'], 
                   iou=config['nms_iou_threshold'], verbose=False)
    
    raw_detections = []
    
    if results[0].boxes is not None:
        boxes = results[0].boxes
        print(f"YOLO検出数: {len(boxes)}")
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            
            raw_detections.append({
                'bbox': [x1, y1, x2, y2],
                'conf': conf,
                'index': i
            })
    
    # フィルタリング
    filtered_detections = filter_detections(raw_detections, image.shape, config)
    
    # 特徴量抽出と可視化
    final_detections = []
    features = []
    
    if filtered_detections:
        vis_image = image.copy()
        
        for i, detection in enumerate(filtered_detections):
            x1, y1, x2, y2 = detection['bbox']
            conf = detection['conf']
            
            # OSNet特徴量抽出
            feature = extractor.extract_features(image, [x1, y1, x2, y2])
            
            if feature is not None:
                final_detections.append(detection)
                features.append(feature)
                
                # 可視化（めがね専用カラーリング）
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                # めがね専用色：紫、オレンジ、緑
                colors = [(128, 0, 128), (0, 165, 255), (0, 255, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
                color = colors[i % len(colors)]
                
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 4)
                
                # ラベル（めがね専用マーク付き）
                model_mark = "👓🧠" if extractor.model is not None else "👓📝"
                label = f"Megane Person {i+1} ({conf:.2f}) {model_mark}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                
                cv2.rectangle(vis_image, (x1, y1-35), (x1+label_size[0]+10, y1), color, -1)
                cv2.putText(vis_image, label, (x1+5, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # 中央番号
                number_text = str(i+1)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                cv2.circle(vis_image, (center_x, center_y), 40, color, -1)
                cv2.circle(vis_image, (center_x, center_y), 40, (255, 255, 255), 4)
                
                text_size = cv2.getTextSize(number_text, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 4)[0]
                text_x = center_x - text_size[0] // 2
                text_y = center_y + text_size[1] // 2
                cv2.putText(vis_image, number_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 4)
                
                print(f"  👓 人物{i+1}: 特徴量抽出成功（サイズ: {len(feature)}）")
        
        # 画像保存
        filename = os.path.basename(image_path).split('.')[0]
        output_path = os.path.join(output_dir, f"{filename}_megane_detected.jpg")
        cv2.imwrite(output_path, vis_image)
        print(f"検出結果を保存: {output_path}")
    
    return final_detections, features

def analyze_megane_similarity(all_results, extractor):
    """megane特徴量類似度分析（シンプル版）"""
    print("\n" + "="*80)
    print("👓 Megane Similarity Analysis (megane01~03)")
    print("="*80)

    # 1. megane系列の特徴量収集
    print("📊 megane系列特徴量の収集...")
    
    # 各megane画像から最初の人物を抽出（メインの人物と仮定）
    megane_main_persons = {}
    
    for img_name in ['megane01', 'megane02', 'megane03']:
        if img_name in all_results and len(all_results[img_name]['features']) > 0:
            # 最初の人物（通常は最も信頼度が高い）をメイン人物とする
            megane_main_persons[f"{img_name}_main"] = all_results[img_name]['features'][0]
            print(f"  ✅ {img_name}: メイン人物の特徴量を抽出")
    
    if len(megane_main_persons) < 2:
        print("❌ 比較に必要なmegane特徴量が不足しています。")
        return

    # 2. megane間の類似度マトリックス作成
    print(f"\n👓 megane間の類似度マトリックス:")
    megane_ids = list(megane_main_persons.keys())
    
    print("    ", end="")
    for mid in megane_ids:
        print(f"{mid:>15}", end="")
    print()
    
    high_similarity_pairs = []
    similarity_matrix = []
    
    for i, id1 in enumerate(megane_ids):
        print(f"{id1:>15}", end="")
        row = []
        for j, id2 in enumerate(megane_ids):
            if i == j:
                print(f"{'1.000':>15}", end="")
                row.append(1.0)
            else:
                feat1 = megane_main_persons[id1]
                feat2 = megane_main_persons[id2]
                sim = extractor.calculate_cosine_similarity(feat1, feat2)
                print(f"{sim:>15.3f}", end="")
                row.append(sim)
                
                # 高類似度ペアを記録
                if sim > 0.7 and i < j:
                    high_similarity_pairs.append((id1, id2, sim))
        similarity_matrix.append(row)
        print()
    
    # 3. 同一人物判定分析
    print(f"\n🔍 同一人物判定分析:")
    
    if high_similarity_pairs:
        print(f"高類似度ペア（閾値0.7以上）:")
        high_similarity_pairs.sort(key=lambda x: x[2], reverse=True)
        
        for id1, id2, sim in high_similarity_pairs:
            print(f"\n【{id1} ↔ {id2}】")
            print(f"  類似度: {sim:.4f}")
            
            if sim > 0.9:
                print("  🎯 極めて高い類似度 → 同一人物の可能性が非常に高い")
            elif sim > 0.85:
                print("  🚀 非常に高い類似度 → 同一人物の可能性が高い")
            elif sim > 0.8:
                print("  ✅ 高い類似度 → 同一人物の可能性あり")
            else:
                print("  ⚠️ 中程度の類似度 → 要検討")
    else:
        print("高類似度ペア（0.7以上）は検出されませんでした")
        print("→ 各megane画像の人物は十分に区別されています")
    
    # 4. 最小類似度分析
    min_similarity = 1.0
    min_pair = None
    
    for i, id1 in enumerate(megane_ids):
        for j, id2 in enumerate(megane_ids):
            if i < j:
                sim = similarity_matrix[i][j]
                if sim < min_similarity:
                    min_similarity = sim
                    min_pair = (id1, id2)
    
    print(f"\n📈 分離性能評価:")
    print(f"  最小類似度: {min_similarity:.4f} ({min_pair[0]} ↔ {min_pair[1]})")
    
    if min_similarity < 0.5:
        print("  🎉 優秀な分離性能 - 各megane人物が明確に区別されています")
    elif min_similarity < 0.7:
        print("  ✅ 良好な分離性能 - 実用的なレベルです")
    else:
        print("  ⚠️ 注意が必要 - 一部megane人物の区別が曖昧です")
    
    # 5. 全人物統合分析
    print(f"\n🧑‍🤝‍🧑 全megane人物統合分析:")
    
    all_megane_features = []
    all_megane_labels = []
    
    for img_name in ['megane01', 'megane02', 'megane03']:
        if img_name in all_results:
            for i, feature in enumerate(all_results[img_name]['features']):
                all_megane_features.append(feature)
                all_megane_labels.append(f"{img_name}_人物{i+1}")
    
    print(f"総人物数: {len(all_megane_features)}人")
    
    if len(all_megane_features) > 1:
        print(f"\n全人物間類似度マトリックス:")
        print("    ", end="")
        for label in all_megane_labels:
            print(f"{label[:12]:>12}", end="")
        print()
        
        for i, label1 in enumerate(all_megane_labels):
            print(f"{label1[:12]:>12}", end="")
            for j, label2 in enumerate(all_megane_labels):
                if i == j:
                    print(f"{'1.00':>12}", end="")
                else:
                    feat1 = all_megane_features[i]
                    feat2 = all_megane_features[j]
                    sim = extractor.calculate_cosine_similarity(feat1, feat2)
                    print(f"{sim:>12.3f}", end="")
            print()

def main():
    print("👓 megane01~03 特徴量比較システムを開始します...")
    print(f"設定パラメータ: {CONFIG}")
    
    # YOLOv8モデルをロード
    model = YOLO('yolov8m.pt')
    
    # OSNet特徴量抽出器を初期化
    extractor = OSNetFeatureExtractor(CONFIG['reid_model'])
    
    # megane画像パス
    image_paths = {
        'megane01': "../videos/megane01.png",
        'megane02': "../videos/megane02.png", 
        'megane03': "../videos/megane03.png"
    }
    
    # 全結果を格納
    all_results = {}
    
    # 各画像を処理
    for img_name, img_path in image_paths.items():
        print("=" * 60)
        detections, features = detect_and_extract_features_megane(img_path, model, extractor, CONFIG)
        
        if detections and features:
            all_results[img_name] = {
                'detections': detections,
                'features': features,
                'path': img_path
            }
        else:
            print(f"⚠️  {img_name} の処理に失敗しました")
    
    if len(all_results) < 1:
        print("❌ 比較に必要なmegane画像データが不足しています")
        return
    
    # megane類似度分析を実行
    analyze_megane_similarity(all_results, extractor)
    
    # 結果サマリー
    print("\n" + "="*80)
    print("📋 megane特徴量比較結果サマリー")
    print("="*80)
    
    model_info = "👓🧠 OSNet" if extractor.model is not None else "👓📝 Fallback"
    print(f"使用モデル: {model_info}")
    print(f"処理画像数: {len(all_results)}枚")
    
    total_detections = sum(len(result['detections']) for result in all_results.values())
    print(f"総検出人数: {total_detections}人")
    
    for img_name, result in all_results.items():
        print(f"\n{img_name}: {len(result['detections'])}人検出（フィルタ後）")
        for i, detection in enumerate(result['detections']):
            x1, y1, x2, y2 = detection['bbox']
            conf = detection['conf']
            w, h = x2 - x1, y2 - y1
            print(f"  👓人物{i+1}: 座標({x1:.0f},{y1:.0f}) サイズ{w:.0f}x{h:.0f} 信頼度{conf:.3f}")
    
    print(f"\n🖼️  megane検出結果画像は 'detection_results_megane' フォルダに保存されました")
    print("📁 ファイル一覧:")
    for img_name in all_results.keys():
        print(f"   - {img_name}_megane_detected.jpg")

if __name__ == "__main__":
    main() 
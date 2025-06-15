import cv2
import numpy as np
from ultralytics import YOLO
import torch
import torch.nn.functional as F
import os
import time
from pathlib import Path

# boxmotのReIDモデルを使用
try:
    from boxmot.appearance.reid.factory import osnet_x0_25
    print("✅ boxmot imported successfully")
    BOXMOT_AVAILABLE = True
except ImportError:
    print("⚠️ boxmot not available, using fallback")
    BOXMOT_AVAILABLE = False

# 設定パラメータ（修正版）
CONFIG = {
    'min_confidence': 0.6,          # 最小信頼度閾値（少し上げる）
    'min_area_ratio': 0.01,         # 画像全体に対する最小面積比（上げる）
    'min_width': 60,                # 最小幅（上げる）
    'min_height': 120,              # 最小高さ（上げる）
    'cosine_threshold': 0.85,       # コサイン類似度閾値
    'nms_iou_threshold': 0.3,       # NMS IoU閾値（厳しくする）
    'reid_model': 'osnet_x0_25',    # OSNetモデル
    'gallery_threshold': 0.7,       # ギャラリーマッチング閾値（厳しくする）
    'max_gallery_size': 20,         # 1人当たりの最大ギャラリーサイズ（削減）
    'update_interval': 3,           # 特徴量更新間隔（フレーム数）
    'max_missing_frames': 9000,     # ID消失判定フレーム数（5分@30FPS想定: 5*60*30=9000）
    'video_fps': 10,                # 出力動画のFPS
    'iou_threshold': 0.3,           # 検出間のIoU重複除去閾値
}

class OSNetFeatureExtractor:
    """OSNetベースの特徴量抽出器（動画用）"""
    
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
        """フォールバック特徴量抽出"""
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

def calculate_iou(box1, box2):
    """IoU (Intersection over Union) を計算"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # 交差領域
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    # 交差面積
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # 各ボックスの面積
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # 和集合面積
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

class VideoGalleryTracker:
    """修正版：動画用ギャラリー追跡システム"""
    
    def __init__(self, extractor, config):
        self.extractor = extractor
        self.config = config
        self.galleries = {}  # ID -> {'features': [特徴量リスト], 'avg_feature': 平均特徴量, 'last_seen': フレーム番号, 'bbox': 最後のbbox}
        self.next_id = 1
        self.frame_count = 0
        self.color_map = {}  # ID -> 色のマッピング
        self.colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), 
                      (0, 255, 255), (128, 255, 0), (255, 128, 0), (128, 0, 255), (0, 128, 255)]
        self.debug_mode = True  # デバッグモード
    
    def update_frame(self, frame_num):
        """フレーム番号を更新"""
        self.frame_count = frame_num
        
        # 長期間見えていないIDを削除
        to_remove = []
        for person_id, data in self.galleries.items():
            if frame_num - data['last_seen'] > self.config['max_missing_frames']:
                to_remove.append(person_id)
        
        for person_id in to_remove:
            if self.debug_mode:
                print(f"  🗑️ ID {person_id}: {self.config['max_missing_frames']}フレーム見えないため削除")
            del self.galleries[person_id]
            if person_id in self.color_map:
                del self.color_map[person_id]
    
    def add_to_gallery(self, person_id, feature, bbox):
        """ギャラリーに特徴量を追加"""
        if person_id not in self.galleries:
            self.galleries[person_id] = {
                'features': [], 
                'avg_feature': None, 
                'last_seen': self.frame_count,
                'bbox': bbox
            }
            # 色を割り当て
            self.color_map[person_id] = self.colors[len(self.color_map) % len(self.colors)]
        
        self.galleries[person_id]['features'].append(feature)
        self.galleries[person_id]['last_seen'] = self.frame_count
        self.galleries[person_id]['bbox'] = bbox
        
        # ギャラリーサイズ制限
        if len(self.galleries[person_id]['features']) > self.config['max_gallery_size']:
            self.galleries[person_id]['features'].pop(0)
        
        self._update_average_feature(person_id)
    
    def _update_average_feature(self, person_id):
        """平均特徴量を更新"""
        features = self.galleries[person_id]['features']
        if features:
            avg_feature = np.mean(np.array(features), axis=0)
            avg_feature = avg_feature / np.linalg.norm(avg_feature)
            self.galleries[person_id]['avg_feature'] = avg_feature
    
    def track_detections(self, detections, features):
        """検出結果を追跡（修正版：一対一対応を保証）"""
        if not detections or not features:
            return []
        
        if self.debug_mode and self.frame_count % 30 == 0:
            print(f"  フレーム{self.frame_count}: {len(detections)}個の検出を処理中...")
        
        tracking_results = []
        
        if not self.galleries:
            # 新規ID発行（初回）
            for i, (detection, feature) in enumerate(zip(detections, features)):
                new_id = f"ID_{self.next_id:03d}"
                self.next_id += 1
                self.add_to_gallery(new_id, feature, detection['bbox'])
                tracking_results.append({
                    'id': new_id,
                    'bbox': detection['bbox'],
                    'conf': detection['conf'],
                    'similarity': 1.0,
                    'status': 'new'
                })
                if self.debug_mode and self.frame_count % 30 == 0:
                    print(f"    新規ID発行: {new_id}")
            return tracking_results
        
        # 類似度マトリックス作成
        similarity_matrix = []
        gallery_ids = list(self.galleries.keys())
        
        for i, feature in enumerate(features):
            similarities = []
            for person_id in gallery_ids:
                gallery_data = self.galleries[person_id]
                if gallery_data['avg_feature'] is not None:
                    sim = self.extractor.calculate_cosine_similarity(
                        feature, gallery_data['avg_feature']
                    )
                    similarities.append(sim if sim is not None else 0.0)
                else:
                    similarities.append(0.0)
            similarity_matrix.append(similarities)
        
        # ハンガリアンアルゴリズム（簡易版）による最適割り当て
        assignments = self._simple_assignment(similarity_matrix, self.config['gallery_threshold'])
        
        # 割り当て結果を処理
        assigned_gallery_indices = set()
        
        for detection_idx, gallery_idx in assignments.items():
            detection = detections[detection_idx]
            feature = features[detection_idx]
            
            if gallery_idx is not None:
                # 既存IDに割り当て
                person_id = gallery_ids[gallery_idx]
                similarity = similarity_matrix[detection_idx][gallery_idx]
                
                self.add_to_gallery(person_id, feature, detection['bbox'])
                tracking_results.append({
                    'id': person_id,
                    'bbox': detection['bbox'],
                    'conf': detection['conf'],
                    'similarity': similarity,
                    'status': 'matched'
                })
                assigned_gallery_indices.add(gallery_idx)
                
                if self.debug_mode and self.frame_count % 30 == 0:
                    print(f"    検出{detection_idx} -> {person_id} (類似度: {similarity:.3f})")
            else:
                # 新規ID発行
                new_id = f"ID_{self.next_id:03d}"
                self.next_id += 1
                self.add_to_gallery(new_id, feature, detection['bbox'])
                tracking_results.append({
                    'id': new_id,
                    'bbox': detection['bbox'],
                    'conf': detection['conf'],
                    'similarity': 0.0,
                    'status': 'new'
                })
                
                if self.debug_mode and self.frame_count % 30 == 0:
                    print(f"    検出{detection_idx} -> 新規ID {new_id}")
        
        return tracking_results
    
    def _simple_assignment(self, similarity_matrix, threshold):
        """簡易版ハンガリアンアルゴリズム（貪欲法）"""
        assignments = {}
        used_galleries = set()
        
        # 検出と既存IDのペアを類似度でソート
        pairs = []
        for det_idx, similarities in enumerate(similarity_matrix):
            for gal_idx, sim in enumerate(similarities):
                if sim >= threshold:
                    pairs.append((sim, det_idx, gal_idx))
        
        # 類似度降順でソート
        pairs.sort(reverse=True)
        
        # 貪欲法で割り当て
        for sim, det_idx, gal_idx in pairs:
            if det_idx not in assignments and gal_idx not in used_galleries:
                assignments[det_idx] = gal_idx
                used_galleries.add(gal_idx)
        
        # 割り当てられなかった検出はNone
        for det_idx in range(len(similarity_matrix)):
            if det_idx not in assignments:
                assignments[det_idx] = None
        
        return assignments
    
    def get_gallery_stats(self):
        """ギャラリー統計情報を取得"""
        stats = {}
        for person_id, data in self.galleries.items():
            stats[person_id] = {
                'gallery_size': len(data['features']),
                'last_seen': data['last_seen'],
                'frames_since_seen': self.frame_count - data['last_seen']
            }
        return stats
    
    def analyze_final_similarities(self):
        """最終ID間類似度分析"""
        if len(self.galleries) < 2:
            print("\n📊 最終類似度分析: アクティブIDが1個以下のため分析をスキップします")
            return
        
        print("\n" + "="*80)
        print("📊 最終ID間類似度マトリックス（静的比較）")
        print("="*80)
        
        # アクティブIDのリスト
        active_ids = list(self.galleries.keys())
        print(f"アクティブID数: {len(active_ids)}")
        
        # 各IDの平均特徴量を取得
        avg_features = {}
        for person_id in active_ids:
            gallery_data = self.galleries[person_id]
            if gallery_data['avg_feature'] is not None:
                avg_features[person_id] = gallery_data['avg_feature']
                print(f"  {person_id}: ギャラリーサイズ{len(gallery_data['features'])}, 平均特徴量準備完了")
        
        if len(avg_features) < 2:
            print("\n⚠️ 平均特徴量が利用可能なIDが不足しています")
            return
        
        # 類似度マトリックス作成
        print(f"\n🤝 ID間コサイン類似度マトリックス:")
        print("    ", end="")
        for pid in active_ids:
            print(f"{pid:>12}", end="")
        print()
        
        high_similarity_pairs = []
        similarity_matrix = []
        
        for i, id1 in enumerate(active_ids):
            print(f"{id1:>12}", end="")
            row = []
            
            for j, id2 in enumerate(active_ids):
                if i == j:
                    print(f"{'1.000':>12}", end="")
                    row.append(1.0)
                elif id1 in avg_features and id2 in avg_features:
                    feat1 = avg_features[id1]
                    feat2 = avg_features[id2]
                    sim = self.extractor.calculate_cosine_similarity(feat1, feat2)
                    print(f"{sim:>12.3f}", end="")
                    row.append(sim)
                    
                    # 高類似度ペアを記録（重複を避けるため i < j）
                    if sim > 0.7 and i < j:
                        high_similarity_pairs.append((id1, id2, sim))
                else:
                    print(f"{'N/A':>12}", end="")
                    row.append(0.0)
            
            similarity_matrix.append(row)
            print()
        
        # 高類似度ペアの分析
        if high_similarity_pairs:
            print(f"\n🎯 高類似度ペア分析（閾値0.7以上）:")
            high_similarity_pairs.sort(key=lambda x: x[2], reverse=True)
            
            for id1, id2, sim in high_similarity_pairs:
                print(f"\n【{id1} ↔ {id2}】")
                print(f"  類似度: {sim:.4f}")
                
                # ギャラリー情報
                gallery1 = self.galleries[id1]
                gallery2 = self.galleries[id2]
                print(f"  {id1}: ギャラリーサイズ{len(gallery1['features'])}, 最終確認フレーム{gallery1['last_seen']}")
                print(f"  {id2}: ギャラリーサイズ{len(gallery2['features'])}, 最終確認フレーム{gallery2['last_seen']}")
                
                if sim > 0.9:
                    print("  🔥 極めて高い類似度 → 同一人物の可能性が非常に高い")
                elif sim > 0.85:
                    print("  🚀 非常に高い類似度 → 同一人物の可能性が高い")
                elif sim > 0.8:
                    print("  ✅ 高い類似度 → 同一人物の可能性あり")
                else:
                    print("  ⚠️ 中程度の類似度 → 要注意ペア")
        else:
            print(f"\n✅ 高類似度ペア（0.7以上）は検出されませんでした")
            print("  → 各IDは十分に区別されています")
        
        # 分離性能評価
        if len(active_ids) >= 2:
            # 最小ID間距離を計算
            min_similarity = 1.0
            min_pair = None
            
            for i, id1 in enumerate(active_ids):
                for j, id2 in enumerate(active_ids):
                    if i < j and id1 in avg_features and id2 in avg_features:
                        sim = similarity_matrix[i][j]
                        if sim < min_similarity:
                            min_similarity = sim
                            min_pair = (id1, id2)
            
            print(f"\n📈 分離性能評価:")
            print(f"  最小ID間類似度: {min_similarity:.4f} ({min_pair[0]} ↔ {min_pair[1]})")
            
            if min_similarity < 0.5:
                print("  🎉 優秀な分離性能 - IDが明確に区別されています")
            elif min_similarity < 0.7:
                print("  ✅ 良好な分離性能 - 実用的なレベルです")
            elif min_similarity < 0.8:
                print("  ⚠️ 注意が必要 - 一部IDの区別が曖昧です")
            else:
                print("  🚨 分離性能不足 - 閾値調整が必要です")
        
        # 推奨設定
        if high_similarity_pairs:
            max_sim = max([sim for _, _, sim in high_similarity_pairs])
            recommended_threshold = max_sim * 0.95  # 最高類似度の95%
            print(f"\n💡 推奨ギャラリー閾値: {recommended_threshold:.3f}")
            print(f"  （現在の設定: {self.config['gallery_threshold']}）")

def filter_detections_advanced(detections, image_shape, config):
    """高度な検出結果フィルタリング（IoU-NMS付き）"""
    h, w = image_shape[:2]
    image_area = h * w
    
    # 基本フィルタリング
    filtered = []
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        conf = detection['conf']
        
        if conf < config['min_confidence']:
            continue
        
        width = x2 - x1
        height = y2 - y1
        area = width * height
        area_ratio = area / image_area
        
        if (width < config['min_width'] or 
            height < config['min_height'] or 
            area_ratio < config['min_area_ratio']):
            continue
        
        filtered.append(detection)
    
    # IoU-NMSによる重複除去
    if len(filtered) <= 1:
        return filtered
    
    # 信頼度でソート
    filtered.sort(key=lambda x: x['conf'], reverse=True)
    
    final_detections = []
    for i, detection in enumerate(filtered):
        keep = True
        for kept_detection in final_detections:
            iou = calculate_iou(detection['bbox'], kept_detection['bbox'])
            if iou > config['iou_threshold']:
                keep = False
                break
        
        if keep:
            final_detections.append(detection)
    
    return final_detections

def process_video_frame(frame, model, tracker, extractor, config):
    """フレーム処理（修正版）"""
    # YOLOv8で人物検出
    results = model(frame, classes=0, conf=config['min_confidence'], 
                   iou=config['nms_iou_threshold'], verbose=False)
    
    raw_detections = []
    
    if results[0].boxes is not None:
        boxes = results[0].boxes
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            
            raw_detections.append({
                'bbox': [x1, y1, x2, y2],
                'conf': conf
            })
    
    # 高度なフィルタリング
    filtered_detections = filter_detections_advanced(raw_detections, frame.shape, config)
    
    # 特徴量抽出
    valid_detections = []
    valid_features = []
    
    for detection in filtered_detections:
        feature = extractor.extract_features(frame, detection['bbox'])
        if feature is not None:
            valid_detections.append(detection)
            valid_features.append(feature)
    
    # 追跡実行
    tracking_results = tracker.track_detections(valid_detections, valid_features)
    
    return tracking_results

def draw_tracking_results_debug(frame, tracking_results, tracker, frame_num):
    """追跡結果を描画（デバッグ情報付き）"""
    vis_frame = frame.copy()
    
    # 重複ID検証
    used_ids = set()
    duplicate_ids = set()
    
    for result in tracking_results:
        person_id = result['id']
        if person_id in used_ids:
            duplicate_ids.add(person_id)
        used_ids.add(person_id)
    
    for result in tracking_results:
        person_id = result['id']
        x1, y1, x2, y2 = map(int, result['bbox'])
        conf = result['conf']
        similarity = result['similarity']
        status = result['status']
        
        # 色を取得（重複IDは赤で警告）
        if person_id in duplicate_ids:
            color = (0, 0, 255)  # 赤色で警告
        else:
            color = tracker.color_map.get(person_id, (255, 255, 255))
        
        # バウンディングボックス（重複IDは太い線）
        thickness = 5 if person_id in duplicate_ids else 3
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, thickness)
        
        # ラベル作成
        status_mark = "🆕" if status == 'new' else "✅"
        model_mark = "🧠" if tracker.extractor.model is not None else "📝"
        duplicate_mark = "⚠️" if person_id in duplicate_ids else ""
        gallery_size = len(tracker.galleries[person_id]['features'])
        
        label = f"{person_id} ({conf:.2f}) {status_mark}{model_mark}{duplicate_mark}"
        sub_label = f"Gallery:{gallery_size} Sim:{similarity:.3f}"
        
        # ラベル背景
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        sub_label_size = cv2.getTextSize(sub_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        
        cv2.rectangle(vis_frame, (x1, y1-60), (x1+max(label_size[0], sub_label_size[0])+10, y1), color, -1)
        cv2.putText(vis_frame, label, (x1+5, y1-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_frame, sub_label, (x1+5, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # フレーム情報とアラート表示
    info_text = f"Frame: {frame_num} | Active IDs: {len(tracker.galleries)} | Detections: {len(tracking_results)}"
    if duplicate_ids:
        info_text += f" | ⚠️DUPLICATE: {duplicate_ids}"
    
    cv2.putText(vis_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return vis_frame

def main():
    print("🎬 修正版：動画ギャラリー追跡システムを開始します...")
    print(f"設定パラメータ: {CONFIG}")
    
    # 動画ファイルパス
    video_path = "../videos/megane-test.mov"
        # ../videos/classtest04.mp4: 教室の動画　重なり強
        # ../videos/classtest06-resized.mov: 教室の動画 重なり弱
        # ../videos/test04.mov: 広場の動画 重なり中
    output_path = "output_gallery_tracking_fixed.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ 動画ファイルが見つかりません: {video_path}")
        return
    
    # YOLOv8モデルをロード
    model = YOLO('yolov8m.pt')
    
    # OSNet特徴量抽出器を初期化
    extractor = OSNetFeatureExtractor(CONFIG['reid_model'])
    
    # ギャラリー追跡システムを初期化
    tracker = VideoGalleryTracker(extractor, CONFIG)
    
    # 動画読み込み
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ 動画ファイルを開けません: {video_path}")
        return
    
    # 動画情報取得
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 動画情報: {width}x{height}, {fps}FPS, {total_frames}フレーム")
    
    # 出力動画設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, CONFIG['video_fps'], (width, height))
    
    frame_num = 0
    process_start_time = time.time()
    
    print("🚀 修正版動画処理を開始...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            tracker.update_frame(frame_num)
            
            # フレーム処理
            tracking_results = process_video_frame(frame, model, tracker, extractor, CONFIG)
            
            # 結果描画（デバッグ付き）
            vis_frame = draw_tracking_results_debug(frame, tracking_results, tracker, frame_num)
            
            # 出力動画に書き込み
            out.write(vis_frame)
            
            # 進捗表示
            if frame_num % 30 == 0:
                elapsed_time = time.time() - process_start_time
                fps_current = frame_num / elapsed_time
                progress = (frame_num / total_frames) * 100
                
                print(f"  📊 フレーム {frame_num}/{total_frames} ({progress:.1f}%) | "
                      f"処理速度: {fps_current:.1f}FPS | アクティブID数: {len(tracker.galleries)}")
                
                # ギャラリー統計表示
                stats = tracker.get_gallery_stats()
                for person_id, stat in stats.items():
                    print(f"    {person_id}: ギャラリー{stat['gallery_size']}個, "
                          f"最終確認から{stat['frames_since_seen']}フレーム")
    
    except KeyboardInterrupt:
        print("\n⏹️ ユーザーにより中断されました")
    
    finally:
        # リソースを解放
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        total_time = time.time() - process_start_time
        avg_fps = frame_num / total_time
        
        print(f"\n🎬 修正版動画処理完了!")
        print(f"📊 処理統計:")
        print(f"  総フレーム数: {frame_num}")
        print(f"  総処理時間: {total_time:.1f}秒")
        print(f"  平均処理速度: {avg_fps:.1f}FPS")
        print(f"  発行ID数: {tracker.next_id - 1}")
        print(f"  最終アクティブID数: {len(tracker.galleries)}")
        print(f"  出力動画: {output_path}")
        
        # 最終ギャラリー統計
        print(f"\n🏛️ 最終ギャラリー統計:")
        stats = tracker.get_gallery_stats()
        for person_id, stat in stats.items():
            print(f"  {person_id}: ギャラリーサイズ{stat['gallery_size']}, "
                  f"最終確認フレーム{stat['last_seen']}")
        
        # 最終ID間類似度分析
        tracker.analyze_final_similarities()

if __name__ == "__main__":
    main() 
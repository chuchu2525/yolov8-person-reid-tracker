# YOLOv8 と Re-ID による人物追跡システム

YOLOv8 によるリアルタイム人物検出と、Re-ID（人物再識別）モデル(OSNet)を組み合わせた、高精度な動画人物追跡システムです。検出した人物の外見的特徴を「ギャラリー」として管理し、フレームをまたいで同一人物を追跡することで、ID のスイッチングを抑制し、安定した追跡を実現します。

## 主な機能

- **YOLOv8 によるリアルタイム人物検出**: 高速かつ高精度な人物検出。
- **高精度な人物再識別**: `boxmot`ライブラリの OSNet モデルを利用した、外見特徴による人物の再識別。
- **ギャラリーベースの追跡**: 一時的に隠れたり見えなくなったりした人物でも、ID を維持して追跡。
- **高度なフィルタリング**: IoU ベースの NMS（非最大抑制）により、不要な検出ボックスを除去。
- **詳細な最終分析**: 追跡終了後、ID 間の類似度マトリックスを含む詳細な分析レポートを出力。

## セットアップ

### 必要なもの

- Python 3.8 以上
- PyTorch
- OpenCV

### インストール手順

1.  **リポジトリのクローン:**
    このスクリプトファイルがあれば、準備完了です。

2.  **必要な Python パッケージのインストール:**
    仮想環境の利用を推奨します。
    ```bash
    pip install ultralytics torch opencv-python boxmot numpy
    ```

## 使い方

1.  **動画ファイルの準備:**
    処理したい動画ファイルを、スクリプトからアクセス可能な場所に配置します。

2.  **スクリプト内の動画パスを更新:**
    `video_gallery_tracker_fixed.py` を開き、`video_path` の値を、使用する動画ファイルのパスに書き換えてください。デフォルトでは、親ディレクトリの `videos` フォルダ内を指しています。

    ```python
    # 700行目あたり
    video_path = "../videos/your_video_file.mp4"
    ```

3.  **スクリプトの実行:**
    ターミナルで `yolov8` ディレクトリに移動し、以下のコマンドを実行します。

    ```bash
    python video_gallery_tracker_fixed.py
    ```

4.  **出力の確認:**
    処理後の動画（追跡ボックスや ID が描画されたもの）が、`output_gallery_tracking_fixed.mp4` という名前で同じディレクトリ内に保存されます。

## 設定

`video_gallery_tracker_fixed.py` スクリプトの冒頭にある `CONFIG` 辞書の値を調整することで、トラッカーの挙動を細かく設定できます。

```python
CONFIG = {
    'min_confidence': 0.6,          # 検出の最小信頼度
    'min_area_ratio': 0.01,         # 画像全体に対する最小面積の割合
    'min_width': 60,                # 検出ボックスの最小幅
    'min_height': 120,              # 検出ボックスの最小高さ
    'cosine_threshold': 0.85,       # 特徴量のコサイン類似度の閾値
    'nms_iou_threshold': 0.3,       # NMSで使うIoUの閾値
    'reid_model': 'osnet_x0_25',    # Re-IDモデル名
    'gallery_threshold': 0.7,       # ギャラリーとのマッチング閾値
    'max_gallery_size': 20,         # 1人あたりの最大ギャラリーサイズ
    'update_interval': 3,           # 特徴量更新のフレーム間隔
    'max_missing_frames': 300,      # IDが消失したと判断するまでのフレーム数
    'video_fps': 10,                # 出力動画のFPS
    'iou_threshold': 0.3,           # 検出間の重複除去で使うIoUの閾値
}
```

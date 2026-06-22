# Cricket LBW Decision Review System 🏏

> An automated LBW (Leg Before Wicket) decision engine powered by YOLOv8 object detection, Kalman filtering, and ensemble machine learning — implementing **MCC Law 36** with computer vision.

---

## 📋 Overview

The Cricket LBW Decision Review System processes match footage to automatically determine LBW decisions without human intervention. It detects the ball and stumps in video, tracks the ball's trajectory using a Kalman filter, predicts where it would go, and applies both rule-based (MCC Law 36) and ML-based reasoning to deliver a verdict.

**Pipeline at a glance:**

```
Video Input
    │
    ▼
YOLOv8 Detection  ──→  Ball + Stump Bounding Boxes
    │
    ▼
Kalman Filter     ──→  Smoothed Ball Trajectory
    │
    ▼
Auto-Calibration  ──→  Stump Positions (YOLO or trajectory fallback)
    │
    ▼
Bounce & Impact Detection
    │
    ▼
Polynomial Trajectory Prediction
    │
    ├──→  Rule Engine (MCC Law 36)  ──→  Verdict
    └──→  Ensemble ML (XGBoost + RF) ──→  Verdict
```

---

## ✨ Key Features

- **Custom YOLOv8 Model** — Trained to detect cricket ball and stumps (mAP50: 0.648)
- **Kalman Filter Tracking** — Robust ball tracking with noise reduction across frames
- **Auto-Calibration** — Uses YOLO-detected stump positions with trajectory-based fallback
- **14 Engineered Features** — Position, velocity, curvature, stump intersection, and more
- **Ensemble ML** — XGBoost + Random Forest trained with SMOTE class balancing
- **MCC Law 36 Engine** — Pitch in line + Impact in line + Hitting stumps → OUT



## 🗂️ Project Structure

```
Cricket-LBW-Decision-Review-System/
├── models/
│   └── cricket_detector.pt       # Trained YOLOv8 model
├── src/
│   ├── config.py                 # Paths and hyperparameters
│   ├── detector.py               # YOLOv8 ball + stump detection
│   ├── tracker.py                # Kalman filter tracking
│   ├── calibration.py            # Auto stump calibration
│   ├── trajectory.py             # Bounce/impact detection + prediction
│   ├── feature_extraction.py     # 14-feature engineering
│   ├── ml_models.py              # XGBoost + RF ensemble
│   ├── decision_engine.py        # MCC Law 36 rule engine
│   └── pipeline.py               # End-to-end orchestration
├── scripts/
│   ├── process_videos.py         # Batch video feature extraction
│   ├── train_models.py           # Model training CLI
│   └── run_pipeline.py           # Single video inference CLI
├── utils/
│   ├── visualization.py          # Trajectory and confusion matrix plots
│   └── helpers.py                # Video I/O, normalization utilities
├── notebooks/
│   └── LBW_System.ipynb          # Full interactive notebook
├── tests/
│   ├── test_detector.py
│   └── test_pipeline.py
└── docs/
    └── *.png                     # Architecture and result diagrams
```

---

## 🚀 Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/yourusername/Cricket-LBW-Decision-Review-System.git
cd Cricket-LBW-Decision-Review-System
pip install -r requirements.txt
```

### 2. Run on a Single Video

```python
from src.pipeline import run_full_pipeline

result = run_full_pipeline("path/to/match_clip.mp4")
print(f"Verdict: {result['decision']['Verdict']}")
# → Verdict: OUT
```

Or via CLI:

```bash
python scripts/run_pipeline.py --video match_clip.mp4 --output-json result.json
```

### 3. Batch Process Videos

```python
from scripts.process_videos import batch_test_videos

results = batch_test_videos(video_list=["video1.mp4", "video2.mp4"])
```

Or via CLI:

```bash
python scripts/process_videos.py --input-dir ./videos --output-dir ./outputs --model-path models/cricket_detector.pt
```

### 4. Train Models on Your Own Data

```bash
# First extract features from labeled videos
python scripts/process_videos.py --input-dir ./labeled_videos --output-dir ./outputs

# Then train
python scripts/train_models.py --data outputs/features_<timestamp>.csv --output-dir outputs/models
```

> **Video labeling convention:** include `out` or `not_out` in filenames for auto-labeling (e.g. `clip_out_01.mp4`, `clip_not_out_07.mp4`).

---

## ⚙️ How It Works

### 1. Detection
YOLOv8 runs on every frame to detect the ball (class 0) and stumps (class 1), returning bounding boxes and confidence scores.

### 2. Tracking
A 4-state Kalman filter `[x, y, vx, vy]` smooths noisy detections and interpolates across frames where the ball is occluded.

### 3. Calibration
Stump positions are determined either from YOLO detections (preferred) or estimated from ball trajectory when stump detections are sparse.

### 4. Trajectory Prediction
After identifying the bounce point (sign change in vertical velocity) and impact point (last detection), a quadratic polynomial is fitted to the post-bounce trajectory and extrapolated to the stump plane.

### 5. Decision

**Rule-based (MCC Law 36):**
```
Pitched in line AND Impact in line AND Trajectory hitting stumps → OUT
```

**ML ensemble:**  
XGBoost + Random Forest trained on 14 features. Predictions are averaged and thresholded at 0.5. Both models and the rule engine run in parallel; agreements are flagged.

---

## 🛠️ Requirements

Key dependencies (see `requirements.txt` for pinned versions):

```
torch>=2.0.0
ultralytics>=8.0.0       # YOLOv8
scikit-learn>=1.2.0
xgboost>=1.7.0
imbalanced-learn>=0.10.0 # SMOTE
opencv-python>=4.8.0
filterpy>=1.4.5          # Kalman filter
scipy>=1.10.0
numpy>=1.24.0
pandas>=2.0.0
```

Python 3.8+ required.

---

## 📁 Model

Place your trained YOLO model at:

```
models/cricket_detector.pt
```

The model is not included in this repository due to file size. To train your own, prepare a YOLO-format dataset with `ball` and `stump` classes and run:

```bash
yolo train model=yolov8n.pt data=yolo_dataset/data.yaml epochs=100 imgsz=640
```

---

## 📝 Configuration

Edit `src/config.py` to set paths and thresholds:

```python
CONFIDENCE_THRESHOLD = 0.25   # YOLO confidence cutoff
KALMAN_PROCESS_NOISE = 0.1    # Kalman filter tuning
KALMAN_MEASUREMENT_NOISE = 1.0
```

---

## 🧪 Testing

```bash
pytest tests/
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [FilterPy](https://github.com/rlabbe/filterpy) — Kalman filter implementation
- [MCC Laws of Cricket](https://www.lords.org/mcc/the-laws-of-cricket) — Law 36 (LBW)

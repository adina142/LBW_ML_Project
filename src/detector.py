"""
YOLO-based ball and stump detection module
"""

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm
import torch


class CricketDetector:
    """Detects cricket ball and stumps using YOLOv8"""
    
    def __init__(self, model_path, conf_threshold=0.25):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 Detector using device: {self.device}")
    
    def detect_ball_and_stumps(self, video_path):
        """
        Detect both cricket ball AND stumps
        
        Args:
            video_path: Path to video file
            
        Returns:
            ball_df: DataFrame with ball detections
            stump_df: DataFrame with stump detections
            video_info: Dict with video metadata
        """
        print(f"\n🎥 Processing: {os.path.basename(video_path)}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Could not open video!")
            return None, None, None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"   {width}x{height}, {total_frames} frames, {fps:.1f} FPS")
        
        ball_detections = []
        stump_detections = []
        frame_count = 0
        
        with tqdm(total=total_frames, desc="Detecting ball+stumps") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                pbar.update(1)
                
                results = self.model(frame, verbose=False, conf=self.conf_threshold)
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None and len(boxes) > 0:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            
                            if class_id == 0:  # Ball class ID
                                ball_detections.append({
                                    'Frame': frame_count,
                                    'Ball_X': (x1 + x2) / 2,
                                    'Ball_Y': (y1 + y2) / 2,
                                    'Confidence': confidence
                                })
                            elif class_id == 1:  # Stump class ID
                                stump_detections.append({
                                    'Frame': frame_count,
                                    'Stump_X1': x1, 'Stump_Y1': y1,
                                    'Stump_X2': x2, 'Stump_Y2': y2,
                                    'Stump_Center_X': (x1 + x2) / 2,
                                    'Confidence': confidence
                                })
        
        cap.release()
        
        ball_df = pd.DataFrame(ball_detections) if ball_detections else pd.DataFrame()
        stump_df = pd.DataFrame(stump_detections) if stump_detections else pd.DataFrame()
        
        if len(ball_df) == 0:
            print("⚠️ No ball detections!")
            return None, stump_df, {'width': width, 'height': height, 'fps': fps}
        
        print(f"✅ Ball: {len(ball_df)}/{total_frames} frames ({len(ball_df)/total_frames*100:.1f}%), "
              f"Avg conf: {ball_df['Confidence'].mean():.3f}")
        if len(stump_df) > 0:
            print(f"✅ Stumps: {len(stump_df)} detections, "
                  f"Avg conf: {stump_df['Confidence'].mean():.3f}")
        
        return ball_df, stump_df, {'width': width, 'height': height, 'fps': fps}
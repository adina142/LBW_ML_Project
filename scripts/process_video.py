#!/usr/bin/env python
"""
Batch process videos for feature extraction
"""

import os
import sys
import argparse
import pandas as pd
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import load_config, CricketDetector, apply_kalman_tracking
from src import AutoCalibrator, extract_batch_features


def find_videos(videos_dir):
    """Find all video files in directory"""
    videos = []
    for root, dirs, files in os.walk(videos_dir):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov', '.MOV')):
                videos.append(os.path.join(root, file))
    return videos


def label_videos(videos):
    """Auto-label videos based on filename"""
    labels = {}
    for vp in videos:
        vn = os.path.basename(vp).lower()
        if 'not_out' in vn or 'notout' in vn:
            labels[vp] = 0
        elif 'out' in vn:
            labels[vp] = 1
        else:
            labels[vp] = None  # Unknown
    return labels


def main():
    parser = argparse.ArgumentParser(description='Process cricket videos for LBW analysis')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Directory containing videos')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory for results')
    parser.add_argument('--model-path', type=str, default='models/cricket_detector.pt',
                       help='Path to YOLO model')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--max-videos', type=int, default=50,
                       help='Maximum videos to process')
    
    args = parser.parse_args()
    
    # Setup
    config = load_config()
    config.VIDEOS_DIR = args.input_dir
    config.OUTPUT_DIR = args.output_dir
    config.MODEL_PATH = args.model_path
    config.CONFIDENCE_THRESHOLD = args.conf
    config.setup_directories()
    
    # Find videos
    videos = find_videos(args.input_dir)
    labels = label_videos(videos)
    
    # Filter labeled videos
    labeled_videos = {k: v for k, v in labels.items() if v is not None}
    video_list = list(labeled_videos.keys())[:args.max_videos]
    
    print(f"📹 Found {len(video_list)} labeled videos")
    
    # Process
    detector = CricketDetector(config.MODEL_PATH, config.CONFIDENCE_THRESHOLD)
    
    features_list = []
    failed_videos = []
    
    for i, vp in enumerate(video_list, 1):
        print(f"\n[{i}/{len(video_list)}] {os.path.basename(vp)}")
        try:
            # Detect
            ball_df, stump_df, vid_info = detector.detect_ball_and_stumps(vp)
            if ball_df is None or len(ball_df) == 0:
                failed_videos.append(vp)
                continue
            
            # Track
            track_df = apply_kalman_tracking(ball_df)
            
            # Calibrate
            calibrator = AutoCalibrator(vid_info['width'], vid_info['height'])
            stump_data = calibrator.calibrate(stump_df, track_df)
            
            # Extract features
            from src.trajectory import detect_bounce_point, detect_impact_point, predict_trajectory
            from src.feature_extraction import extract_features
            
            pitch_data = detect_bounce_point(track_df)
            impact_data = detect_impact_point(track_df, pitch_data)
            pred_data = predict_trajectory(track_df, impact_data, stump_data)
            features = extract_features(track_df, pitch_data, impact_data, stump_data, pred_data)
            
            features['Video_Name'] = os.path.basename(vp)
            features['Label'] = labels[vp]
            features['Calibration_Method'] = stump_data['Calibration_Method']
            
            features_list.append(features)
            print(f"   ✅ Success")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:80]}")
            failed_videos.append(vp)
    
    # Save results
    if features_list:
        df = pd.DataFrame(features_list)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{config.OUTPUT_DIR}/features_{timestamp}.csv"
        df.to_csv(output_path, index=False)
        print(f"\n✅ Saved {len(df)} samples to {output_path}")
        print(f"   ❌ {len(failed_videos)} failed")
    else:
        print("\n❌ No features extracted!")


if __name__ == "__main__":
    main()
"""
Feature extraction for ML models
"""

import numpy as np
import pandas as pd
from .trajectory import check_stump_intersection


def extract_features(tracking_df, pitch_data, impact_data, stump_data, prediction_data):
    """
    Extract features for ML model
    
    Args:
        tracking_df: DataFrame with Kalman tracking
        pitch_data: Dict with bounce point
        impact_data: Dict with impact point
        stump_data: Dict with stump positions
        prediction_data: Dict with predicted trajectory
        
    Returns:
        Dict with extracted features
    """
    features = {}
    
    # Position features
    features['Pitch_X'] = pitch_data['Pitch_X']
    features['Pitch_Y'] = pitch_data['Pitch_Y']
    features['Impact_X'] = impact_data['Impact_X']
    features['Impact_Y'] = impact_data['Impact_Y']
    
    # Stump center
    scx = stump_data['Stump_Middle_X']
    scy = (stump_data['Stump_Y_Bottom'] + stump_data['Stump_Y_Top']) / 2
    
    # Distance features
    features['Distance_Pitch_To_Stumps'] = np.sqrt(
        (features['Pitch_X'] - scx) ** 2 + (features['Pitch_Y'] - scy) ** 2
    )
    features['Distance_Impact_To_Stumps'] = np.sqrt(
        (features['Impact_X'] - scx) ** 2 + (features['Impact_Y'] - scy) ** 2
    )
    
    # Velocity features
    vx = np.nanmean(np.abs(tracking_df['Kalman_VX']))
    vy = np.nanmean(np.abs(tracking_df['Kalman_VY']))
    features['Ball_Speed'] = np.sqrt(vx ** 2 + vy ** 2)
    features['Ball_Angle'] = np.degrees(np.arctan2(vy, vx))
    
    imp_idx = min(impact_data['Impact_Frame'] - 1, len(tracking_df) - 1)
    features['Horizontal_Velocity'] = tracking_df['Kalman_VX'].iloc[imp_idx] if imp_idx < len(tracking_df) else vx
    features['Vertical_Velocity'] = tracking_df['Kalman_VY'].iloc[imp_idx] if imp_idx < len(tracking_df) else vy
    
    # Trajectory features
    features['Trajectory_Curvature'] = prediction_data['Poly_Coeffs'][0] if prediction_data is not None else 0
    
    # Stump intersection
    hits, ix, iy = check_stump_intersection(prediction_data, stump_data)
    features['Predicted_Stump_Intersection_X'] = ix
    features['Predicted_Stump_Intersection_Y'] = iy
    features['Wicket_Hit_Flag'] = 1 if hits else 0
    
    # Clean NaN values
    for k in features:
        if np.isnan(features[k]) or np.isinf(features[k]):
            features[k] = 0.0
    
    return features


def extract_batch_features(video_list, detector, tracker, calibrator, config):
    """
    Extract features from multiple videos
    
    Args:
        video_list: List of video paths
        detector: CricketDetector instance
        tracker: BallTracker instance
        calibrator: AutoCalibrator instance
        config: LBWConfig instance
        
    Returns:
        DataFrame with extracted features
    """
    from .trajectory import detect_bounce_point, detect_impact_point, predict_trajectory
    from .calibration import AutoCalibrator
    
    features_list = []
    failed_videos = []
    
    for video_path in video_list:
        print(f"\n📹 Processing: {os.path.basename(video_path)}")
        
        try:
            # Detect ball and stumps
            ball_df, stump_df, video_info = detector.detect_ball_and_stumps(video_path)
            
            if ball_df is None or len(ball_df) == 0:
                print("   ❌ Detection failed")
                failed_videos.append(video_path)
                continue
            
            # Kalman tracking
            track_df = tracker.apply_kalman_tracking(ball_df)
            
            # Detect key points
            pitch_data = detect_bounce_point(track_df)
            impact_data = detect_impact_point(track_df, pitch_data)
            
            # Calibrate
            calibrator = AutoCalibrator(video_info['width'], video_info['height'])
            stump_data = calibrator.calibrate(stump_df, track_df)
            
            # Predict trajectory
            pred_data = predict_trajectory(track_df, impact_data, stump_data)
            
            # Extract features
            features = extract_features(track_df, pitch_data, impact_data, stump_data, pred_data)
            features['Video_Name'] = os.path.basename(video_path)
            features['Calibration_Method'] = stump_data['Calibration_Method']
            
            features_list.append(features)
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:80]}")
            failed_videos.append(video_path)
    
    return pd.DataFrame(features_list), failed_videos
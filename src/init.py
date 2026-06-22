"""
Main pipeline for LBW Decision Review System
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime

from .config import LBWConfig, load_config
from .detector import CricketDetector
from .tracker import apply_kalman_tracking
from .calibration import AutoCalibrator
from .trajectory import (
    detect_bounce_point, detect_impact_point, 
    predict_trajectory, visualize_trajectory
)
from .feature_extraction import extract_features
from .ml_models import LBWMLModels
from .decision_engine import LBWDecisionEngine


class LBWPipeline:
    """Complete LBW decision pipeline"""
    
    def __init__(self, config=None, load_ml=True):
        self.config = config or load_config()
        self.config.setup_directories()
        
        # Initialize components
        self.detector = CricketDetector(
            self.config.MODEL_PATH, 
            self.config.CONFIDENCE_THRESHOLD
        )
        
        self.ml_models = LBWMLModels()
        if load_ml and os.path.exists(f"{self.config.OUTPUT_DIR}/models"):
            try:
                self.ml_models.load(f"{self.config.OUTPUT_DIR}/models")
                print("✅ ML models loaded")
            except:
                print("⚠️ ML models not found, using rule-based only")
                self.ml_models = None
    
    def process_video(self, video_path):
        """
        Process a single video through the complete pipeline
        
        Returns:
            Dict with results
        """
        print("\n" + "="*70)
        print(f"🚀 RUNNING LBW PIPELINE: {os.path.basename(video_path)}")
        print("="*70)
        
        # 1. Detect ball and stumps
        print("\n[1/6] Detecting ball & stumps...")
        ball_df, stump_df, video_info = self.detector.detect_ball_and_stumps(video_path)
        
        if ball_df is None or len(ball_df) == 0:
            # Try with lower threshold
            self.detector.conf_threshold = 0.15
            ball_df, stump_df, video_info = self.detector.detect_ball_and_stumps(video_path)
            self.detector.conf_threshold = self.config.CONFIDENCE_THRESHOLD
            
            if ball_df is None or len(ball_df) == 0:
                print("❌ Detection failed!")
                return None
        
        # 2. Kalman tracking
        print("\n[2/6] Kalman tracking...")
        track_df = apply_kalman_tracking(ball_df, self.config.KALMAN_DT)
        
        # 3. Detect key points
        print("\n[3/6] Key points...")
        pitch_data = detect_bounce_point(track_df)
        impact_data = detect_impact_point(track_df, pitch_data)
        
        # 4. Calibrate
        print("\n[4/6] Calibrating...")
        calibrator = AutoCalibrator(video_info['width'], video_info['height'])
        stump_data = calibrator.calibrate(stump_df, track_df)
        
        # 5. Trajectory prediction
        print("\n[5/6] Trajectory prediction...")
        pred_data = predict_trajectory(track_df, impact_data, stump_data)
        hits_stumps = visualize_trajectory(track_df, pitch_data, impact_data, pred_data, stump_data)
        
        # 6. Feature extraction
        print("\n[6/6] Feature extraction & Decision...")
        features = extract_features(track_df, pitch_data, impact_data, stump_data, pred_data)
        
        # ML prediction
        ml_result = None
        if self.ml_models is not None:
            try:
                ml_result = self.ml_models.predict(features)
                print(f"\n🤖 ML: XGB={'🟥 OUT' if ml_result['xgb_pred']==1 else '🟩 NOT OUT'} "
                      f"| RF={'🟥 OUT' if ml_result['rf_pred']==1 else '🟩 NOT OUT'} "
                      f"| Ensemble: {'OUT' if ml_result['ensemble_pred'] else 'NOT OUT'} "
                      f"({ml_result['ensemble_confidence']:.0f}%)")
            except Exception as e:
                print(f"\n⚠️ ML unavailable: {str(e)[:50]}")
        
        # Rule-based decision
        engine = LBWDecisionEngine(stump_data)
        decision = engine.make_decision(
            pitch_data['Pitch_X'], 
            impact_data['Impact_X'], 
            hits_stumps
        )
        
        # Combine results
        result = {
            'video_name': os.path.basename(video_path),
            'pitch': pitch_data,
            'impact': impact_data,
            'stumps': stump_data,
            'features': features,
            'decision': decision,
            'ml_result': ml_result,
            'hits_stumps': hits_stumps,
            'track_df': track_df,
            'prediction_data': pred_data
        }
        
        # Print summary
        self._print_summary(result)
        
        return result
    
    def _print_summary(self, result):
        """Print summary of results"""
        print(f"\n{'='*70}")
        print(f"📊 SUMMARY")
        print(f"{'='*70}")
        print(f"   Video: {result['video_name']}")
        print(f"   Pitch: ({result['pitch']['Pitch_X']:.0f},{result['pitch']['Pitch_Y']:.0f})")
        print(f"   Impact: ({result['impact']['Impact_X']:.0f},{result['impact']['Impact_Y']:.0f})")
        print(f"   Stumps: X={result['stumps']['Stump_Middle_X']:.0f} ({result['stumps']['Calibration_Method']})")
        print(f"   Hitting: {'YES' if result['hits_stumps'] else 'NO'}")
        
        if result['ml_result'] is not None:
            ml_verdict = 'OUT' if result['ml_result']['ensemble_pred'] else 'NOT OUT'
            print(f"   ML: {ml_verdict} ({result['ml_result']['ensemble_confidence']:.0f}%)")
        
        print(f"   RULE: {result['decision']['Verdict']}")
        
        if result['ml_result'] is not None:
            ml_verdict = 'OUT' if result['ml_result']['ensemble_pred'] else 'NOT OUT'
            agreement = '✅' if ml_verdict == result['decision']['Verdict'] else '⚠️ DISAGREE'
            print(f"   Agreement: {agreement}")
        
        print("="*70)


def run_full_pipeline(video_path, drive_root=None):
    """Convenience function to run pipeline"""
    config = load_config(drive_root) if drive_root else load_config()
    pipeline = LBWPipeline(config)
    return pipeline.process_video(video_path)
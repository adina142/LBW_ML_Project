"""
Kalman Filter tracking for cricket ball
"""

import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter


class BallTracker:
    """Kalman Filter for ball tracking"""
    
    def __init__(self, dt=1.0, process_noise=0.1, measurement_noise=1.0):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # State transition matrix
        self.kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise
        q = process_noise
        self.kf.Q = np.array([
            [q, 0, 0, 0],
            [0, q, 0, 0],
            [0, 0, q*0.1, 0],
            [0, 0, 0, q*0.1]
        ])
        
        # Measurement noise
        self.kf.R = np.array([
            [measurement_noise, 0],
            [0, measurement_noise]
        ])
        
        self.kf.P *= 100
        self.kf.x = np.array([0, 0, 0, 0])
        self.initialized = False
    
    def predict(self):
        """Predict next state"""
        self.kf.predict()
        return self.kf.x[:2]
    
    def update(self, measurement):
        """Update with measurement"""
        if not self.initialized:
            self.kf.x[:2] = measurement
            self.initialized = True
        self.kf.update(measurement)
        return self.kf.x[:2]
    
    def get_state(self):
        """Get current state [x, y, vx, vy]"""
        return self.kf.x


def apply_kalman_tracking(detections_df, dt=1.0):
    """
    Apply Kalman filtering to ball detections
    
    Args:
        detections_df: DataFrame with 'Frame', 'Ball_X', 'Ball_Y' columns
        dt: Time step
        
    Returns:
        DataFrame with Kalman filtered positions and velocities
    """
    print("\n🔧 Kalman Filter...")
    
    tracker = BallTracker(dt=dt)
    n_frames = int(detections_df['Frame'].max())
    
    kx = np.full(n_frames + 1, np.nan)
    ky = np.full(n_frames + 1, np.nan)
    kvx = np.full(n_frames + 1, np.nan)
    kvy = np.full(n_frames + 1, np.nan)
    has_det = np.zeros(n_frames + 1, dtype=bool)
    
    det_dict = dict(zip(detections_df['Frame'], 
                       zip(detections_df['Ball_X'], detections_df['Ball_Y'])))
    
    for f in range(1, n_frames + 1):
        tracker.predict()
        if f in det_dict:
            tracker.update(np.array(det_dict[f]))
            has_det[f] = True
        s = tracker.get_state()
        kx[f] = s[0]
        ky[f] = s[1]
        kvx[f] = s[2]
        kvy[f] = s[3]
    
    result_df = pd.DataFrame({
        'Frame': range(1, n_frames + 1),
        'Kalman_X': kx[1:],
        'Kalman_Y': ky[1:],
        'Kalman_VX': kvx[1:],
        'Kalman_VY': kvy[1:],
        'Has_Detection': has_det[1:]
    })
    
    merged = pd.merge(result_df, 
                     detections_df[['Frame', 'Ball_X', 'Ball_Y']], 
                     on='Frame', how='left')
    
    print(f"✅ Kalman complete. {n_frames} frames.")
    return merged
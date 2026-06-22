"""
Auto-calibration using YOLO-detected stumps or trajectory fallback
"""

import numpy as np


class AutoCalibrator:
    """
    Auto-calibrates stump positions using YOLO detections or trajectory
    """
    
    def __init__(self, video_width, video_height):
        self.width = video_width
        self.height = video_height
    
    def calibrate(self, stump_df=None, tracking_df=None):
        """
        Main calibration method
        
        Returns:
            Dict with stump position data
        """
        if stump_df is not None and len(stump_df) >= 3:
            return self._calibrate_from_yolo_stumps(stump_df)
        else:
            return self._calibrate_from_trajectory(tracking_df)
    
    def _calibrate_from_yolo_stumps(self, stump_df):
        """Use YOLO-detected stumps for precise calibration"""
        x1 = stump_df['Stump_X1'].median()
        x2 = stump_df['Stump_X2'].median()
        y1 = stump_df['Stump_Y1'].min()
        y2 = stump_df['Stump_Y2'].max()
        center_x = (x1 + x2) / 2
        stump_width = x2 - x1
        stump_spacing = stump_width / 6
        
        print(f"\n📐 YOLO-Detected Stumps:")
        print(f"   Center X: {center_x:.0f}, Width: {stump_width:.0f}px")
        print(f"   Bounds: X={x1:.0f}-{x2:.0f}, Y={y1:.0f}-{y2:.0f}")
        
        return {
            'Stump_Off_X': float(center_x - stump_spacing),
            'Stump_Middle_X': float(center_x),
            'Stump_Leg_X': float(center_x + stump_spacing),
            'Stump_Y_Top': float(y1),
            'Stump_Y_Bottom': float(y2),
            'Stump_Width': float(stump_width),
            'Stump_Left_Boundary': float(x1),
            'Stump_Right_Boundary': float(x2),
            'Video_Width': self.width,
            'Video_Height': self.height,
            'Calibration_Method': 'yolo_detected'
        }
    
    def _calibrate_from_trajectory(self, tracking_df):
        """Fallback: estimate from ball trajectory"""
        if tracking_df is not None:
            valid_x = tracking_df['Kalman_X'].values
            valid_x = valid_x[(np.abs(valid_x) > 1) & (~np.isnan(valid_x))]
            if len(valid_x) >= 10:
                middle_x = float(np.median(valid_x[int(len(valid_x) * 0.75):]))
            elif len(valid_x) >= 3:
                middle_x = float(np.median(valid_x))
            else:
                middle_x = self.width * 0.50
        else:
            middle_x = self.width * 0.50
        
        stump_top = self.height * 0.35
        stump_bottom = self.height * 0.60
        stump_spacing = int(self.width * 0.04)
        
        if stump_bottom - stump_top < 100:
            stump_top = self.height * 0.30
            stump_bottom = self.height * 0.65
        
        wm = stump_spacing * 0.8
        
        print(f"\n📐 Trajectory-based Calibration:")
        print(f"   Center X: {middle_x:.0f}")
        print(f"   Y: {stump_top:.0f}-{stump_bottom:.0f}")
        
        return {
            'Stump_Off_X': float(middle_x - stump_spacing),
            'Stump_Middle_X': float(middle_x),
            'Stump_Leg_X': float(middle_x + stump_spacing),
            'Stump_Y_Top': float(stump_top),
            'Stump_Y_Bottom': float(stump_bottom),
            'Stump_Width': float(stump_spacing * 2),
            'Stump_Left_Boundary': float(middle_x - stump_spacing - wm),
            'Stump_Right_Boundary': float(middle_x + stump_spacing + wm),
            'Video_Width': self.width,
            'Video_Height': self.height,
            'Calibration_Method': 'trajectory_based'
        }
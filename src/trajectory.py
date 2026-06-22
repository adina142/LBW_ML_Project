"""
Trajectory prediction and visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import uniform_filter1d
import pandas as pd


def detect_bounce_point(tracking_df):
    """
    Detect ball bounce point from trajectory
    
    Args:
        tracking_df: DataFrame with Kalman tracking data
        
    Returns:
        Dict with bounce point info
    """
    print("\n🔍 Detecting Bounce...")
    
    vy = tracking_df['Kalman_VY'].values
    valid_idx = np.where(tracking_df['Has_Detection'].values)[0]
    
    if len(valid_idx) < 3:
        bidx = len(tracking_df) // 3
    else:
        vy_v = vy[valid_idx]
        vy_s = uniform_filter1d(np.nan_to_num(vy_v, nan=0), size=5)
        if len(vy_s) > 2:
            zc = np.where(np.diff(np.signbit(vy_s)))[0]
            if len(zc) > 0:
                bidx = valid_idx[min(zc[0], len(valid_idx) - 1)]
            else:
                bidx = valid_idx[np.argmax(tracking_df['Kalman_Y'].values[valid_idx][:int(0.6 * len(valid_idx))])]
        else:
            bidx = valid_idx[len(valid_idx) // 3]
    
    bidx = min(bidx, len(tracking_df) - 2)
    
    pitch_data = {
        'Pitch_Frame': int(tracking_df['Frame'].iloc[bidx]),
        'Pitch_X': float(tracking_df['Kalman_X'].iloc[bidx]),
        'Pitch_Y': float(tracking_df['Kalman_Y'].iloc[bidx])
    }
    
    print(f"✅ Bounce Frame {pitch_data['Pitch_Frame']}: "
          f"({pitch_data['Pitch_X']:.0f},{pitch_data['Pitch_Y']:.0f})")
    return pitch_data


def detect_impact_point(tracking_df, pitch_data):
    """
    Detect impact point from trajectory
    
    Args:
        tracking_df: DataFrame with Kalman tracking data
        pitch_data: Dict with bounce point info
        
    Returns:
        Dict with impact point info
    """
    print("\n🎯 Detecting Impact...")
    
    valid_frames = tracking_df[tracking_df['Has_Detection'] == True]
    if len(valid_frames) > 0:
        imp_idx = min(valid_frames.index[-1] + 2, len(tracking_df) - 1)
    else:
        imp_idx = len(tracking_df) - 1
    
    impact_data = {
        'Impact_Frame': int(tracking_df['Frame'].iloc[imp_idx]),
        'Impact_X': float(tracking_df['Kalman_X'].iloc[imp_idx]),
        'Impact_Y': float(tracking_df['Kalman_Y'].iloc[imp_idx])
    }
    
    print(f"✅ Impact Frame {impact_data['Impact_Frame']}: "
          f"({impact_data['Impact_X']:.0f},{impact_data['Impact_Y']:.0f})")
    return impact_data


def predict_trajectory(tracking_df, impact_data, stump_data, n_points=200):
    """
    Predict future trajectory using polynomial fitting
    
    Args:
        tracking_df: DataFrame with Kalman tracking data
        impact_data: Dict with impact point
        stump_data: Dict with stump positions
        n_points: Number of points to predict
        
    Returns:
        Dict with predicted trajectory
    """
    print("\n🔮 Predicting Trajectory...")
    
    imp_idx = impact_data['Impact_Frame'] - 1
    valid_track = tracking_df[
        (tracking_df['Kalman_X'].abs() > 1) & 
        (tracking_df['Kalman_Y'].abs() > 1)
    ].copy()
    
    if len(valid_track) < 5:
        print("⚠️ Insufficient points")
        return None
    
    pre_impact = valid_track[valid_track.index <= imp_idx]
    if len(pre_impact) < 5:
        pre_impact = valid_track.iloc[:max(5, len(valid_track))]
    
    x_fit = pre_impact['Kalman_X'].values[-40:]
    y_fit = pre_impact['Kalman_Y'].values[-40:]
    
    if len(x_fit) < 5:
        print("⚠️ Insufficient fitting points")
        return None
    
    try:
        # Normalize for numerical stability
        x_mean = np.mean(x_fit)
        x_std = np.std(x_fit) if np.std(x_fit) > 0 else 1
        x_norm = (x_fit - x_mean) / x_std
        coeffs_norm = np.polyfit(x_norm, y_fit, 2)
        
        # Convert back to original scale
        a = coeffs_norm[0] / (x_std ** 2)
        b = coeffs_norm[1] / x_std - 2 * coeffs_norm[0] * x_mean / (x_std ** 2)
        c = coeffs_norm[2] - coeffs_norm[1] * x_mean / x_std + coeffs_norm[0] * (x_mean ** 2) / (x_std ** 2)
        coeffs = np.array([a, b, c])
    except:
        coeffs = np.polyfit(x_fit, y_fit, 1)
        coeffs = np.array([0, coeffs[0], coeffs[1]])
    
    # Calculate R-squared
    poly_func = np.poly1d(coeffs)
    y_pred = poly_func(x_fit)
    ss_res = np.sum((y_fit - y_pred) ** 2)
    ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Extend trajectory to stumps
    impact_x = impact_data['Impact_X']
    stump_mx = stump_data['Stump_Middle_X']
    x_extend = np.linspace(impact_x, stump_mx + 300, n_points)
    y_extend = poly_func(x_extend)
    
    pred_data = {
        'X_Predicted': x_extend,
        'Y_Predicted': y_extend,
        'Poly_Coeffs': coeffs,
        'R_Squared': r_squared
    }
    
    print(f"✅ Trajectory predicted (R²={r_squared:.3f})")
    return pred_data


def check_stump_intersection(prediction_data, stump_data):
    """
    Check if predicted trajectory hits stumps
    
    Returns:
        Tuple (hits_stumps, intersection_x, intersection_y)
    """
    if prediction_data is None:
        return False, 0, 0
    
    x_pred = prediction_data['X_Predicted']
    y_pred = prediction_data['Y_Predicted']
    
    sl = stump_data['Stump_Left_Boundary']
    sr = stump_data['Stump_Right_Boundary']
    st = stump_data['Stump_Y_Top']
    sb = stump_data['Stump_Y_Bottom']
    
    for x, y in zip(x_pred, y_pred):
        if (sl <= x <= sr) and (st <= y <= sb):
            return True, x, y
    
    # Check line segment intersection
    for i in range(len(x_pred) - 1):
        x1, y1 = x_pred[i], y_pred[i]
        x2, y2 = x_pred[i + 1], y_pred[i + 1]
        if (min(x1, x2) <= sr and max(x1, x2) >= sl and
            min(y1, y2) <= sb and max(y1, y2) >= st):
            return True, (x1 + x2) / 2, (y1 + y2) / 2
    
    return False, 0, 0


def visualize_trajectory(tracking_df, pitch_data, impact_data, prediction_data, stump_data):
    """
    Visualize ball trajectory with stumps
    
    Returns:
        bool: Whether trajectory hits stumps
    """
    hits_stumps, ix, iy = check_stump_intersection(prediction_data, stump_data)
    
    if prediction_data is None:
        return False
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot tracking data
    valid = tracking_df[tracking_df['Has_Detection'] == True]
    ax.scatter(valid['Kalman_X'], valid['Kalman_Y'], 
              c='blue', s=15, alpha=0.6, label='Detected Ball')
    ax.plot(tracking_df['Kalman_X'], tracking_df['Kalman_Y'], 
            'b-', alpha=0.3, linewidth=1, label='Kalman Track')
    
    # Key points
    ax.scatter(pitch_data['Pitch_X'], pitch_data['Pitch_Y'], 
              c='orange', s=150, marker='s', label='Pitch Point', 
              zorder=5, edgecolors='black')
    ax.scatter(impact_data['Impact_X'], impact_data['Impact_Y'], 
              c='red', s=150, marker='X', label='Impact Point', 
              zorder=5, edgecolors='black')
    
    # Predicted trajectory
    ax.plot(prediction_data['X_Predicted'], prediction_data['Y_Predicted'], 
            'r--', linewidth=3, label='Predicted Trajectory')
    
    # Stump zone
    from matplotlib.patches import Rectangle
    stump_rect = Rectangle(
        (stump_data['Stump_Left_Boundary'], stump_data['Stump_Y_Top']),
        stump_data['Stump_Right_Boundary'] - stump_data['Stump_Left_Boundary'],
        stump_data['Stump_Y_Bottom'] - stump_data['Stump_Y_Top'],
        linewidth=2, edgecolor='brown', facecolor='brown', 
        alpha=0.4, label='Stump Zone'
    )
    ax.add_patch(stump_rect)
    
    # Stump lines
    for sx in [stump_data['Stump_Off_X'], stump_data['Stump_Middle_X'], 
               stump_data['Stump_Leg_X']]:
        ax.axvline(x=sx, ymin=0.3, ymax=0.7, color='darkred', 
                  linewidth=2, linestyle='-')
    
    if hits_stumps and ix > 0:
        ax.scatter(ix, iy, c='lime', s=300, marker='*', 
                  label='HITTING STUMPS!', zorder=10, edgecolors='black')
    
    ax.set_xlabel('X Position (pixels)', fontsize=12)
    ax.set_ylabel('Y Position (pixels)', fontsize=12)
    ax.set_title(f'Ball Trajectory - {"🟥 HITTING" if hits_stumps else "🟩 MISSING"}', 
                fontsize=14, weight='bold')
    ax.legend(loc='upper right')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return hits_stumps
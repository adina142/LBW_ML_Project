"""
Visualization utilities for LBW system
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns


def plot_trajectory(tracking_df, pitch_data, impact_data, pred_data, stump_data):
    """Plot ball trajectory with stumps"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Tracking data
    valid = tracking_df[tracking_df['Has_Detection'] == True]
    ax.scatter(valid['Kalman_X'], valid['Kalman_Y'], 
              c='blue', s=15, alpha=0.6, label='Detected')
    ax.plot(tracking_df['Kalman_X'], tracking_df['Kalman_Y'], 
            'b-', alpha=0.3, linewidth=1, label='Kalman')
    
    # Key points
    ax.scatter(pitch_data['Pitch_X'], pitch_data['Pitch_Y'], 
              c='orange', s=150, marker='s', label='Pitch')
    ax.scatter(impact_data['Impact_X'], impact_data['Impact_Y'], 
              c='red', s=150, marker='X', label='Impact')
    
    # Prediction
    if pred_data is not None:
        ax.plot(pred_data['X_Predicted'], pred_data['Y_Predicted'], 
                'r--', linewidth=3, label='Predicted')
    
    # Stumps
    rect = patches.Rectangle(
        (stump_data['Stump_Left_Boundary'], stump_data['Stump_Y_Top']),
        stump_data['Stump_Width'], 
        stump_data['Stump_Y_Bottom'] - stump_data['Stump_Y_Top'],
        linewidth=2, edgecolor='brown', facecolor='brown', alpha=0.4
    )
    ax.add_patch(rect)
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.invert_yaxis()
    ax.legend()
    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm, title='Confusion Matrix'):
    """Plot confusion matrix"""
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    return fig


def plot_feature_importance(model, feature_names, title='Feature Importance'):
    """Plot feature importance"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    ax.bar(range(len(importances)), importances[indices])
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
    ax.set_title(title)
    plt.tight_layout()
    return fig
"""
Helper utilities for LBW system
"""

import os
import cv2
import numpy as np
from datetime import datetime


def get_video_info(video_path):
    """Get video metadata"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    }
    cap.release()
    return info


def extract_frame(video_path, frame_number):
    """Extract a specific frame from video"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def create_timestamp():
    """Create timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_divide(a, b, default=0):
    """Safe division with fallback"""
    if b == 0:
        return default
    return a / b


def moving_average(data, window_size):
    """Compute moving average"""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def normalize_to_range(data, min_val, max_val):
    """Normalize data to range [min_val, max_val]"""
    if len(data) == 0:
        return data
    data_min, data_max = np.min(data), np.max(data)
    if data_max == data_min:
        return np.full_like(data, (min_val + max_val) / 2)
    return min_val + (data - data_min) * (max_val - min_val) / (data_max - data_min)


def find_videos(directory):
    """Find all video files in directory recursively"""
    videos = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                videos.append(os.path.join(root, file))
    return videos


def label_from_filename(filename):
    """Extract label from filename"""
    filename = os.path.basename(filename).lower()
    if 'not_out' in filename or 'notout' in filename:
        return 0
    if 'out' in filename:
        return 1
    return None
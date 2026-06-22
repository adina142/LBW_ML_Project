"""
Configuration for LBW Decision Review System
"""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LBWConfig:
    """Configuration class for LBW system"""
    
    # Paths
    DRIVE_ROOT: str = "/content/drive/MyDrive/LBW_2"
    MODEL_PATH: str = ""
    YAML_PATH: str = ""
    OUTPUT_DIR: str = ""
    DATA_DIR: str = ""
    VIDEOS_DIR: str = ""
    
    # Detection parameters
    CONFIDENCE_THRESHOLD: float = 0.25
    BALL_CLASS_ID: int = 0
    STUMP_CLASS_ID: int = 1
    
    # Kalman parameters
    KALMAN_DT: float = 1.0
    KALMAN_PROCESS_NOISE: float = 0.1
    KALMAN_MEASUREMENT_NOISE: float = 1.0
    
    # Model paths
    RF_MODEL_PATH: str = ""
    XGB_MODEL_PATH: str = ""
    SCALER_PATH: str = ""
    
    def __post_init__(self):
        """Initialize derived paths"""
        self.MODEL_PATH = f"{self.DRIVE_ROOT}/models/cricket_detector.pt"
        self.YAML_PATH = f"{self.DRIVE_ROOT}/yolo_dataset/data.yaml"
        self.OUTPUT_DIR = f"{self.DRIVE_ROOT}/outputs"
        self.DATA_DIR = f"{self.DRIVE_ROOT}/data"
        self.VIDEOS_DIR = f"{self.DRIVE_ROOT}/videos"
        self.RF_MODEL_PATH = f"{self.OUTPUT_DIR}/models/rf_model.pkl"
        self.XGB_MODEL_PATH = f"{self.OUTPUT_DIR}/models/xgb_model.pkl"
        self.SCALER_PATH = f"{self.OUTPUT_DIR}/models/scaler.pkl"
    
    def setup_directories(self):
        """Create necessary directories"""
        for d in [self.OUTPUT_DIR, self.DATA_DIR, 
                  f"{self.OUTPUT_DIR}/figures",
                  f"{self.OUTPUT_DIR}/models"]:
            os.makedirs(d, exist_ok=True)


def load_config(drive_root="/content/drive/MyDrive/LBW_2"):
    """Load configuration with custom root path"""
    return LBWConfig(DRIVE_ROOT=drive_root)
"""
Unit tests for detector module
"""

import unittest
import os
import numpy as np
from src.detector import CricketDetector


class TestDetector(unittest.TestCase):
    
    def setUp(self):
        self.model_path = "models/cricket_detector.pt"
        if not os.path.exists(self.model_path):
            self.skipTest("Model file not found")
        self.detector = CricketDetector(self.model_path)
    
    def test_initialization(self):
        self.assertIsNotNone(self.detector.model)
        self.assertEqual(self.detector.conf_threshold, 0.25)
    
    def test_detect_ball_and_stumps(self):
        # This would need a test video
        pass


if __name__ == '__main__':
    unittest.main()
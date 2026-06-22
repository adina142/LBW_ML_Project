"""
Integration tests for pipeline
"""

import unittest
import os
from src import LBWPipeline, load_config


class TestPipeline(unittest.TestCase):
    
    def setUp(self):
        self.config = load_config()
        self.pipeline = LBWPipeline(self.config)
    
    def test_config(self):
        self.assertIsNotNone(self.config)
        self.assertTrue(os.path.exists(self.config.MODEL_PATH))
    
    # Add more tests...


if __name__ == '__main__':
    unittest.main()
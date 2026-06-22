#!/usr/bin/env python
"""
Run LBW pipeline on a single video
"""

import os
import sys
import argparse
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import run_full_pipeline, load_config


def main():
    parser = argparse.ArgumentParser(description='Run LBW pipeline on video')
    parser.add_argument('--video', type=str, required=True,
                       help='Path to video file')
    parser.add_argument('--output-json', type=str, default=None,
                       help='Save results to JSON file')
    parser.add_argument('--drive-root', type=str, default='/content/drive/MyDrive/LBW_2',
                       help='Root directory for data')
    
    args = parser.parse_args()
    
    # Check video
    if not os.path.exists(args.video):
        print(f"❌ Video not found: {args.video}")
        sys.exit(1)
    
    # Run pipeline
    result = run_full_pipeline(args.video, args.drive_root)
    
    if result is None:
        print("❌ Pipeline failed!")
        sys.exit(1)
    
    # Save results
    if args.output_json:
        # Convert to serializable format
        serializable = {
            'video_name': result['video_name'],
            'decision': result['decision'],
            'hits_stumps': result['hits_stumps'],
            'features': result['features']
        }
        
        if result['ml_result'] is not None:
            serializable['ml_result'] = {
                'ensemble_pred': result['ml_result']['ensemble_pred'],
                'ensemble_confidence': result['ml_result']['ensemble_confidence']
            }
        
        with open(args.output_json, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"✅ Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
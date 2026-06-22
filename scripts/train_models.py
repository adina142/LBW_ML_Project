#!/usr/bin/env python
"""
Train ML models on extracted features
"""

import os
import sys
import argparse
import pandas as pd
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import LBWMLModels, load_config


def main():
    parser = argparse.ArgumentParser(description='Train ML models for LBW')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to features CSV file')
    parser.add_argument('--output-dir', type=str, default='outputs/models',
                       help='Output directory for models')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test split proportion')
    
    args = parser.parse_args()
    
    # Load data
    print(f"📊 Loading data from {args.data}")
    data = pd.read_csv(args.data)
    print(f"   {len(data)} samples, {data['Label'].sum()} OUT, {len(data) - data['Label'].sum()} NOT OUT")
    
    # Train models
    ml_models = LBWMLModels()
    results = ml_models.train(data, test_size=args.test_size)
    
    # Save models
    os.makedirs(args.output_dir, exist_ok=True)
    ml_models.save(args.output_dir)
    print(f"\n✅ Models saved to {args.output_dir}")
    
    # Save results
    results_df = pd.DataFrame([results])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(f"{args.output_dir}/training_results_{timestamp}.csv", index=False)


if __name__ == "__main__":
    main()
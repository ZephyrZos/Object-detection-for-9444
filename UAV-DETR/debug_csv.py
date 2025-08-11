#!/usr/bin/env python3
"""
Debug CSV file to understand the data structure
"""

import pandas as pd
import glob

# Find the results file
results_files = glob.glob('runs/train/uavdetr_single_gpu_memory_opt_*/results.csv')
if results_files:
    csv_file = results_files[0]
    print(f"Reading: {csv_file}")
    
    # Read raw CSV
    df = pd.read_csv(csv_file)
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Show first few rows raw
    print(f"\nFirst 3 rows (raw):")
    for i in range(min(3, len(df))):
        print(f"Row {i}: {df.iloc[i].to_dict()}")
    
    # Check specific columns
    loss_cols = ['train/giou_loss', 'train/cls_loss', 'train/l1_loss']
    for col in loss_cols:
        if col in df.columns:
            print(f"\n{col}:")
            print(f"  Type: {df[col].dtype}")
            print(f"  Unique values: {df[col].unique()}")
            print(f"  First 5 values: {df[col].head().tolist()}")
    
    metric_cols = ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
    for col in metric_cols:
        if col in df.columns:
            print(f"\n{col}:")
            print(f"  Type: {df[col].dtype}")
            print(f"  Unique values: {df[col].unique()}")
            print(f"  First 5 values: {df[col].head().tolist()}")
    
else:
    print("No results files found!")
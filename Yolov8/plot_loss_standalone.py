#!/usr/bin/env python3
import sys
import os
import glob
import yaml
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_config(config_path='config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def find_latest_results_csv():
    # Find the latest results.csv
    csv_patterns = [
        "visdrone_detection/train*/results.csv",
        "runs/detect/train*/results.csv"
    ]
    
    all_csvs = []
    for pattern in csv_patterns:
        all_csvs.extend(glob.glob(pattern))
    
    if all_csvs:
        return max(all_csvs, key=os.path.getmtime)
    return None

def plot_training_results(csv_path):
    if not os.path.exists(csv_path):
        print(f"❌ CSV file does not exist: {csv_path}")
        return False
    
    try:
        df = pd.read_csv(csv_path)
        print(f"📊 Reading training data: {len(df)} epochs")
        
        # Ensure save directory exists
        save_dir = os.path.dirname(csv_path)
        save_path = os.path.join(save_dir, 'results.png')
        
        print(f"📁 Save directory: {save_dir}")
        print(f"💾 Save path: {save_path}")
        
        # Check directory permissions
        if not os.path.exists(save_dir):
            print(f"❌ Directory does not exist: {save_dir}")
            # Try to save in current directory
            save_path = 'results.png'
            print(f"🔄 Changed to save in current directory: {save_path}")
        elif not os.access(save_dir, os.W_OK):
            print(f"❌ Directory has no write permission: {save_dir}")
            # Try to save in current directory
            save_path = 'results.png'
            print(f"🔄 Changed to save in current directory: {save_path}")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Results', fontsize=16)
        
        columns = df.columns.tolist()
        print(f"📋 Available columns: {columns}")
        
        # Box Loss
        if 'train/box_loss' in columns and 'val/box_loss' in columns:
            axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', color='blue', linewidth=2)
            axes[0, 0].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', color='red', linewidth=2)
            axes[0, 0].set_title('Box Loss', fontsize=12)
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Classification Loss
        if 'train/cls_loss' in columns and 'val/cls_loss' in columns:
            axes[0, 1].plot(df['epoch'], df['train/cls_loss'], label='Train Cls Loss', color='blue', linewidth=2)
            axes[0, 1].plot(df['epoch'], df['val/cls_loss'], label='Val Cls Loss', color='red', linewidth=2)
            axes[0, 1].set_title('Classification Loss', fontsize=12)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # mAP
        map50_col = None
        map5095_col = None
        for col in columns:
            if 'mAP50' in col and 'mAP50-95' not in col:
                map50_col = col
            elif 'mAP50-95' in col:
                map5095_col = col
        
        if map50_col and map5095_col:
            axes[1, 0].plot(df['epoch'], df[map50_col], label='mAP@0.5', color='green', linewidth=2)
            axes[1, 0].plot(df['epoch'], df[map5095_col], label='mAP@0.5:0.95', color='orange', linewidth=2)
            axes[1, 0].set_title('mAP Metrics', fontsize=12)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('mAP')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate
        lr_cols = [col for col in columns if col.startswith('lr/')]
        if lr_cols:
            axes[1, 1].plot(df['epoch'], df[lr_cols[0]], label='Learning Rate', color='purple', linewidth=2)
            axes[1, 1].set_title('Learning Rate', fontsize=12)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('LR')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Try to save, if it fails try other locations
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Training chart generated: {save_path}")
        except Exception as save_error:
            print(f"❌ Failed to save to {save_path}: {save_error}")
            # Try to save to current directory
            fallback_path = 'training_results.png'
            try:
                plt.savefig(fallback_path, dpi=300, bbox_inches='tight')
                print(f"✅ Training chart generated: {fallback_path}")
            except Exception as fallback_error:
                print(f"❌ Failed to save to {fallback_path} as well: {fallback_error}")
                return False
        
        plt.close()
        return True
        
    except Exception as e:
        print(f"❌ Error generating chart: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = find_latest_results_csv()
        if not csv_path:
            print("❌ No results.csv file found")
            print("Please manually specify path: python3 plot_loss_standalone.py path/to/results.csv")
            return
        print(f"🔍 Found latest results.csv: {csv_path}")
    
    plot_training_results(csv_path)

if __name__ == "__main__":
    main()

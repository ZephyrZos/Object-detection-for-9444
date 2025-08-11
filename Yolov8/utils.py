import os
import glob
import yaml
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-GUI environment setting
import matplotlib.pyplot as plt
from pathlib import Path
import re  # Add regex module

def load_config(config_path='config.yaml'):
    """Load configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_device(config_device):
    """Auto detect the most suitable device"""
    if config_device != 'auto':
        return config_device

    if torch.cuda.is_available():
        # Check if there are multiple GPUs
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            print(f"Detected {gpu_count} GPUs, will use multi-GPU training")
            return [0, 1]  # Use first two GPUs
        else:
            return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

def find_latest_weights(config):
    """Auto find latest training weights"""
    project_name = config.get('project_name', 'runs/detect')

    # Find all possible weight files
    weight_patterns = [
        f"{project_name}/train*/weights/best.pt",
        f"{project_name}/train*/weights/last.pt",
        "runs/detect/train*/weights/best.pt",  # Backup path
        "runs/detect/train*/weights/last.pt"   # Backup path
    ]

    all_weights = []
    for pattern in weight_patterns:
        all_weights.extend(glob.glob(pattern))

    if all_weights:
        # Sort by modification time, return the latest
        latest_weight = max(all_weights, key=os.path.getmtime)
        return latest_weight

    return None

def find_latest_train_dir(config):
    """Find latest training directory (based on numbers in directory name, not modification time)"""
    project_name = config.get('project_name', 'runs/detect')

    # Find all training directories
    train_patterns = [
        f"{project_name}/train*",
        "runs/detect/train*"  # Backup path
    ]

    all_train_dirs = []
    for pattern in train_patterns:
        all_train_dirs.extend(glob.glob(pattern))

    if all_train_dirs:
        # Find latest based on numbers in directory name
        def extract_train_number(path):
            # Get directory name
            dirname = os.path.basename(path)
            # Extract number after 'train'
            match = re.match(r'train(\d+)', dirname)
            if match:
                return int(match.group(1))
            else:
                # If only 'train' without number, consider it as 0
                return 0 if dirname == 'train' else -1

        # Filter out invalid directory names, then sort by number
        valid_dirs = [d for d in all_train_dirs if extract_train_number(d) >= 0]
        if valid_dirs:
            latest_train_dir = max(valid_dirs, key=extract_train_number)
            return latest_train_dir

    return None

def find_latest_results_csv(config):
    """Find results.csv file in latest training directory"""
    latest_train_dir = find_latest_train_dir(config)
    print(f"latest_train_dir: {latest_train_dir}")
    if latest_train_dir:
        results_csv = os.path.join(latest_train_dir, 'results.csv')
        if os.path.exists(results_csv):
            return results_csv
    return None

def plot_training_results(csv_path, save_path=None):
    """Generate training charts from results.csv - YOLO official style"""
    if not os.path.exists(csv_path):
        print(f"❌ CSV file does not exist: {csv_path}")
        return False

    try:
        # Read data
        df = pd.read_csv(csv_path)
        print(f"📊 Reading training data: {len(df)} epochs")

        # If save path not specified, use CSV's directory
        if save_path is None:
            save_path = os.path.join(os.path.dirname(csv_path), 'results.png')

        # Check column names
        columns = df.columns.tolist()
        print(f"📋 Available columns: {columns}")

                        # Create charts - 2 rows 5 columns layout
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle('Training Results', fontsize=16, y=0.98)

        # Set colors
        results_color = '#1f77b4'  # Blue
        smooth_color = '#ff7f0e'   # Orange

        # Top row: Training loss and metrics (5 items)
        top_metrics = [
            ('train/box_loss', 'train/box_loss'),
            ('train/cls_loss', 'train/cls_loss'),
            ('train/dfl_loss', 'train/dfl_loss'),
            ('metrics/precision(B)', 'metrics/precision(B)'),
            ('metrics/recall(B)', 'metrics/recall(B)')
        ]

        for i, (metric, title) in enumerate(top_metrics):
            ax = axes[0, i]

            if metric in columns:
                # Plot original data line
                ax.plot(df['epoch'], df[metric], color=results_color, linewidth=1.5, label='results')
                # Plot smoothed line
                ax.plot(df['epoch'], df[metric].rolling(window=5, center=True).mean(),
                       color=smooth_color, linewidth=1, linestyle='--', alpha=0.8, label='smooth')

            ax.set_title(title, fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=9)

            # Show legend only in first subplot
            if i == 0:
                ax.legend(fontsize=9)

        # Bottom row: Validation loss and metrics (5 items)
        bottom_metrics = [
            ('val/box_loss', 'val/box_loss'),
            ('val/cls_loss', 'val/cls_loss'),
            ('val/dfl_loss', 'val/dfl_loss'),
            ('metrics/mAP50(B)', 'metrics/mAP50(B)'),
            ('metrics/mAP50-95(B)', 'metrics/mAP50-95(B)')
        ]

        # Find actual column names
        actual_cols = {}
        for col in columns:
            if 'precision' in col.lower():
                actual_cols['precision'] = col
            elif 'recall' in col.lower():
                actual_cols['recall'] = col
            elif 'mAP50' in col and 'mAP50-95' not in col:
                actual_cols['mAP50'] = col
            elif 'mAP50-95' in col or 'mAP_0.5:0.95' in col:
                actual_cols['mAP50-95'] = col

        for i, (metric, title) in enumerate(bottom_metrics):
            ax = axes[1, i]

            # Check if there are corresponding actual column names
            if metric in columns:
                col = metric
            elif 'mAP50' in metric and 'mAP50' in actual_cols:
                col = actual_cols['mAP50']
            elif 'mAP50-95' in metric and 'mAP50-95' in actual_cols:
                col = actual_cols['mAP50-95']
            else:
                col = None

            if col:
                # Plot original data line
                ax.plot(df['epoch'], df[col], color=results_color, linewidth=1.5, label='results')
                # Plot smoothed line
                ax.plot(df['epoch'], df[col].rolling(window=5, center=True).mean(),
                       color=smooth_color, linewidth=1, linestyle='--', alpha=0.8, label='smooth')

            ax.set_title(title, fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=9)

            # Show legend only in first subplot
            if i == 0:
                ax.legend(fontsize=9)

        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.93, hspace=0.3, wspace=0.3)

        # Save chart
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"✅ Training chart generated: {save_path}")
        return True

    except Exception as e:
        print(f"❌ Error generating chart: {e}")
        return False

def get_cpu_optimized_params(config):
    """Adjust parameters for CPU environment"""
    print("Warning: CPU training will be slow, GPU is recommended")
    print("If you have Apple Silicon Mac, ensure PyTorch with MPS support is installed")
    # Use smaller batch size and fewer workers for CPU to avoid memory issues
    config['training']['batch'] = 2
    config['training']['workers'] = 2
    return config

def print_training_info(config, device):
    """Print training information"""
    training_params = config['training']
    print("="*50)
    print("🚀 Training Configuration Info")
    print("="*50)
    print(f"📱 Device: {device}")
    print(f"🗂️ Project Name: {config.get('project_name', 'runs/detect')}")
    print(f"📊 Data Config: {config['data_config']}")
    print(f"🎯 Model: {config['model_pretrained']}")
    print(f"📈 Epochs: {training_params['epochs']}")
    print(f"📦 Batch Size: {training_params['batch']}")
    print(f"🖼️ Image Size: {training_params['imgsz']}")
    print(f"👥 Workers: {training_params['workers']}")
    print(f"🌱 Random Seed: {config.get('seed', 0)}")
    print("="*50)

def create_loss_plot_script(results_csv_path, output_dir=None):
    """Create an independent loss plotting script"""
    if output_dir is None:
        output_dir = os.path.dirname(results_csv_path)

    script_path = os.path.join(output_dir, 'plot_loss.py')

    script_content = f'''#!/usr/bin/env python3
"""
Auto-generated loss plotting script
Usage: python plot_loss.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import plot_training_results

if __name__ == "__main__":
    csv_path = "{results_csv_path}"
    success = plot_training_results(csv_path)
    if success:
        print("🎉 Loss chart generation completed!")
    else:
        print("❌ Generation failed")
'''

    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)

    # Add execute permission
    os.chmod(script_path, 0o755)
    print(f"📝 Created plotting script: {script_path}")
    return script_path
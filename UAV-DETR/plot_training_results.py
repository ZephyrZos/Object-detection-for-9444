#!/usr/bin/env python3
"""
UAV-DETR Training Results Visualization Script
Plots training curves similar to YOLOv5 format for UAV-DETR training results
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def setup_matplotlib():
    """Setup matplotlib for professional plots"""
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        plt.style.use('default')
    
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'lines.linewidth': 2.5,
        'axes.grid': True,
        'grid.alpha': 0.3
    })

def load_training_data(csv_file):
    """Load training data from CSV file"""
    try:
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.strip()  # Remove any whitespace
        
        print(f"Loaded {len(df)} rows of data")
        print(f"Columns: {list(df.columns)}")
        
        # All columns should already be numeric in this format
        # Calculate total losses
        df['train_total_loss'] = df['train/giou_loss'] + df['train/cls_loss'] + df['train/l1_loss']
        df['val_total_loss'] = df['val/giou_loss'] + df['val/cls_loss'] + df['val/l1_loss']
        
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_training_plots(df, save_path, max_epochs=120):
    """Create training visualization plots"""
    # Limit to first max_epochs or available data
    df_plot = df.head(min(max_epochs, len(df))).copy()
    epochs = df_plot['epoch'].values
    
    print(f"Plotting {len(df_plot)} epochs of data")
    print(f"Available metrics: {[col for col in df_plot.columns if 'metrics/' in col]}")
    
    # Create figure with 2x3 layout
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('UAV-DETR Training Results - VisDrone Dataset (120 Epochs)', fontsize=18, fontweight='bold', y=0.96)
    
    # Color palette (similar to YOLOv5)
    colors = {
        'train': '#1f77b4',      # Blue
        'val': '#ff7f0e',        # Orange
        'precision': '#2ca02c',  # Green
        'recall': '#d62728',     # Red
        'map50': '#9467bd',      # Purple
        'map95': '#8c564b'       # Brown
    }
    
    # Show data ranges
    print(f"Data ranges:")
    print(f"  Train Loss: {df_plot['train_total_loss'].min():.4f} - {df_plot['train_total_loss'].max():.4f}")
    print(f"  Val Loss: {df_plot['val_total_loss'].min():.4f} - {df_plot['val_total_loss'].max():.4f}")
    print(f"  Precision: {df_plot['metrics/precision(B)'].min():.4f} - {df_plot['metrics/precision(B)'].max():.4f}")
    print(f"  Recall: {df_plot['metrics/recall(B)'].min():.4f} - {df_plot['metrics/recall(B)'].max():.4f}")
    print(f"  mAP@0.5: {df_plot['metrics/mAP50(B)'].min():.4f} - {df_plot['metrics/mAP50(B)'].max():.4f}")
    print(f"  mAP@0.5:0.95: {df_plot['metrics/mAP50-95(B)'].min():.4f} - {df_plot['metrics/mAP50-95(B)'].max():.4f}")
    
    # 1. Training Loss (top left)
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(epochs, df_plot['train_total_loss'], color=colors['train'], 
             label='Train Loss', linewidth=2.5, alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss', fontweight='bold', fontsize=14)
    ax1.grid(True, alpha=0.3)
    # Set tight y-axis limits
    train_min, train_max = df_plot['train_total_loss'].min(), df_plot['train_total_loss'].max()
    margin = (train_max - train_min) * 0.05
    ax1.set_ylim(train_min - margin, train_max + margin)
    
    # 2. Validation Loss (top middle)
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(epochs, df_plot['val_total_loss'], color=colors['val'], 
             label='Val Loss', linewidth=2.5, alpha=0.8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Validation Loss', fontweight='bold', fontsize=14)
    ax2.grid(True, alpha=0.3)
    # Set tight y-axis limits
    val_min, val_max = df_plot['val_total_loss'].min(), df_plot['val_total_loss'].max()
    margin = (val_max - val_min) * 0.05
    ax2.set_ylim(val_min - margin, val_max + margin)
    
    # 3. Precision (top right)
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(epochs, df_plot['metrics/precision(B)'], color=colors['precision'], 
             linewidth=2.5, alpha=0.8)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Precision')
    ax3.set_title('Precision', fontweight='bold', fontsize=14)
    ax3.grid(True, alpha=0.3)
    # Set tight y-axis limits
    prec_min, prec_max = df_plot['metrics/precision(B)'].min(), df_plot['metrics/precision(B)'].max()
    margin = (prec_max - prec_min) * 0.05
    ax3.set_ylim(max(0, prec_min - margin), prec_max + margin)
    
    # 4. Recall (bottom left)
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(epochs, df_plot['metrics/recall(B)'], color=colors['recall'], 
             linewidth=2.5, alpha=0.8)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Recall')
    ax4.set_title('Recall', fontweight='bold', fontsize=14)
    ax4.grid(True, alpha=0.3)
    # Set tight y-axis limits
    recall_min, recall_max = df_plot['metrics/recall(B)'].min(), df_plot['metrics/recall(B)'].max()
    margin = (recall_max - recall_min) * 0.05
    ax4.set_ylim(max(0, recall_min - margin), recall_max + margin)
    
    # 5. mAP@0.5 (bottom middle)
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(epochs, df_plot['metrics/mAP50(B)'], color=colors['map50'], 
             linewidth=2.5, alpha=0.8)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('mAP@0.5')
    ax5.set_title('mAP@0.5', fontweight='bold', fontsize=14)
    ax5.grid(True, alpha=0.3)
    # Set tight y-axis limits
    map50_min, map50_max = df_plot['metrics/mAP50(B)'].min(), df_plot['metrics/mAP50(B)'].max()
    margin = (map50_max - map50_min) * 0.05
    ax5.set_ylim(max(0, map50_min - margin), map50_max + margin)
    
    # 6. mAP@0.5:0.95 (bottom right)
    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(epochs, df_plot['metrics/mAP50-95(B)'], color=colors['map95'], 
             linewidth=2.5, alpha=0.8)
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('mAP@0.5:0.95')
    ax6.set_title('mAP@0.5:0.95', fontweight='bold', fontsize=14)
    ax6.grid(True, alpha=0.3)
    # Set tight y-axis limits
    map95_min, map95_max = df_plot['metrics/mAP50-95(B)'].min(), df_plot['metrics/mAP50-95(B)'].max()
    margin = (map95_max - map95_min) * 0.05
    ax6.set_ylim(max(0, map95_min - margin), map95_max + margin)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.3, wspace=0.3)
    
    # Save plot
    save_file = Path(save_path) / 'uavdetr_training_results.png'
    plt.savefig(save_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Training plots saved to: {save_file}")
    
    # Show plot if in interactive mode
    try:
        plt.show()
    except:
        print("Plot display not available (non-interactive environment)")
    
    return fig

def safe_format_value(value, decimals=4):
    """Safely format numeric values, handling NaN and non-numeric types"""
    try:
        if pd.isna(value):
            return "NaN"
        elif isinstance(value, (int, float)):
            return f"{float(value):.{decimals}f}"
        else:
            # Try to convert to float
            return f"{float(value):.{decimals}f}"
    except (ValueError, TypeError):
        return str(value)

def print_training_summary(df, max_epochs=120):
    """Print training summary statistics"""
    df_summary = df.head(min(max_epochs, len(df)))
    actual_epochs = len(df_summary)
    
    print("\n" + "="*60)
    print(f"UAV-DETR TRAINING SUMMARY (First {actual_epochs} Epochs)")
    print("="*60)
    
    # Final epoch metrics
    final_epoch = df_summary.iloc[-1]
    print(f"Final Epoch: {int(final_epoch['epoch'])}")
    print(f"Training Loss: {safe_format_value(final_epoch['train_total_loss'])}")
    print(f"Validation Loss: {safe_format_value(final_epoch['val_total_loss'])}")
    print(f"Precision: {safe_format_value(final_epoch['metrics/precision(B)'])}")
    print(f"Recall: {safe_format_value(final_epoch['metrics/recall(B)'])}")
    print(f"mAP@0.5: {safe_format_value(final_epoch['metrics/mAP50(B)'])}")
    print(f"mAP@0.5:0.95: {safe_format_value(final_epoch['metrics/mAP50-95(B)'])}")
    
    # Check if we have valid data for best metrics
    numeric_cols = ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
    valid_data = {}
    
    for col in numeric_cols:
        if col in df_summary.columns:
            # Get non-NaN values
            valid_values = df_summary[col].dropna()
            if len(valid_values) > 0:
                valid_data[col] = valid_values
    
    if valid_data:
        print(f"\n📊 Best Performance:")
        for col, values in valid_data.items():
            best_val = values.max()
            best_epoch = df_summary[df_summary[col] == best_val]['epoch'].iloc[0]
            col_name = col.split('/')[-1].replace('(B)', '').replace('mAP50-95', 'mAP@0.5:0.95').replace('mAP50', 'mAP@0.5')
            print(f"Best {col_name}: {safe_format_value(best_val)} (Epoch {int(best_epoch)})")
    
    # Training improvement (if we have at least 2 epochs)
    if len(df_summary) > 1:
        initial_metrics = df_summary.iloc[0]
        print(f"\n📈 Training Progress:")
        
        for col in numeric_cols:
            if col in df_summary.columns:
                initial_val = initial_metrics[col]
                final_val = final_epoch[col]
                
                if not pd.isna(initial_val) and not pd.isna(final_val):
                    improvement = float(final_val) - float(initial_val)
                    col_name = col.split('/')[-1].replace('(B)', '').replace('mAP50-95', 'mAP@0.5:0.95').replace('mAP50', 'mAP@0.5')
                    print(f"{col_name} Improvement: {safe_format_value(improvement, 4)}")
        
        # Loss reduction
        if 'train_total_loss' in df_summary.columns and 'val_total_loss' in df_summary.columns:
            train_initial = initial_metrics['train_total_loss']
            train_final = final_epoch['train_total_loss']
            val_initial = initial_metrics['val_total_loss']
            val_final = final_epoch['val_total_loss']
            
            if not pd.isna(train_initial) and not pd.isna(train_final):
                train_reduction = float(train_initial) - float(train_final)
                print(f"Training Loss Reduction: {safe_format_value(train_reduction)}")
            
            if not pd.isna(val_initial) and not pd.isna(val_final):
                val_reduction = float(val_initial) - float(val_final)
                print(f"Validation Loss Reduction: {safe_format_value(val_reduction)}")
    
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Plot UAV-DETR training results')
    parser.add_argument('--results', type=str, 
                       default='runs/train/uavdetr_single_gpu_batch8_20250726_014942/results.csv',
                       help='Path to results CSV file')
    parser.add_argument('--save-dir', type=str, default='.',
                       help='Directory to save plots')
    parser.add_argument('--epochs', type=int, default=120,
                       help='Number of epochs to plot')
    parser.add_argument('--show-summary', action='store_true', default=True,
                       help='Show training summary')
    
    args = parser.parse_args()
    
    # Setup matplotlib
    setup_matplotlib()
    
    # Check if results file exists
    results_file = args.results
    if not Path(results_file).exists():
        print(f"Results file not found: {results_file}")
        return
    print(f"Loading training results from: {results_file}")
    
    # Load data
    df = load_training_data(results_file)
    if df is None:
        return
    
    print(f"Loaded {len(df)} epochs of training data")
    
    # Show training summary
    if args.show_summary:
        print_training_summary(df, args.epochs)
    
    # Create plots
    create_training_plots(df, args.save_dir, args.epochs)

if __name__ == '__main__':
    main()
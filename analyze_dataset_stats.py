#!/usr/bin/env python3
"""
VisDrone Dataset Analysis and Visualization Script
Analyze train/val/test split statistics and image size distribution
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import pandas as pd
from collections import defaultdict
import numpy as np

def count_files_in_directory(directory):
    """Count image and label files in a directory"""
    if not directory.exists():
        return 0, 0
    
    # Count images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_count = len([f for f in directory.iterdir() 
                      if f.is_file() and f.suffix.lower() in image_extensions])
    
    # Count labels
    label_count = len([f for f in directory.iterdir() 
                      if f.is_file() and f.suffix.lower() == '.txt'])
    
    return image_count, label_count

def get_image_sizes(image_directory, max_sample=1000):
    """Get image size distribution from a sample of images"""
    if not image_directory.exists():
        return []
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in image_directory.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    # Sample images if too many
    if len(image_files) > max_sample:
        image_files = np.random.choice(image_files, max_sample, replace=False)
    
    sizes = []
    for img_file in image_files:
        try:
            with Image.open(img_file) as img:
                sizes.append(img.size)  # (width, height)
        except Exception as e:
            print(f"Warning: Could not read {img_file}: {e}")
            continue
    
    return sizes

def analyze_dataset():
    """Main analysis function"""
    dataset_path = Path("/g/data/zq94/zz9919/Drone_ObjectDetection/Yolo_method/dataset")
    
    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        return
    
    print("🔍 Analyzing VisDrone Dataset Structure")
    print("=" * 50)
    
    # Analyze each split
    stats = {}
    splits = ['train', 'val', 'test']
    
    for split in splits:
        split_path = dataset_path / split
        
        # Try both structures: split/images/ and split/
        images_path = split_path / "images" if (split_path / "images").exists() else split_path
        labels_path = split_path / "labels" if (split_path / "labels").exists() else split_path
        
        if split_path.exists():
            img_count, _ = count_files_in_directory(images_path)
            _, label_count = count_files_in_directory(labels_path)
            
            # Check for cache files
            cache_file = split_path / "labels.cache"
            has_cache = cache_file.exists()
            
            stats[split] = {
                'images': img_count,
                'labels': label_count,
                'has_cache': has_cache,
                'images_path': images_path
            }
            
            print(f"📁 {split.upper()} Split:")
            print(f"   Images: {img_count:,}")
            print(f"   Labels: {label_count:,}")
            print(f"   Cache: {'✅' if has_cache else '❌'}")
            print()
        else:
            stats[split] = {'images': 0, 'labels': 0, 'has_cache': False, 'images_path': None}
            print(f"📁 {split.upper()} Split: ❌ Not found")
            print()
    
    # Total statistics
    total_images = sum(stats[split]['images'] for split in splits)
    total_labels = sum(stats[split]['labels'] for split in splits)
    
    print("📊 Dataset Summary:")
    print(f"   Total Images: {total_images:,}")
    print(f"   Total Labels: {total_labels:,}")
    print("=" * 50)
    
    return stats

def create_visualizations(stats):
    """Create visualization plots"""
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('VisDrone Dataset Analysis', fontsize=16, fontweight='bold')
    
    # 1. Dataset split distribution (bar chart)
    splits = ['train', 'val', 'test']
    image_counts = [stats[split]['images'] for split in splits]
    
    ax1 = axes[0, 0]
    bars = ax1.bar(splits, image_counts, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax1.set_title('Dataset Split Distribution', fontweight='bold')
    ax1.set_ylabel('Number of Images')
    
    # Add value labels on bars
    for bar, count in zip(bars, image_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Pie chart of dataset distribution
    ax2 = axes[0, 1]
    valid_counts = [(split, count) for split, count in zip(splits, image_counts) if count > 0]
    if valid_counts:
        labels, counts = zip(*valid_counts)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c'][:len(labels)]
        wedges, texts, autotexts = ax2.pie(counts, labels=labels, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax2.set_title('Dataset Distribution', fontweight='bold')
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_fontweight('bold')
    else:
        ax2.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Dataset Distribution', fontweight='bold')
    
    # 3. Image size analysis for each split
    all_sizes = []
    size_data = []
    
    for i, split in enumerate(splits):
        if stats[split]['images'] > 0 and stats[split]['images_path']:
            print(f"🔍 Analyzing image sizes for {split} split...")
            sizes = get_image_sizes(stats[split]['images_path'], max_sample=500)
            
            if sizes:
                widths, heights = zip(*sizes)
                all_sizes.extend(sizes)
                
                for w, h in sizes:
                    size_data.append({
                        'split': split,
                        'width': w,
                        'height': h,
                        'aspect_ratio': w/h,
                        'area': w*h
                    })
    
    df_sizes = pd.DataFrame(size_data)
    
    # 4. Image width distribution
    ax3 = axes[0, 2]
    if not df_sizes.empty:
        for split in splits:
            split_data = df_sizes[df_sizes['split'] == split]
            if not split_data.empty:
                ax3.hist(split_data['width'], bins=30, alpha=0.7, label=split)
        ax3.set_title('Image Width Distribution', fontweight='bold')
        ax3.set_xlabel('Width (pixels)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'No size data available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Image Width Distribution', fontweight='bold')
    
    # 5. Image height distribution
    ax4 = axes[1, 0]
    if not df_sizes.empty:
        for split in splits:
            split_data = df_sizes[df_sizes['split'] == split]
            if not split_data.empty:
                ax4.hist(split_data['height'], bins=30, alpha=0.7, label=split)
        ax4.set_title('Image Height Distribution', fontweight='bold')
        ax4.set_xlabel('Height (pixels)')
        ax4.set_ylabel('Frequency')
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'No size data available', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Image Height Distribution', fontweight='bold')
    
    # 6. Aspect ratio distribution
    ax5 = axes[1, 1]
    if not df_sizes.empty:
        for split in splits:
            split_data = df_sizes[df_sizes['split'] == split]
            if not split_data.empty:
                ax5.hist(split_data['aspect_ratio'], bins=30, alpha=0.7, label=split)
        ax5.set_title('Aspect Ratio Distribution', fontweight='bold')
        ax5.set_xlabel('Aspect Ratio (width/height)')
        ax5.set_ylabel('Frequency')
        ax5.legend()
    else:
        ax5.text(0.5, 0.5, 'No size data available', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Aspect Ratio Distribution', fontweight='bold')
    
    # 7. Summary statistics table
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    if not df_sizes.empty:
        summary_stats = []
        for split in splits:
            split_data = df_sizes[df_sizes['split'] == split]
            if not split_data.empty:
                summary_stats.append([
                    split.upper(),
                    f"{len(split_data)}",
                    f"{split_data['width'].mean():.0f}",
                    f"{split_data['height'].mean():.0f}",
                    f"{split_data['aspect_ratio'].mean():.2f}"
                ])
        
        if summary_stats:
            table = ax6.table(cellText=summary_stats,
                            colLabels=['Split', 'Samples', 'Avg Width', 'Avg Height', 'Avg Ratio'],
                            cellLoc='center',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            
            # Style the table
            for i in range(len(summary_stats) + 1):
                for j in range(5):
                    cell = table[(i, j)]
                    if i == 0:  # Header row
                        cell.set_facecolor('#4CAF50')
                        cell.set_text_props(weight='bold', color='white')
                    else:
                        cell.set_facecolor('#f0f0f0')
    
    ax6.set_title('Size Statistics Summary', fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = Path("/g/data/zq94/zz9919/Drone_ObjectDetection/dataset_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Visualization saved to: {output_path}")
    
    plt.show()
    
    return df_sizes

def print_detailed_stats(df_sizes, stats):
    """Print detailed statistics"""
    print("\n" + "="*60)
    print("📈 DETAILED STATISTICS")
    print("="*60)
    
    if not df_sizes.empty:
        for split in ['train', 'val', 'test']:
            split_data = df_sizes[df_sizes['split'] == split]
            if not split_data.empty:
                print(f"\n🔍 {split.upper()} SPLIT ANALYSIS:")
                print(f"   Sample Size: {len(split_data)} images")
                print(f"   Width  - Mean: {split_data['width'].mean():.1f}, Std: {split_data['width'].std():.1f}")
                print(f"   Height - Mean: {split_data['height'].mean():.1f}, Std: {split_data['height'].std():.1f}")
                print(f"   Aspect Ratio - Mean: {split_data['aspect_ratio'].mean():.3f}")
                print(f"   Common sizes:")
                size_counts = split_data.groupby(['width', 'height']).size().sort_values(ascending=False).head(3)
                for (w, h), count in size_counts.items():
                    print(f"     {w}x{h}: {count} images")
    
    # Dataset health check
    print(f"\n🏥 DATASET HEALTH CHECK:")
    total_images = sum(stats[split]['images'] for split in ['train', 'val', 'test'])
    if total_images > 0:
        train_pct = stats['train']['images'] / total_images * 100
        val_pct = stats['val']['images'] / total_images * 100
        test_pct = stats['test']['images'] / total_images * 100
        
        print(f"   Split Ratios: Train {train_pct:.1f}% | Val {val_pct:.1f}% | Test {test_pct:.1f}%")
        
        # Check if ratios are reasonable
        if train_pct < 60:
            print("   ⚠️  Warning: Training set might be too small")
        if val_pct < 5:
            print("   ⚠️  Warning: Validation set might be too small")
        if val_pct > 30:
            print("   ⚠️  Warning: Validation set might be too large")
    
    print("="*60)

def main():
    """Main function"""
    print("🚀 Starting VisDrone Dataset Analysis")
    print("This may take a few minutes to analyze image sizes...")
    
    # Analyze dataset structure
    stats = analyze_dataset()
    
    if sum(stats[split]['images'] for split in ['train', 'val', 'test']) == 0:
        print("❌ No images found in dataset. Please check the dataset path.")
        return
    
    # Create visualizations
    df_sizes = create_visualizations(stats)
    
    # Print detailed statistics
    print_detailed_stats(df_sizes, stats)
    
    print("\n✅ Dataset analysis completed!")
    print("📊 Check dataset_analysis.png for visualizations")

if __name__ == "__main__":
    main()
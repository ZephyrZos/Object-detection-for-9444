#!/usr/bin/env python3
"""
Enhanced VisDrone Dataset Resolution Distribution Visualization
Comprehensive analysis of image resolutions across train/val/test splits
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import pandas as pd
from collections import defaultdict, Counter
import numpy as np

def analyze_all_images():
    """Analyze all images in the dataset for comprehensive resolution analysis"""
    dataset_path = Path("/g/data/zq94/zz9919/Drone_ObjectDetection/Yolo_method/dataset")
    
    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        return None
    
    print("🔍 Analyzing ALL images in VisDrone dataset...")
    print("This may take several minutes...")
    
    all_data = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        split_path = dataset_path / split
        
        # Try both structures: split/images/ and split/
        images_path = split_path / "images" if (split_path / "images").exists() else split_path
        
        if not images_path.exists():
            continue
            
        print(f"📁 Processing {split} split...")
        
        image_files = [f for f in images_path.iterdir() 
                       if f.is_file() and f.suffix.lower() in image_extensions]
        
        print(f"   Found {len(image_files)} images")
        
        for i, img_file in enumerate(image_files):
            if i % 1000 == 0:
                print(f"   Progress: {i}/{len(image_files)} images")
                
            try:
                with Image.open(img_file) as img:
                    width, height = img.size
                    all_data.append({
                        'split': split,
                        'width': width,
                        'height': height,
                        'resolution': f"{width}x{height}",
                        'aspect_ratio': width / height,
                        'area': width * height,
                        'megapixels': (width * height) / 1_000_000
                    })
            except Exception as e:
                print(f"   Warning: Could not read {img_file}: {e}")
                continue
    
    return pd.DataFrame(all_data)

def create_comprehensive_visualizations(df):
    """Create comprehensive resolution distribution visualizations"""
    if df.empty:
        print("❌ No data to visualize")
        return
    
    plt.style.use('seaborn-v0_8')
    
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    fig.suptitle('VisDrone Dataset - Comprehensive Resolution Analysis', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    splits = ['train', 'val', 'test']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # 1. Overall resolution distribution (scatter plot)
    ax1 = fig.add_subplot(gs[0, :2])
    for i, split in enumerate(splits):
        split_data = df[df['split'] == split]
        if not split_data.empty:
            ax1.scatter(split_data['width'], split_data['height'], 
                       alpha=0.6, s=30, label=split, color=colors[i])
    
    ax1.set_xlabel('Width (pixels)')
    ax1.set_ylabel('Height (pixels)')
    ax1.set_title('Resolution Distribution Scatter Plot', fontweight='bold', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Top 10 most common resolutions (horizontal bar chart)
    ax2 = fig.add_subplot(gs[0, 2:])
    resolution_counts = df['resolution'].value_counts().head(10)
    
    bars = ax2.barh(range(len(resolution_counts)), resolution_counts.values, 
                    color='skyblue', edgecolor='navy', alpha=0.7)
    ax2.set_yticks(range(len(resolution_counts)))
    ax2.set_yticklabels(resolution_counts.index, fontsize=10)
    ax2.set_xlabel('Number of Images')
    ax2.set_title('Top 10 Most Common Resolutions', fontweight='bold', fontsize=14)
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, resolution_counts.values)):
        ax2.text(count + max(resolution_counts.values) * 0.01, i, 
                f'{count}', va='center', fontweight='bold')
    
    # 3. Resolution distribution by split (stacked bar)
    ax3 = fig.add_subplot(gs[1, :2])
    top_resolutions = df['resolution'].value_counts().head(8).index
    
    split_resolution_data = []
    for resolution in top_resolutions:
        row_data = {'resolution': resolution}
        for split in splits:
            count = len(df[(df['split'] == split) & (df['resolution'] == resolution)])
            row_data[split] = count
        split_resolution_data.append(row_data)
    
    split_df = pd.DataFrame(split_resolution_data)
    split_df.set_index('resolution')[splits].plot(kind='bar', stacked=True, 
                                                  ax=ax3, color=colors)
    ax3.set_title('Resolution Distribution by Split (Top 8)', fontweight='bold', fontsize=14)
    ax3.set_xlabel('Resolution')
    ax3.set_ylabel('Number of Images')
    ax3.legend(title='Split')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Aspect ratio distribution
    ax4 = fig.add_subplot(gs[1, 2:])
    for i, split in enumerate(splits):
        split_data = df[df['split'] == split]
        if not split_data.empty:
            ax4.hist(split_data['aspect_ratio'], bins=30, alpha=0.7, 
                    label=split, color=colors[i], density=True)
    
    ax4.set_xlabel('Aspect Ratio (width/height)')
    ax4.set_ylabel('Density')
    ax4.set_title('Aspect Ratio Distribution', fontweight='bold', fontsize=14)
    ax4.legend()
    ax4.axvline(x=16/9, color='red', linestyle='--', alpha=0.7, label='16:9')
    ax4.axvline(x=4/3, color='purple', linestyle='--', alpha=0.7, label='4:3')
    
    # 5. Megapixels distribution
    ax5 = fig.add_subplot(gs[2, :2])
    for i, split in enumerate(splits):
        split_data = df[df['split'] == split]
        if not split_data.empty:
            ax5.hist(split_data['megapixels'], bins=30, alpha=0.7, 
                    label=split, color=colors[i])
    
    ax5.set_xlabel('Megapixels')
    ax5.set_ylabel('Number of Images')
    ax5.set_title('Image Size Distribution (Megapixels)', fontweight='bold', fontsize=14)
    ax5.legend()
    
    # 6. Resolution heatmap
    ax6 = fig.add_subplot(gs[2, 2:])
    
    # Create resolution bins for heatmap
    width_bins = np.linspace(df['width'].min(), df['width'].max(), 20)
    height_bins = np.linspace(df['height'].min(), df['height'].max(), 15)
    
    # Create 2D histogram
    hist_data = []
    for split in splits:
        split_data = df[df['split'] == split]
        if not split_data.empty:
            hist, _, _ = np.histogram2d(split_data['width'], split_data['height'], 
                                      bins=[width_bins, height_bins])
            hist_data.append(hist.T)
    
    if hist_data:
        combined_hist = sum(hist_data)
        im = ax6.imshow(combined_hist, origin='lower', aspect='auto', 
                       cmap='YlOrRd', interpolation='nearest')
        
        # Set ticks and labels
        ax6.set_xticks(np.linspace(0, len(width_bins)-1, 5))
        ax6.set_xticklabels([f'{int(w)}' for w in np.linspace(width_bins[0], width_bins[-1], 5)])
        ax6.set_yticks(np.linspace(0, len(height_bins)-1, 5))
        ax6.set_yticklabels([f'{int(h)}' for h in np.linspace(height_bins[0], height_bins[-1], 5)])
        
        ax6.set_xlabel('Width (pixels)')
        ax6.set_ylabel('Height (pixels)')
        ax6.set_title('Resolution Density Heatmap', fontweight='bold', fontsize=14)
        
        plt.colorbar(im, ax=ax6, label='Number of Images')
    
    # 7. Statistics table for each split
    ax7 = fig.add_subplot(gs[3, :2])
    ax7.axis('off')
    
    stats_data = []
    for split in splits:
        split_data = df[df['split'] == split]
        if not split_data.empty:
            stats_data.append([
                split.upper(),
                f"{len(split_data):,}",
                f"{split_data['width'].mean():.0f}±{split_data['width'].std():.0f}",
                f"{split_data['height'].mean():.0f}±{split_data['height'].std():.0f}",
                f"{split_data['aspect_ratio'].mean():.3f}",
                f"{split_data['megapixels'].mean():.2f}±{split_data['megapixels'].std():.2f}",
                split_data['resolution'].mode().iloc[0] if len(split_data) > 0 else 'N/A'
            ])
    
    if stats_data:
        table = ax7.table(cellText=stats_data,
                         colLabels=['Split', 'Count', 'Width (μ±σ)', 'Height (μ±σ)', 
                                   'Aspect Ratio', 'Megapixels (μ±σ)', 'Most Common'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 2)
        
        # Style the table
        for i in range(len(stats_data) + 1):
            for j in range(7):
                cell = table[(i, j)]
                if i == 0:  # Header row
                    cell.set_facecolor('#2E86AB')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor(['#E8F4F8', '#F8E8E8', '#E8F8E8'][i-1])
    
    ax7.set_title('Detailed Statistics by Split', fontweight='bold', fontsize=14)
    
    # 8. Resolution variety analysis
    ax8 = fig.add_subplot(gs[3, 2:])
    
    # Count unique resolutions per split
    unique_resolutions = []
    resolution_diversity = []
    
    for split in splits:
        split_data = df[df['split'] == split]
        if not split_data.empty:
            unique_count = len(split_data['resolution'].unique())
            total_count = len(split_data)
            diversity_ratio = unique_count / total_count
            
            unique_resolutions.append(unique_count)
            resolution_diversity.append(diversity_ratio)
    
    # Dual y-axis plot
    ax8_twin = ax8.twinx()
    
    x_pos = np.arange(len(splits))
    bars1 = ax8.bar(x_pos - 0.2, unique_resolutions, 0.4, 
                   label='Unique Resolutions', color='steelblue', alpha=0.7)
    bars2 = ax8_twin.bar(x_pos + 0.2, resolution_diversity, 0.4, 
                        label='Diversity Ratio', color='orange', alpha=0.7)
    
    ax8.set_xlabel('Dataset Split')
    ax8.set_ylabel('Number of Unique Resolutions', color='steelblue')
    ax8_twin.set_ylabel('Resolution Diversity Ratio', color='orange')
    ax8.set_title('Resolution Variety Analysis', fontweight='bold', fontsize=14)
    ax8.set_xticks(x_pos)
    ax8.set_xticklabels([s.upper() for s in splits])
    
    # Add value labels
    for bar, val in zip(bars1, unique_resolutions):
        ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val}', ha='center', va='bottom', color='steelblue', fontweight='bold')
    
    for bar, val in zip(bars2, resolution_diversity):
        ax8_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                     f'{val:.3f}', ha='center', va='bottom', color='orange', fontweight='bold')
    
    plt.tight_layout()
    
    # Save the comprehensive visualization
    output_path = Path("/g/data/zq94/zz9919/Drone_ObjectDetection/comprehensive_resolution_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Comprehensive visualization saved to: {output_path}")
    
    return df

def print_resolution_insights(df):
    """Print detailed insights about resolution distribution"""
    print("\n" + "="*80)
    print("🔍 COMPREHENSIVE RESOLUTION ANALYSIS INSIGHTS")
    print("="*80)
    
    # Overall statistics
    total_images = len(df)
    unique_resolutions = df['resolution'].nunique()
    most_common_res = df['resolution'].mode().iloc[0]
    most_common_count = (df['resolution'] == most_common_res).sum()
    
    print(f"\n📊 OVERALL DATASET STATISTICS:")
    print(f"   Total Images Analyzed: {total_images:,}")
    print(f"   Unique Resolutions: {unique_resolutions}")
    print(f"   Most Common Resolution: {most_common_res} ({most_common_count:,} images, {most_common_count/total_images*100:.1f}%)")
    
    # Resolution analysis by split
    print(f"\n🎯 RESOLUTION DISTRIBUTION BY SPLIT:")
    for split in ['train', 'val', 'test']:
        split_data = df[df['split'] == split]
        if not split_data.empty:
            unique_in_split = split_data['resolution'].nunique()
            most_common_in_split = split_data['resolution'].mode().iloc[0]
            most_common_count_split = (split_data['resolution'] == most_common_in_split).sum()
            
            print(f"   {split.upper()} Split:")
            print(f"     Images: {len(split_data):,}")
            print(f"     Unique Resolutions: {unique_in_split}")
            print(f"     Most Common: {most_common_in_split} ({most_common_count_split} images)")
            print(f"     Resolution Diversity: {unique_in_split/len(split_data):.3f}")
    
    # Aspect ratio analysis
    print(f"\n📐 ASPECT RATIO INSIGHTS:")
    aspect_ratios = df['aspect_ratio']
    common_ratios = {
        '16:9': 16/9,
        '4:3': 4/3,
        '3:2': 3/2,
        '1:1': 1.0
    }
    
    for name, ratio in common_ratios.items():
        close_to_ratio = abs(aspect_ratios - ratio) < 0.05
        count = close_to_ratio.sum()
        if count > 0:
            print(f"   ~{name} aspect ratio: {count:,} images ({count/total_images*100:.1f}%)")
    
    # Size categories
    print(f"\n📏 IMAGE SIZE CATEGORIES:")
    size_categories = {
        'Small (< 1 MP)': df['megapixels'] < 1,
        'Medium (1-2 MP)': (df['megapixels'] >= 1) & (df['megapixels'] < 2),
        'Large (2-4 MP)': (df['megapixels'] >= 2) & (df['megapixels'] < 4),
        'Extra Large (≥ 4 MP)': df['megapixels'] >= 4
    }
    
    for category, mask in size_categories.items():
        count = mask.sum()
        print(f"   {category}: {count:,} images ({count/total_images*100:.1f}%)")
    
    # Training implications
    print(f"\n🎯 TRAINING IMPLICATIONS:")
    avg_megapixels = df['megapixels'].mean()
    std_megapixels = df['megapixels'].std()
    
    print(f"   Average Image Size: {avg_megapixels:.2f} ± {std_megapixels:.2f} MP")
    
    if std_megapixels / avg_megapixels > 0.5:
        print("   ⚠️  High size variation - consider adaptive input sizing")
    
    if avg_megapixels > 2:
        print("   💾 Large images - may need memory optimization")
    
    if unique_resolutions > total_images * 0.1:
        print("   📐 High resolution diversity - good for model robustness")
    
    print("="*80)

def main():
    """Main function"""
    print("🚀 Starting Comprehensive Resolution Analysis")
    print("This will analyze ALL images in the dataset - may take 10-15 minutes...")
    
    # Analyze all images
    df = analyze_all_images()
    
    if df is None or df.empty:
        print("❌ No data found. Please check the dataset path.")
        return
    
    print(f"\n✅ Successfully analyzed {len(df):,} images")
    
    # Create comprehensive visualizations
    create_comprehensive_visualizations(df)
    
    # Print detailed insights
    print_resolution_insights(df)
    
    print("\n🎉 Comprehensive resolution analysis completed!")
    print("📊 Check comprehensive_resolution_analysis.png for detailed visualizations")

if __name__ == "__main__":
    main()
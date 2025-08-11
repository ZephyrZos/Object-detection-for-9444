#!/usr/bin/env python3
"""
Quick test script for visualization
"""

import os
import glob
import subprocess
import sys

def find_test_image():
    """Find a test image from the dataset"""
    # Look for images in common locations
    image_paths = [
        "faster_rcnn_dataset/val/images/*.jpg",
        "faster_rcnn_dataset/train/images/*.jpg",
        "fastrcnn_dataset/val/images/*.jpg",
        "fastrcnn_dataset/train/images/*.jpg",
        "*.jpg",
        "*.png"
    ]

    for pattern in image_paths:
        images = glob.glob(pattern)
        if images:
            return images[0]

    return None

def find_best_model():
    """Find the best model checkpoint"""
    model_paths = glob.glob("checkpoints/*/best_model.pth")
    if model_paths:
        # Return the most recent one
        return max(model_paths, key=os.path.getmtime)
    return None

def main():
    print("🎨 Quick Visualization Test")
    print("=" * 50)

    # Find test image
    test_image = find_test_image()
    if not test_image:
        print("❌ No test image found!")
        print("Please provide an image path or ensure dataset is available")
        return

    print(f"✅ Found test image: {test_image}")

    # Find model
    model_path = find_best_model()
    if not model_path:
        print("❌ No trained model found!")
        print("Please train a model first using: python train.py")
        return

    print(f"✅ Found model: {model_path}")

    # Run visualization
    print("\n🚀 Generating visualizations...")
    print("This might take a moment...\n")

    cmd = [
        sys.executable,
        "visualize_intermediate.py",
        "--image", test_image,
        "--model", model_path,
        "--all",
        "--output", "test_visualizations"
    ]

    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Visualizations generated successfully!")
        print("📁 Check the 'test_visualizations' folder for results:")
        print("  - feature_maps/: Feature maps from different layers")
        print("  - rpn_attention/: RPN objectness scores")
        print("  - confidence_heatmap/: Detection confidence heatmaps")
        print("  - anchor_boxes/: Anchor box visualizations")

    except subprocess.CalledProcessError:
        print("❌ Visualization failed!")
        print("Please check the error messages above")
    except FileNotFoundError:
        print("❌ visualize_intermediate.py not found!")
        print("Please ensure the script is in the current directory")

if __name__ == "__main__":
    main()
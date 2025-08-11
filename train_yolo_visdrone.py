#!/usr/bin/env python3
"""
Simple YOLO Training Script for VisDrone Dataset
Optimized for single GPU training with path matching
"""

import os
import subprocess
import sys
import time
import torch
from pathlib import Path

# Must set these environment variables before importing any libraries that use MKL!
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from ultralytics import YOLO

def get_device():
    """Auto-detect the best available device"""
    if torch.cuda.is_available():
        device = 0  # Use first GPU
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = 'cpu'
        print("⚠️  CUDA not available, using CPU")
    return device

def find_dataset_path():
    """Find the correct dataset path in the project"""
    current_dir = Path.cwd()
    possible_paths = [
        Path("/g/data/zq94/zz9919/Drone_ObjectDetection/Yolo_method/dataset"),
        current_dir / "Yolo_method" / "dataset",
        current_dir / "dataset",
        current_dir / "data" / "visdrone",
        current_dir / "VisDrone_YOLO",
    ]
    
    for path in possible_paths:
        if path.exists():
            # Check if has train directory and either images or labels
            train_path = path / "train"
            if train_path.exists():
                # Check for images directory or labels.cache (indicating processed dataset)
                if (train_path / "images").exists() or (train_path / "labels.cache").exists():
                    print(f"✅ Found dataset at: {path}")
                    return str(path)
    
    print("❌ Dataset not found! Checked paths:")
    for path in possible_paths:
        exists = "✅" if path.exists() else "❌"
        print(f"  {exists} {path}")
    return None

def create_data_yaml(dataset_path):
    """Create or update data.yaml configuration file"""
    dataset_path = Path(dataset_path)
    
    # Check actual structure and adapt paths
    train_images_path = "train/images" if (dataset_path / "train" / "images").exists() else "train"
    val_images_path = "val/images" if (dataset_path / "val" / "images").exists() else "val" 
    test_images_path = "test/images" if (dataset_path / "test" / "images").exists() else "test"
    
    print(f"📂 Dataset structure detected:")
    print(f"   Train path: {train_images_path}")
    print(f"   Val path: {val_images_path}")
    print(f"   Test path: {test_images_path}")
    
    data_yaml_content = f"""# VisDrone dataset configuration
path: {dataset_path}  # dataset root dir
train: {train_images_path}  # train images (relative to 'path')
val: {val_images_path}      # val images (relative to 'path')
test: {test_images_path}    # test images (relative to 'path') - optional

# Classes (VisDrone dataset - 10 classes)
nc: 10  # number of classes
names:
  0: pedestrian
  1: people
  2: bicycle  
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
"""
    
    yaml_path = dataset_path / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(data_yaml_content)
    
    print(f"✅ Created data config: {yaml_path}")
    return str(yaml_path)

def main():
    """Main training function"""
    print("=" * 60)
    print("🚀 YOLO VisDrone Training Script")
    print("=" * 60)
    
    # Auto-detect device
    device = get_device()
    
    # Find dataset path
    dataset_path = find_dataset_path()
    if not dataset_path:
        print("❌ Dataset not found! Available options:")
        print("   1. Convert VisDrone dataset: python convert_visdrone_to_yolo.py")
        print("   2. Check if dataset exists at: /g/data/zq94/zz9919/Drone_ObjectDetection/Yolo_method/dataset")
        return
    
    # Check dataset health
    dataset_path_obj = Path(dataset_path)
    train_exists = (dataset_path_obj / "train").exists()
    val_exists = (dataset_path_obj / "val").exists()
    
    if not (train_exists and val_exists):
        print(f"⚠️  Dataset structure incomplete:")
        print(f"   Train exists: {train_exists}")
        print(f"   Val exists: {val_exists}")
        print(f"   Please ensure dataset conversion completed successfully")
        return
    
    # Create data.yaml config
    data_config = create_data_yaml(dataset_path)
    
    # Find local model weights
    local_weights = [
        Path("/g/data/zq94/zz9919/Drone_ObjectDetection/Yolov8/yolov8m.pt"),
        Path("Yolov8/yolov8m.pt"),
        Path("yolov8m.pt"),
        Path("weights/yolov8m.pt"),
    ]
    
    model_path = 'yolov8m.pt'  # Default fallback
    for weight_path in local_weights:
        if weight_path.exists():
            model_path = str(weight_path)
            print(f"🎯 Found local weights: {model_path}")
            break
    else:
        print("⚠️ No local weights found, will try to download (may fail on compute nodes)")
    
    # Training configuration
    config = {
        'model': model_path,  # Use local model weights
        'data': data_config,
        'epochs': 100,
        'batch': 8 if device != 'cpu' else 4,
        'imgsz': 1280,
        'workers': 4 if device != 'cpu' else 2,
        'device': device,
        'project': 'runs/detect',
        'name': 'yolo_visdrone_train',
        'save': True,
        'save_period': 10,
        'cache': False,  # Disable cache to save memory
        'patience': 20,  # Early stopping patience
        'optimizer': 'AdamW',
        'lr0': 0.001,  # Initial learning rate
        'lrf': 0.01,   # Final learning rate factor
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'warmup_epochs': 5,
        'cos_lr': True,  # Use cosine learning rate scheduler
        # Data augmentation
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'single_cls': False,
    }
    
    # Check for existing configs first
    existing_configs = [
        Path("/g/data/zq94/zz9919/Drone_ObjectDetection/Yolo_method/data/VisDrone_converted.yaml"),
        Path("/g/data/zq94/zz9919/Drone_ObjectDetection/Yolov8/visdrone.yaml"),
    ]
    
    data_config_to_use = data_config
    for existing_config in existing_configs:
        if existing_config.exists():
            print(f"📄 Found existing config: {existing_config}")
            data_config_to_use = str(existing_config)
            config['data'] = data_config_to_use
            break
    
    print("\n📋 Final Training Configuration:")
    print(f"  🎯 Model: {config['model']}")
    print(f"  📊 Dataset: {data_config_to_use}")
    print(f"  🖥️  Device: {device}")
    print(f"  🔄 Epochs: {config['epochs']}")
    print(f"  📦 Batch size: {config['batch']}")
    print(f"  📐 Image size: {config['imgsz']}")
    print(f"  ⚙️  Workers: {config['workers']}")
    print(f"  💾 Cache: {config['cache']}")
    print("=" * 60)
    
    try:
        # Load model
        print(f"🔄 Loading model: {config['model']}")
        model = YOLO(config['model'])
        
        # Start training
        print("🚀 Starting training...")
        results = model.train(**config)
        
        print("=" * 60)
        print("🎉 Training completed successfully!")
        print(f"📁 Results saved to: {results.save_dir}")
        print(f"🏆 Best weights: {results.save_dir}/weights/best.pt")
        print(f"📝 Last weights: {results.save_dir}/weights/last.pt")
        print("=" * 60)
        
        # Print final metrics if available
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            print("📊 Final Training Metrics:")
            print(f"  mAP@0.5: {metrics.get('metrics/mAP50(B)', 'N/A')}")
            print(f"  mAP@0.5:0.95: {metrics.get('metrics/mAP50-95(B)', 'N/A')}")
            print(f"  Precision: {metrics.get('metrics/precision(B)', 'N/A')}")
            print(f"  Recall: {metrics.get('metrics/recall(B)', 'N/A')}")
        
    except KeyboardInterrupt:
        print("\\n⚠️ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
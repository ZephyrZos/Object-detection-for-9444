#!/usr/bin/env python3
"""
UAV-DETR Training Script for 1280x1280 Image Size
Optimized for high-resolution drone object detection with VisDrone dataset
"""

import warnings
import os
from pathlib import Path
from ultralytics import RTDETR
import torch
import gc

warnings.filterwarnings('ignore')

def check_path(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")

def setup_device():
    """Setup device with automatic fallback and memory optimization"""
    if torch.cuda.is_available():
        device = '0'
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Using GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        
        # Clear cache for clean start
        torch.cuda.empty_cache()
        
        return device, gpu_memory
    else:
        device = 'cpu'
        print("CUDA not available, using CPU")
        return device, 0

def get_optimal_batch_size(gpu_memory_gb, imgsz=1280):
    """Calculate optimal batch size based on GPU memory and image size"""
    if gpu_memory_gb == 0:  # CPU
        return 1
    elif gpu_memory_gb < 8:  # < 8GB
        return 1 if imgsz >= 1280 else 2
    elif gpu_memory_gb < 16:  # 8-16GB
        return 2 if imgsz >= 1280 else 4
    elif gpu_memory_gb < 24:  # 16-24GB
        return 3 if imgsz >= 1280 else 6
    else:  # >= 24GB
        return 4 if imgsz >= 1280 else 8

def optimize_training_params(device, gpu_memory_gb, imgsz=1280):
    """Optimize training parameters for high-resolution images"""
    params = {
        'batch': get_optimal_batch_size(gpu_memory_gb, imgsz),
        'workers': min(8, os.cpu_count()) if device != 'cpu' else 2,
        'cache': False,  # Disable caching for large images to save memory
        'amp': True if device != 'cpu' else False,  # Mixed precision training
    }
    
    print(f"Optimized parameters for {imgsz}x{imgsz} images:")
    print(f"  Batch size: {params['batch']}")
    print(f"  Workers: {params['workers']}")
    print(f"  Cache: {params['cache']}")
    print(f"  Mixed precision: {params['amp']}")
    
    return params

if __name__ == '__main__':
    # Memory cleanup
    torch.cuda.empty_cache()
    gc.collect()
    
    # Get current script directory
    current_dir = Path(__file__).parent
    
    # Setup paths for VisDrone dataset
    data_yaml = current_dir / 'data/visdrone.yaml'
    model_config = current_dir / 'ultralytics/cfg/models/uavdetr-r18.yaml'
    
    # Check if paths exist
    if not data_yaml.exists():
        print(f"Error: Data config {data_yaml} not found!")
        print("Please run: python convert_visdrone_to_detr.py first")
        exit(1)
    
    if not model_config.exists():
        print(f"Error: Model config {model_config} not found!")
        exit(1)
    
    # Setup device and get GPU memory info
    device, gpu_memory = setup_device()
    
    # Image size for high-resolution training
    imgsz = 1280
    
    # Get optimized training parameters
    train_params = optimize_training_params(device, gpu_memory, imgsz)
    
    print(f"\n{'='*60}")
    print(f"UAV-DETR High-Resolution Training Configuration")
    print(f"{'='*60}")
    print(f"Data config: {data_yaml}")
    print(f"Model config: {model_config}")
    print(f"Device: {device}")
    print(f"Image size: {imgsz}x{imgsz}")
    print(f"Batch size: {train_params['batch']}")
    print(f"Workers: {train_params['workers']}")
    print(f"Mixed precision: {train_params['amp']}")
    print(f"{'='*60}\n")
    
    try:
        # Initialize model
        print("Initializing UAV-DETR model...")
        model = RTDETR(str(model_config))
        
        # Update model config for VisDrone (10 classes)
        model.model.nc = 10
        print(f"Model configured for {model.model.nc} classes (VisDrone)")
        
        # Start training with optimized parameters
        print("\nStarting high-resolution training...")
        results = model.train(
            data=str(data_yaml),
            cache=train_params['cache'],
            imgsz=imgsz,
            epochs=120,  # Reduced epochs for 1280 due to longer training time
            batch=train_params['batch'],
            workers=train_params['workers'],
            device=device,
            amp=train_params['amp'],  # Mixed precision training
            resume='runs/train/uavdetr_visdrone_1280_batch4/weights/last.pt',  # Resume from checkpoint
            project='runs/train',
            name=f'uavdetr_visdrone_1280_batch{train_params["batch"]}',
            patience=30,  # Early stopping patience
            optimizer='AdamW',  # AdamW optimizer for better performance
            lr0=0.001,  # Base learning rate
            lrf=0.01,  # Final learning rate factor
            warmup_epochs=5,  # Warmup epochs
            cos_lr=True,  # Cosine learning rate scheduler
            save_period=10,  # Save checkpoint every 10 epochs
            # Additional optimizations for high-resolution training
            close_mosaic=0,  # Disable mosaic in final epochs
            # Data augmentation adjustments for large images
            hsv_h=0.015,  # HSV-Hue augmentation
            hsv_s=0.7,    # HSV-Saturation augmentation
            hsv_v=0.4,    # HSV-Value augmentation
            degrees=0.0,  # Rotation degrees
            translate=0.1,  # Translation fraction
            scale=0.5,    # Scale fraction
            shear=0.0,    # Shear degrees
            perspective=0.0,  # Perspective transformation
            flipud=0.0,   # Vertical flip probability
            fliplr=0.5,   # Horizontal flip probability
            mosaic=1.0,   # Mosaic augmentation probability
            mixup=0.0,    # MixUp augmentation probability
        )
        
        print(f"\n{'='*60}")
        print("Training completed successfully!")
        print(f"Training resumed from: runs/train/uavdetr_visdrone_1280_batch4/weights/last.pt")
        print(f"Best model saved to: runs/train/uavdetr_visdrone_1280_batch{train_params['batch']}/weights/best.pt")
        print(f"Last model saved to: runs/train/uavdetr_visdrone_1280_batch{train_params['batch']}/weights/last.pt")
        print(f"{'='*60}")
        
        # Final memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n❌ GPU Out of Memory Error!")
            print(f"Current batch size: {train_params['batch']}")
            print(f"Recommended solutions:")
            print(f"1. Reduce batch size to {max(1, train_params['batch']//2)}")
            print(f"2. Use gradient accumulation: set batch=1 and add gradient accumulation steps")
            print(f"3. Use smaller image size (e.g., 960 or 1024)")
            print(f"4. Enable model sharding or use CPU training")
        else:
            print(f"Training error: {e}")
        
        # Emergency memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
        exit(1)
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        exit(1)
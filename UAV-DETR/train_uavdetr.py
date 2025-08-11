#!/usr/bin/env python3
"""
UAV-DETR Training Script for VisDrone Dataset

This script provides a convenient interface for training UAV-DETR models
on the VisDrone dataset with optimized parameters.

Usage:
    python train_uavdetr.py --model uavdetr-r50 --epochs 300 --batch 4
    python train_uavdetr.py --model uavdetr-r18 --epochs 200 --batch 8
    python train_uavdetr.py --resume runs/train/exp/weights/last.pt
"""

import warnings
import argparse
import os
from pathlib import Path
from ultralytics import RTDETR
import torch
import torch.distributed as dist

warnings.filterwarnings('ignore')

def setup_distributed():
    """Check if distributed training is already initialized"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
        
        # Check if already initialized
        if not dist.is_initialized():
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=world_size,
                rank=rank
            )
            print(f"Distributed training initialized: rank {rank}, world_size {world_size}, local_rank {local_rank}")
        else:
            print(f"Distributed training already initialized: rank {rank}, world_size {world_size}")
        
        torch.cuda.set_device(local_rank)
        return True
    return False

def setup_device():
    """Setup device with multi-GPU support"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Available GPUs: {gpu_count}")
        for i in range(gpu_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Check if multiple GPUs are available and requested
        if gpu_count > 1:
            print(f"Multi-GPU training available with {gpu_count} GPUs")
            # Return all available GPUs as string for Ultralytics
            device = ','.join(str(i) for i in range(gpu_count))
            print(f"Using multiple GPUs: {device}")
            return device, True
        else:
            # Single GPU
            device = '0'
            print(f"Using single GPU: {torch.cuda.get_device_name(0)}")
            return device, True
    else:
        device = 'cpu'
        print("CUDA not available, using CPU")
        return device, False

def get_model_config(model_name):
    """Get model configuration path"""
    model_configs = {
        'uavdetr-r18': 'ultralytics/cfg/models/uavdetr-r18.yaml',
        'uavdetr-r50': 'ultralytics/cfg/models/uavdetr-r50.yaml',
        'uavdetr-ev2': 'ultralytics/cfg/models/uavdetr-ev2.yaml',
    }
    return model_configs.get(model_name, 'ultralytics/cfg/models/uavdetr-r50.yaml')

def main():
    parser = argparse.ArgumentParser(description='Train UAV-DETR on VisDrone dataset')
    
    # Model configuration
    parser.add_argument('--model', type=str, default='uavdetr-r50',
                       choices=['uavdetr-r18', 'uavdetr-r50', 'uavdetr-ev2'],
                       help='UAV-DETR model variant')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=120,
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=4,
                       help='Batch size for training (memory optimized for high resolution)')
    parser.add_argument('--imgsz', type=int, default=1280,
                       help='Input image size (high resolution for small objects)')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of data loading workers (reduced for memory efficiency)')
    
    # Hardware configuration
    parser.add_argument('--device', type=str, default='0',
                       help='Device to use for training (auto/0/cpu)')
    
    # Optimization parameters
    parser.add_argument('--optimizer', type=str, default='AdamW',
                       choices=['SGD', 'Adam', 'AdamW'],
                       help='Optimizer type')
    parser.add_argument('--lr0', type=float, default=0.00001,
                       help='Ultra-small learning rate to prevent NaN (1e-5 for UAV-DETR stability)')
    parser.add_argument('--lrf', type=float, default=0.01,
                       help='Final learning rate factor')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                       help='Warmup epochs')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                       help='Weight decay for regularization')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='Momentum for SGD optimizer')
    
    # Small object detection optimizations
    parser.add_argument('--close-mosaic', type=int, default=20,
                       help='Disable mosaic augmentation in last N epochs')
    parser.add_argument('--amp', action='store_true', default=True,
                       help='Use automatic mixed precision')
    parser.add_argument('--multi-scale', action='store_true', default=False,
                       help='Use multi-scale training (disabled for memory efficiency)')
    parser.add_argument('--copy-paste', type=float, default=0.0,
                       help='Copy-paste augmentation probability (disabled for memory efficiency)')
    parser.add_argument('--mixup', type=float, default=0.0,
                       help='Mixup augmentation probability (disabled for memory efficiency)')
    
    # Training options - CRITICAL: Default to no-cache to prevent GPU memory explosion
    parser.add_argument('--cache', action='store_true', default=False,
                       help='Cache images for faster training (DISABLED by default to prevent OOM)')
    parser.add_argument('--no-cache', action='store_true', default=True,
                       help='Disable image caching for memory efficiency (DEFAULT: True to prevent 28GB GPU usage)')
    parser.add_argument('--cos-lr', action='store_true', default=True,
                       help='Use cosine learning rate scheduler')
    parser.add_argument('--patience', type=int, default=30,
                       help='Early stopping patience')
    parser.add_argument('--save-period', type=int, default=10,
                       help='Save checkpoint every N epochs')
    
    # Output configuration
    parser.add_argument('--project', type=str, default='runs/train',
                       help='Project directory')
    parser.add_argument('--name', type=str, default='uavdetr_visdrone',
                       help='Experiment name')
    
    # Resume training
    parser.add_argument('--resume', type=str, default='',
                       help='Resume training from checkpoint')
    
    args = parser.parse_args()
    
    print("UAV-DETR Training on VisDrone Dataset")
    print("=" * 50)
    
    # Setup paths
    current_dir = Path(__file__).parent
    data_yaml = current_dir / 'data/visdrone.yaml'
    model_config = current_dir / get_model_config(args.model)
    
    # Check if paths exist
    if not data_yaml.exists():
        print(f"Error: Data config {data_yaml} not found!")
        print("Please run: python convert_visdrone_to_detr.py first")
        return
    
    if not model_config.exists():
        print(f"Error: Model config {model_config} not found!")
        return
    
    # Setup device
    if args.device == 'auto':
        device, has_gpu = setup_device()
    else:
        device = args.device
        has_gpu = device != 'cpu' and torch.cuda.is_available()
    
    # Adjust parameters based on device
    batch_size = args.batch if has_gpu else max(1, args.batch // 2)
    workers = args.workers if has_gpu else max(1, args.workers // 2)
    
    print(f"Model: {args.model}")
    print(f"Data config: {data_yaml}")
    print(f"Model config: {model_config}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Workers: {workers}")
    print(f"Epochs: {args.epochs}")
    print("=" * 50)
    
    try:
        # Setup distributed training if available (but avoid conflicts)
        is_distributed = setup_distributed() if 'RANK' in os.environ else False
        
        # Aggressive memory optimizations - prevent cache-induced OOM
        if has_gpu:
            torch.cuda.empty_cache()
            # Force disable caching environment variables to prevent 28GB GPU memory usage
            os.environ['YOLO_CACHE'] = 'False'
            os.environ['ULTRALYTICS_CACHE'] = 'False' 
            os.environ['ULTRALYTICS_CACHE_DISABLE'] = '1'
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:256'
        
        # Initialize model
        model = RTDETR(str(model_config))
        
        # Update model for VisDrone classes
        if hasattr(model.model, 'nc'):
            model.model.nc = 10
        
        # Optimize image size for small object detection
        if args.imgsz < 1024:
            print(f"⚠️  Warning: Image size {args.imgsz} may be too small for optimal small object detection")
            print(f"   Recommended: 1280+ for best small object performance")
        
        # Adjust batch size based on image size and available memory
        if args.imgsz >= 1280 and batch_size > 8:
            suggested_batch = max(4, batch_size // 2)
            print(f"⚠️  Large image size ({args.imgsz}) with batch size {batch_size}")
            print(f"   Consider reducing batch size to {suggested_batch} to avoid OOM")
        
        # UAV-DETR training optimizations
        print(f"🎯 Training with UAV-DETR optimizations:")
        print(f"   - Image size: {args.imgsz} (higher resolution for small objects)")
        print(f"   - Learning rate: {args.lr0} (optimized for UAV-DETR)")
        print(f"   - Close mosaic: {args.close_mosaic} epochs (preserve small objects)")
        print(f"   - Multi-scale training: {args.multi_scale}")
        print(f"   - Mixed precision: {args.amp}")
        print()
        
        # CRITICAL: Force disable caching to prevent GPU memory explosion
        cache_setting = False  # Always disabled to prevent 28GB GPU memory usage
        print(f"🚫 Image caching: DISABLED (prevents GPU memory explosion)")
        print(f"   Previous issue: Data caching consumed 28GB GPU memory")
        print(f"   Solution: Force cache=False for all training")
        
        results = model.train(
            data=str(data_yaml),
            cache=cache_setting,
            imgsz=args.imgsz,
            epochs=args.epochs,
            batch=batch_size,
            workers=workers,
            device=device,
            resume=args.resume if args.resume else False,
            project=args.project,
            name=args.name,
            patience=args.patience,
            optimizer=args.optimizer,
            lr0=args.lr0,
            lrf=args.lrf,
            warmup_epochs=args.warmup_epochs,
            cos_lr=args.cos_lr,
            save_period=args.save_period,
            exist_ok=True,
            verbose=True,
            # UAV-DETR optimizations
            close_mosaic=args.close_mosaic,
            amp=args.amp,
            copy_paste=args.copy_paste,
            mixup=args.mixup,
        )
        
        print("\n" + "=" * 50)
        print("Training completed successfully!")
        print(f"Results saved to: {results.save_dir}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise

if __name__ == '__main__':
    main()
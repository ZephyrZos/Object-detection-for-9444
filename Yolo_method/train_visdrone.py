#!/usr/bin/env python3
"""
VisDrone Dataset Training Script for YOLOv5

Simplified training script that leverages the optimized defaults in train.py
for the VisDrone dataset configuration.

Usage:
    python train_visdrone.py --epochs 120 --batch-size 16
    python train_visdrone.py --epochs 120 --batch-size 8 --multi-gpu

Features:
    - Uses optimized VisDrone defaults from train.py
    - Support for multi-GPU training
    - Resume training capability
    - Automatic device detection (CUDA/CPU)
"""
from pathlib import Path
import sys
import os

FILE = Path(__file__).resolve()
ROOT = FILE.parent  
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# Set YOLO_CONFIG_DIR to ROOT path to avoid home quota issues
os.environ["YOLO_CONFIG_DIR"] = str(ROOT)

import argparse
import subprocess
import sys
from pathlib import Path

def build_train_command(args):
    """Build the training command using simplified parameters."""
    cmd = [sys.executable, str(ROOT / "train.py")]
    
    # Only add non-default parameters since train.py now has VisDrone defaults
    if args.epochs != 100:
        cmd.extend(["--epochs", str(args.epochs)])
    
    if args.batch_size != 16:
        cmd.extend(["--batch-size", str(args.batch_size)])
    
    if args.img_size != 640:
        cmd.extend(["--img", str(args.img_size)])
    
    if args.device != "auto":
        cmd.extend(["--device", args.device])
    
    if args.workers != 8:
        cmd.extend(["--workers", str(args.workers)])
    
    # Handle resume training
    if args.resume:
        cmd.extend(["--resume", args.resume])
    
    # Handle different model weights if not using default yolov5s
    if args.weights and args.weights != "yolov5s.pt":
        cmd.extend(["--weights", args.weights])
    
    # Handle custom configuration
    if args.cfg:
        cmd.extend(["--cfg", args.cfg])
    
    # Add optional parameters
    if args.cache:
        cmd.append("--cache")
    
    if args.multi_scale:
        cmd.append("--multi-scale")
    
    if args.sync_bn:
        cmd.append("--sync-bn")
    
    if args.single_cls:
        cmd.append("--single-cls")
    
    # Custom project and name
    if args.project != "runs/train":
        cmd.extend(["--project", args.project])
    
    if args.name != "visdrone":
        cmd.extend(["--name", args.name])
    
    return cmd

def setup_distributed_training(args):
    """Setup distributed training command for multi-GPU."""
    if args.multi_gpu:
        # Count available GPUs
        try:
            import torch
            num_gpus = torch.cuda.device_count()
            
            if num_gpus < 2:
                print(f"Warning: Only {num_gpus} GPU(s) available. Multi-GPU training disabled.")
                return None
            
            # Build distributed command
            dist_cmd = [
                sys.executable, "-m", "torch.distributed.run",
                "--nproc_per_node", str(num_gpus),
                "--master_port", str(args.master_port)
            ]
            
            return dist_cmd
        except ImportError:
            print("Warning: PyTorch not available. Multi-GPU training disabled.")
            return None
    
    return None

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv5 on VisDrone dataset (using optimized defaults)')
    
    # Training parameters (with train.py defaults)
    parser.add_argument('--epochs', type=int, default=120,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training (default: 16)')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Input image size (default: 640)')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of data loading workers (default: 8)')
    
    # Model configuration (optional overrides)
    parser.add_argument('--weights', type=str, default='',
                       help='Model weights path (default: yolov5s.pt from train.py)')
    parser.add_argument('--cfg', type=str, default='',
                       help='Model config path (default: yolov5s-visdrone.yaml from train.py)')
    
    # Hardware configuration
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (default: auto-detect from train.py)')
    parser.add_argument('--multi-gpu', action='store_true',
                       help='Use all available GPUs for distributed training')
    parser.add_argument('--master-port', type=int, default=12355,
                       help='Master port for distributed training')
    
    # Optimization options
    parser.add_argument('--cache', action='store_true',
                       help='Cache images for faster training')
    parser.add_argument('--multi-scale', action='store_true',
                       help='Use multi-scale training')
    parser.add_argument('--sync-bn', action='store_true',
                       help='Use synchronized batch normalization')
    parser.add_argument('--single-cls', action='store_true',
                       help='Train as single-class detector')
    
    # Output configuration
    parser.add_argument('--name', type=str, default='visdrone',
                       help='Experiment name (default: visdrone)')
    parser.add_argument('--project', type=str, default='runs/train',
                       help='Project directory (default: runs/train)')
    
    # Resume training
    parser.add_argument('--resume', type=str, default='',
                       help='Resume training from checkpoint')
    
    args = parser.parse_args()
    
    print("VisDrone Training Script - Using optimized defaults from train.py")
    print("=" * 60)
    
    # Validate arguments
    if args.multi_gpu and args.device != 'auto':
        print("Warning: --device argument may be ignored when using --multi-gpu")
    
    # Build training command
    train_cmd = build_train_command(args)
    
    # Setup distributed training if requested
    dist_cmd = setup_distributed_training(args)
    if dist_cmd:
        train_cmd = dist_cmd + train_cmd[1:]  # Remove 'python' from train_cmd
        print("Multi-GPU training enabled")
    
    # Print command
    print("Training command:")
    print(" ".join(train_cmd))
    print("=" * 60)
    
    # Start training
    try:
        subprocess.run(train_cmd, check=True)
        print("\n" + "=" * 60)
        print("Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with error code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
UAV-DETR Resume Training Script with OOM Prevention
Based on train.py, designed for continuing training from checkpoints
Includes memory optimization and automatic batch size adjustment
"""

import warnings
import os
import argparse
import gc
import psutil
from pathlib import Path
from ultralytics import RTDETR
import torch
from ultralytics.nn.tasks import RTDETRDetectionModel

# Add safe globals for PyTorch serialization
torch.serialization.add_safe_globals({'RTDETRDetectionModel': RTDETRDetectionModel})

# Force single-GPU training to prevent DDP serialization issues
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Force single GPU
os.environ['LOCAL_RANK'] = '0'  # Force single-process training
os.environ['WORLD_SIZE'] = '1'  # Explicitly set world size to 1
os.environ['RANK'] = '0'  # Set rank to 0

warnings.filterwarnings('ignore')

def get_gpu_memory_info():
    """Get GPU memory information"""
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        allocated_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        cached_memory = torch.cuda.memory_reserved() / 1024**3  # GB
        free_memory = total_memory - cached_memory
        return {
            'total': total_memory,
            'allocated': allocated_memory,
            'cached': cached_memory,
            'free': free_memory
        }
    return None

def optimize_memory():
    """Clean GPU and CPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def estimate_optimal_batch_size(gpu_memory_gb, img_size=640, safety_factor=0.7):
    """Estimate optimal batch size based on GPU memory"""
    # Empirical formula: UAV-DETR needs ~1.2GB per sample at 640x640
    memory_per_sample = 1.2 * (img_size / 640) ** 2
    max_batch = int((gpu_memory_gb * safety_factor) / memory_per_sample)
    
    # Ensure reasonable batch size
    return max(1, min(max_batch, 16))  # Limit between 1-16

def monitor_memory_usage():
    """Monitor and log current memory usage"""
    if torch.cuda.is_available():
        gpu_info = get_gpu_memory_info()
        if gpu_info:
            print(f"Memory Usage - GPU: {gpu_info['allocated']:.1f}GB/{gpu_info['total']:.1f}GB "
                  f"({gpu_info['allocated']/gpu_info['total']*100:.1f}%)")
    
    cpu_memory = psutil.virtual_memory()
    print(f"Memory Usage - CPU: {(cpu_memory.total-cpu_memory.available)/1024**3:.1f}GB/"
          f"{cpu_memory.total/1024**3:.1f}GB ({cpu_memory.percent:.1f}%)")

def check_path(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")

def setup_device(device_id=None):
    """Setup device with automatic fallback and memory check"""
    if torch.cuda.is_available():
        if device_id is not None:
            device = str(device_id)
            torch.cuda.set_device(device_id)
            print(f"Using GPU {device_id}: {torch.cuda.get_device_name(device_id)}")
        else:
            device = '0'
            torch.cuda.set_device(0)
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        
        # Clean memory and display memory info
        optimize_memory()
        gpu_info = get_gpu_memory_info()
        if gpu_info:
            print(f"GPU Memory - Total: {gpu_info['total']:.1f}GB, "
                  f"Free: {gpu_info['free']:.1f}GB, "
                  f"Allocated: {gpu_info['allocated']:.1f}GB")
    else:
        device = 'cpu'
        print("CUDA not available, using CPU")
    return device

def find_latest_checkpoint(run_dir):
    """Find the latest checkpoint in a run directory"""
    run_path = Path(run_dir)
    weights_dir = run_path / 'weights'
    
    if not weights_dir.exists():
        return None
    
    # Check for last.pt first (most recent)
    last_pt = weights_dir / 'last.pt'
    if last_pt.exists():
        return str(last_pt)
    
    # If no last.pt, find highest epoch number
    epoch_files = list(weights_dir.glob('epoch*.pt'))
    if epoch_files:
        epoch_nums = []
        for f in epoch_files:
            try:
                epoch_num = int(f.stem.replace('epoch', ''))
                epoch_nums.append((epoch_num, f))
            except ValueError:
                continue
        
        if epoch_nums:
            latest_epoch, latest_file = max(epoch_nums, key=lambda x: x[0])
            return str(latest_file)
    
    return None

def main():
    parser = argparse.ArgumentParser(description='UAV-DETR Resume Training')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file (.pt) or run directory')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Total epochs to train (default: 200)')
    parser.add_argument('--batch-size', type=int, default=0,
                       help='Batch size (0 for auto-detection, default: 0)')
    parser.add_argument('--max-batch-size', type=int, default=8,
                       help='Maximum allowed batch size (default: 8)')
    parser.add_argument('--device', type=int, default=0,
                       help='GPU device ID (default: 0)')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of workers (default: 4, reduced for memory)')
    parser.add_argument('--project', type=str, default='runs/train',
                       help='Project directory (default: runs/train)')
    parser.add_argument('--name', type=str, default=None,
                       help='Experiment name (auto-generated if not provided)')
    parser.add_argument('--patience', type=int, default=50,
                       help='Early stopping patience (default: 50)')
    parser.add_argument('--lr0', type=float, default=0.0001,
                       help='Initial learning rate (default: 0.0001)')
    parser.add_argument('--save-period', type=int, default=10,
                       help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Image size for training (default: 640)')
    parser.add_argument('--gradient-accumulation', type=int, default=1,
                       help='Gradient accumulation steps (default: 1)')
    parser.add_argument('--memory-fraction', type=float, default=0.9,
                       help='Fraction of GPU memory to use (default: 0.9)')
    
    args = parser.parse_args()
    
    # Initial memory optimization
    optimize_memory()
    
    # Get current script directory
    current_dir = Path(__file__).parent
    
    # Setup paths
    data_yaml = current_dir / 'data/visdrone.yaml'
    
    # Check if data config exists
    if not data_yaml.exists():
        print(f"Error: Data config {data_yaml} not found!")
        print("Please run: python convert_visdrone_to_detr.py first")
        exit(1)
    
    # Setup device
    device = setup_device(args.device)
    
    # Auto-detect optimal batch size
    if args.batch_size == 0 and torch.cuda.is_available():
        gpu_info = get_gpu_memory_info()
        if gpu_info:
            optimal_batch = estimate_optimal_batch_size(
                gpu_info['free'], args.img_size, args.memory_fraction
            )
            args.batch_size = min(optimal_batch, args.max_batch_size)
            print(f"Auto-detected optimal batch size: {args.batch_size}")
        else:
            args.batch_size = 2  # Conservative default
            print("Could not detect GPU memory, using conservative batch size: 2")
    elif args.batch_size == 0:
        args.batch_size = 1  # CPU fallback
        print("Using CPU, setting batch size to 1")
    
    # Configure PyTorch memory management
    if torch.cuda.is_available():
        # Limit GPU memory growth
        torch.backends.cudnn.benchmark = False  # Reduce memory fragmentation
        torch.backends.cudnn.deterministic = True
        
        # Set memory allocation strategy
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    # Configure DataLoader settings for memory optimization
    # These will be used by the underlying training code
    os.environ['ULTRALYTICS_PIN_MEMORY'] = 'False'
    os.environ['ULTRALYTICS_PERSISTENT_WORKERS'] = 'False'

    # Setup model config path
    model_config = current_dir / 'ultralytics/cfg/models/uavdetr-r50.yaml'
    if not model_config.exists():
        print(f"Error: Model config {model_config} not found!")
        exit(1)

    # Determine checkpoint path
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.is_dir():
        # If directory provided, find latest checkpoint
        latest_checkpoint = find_latest_checkpoint(checkpoint_path)
        if latest_checkpoint is None:
            print(f"Error: No checkpoint found in directory {checkpoint_path}")
            print("Available files:")
            weights_dir = checkpoint_path / 'weights'
            if weights_dir.exists():
                for f in weights_dir.iterdir():
                    print(f"  {f}")
            exit(1)
        checkpoint_path = Path(latest_checkpoint)

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint {checkpoint_path} not found!")
        exit(1)

    # Generate experiment name if not provided
    if args.name is None:
        base_name = checkpoint_path.parent.parent.name  # Get original run name
        args.name = f"{base_name}_resume"

    print("=" * 70)
    print("UAV-DETR Resume Training with OOM Prevention")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data config: {data_yaml}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size} (max: {args.max_batch_size})")
    print(f"Workers: {args.workers}")
    print(f"Total epochs: {args.epochs}")
    print(f"Image size: {args.img_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation}")
    print(f"Memory fraction: {args.memory_fraction}")
    print(f"Experiment name: {args.name}")
    
    # Display system resource info
    if torch.cuda.is_available():
        gpu_info = get_gpu_memory_info()
        if gpu_info:
            print(f"GPU Memory: {gpu_info['free']:.1f}GB free / {gpu_info['total']:.1f}GB total")
    
    # CPU memory info
    cpu_memory = psutil.virtual_memory()
    print(f"CPU Memory: {cpu_memory.available/1024**3:.1f}GB free / {cpu_memory.total/1024**3:.1f}GB total")
    print("=" * 70)

    try:
        # Initialize model from config (yaml)
        print("Loading model from config...")
        model = RTDETR(str(model_config))

        # Update model config for VisDrone (10 classes)
        if hasattr(model.model, 'nc'):
            model.model.nc = 10

        print("Starting resume training...")

        # Optimize memory before training
        optimize_memory()
        
        # Prepare training parameters (only verified ones)
        # Force single device to prevent DDP serialization issues
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device) if ',' not in str(device) else str(device).split(',')[0]
        os.environ['LOCAL_RANK'] = '0'  # Force single-process training to disable DDP
        os.environ['WORLD_SIZE'] = '1'  # Explicitly set world size to 1
        os.environ['RANK'] = '0'  # Set rank to 0
        
        train_params = {
            'data': str(data_yaml),
            'epochs': args.epochs,
            'batch': args.batch_size,
            'workers': args.workers,
            'device': 0,  # Force single GPU index to prevent DDP
            'resume': str(checkpoint_path),
            'project': args.project,
            'name': args.name,
            'exist_ok': True,
            'patience': args.patience,
            'optimizer': 'AdamW',
            'lr0': args.lr0,
            'lrf': 0.01,
            'warmup_epochs': 3,
            'save_period': args.save_period,
            'verbose': True,
            'single_cls': False,  # Ensure multi-class training
            'rect': False,  # Disable rectangular training to prevent issues
        }
        
        print("Training parameters:")
        for key, value in train_params.items():
            print(f"  {key}: {value}")
        print()
        
        # Start training with verified parameters
        results = model.train(**train_params)

        print("\n" + "=" * 70)
        print("Resume training completed successfully!")
        print(f"Results saved to: {results.save_dir}")
        
        # Final memory cleanup and monitoring
        optimize_memory()
        print("\nFinal memory status:")
        monitor_memory_usage()
        print("=" * 70)

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n❌ CUDA Out of Memory Error!")
            print(f"Current batch size: {args.batch_size}")
            print(f"Suggested solutions:")
            print(f"1. Reduce batch size: --batch-size {max(1, args.batch_size//2)}")
            print(f"2. Reduce image size: --img-size {args.img_size//2}")
            print(f"3. Use gradient accumulation: --gradient-accumulation 2")
            print(f"4. Reduce workers: --workers {max(1, args.workers//2)}")
            monitor_memory_usage()
        else:
            print(f"\nRuntime Error: {e}")
        exit(1)
    except TypeError as e:
        if "'str' object has no attribute '__module__'" in str(e) or "Expected trainer object, got string" in str(e):
            print(f"\n❌ Trainer Object Serialization Error!")
            print(f"This error occurs when the trainer object is corrupted during serialization.")
            print(f"Solutions:")
            print(f"1. The script has already been configured to force single-GPU training")
            print(f"2. Try reducing batch size: --batch-size {max(1, args.batch_size//2)}")
            print(f"3. Try reducing workers: --workers {max(1, args.workers//2)}")
            print(f"4. Check if your checkpoint file is corrupted")
            monitor_memory_usage()
        else:
            print(f"\nTypeError: {e}")
        exit(1)
    except Exception as e:
        print(f"\nError during training: {e}")
        print("Please check your checkpoint file and configuration.")
        monitor_memory_usage()
        exit(1)

if __name__ == '__main__':
    main()
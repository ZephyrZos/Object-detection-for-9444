import warnings
import os
from pathlib import Path
from ultralytics import RTDETR
import torch

warnings.filterwarnings('ignore')

def check_path(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")

def setup_device():
    """Setup device with automatic fallback"""
    if torch.cuda.is_available():
        device = '0'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("CUDA not available, using CPU")
    return device

if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    # Get current script directory
    current_dir = Path(__file__).parent
    
    # Setup paths for VisDrone dataset
    data_yaml = current_dir / 'data/visdrone.yaml'
    model_config = current_dir / 'ultralytics/cfg/models/uavdetr-r50.yaml'
    
    # Check if paths exist
    if not data_yaml.exists():
        print(f"Error: Data config {data_yaml} not found!")
        print("Please run: python convert_visdrone_to_detr.py first")
        exit(1)
    
    if not model_config.exists():
        print(f"Error: Model config {model_config} not found!")
        exit(1)
    
    # Setup device
    device = setup_device()
    
    # Adjust batch size based on device
    batch_size = 4 if device != 'cpu' else 2
    workers = 8 if device != 'cpu' else 4
    
    print(f"Training UAV-DETR on VisDrone dataset")
    print(f"Data config: {data_yaml}")
    print(f"Model config: {model_config}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    
    # Initialize model
    model = RTDETR(str(model_config))
    
    # Update model config for VisDrone (10 classes)
    model.model.nc = 10
    
    # Start training
    model.train(
        data=str(data_yaml),
        cache=False,
        imgsz=640,
        epochs=300,  # Reduced from 400 for initial training
        batch=batch_size,
        workers=workers,
        device=device,
        # resume='', # last.pt path for resume training
        project='runs/train',
        name='uavdetr_visdrone',
        patience=50,  # Early stopping patience
        optimizer='AdamW',  # Use AdamW optimizer for better performance
        lr0=0.0001,  # Learning rate
        lrf=0.01,  # Final learning rate factor
        warmup_epochs=5,  # Warmup epochs
        cos_lr=True,  # Use cosine learning rate scheduler
        save_period=10,  # Save checkpoint every 10 epochs
    )
#!/usr/bin/env python3
"""
Debug script to test resume functionality
"""

import warnings
import os
import argparse
from pathlib import Path
import torch
from ultralytics.nn.tasks import RTDETRDetectionModel

# Add safe globals for PyTorch 2.6 - Fix the format and add dill support
try:
    import dill
    torch.serialization.add_safe_globals([RTDETRDetectionModel, dill._dill._load_type])
except ImportError:
    torch.serialization.add_safe_globals([RTDETRDetectionModel])

warnings.filterwarnings('ignore')

def test_checkpoint_loading():
    """Test loading the checkpoint file"""
    checkpoint_path = 'runs/train/uavdetr_single_gpu_batch8_20250726_014942/weights/last.pt'
    
    print("Testing checkpoint loading...")
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print(f"✅ Checkpoint loaded successfully")
        print(f"Checkpoint keys: {list(ckpt.keys())}")
        print(f"train_args exists: {'train_args' in ckpt}")
        return ckpt
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return None

def test_attempt_load_weights():
    """Test the attempt_load_weights function"""
    from ultralytics.nn.tasks import attempt_load_weights
    
    checkpoint_path = 'runs/train/uavdetr_single_gpu_batch8_20250726_014942/weights/last.pt'
    
    print("\nTesting attempt_load_weights...")
    try:
        model = attempt_load_weights(checkpoint_path, device='cpu')
        print(f"✅ attempt_load_weights successful")
        return model
    except Exception as e:
        print(f"❌ attempt_load_weights failed: {e}")
        return None

def test_check_resume_logic():
    """Test the check_resume logic manually"""
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import attempt_load_weights
    from ultralytics.utils.checks import check_file
    from ultralytics.utils.files import get_latest_run
    from pathlib import Path
    
    checkpoint_path = 'runs/train/uavdetr_single_gpu_batch8_20250726_014942/weights/last.pt'
    
    print("\nTesting check_resume logic...")
    try:
        # Step 1: Check if resume path exists
        exists = isinstance(checkpoint_path, (str, Path)) and Path(checkpoint_path).exists()
        print(f"Checkpoint exists: {exists}")
        
        # Step 2: Get the checkpoint path
        last = Path(check_file(checkpoint_path) if exists else get_latest_run())
        print(f"Resolved checkpoint path: {last}")
        
        # Step 3: Load checkpoint and get args
        ckpt_args = attempt_load_weights(last).args
        print(f"✅ Checkpoint args loaded successfully")
        print(f"Args keys: {list(ckpt_args.keys()) if ckpt_args else 'None'}")
        
        # Step 4: Check if data YAML exists
        if ckpt_args and 'data' in ckpt_args:
            data_path = Path(ckpt_args['data'])
            data_exists = data_path.exists()
            print(f"Data path: {data_path}")
            print(f"Data path exists: {data_exists}")
            
            if not data_exists:
                print("⚠️ Data path doesn't exist, would use current args.data")
        
        return True
        
    except Exception as e:
        print(f"❌ check_resume logic failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ultralytics_resume():
    """Test Ultralytics resume functionality"""
    from ultralytics import RTDETR
    
    print("\nTesting Ultralytics resume...")
    try:
        # Initialize model
        model = RTDETR('ultralytics/cfg/models/uavdetr-r50.yaml')
        
        # Test resume
        checkpoint_path = 'runs/train/uavdetr_single_gpu_batch8_20250726_014942/weights/last.pt'
        
        print("Starting resume test...")
        results = model.train(
            data='data/visdrone.yaml',
            epochs=1,  # Just 1 epoch for testing
            batch=2,   # Small batch for testing
            device='cpu',
            resume=checkpoint_path,
            project='runs/train',
            name='debug_resume_test',
            exist_ok=True,
            verbose=True,
        )
        print("✅ Ultralytics resume successful")
        return True
        
    except Exception as e:
        print(f"❌ Ultralytics resume failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("UAV-DETR Resume Debug Script")
    print("=" * 60)
    
    # Test 1: Basic checkpoint loading
    ckpt = test_checkpoint_loading()
    
    # Test 2: attempt_load_weights
    model = test_attempt_load_weights()
    
    # Test 3: Manual check_resume logic
    check_resume_success = test_check_resume_logic()
    
    # Test 4: Full Ultralytics resume (only if previous tests pass)
    if check_resume_success:
        ultralytics_success = test_ultralytics_resume()
    else:
        print("\nSkipping Ultralytics resume test due to previous failures")
        ultralytics_success = False
    
    print("\n" + "=" * 60)
    print("Debug Summary:")
    print(f"Checkpoint loading: {'✅' if ckpt else '❌'}")
    print(f"attempt_load_weights: {'✅' if model else '❌'}")
    print(f"check_resume logic: {'✅' if check_resume_success else '❌'}")
    print(f"Ultralytics resume: {'✅' if ultralytics_success else '❌'}")
    print("=" * 60)

if __name__ == '__main__':
    main() 
#!/usr/bin/env python3
"""
Data Integrity Check Script for UAV-DETR Training
Diagnoses potential data issues causing NaN losses
"""

import os
import glob
import numpy as np
from pathlib import Path
import yaml
from PIL import Image

def check_dataset_integrity():
    """Comprehensive dataset integrity check"""
    print("=" * 60)
    print("UAV-DETR Dataset Integrity Check")
    print("=" * 60)
    
    # Load dataset configuration
    data_yaml = Path("data/visdrone.yaml")
    if not data_yaml.exists():
        print("❌ Dataset config file not found!")
        return False
    
    with open(data_yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset_path = Path(config['path'])
    train_img_dir = dataset_path / config['train']
    train_lbl_dir = dataset_path / config['train'].replace('images', 'labels')
    val_img_dir = dataset_path / config['val']
    val_lbl_dir = dataset_path / config['val'].replace('images', 'labels')
    
    print(f"📁 Dataset Configuration:")
    print(f"   Root path: {dataset_path}")
    print(f"   Train images: {train_img_dir}")
    print(f"   Train labels: {train_lbl_dir}")
    print(f"   Val images: {val_img_dir}")
    print(f"   Val labels: {val_lbl_dir}")
    print(f"   Number of classes: {config['nc']}")
    print(f"   Class names: {config['names']}")
    
    # Check if directories exist
    dirs_exist = True
    for path, name in [(train_img_dir, "Train images"), (train_lbl_dir, "Train labels"), 
                       (val_img_dir, "Val images"), (val_lbl_dir, "Val labels")]:
        if path.exists():
            print(f"   ✅ {name}: {path} exists")
        else:
            print(f"   ❌ {name}: {path} NOT FOUND")
            dirs_exist = False
    
    if not dirs_exist:
        return False
    
    return check_train_data(train_img_dir, train_lbl_dir, config['nc'])

def check_train_data(img_dir, lbl_dir, num_classes):
    """Check training data integrity"""
    print(f"\n🔍 Training Data Integrity Check:")
    
    # Get all image files
    img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    img_files = []
    for ext in img_extensions:
        img_files.extend(glob.glob(str(img_dir / ext)))
    
    img_files.sort()
    print(f"   Found {len(img_files)} image files")
    
    if len(img_files) == 0:
        print("   ❌ No training images found!")
        return False
    
    # Check corresponding label files
    missing_labels = []
    invalid_labels = []
    coord_issues = []
    class_issues = []
    empty_labels = 0
    
    sample_size = min(100, len(img_files))  # Check first 100 files for speed
    print(f"   Checking first {sample_size} files for issues...")
    
    for i, img_file in enumerate(img_files[:sample_size]):
        if i % 20 == 0:
            print(f"   Progress: {i}/{sample_size}")
        
        # Get corresponding label file
        img_name = Path(img_file).stem
        lbl_file = lbl_dir / f"{img_name}.txt"
        
        if not lbl_file.exists():
            missing_labels.append(img_name)
            continue
        
        # Check label file content
        try:
            with open(lbl_file, 'r') as f:
                lines = f.readlines()
            
            if len(lines) == 0:
                empty_labels += 1
                continue
            
            # Check each annotation line
            for line_num, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) != 5:
                    invalid_labels.append(f"{img_name}:{line_num} - Wrong format: {line}")
                    continue
                
                try:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])
                    
                    # Check class ID range
                    if class_id < 0 or class_id >= num_classes:
                        class_issues.append(f"{img_name}:{line_num} - Invalid class {class_id}")
                    
                    # Check coordinate ranges [0, 1]
                    coords = [x_center, y_center, width, height]
                    for coord_name, coord_val in zip(['x', 'y', 'w', 'h'], coords):
                        if coord_val < 0 or coord_val > 1:
                            coord_issues.append(f"{img_name}:{line_num} - {coord_name}={coord_val:.4f} out of range")
                            break
                
                except ValueError as e:
                    invalid_labels.append(f"{img_name}:{line_num} - Parse error: {line}")
        
        except Exception as e:
            invalid_labels.append(f"{img_name} - File read error: {e}")
    
    # Report results
    print(f"\n📊 Data Check Results:")
    print(f"   Total images checked: {sample_size}")
    print(f"   Missing labels: {len(missing_labels)}")
    print(f"   Empty label files: {empty_labels}")
    print(f"   Invalid label format: {len(invalid_labels)}")
    print(f"   Coordinate issues: {len(coord_issues)}")
    print(f"   Class ID issues: {len(class_issues)}")
    
    # Show sample issues
    if missing_labels:
        print(f"\n❌ Sample missing labels: {missing_labels[:5]}")
    if invalid_labels:
        print(f"\n❌ Sample invalid labels: {invalid_labels[:5]}")
    if coord_issues:
        print(f"\n❌ Sample coordinate issues: {coord_issues[:5]}")
    if class_issues:
        print(f"\n❌ Sample class issues: {class_issues[:5]}")
    
    # Check a few sample images
    print(f"\n🖼️  Sample Image Check:")
    for img_file in img_files[:3]:
        try:
            img = Image.open(img_file)
            print(f"   {Path(img_file).name}: {img.size} pixels, mode: {img.mode}")
        except Exception as e:
            print(f"   ❌ {Path(img_file).name}: Error loading - {e}")
    
    # Determine if data is healthy
    total_issues = len(missing_labels) + len(invalid_labels) + len(coord_issues) + len(class_issues)
    if total_issues > sample_size * 0.1:  # More than 10% issues
        print(f"\n❌ DATA ISSUES DETECTED: {total_issues} issues in {sample_size} samples")
        print("   This could cause NaN losses during training!")
        return False
    else:
        print(f"\n✅ Data appears healthy: {total_issues} minor issues in {sample_size} samples")
        return True

def check_model_config():
    """Check model configuration"""
    print(f"\n🔧 Model Configuration Check:")
    
    model_config = Path("ultralytics/cfg/models/uavdetr-r18.yaml")
    if not model_config.exists():
        print(f"   ❌ Model config not found: {model_config}")
        return False
    
    with open(model_config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"   ✅ Model config found: {model_config}")
    print(f"   Number of classes: {config.get('nc', 'Not specified')}")
    
    # Check for potential problematic modules
    problematic_modules = ['DySample', 'MFFF', 'FrequencyFocusedDownSampling', 'SemanticAlignmenCalibration']
    found_modules = []
    
    if 'head' in config:
        for layer in config['head']:
            if len(layer) > 2 and layer[2] in problematic_modules:
                found_modules.append(layer[2])
    
    if found_modules:
        print(f"   ⚠️  Complex modules found: {found_modules}")
        print("   These modules might cause numerical instability")
    
    return True

def main():
    """Main diagnostic function"""
    try:
        # Change to UAV-DETR directory
        os.chdir('/g/data/zq94/zz9919/Drone_ObjectDetection/UAV-DETR')
        
        print("Starting comprehensive dataset check...")
        
        # Check dataset integrity
        data_ok = check_dataset_integrity()
        
        # Check model configuration
        model_ok = check_model_config()
        
        print(f"\n" + "=" * 60)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 60)
        
        if data_ok and model_ok:
            print("✅ No obvious data/config issues detected")
            print("\n🔍 NaN losses might be caused by:")
            print("   • Model architecture numerical instability")
            print("   • Mixed precision training issues")
            print("   • High resolution (1280px) computation overflow")
            print("   • Small batch size (4) causing unstable batch norm")
            
            print(f"\n💡 Suggested fixes:")
            print("   1. Try lower resolution (640px)")
            print("   2. Disable AMP (remove --amp flag)")
            print("   3. Increase batch size to 8")
            print("   4. Use simpler model variant if available")
            
        else:
            print("❌ Data or configuration issues detected!")
            print("   Fix these issues before training")
        
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
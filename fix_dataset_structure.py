#!/usr/bin/env python3
"""
Dataset Structure Fix Script
Fix YOLO dataset structure for VisDrone training
"""

import os
import shutil
from pathlib import Path
import yaml

def check_dataset_structure():
    """Check current dataset structure"""
    base_dir = Path("/g/data/zq94/zz9919/Drone_ObjectDetection")
    dataset_paths = [
        base_dir / "Yolo_method" / "dataset",
        base_dir / "dataset",
    ]
    
    print("🔍 Checking dataset structure...")
    
    for dataset_path in dataset_paths:
        if dataset_path.exists():
            print(f"\n📁 Found dataset at: {dataset_path}")
            
            # Check subdirectories
            for subdir in ["train", "val", "test"]:
                sub_path = dataset_path / subdir
                if sub_path.exists():
                    print(f"  ✅ {subdir}/")
                    
                    # Check for images and labels
                    images_path = sub_path / "images"
                    labels_path = sub_path / "labels"
                    cache_file = sub_path / "labels.cache"
                    
                    print(f"    {'✅' if images_path.exists() else '❌'} images/ ({len(list(images_path.glob('*.jpg'))) if images_path.exists() else 0} files)")
                    print(f"    {'✅' if labels_path.exists() else '❌'} labels/ ({len(list(labels_path.glob('*.txt'))) if labels_path.exists() else 0} files)")  
                    print(f"    {'✅' if cache_file.exists() else '❌'} labels.cache")
                else:
                    print(f"  ❌ {subdir}/ (missing)")
            
            return dataset_path
    
    print("❌ No valid dataset found!")
    return None

def suggest_fix_options(dataset_path):
    """Suggest fix options based on current structure"""
    print(f"\n💡 Suggested fixes for {dataset_path}:")
    
    # Option 1: Use existing converted data if available
    yolo_converted = Path("/g/data/zq94/zz9919/Drone_ObjectDetection/Yolo_method/data/VisDrone_converted.yaml")
    if yolo_converted.exists():
        print("1. Use existing converted VisDrone dataset (recommended)")
        print("   - Dataset appears to be partially converted")
        print("   - Check if original conversion completed successfully")
    
    # Option 2: Re-convert from original data
    original_data_paths = [
        Path("/g/data/zq94/zz9919/Drone_ObjectDetection/OriginalData"),
        Path("/g/data/zq94/zz9919/OriginalData"),
    ]
    
    for orig_path in original_data_paths:
        if orig_path.exists():
            print(f"2. Re-convert from original data at: {orig_path}")
            print("   python convert_visdrone_to_yolo.py --original_data OriginalData --output_dir Yolo_method/dataset")
            break
    
    # Option 3: Use alternative dataset
    print("3. Alternative dataset locations:")
    alt_paths = [
        "/g/data/zq94/zz9919/Drone_ObjectDetection/Yolov8/yolo_dataset",
        "/g/data/zq94/zz9919/Drone_ObjectDetection/esod/data",
    ]
    
    for alt_path in alt_paths:
        if Path(alt_path).exists():
            print(f"   - Found alternative at: {alt_path}")

def create_minimal_dataset_config(dataset_path):
    """Create a working dataset configuration based on available data"""
    dataset_path = Path(dataset_path)
    
    # Check what's actually available
    available_structure = {}
    for split in ["train", "val", "test"]:
        split_path = dataset_path / split
        if split_path.exists():
            # Check if images directory exists or if files are directly in split directory
            if (split_path / "images").exists():
                available_structure[split] = f"{split}/images"
            elif any(split_path.glob("*.jpg")):
                available_structure[split] = split
            else:
                available_structure[split] = None
        else:
            available_structure[split] = None
    
    print(f"\n📝 Creating dataset config based on available structure:")
    for split, path in available_structure.items():
        status = "✅" if path else "❌"
        print(f"  {status} {split}: {path or 'not found'}")
    
    # Create YAML config
    config = {
        'path': str(dataset_path),
        'train': available_structure.get('train', 'train/images'),
        'val': available_structure.get('val', 'val/images'),
        'test': available_structure.get('test', 'test/images'),
        'nc': 10,
        'names': {
            0: 'pedestrian',
            1: 'people', 
            2: 'bicycle',
            3: 'car',
            4: 'van',
            5: 'truck',
            6: 'tricycle',
            7: 'awning-tricycle',
            8: 'bus',
            9: 'motor'
        }
    }
    
    # Save config
    yaml_path = dataset_path / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ Created dataset config: {yaml_path}")
    return yaml_path

def main():
    """Main function"""
    print("🔧 Dataset Structure Diagnostic Tool")
    print("=" * 50)
    
    # Check current structure
    dataset_path = check_dataset_structure()
    
    if dataset_path:
        # Suggest fixes
        suggest_fix_options(dataset_path)
        
        # Create working config
        config_path = create_minimal_dataset_config(dataset_path)
        
        print(f"\n✅ Next steps:")
        print(f"1. Review the generated config: {config_path}")
        print(f"2. If structure is incomplete, run data conversion")
        print(f"3. Test with: python train_yolo_visdrone.py")
    else:
        print("\n❌ No usable dataset found!")
        print("Please run data conversion first:")
        print("python convert_visdrone_to_yolo.py --original_data OriginalData --output_dir Yolo_method/dataset")

if __name__ == "__main__":
    main()
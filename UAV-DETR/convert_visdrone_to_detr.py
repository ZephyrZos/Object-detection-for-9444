#!/usr/bin/env python3
"""
Convert VisDrone dataset format to UAV-DETR/RT-DETR format.
This script converts VisDrone annotations to COCO format for UAV-DETR training.

VisDrone format: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
COCO format: JSON with images, annotations, and categories
"""

import os
import json
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# VisDrone class mapping (excluding ignored classes 0 and 11)
VISDRONE_CLASSES = {
    1: 'pedestrian',
    2: 'people', 
    3: 'bicycle',
    4: 'car',
    5: 'van',
    6: 'truck',
    7: 'tricycle',
    8: 'awning-tricycle',
    9: 'bus',
    10: 'motor'
}

# Map VisDrone class IDs to COCO class IDs (0-based)
CLASS_MAPPING = {
    1: 0,  # pedestrian
    2: 1,  # people
    3: 2,  # bicycle
    4: 3,  # car
    5: 4,  # van
    6: 5,  # truck
    7: 6,  # tricycle
    8: 7,  # awning-tricycle
    9: 8,  # bus
    10: 9  # motor
}

def create_coco_categories():
    """Create COCO format categories"""
    categories = []
    for visdrone_id, class_name in VISDRONE_CLASSES.items():
        coco_id = CLASS_MAPPING[visdrone_id]
        categories.append({
            'id': coco_id + 1,  # COCO categories are 1-based
            'name': class_name,
            'supercategory': 'object'
        })
    return categories

def parse_visdrone_annotation(annotation_path):
    """Parse VisDrone annotation file"""
    annotations = []
    if not os.path.exists(annotation_path):
        return annotations
    
    with open(annotation_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',')
            if len(parts) != 8:
                continue
            
            bbox_left = int(parts[0])
            bbox_top = int(parts[1])
            bbox_width = int(parts[2])
            bbox_height = int(parts[3])
            score = int(parts[4])  # confidence score (0-1)
            object_category = int(parts[5])
            truncation = int(parts[6])  # 0: no truncation, 1: partial, 2: heavy
            occlusion = int(parts[7])   # 0: no occlusion, 1: partial, 2: heavy
            
            # Skip ignored regions and others class
            if object_category == 0 or object_category == 11:
                continue
            
            # Skip objects that are too small or heavily occluded/truncated
            if bbox_width < 3 or bbox_height < 3:
                continue
            
            if object_category in CLASS_MAPPING:
                annotations.append({
                    'bbox': [bbox_left, bbox_top, bbox_width, bbox_height],
                    'category_id': object_category,
                    'score': score,
                    'truncation': truncation,
                    'occlusion': occlusion
                })
    
    return annotations

def convert_split(original_data_path, split_name, output_path):
    """Convert a single split (train/val/test) to COCO format"""
    
    images_dir = os.path.join(original_data_path, f'VisDrone2019-DET-{split_name}', 'images')
    annotations_dir = os.path.join(original_data_path, f'VisDrone2019-DET-{split_name}', 'annotations')
    
    if not os.path.exists(images_dir):
        print(f"Warning: {images_dir} not found, skipping {split_name}")
        return
    
    # Initialize COCO format structure
    coco_data = {
        'images': [],
        'annotations': [],
        'categories': create_coco_categories(),
        'info': {
            'description': f'VisDrone2019-DET {split_name} set',
            'version': '1.0',
            'year': 2019,
            'contributor': 'VisDrone',
            'date_created': '2019'
        }
    }
    
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()
    
    annotation_id = 1
    
    print(f"Converting {split_name} split...")
    for image_id, image_file in enumerate(tqdm(image_files), 1):
        image_path = os.path.join(images_dir, image_file)
        
        # Get image dimensions
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Error reading image {image_path}: {e}")
            continue
        
        # Add image info
        coco_data['images'].append({
            'id': image_id,
            'file_name': image_file,
            'width': width,
            'height': height,
            'date_captured': '',
            'flickr_url': '',
            'coco_url': '',
            'license': 1
        })
        
        # Parse annotations
        annotation_file = os.path.splitext(image_file)[0] + '.txt'
        annotation_path = os.path.join(annotations_dir, annotation_file)
        
        visdrone_annotations = parse_visdrone_annotation(annotation_path)
        
        for ann in visdrone_annotations:
            bbox = ann['bbox']
            category_id = CLASS_MAPPING[ann['category_id']] + 1  # COCO categories are 1-based
            
            # Calculate area
            area = bbox[2] * bbox[3]
            
            # Add COCO annotation
            coco_data['annotations'].append({
                'id': annotation_id,
                'image_id': image_id,
                'category_id': category_id,
                'bbox': bbox,  # [x, y, width, height]
                'area': area,
                'iscrowd': 0,
                'segmentation': []
            })
            annotation_id += 1
    
    # Save COCO format JSON
    output_file = os.path.join(output_path, f'{split_name}.json')
    os.makedirs(output_path, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"Converted {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations for {split_name}")
    print(f"Saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Convert VisDrone dataset to COCO format for UAV-DETR')
    parser.add_argument('--original_data', type=str, default='../OriginalData',
                       help='Path to original VisDrone data directory')
    parser.add_argument('--output_dir', type=str, default='./annotations',
                       help='Output directory for COCO format annotations')
    
    args = parser.parse_args()
    
    original_data_path = Path(args.original_data).resolve()
    output_path = Path(args.output_dir).resolve()
    
    print(f"Converting VisDrone dataset from {original_data_path}")
    print(f"Output will be saved to {output_path}")
    
    # Check if original data exists
    if not original_data_path.exists():
        print(f"Error: Original data path {original_data_path} does not exist!")
        return
    
    # Convert each split
    splits = ['train', 'val', 'test-dev']
    
    for split in splits:
        try:
            convert_split(str(original_data_path), split, str(output_path))
        except Exception as e:
            print(f"Error converting {split}: {e}")
    
    print("\nConversion completed!")
    print(f"COCO format annotations saved to {output_path}")

if __name__ == '__main__':
    main()
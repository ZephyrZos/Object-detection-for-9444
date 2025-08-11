#!/usr/bin/env python3
"""
VisDrone数据集转换为YOLOv5格式脚本

VisDrone格式: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
YOLOv5格式: <class_id> <x_center> <y_center> <width> <height> (归一化坐标)

作者: Claude Code
"""

import os
import shutil
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# VisDrone类别映射到YOLOv5 (0-based indexing)
VISDRONE_CLASSES = {
    0: -1,  # ignored regions
    1: 0,   # pedestrian
    2: 1,   # people
    3: 2,   # bicycle
    4: 3,   # car
    5: 4,   # van
    6: 5,   # truck
    7: 6,   # tricycle
    8: 7,   # awning-tricycle
    9: 8,   # bus
    10: 9,  # motor
}

def convert_bbox_to_yolo(bbox, img_width, img_height):
    """
    将VisDrone bbox格式转换为YOLOv5格式
    
    Args:
        bbox: [x_left, y_top, width, height] - VisDrone格式
        img_width: 图像宽度
        img_height: 图像高度
    
    Returns:
        [x_center, y_center, width, height] - YOLOv5格式（归一化）
    """
    x_left, y_top, width, height = bbox
    
    # 计算中心点坐标
    x_center = x_left + width / 2
    y_center = y_top + height / 2
    
    # 归一化
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    
    return [x_center_norm, y_center_norm, width_norm, height_norm]

def process_annotation_file(ann_file, img_file, output_label_file):
    """
    处理单个标注文件
    
    Args:
        ann_file: VisDrone标注文件路径
        img_file: 对应的图像文件路径
        output_label_file: 输出的YOLOv5标签文件路径
    """
    # 获取图像尺寸
    try:
        with Image.open(img_file) as img:
            img_width, img_height = img.size
    except Exception as e:
        print(f"Error reading image {img_file}: {e}")
        return False
    
    yolo_annotations = []
    
    # 读取VisDrone标注
    try:
        with open(ann_file, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading annotation {ann_file}: {e}")
        return False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        try:
            parts = line.split(',')
            if len(parts) < 6:
                continue
                
            bbox_left = int(parts[0])
            bbox_top = int(parts[1])
            bbox_width = int(parts[2])
            bbox_height = int(parts[3])
            score = int(parts[4])  # 通常为0或1
            object_category = int(parts[5])
            
            # 跳过ignored regions (category 0)
            if object_category == 0 or object_category not in VISDRONE_CLASSES:
                continue
            
            # 跳过无效的边界框
            if bbox_width <= 0 or bbox_height <= 0:
                continue
            
            # 转换为YOLOv5格式
            yolo_class = VISDRONE_CLASSES[object_category]
            if yolo_class == -1:
                continue
                
            yolo_bbox = convert_bbox_to_yolo([bbox_left, bbox_top, bbox_width, bbox_height], 
                                           img_width, img_height)
            
            # 确保坐标在有效范围内
            if all(0 <= coord <= 1 for coord in yolo_bbox):
                yolo_annotations.append(f"{yolo_class} {' '.join(f'{x:.6f}' for x in yolo_bbox)}")
                
        except ValueError as e:
            print(f"Error parsing line '{line}' in {ann_file}: {e}")
            continue
    
    # 写入YOLOv5标签文件
    try:
        output_label_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_label_file, 'w') as f:
            f.write('\n'.join(yolo_annotations))
            if yolo_annotations:  # 只有当有标注时才添加换行符
                f.write('\n')
    except Exception as e:
        print(f"Error writing label file {output_label_file}: {e}")
        return False
    
    return True

def convert_dataset(original_data_dir, output_dir):
    """
    转换整个VisDrone数据集
    
    Args:
        original_data_dir: OriginalData目录路径
        output_dir: 输出目录路径（Yolo_method/dataset）
    """
    original_data_dir = Path(original_data_dir)
    output_dir = Path(output_dir)
    
    # 数据集分割映射
    dataset_splits = {
        'VisDrone2019-DET-train': 'train',
        'VisDrone2019-DET-val': 'val',
        'VisDrone2019-DET-test-dev': 'test'
    }
    
    total_processed = 0
    total_images = 0
    
    for visdrone_split, yolo_split in dataset_splits.items():
        visdrone_dir = original_data_dir / visdrone_split
        
        if not visdrone_dir.exists():
            print(f"Warning: {visdrone_dir} does not exist, skipping...")
            continue
            
        images_dir = visdrone_dir / 'images'
        annotations_dir = visdrone_dir / 'annotations'
        
        if not images_dir.exists():
            print(f"Warning: {images_dir} does not exist, skipping...")
            continue
            
        # 输出目录
        output_images_dir = output_dir / yolo_split / 'images'
        output_labels_dir = output_dir / yolo_split / 'labels'
        
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取所有图像文件
        image_files = list(images_dir.glob('*.jpg'))
        total_images += len(image_files)
        
        print(f"\n处理 {yolo_split} 数据集 ({len(image_files)} 张图像)...")
        
        processed_count = 0
        for img_file in tqdm(image_files, desc=f"Converting {yolo_split}"):
            # 复制图像文件
            output_img_file = output_images_dir / img_file.name
            try:
                shutil.copy2(img_file, output_img_file)
            except Exception as e:
                print(f"Error copying image {img_file}: {e}")
                continue
            
            # 处理标注文件
            ann_file = annotations_dir / (img_file.stem + '.txt')
            output_label_file = output_labels_dir / (img_file.stem + '.txt')
            
            if ann_file.exists():
                success = process_annotation_file(ann_file, img_file, output_label_file)
                if success:
                    processed_count += 1
            else:
                # 如果没有标注文件，创建空的标签文件
                output_label_file.touch()
                processed_count += 1
        
        print(f"{yolo_split} 数据集处理完成: {processed_count}/{len(image_files)} 图像成功处理")
        total_processed += processed_count
    
    print(f"\n数据转换完成!")
    print(f"总计处理: {total_processed}/{total_images} 图像")
    print(f"输出目录: {output_dir}")

def create_dataset_yaml(output_dir):
    """
    创建数据集配置文件
    """
    yaml_content = f"""# VisDrone数据集配置文件
# 用于YOLOv5训练

# 数据集路径
path: {output_dir.absolute()}  # 数据集根目录
train: train/images  # 训练图像路径（相对于path）
val: val/images      # 验证图像路径（相对于path）
test: test/images    # 测试图像路径（相对于path）

# 类别数量
nc: 10

# 类别名称
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
    
    yaml_file = output_dir.parent / 'data' / 'VisDrone_converted.yaml'
    yaml_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(yaml_file, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"数据集配置文件已创建: {yaml_file}")

def main():
    parser = argparse.ArgumentParser(description='Convert VisDrone dataset to YOLOv5 format')
    parser.add_argument('--original_data', 
                       default='/g/data/zq94/zz9919/Drone_ObjectDetection/OriginalData',
                       help='Path to OriginalData directory')
    parser.add_argument('--output_dir',
                       default='/g/data/zq94/zz9919/Drone_ObjectDetection/Yolo_method/dataset',
                       help='Output directory for converted dataset')
    
    args = parser.parse_args()
    
    original_data_dir = Path(args.original_data)
    output_dir = Path(args.output_dir)
    
    if not original_data_dir.exists():
        print(f"Error: Original data directory {original_data_dir} does not exist!")
        return
    
    print(f"开始转换VisDrone数据集...")
    print(f"源目录: {original_data_dir}")
    print(f"输出目录: {output_dir}")
    
    # 转换数据集
    convert_dataset(original_data_dir, output_dir)
    
    # 创建数据集配置文件
    create_dataset_yaml(output_dir)
    
    print("\n转换完成! 你现在可以使用以下命令训练模型:")
    print(f"cd {output_dir.parent}")
    print("python train.py --data data/VisDrone_converted.yaml --weights yolov5s.pt --img 640")

if __name__ == '__main__':
    main()
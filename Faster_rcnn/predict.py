#!/usr/bin/env python3
"""
Faster R-CNN预测脚本
使用方法: python predict.py --image path/to/image.jpg
"""

import argparse
import torch
import cv2
import numpy as np
from PIL import Image
import yaml
import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import load_pretrained_model
from utils import draw_boxes, get_device
from dataset import get_transform


def predict_image(model, image_path, config, device):
    """预测单张图像"""
    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法加载图像: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 预处理
    transform = get_transform(train=False, config=config)
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 预测
    model.eval()
    with torch.no_grad():
        predictions = model(image_tensor)

    # 处理预测结果
    boxes = predictions[0]['boxes'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()

    # 过滤低置信度预测
    confidence_threshold = config['prediction']['confidence_threshold']
    mask = scores > confidence_threshold
    boxes = boxes[mask]
    labels = labels[mask]
    scores = scores[mask]

    # 绘制结果
    class_names = list(config['classes'].values())
    result_image = draw_boxes(image, boxes, labels, scores, class_names)

    return result_image, boxes, labels, scores


def predict_batch(model, image_dir, config, device, output_dir):
    """批量预测"""
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 支持的图像格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    # 获取所有图像文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(image_dir).glob(f'*{ext}'))
        image_files.extend(Path(image_dir).glob(f'*{ext.upper()}'))

    print(f"找到 {len(image_files)} 张图像")

    results = []

    for i, image_file in enumerate(image_files):
        print(f"处理图像 {i+1}/{len(image_files)}: {image_file.name}")

        try:
            # 预测
            result_image, boxes, labels, scores = predict_image(
                model, str(image_file), config, device
            )

            # 保存结果图像
            output_file = output_path / f"pred_{image_file.name}"
            result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_file), result_image_bgr)

            # 记录结果
            results.append({
                'image': image_file.name,
                'num_detections': len(boxes),
                'detections': []
            })

            for box, label, score in zip(boxes, labels, scores):
                class_name = config['classes'].get(str(label), f'Class {label}')
                results[-1]['detections'].append({
                    'class': class_name,
                    'confidence': float(score),
                    'bbox': box.tolist()
                })

            print(f"  检测到 {len(boxes)} 个目标")

        except Exception as e:
            print(f"  处理图像时出错: {e}")
            continue

    # 保存结果报告
    report_file = output_path / "prediction_report.json"
    import json
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"批量预测完成，结果保存在: {output_path}")
    print(f"预测报告: {report_file}")


def main():
    parser = argparse.ArgumentParser(description='Faster R-CNN预测')
    parser.add_argument('--image', type=str, help='输入图像路径')
    parser.add_argument('--dir', type=str, help='输入图像目录路径（批量预测）')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth', help='模型路径')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--output', type=str, default='output.jpg', help='输出图像路径')
    parser.add_argument('--output_dir', type=str, default='predictions', help='批量预测输出目录')
    parser.add_argument('--confidence', type=float, help='置信度阈值（覆盖配置文件）')

    args = parser.parse_args()

    # 检查输入
    if not args.image and not args.dir:
        parser.error("必须指定 --image 或 --dir 参数")

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 覆盖置信度阈值
    if args.confidence is not None:
        config['prediction']['confidence_threshold'] = args.confidence

    # 获取设备
    device = get_device(config['device'])
    print(f"使用设备: {device}")

    # 检查模型文件
    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在: {args.model}")
        print("请先训练模型或指定正确的模型路径")
        return

    # 加载模型
    print(f"加载模型: {args.model}")
    model = load_pretrained_model(args.model, config, device)

    if args.image:
        # 单张图像预测
        print(f"预测图像: {args.image}")

        try:
            result_image, boxes, labels, scores = predict_image(
                model, args.image, config, device
            )

            # 保存结果
            result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(args.output, result_image_bgr)

            print(f"预测结果已保存到: {args.output}")
            print(f"检测到 {len(boxes)} 个目标")

            # 打印检测结果
            class_names = list(config['classes'].values())
            for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
                class_name = class_names[label] if label < len(class_names) else f'Class {label}'
                print(f"  目标 {i+1}: {class_name}, 置信度: {score:.3f}, 边界框: {box}")

        except Exception as e:
            print(f"预测时出错: {e}")

    elif args.dir:
        # 批量预测
        print(f"批量预测目录: {args.dir}")
        predict_batch(model, args.dir, config, device, args.output_dir)


if __name__ == "__main__":
    main()
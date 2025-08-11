
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path


def find_latest_model_for_inference(checkpoint_base_dir="checkpoints"):
    """
    Finds the latest and most suitable model for inference or visualization.

    It searches for the latest training run directory inside `checkpoint_base_dir`
    and returns the path to `final_model.pth` if it exists, otherwise `best_model.pth`.

    Args:
        checkpoint_base_dir (str): The base directory where training runs are stored.

    Returns:
        str: The full path to the selected model checkpoint file.

    Raises:
        FileNotFoundError: If no suitable model file is found in the latest run directory.
    """
    # Find all training run directories
    run_dirs = [d for d in os.listdir(checkpoint_base_dir) if os.path.isdir(os.path.join(checkpoint_base_dir, d)) and d.startswith('faster_rcnn_')]
    if not run_dirs:
        raise FileNotFoundError(f"No training run directories found in '{checkpoint_base_dir}'.")

    # Sort them to find the latest one
    latest_run_dir_name = sorted(run_dirs)[-1]
    latest_run_path = os.path.join(checkpoint_base_dir, latest_run_dir_name)

    # Prioritize 'final_model.pth', then 'best_model.pth'
    model_path_final = os.path.join(latest_run_path, 'final_model.pth')
    model_path_best = os.path.join(latest_run_path, 'best_model.pth')

    if os.path.exists(model_path_final):
        print(f"Found final model in the latest run: {model_path_final}")
        return model_path_final
    elif os.path.exists(model_path_best):
        print(f"Found best model in the latest run: {model_path_best}")
        return model_path_best
    else:
        raise FileNotFoundError(f"No 'final_model.pth' or 'best_model.pth' found in the latest run directory: {latest_run_path}")


class AverageMeter:
    """计算和存储平均值和当前值"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False


def get_device(device_config):
    """获取设备"""
    if device_config == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    else:
        return torch.device(device_config)


def draw_boxes(image, boxes, labels, scores=None, class_names=None, color=(255, 0, 0)):
    """在图像上绘制边界框"""
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()

    if len(image.shape) == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))

    # 反归一化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)

    # 转换为PIL图像
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    # 尝试加载字体
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()

    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box

        # 绘制边界框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # 准备标签文本
        if class_names and label < len(class_names):
            label_text = class_names[label]
        else:
            label_text = f"Class {label}"

        if scores is not None and i < len(scores):
            label_text += f" {scores[i]:.2f}"

        # 绘制标签背景
        bbox = draw.textbbox((x1, y1 - 20), label_text, font=font)
        draw.rectangle(bbox, fill=color)

        # 绘制标签文本
        draw.text((x1, y1 - 20), label_text, fill=(255, 255, 255), font=font)

    return np.array(pil_image)


def visualize_predictions(model, images, targets, device, config, num_images=4):
    """可视化预测结果"""
    model.eval()

    # 获取类别名称
    class_names = list(config['classes'].values())

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    with torch.no_grad():
        for i in range(min(num_images, len(images))):
            # 准备输入
            image = images[i].unsqueeze(0).to(device)

            # 预测
            predictions = model(image)

            # 获取预测结果
            boxes = predictions[0]['boxes'].cpu().numpy()
            labels = predictions[0]['labels'].cpu().numpy()
            scores = predictions[0]['scores'].cpu().numpy()

            # 过滤低置信度预测
            confidence_threshold = config['prediction']['confidence_threshold']
            mask = scores > confidence_threshold
            boxes = boxes[mask]
            labels = labels[mask]
            scores = scores[mask]

            # 绘制图像
            image_np = images[i].cpu().numpy()
            result_image = draw_boxes(image_np, boxes, labels, scores, class_names)

            axes[i].imshow(result_image)
            axes[i].set_title(f'Image {i+1}')
            axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def evaluate_model(model, val_loader, device, config):
    """评估模型性能"""
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = [image.to(device) for image in images]

            # 预测
            predictions = model(images)

            # 收集预测和目标
            for pred, target in zip(predictions, targets):
                all_predictions.append(pred)
                all_targets.append(target)

    # 计算mAP
    mAP = calculate_map(all_predictions, all_targets, config)

    return mAP


def calculate_map(predictions, targets, config, iou_thresholds=None):
    """
    计算mAP指标 - 使用VOC风格的AP计算

    Args:
        predictions: 模型预测结果列表
        targets: 真实标签列表
        config: 配置字典
        iou_thresholds: IoU阈值列表，默认为[0.5]用于mAP50

    Returns:
        tuple: (mAP50, mAP50_95, precision, recall)
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5]

    try:
        class_names = list(config['classes'].values())[1:]  # 排除background
        num_classes = len(class_names)

        # 收集所有预测和真实标注
        all_predictions = []
        all_targets = []

        for pred, target in zip(predictions, targets):
            # 处理预测结果
            pred_boxes = pred['boxes'].cpu().numpy() if torch.is_tensor(pred['boxes']) else pred['boxes']
            pred_labels = pred['labels'].cpu().numpy() if torch.is_tensor(pred['labels']) else pred['labels']
            pred_scores = pred['scores'].cpu().numpy() if torch.is_tensor(pred['scores']) else pred['scores']

            # 处理真实标注
            target_boxes = target['boxes'].cpu().numpy() if torch.is_tensor(target['boxes']) else target['boxes']
            target_labels = target['labels'].cpu().numpy() if torch.is_tensor(target['labels']) else target['labels']

            # 添加图像ID
            image_id = len(all_predictions)

            # 添加预测结果
            for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                all_predictions.append({
                    'image_id': image_id,
                    'category_id': int(label),
                    'bbox': box,
                    'score': float(score)
                })

            # 添加真实标注
            for box, label in zip(target_boxes, target_labels):
                all_targets.append({
                    'image_id': image_id,
                    'category_id': int(label),
                    'bbox': box
                })

        # 计算每个IoU阈值下的AP
        aps_per_iou = []

        for iou_threshold in iou_thresholds:
            class_aps = []

            for class_id in range(1, num_classes + 1):
                # 获取该类别的预测和真实标注
                class_predictions = [p for p in all_predictions if p['category_id'] == class_id]
                class_targets = [t for t in all_targets if t['category_id'] == class_id]

                if len(class_targets) == 0:
                    class_aps.append(0.0)
                    continue

                # 按置信度排序预测结果
                class_predictions.sort(key=lambda x: x['score'], reverse=True)

                # 计算AP
                ap = calculate_ap_for_class(class_predictions, class_targets, iou_threshold)
                class_aps.append(ap)

            # 计算该IoU阈值下的mAP
            mean_ap = np.mean(class_aps) if class_aps else 0.0
            aps_per_iou.append(mean_ap)

        # mAP50 (IoU=0.5)
        mAP50 = aps_per_iou[0] if len(aps_per_iou) > 0 else 0.0

        # mAP50-95 (IoU从0.5到0.95，步长0.05)
        if len(iou_thresholds) == 1:
            # 如果只有一个阈值，计算多个阈值的mAP
            iou_range = np.arange(0.5, 1.0, 0.05)
            map_values = []

            for iou_thresh in iou_range:
                class_aps = []
                for class_id in range(1, num_classes + 1):
                    class_predictions = [p for p in all_predictions if p['category_id'] == class_id]
                    class_targets = [t for t in all_targets if t['category_id'] == class_id]

                    if len(class_targets) == 0:
                        class_aps.append(0.0)
                        continue

                    class_predictions.sort(key=lambda x: x['score'], reverse=True)
                    ap = calculate_ap_for_class(class_predictions, class_targets, iou_thresh)
                    class_aps.append(ap)

                map_values.append(np.mean(class_aps) if class_aps else 0.0)

            mAP50_95 = np.mean(map_values)
        else:
            mAP50_95 = np.mean(aps_per_iou)

        # 计算整体precision和recall
        precision, recall = calculate_precision_recall_overall(all_predictions, all_targets, iou_thresholds[0])

        return mAP50, mAP50_95, precision, recall

    except Exception as e:
        print(f"计算mAP时出错: {e}")
        return 0.0, 0.0, 0.0, 0.0


def calculate_ap_for_class(predictions, targets, iou_threshold):
    """
    计算单个类别的AP (Average Precision)
    使用VOC风格的11点插值方法
    """
    if len(predictions) == 0:
        return 0.0

    # 创建目标框的匹配状态
    target_matched = [False] * len(targets)

    # 计算每个预测的TP/FP
    tp = np.zeros(len(predictions))
    fp = np.zeros(len(predictions))

    for pred_idx, pred in enumerate(predictions):
        # 找到最佳匹配的真实框
        best_iou = 0.0
        best_target_idx = -1

        for target_idx, target in enumerate(targets):
            if target_matched[target_idx]:
                continue

            # 计算IoU
            iou = calculate_bbox_iou(pred['bbox'], target['bbox'])

            if iou > best_iou:
                best_iou = iou
                best_target_idx = target_idx

        # 判断TP或FP
        if best_iou >= iou_threshold and best_target_idx != -1:
            tp[pred_idx] = 1
            target_matched[best_target_idx] = True
        else:
            fp[pred_idx] = 1

    # 计算累积TP和FP
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    # 计算precision和recall
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
    recall = tp_cumsum / (len(targets) + 1e-8)

    # 使用VOC风格的11点插值计算AP
    ap = compute_voc_ap(recall, precision)

    return ap


def compute_voc_ap(recall, precision):
    """
    计算VOC风格的AP (11点插值)
    """
    # 添加边界点
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    # 计算precision的单调递减包络
    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])

    # 找到recall变化的点
    i = np.where(recall[1:] != recall[:-1])[0]

    # 计算AP (曲线下面积)
    ap = np.sum((recall[i + 1] - recall[i]) * precision[i + 1])

    return ap


def calculate_bbox_iou(box1, box2):
    """
    计算两个边界框的IoU
    box格式: [x1, y1, x2, y2]
    """
    # 计算交集
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)

    # 计算并集
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / (union + 1e-8)


def calculate_precision_recall_overall(predictions, targets, iou_threshold=0.5):
    """
    计算整体的precision和recall
    """
    if len(predictions) == 0:
        return 0.0, 0.0

    # 按置信度排序所有预测
    predictions.sort(key=lambda x: x['score'], reverse=True)

    # 为每个图像的每个类别创建目标匹配状态
    target_matched = {}
    for target in targets:
        img_id = target['image_id']
        cat_id = target['category_id']
        key = f"{img_id}_{cat_id}"
        if key not in target_matched:
            target_matched[key] = []
        target_matched[key].append(False)

    tp_count = 0
    fp_count = 0

    for pred in predictions:
        img_id = pred['image_id']
        cat_id = pred['category_id']
        key = f"{img_id}_{cat_id}"

        # 找到该图像该类别的所有真实框
        matching_targets = [t for t in targets
                          if t['image_id'] == img_id and t['category_id'] == cat_id]

        if not matching_targets:
            fp_count += 1
            continue

        # 找到最佳匹配
        best_iou = 0.0
        best_idx = -1

        for idx, target in enumerate(matching_targets):
            if target_matched[key][idx]:
                continue

            iou = calculate_bbox_iou(pred['bbox'], target['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        # 判断TP或FP
        if best_iou >= iou_threshold and best_idx != -1:
            tp_count += 1
            target_matched[key][best_idx] = True
        else:
            fp_count += 1

    # 计算precision和recall
    total_targets = len(targets)
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
    recall = tp_count / total_targets if total_targets > 0 else 0.0

    return precision, recall


def save_predictions(predictions, save_path, config):
    """保存预测结果"""
    results = []

    for i, pred in enumerate(predictions):
        boxes = pred['boxes'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()

        # 过滤低置信度预测
        confidence_threshold = config['prediction']['confidence_threshold']
        mask = scores > confidence_threshold
        boxes = boxes[mask]
        labels = labels[mask]
        scores = scores[mask]

        # 转换为COCO格式
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box
            results.append({
                'image_id': i,
                'category_id': int(label),
                'bbox': [x1, y1, x2 - x1, y2 - y1],  # [x, y, width, height]
                'score': float(score)
            })

    # 保存为JSON文件
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"预测结果已保存到: {save_path}")


def plot_training_curves(log_file, save_path=None):
    """绘制训练曲线"""
    import pandas as pd

    # 读取日志文件
    df = pd.read_csv(log_file)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 损失曲线
    axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train Loss')
    axes[0, 0].plot(df['epoch'], df['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 学习率曲线
    axes[0, 1].plot(df['epoch'], df['learning_rate'])
    axes[0, 1].set_title('Learning Rate')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Learning Rate')
    axes[0, 1].grid(True)

    # RPN损失
    axes[1, 0].plot(df['epoch'], df['rpn_loss'], label='RPN Loss')
    axes[1, 0].set_title('RPN Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # ROI损失
    axes[1, 1].plot(df['epoch'], df['roi_loss'], label='ROI Loss')
    axes[1, 1].set_title('ROI Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线已保存到: {save_path}")
    else:
        plt.show()


def create_inference_script(model_path, config_path, output_path):
    """创建推理脚本"""
    script_content = f'''#!/usr/bin/env python3
"""
Faster R-CNN推理脚本
使用方法: python {output_path} --image path/to/image.jpg
"""

import argparse
import torch
import cv2
import numpy as np
from PIL import Image
import yaml
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import load_pretrained_model
from utils import draw_boxes, get_device

def predict_image(model, image_path, config, device):
    """预测单张图像"""
    # 加载图像
    image = cv2.imread(image_path)
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

def main():
    parser = argparse.ArgumentParser(description='Faster R-CNN推理')
    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--model', type=str, default='{model_path}', help='模型路径')
    parser.add_argument('--config', type=str, default='{config_path}', help='配置文件路径')
    parser.add_argument('--output', type=str, default='output.jpg', help='输出图像路径')

    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 获取设备
    device = get_device(config['device'])

    # 加载模型
    model = load_pretrained_model(args.model, config, device)

    # 预测
    result_image, boxes, labels, scores = predict_image(
        model, args.image, config, device
    )

    # 保存结果
    result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, result_image)

    print(f"预测结果已保存到: {{args.output}}")
    print(f"检测到 {{len(boxes)}} 个目标")

if __name__ == "__main__":
    main()
'''

    with open(output_path, 'w') as f:
        f.write(script_content)

    # 添加执行权限
    os.chmod(output_path, 0o755)
    print(f"推理脚本已创建: {output_path}")


def generate_industry_standard_plots(log_dir, save_path='faster_rcnn_training.png', config=None):
    """
    生成业界标准的Faster R-CNN训练监控图表 - 2行5列布局

    Args:
        log_dir: TensorBoard日志目录路径
        save_path: 保存图表的路径
        config: 配置文件，用于获取训练统计信息
    """
    try:
        import pandas as pd
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        import seaborn as sns
    except ImportError as e:
        print(f"❌ 缺少依赖包: {e}")
        print("请安装: pip install pandas tensorboard seaborn")
        return

    # 设置matplotlib样式
    plt.style.use('default')
    sns.set_palette("husl")

    # 加载TensorBoard日志
    try:
        ea = EventAccumulator(log_dir)
        ea.Reload()
    except Exception as e:
        print(f"❌ 无法加载TensorBoard日志: {e}")
        return

    # 创建2行5列的图表（类似YOLO风格）
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    fig.suptitle('Faster R-CNN Training Monitor', fontsize=16, fontweight='bold')

    # 设置颜色主题
    colors = {
        'train': '#1f77b4',      # 蓝色
        'val': '#ff7f0e',        # 橙色
        'smooth': '#2ca02c',     # 绿色
        'lr': '#d62728',         # 红色
        'rpn': '#9467bd',        # 紫色
        'roi': '#8c564b',        # 棕色
        'map50': '#ff7f0e',      # 橙色
        'map50_95': '#1f77b4',   # 蓝色
        'precision': '#2ca02c',  # 绿色
        'recall': '#d62728'      # 红色
    }

    # 第一行：训练指标
    # 1. 训练损失
    ax1 = axes[0, 0]
    if 'Loss/Train' in ea.Tags()['scalars']:
        train_data = [(e.step, e.value) for e in ea.Scalars('Loss/Train')]
        epochs, values = zip(*train_data)
        ax1.plot(epochs, values, color=colors['train'], linewidth=2, label='Train Loss')

        # 平滑线
        if len(values) > 5:
            smooth_values = pd.Series(values).rolling(window=5, center=True).mean()
            ax1.plot(epochs, smooth_values, color=colors['smooth'], linewidth=1,
                    linestyle='--', alpha=0.7, label='Smooth')

    ax1.set_title('Train Loss', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. RPN损失
    ax2 = axes[0, 1]
    if 'Loss/RPN' in ea.Tags()['scalars']:
        rpn_data = [(e.step, e.value) for e in ea.Scalars('Loss/RPN')]
        epochs, values = zip(*rpn_data)
        ax2.plot(epochs, values, color=colors['rpn'], linewidth=2, label='RPN Loss')

        # 平滑线
        if len(values) > 5:
            smooth_values = pd.Series(values).rolling(window=5, center=True).mean()
            ax2.plot(epochs, smooth_values, color=colors['smooth'], linewidth=1,
                    linestyle='--', alpha=0.7, label='Smooth')

    ax2.set_title('RPN Loss', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. ROI损失
    ax3 = axes[0, 2]
    if 'Loss/ROI' in ea.Tags()['scalars']:
        roi_data = [(e.step, e.value) for e in ea.Scalars('Loss/ROI')]
        epochs, values = zip(*roi_data)
        ax3.plot(epochs, values, color=colors['roi'], linewidth=2, label='ROI Loss')

        # 平滑线
        if len(values) > 5:
            smooth_values = pd.Series(values).rolling(window=5, center=True).mean()
            ax3.plot(epochs, smooth_values, color=colors['smooth'], linewidth=1,
                    linestyle='--', alpha=0.7, label='Smooth')

    ax3.set_title('ROI Loss', fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Precision
    ax4 = axes[0, 3]
    if 'Metrics/Precision' in ea.Tags()['scalars']:
        prec_data = [(e.step, e.value) for e in ea.Scalars('Metrics/Precision')]
        epochs, values = zip(*prec_data)
        ax4.plot(epochs, values, color=colors['precision'], linewidth=2, label='Precision')

        # 平滑线
        if len(values) > 5:
            smooth_values = pd.Series(values).rolling(window=5, center=True).mean()
            ax4.plot(epochs, smooth_values, color=colors['smooth'], linewidth=1,
                    linestyle='--', alpha=0.7, label='Smooth')

    ax4.set_title('Precision', fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Score')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Recall
    ax5 = axes[0, 4]
    if 'Metrics/Recall' in ea.Tags()['scalars']:
        recall_data = [(e.step, e.value) for e in ea.Scalars('Metrics/Recall')]
        epochs, values = zip(*recall_data)
        ax5.plot(epochs, values, color=colors['recall'], linewidth=2, label='Recall')

        # 平滑线
        if len(values) > 5:
            smooth_values = pd.Series(values).rolling(window=5, center=True).mean()
            ax5.plot(epochs, smooth_values, color=colors['smooth'], linewidth=1,
                    linestyle='--', alpha=0.7, label='Smooth')

    ax5.set_title('Recall', fontweight='bold')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Score')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 第二行：验证指标
    # 6. 验证损失
    ax6 = axes[1, 0]
    if 'Loss/Val' in ea.Tags()['scalars']:
        val_data = [(e.step, e.value) for e in ea.Scalars('Loss/Val')]
        epochs, values = zip(*val_data)
        ax6.plot(epochs, values, color=colors['val'], linewidth=2, label='Val Loss')

        # 平滑线
        if len(values) > 5:
            smooth_values = pd.Series(values).rolling(window=5, center=True).mean()
            ax6.plot(epochs, smooth_values, color=colors['smooth'], linewidth=1,
                    linestyle='--', alpha=0.7, label='Smooth')

    ax6.set_title('Val Loss', fontweight='bold')
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Loss')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 7. 学习率
    ax7 = axes[1, 1]
    if 'LR' in ea.Tags()['scalars']:
        lr_data = [(e.step, e.value) for e in ea.Scalars('LR')]
        epochs, values = zip(*lr_data)
        ax7.plot(epochs, values, color=colors['lr'], linewidth=2, label='Learning Rate')
        ax7.set_yscale('log')

    ax7.set_title('Learning Rate', fontweight='bold')
    ax7.set_xlabel('Epoch')
    ax7.set_ylabel('Learning Rate')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 8. mAP50
    ax8 = axes[1, 2]
    if 'Metrics/mAP50' in ea.Tags()['scalars']:
        map50_data = [(e.step, e.value) for e in ea.Scalars('Metrics/mAP50')]
        epochs, values = zip(*map50_data)
        ax8.plot(epochs, values, color=colors['map50'], linewidth=2, label='mAP50')

        # 平滑线
        if len(values) > 5:
            smooth_values = pd.Series(values).rolling(window=5, center=True).mean()
            ax8.plot(epochs, smooth_values, color=colors['smooth'], linewidth=1,
                    linestyle='--', alpha=0.7, label='Smooth')

    ax8.set_title('mAP50', fontweight='bold')
    ax8.set_xlabel('Epoch')
    ax8.set_ylabel('mAP')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # 9. mAP50-95
    ax9 = axes[1, 3]
    if 'Metrics/mAP50-95' in ea.Tags()['scalars']:
        map50_95_data = [(e.step, e.value) for e in ea.Scalars('Metrics/mAP50-95')]
        epochs, values = zip(*map50_95_data)
        ax9.plot(epochs, values, color=colors['map50_95'], linewidth=2, label='mAP50-95')

        # 平滑线
        if len(values) > 5:
            smooth_values = pd.Series(values).rolling(window=5, center=True).mean()
            ax9.plot(epochs, smooth_values, color=colors['smooth'], linewidth=1,
                    linestyle='--', alpha=0.7, label='Smooth')

    ax9.set_title('mAP50-95', fontweight='bold')
    ax9.set_xlabel('Epoch')
    ax9.set_ylabel('mAP')
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    # 10. 训练统计
    ax10 = axes[1, 4]

    # 计算训练统计信息
    stats_text = "Training Summary\n\n"

    # 获取训练轮数
    if 'Loss/Train' in ea.Tags()['scalars']:
        total_epochs = len(ea.Scalars('Loss/Train'))
        stats_text += f"• Total Epochs: {total_epochs}\n"

    # 获取最佳mAP
    best_map = 0
    if 'Metrics/mAP50' in ea.Tags()['scalars']:
        map_values = [e.value for e in ea.Scalars('Metrics/mAP50')]
        best_map = max(map_values) if map_values else 0
        stats_text += f"• Best mAP50: {best_map:.3f}\n"

    # 获取最终损失
    final_loss = 0
    if 'Loss/Val' in ea.Tags()['scalars']:
        val_losses = [e.value for e in ea.Scalars('Loss/Val')]
        final_loss = val_losses[-1] if val_losses else 0
        stats_text += f"• Final Val Loss: {final_loss:.4f}\n"

    # 如果有配置文件，添加更多信息
    if config:
        if 'training' in config:
            stats_text += f"• Learning Rate: {config['training']['learning_rate']:.6f}\n"
            stats_text += f"• Optimizer: {config['training'].get('optimizer', 'SGD')}\n"
        if 'model' in config:
            stats_text += f"• Backbone: {config['model'].get('backbone', 'ResNet50')}\n"

    ax10.text(0.5, 0.5, stats_text,
             ha='center', va='center', transform=ax10.transAxes, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    ax10.set_title('Training Summary', fontweight='bold')
    ax10.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.93, hspace=0.3, wspace=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    print(f"✅ 2行5列训练图表已生成: {save_path}")


def generate_training_report(log_dir, config, save_dir='reports'):
    """
    生成完整的训练报告

    Args:
        log_dir: TensorBoard日志目录
        config: 配置文件
        save_dir: 报告保存目录
    """
    import os
    from datetime import datetime

    # 创建报告目录
    os.makedirs(save_dir, exist_ok=True)

    # 生成时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 生成图表
    plot_path = os.path.join(save_dir, f'training_plots_{timestamp}.png')
    generate_industry_standard_plots(log_dir, plot_path, config)

    # 生成文本报告
    report_path = os.path.join(save_dir, f'training_report_{timestamp}.txt')

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(log_dir)
        ea.Reload()

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("Faster R-CNN Training Report\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"日志目录: {log_dir}\n\n")

            # 训练配置
            f.write("训练配置:\n")
            f.write("-" * 20 + "\n")
            if 'training' in config:
                f.write(f"学习率: {config['training']['learning_rate']}\n")
                f.write(f"优化器: {config['training'].get('optimizer', 'SGD')}\n")
                f.write(f"训练轮数: {config['training']['epochs']}\n")
                f.write(f"批次大小: {config['training']['batch_size']}\n\n")

            # 模型配置
            f.write("模型配置:\n")
            f.write("-" * 20 + "\n")
            if 'model' in config:
                f.write(f"骨干网络: {config['model'].get('backbone', 'ResNet50')}\n")
                f.write(f"类别数量: {config['model']['num_classes']}\n\n")

            # 训练结果
            f.write("训练结果:\n")
            f.write("-" * 20 + "\n")

            if 'Loss/Train' in ea.Tags()['scalars']:
                train_losses = [e.value for e in ea.Scalars('Loss/Train')]
                f.write(f"训练轮数: {len(train_losses)}\n")
                f.write(f"初始训练损失: {train_losses[0]:.4f}\n")
                f.write(f"最终训练损失: {train_losses[-1]:.4f}\n")
                f.write(f"损失下降: {train_losses[0] - train_losses[-1]:.4f}\n\n")

            if 'Loss/Val' in ea.Tags()['scalars']:
                val_losses = [e.value for e in ea.Scalars('Loss/Val')]
                best_val_loss = min(val_losses)
                best_epoch = val_losses.index(best_val_loss) + 1
                f.write(f"最佳验证损失: {best_val_loss:.4f} (Epoch {best_epoch})\n")
                f.write(f"最终验证损失: {val_losses[-1]:.4f}\n\n")

            if 'Metrics/mAP50' in ea.Tags()['scalars']:
                map_values = [e.value for e in ea.Scalars('Metrics/mAP50')]
                best_map = max(map_values)
                best_map_epoch = map_values.index(best_map) + 1
                f.write(f"最佳mAP@0.5: {best_map:.3f} (Epoch {best_map_epoch})\n")
                f.write(f"最终mAP@0.5: {map_values[-1]:.3f}\n\n")

            f.write("=" * 60 + "\n")
            f.write("报告生成完成\n")
            f.write("=" * 60 + "\n")

        print(f"✅ 训练报告已生成: {report_path}")

    except Exception as e:
        print(f"❌ 生成训练报告时出错: {e}")


def train_one_epoch(model, train_loader, optimizer, device, epoch, config):
    """训练一个epoch"""
    model.train()

    # 损失记录器
    loss_meter = AverageMeter()
    rpn_loss_meter = AverageMeter()
    roi_loss_meter = AverageMeter()

    # 进度条
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    try:
        for batch_idx, (images, targets) in enumerate(pbar):
            try:
                # 移动数据到设备
                images = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # 检查数据有效性
                if not images or not targets:
                    print(f"警告: 批次 {batch_idx} 数据为空，跳过")
                    continue

                # 前向传播
                loss_dict = model(images, targets)

                # 检查损失字典
                if not loss_dict:
                    print(f"警告: 批次 {batch_idx} 损失字典为空，跳过")
                    continue

                # 计算总损失
                losses = sum(loss for loss in loss_dict.values())

                # 检查损失是否有效
                if torch.isnan(losses) or torch.isinf(losses):
                    print(f"警告: 批次 {batch_idx} 出现无效损失值: {losses.item()}")
                    continue

                # 反向传播
                optimizer.zero_grad()
                losses.backward()

                # 梯度裁剪
                if config["training"].get("gradient_clip_val", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config["training"]["gradient_clip_val"]
                    )

                optimizer.step()

                # 更新损失记录
                loss_meter.update(losses.item())

                # 安全地获取RPN和ROI损失
                rpn_loss = 0.0
                roi_loss = 0.0

                if "loss_objectness" in loss_dict and "loss_rpn_box_reg" in loss_dict:
                    rpn_loss = loss_dict["loss_objectness"].item() + loss_dict["loss_rpn_box_reg"].item()

                if "loss_classifier" in loss_dict and "loss_box_reg" in loss_dict:
                    roi_loss = loss_dict["loss_classifier"].item() + loss_dict["loss_box_reg"].item()

                rpn_loss_meter.update(rpn_loss)
                roi_loss_meter.update(roi_loss)

                # 更新进度条
                pbar.set_postfix({
                    'Loss': f'{loss_meter.avg:.4f}',
                    'RPN': f'{rpn_loss_meter.avg:.4f}',
                    'ROI': f'{roi_loss_meter.avg:.4f}'
                })

            except Exception as e:
                print(f"处理批次 {batch_idx} 时出错: {e}")
                continue

        return {
            'loss': loss_meter.avg,
            'rpn_loss': rpn_loss_meter.avg,
            'roi_loss': roi_loss_meter.avg
        }

    except Exception as e:
        print(f"训练第{epoch+1}轮时出错: {e}")
        print(f"错误类型: {type(e).__name__}")
        import traceback
        traceback.print_exc()

        # 返回默认值避免程序崩溃
        return {
            'loss': float('inf'),
            'rpn_loss': float('inf'),
            'roi_loss': float('inf')
        }





if __name__ == "__main__":
    # 测试工具函数
    print("工具函数测试完成")

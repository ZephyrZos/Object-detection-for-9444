
import os
import json
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import random
from typing import Dict, List, Tuple, Optional


class VisDroneDataset(Dataset):
    """VisDrone数据集类，支持COCO格式标注"""

    def __init__(self,
                 root_dir: str,
                 annotation_file: str,
                 transform=None,
                 augmentation_config: Optional[Dict] = None):
        """
        Args:
            root_dir: 数据集根目录
            annotation_file: COCO格式标注文件路径
            transform: 图像变换
            augmentation_config: 数据增强配置
        """
        self.root_dir = root_dir
        self.transform = transform
        self.augmentation_config = augmentation_config

        # 加载COCO格式标注
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)

        # 创建图像ID到标注的映射
        self.img_to_anns = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

        # 创建类别ID映射
        self.cat_id_to_label = {}
        for cat in self.coco_data['categories']:
            self.cat_id_to_label[cat['id']] = cat['name']

        # 图像列表
        self.images = self.coco_data['images']

        # 设置数据增强
        self.setup_augmentation()

    def setup_augmentation(self):
        """设置数据增强"""
        if self.augmentation_config and self.augmentation_config.get('enabled', False):
            # 获取配置参数
            scale_range = self.augmentation_config.get('scale', [0.8, 1.2])
            translate_range = self.augmentation_config.get('translate', [0.1, 0.1])

            # 计算scale_limit (RandomScale需要的是偏差值，不是范围)
            scale_limit = (scale_range[1] - 1.0) if isinstance(scale_range, list) else 0.2

            self.aug_transform = A.Compose([
                A.HorizontalFlip(p=self.augmentation_config.get('horizontal_flip', 0.5)),
                A.VerticalFlip(p=self.augmentation_config.get('vertical_flip', 0.0)),
                A.Rotate(
                    limit=self.augmentation_config.get('rotation', 10),
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT
                    # value参数在新版本中已移除，使用默认填充
                ),
                A.ColorJitter(
                    brightness=self.augmentation_config.get('brightness', 0.2),
                    contrast=self.augmentation_config.get('contrast', 0.2),
                    saturation=self.augmentation_config.get('saturation', 0.2),
                    hue=self.augmentation_config.get('hue', 0.1),
                    p=0.5
                ),
                # 修复RandomScale - 使用正确的参数格式
                A.RandomScale(
                    scale_limit=scale_limit,  # 单个值，表示缩放的最大偏差
                    p=0.5
                ),
                # 修复ShiftScaleRotate - 使用正确的参数
                A.Affine(
                    translate_percent=translate_range[0] if isinstance(translate_range, list) else 0.1,
                    scale=1.0,  # 不进行缩放，已经用RandomScale了
                    rotate=0,   # 不进行旋转，已经用Rotate了
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT  # 修复：使用border_mode而不是mode
                    # fill参数替代了旧的value参数
                ),
                # 添加更多增强选项
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.1,
                    p=0.3
                ),
                A.GaussNoise(
                    std_range=(0.01, 0.05),  # 修复：使用std_range参数而不是noise_scale_factor
                    noise_scale_factor=1.0,  # 噪声缩放因子，单个float值
                    p=0.2
                ),
                A.Blur(
                    blur_limit=3,
                    p=0.1
                )
            ], bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['labels'],
                min_visibility=0.3,  # 确保增强后的框至少有30%可见
                min_area=100  # 最小框面积
            ))
        else:
            self.aug_transform = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 获取图像信息
        img_info = self.images[idx]
        img_path = os.path.join(self.root_dir, img_info['file_name'])

        # 加载图像
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 获取该图像的标注
        img_id = img_info['id']
        annotations = self.img_to_anns.get(img_id, [])

        # 准备边界框和标签
        boxes = []
        labels = []

        for ann in annotations:
            bbox = ann['bbox']  # [x, y, width, height]
            # 转换为 [x1, y1, x2, y2] 格式
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
            boxes.append([x1, y1, x2, y2])
            labels.append(ann['category_id'])

        # 转换为numpy数组
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        # 应用数据增强
        if self.aug_transform and len(boxes) > 0:
            augmented = self.aug_transform(image=image, bboxes=boxes, labels=labels)
            image = augmented['image']
            boxes = np.array(augmented['bboxes'], dtype=np.float32)
            labels = np.array(augmented['labels'], dtype=np.int64)

        # 应用变换
        if self.transform:
            image = self.transform(image)

        # 转换为tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # 创建目标字典
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id]),
            'area': torch.as_tensor([(box[2] - box[0]) * (box[3] - box[1]) for box in boxes], dtype=torch.float32),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }

        return image, target


def get_transform(train: bool, config: Dict):
    """获取图像变换"""
    if train:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config['model']['transform']['image_mean'],
                std=config['model']['transform']['image_std']
            )
        ])
    else:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config['model']['transform']['image_mean'],
                std=config['model']['transform']['image_std']
            )
        ])

    return transform


def collate_fn(batch):
    """自定义collate函数，处理不同数量的目标"""
    return tuple(zip(*batch))


def create_data_loaders(config: Dict):
    """创建数据加载器"""
    # 获取变换
    train_transform = get_transform(train=True, config=config)
    val_transform = get_transform(train=False, config=config)

    # 创建数据集
    train_dataset = VisDroneDataset(
        root_dir=os.path.join(config['convert_output_path'], config['train_images']),
        annotation_file=os.path.join(config['convert_output_path'], config['train_annotations']),
        transform=train_transform,
        augmentation_config=config.get('augmentation', {})
    )

    val_dataset = VisDroneDataset(
        root_dir=os.path.join(config['convert_output_path'], config['val_images']),
        annotation_file=os.path.join(config['convert_output_path'], config['val_annotations']),
        transform=val_transform
    )

    # 检查是否启用测试模式
    test_mode = config.get('test_mode', {})
    if test_mode.get('enabled', False):
        print("🧪 测试模式已启用")

        # 获取测试模式参数
        max_train_samples = test_mode.get('max_train_samples', 100)
        max_val_samples = test_mode.get('max_val_samples', 50)
        test_batch_size = test_mode.get('batch_size', 2)
        test_num_workers = test_mode.get('num_workers', 2)

        # 创建随机索引（确保可重现）
        random.seed(config.get('seed', 42))

        # 限制训练集大小
        if len(train_dataset) > max_train_samples:
            train_indices = random.sample(range(len(train_dataset)), max_train_samples)
            train_dataset = Subset(train_dataset, train_indices)
            print(f"📊 训练集已限制为 {max_train_samples} 个样本")

        # 限制验证集大小
        if len(val_dataset) > max_val_samples:
            val_indices = random.sample(range(len(val_dataset)), max_val_samples)
            val_dataset = Subset(val_dataset, val_indices)
            print(f"📊 验证集已限制为 {max_val_samples} 个样本")

        # 使用测试模式的批次大小和工作进程数
        train_batch_size = test_batch_size
        val_batch_size = test_batch_size
        num_workers = test_num_workers

        print(f"📊 测试模式配置:")
        print(f"  - 训练样本数: {len(train_dataset)}")
        print(f"  - 验证样本数: {len(val_dataset)}")
        print(f"  - 批次大小: {test_batch_size}")
        print(f"  - 工作进程数: {test_num_workers}")

    else:
        # 正常模式
        train_batch_size = config['training']['batch_size']
        val_batch_size = config['validation']['batch_size']
        num_workers = config['training']['num_workers']

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader

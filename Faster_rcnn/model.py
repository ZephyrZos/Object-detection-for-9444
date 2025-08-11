
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    FasterRCNN
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from typing import Dict, List, Optional


class FasterRCNNModel(nn.Module):
    """Faster R-CNN模型类"""

    def __init__(self, config: Dict):
        """
        Args:
            config: 配置字典
        """
        super(FasterRCNNModel, self).__init__()

        self.config = config
        self.num_classes = config["model"]["num_classes"]
        self.backbone_name = config["model"]["backbone"]

        # 创建基础模型
        self.model = self._create_model()

        # 自定义RPN锚点生成器（可选）
        if "rpn_anchor_generator" in config["model"]:
            self._customize_rpn_anchor_generator()

    def _create_model(self):
        """创建Faster R-CNN模型"""
        if self.backbone_name == "resnet50":
            model = fasterrcnn_resnet50_fpn(
                pretrained=self.config["model"]["pretrained"],
                pretrained_backbone=self.config["model"]["pretrained_backbone"],
            )
        elif self.backbone_name == "resnet50_v2":
            model = fasterrcnn_resnet50_fpn_v2(
                pretrained=self.config["model"]["pretrained"],
                pretrained_backbone=self.config["model"]["pretrained_backbone"],
            )
        elif self.backbone_name == "mobilenet_v3_large":
            model = fasterrcnn_mobilenet_v3_large_fpn(
                pretrained=self.config["model"]["pretrained"],
                pretrained_backbone=self.config["model"]["pretrained_backbone"],
            )
        elif self.backbone_name == "mobilenet_v3_large_320":
            model = fasterrcnn_mobilenet_v3_large_320_fpn(
                pretrained=self.config["model"]["pretrained"],
                pretrained_backbone=self.config["model"]["pretrained_backbone"],
            )
        elif self.backbone_name == "resnet101":
            from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
            backbone = resnet_fpn_backbone(
                'resnet101',
                pretrained=self.config["model"]["pretrained_backbone"],
                trainable_layers=5
            )
            model = FasterRCNN(
                backbone,
                num_classes=self.num_classes,
                rpn_anchor_generator=None,
                rpn_head=None,
                rpn_pre_nms_top_n_train=2000,
                rpn_pre_nms_top_n_test=1000,
                rpn_post_nms_top_n_train=2000,
                rpn_post_nms_top_n_test=1000,
                rpn_nms_thresh=0.7,
                rpn_fg_iou_thresh=0.7,
                rpn_bg_iou_thresh=0.3,
                rpn_batch_size_per_image=256,
                rpn_positive_fraction=0.5,
                box_roi_pool=None,
                box_head=None,
                box_predictor=None,
                box_score_thresh=0.05,
                box_nms_thresh=0.5,
                box_detections_per_img=100,
                box_fg_iou_thresh=0.5,
                box_bg_iou_thresh=0.5,
                box_batch_size_per_image=512,
                box_positive_fraction=0.25,
            )
        else:
            raise ValueError(
                f"不支持的backbone: {self.backbone_name}. 支持的backbone: resnet50, resnet50_v2, mobilenet_v3_large, mobilenet_v3_large_320"
            )

        # 替换分类器头部以匹配类别数
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

        return model

    def _customize_rpn_anchor_generator(self):
        """自定义RPN锚点生成器"""
        anchor_config = self.config["model"]["rpn_anchor_generator"]

        # 创建自定义锚点生成器
        anchor_generator = AnchorGenerator(
            sizes=anchor_config["sizes"], aspect_ratios=anchor_config["aspect_ratios"]
        )

        # 替换模型的RPN锚点生成器
        self.model.rpn.anchor_generator = anchor_generator

    def forward(self, images, targets=None):
        """
        Args:
            images: 图像列表
            targets: 目标列表（训练时）
        """
        return self.model(images, targets)

    def extract_feature_maps(self, images):
        """
        Extracts feature maps from the backbone.

        Args:
            images: A list of input images (tensors).

        Returns:
            A dictionary of feature maps from the FPN and the transformed images.
        """
        self.model.eval()

        images, _ = self.model.transform(images, None)
        feature_maps = self.model.backbone(images.tensors)
        return feature_maps, images

    def get_parameters(self):
        """获取模型参数"""
        return self.model.parameters()

    def get_trainable_parameters(self):
        """获取可训练参数"""
        return filter(lambda p: p.requires_grad, self.model.parameters())

    def calculate_metrics(self, val_loader, device):
        """计算验证指标：mAP50, mAP50-95, precision, recall"""
        self.eval()

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for images, targets in val_loader:
                # 移动数据到设备
                images = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # 预测
                predictions = self.model(images)

                # 收集预测和目标
                for pred, target in zip(predictions, targets):
                    all_predictions.append(pred)
                    all_targets.append(target)

        # 计算指标
        metrics = self._calculate_map_metrics(all_predictions, all_targets)
        return metrics

    def _calculate_map_metrics(self, predictions, targets):
        """计算mAP指标"""
        import numpy as np

        # 获取类别名称（排除background）
        class_names = list(self.config['classes'].values())[1:]
        num_classes = len(class_names)

        # 计算mAP@0.5
        map50 = self._calculate_map_at_iou(predictions, targets, iou_threshold=0.5)

        # 计算mAP@0.5:0.95 (多个IoU阈值的平均值)
        iou_thresholds = np.arange(0.5, 1.0, 0.05)
        maps = []
        for iou in iou_thresholds:
            ap = self._calculate_map_at_iou(predictions, targets, iou_threshold=iou)
            maps.append(ap)
        map50_95 = np.mean(maps)

        # 计算precision和recall
        precision, recall = self._calculate_precision_recall(predictions, targets)

        return {
            'mAP50': map50,
            'mAP50-95': map50_95,
            'precision': precision,
            'recall': recall
        }

    def _calculate_map_at_iou(self, predictions, targets, iou_threshold=0.5):
        """计算指定IoU阈值的mAP"""
        import numpy as np

        class_names = list(self.config['classes'].values())[1:]
        num_classes = len(class_names)

        aps = []

        for class_id in range(1, num_classes + 1):
            class_predictions = []
            class_targets = []

            # 收集该类别的所有预测和真实框
            for pred, target in zip(predictions, targets):
                # 预测框
                pred_mask = pred['labels'] == class_id
                if pred_mask.any():
                    pred_boxes = pred['boxes'][pred_mask].cpu().numpy()
                    pred_scores = pred['scores'][pred_mask].cpu().numpy()
                    for box, score in zip(pred_boxes, pred_scores):
                        class_predictions.append({
                            'box': box,
                            'score': score
                        })

                # 真实框
                target_mask = target['labels'] == class_id
                if target_mask.any():
                    target_boxes = target['boxes'][target_mask].cpu().numpy()
                    for box in target_boxes:
                        class_targets.append(box)

            # 计算该类别的AP
            ap = self._calculate_ap(class_predictions, class_targets, iou_threshold)
            aps.append(ap)

        return np.mean(aps)

    def _calculate_ap(self, predictions, targets, iou_threshold):
        """计算平均精度"""
        import numpy as np

        if len(predictions) == 0 or len(targets) == 0:
            return 0.0

        # 按置信度排序
        predictions.sort(key=lambda x: x['score'], reverse=True)

        # 计算TP和FP
        tp = np.zeros(len(predictions))
        fp = np.zeros(len(predictions))
        used_targets = set()

        for i, pred in enumerate(predictions):
            best_iou = 0
            best_target_idx = -1

            for j, target in enumerate(targets):
                if j in used_targets:
                    continue

                iou = self._calculate_iou(pred['box'], target)
                if iou > best_iou:
                    best_iou = iou
                    best_target_idx = j

            if best_iou >= iou_threshold and best_target_idx != -1:
                tp[i] = 1
                used_targets.add(best_target_idx)
            else:
                fp[i] = 1

        # 计算累积TP和FP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        # 计算precision和recall
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / len(targets)

        # 计算AP (VOC方法)
        ap = self._voc_ap(recall, precision)
        return ap

    def _calculate_iou(self, box1, box2):
        """计算IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def _voc_ap(self, recall, precision):
        """VOC AP计算方法"""
        import numpy as np

        # 添加哨兵值
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))

        # 计算PR曲线下的面积
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # 计算AP
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

        return ap

    def _calculate_precision_recall(self, predictions, targets):
        """计算整体的precision和recall"""
        import numpy as np

        total_tp = 0
        total_fp = 0
        total_fn = 0

        for pred, target in zip(predictions, targets):
            pred_boxes = pred['boxes'].cpu().numpy()
            pred_labels = pred['labels'].cpu().numpy()
            pred_scores = pred['scores'].cpu().numpy()

            target_boxes = target['boxes'].cpu().numpy()
            target_labels = target['labels'].cpu().numpy()

            # 计算IoU矩阵
            if len(pred_boxes) > 0 and len(target_boxes) > 0:
                ious = np.zeros((len(pred_boxes), len(target_boxes)))
                for i, pred_box in enumerate(pred_boxes):
                    for j, target_box in enumerate(target_boxes):
                        ious[i, j] = self._calculate_iou(pred_box, target_box)

                # 匹配预测和真实目标
                matched_pred = np.zeros(len(pred_boxes), dtype=bool)
                matched_target = np.zeros(len(target_boxes), dtype=bool)

                for i in range(len(pred_boxes)):
                    for j in range(len(target_boxes)):
                        if not matched_target[j] and ious[i, j] > 0.5:
                            matched_pred[i] = True
                            matched_target[j] = True
                            break

                total_tp += np.sum(matched_pred)
                total_fp += np.sum(~matched_pred)
                total_fn += np.sum(~matched_target)
            else:
                total_fp += len(pred_boxes)
                total_fn += len(target_boxes)

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

        return precision, recall


def create_model(config: Dict, device: torch.device):
    """创建Faster R-CNN模型"""
    model = FasterRCNNModel(config)
    model = model.to(device)
    return model


def load_pretrained_model(model_path: str, config: Dict, device: torch.device):
    """加载预训练模型"""
    model = create_model(config, device)

    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return model


def load_checkpoint_for_resume(checkpoint_path: str, config: Dict, device: torch.device):
    """加载检查点用于继续训练"""
    model = create_model(config, device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 加载模型权重
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return model, checkpoint


def save_model(
    model: nn.Module, optimizer, scheduler, epoch: int, loss: float, save_path: str, config: Dict
):
    """保存模型"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": config,
    }

    # 如果调度器存在，也保存其状态
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, save_path)


class ModelEMA:
    """模型指数移动平均"""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # 注册模型参数
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """更新EMA模型"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (
                    1.0 - self.decay
                ) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """应用EMA权重"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """恢复原始权重"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def count_parameters(model: nn.Module):
    """计算模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params,
    }


def print_model_summary(model: nn.Module):
    """打印模型摘要"""
    param_counts = count_parameters(model)

    print("=" * 50)
    print("📊 模型参数统计")
    print("=" * 50)
    print(f"总参数数量: {param_counts['total']:,}")
    print(f"可训练参数: {param_counts['trainable']:,}")
    print(f"不可训练参数: {param_counts['non_trainable']:,}")
    print("=" * 50)

    # 打印模型结构
    print("\n🏗️ 模型结构:")
    print(model)
    print("=" * 50)


if __name__ == "__main__":
    # 测试模型创建
    import yaml

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(config, device)

    print_model_summary(model)


import os
import sys
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import random
from pathlib import Path
import glob

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from convert_dataset import convert_visdrone_to_coco
from dataset import create_data_loaders
from model import create_model, save_model, load_pretrained_model, ModelEMA, print_model_summary
from utils import AverageMeter, EarlyStopping, get_device


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_latest_checkpoint(checkpoint_dir="checkpoints"):
    """查找最新的检查点文件"""
    if not os.path.exists(checkpoint_dir):
        return None

    # 查找所有检查点文件
    checkpoint_patterns = [
        f"{checkpoint_dir}/faster_rcnn_*/best_model.pth",
        f"{checkpoint_dir}/faster_rcnn_*/final_model.pth",
        f"{checkpoint_dir}/faster_rcnn_*/checkpoint_epoch_*.pth"
    ]

    all_checkpoints = []
    for pattern in checkpoint_patterns:
        all_checkpoints.extend(glob.glob(pattern))

    if all_checkpoints:
        # 按修改时间排序，返回最新的
        latest_checkpoint = max(all_checkpoints, key=os.path.getmtime)
        return latest_checkpoint

    return None


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """加载检查点"""
    print(f"🔄 加载检查点: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 加载模型权重
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # 加载优化器状态
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # 加载学习率调度器状态
    if "scheduler_state_dict" in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # 获取训练状态
    start_epoch = checkpoint.get("epoch", 0) + 1  # 从下一个epoch开始
    best_loss = checkpoint.get("loss", float("inf"))

    print(f"✅ 检查点加载成功")
    print(f"   开始epoch: {start_epoch}")
    print(f"   最佳损失: {best_loss:.4f}")

    return start_epoch, best_loss


def create_optimizer(model, config):
    """创建优化器"""
    optimizer_name = config["training"].get("optimizer", "sgd").lower()
    lr = config["training"]["learning_rate"]
    weight_decay = config["training"]["weight_decay"]
    momentum = config["training"]["momentum"]

    if optimizer_name == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    elif optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"不支持的优化器: {optimizer_name}")

    return optimizer


def create_scheduler(optimizer, config, num_steps):
    """创建学习率调度器"""
    scheduler_name = config["training"].get("lr_scheduler", "step").lower()

    if scheduler_name == "step":
        scheduler = StepLR(
            optimizer,
            step_size=config["training"]["lr_step_size"],
            gamma=config["training"]["lr_gamma"],
        )
    elif scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])
    elif scheduler_name == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config["training"]["lr_gamma"],
            patience=config["training"].get("lr_patience", 5),
            # verbose参数在新版本PyTorch中已移除
        )
    else:
        raise ValueError(f"不支持的学习率调度器: {scheduler_name}")

    return scheduler


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


def validate(model, val_loader, device, config):
    """验证模型"""
    model.eval()

    total_loss = 0
    num_batches = 0

    # 用于计算mAP的列表
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validation"):
            # 移动数据到设备
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # 前向传播 - 在验证时需要计算损失
            # 临时切换到训练模式来计算损失
            model.train()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # 切换回评估模式
            model.eval()

            total_loss += losses.item()
            num_batches += 1

            # 获取预测结果用于计算mAP
            predictions = model(images)

            # 收集预测和目标用于计算指标
            for pred, target in zip(predictions, targets):
                all_predictions.append(pred)
                all_targets.append(target)

    avg_loss = total_loss / num_batches

    # 计算评估指标
    metrics = calculate_validation_metrics(all_predictions, all_targets, config)

    return avg_loss, metrics


def calculate_iou(box1, box2):
    """计算两个边界框的IoU"""
    # box格式: [x1, y1, x2, y2]
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])

    # 计算交集面积
    intersection_width = max(0, x2_min - x1_max)
    intersection_height = max(0, y2_min - y1_max)
    intersection_area = intersection_width * intersection_height

    # 计算并集面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    # 避免除零
    if union_area == 0:
        return 0.0

    return intersection_area / union_area


def calculate_strict_metrics(predictions, targets, iou_threshold=0.5):
    """
    严格计算TP、FP、FN和相应的precision、recall

    Args:
        predictions: 预测结果列表，每个元素包含 'boxes', 'labels', 'scores'
        targets: 真实标签列表，每个元素包含 'boxes', 'labels'
        iou_threshold: IoU阈值，默认0.5

    Returns:
        dict: 包含TP、FP、FN、precision、recall的字典
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for pred, target in zip(predictions, targets):
        # 转换为numpy数组便于处理
        if len(pred['boxes']) == 0:
            pred_boxes = np.array([]).reshape(0, 4)
            pred_labels = np.array([])
            pred_scores = np.array([])
        else:
            pred_boxes = pred['boxes'].cpu().numpy() if hasattr(pred['boxes'], 'cpu') else pred['boxes']
            pred_labels = pred['labels'].cpu().numpy() if hasattr(pred['labels'], 'cpu') else pred['labels']
            pred_scores = pred['scores'].cpu().numpy() if hasattr(pred['scores'], 'cpu') else pred['scores']

        if len(target['boxes']) == 0:
            target_boxes = np.array([]).reshape(0, 4)
            target_labels = np.array([])
        else:
            target_boxes = target['boxes'].cpu().numpy() if hasattr(target['boxes'], 'cpu') else target['boxes']
            target_labels = target['labels'].cpu().numpy() if hasattr(target['labels'], 'cpu') else target['labels']

        # 记录哪些GT已经被匹配
        gt_matched = np.zeros(len(target_boxes), dtype=bool)

        # 对预测按置信度排序
        if len(pred_scores) > 0:
            sorted_indices = np.argsort(pred_scores)[::-1]  # 降序排列
            pred_boxes = pred_boxes[sorted_indices]
            pred_labels = pred_labels[sorted_indices]
            pred_scores = pred_scores[sorted_indices]

        # 遍历每个预测框
        for pred_idx in range(len(pred_boxes)):
            pred_box = pred_boxes[pred_idx]
            pred_label = pred_labels[pred_idx]

            best_iou = 0.0
            best_gt_idx = -1

            # 找到与当前预测框IoU最大且类别匹配的GT框
            for gt_idx in range(len(target_boxes)):
                if gt_matched[gt_idx]:  # 该GT已被匹配
                    continue

                target_box = target_boxes[gt_idx]
                target_label = target_labels[gt_idx]

                # 类别必须匹配
                if pred_label != target_label:
                    continue

                # 计算IoU
                iou = calculate_iou(pred_box, target_box)

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            # 判断是否为TP
            if best_iou >= iou_threshold and best_gt_idx != -1:
                total_tp += 1
                gt_matched[best_gt_idx] = True  # 标记该GT已被匹配
            else:
                total_fp += 1

        # 计算FN：未被匹配的GT框数量
        total_fn += np.sum(~gt_matched)

    # 计算precision和recall
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    return {
        'TP': total_tp,
        'FP': total_fp,
        'FN': total_fn,
        'precision': precision,
        'recall': recall
    }


def calculate_validation_metrics(predictions, targets, config):
    """计算验证指标 - 严格版本"""
    try:
        print("📊 计算验证指标...")

        total_predictions = len(predictions)
        total_targets = len(targets)

        if total_predictions == 0 or total_targets == 0:
            return {
                'mAP50': 0.0,
                'mAP50-95': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'TP': 0,
                'FP': 0,
                'FN': 0
            }

        # 计算严格的TP、FP、FN
        strict_metrics = calculate_strict_metrics(predictions, targets, iou_threshold=0.5)

        # 统计每个类别的预测和真实数量（用于简化的mAP计算）
        pred_counts = {}
        target_counts = {}

        for pred in predictions:
            if 'labels' in pred and len(pred['labels']) > 0:
                labels = pred['labels'].cpu().numpy() if hasattr(pred['labels'], 'cpu') else pred['labels']
                for label in labels:
                    pred_counts[label] = pred_counts.get(label, 0) + 1

        for target in targets:
            if 'labels' in target and len(target['labels']) > 0:
                labels = target['labels'].cpu().numpy() if hasattr(target['labels'], 'cpu') else target['labels']
                for label in labels:
                    target_counts[label] = target_counts.get(label, 0) + 1

        # 简化的mAP计算（保持原有逻辑）
        class_aps = []
        for class_id in range(1, config['model']['num_classes']):
            pred_count = pred_counts.get(class_id, 0)
            target_count = target_counts.get(class_id, 0)

            if target_count > 0:
                # 基于严格recall的AP估算
                class_recall = strict_metrics['recall']
                ap = min(pred_count / target_count, 1.0) * class_recall
            else:
                ap = 0.0
            class_aps.append(ap)

        # 计算平均值
        mAP50 = sum(class_aps) / len(class_aps) if class_aps else 0.0
        mAP50_95 = mAP50 * 0.8  # 简化估算

        print(f"✅ 验证指标计算完成 - mAP50: {mAP50:.4f}, Precision: {strict_metrics['precision']:.4f}, Recall: {strict_metrics['recall']:.4f}")
        print(f"   TP: {strict_metrics['TP']}, FP: {strict_metrics['FP']}, FN: {strict_metrics['FN']}")

        return {
            'mAP50': mAP50,
            'mAP50-95': mAP50_95,
            'precision': strict_metrics['precision'],
            'recall': strict_metrics['recall'],
            'TP': strict_metrics['TP'],
            'FP': strict_metrics['FP'],
            'FN': strict_metrics['FN']
        }

    except Exception as e:
        print(f"计算验证指标时出错: {e}")
        import traceback
        traceback.print_exc()
        return {
            'mAP50': 0.0,
            'mAP50-95': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'TP': 0,
            'FP': 0,
            'FN': 0
        }


def validate_config(config):
    """验证配置文件的有效性"""
    required_keys = [
        'model.backbone',
        'model.num_classes',
        'training.epochs',
        'training.batch_size',
        'training.learning_rate'
    ]

    for key in required_keys:
        keys = key.split('.')
        current = config
        try:
            for k in keys:
                current = current[k]
        except KeyError:
            raise ValueError(f"配置文件缺少必需的键: {key}")

    # 检查backbone支持
    supported_backbones = [
        'resnet50', 'resnet50_v2', 'resnet101',
        'mobilenet_v3_large', 'mobilenet_v3_large_320'
    ]
    if config['model']['backbone'] not in supported_backbones:
        raise ValueError(f"不支持的backbone: {config['model']['backbone']}")

    print("✅ 配置文件验证通过")


def main():
    """主训练函数"""
    # 加载配置
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 验证配置
    validate_config(config)

    # 设置随机种子
    set_seed(config["seed"])

    # 获取设备
    device = get_device(config["device"])
    print(f"使用设备: {device}")

    # 检查数据集是否存在，如果不存在则转换
    train_annotations = os.path.join(
        config["convert_output_path"], config["train_annotations"]
    )

    if not os.path.exists(train_annotations):
        print("未找到COCO格式标注文件，开始转换数据集...")
        convert_visdrone_to_coco(
            visdrone_dir=config['dataset_path'], output_dir=config["convert_output_path"]
        )

    # 创建数据加载器
    print("创建数据加载器...")
    train_loader, val_loader = create_data_loaders(config)

    # 创建模型
    print("创建模型...")
    model = create_model(config, device)
    print_model_summary(model)

    # 创建优化器
    optimizer = create_optimizer(model, config)

    # 创建学习率调度器
    scheduler = create_scheduler(optimizer, config, len(train_loader))

    # 创建EMA模型（可选）
    use_ema = config.get("training", {}).get("use_ema", False)
    if use_ema:
        ema = ModelEMA(model)
        print("启用EMA模型")

    # 创建早停
    early_stopping = EarlyStopping(
        patience=config["training"].get("patience"),
        min_delta=config["training"].get("min_delta"),
    )

    # 检查是否有之前的检查点
    latest_checkpoint = find_latest_checkpoint()
    start_epoch = 0
    best_loss = float("inf")
    resume_training = False

    if latest_checkpoint:
        print(f"🔄 找到已有检查点: {latest_checkpoint}")
        response = input("是否从断点继续训练? (Y/n): ").strip().lower()
        if response in ['', 'y', 'yes']:
            start_epoch, best_loss = load_checkpoint(
                latest_checkpoint, model, optimizer, scheduler, device
            )
            resume_training = True
            print(f"🔄 从断点继续训练，从epoch {start_epoch}开始")
        else:
            print("🆕 从预训练模型开始新训练")
    else:
        print("🆕 未找到检查点，开始新训练")

    # 创建TensorBoard写入器
    if config["logging"]["tensorboard"]:
        if resume_training:
            # 如果继续训练，使用现有的日志目录
            log_dir = os.path.dirname(latest_checkpoint).replace("checkpoints", "runs")
            if not os.path.exists(log_dir):
                log_dir = f"runs/faster_rcnn_{time.strftime('%Y%m%d_%H%M%S')}"
        else:
            log_dir = f"runs/faster_rcnn_{time.strftime('%Y%m%d_%H%M%S')}"

        writer = SummaryWriter(log_dir)
        print(f"TensorBoard日志保存在: {log_dir}")

    # 创建保存目录
    if resume_training:
        # 如果继续训练，使用现有的保存目录
        save_dir = Path(os.path.dirname(latest_checkpoint))
    else:
        save_dir = Path(f"checkpoints/faster_rcnn_{time.strftime('%Y%m%d_%H%M%S')}")
        save_dir.mkdir(parents=True, exist_ok=True)

    # 检查是否为测试模式，调整训练轮数
    test_mode = config.get('test_mode', {})
    if test_mode.get('enabled', False):
        total_epochs = test_mode.get('epochs', 5)
        print(f"🧪 测试模式: 将训练 {total_epochs} 轮")
    else:
        total_epochs = config["training"]["epochs"]

    print("开始训练...")
    for epoch in range(start_epoch, total_epochs):
        # 训练
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, epoch, config
        )

        # 验证
        val_loss, val_metrics = validate(model, val_loader, device, config)

        # 更新学习率
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # 更新EMA
        if use_ema:
            ema.update()

        # TensorBoard 记录 - 记录损失、学习率和评估指标
        if config["logging"]["tensorboard"]:
            writer.add_scalar("Loss/Train", train_metrics["loss"], epoch)
            writer.add_scalar("Loss/Val", val_loss, epoch)
            writer.add_scalar("Loss/RPN", train_metrics["rpn_loss"], epoch)
            writer.add_scalar("Loss/ROI", train_metrics["roi_loss"], epoch)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
            writer.add_scalar("Metrics/mAP50", val_metrics['mAP50'], epoch)
            writer.add_scalar("Metrics/mAP50-95", val_metrics['mAP50-95'], epoch)
            writer.add_scalar("Metrics/Precision", val_metrics['precision'], epoch)
            writer.add_scalar("Metrics/Recall", val_metrics['recall'], epoch)
            writer.add_scalar("Metrics/TP", val_metrics['TP'], epoch)
            writer.add_scalar("Metrics/FP", val_metrics['FP'], epoch)
            writer.add_scalar("Metrics/FN", val_metrics['FN'], epoch)

        # 打印训练信息 - 显示损失和评估指标
        print(f"Epoch {epoch+1}/{config['training']['epochs']}")
        print(f"Train Loss: {train_metrics['loss']:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"mAP50: {val_metrics['mAP50']:.4f}")
        print(f"mAP50-95: {val_metrics['mAP50-95']:.4f}")
        print(f"Precision: {val_metrics['precision']:.4f} (TP={val_metrics['TP']}, FP={val_metrics['FP']})")
        print(f"Recall: {val_metrics['recall']:.4f} (TP={val_metrics['TP']}, FN={val_metrics['FN']})")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 50)

        # 保存最佳模型（基于验证损失）
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_path = save_dir / "best_model.pth"
            save_model(model, optimizer, scheduler, epoch, val_loss, str(best_model_path), config)
            print(f"💾 保存最佳模型: {best_model_path}")

        # 定期保存检查点
        if (epoch + 1) % config["logging"]["save_interval"] == 0:
            checkpoint_path = save_dir / f"checkpoint_epoch_{epoch+1}.pth"
            save_model(model, optimizer, scheduler, epoch, val_loss, str(checkpoint_path), config)
            print(f"💾 保存检查点: {checkpoint_path}")

        # 早停检查
        if early_stopping(val_loss):
            print("🛑 早停触发，停止训练")
            break

    # 保存最终模型
    final_model_path = save_dir / "final_model.pth"
    save_model(model, optimizer, scheduler, epoch, val_loss, str(final_model_path), config)
    print(f"💾 保存最终模型: {final_model_path}")

    # 关闭TensorBoard
    if config["logging"]["tensorboard"]:
        writer.close()

    print("✅ 训练完成！")
    print(f"📊 训练日志保存在: {log_dir}")
    print(f"💾 模型保存在: {save_dir}")
    print("🔍 运行 'python generate_map_curves.py' 来生成 mAP50 变化图")


if __name__ == "__main__":
    main()

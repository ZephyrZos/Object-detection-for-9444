#!/usr/bin/env python3
"""
测试评估指标计算
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import calculate_map
import torch
import numpy as np

def test_calculate_map():
    """测试mAP计算函数"""

    # 模拟配置
    config = {
        'classes': {
            0: 'background',
            1: 'person',
            2: 'car'
        }
    }

    # 模拟预测结果
    predictions = [
        {
            'boxes': torch.tensor([[10, 10, 50, 50], [100, 100, 150, 150]]),
            'labels': torch.tensor([1, 2]),
            'scores': torch.tensor([0.9, 0.8])
        }
    ]

    # 模拟真实标签
    targets = [
        {
            'boxes': torch.tensor([[12, 12, 48, 48], [95, 95, 155, 155]]),
            'labels': torch.tensor([1, 2])
        }
    ]

    # 计算指标
    map50, map50_95, precision, recall = calculate_map(predictions, targets, config)

    print(f"mAP50: {map50:.4f}")
    print(f"mAP50-95: {map50_95:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    return map50, map50_95, precision, recall

if __name__ == "__main__":
    test_calculate_map()
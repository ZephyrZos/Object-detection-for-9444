#!/usr/bin/env python3
"""
业界标准的 Faster R-CNN 训练监控图表生成器
使用方法: python generate_plots.py --log_dir runs/faster_rcnn_20250728_021257
"""

import argparse
import os
import sys
import yaml
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import generate_industry_standard_plots, generate_training_report


def main():
    parser = argparse.ArgumentParser(description='生成Faster R-CNN训练监控图表')
    parser.add_argument('--log_dir', type=str, required=True,
                       help='TensorBoard日志目录路径')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径')
    parser.add_argument('--output', type=str, default='training_plots.png',
                       help='输出图表路径')
    parser.add_argument('--report', action='store_true',
                       help='同时生成训练报告')
    parser.add_argument('--report_dir', type=str, default='reports',
                       help='报告保存目录')

    args = parser.parse_args()

    # 检查日志目录是否存在
    if not os.path.exists(args.log_dir):
        print(f"❌ 日志目录不存在: {args.log_dir}")
        return

    # 加载配置
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ 无法加载配置文件: {e}")
        config = None

    # 生成图表
    print(f"📊 正在生成训练图表...")
    print(f"   日志目录: {args.log_dir}")
    print(f"   输出路径: {args.output}")

    generate_industry_standard_plots(args.log_dir, args.output, config)

    # 生成报告（如果请求）
    if args.report:
        print(f"📋 正在生成训练报告...")
        generate_training_report(args.log_dir, config, args.report_dir)

    print("✅ 完成！")


if __name__ == "__main__":
    main()
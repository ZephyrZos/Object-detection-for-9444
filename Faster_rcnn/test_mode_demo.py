#!/usr/bin/env python3
"""
测试模式演示脚本
展示如何使用test mode进行快速测试
"""

import os
import yaml
import torch
from dataset import create_data_loaders
from model import create_model
from utils import get_device

def demo_test_mode():
    """演示测试模式的使用"""
    
    print("🧪 测试模式演示")
    print("=" * 50)
    
    # 加载配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 显示当前配置
    test_mode = config.get('test_mode', {})
    print(f"测试模式状态: {'启用' if test_mode.get('enabled', False) else '禁用'}")
    
    if test_mode.get('enabled', False):
        print(f"测试模式配置:")
        print(f"  - 最大训练样本数: {test_mode.get('max_train_samples', 100)}")
        print(f"  - 最大验证样本数: {test_mode.get('max_val_samples', 50)}")
        print(f"  - 批次大小: {test_mode.get('batch_size', 2)}")
        print(f"  - 训练轮数: {test_mode.get('epochs', 5)}")
        print(f"  - 工作进程数: {test_mode.get('num_workers', 2)}")
    
    print("\n" + "=" * 50)
    
    # 检查数据集是否存在
    train_annotations = os.path.join(
        config["convert_output_path"], config["train_annotations"]
    )
    
    if not os.path.exists(train_annotations):
        print("❌ 未找到COCO格式标注文件")
        print("请先运行数据集转换:")
        print("python convert_dataset.py")
        return
    
    try:
        # 创建数据加载器
        print("📊 创建数据加载器...")
        train_loader, val_loader = create_data_loaders(config)
        
        print(f"✅ 数据加载器创建成功!")
        print(f"  - 训练批次数: {len(train_loader)}")
        print(f"  - 验证批次数: {len(val_loader)}")
        
        # 获取一个批次的数据进行测试
        print("\n📋 测试数据加载...")
        images, targets = next(iter(train_loader))
        print(f"  - 批次大小: {len(images)}")
        print(f"  - 图像形状: {images[0].shape}")
        print(f"  - 目标数量: {len(targets[0]['boxes'])}")
        
        # 测试模型创建
        print("\n🤖 测试模型创建...")
        device = get_device(config["device"])
        model = create_model(config, device)
        print(f"✅ 模型创建成功! 设备: {device}")
        
        # 测试前向传播
        print("\n🔄 测试前向传播...")
        model.eval()
        with torch.no_grad():
            # 只取第一张图像进行测试
            test_image = images[0].unsqueeze(0).to(device)
            predictions = model(test_image)
            print(f"✅ 前向传播成功!")
            print(f"  - 预测框数量: {len(predictions[0]['boxes'])}")
            print(f"  - 预测分数范围: {predictions[0]['scores'].min():.3f} - {predictions[0]['scores'].max():.3f}")
        
        print("\n🎉 测试模式演示完成!")
        
        if test_mode.get('enabled', False):
            print("\n💡 提示:")
            print("  - 当前为测试模式，适合快速验证代码")
            print("  - 要进行完整训练，请在config.yaml中设置 test_mode.enabled: False")
            print("  - 运行 'python train.py' 开始训练")
        else:
            print("\n💡 提示:")
            print("  - 要启用测试模式，请在config.yaml中设置 test_mode.enabled: True")
            print("  - 测试模式可以快速验证代码是否正常工作")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

def toggle_test_mode(enable: bool = True):
    """切换测试模式"""
    
    # 读取配置文件
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 修改测试模式设置
    if 'test_mode' not in config:
        config['test_mode'] = {}
    
    config['test_mode']['enabled'] = enable
    
    # 写回配置文件
    with open('config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    status = "启用" if enable else "禁用"
    print(f"✅ 测试模式已{status}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试模式演示和控制")
    parser.add_argument('--demo', action='store_true', help='运行测试模式演示')
    parser.add_argument('--enable', action='store_true', help='启用测试模式')
    parser.add_argument('--disable', action='store_true', help='禁用测试模式')
    
    args = parser.parse_args()
    
    if args.enable:
        toggle_test_mode(True)
    elif args.disable:
        toggle_test_mode(False)
    elif args.demo:
        demo_test_mode()
    else:
        # 默认运行演示
        demo_test_mode()

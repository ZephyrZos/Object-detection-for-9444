#!/usr/bin/env python3
"""
集成测试脚本
验证test mode的完整功能
"""

import os
import sys
import yaml
import torch
import tempfile
import shutil
from pathlib import Path

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_config_modification():
    """测试配置文件修改功能"""
    print("🧪 测试配置文件修改...")
    
    # 备份原配置
    backup_config = None
    with open('config.yaml', 'r', encoding='utf-8') as f:
        backup_config = f.read()
    
    try:
        # 测试启用test mode
        from test_mode_demo import toggle_test_mode
        toggle_test_mode(True)
        
        # 验证配置是否正确修改
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        assert config['test_mode']['enabled'] == True, "启用test mode失败"
        print("✅ 启用test mode成功")
        
        # 测试禁用test mode
        toggle_test_mode(False)
        
        # 验证配置是否正确修改
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        assert config['test_mode']['enabled'] == False, "禁用test mode失败"
        print("✅ 禁用test mode成功")
        
    finally:
        # 恢复原配置
        if backup_config:
            with open('config.yaml', 'w', encoding='utf-8') as f:
                f.write(backup_config)

def test_data_loader_creation():
    """测试数据加载器创建"""
    print("\n🧪 测试数据加载器创建...")
    
    # 加载配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 检查数据集是否存在
    train_annotations = os.path.join(
        config["convert_output_path"], config["train_annotations"]
    )
    
    if not os.path.exists(train_annotations):
        print("⚠️ 跳过数据加载器测试：未找到数据集")
        return
    
    from dataset import create_data_loaders
    
    # 测试正常模式
    config['test_mode']['enabled'] = False
    train_loader_normal, val_loader_normal = create_data_loaders(config)
    normal_train_size = len(train_loader_normal.dataset)
    normal_val_size = len(val_loader_normal.dataset)
    print(f"✅ 正常模式 - 训练集: {normal_train_size}, 验证集: {normal_val_size}")
    
    # 测试test mode
    config['test_mode']['enabled'] = True
    train_loader_test, val_loader_test = create_data_loaders(config)
    test_train_size = len(train_loader_test.dataset)
    test_val_size = len(val_loader_test.dataset)
    print(f"✅ 测试模式 - 训练集: {test_train_size}, 验证集: {test_val_size}")
    
    # 验证test mode确实减少了数据集大小
    max_train = config['test_mode']['max_train_samples']
    max_val = config['test_mode']['max_val_samples']
    
    assert test_train_size <= max_train, f"训练集大小超出限制: {test_train_size} > {max_train}"
    assert test_val_size <= max_val, f"验证集大小超出限制: {test_val_size} > {max_val}"
    assert test_train_size < normal_train_size, "测试模式未减少训练集大小"
    assert test_val_size < normal_val_size, "测试模式未减少验证集大小"
    
    print("✅ 数据集大小限制正常工作")

def test_batch_processing():
    """测试批次处理"""
    print("\n🧪 测试批次处理...")
    
    # 加载配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 检查数据集是否存在
    train_annotations = os.path.join(
        config["convert_output_path"], config["train_annotations"]
    )
    
    if not os.path.exists(train_annotations):
        print("⚠️ 跳过批次处理测试：未找到数据集")
        return
    
    from dataset import create_data_loaders
    
    # 启用test mode
    config['test_mode']['enabled'] = True
    train_loader, val_loader = create_data_loaders(config)
    
    try:
        # 测试获取一个批次
        images, targets = next(iter(train_loader))
        
        batch_size = len(images)
        expected_batch_size = config['test_mode']['batch_size']
        
        print(f"✅ 成功获取批次，大小: {batch_size}")
        print(f"✅ 图像形状: {images[0].shape}")
        print(f"✅ 目标数量: {len(targets)}")
        
        # 验证批次大小
        assert batch_size <= expected_batch_size, f"批次大小超出预期: {batch_size} > {expected_batch_size}"
        
        # 验证数据类型
        assert torch.is_tensor(images[0]), "图像不是tensor类型"
        assert isinstance(targets[0], dict), "目标不是字典类型"
        assert 'boxes' in targets[0], "目标中缺少boxes"
        assert 'labels' in targets[0], "目标中缺少labels"
        
        print("✅ 批次数据格式正确")
        
    except Exception as e:
        print(f"❌ 批次处理测试失败: {e}")
        raise

def test_model_integration():
    """测试模型集成"""
    print("\n🧪 测试模型集成...")
    
    try:
        from model import create_model
        from utils import get_device
        
        # 加载配置
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 创建模型
        device = get_device(config["device"])
        model = create_model(config, device)
        
        print(f"✅ 模型创建成功，设备: {device}")
        
        # 测试模型前向传播
        model.eval()
        
        # 创建虚拟输入
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        
        with torch.no_grad():
            predictions = model(dummy_input)
        
        print("✅ 模型前向传播成功")
        print(f"✅ 预测结果包含 {len(predictions)} 个检测结果")
        
        # 验证输出格式
        assert isinstance(predictions, list), "预测结果应该是列表"
        assert len(predictions) == 1, "批次大小为1时应该返回1个预测结果"
        assert 'boxes' in predictions[0], "预测结果中缺少boxes"
        assert 'labels' in predictions[0], "预测结果中缺少labels"
        assert 'scores' in predictions[0], "预测结果中缺少scores"
        
        print("✅ 模型输出格式正确")
        
    except Exception as e:
        print(f"❌ 模型集成测试失败: {e}")
        raise

def run_all_tests():
    """运行所有测试"""
    print("🚀 开始集成测试")
    print("=" * 60)
    
    tests = [
        ("配置文件修改", test_config_modification),
        ("数据加载器创建", test_data_loader_creation),
        ("批次处理", test_batch_processing),
        ("模型集成", test_model_integration),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\n📋 运行测试: {test_name}")
            test_func()
            print(f"✅ {test_name} 测试通过")
            passed += 1
        except Exception as e:
            print(f"❌ {test_name} 测试失败: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"🎯 测试结果: {passed} 通过, {failed} 失败")
    
    if failed == 0:
        print("🎉 所有测试通过！test mode功能正常")
    else:
        print("⚠️ 部分测试失败，请检查错误信息")
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

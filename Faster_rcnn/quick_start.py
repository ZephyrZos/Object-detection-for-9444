#!/usr/bin/env python3
"""
快速开始脚本
帮助用户快速上手test mode
"""

import os
import sys
import yaml
import subprocess
from pathlib import Path

def print_banner():
    """打印欢迎横幅"""
    print("🚀 Faster R-CNN 快速开始")
    print("=" * 50)
    print("这个脚本将帮助你快速开始使用test mode进行测试")
    print("=" * 50)

def check_environment():
    """检查环境"""
    print("\n🔍 检查环境...")
    
    # 检查Python版本
    python_version = sys.version_info
    print(f"✅ Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 检查PyTorch
    try:
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")
        print(f"✅ CUDA可用: {'是' if torch.cuda.is_available() else '否'}")
        if torch.cuda.is_available():
            print(f"✅ GPU数量: {torch.cuda.device_count()}")
    except ImportError:
        print("❌ PyTorch未安装")
        return False
    
    # 检查其他依赖
    required_packages = ['cv2', 'numpy', 'PIL', 'yaml', 'tqdm']
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} 已安装")
        except ImportError:
            print(f"❌ {package} 未安装")
            return False
    
    return True

def check_dataset():
    """检查数据集"""
    print("\n📊 检查数据集...")
    
    # 加载配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 检查转换后的数据集
    train_annotations = os.path.join(
        config["convert_output_path"], config["train_annotations"]
    )
    val_annotations = os.path.join(
        config["convert_output_path"], config["val_annotations"]
    )
    
    if os.path.exists(train_annotations) and os.path.exists(val_annotations):
        print("✅ 找到转换后的数据集")
        return True
    else:
        print("❌ 未找到转换后的数据集")
        print("请先运行数据集转换:")
        print("  python convert_dataset.py")
        return False

def setup_test_mode():
    """设置测试模式"""
    print("\n🧪 设置测试模式...")
    
    # 启用测试模式
    from test_mode_demo import toggle_test_mode
    toggle_test_mode(True)
    
    print("✅ 测试模式已启用")
    
    # 显示测试模式配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    test_mode = config['test_mode']
    print(f"📋 测试模式配置:")
    print(f"  - 训练样本数: {test_mode['max_train_samples']}")
    print(f"  - 验证样本数: {test_mode['max_val_samples']}")
    print(f"  - 批次大小: {test_mode['batch_size']}")
    print(f"  - 训练轮数: {test_mode['epochs']}")

def run_demo():
    """运行演示"""
    print("\n🎬 运行测试模式演示...")
    
    try:
        from test_mode_demo import demo_test_mode
        demo_test_mode()
        return True
    except Exception as e:
        print(f"❌ 演示运行失败: {e}")
        return False

def ask_user_choice():
    """询问用户选择"""
    print("\n🤔 你想要做什么？")
    print("1. 运行快速训练测试（推荐）")
    print("2. 只运行演示，不训练")
    print("3. 退出")
    
    while True:
        choice = input("请选择 (1-3): ").strip()
        if choice in ['1', '2', '3']:
            return int(choice)
        print("请输入有效选择 (1-3)")

def run_quick_training():
    """运行快速训练"""
    print("\n🚀 开始快速训练...")
    print("这将运行5轮训练，使用100个训练样本")
    print("预计需要几分钟时间...")
    
    try:
        # 运行训练
        result = subprocess.run([sys.executable, 'train.py'], 
                              capture_output=False, 
                              text=True)
        
        if result.returncode == 0:
            print("✅ 快速训练完成！")
            return True
        else:
            print("❌ 训练过程中出现错误")
            return False
            
    except Exception as e:
        print(f"❌ 训练启动失败: {e}")
        return False

def main():
    """主函数"""
    print_banner()
    
    # 检查环境
    if not check_environment():
        print("\n❌ 环境检查失败，请安装缺失的依赖")
        return
    
    # 检查数据集
    if not check_dataset():
        print("\n❌ 数据集检查失败，请先准备数据集")
        return
    
    # 设置测试模式
    setup_test_mode()
    
    # 运行演示
    if not run_demo():
        print("\n❌ 演示运行失败")
        return
    
    # 询问用户选择
    choice = ask_user_choice()
    
    if choice == 1:
        # 运行快速训练
        if run_quick_training():
            print("\n🎉 快速开始完成！")
            print("\n💡 下一步建议:")
            print("1. 查看训练日志和结果")
            print("2. 如果一切正常，可以禁用测试模式进行完整训练:")
            print("   python test_mode_demo.py --disable")
            print("   python train.py")
        else:
            print("\n❌ 快速训练失败")
    
    elif choice == 2:
        print("\n✅ 演示完成！")
        print("\n💡 如果要开始训练，运行:")
        print("   python train.py")
    
    else:
        print("\n👋 再见！")

if __name__ == "__main__":
    main()

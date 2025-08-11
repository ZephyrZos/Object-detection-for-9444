#!/usr/bin/env python3
"""
脚本用于分析Yolo_method/dataset/train/labels目录下所有标签文件中的类别
YOLO格式: 每行为 class_id x_center y_center width height（归一化坐标）
"""

import os
import glob
from collections import Counter
import matplotlib.pyplot as plt

def analyze_yolo_labels(labels_dir):
    """
    分析YOLO格式标签文件，统计所有类别
    
    Args:
        labels_dir: 标签文件目录路径
    
    Returns:
        dict: 类别统计结果
    """
    # VisDrone数据集类别映射
    CLASS_NAMES = {
        0: 'ignored regions',
        1: 'pedestrian', 
        2: 'people',
        3: 'bicycle',
        4: 'car',
        5: 'van',
        6: 'truck',
        7: 'tricycle',
        8: 'awning-tricycle', 
        9: 'bus',
        10: 'motor'
    }
    
    print(f"正在分析标签目录: {labels_dir}")
    
    # 获取所有txt标签文件
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    print(f"找到标签文件数量: {len(label_files)}")
    
    if len(label_files) == 0:
        print("错误: 未找到任何标签文件!")
        return {}
    
    # 统计类别和对象数量
    class_counter = Counter()
    total_objects = 0
    files_processed = 0
    
    for label_file in label_files:
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for line in lines:
                line = line.strip()
                if line:  # 跳过空行
                    parts = line.split()
                    if len(parts) >= 5:  # YOLO格式至少有5个值
                        class_id = int(parts[0])
                        class_counter[class_id] += 1
                        total_objects += 1
            
            files_processed += 1
            if files_processed % 1000 == 0:
                print(f"已处理 {files_processed} 个文件...")
                
        except Exception as e:
            print(f"处理文件 {label_file} 时出错: {e}")
    
    print(f"处理完成! 共处理 {files_processed} 个文件，发现 {total_objects} 个标注对象")
    
    # 输出统计结果
    print(f"\n{'='*60}")
    print("VisDrone数据集类别统计结果:")
    print(f"{'='*60}")
    print(f"{'类别ID':<6} {'类别名称':<20} {'数量':<10} {'百分比':<10}")
    print(f"{'-'*60}")
    
    # 按类别ID排序
    for class_id in sorted(class_counter.keys()):
        count = class_counter[class_id]
        percentage = (count / total_objects) * 100 if total_objects > 0 else 0
        class_name = CLASS_NAMES.get(class_id, f'unknown_{class_id}')
        print(f"{class_id:<6} {class_name:<20} {count:<10} {percentage:>6.2f}%")
    
    print(f"{'-'*60}")
    print(f"总计: {total_objects} 个标注对象")
    print(f"出现的类别数: {len(class_counter)}")
    print(f"{'='*60}")
    
    return dict(class_counter), CLASS_NAMES, total_objects

def create_visualization(class_counter, class_names, total_objects, output_dir="./"):
    """
    创建类别分布可视化图表
    
    Args:
        class_counter: 类别计数字典
        class_names: 类别名称映射
        total_objects: 总对象数
        output_dir: 输出目录
    """
    if not class_counter:
        print("没有数据用于可视化")
        return
    
    # 准备数据
    sorted_classes = sorted(class_counter.keys())
    counts = [class_counter[cid] for cid in sorted_classes]
    names = [class_names.get(cid, f'Class_{cid}') for cid in sorted_classes]
    percentages = [(count/total_objects)*100 for count in counts]
    
    # 创建图表
    plt.figure(figsize=(15, 10))
    
    # 子图1: 柱状图
    plt.subplot(2, 1, 1)
    bars = plt.bar(range(len(sorted_classes)), counts, color='steelblue', alpha=0.7)
    plt.xlabel('类别')
    plt.ylabel('标注对象数量')
    plt.title('VisDrone数据集训练集类别分布统计')
    plt.xticks(range(len(sorted_classes)), 
               [f'{cid}\n{names[i][:10]}' for i, cid in enumerate(sorted_classes)], 
               rotation=45, ha='right')
    
    # 在柱子上显示数值
    for i, (bar, count) in enumerate(zip(bars, counts)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                f'{count}\n({percentages[i]:.1f}%)', 
                ha='center', va='bottom', fontsize=9)
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # 子图2: 饼图 (只显示主要类别)
    plt.subplot(2, 1, 2)
    
    # 过滤小类别，将占比小于1%的类别合并为"其他"
    main_classes = []
    main_counts = []
    main_names = []
    others_count = 0
    
    for i, (cid, count) in enumerate(zip(sorted_classes, counts)):
        if percentages[i] >= 1.0:  # 占比>=1%的类别单独显示
            main_classes.append(cid)
            main_counts.append(count)
            main_names.append(names[i])
        else:
            others_count += count
    
    if others_count > 0:
        main_counts.append(others_count)
        main_names.append('其他类别')
    
    colors = plt.cm.Set3(range(len(main_counts)))
    wedges, texts, autotexts = plt.pie(main_counts, labels=main_names, autopct='%1.1f%%', 
                                       colors=colors, startangle=90)
    
    plt.title('主要类别分布 (占比>=1%的类别)')
    
    # 保存图表
    output_file = os.path.join(output_dir, 'yolo_class_analysis.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n可视化图表已保存到: {output_file}")
    
    # 显示图表 (如果在交互环境中)
    try:
        plt.show()
    except:
        print("注意: 无法显示图表，但已保存到文件")

def main():
    """主函数"""
    # 设置标签文件目录路径
    labels_dir = "Yolo_method/dataset/train/labels"
    
    # 检查目录是否存在
    if not os.path.exists(labels_dir):
        print(f"错误: 目录不存在 - {labels_dir}")
        print("请确保从项目根目录运行此脚本")
        return
    
    # 分析标签文件
    class_counter, class_names, total_objects = analyze_yolo_labels(labels_dir)
    
    if class_counter:
        # 创建可视化
        create_visualization(class_counter, class_names, total_objects)
        
        # 保存统计结果到文件
        output_file = "yolo_class_statistics.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("VisDrone数据集训练集类别统计结果\n")
            f.write("="*60 + "\n")
            f.write(f"{'类别ID':<6} {'类别名称':<20} {'数量':<10} {'百分比':<10}\n")
            f.write("-"*60 + "\n")
            
            for class_id in sorted(class_counter.keys()):
                count = class_counter[class_id]
                percentage = (count / total_objects) * 100
                class_name = class_names.get(class_id, f'unknown_{class_id}')
                f.write(f"{class_id:<6} {class_name:<20} {count:<10} {percentage:>6.2f}%\n")
            
            f.write("-"*60 + "\n")
            f.write(f"总计: {total_objects} 个标注对象\n")
            f.write(f"出现的类别数: {len(class_counter)}\n")
            
        print(f"统计结果已保存到: {output_file}")

if __name__ == "__main__":
    main()
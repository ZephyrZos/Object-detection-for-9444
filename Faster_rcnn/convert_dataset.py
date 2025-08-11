#!/usr/bin/env python3
"""
VisDrone数据集转换脚本
将VisDrone格式转换为COCO格式，用于Faster R-CNN训练
同时复制图像文件到输出目录
"""

import os
import json
import glob
import zipfile
import shutil
from PIL import Image
import argparse
from pathlib import Path
import yaml
from tqdm import tqdm


def convert_visdrone_to_coco(visdrone_dir: str, output_dir: str):
    """将VisDrone格式转换为COCO格式，同时复制图像文件"""
    print(f"开始转换数据集...")
    print(f"输入目录: {visdrone_dir}")
    print(f"输出目录: {output_dir}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)

    # 类别映射
    class_names = [
        "person",
        "bicycle",
        "car",
        "van",
        "truck",
        "tricycle",
        "awning-tricycle",
        "bus",
        "motor",
    ]

    # 处理训练、验证、测试集、挑战集
    for split in ["train", "val", "test-dev", "test-challenge"]:
        print(f"\n处理 {split} 集...")

        # 原始数据路径
        img_dir = os.path.join(visdrone_dir, f"VisDrone2019-DET-{split}", "images")
        ann_dir = os.path.join(visdrone_dir, f"VisDrone2019-DET-{split}", "annotations")

        # 输出路径
        output_img_dir = os.path.join(output_dir, split, "images")
        os.makedirs(output_img_dir, exist_ok=True)

        # 检查并解压数据
        if not os.path.exists(img_dir):
            zip_file = f'{visdrone_dir}/VisDrone2019-DET-{split}.zip'
            if not os.path.exists(zip_file):
                print(f"跳过 {split}，目录不存在: {img_dir}.zip")
                continue
            else:
                print(f"解压 {split} 集...")
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(visdrone_dir)
                print(f"解压 {split} 集完成")

        # COCO格式数据
        coco_data = {"images": [], "annotations": [], "categories": []}

        # 添加类别
        for i, name in enumerate(class_names, 1):
            coco_data["categories"].append(
                {"id": i, "name": name, "supercategory": "vehicle"}
            )

        # 处理图像和标注
        img_files = glob.glob(os.path.join(img_dir, "*.jpg"))
        annotation_id = 1

        print(f"找到 {len(img_files)} 张图像")

        # 复制图像文件
        print(f"  复制图像文件到 {output_img_dir}...")
        copied_count = 0
        for img_file in tqdm(img_files, desc=f"复制 {split} 图像"):
            img_name = os.path.basename(img_file)
            output_img_path = os.path.join(output_img_dir, img_name)

            try:
                shutil.copy2(img_file, output_img_path)
                copied_count += 1
            except Exception as e:
                print(f"    复制图像失败 {img_name}: {e}")
                continue

        print(f"  ✅ 成功复制 {copied_count} 张图像")

        # 处理标注
        print(f"  处理标注文件...")
        for img_file in tqdm(img_files, desc=f"处理 {split} 标注"):
            img_name = os.path.basename(img_file)
            ann_file = os.path.join(ann_dir, img_name.replace(".jpg", ".txt"))

            # 读取图像尺寸
            try:
                img = Image.open(img_file)
                width, height = img.size
            except Exception as e:
                print(f"  跳过图像 {img_name}: {e}")
                continue

            # 添加图像信息到COCO格式
            image_id = len(coco_data["images"]) + 1
            coco_data["images"].append(
                {
                    "id": image_id,
                    "file_name": img_name,
                    "width": width,
                    "height": height,
                }
            )

            # 读取并转换标注格式
            if os.path.exists(ann_file):
                try:
                    with open(ann_file, "r") as f:
                        lines = f.readlines()

                    for line in lines:
                        parts = line.strip().split(",")
                        if len(parts) >= 6:
                            x, y, w, h = map(float, parts[:4])
                            class_id = int(parts[5])

                            # 只处理有效的类别
                            if 1 <= class_id <= len(class_names):
                                # 确保边界框在图像范围内
                                x = max(0, x)
                                y = max(0, y)
                                w = min(w, width - x)
                                h = min(h, height - y)

                                if w > 0 and h > 0:  # 确保边界框有效
                                    coco_data["annotations"].append(
                                        {
                                            "id": annotation_id,
                                            "image_id": image_id,
                                            "category_id": class_id,
                                            "bbox": [x, y, w, h],
                                            "area": w * h,
                                            "iscrowd": 0,
                                        }
                                    )
                                    annotation_id += 1
                except Exception as e:
                    print(f"  处理标注文件 {ann_file} 时出错: {e}")
                    continue

        # 保存COCO格式文件
        output_file = os.path.join(output_dir, "annotations", f"{split}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)

        print(f"✅ {split} 集处理完成:")
        print(f"  图像数量: {len(coco_data['images'])}")
        print(f"  标注数量: {len(coco_data['annotations'])}")
        print(f"  复制图像: {copied_count} 张")
        print(f"  保存到: {output_file}")

    print(f"\n✅ 所有数据集转换完成！")
    print(f"结果保存在: {output_dir}")


def validate_conversion(output_dir: str):
    """验证转换结果"""
    print(f"\n验证转换结果...")

    for split in ["train", "val", "test-dev", "test-challenge"]:
        json_file = os.path.join(output_dir, "annotations", f"{split}.json")
        img_dir = os.path.join(output_dir, split, "images")

        if not os.path.exists(json_file):
            print(f"⚠️ {split} 集JSON文件不存在")
            continue

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 检查图像文件
            expected_images = len(data['images'])
            actual_images = len(glob.glob(os.path.join(img_dir, "*.jpg"))) if os.path.exists(img_dir) else 0

            print(f"✅ {split} 集验证通过:")
            print(f"  图像数量: {len(data['images'])}")
            print(f"  标注数量: {len(data['annotations'])}")
            print(f"  类别数量: {len(data['categories'])}")
            print(f"  实际图像文件: {actual_images}/{expected_images}")

            # 检查标注分布
            class_counts = {}
            for ann in data["annotations"]:
                cat_id = ann["category_id"]
                class_counts[cat_id] = class_counts.get(cat_id, 0) + 1

            print(f"  类别分布: {class_counts}")

            if actual_images != expected_images:
                print(f"  ⚠️ 警告: 图像文件数量不匹配")

        except Exception as e:
            print(f"❌ {split} 集验证失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="VisDrone数据集转换工具")
    parser.add_argument("--input", type=str, help="VisDrone数据集目录路径")
    parser.add_argument("--output", type=str, help="输出目录路径")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="配置文件路径"
    )
    parser.add_argument("--validate", action="store_true", help="转换后验证结果")

    args = parser.parse_args()

    # 加载配置文件
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"错误: 无法加载配置文件 {args.config}: {e}")
        return

    # 从配置文件读取路径
    input_path = config.get("dataset_path")
    # 输出到Faster R-CNN项目目录下的专门文件夹
    output_path = config.get("convert_output_path")

    print(f"输入路径: {input_path}")
    print(f"输出路径: {output_path}")

    # 检查输入目录
    if not os.path.exists(input_path):
        print(f"错误: 输入目录不存在: {input_path}")
        return

    # 转换数据集
    convert_visdrone_to_coco(input_path, output_path)

    # 验证结果
    if args.validate:
        validate_conversion(output_path)

    print(f"\n🎉 数据集转换完成！")
    print(f"COCO格式数据集保存在: {output_path}")
    print(f"图像文件已复制到: {output_path}/[split]/images/")
    print(f"现在可以开始训练Faster R-CNN模型了。")


if __name__ == "__main__":
    main()

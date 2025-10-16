"""
数据集分配脚本

将标注好的图片和标注文件按照8:2的比例分为训练集和验证集
"""

import shutil
from pathlib import Path
import random

def split_dataset(
    source_images_dir="dataset/test_baidu_bg",
    source_labels_dir="dataset/watermark_detection/labels/train",
    train_images_dir="dataset/watermark_detection/images/train",
    val_images_dir="dataset/watermark_detection/images/val",
    train_labels_dir="dataset/watermark_detection/labels/train",
    val_labels_dir="dataset/watermark_detection/labels/val",
    split_ratio=0.8,
    seed=42
):
    """
    分配数据集
    
    Args:
        source_images_dir: 源图片目录
        source_labels_dir: 源标注目录
        train_images_dir: 训练图片目录
        val_images_dir: 验证图片目录
        train_labels_dir: 训练标注目录
        val_labels_dir: 验证标注目录
        split_ratio: 训练集比例（默认0.8，即80%）
        seed: 随机种子
    """
    # 转换为Path对象
    source_images = Path(source_images_dir)
    source_labels = Path(source_labels_dir)
    train_images = Path(train_images_dir)
    val_images = Path(val_images_dir)
    train_labels = Path(train_labels_dir)
    val_labels = Path(val_labels_dir)
    
    # 创建目录
    train_images.mkdir(parents=True, exist_ok=True)
    val_images.mkdir(parents=True, exist_ok=True)
    train_labels.mkdir(parents=True, exist_ok=True)
    val_labels.mkdir(parents=True, exist_ok=True)
    
    # 获取所有已标注的图片
    label_files = list(source_labels.glob("*.txt"))
    
    if not label_files:
        print("❌ 错误：未找到标注文件！")
        print(f"   请先运行标注工具: python scripts/create_annotations.py")
        return
    
    print(f"找到 {len(label_files)} 个标注文件")
    
    # 检查对应的图片是否存在
    valid_pairs = []
    for label_file in label_files:
        # 查找对应的图片（可能是jpg或png）
        img_name_jpg = source_images / f"{label_file.stem}.jpg"
        img_name_png = source_images / f"{label_file.stem}.png"
        
        if img_name_jpg.exists():
            valid_pairs.append((img_name_jpg, label_file))
        elif img_name_png.exists():
            valid_pairs.append((img_name_png, label_file))
        else:
            print(f"⚠️  警告：未找到图片 {label_file.stem}")
    
    print(f"找到 {len(valid_pairs)} 对有效的图片-标注对")
    
    if len(valid_pairs) == 0:
        print("❌ 错误：没有找到有效的图片-标注对！")
        return
    
    # 随机打乱
    random.seed(seed)
    random.shuffle(valid_pairs)
    
    # 计算分割点
    split_point = int(len(valid_pairs) * split_ratio)
    train_pairs = valid_pairs[:split_point]
    val_pairs = valid_pairs[split_point:]
    
    print(f"\n开始分配数据集...")
    print(f"  训练集: {len(train_pairs)} 对")
    print(f"  验证集: {len(val_pairs)} 对")
    
    # 复制训练集
    print("\n正在复制训练集...")
    for img_path, label_path in train_pairs:
        # 复制图片
        shutil.copy2(img_path, train_images / img_path.name)
        # 标注文件已经在train目录，不需要复制
        # 但如果需要重新组织，可以复制
    
    # 复制验证集
    print("正在复制验证集...")
    for img_path, label_path in val_pairs:
        # 复制图片
        shutil.copy2(img_path, val_images / img_path.name)
        # 复制标注
        shutil.copy2(label_path, val_labels / label_path.name)
    
    print("\n✓ 数据集分配完成！")
    print(f"\n训练集: {len(train_pairs)} 对")
    print(f"  图片: {train_images}")
    print(f"  标注: {train_labels}")
    print(f"\n验证集: {len(val_pairs)} 对")
    print(f"  图片: {val_images}")
    print(f"  标注: {val_labels}")
    
    # 统计信息
    print("\n" + "=" * 60)
    print("数据集统计")
    print("=" * 60)
    print(f"总计: {len(valid_pairs)} 对")
    print(f"训练集: {len(train_pairs)} 对 ({len(train_pairs)/len(valid_pairs)*100:.1f}%)")
    print(f"验证集: {len(val_pairs)} 对 ({len(val_pairs)/len(valid_pairs)*100:.1f}%)")


if __name__ == "__main__":
    print("=" * 60)
    print("数据集分配工具")
    print("=" * 60)
    print()
    
    split_dataset()

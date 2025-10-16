"""
百度地图水印自动标注脚本

根据生成水印时的固定位置，自动为所有图片生成YOLO格式标注
无需手动标注，完全自动化
"""

from pathlib import Path
from PIL import Image
from tqdm import tqdm

def auto_annotate(
    images_dir="dataset/test_baidu_bg",
    output_dir="dataset/watermark_detection/labels/train",
    watermark_position="bottom-right",
    watermark_width_ratio=0.15,   # 水印宽度占图片宽度的15%
    watermark_height_ratio=0.05,  # 水印高度占图片高度的5%
    margin_x=10,  # 右边距
    margin_y=10   # 下边距
):
    """
    自动标注水印位置
    
    Args:
        images_dir: 图片目录
        output_dir: 标注输出目录
        watermark_position: 水印位置（默认右下角）
        watermark_width_ratio: 水印宽度比例
        watermark_height_ratio: 水印高度比例
        margin_x: 水平边距
        margin_y: 垂直边距
    """
    images_path = Path(images_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图片
    image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
    
    if not image_files:
        print(f"❌ 错误：在 {images_dir} 中未找到图片！")
        return
    
    print(f"找到 {len(image_files)} 张图片")
    print(f"开始自动标注...")
    print(f"  位置: {watermark_position}")
    print(f"  水印大小: {watermark_width_ratio*100:.0f}% x {watermark_height_ratio*100:.0f}%")
    print()
    
    success_count = 0
    skip_count = 0
    
    # 使用进度条
    for img_path in tqdm(image_files, desc="标注进度"):
        try:
            # 读取图片尺寸
            with Image.open(img_path) as img:
                img_width, img_height = img.size
            
            # 计算水印尺寸
            watermark_w = int(img_width * watermark_width_ratio)
            watermark_h = int(img_height * watermark_height_ratio)
            
            # 计算水印位置（右下角）
            if watermark_position == "bottom-right":
                x2 = img_width - margin_x
                y2 = img_height - margin_y
                x1 = x2 - watermark_w
                y1 = y2 - watermark_h
            elif watermark_position == "bottom-left":
                x1 = margin_x
                y2 = img_height - margin_y
                x2 = x1 + watermark_w
                y1 = y2 - watermark_h
            elif watermark_position == "top-right":
                x2 = img_width - margin_x
                y1 = margin_y
                x1 = x2 - watermark_w
                y2 = y1 + watermark_h
            elif watermark_position == "top-left":
                x1 = margin_x
                y1 = margin_y
                x2 = x1 + watermark_w
                y2 = y1 + watermark_h
            else:
                # 默认右下角
                x2 = img_width - margin_x
                y2 = img_height - margin_y
                x1 = x2 - watermark_w
                y1 = y2 - watermark_h
            
            # 转换为YOLO格式（归一化坐标）
            center_x = ((x1 + x2) / 2) / img_width
            center_y = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            
            # 确保坐标在有效范围内
            center_x = max(0, min(1, center_x))
            center_y = max(0, min(1, center_y))
            width = max(0, min(1, width))
            height = max(0, min(1, height))
            
            # 保存标注文件
            label_path = output_path / f"{img_path.stem}.txt"
            with open(label_path, 'w') as f:
                # YOLO格式: class_id center_x center_y width height
                f.write(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
            
            success_count += 1
            
        except Exception as e:
            print(f"\n⚠️  处理失败: {img_path.name} - {e}")
            skip_count += 1
    
    print("\n" + "=" * 60)
    print("自动标注完成！")
    print("=" * 60)
    print(f"✓ 成功: {success_count} 张")
    if skip_count > 0:
        print(f"⚠️  跳过: {skip_count} 张")
    print(f"\n标注文件保存在: {output_path}")
    print(f"\n下一步: 运行 python scripts/split_dataset.py 分配数据集")


def verify_annotations(
    images_dir="dataset/test_baidu_bg",
    labels_dir="dataset/watermark_detection/labels/train",
    num_samples=5
):
    """
    验证标注（可选）
    显示几张带标注框的图片，确保标注正确
    """
    import cv2
    import numpy as np
    
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    
    # 随机选择几张图片
    image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
    import random
    samples = random.sample(image_files, min(num_samples, len(image_files)))
    
    print(f"\n验证标注（显示 {len(samples)} 张示例）...")
    print("按任意键继续，按 'q' 退出")
    
    for img_path in samples:
        label_path = labels_path / f"{img_path.stem}.txt"
        
        if not label_path.exists():
            print(f"⚠️  未找到标注: {img_path.name}")
            continue
        
        # 读取图片
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        h, w = img.shape[:2]
        
        # 读取标注
        with open(label_path, 'r') as f:
            line = f.readline().strip()
            if line:
                parts = line.split()
                if len(parts) == 5:
                    class_id, cx, cy, bw, bh = map(float, parts)
                    
                    # 转换为像素坐标
                    x1 = int((cx - bw/2) * w)
                    y1 = int((cy - bh/2) * h)
                    x2 = int((cx + bw/2) * w)
                    y2 = int((cy + bh/2) * h)
                    
                    # 绘制边界框
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, "baidu_watermark", (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 调整显示大小
        scale = 800 / max(h, w)
        if scale < 1:
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h))
        
        cv2.imshow(f'Annotation Verification - {img_path.name}', img)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    print("验证完成")


if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("百度地图水印自动标注工具")
    print("=" * 60)
    print()
    
    # 自动标注
    auto_annotate()
    
    # 询问是否验证
    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        verify_annotations()
    else:
        print("\n提示: 运行 python scripts/auto_annotate.py --verify 可查看标注效果")

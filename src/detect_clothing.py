"""
人物衣着检测脚本

功能：
1. 检测图片中的人物
2. 识别衣服类型（短袖/长袖，厚重/轻薄）
3. 判断衣着是否符合季节（夏装 vs 冬装）
4. 识别衣服颜色

使用预训练的YOLOv8模型
"""

from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
from pathlib import Path

class ClothingDetector:
    def __init__(self):
        # 加载YOLOv8人物检测模型
        self.person_model = YOLO('yolov8n.pt')  # 预训练COCO模型可检测人
        
    def detect_persons(self, image_path, conf=0.5):
        """
        检测图片中的人物
        
        Returns:
            list: 人物检测框 [(x1, y1, x2, y2, confidence), ...]
        """
        results = self.person_model.predict(image_path, conf=conf, classes=[0])  # class 0 = person
        
        persons = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                persons.append((int(x1), int(y1), int(x2), int(y2), float(confidence)))
        
        return persons
    
    def analyze_clothing_season(self, image, person_box):
        """
        分析衣着季节（夏装/冬装）
        
        判断依据：
        - 皮肤暴露面积（短袖 vs 长袖）
        - 衣服颜色深浅（深色厚重 vs 浅色轻薄）
        - 上半身覆盖程度
        
        Args:
            image: PIL Image或numpy array
            person_box: (x1, y1, x2, y2, conf)
        
        Returns:
            dict: 季节判断结果
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        x1, y1, x2, y2 = person_box[:4]
        person_region = image[y1:y2, x1:x2]
        
        height = y2 - y1
        width = x2 - x1
        
        # 1. 分析上半身（衣服区域）
        upper_body_y2 = y1 + int(height * 0.6)
        upper_body = image[y1:upper_body_y2, x1:x2]
        
        # 2. 分析手臂区域（判断长袖/短袖）
        # 手臂通常在人体两侧，中间30%-70%是躯干
        left_arm_x2 = x1 + int(width * 0.3)
        right_arm_x1 = x1 + int(width * 0.7)
        
        # 下半身开始位置（腰部以下）
        lower_start_y = y1 + int(height * 0.5)
        lower_end_y = y1 + int(height * 0.7)
        
        left_arm = image[lower_start_y:lower_end_y, x1:left_arm_x2]
        right_arm = image[lower_start_y:lower_end_y, right_arm_x1:x2]
        
        # 3. 检测皮肤颜色（肤色检测）
        def is_skin_color(region):
            """简单的肤色检测"""
            if region.size == 0:
                return 0
            
            # 转换到HSV
            hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
            
            # 肤色范围 (HSV)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 170, 255], dtype=np.uint8)
            
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            skin_ratio = np.sum(skin_mask > 0) / skin_mask.size
            
            return skin_ratio
        
        # 计算手臂皮肤暴露比例
        left_skin_ratio = is_skin_color(left_arm) if left_arm.size > 0 else 0
        right_skin_ratio = is_skin_color(right_arm) if right_arm.size > 0 else 0
        avg_skin_ratio = (left_skin_ratio + right_skin_ratio) / 2
        
        # 4. 分析衣服颜色（深浅）
        upper_body_gray = cv2.cvtColor(upper_body, cv2.COLOR_RGB2GRAY)
        avg_brightness = np.mean(upper_body_gray)
        
        # 5. 判断季节
        # 短袖判断：手臂皮肤暴露 > 20%
        is_short_sleeve = avg_skin_ratio > 0.2
        
        # 浅色判断：平均亮度 > 120
        is_light_color = avg_brightness > 120
        
        # 综合判断
        if is_short_sleeve or (is_light_color and avg_skin_ratio > 0.1):
            season = "summer"  # 夏装
            season_cn = "夏装"
            confidence = 0.7 + (avg_skin_ratio * 0.3)
        else:
            season = "winter"  # 冬装
            season_cn = "冬装"
            confidence = 0.7 + ((1 - avg_skin_ratio) * 0.3)
        
        return {
            'season': season,
            'season_cn': season_cn,
            'confidence': confidence,
            'skin_exposure': avg_skin_ratio,
            'brightness': avg_brightness,
            'is_short_sleeve': is_short_sleeve,
            'is_light_color': is_light_color
        }
    
    def check_season_compliance(self, image_path, expected_season="summer"):
        """
        检查衣着是否符合当前季节
        
        Args:
            image_path: 图片路径
            expected_season: 期望的季节 ('summer' 或 'winter')
        
        Returns:
            dict: 检测结果
        """
        # 检测人物
        persons = self.detect_persons(image_path)
        
        if not persons:
            return {
                'has_person': False,
                'compliant': None,
                'message': '未检测到人物'
            }
        
        # 加载图片
        image = Image.open(image_path)
        image_np = np.array(image)
        
        results = []
        for person in persons:
            season_info = self.analyze_clothing_season(image_np, person)
            
            compliant = season_info['season'] == expected_season
            
            results.append({
                'box': person[:4],
                'confidence': person[4],
                'detected_season': season_info['season_cn'],
                'season_confidence': season_info['confidence'],
                'compliant': compliant,
                'details': {
                    'skin_exposure': season_info['skin_exposure'],
                    'is_short_sleeve': season_info['is_short_sleeve']
                }
            })
        
        # 总体判断
        all_compliant = all(r['compliant'] for r in results)
        
        season_cn = "夏季" if expected_season == "summer" else "冬季"
        
        return {
            'has_person': True,
            'person_count': len(persons),
            'persons': results,
            'all_compliant': all_compliant,
            'message': f"检测到{len(persons)}人，{'全部' if all_compliant else '部分'}符合{season_cn}着装要求"
        }
    
    def visualize_results(self, image_path, expected_season="summer", output_path=None):
        """
        可视化检测结果
        """
        result = self.check_season_compliance(image_path, expected_season)
        
        if not result['has_person']:
            print(result['message'])
            return
        
        # 加载图片
        image = cv2.imread(str(image_path))
        
        # 绘制检测框和标签
        for person in result['persons']:
            x1, y1, x2, y2 = person['box']
            detected_season = person['detected_season']
            compliant = person['compliant']
            
            # 颜色：绿色=符合，红色=不符合
            box_color = (0, 255, 0) if compliant else (0, 0, 255)
            
            cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)
            
            label = f"{detected_season} ({'OK' if compliant else 'NG'})"
            cv2.putText(image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
        
        # 添加总体结果
        status_text = result['message']
        status_color = (0, 255, 0) if result['all_compliant'] else (0, 0, 255)
        cv2.putText(image, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        if output_path:
            cv2.imwrite(str(output_path), image)
            print(f"结果保存到: {output_path}")
        else:
            # 显示
            cv2.imshow('Season Compliance Check', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def batch_check_clothing(image_dir, expected_season="summer", output_csv="season_check_results.csv"):
    """
    批量检查图片中的衣着季节符合性
    
    Args:
        image_dir: 图片目录
        expected_season: 期望季节 ('summer' 或 'winter')
        output_csv: 输出CSV文件名
    """
    import csv
    from tqdm import tqdm
    
    detector = ClothingDetector()
    image_dir = Path(image_dir)
    
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    
    results = []
    
    season_cn = "夏季" if expected_season == "summer" else "冬季"
    print(f"开始批量检测 {len(image_files)} 张图片（期望季节：{season_cn}）...")
    
    for img_path in tqdm(image_files):
        try:
            result = detector.check_season_compliance(str(img_path), expected_season)
            
            # 获取第一个人的详细信息
            first_person = result['persons'][0] if result.get('persons') else {}
            
            results.append({
                'filename': img_path.name,
                'has_person': result['has_person'],
                'person_count': result.get('person_count', 0),
                'detected_season': first_person.get('detected_season', 'N/A'),
                'all_compliant': result.get('all_compliant', False),
                'message': result['message']
            })
        except Exception as e:
            results.append({
                'filename': img_path.name,
                'has_person': False,
                'person_count': 0,
                'detected_season': 'Error',
                'all_compliant': False,
                'message': f"处理失败: {e}"
            })
    
    # 保存结果
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'has_person', 'person_count', 'detected_season', 'all_compliant', 'message'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n结果已保存到: {output_csv}")
    
    # 统计
    total = len(results)
    has_person = sum(1 for r in results if r['has_person'])
    compliant = sum(1 for r in results if r['all_compliant'])
    
    print(f"\n统计：")
    print(f"  总图片数: {total}")
    print(f"  有人物: {has_person} ({has_person/total*100:.1f}%)")
    print(f"  符合{season_cn}着装: {compliant} ({compliant/total*100:.1f}%)")


if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("人物衣着检测工具")
    print("=" * 60)
    print()
    
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  单张图片: python scripts/detect_clothing.py image.jpg [--season summer/winter]")
        print("  批量检测: python scripts/detect_clothing.py --batch image_folder/ [--season summer/winter]")
        print("\n示例:")
        print("  python scripts/detect_clothing.py photo.jpg --season summer")
        print("  python scripts/detect_clothing.py --batch dataset/source/ --season winter")
        sys.exit(1)
    
    # 解析季节参数
    expected_season = "summer"  # 默认夏季
    if "--season" in sys.argv:
        season_idx = sys.argv.index("--season")
        if season_idx + 1 < len(sys.argv):
            season_arg = sys.argv[season_idx + 1].lower()
            if season_arg in ["winter", "冬季", "冬"]:
                expected_season = "winter"
    
    if sys.argv[1] == "--batch":
        if len(sys.argv) < 3:
            print("请指定图片目录")
            sys.exit(1)
        batch_check_clothing(sys.argv[2], expected_season)
    else:
        # 单张图片
        detector = ClothingDetector()
        image_path = sys.argv[1]
        
        season_cn = "夏季" if expected_season == "summer" else "冬季"
        print(f"检测图片: {image_path}")
        print(f"期望季节: {season_cn}")
        
        result = detector.check_season_compliance(image_path, expected_season)
        
        print("\n检测结果:")
        print(f"  {result['message']}")
        
        if result['has_person']:
            for i, person in enumerate(result['persons'], 1):
                print(f"\n  人物 {i}:")
                print(f"    检测季节: {person['detected_season']}")
                print(f"    置信度: {person['season_confidence']:.2%}")
                print(f"    是否短袖: {'是' if person['details']['is_short_sleeve'] else '否'}")
                print(f"    符合{season_cn}: {'是' if person['compliant'] else '否'}")
        
        # 显示结果
        detector.visualize_results(image_path, expected_season)

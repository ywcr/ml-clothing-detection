#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版人物衣着季节检测工具

新功能：
1. 根据当前月份自动判断期望季节
2. 支持换季容差（默认20%）
3. 更精确的季节判断算法
4. 批量检测报告
"""

from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import csv
from datetime import datetime
from tqdm import tqdm


class SeasonClothingDetector:
    """季节服装检测器"""
    
    # 季节定义（中国标准）
    SEASON_MAP = {
        'spring': {'months': [3, 4, 5], 'name': '春季', 'expected_clothing': 'light'},
        'summer': {'months': [6, 7, 8], 'name': '夏季', 'expected_clothing': 'summer'},
        'autumn': {'months': [9, 10, 11], 'name': '秋季', 'expected_clothing': 'light'},
        'winter': {'months': [12, 1, 2], 'name': '冬季', 'expected_clothing': 'winter'}
    }
    
    def __init__(self, tolerance=0.2):
        """
        初始化检测器
        
        Args:
            tolerance: 容差比例（0.2 = 20%），允许一定比例的人员不符合季节要求
        """
        self.person_model = YOLO('yolov8n.pt')
        self.tolerance = tolerance
        
    @staticmethod
    def get_current_season(month=None):
        """
        根据月份获取当前季节
        
        Args:
            month: 月份（1-12），如果为None则使用当前月份
            
        Returns:
            tuple: (season_key, season_info)
        """
        if month is None:
            month = datetime.now().month
            
        for season_key, season_info in SeasonClothingDetector.SEASON_MAP.items():
            if month in season_info['months']:
                return season_key, season_info
                
        return 'summer', SeasonClothingDetector.SEASON_MAP['summer']
    
    @staticmethod
    def is_transition_period(month=None):
        """
        判断是否为换季时期（春秋季或季节交替月份）
        
        换季月份：3月、5月、9月、11月
        """
        if month is None:
            month = datetime.now().month
            
        # 春秋季或季节首尾月
        transition_months = [3, 5, 9, 11]
        return month in transition_months
    
    def detect_persons(self, image_path, conf=0.5):
        """检测图片中的人物"""
        results = self.person_model.predict(image_path, conf=conf, classes=[0])
        
        persons = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                persons.append((int(x1), int(y1), int(x2), int(y2), float(confidence)))
        
        return persons
    
    def calculate_skin_exposure(self, region):
        """
        计算皮肤暴露程度
        """
        if region.size == 0:
            return 0
            
        # 转换为HSV色彩空间
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # 多个肤色范围以适应不同肤色
        skin_ranges = [
            ((0, 20, 70), (20, 150, 255)),    # 浅肤色
            ((0, 30, 60), (30, 170, 255)),    # 中等肤色
            ((0, 15, 50), (25, 180, 255)),    # 扩展范围
        ]
        
        skin_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in skin_ranges:
            mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), 
                             np.array(upper, dtype=np.uint8))
            skin_mask = cv2.bitwise_or(skin_mask, mask)
        
        # 计算肤色比例
        total_pixels = region.shape[0] * region.shape[1]
        skin_pixels = np.count_nonzero(skin_mask)
        skin_ratio = skin_pixels / total_pixels if total_pixels > 0 else 0
        
        return skin_ratio
    
    def calculate_brightness(self, region):
        """计算区域亮度"""
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)
    
    def analyze_clothing_features(self, image, person_box):
        """
        分析服装特征
        
        Returns:
            dict: 服装特征信息
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        x1, y1, x2, y2 = person_box[:4]
        person_region = image[y1:y2, x1:x2]
        
        height = y2 - y1
        width = x2 - x1
        
        # 分析上半身（躯干 + 手臂）
        upper_body_y2 = y1 + int(height * 0.65)
        upper_body = image[y1:upper_body_y2, x1:x2]
        
        # 分析手臂区域（判断短袖/长袖）
        arm_y1 = y1 + int(height * 0.25)  # 肩部以下
        arm_y2 = y1 + int(height * 0.65)  # 腰部以上
        
        # 左右手臂区域
        left_arm_x2 = x1 + int(width * 0.25)
        right_arm_x1 = x1 + int(width * 0.75)
        
        left_arm = image[arm_y1:arm_y2, x1:left_arm_x2]
        right_arm = image[arm_y1:arm_y2, right_arm_x1:x2]
        
        # 计算特征
        left_skin = self.calculate_skin_exposure(left_arm) if left_arm.size > 0 else 0
        right_skin = self.calculate_skin_exposure(right_arm) if right_arm.size > 0 else 0
        avg_skin_ratio = (left_skin + right_skin) / 2
        
        # 上半身整体皮肤暴露
        upper_skin_ratio = self.calculate_skin_exposure(upper_body)
        
        # 服装亮度
        brightness = self.calculate_brightness(upper_body)
        
        # 判断短袖：手臂皮肤暴露较高
        is_short_sleeve = avg_skin_ratio > 0.15
        
        # 判断无袖/背心：整体皮肤暴露更高
        is_sleeveless = upper_skin_ratio > 0.25
        
        # 判断浅色衣服
        is_light_colored = brightness > 130
        
        return {
            'arm_skin_exposure': avg_skin_ratio,
            'upper_skin_exposure': upper_skin_ratio,
            'brightness': brightness,
            'is_short_sleeve': is_short_sleeve,
            'is_sleeveless': is_sleeveless,
            'is_light_colored': is_light_colored
        }
    
    def classify_clothing_season(self, features):
        """
        根据服装特征分类季节
        
        Returns:
            tuple: (season, confidence)
            - season: 'summer' 或 'winter'
            - confidence: 0-1之间的置信度
        """
        summer_score = 0
        winter_score = 0
        
        # 1. 皮肤暴露评分（权重最高）
        if features['is_sleeveless']:
            summer_score += 0.5
        elif features['is_short_sleeve']:
            summer_score += 0.35
            winter_score += 0.05
        elif features['arm_skin_exposure'] < 0.08:
            winter_score += 0.4
        else:
            summer_score += 0.15
            winter_score += 0.15
        
        # 2. 整体皮肤暴露评分
        if features['upper_skin_exposure'] > 0.2:
            summer_score += 0.25
        elif features['upper_skin_exposure'] < 0.1:
            winter_score += 0.25
        else:
            summer_score += 0.1
            winter_score += 0.1
        
        # 3. 颜色亮度评分
        if features['is_light_colored']:
            summer_score += 0.15
            winter_score += 0.05
        else:
            summer_score += 0.05
            winter_score += 0.2
        
        # 归一化
        total = summer_score + winter_score
        if total > 0:
            summer_score /= total
            winter_score /= total
        
        if summer_score > winter_score:
            return 'summer', summer_score
        else:
            return 'winter', winter_score
    
    def check_compliance(self, image_path, expected_season=None, current_month=None):
        """
        检查图片中人员衣着是否符合季节要求
        
        Args:
            image_path: 图片路径
            expected_season: 期望季节，如果为None则根据月份自动判断
            current_month: 当前月份，如果为None则使用系统当前月份
            
        Returns:
            dict: 检测结果
        """
        # 自动判断季节
        if expected_season is None:
            season_key, season_info = self.get_current_season(current_month)
            expected_season = season_info['expected_clothing']
            season_name = season_info['name']
        else:
            season_name = '夏季' if expected_season == 'summer' else '冬季'
        
        # 判断是否为换季期
        is_transition = self.is_transition_period(current_month)
        
        # 检测人物
        persons = self.detect_persons(image_path)
        
        if not persons:
            return {
                'image': str(image_path),
                'has_person': False,
                'person_count': 0,
                'expected_season': expected_season,
                'season_name': season_name,
                'is_transition': is_transition,
                'compliant': None,
                'message': '未检测到人员'
            }
        
        # 加载图片
        image = cv2.imread(str(image_path))
        
        # 分析每个人
        person_results = []
        for i, person in enumerate(persons):
            features = self.analyze_clothing_features(image, person)
            detected_season, confidence = self.classify_clothing_season(features)
            
            is_compliant = (detected_season == expected_season)
            
            person_results.append({
                'person_id': i + 1,
                'bbox': person[:4],
                'detection_conf': person[4],
                'detected_season': detected_season,
                'detected_season_cn': '夏装' if detected_season == 'summer' else '冬装',
                'season_confidence': confidence,
                'compliant': is_compliant,
                'features': features
            })
        
        # 计算合规率
        total_persons = len(person_results)
        compliant_persons = sum(1 for p in person_results if p['compliant'])
        compliance_rate = compliant_persons / total_persons if total_persons > 0 else 0
        
        # 判断整体合规性（考虑容差）
        if is_transition:
            # 换季期，容差更宽松
            adjusted_tolerance = self.tolerance * 1.5
        else:
            adjusted_tolerance = self.tolerance
        
        overall_compliant = compliance_rate >= (1 - adjusted_tolerance)
        
        return {
            'image': str(image_path),
            'has_person': True,
            'person_count': total_persons,
            'expected_season': expected_season,
            'season_name': season_name,
            'is_transition': is_transition,
            'persons': person_results,
            'compliant_count': compliant_persons,
            'compliance_rate': compliance_rate,
            'tolerance': adjusted_tolerance,
            'overall_compliant': overall_compliant,
            'message': self._generate_message(total_persons, compliant_persons, 
                                             season_name, is_transition, overall_compliant)
        }
    
    def _generate_message(self, total, compliant, season_name, is_transition, overall_compliant):
        """生成检测消息"""
        transition_note = "（换季期）" if is_transition else ""
        
        if total == compliant:
            return f"检测到{total}人，全部符合{season_name}着装要求{transition_note}"
        elif overall_compliant:
            return f"检测到{total}人，{compliant}人符合{season_name}着装，在容差范围内{transition_note}"
        else:
            return f"检测到{total}人，仅{compliant}人符合{season_name}着装，不合规{transition_note}"
    
    def visualize_results(self, image_path, expected_season=None, current_month=None, 
                         output_path=None, show_features=False):
        """
        可视化检测结果
        
        Args:
            image_path: 图片路径
            expected_season: 期望季节
            current_month: 当前月份
            output_path: 输出路径，如果为None则显示
            show_features: 是否显示详细特征信息
        """
        result = self.check_compliance(image_path, expected_season, current_month)
        
        if not result['has_person']:
            print(result['message'])
            return result
        
        # 加载图片
        image = cv2.imread(str(image_path))
        
        # 绘制每个人的检测框
        for person in result['persons']:
            x1, y1, x2, y2 = person['bbox']
            detected = person['detected_season_cn']
            compliant = person['compliant']
            confidence = person['season_confidence']
            
            # 颜色：绿色=符合，红色=不符合
            color = (0, 255, 0) if compliant else (0, 0, 255)
            
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # 标签
            label = f"P{person['person_id']}: {detected}"
            status = "OK" if compliant else "NG"
            
            cv2.putText(image, label, (x1, y1 - 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(image, f"{status} ({confidence:.0%})", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # 显示特征信息
            if show_features:
                features = person['features']
                feat_text = f"Skin:{features['arm_skin_exposure']:.2f} Bright:{features['brightness']:.0f}"
                cv2.putText(image, feat_text, (x1, y2 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 添加整体状态
        status_color = (0, 255, 0) if result['overall_compliant'] else (0, 0, 255)
        status_text = f"{result['season_name']} | {result['message']}"
        cv2.putText(image, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # 添加统计信息
        stats_text = f"Compliant: {result['compliant_count']}/{result['person_count']} ({result['compliance_rate']:.0%})"
        cv2.putText(image, stats_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if output_path:
            cv2.imwrite(str(output_path), image)
            print(f"结果已保存: {output_path}")
        else:
            cv2.imshow('Season Compliance Check', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return result


def batch_check_season_compliance(image_dir, expected_season=None, current_month=None,
                                  tolerance=0.2, output_csv=None, visualize=False):
    """
    批量检测图片季节合规性
    
    Args:
        image_dir: 图片目录
        expected_season: 期望季节，None表示自动判断
        current_month: 当前月份，None表示使用系统月份
        tolerance: 容差比例
        output_csv: CSV报告输出路径
        visualize: 是否保存可视化结果
    """
    detector = SeasonClothingDetector(tolerance=tolerance)
    image_dir = Path(image_dir)
    
    # 获取所有图片
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")) + \
                  list(image_dir.glob("*.jpeg")) + list(image_dir.glob("*.JPG"))
    
    if len(image_files) == 0:
        print(f"错误: 在 {image_dir} 中未找到图片")
        return
    
    # 准备输出目录
    if visualize:
        output_dir = image_dir / 'season_check_results'
        output_dir.mkdir(exist_ok=True)
    
    # 批量检测
    results = []
    print(f"\n{'='*70}")
    print(f"开始批量检测 {len(image_files)} 张图片")
    
    if expected_season is None:
        season_key, season_info = detector.get_current_season(current_month)
        print(f"当前季节: {season_info['name']} (自动判断)")
    else:
        season_name = '夏季' if expected_season == 'summer' else '冬季'
        print(f"期望季节: {season_name} (手动指定)")
    
    print(f"容差设置: {tolerance*100:.0f}%")
    print(f"{'='*70}\n")
    
    for img_path in tqdm(image_files, desc="处理中"):
        try:
            result = detector.check_compliance(img_path, expected_season, current_month)
            results.append(result)
            
            # 可视化
            if visualize and result['has_person']:
                output_path = output_dir / f"{img_path.stem}_result.jpg"
                detector.visualize_results(img_path, expected_season, current_month, 
                                         output_path=output_path)
        except Exception as e:
            print(f"\n处理失败 {img_path.name}: {e}")
            results.append({
                'image': str(img_path),
                'has_person': False,
                'error': str(e)
            })
    
    # 统计
    total = len(results)
    has_person = sum(1 for r in results if r.get('has_person'))
    overall_compliant = sum(1 for r in results if r.get('overall_compliant'))
    no_person = total - has_person
    non_compliant = has_person - overall_compliant
    
    print(f"\n{'='*70}")
    print(f"批量检测完成")
    print(f"{'='*70}")
    print(f"总图片数: {total}")
    print(f"  有人员: {has_person} ({has_person/total*100:.1f}%)")
    print(f"    合规: {overall_compliant} ({overall_compliant/has_person*100:.1f}%)" if has_person > 0 else "    合规: 0")
    print(f"    不合规: {non_compliant}")
    print(f"  无人员: {no_person}")
    print(f"{'='*70}\n")
    
    # 保存CSV报告
    if output_csv:
        with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['图片文件', '人数', '期望季节', '换季期', '合规人数', 
                           '合规率', '整体合规', '备注'])
            
            for result in results:
                if result.get('has_person'):
                    writer.writerow([
                        Path(result['image']).name,
                        result['person_count'],
                        result['season_name'],
                        '是' if result['is_transition'] else '否',
                        result['compliant_count'],
                        f"{result['compliance_rate']*100:.1f}%",
                        '是' if result['overall_compliant'] else '否',
                        result['message']
                    ])
                else:
                    writer.writerow([
                        Path(result['image']).name,
                        0,
                        '-',
                        '-',
                        0,
                        '0%',
                        '-',
                        result.get('message', result.get('error', '未知错误'))
                    ])
        
        print(f"CSV报告已保存: {output_csv}\n")
    
    if visualize:
        print(f"可视化结果已保存至: {output_dir}\n")
    
    return results


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='增强版人物衣着季节检测工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 单张图片（自动判断当前季节）
  python scripts/detect_clothing_enhanced.py --image photo.jpg
  
  # 指定季节
  python scripts/detect_clothing_enhanced.py --image photo.jpg --season summer
  
  # 指定月份（自动判断季节）
  python scripts/detect_clothing_enhanced.py --image photo.jpg --month 3
  
  # 批量检测
  python scripts/detect_clothing_enhanced.py --batch images/ --visualize --output report.csv
  
  # 调整容差
  python scripts/detect_clothing_enhanced.py --batch images/ --tolerance 0.3
        """
    )
    
    parser.add_argument('--image', type=str, help='单张图片路径')
    parser.add_argument('--batch', type=str, help='图片文件夹路径（批量处理）')
    parser.add_argument('--season', type=str, choices=['summer', 'winter'], 
                       help='期望季节（不指定则自动判断）')
    parser.add_argument('--month', type=int, choices=range(1, 13), 
                       help='指定月份（1-12，用于自动判断季节）')
    parser.add_argument('--tolerance', type=float, default=0.2, 
                       help='容差比例（默认0.2即20%%）')
    parser.add_argument('--visualize', action='store_true', help='保存可视化结果')
    parser.add_argument('--output', type=str, help='CSV报告输出路径')
    parser.add_argument('--show-features', action='store_true', help='显示详细特征信息')
    
    args = parser.parse_args()
    
    if args.image:
        # 单张图片检测
        detector = SeasonClothingDetector(tolerance=args.tolerance)
        
        print(f"\n{'='*70}")
        print(f"图片: {args.image}")
        
        result = detector.visualize_results(
            args.image, 
            expected_season=args.season,
            current_month=args.month,
            show_features=args.show_features
        )
        
        print(f"\n检测结果:")
        print(f"  季节: {result['season_name']}")
        print(f"  人数: {result.get('person_count', 0)}")
        
        if result['has_person']:
            print(f"  合规人数: {result['compliant_count']}")
            print(f"  合规率: {result['compliance_rate']:.0%}")
            print(f"  整体合规: {'是' if result['overall_compliant'] else '否'}")
            print(f"  {result['message']}")
            
            for person in result['persons']:
                print(f"\n  人员 {person['person_id']}:")
                print(f"    检测季节: {person['detected_season_cn']}")
                print(f"    置信度: {person['season_confidence']:.0%}")
                print(f"    合规: {'是' if person['compliant'] else '否'}")
        else:
            print(f"  {result['message']}")
        
        print(f"{'='*70}\n")
    
    elif args.batch:
        # 批量检测
        batch_check_season_compliance(
            args.batch,
            expected_season=args.season,
            current_month=args.month,
            tolerance=args.tolerance,
            output_csv=args.output,
            visualize=args.visualize
        )
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

# ML Clothing Season Detection

机器学习服装季节检测模型训练项目

## 📋 项目概述

本项目用于训练和部署服装季节检测模型，能够自动识别图片中人员的衣着是否符合当前季节要求。

### 核心功能
- 🤖 **人物检测**: 基于YOLOv8检测图片中的人物
- 👕 **服装分析**: 分析皮肤暴露度、服装亮度等特征
- 🌡️ **季节判断**: 自动判断夏装/冬装
- 🎯 **容差机制**: 支持换季期20%容差
- 📊 **批量处理**: 支持大规模图片检测

---

## 🏗️ 项目结构

```
ml-clothing-detection/
├── data/                      # 数据集
│   ├── raw/                  # 原始图片
│   ├── processed/            # 预处理后的数据
│   └── annotations/          # 标注数据
├── models/                    # 训练好的模型
│   ├── yolov8n.pt           # 预训练模型
│   └── clothing_v*.pt        # 训练版本
├── notebooks/                 # Jupyter实验
│   ├── 01_data_exploration.ipynb
│   ├── 02_training.ipynb
│   └── 03_evaluation.ipynb
├── src/                       # 核心代码
│   ├── detect_clothing_enhanced.py  # 检测器
│   ├── train.py              # 训练脚本
│   ├── evaluate.py           # 评估脚本
│   └── utils.py              # 工具函数
├── api/                       # API服务
│   └── clothing_season_api.py
├── configs/                   # 配置文件
│   ├── default.yaml
│   └── training.yaml
├── scripts/                   # 辅助脚本
│   ├── auto_annotate.py
│   ├── split_dataset.py
│   └── generate_watermarks.py
└── docs/                      # 文档
    └── TRAINING_GUIDE.md
```

---

## 🚀 快速开始

### 环境要求
- Python 3.8+
- CUDA 11.8+ (GPU训练，可选)
- 8GB+ RAM
- 20GB+ 磁盘空间

### 安装

```bash
# 克隆仓库
git clone https://github.com/ywcr/ml-clothing-detection.git
cd ml-clothing-detection

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 快速测试

```bash
# 单张图片检测
python src/detect_clothing_enhanced.py --image test.jpg

# 批量检测
python src/detect_clothing_enhanced.py --batch images/ --output report.csv

# 指定季节
python src/detect_clothing_enhanced.py --batch images/ --season summer --tolerance 0.2
```

---

## 📚 使用指南

### 1. 数据准备

```bash
# 准备图片数据
mkdir -p data/raw/summer data/raw/winter

# 复制图片到对应文件夹
cp summer_photos/* data/raw/summer/
cp winter_photos/* data/raw/winter/
```

### 2. 训练模型

```bash
# 使用默认配置训练
python src/train.py --config configs/default.yaml

# 自定义训练
python src/train.py \
  --data data/processed \
  --epochs 100 \
  --batch-size 16 \
  --model yolov8n
```

### 3. 评估模型

```bash
# 评估准确率
python src/evaluate.py --model models/clothing_v1.pt --test-dir data/test/

# 生成评估报告
python src/evaluate.py --model models/clothing_v1.pt --report
```

### 4. 部署API

```bash
# 启动API服务
python api/clothing_season_api.py

# 服务地址: http://localhost:8000
# API文档: http://localhost:8000/docs
```

---

## 🔧 配置说明

### 季节定义

| 季节 | 月份 | 期望着装 | 换季月份 |
|------|------|---------|----------|
| 冬季 | 12, 1, 2 | 长袖/外套 | - |
| 春季 | 3, 4, 5 | 轻便混合 | 3月、5月 |
| 夏季 | 6, 7, 8 | 短袖/无袖 | - |
| 秋季 | 9, 10, 11 | 轻便混合 | 9月、11月 |

### 检测参数

```yaml
# configs/default.yaml
detector:
  model: yolov8n.pt
  confidence: 0.5
  tolerance: 0.2
  
season:
  auto_detect: true
  transition_tolerance_multiplier: 1.5
  
features:
  skin_exposure_threshold: 0.15
  brightness_threshold: 130
```

---

## 📊 模型版本

| 版本 | 日期 | 准确率 | 说明 |
|------|------|--------|------|
| v1.0.0 | 2025-01 | 88% | YOLOv8n基础模型 |
| v1.1.0 | 2025-02 | 90% | 增加换季期检测 |
| v1.2.0 | 2025-03 | 92% | 优化皮肤检测算法 |

---

## 🔗 关联项目

- **生产应用**: [excel-review-app](https://github.com/ywcr/excel-review-app)
- **API部署**: [Railway](https://railway.app) (待部署)

---

## 📖 文档

- [训练指南](docs/TRAINING_GUIDE.md)
- [API文档](http://localhost:8000/docs)
- [数据标注指南](docs/ANNOTATION_GUIDE.md)

---

## 🤝 贡献

欢迎贡献代码、报告问题或提出建议！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启 Pull Request

---

## 📄 许可证

本项目遵循 MIT 许可证。

---

## 👥 作者

- 维护者: hida
- 邮箱: hida@whitesand.online

---

## 🙏 致谢

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [FastAPI](https://fastapi.tiangolo.com/)

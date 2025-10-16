#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
服装季节检测 API 服务
提供RESTful API供前端validation-worker调用
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sys
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent / 'scripts'))

# 导入检测器
try:
    # 尝试导入增强版检测器
    from detect_clothing_enhanced import SeasonClothingDetector
    DETECTOR_VERSION = "enhanced"
except ImportError:
    # 回退到基础版
    from detect_clothing import ClothingDetector as SeasonClothingDetector
    DETECTOR_VERSION = "basic"

app = FastAPI(
    title="服装季节检测API",
    description="提供人物服装季节合规性检测服务",
    version="1.0.0"
)

# CORS配置（允许前端调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局检测器实例（减少初始化开销）
detector = SeasonClothingDetector(tolerance=0.2)


@app.get("/")
async def root():
    """健康检查端点"""
    return {
        "service": "clothing_season_detection",
        "status": "running",
        "version": DETECTOR_VERSION,
        "current_month": datetime.now().month
    }


@app.get("/api/season/current")
async def get_current_season(month: int = None):
    """
    获取当前季节信息
    
    参数:
        month: 可选，指定月份（1-12），默认为当前月份
    """
    try:
        season_key, season_info = detector.get_current_season(month)
        is_transition = detector.is_transition_period(month)
        
        return {
            "success": True,
            "data": {
                "season_key": season_key,
                "season_name": season_info['name'],
                "expected_clothing": season_info['expected_clothing'],
                "is_transition_period": is_transition,
                "month": month or datetime.now().month
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/detect/single")
async def detect_single_image(
    image: UploadFile = File(...),
    expected_season: str = Form(None),
    month: int = Form(None),
    tolerance: float = Form(0.2)
):
    """
    单张图片服装季节检测
    
    参数:
        image: 图片文件
        expected_season: 期望季节 ('summer' 或 'winter')，不指定则自动判断
        month: 月份（1-12），用于自动判断季节
        tolerance: 容差比例（默认0.2即20%）
    """
    try:
        # 读取图片
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="无法解析图片")
        
        # 临时保存（或直接处理）
        temp_path = f"/tmp/temp_image_{datetime.now().timestamp()}.jpg"
        cv2.imwrite(temp_path, img)
        
        # 创建临时检测器实例（使用指定容差）
        temp_detector = SeasonClothingDetector(tolerance=tolerance)
        
        # 执行检测
        result = temp_detector.check_compliance(
            temp_path,
            expected_season=expected_season,
            current_month=month
        )
        
        # 清理临时文件
        Path(temp_path).unlink(missing_ok=True)
        
        # 格式化返回结果
        response_data = {
            "success": True,
            "data": {
                "has_person": result['has_person'],
                "person_count": result.get('person_count', 0),
                "season_name": result.get('season_name'),
                "expected_season": result.get('expected_season'),
                "is_transition_period": result.get('is_transition', False),
                "compliant_count": result.get('compliant_count', 0),
                "compliance_rate": result.get('compliance_rate', 0),
                "overall_compliant": result.get('overall_compliant'),
                "message": result.get('message'),
                "tolerance": tolerance
            }
        }
        
        # 添加详细信息（如果有人）
        if result['has_person'] and result.get('persons'):
            response_data['data']['persons'] = [
                {
                    "person_id": p['person_id'],
                    "detected_season": p['detected_season'],
                    "detected_season_cn": p['detected_season_cn'],
                    "season_confidence": float(p['season_confidence']),
                    "compliant": p['compliant']
                }
                for p in result['persons']
            ]
        
        return response_data
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "message": "服装检测失败"
            }
        )


@app.post("/api/detect/base64")
async def detect_base64_image(
    image_base64: str = Form(...),
    expected_season: str = Form(None),
    month: int = Form(None),
    tolerance: float = Form(0.2)
):
    """
    Base64编码图片的服装季节检测（适合前端直接调用）
    
    参数:
        image_base64: Base64编码的图片数据
        expected_season: 期望季节
        month: 月份
        tolerance: 容差
    """
    try:
        # 解码Base64
        if ',' in image_base64:
            # 处理data:image/jpeg;base64,xxxxx格式
            image_base64 = image_base64.split(',')[1]
        
        img_data = base64.b64decode(image_base64)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="无法解析Base64图片")
        
        # 临时保存
        temp_path = f"/tmp/temp_b64_{datetime.now().timestamp()}.jpg"
        cv2.imwrite(temp_path, img)
        
        # 检测
        temp_detector = SeasonClothingDetector(tolerance=tolerance)
        result = temp_detector.check_compliance(
            temp_path,
            expected_season=expected_season,
            current_month=month
        )
        
        # 清理
        Path(temp_path).unlink(missing_ok=True)
        
        # 返回结果（格式同上）
        response_data = {
            "success": True,
            "data": {
                "has_person": result['has_person'],
                "person_count": result.get('person_count', 0),
                "season_name": result.get('season_name'),
                "expected_season": result.get('expected_season'),
                "is_transition_period": result.get('is_transition', False),
                "compliant_count": result.get('compliant_count', 0),
                "compliance_rate": result.get('compliance_rate', 0),
                "overall_compliant": result.get('overall_compliant'),
                "message": result.get('message'),
                "tolerance": tolerance
            }
        }
        
        if result['has_person'] and result.get('persons'):
            response_data['data']['persons'] = [
                {
                    "person_id": p['person_id'],
                    "detected_season": p['detected_season'],
                    "detected_season_cn": p['detected_season_cn'],
                    "season_confidence": float(p['season_confidence']),
                    "compliant": p['compliant']
                }
                for p in result['persons']
            ]
        
        return response_data
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "message": "Base64图片检测失败"
            }
        )


@app.post("/api/detect/batch")
async def detect_batch_images(
    images: list[UploadFile] = File(...),
    expected_season: str = Form(None),
    month: int = Form(None),
    tolerance: float = Form(0.2)
):
    """
    批量图片检测
    
    参数:
        images: 多个图片文件
        expected_season: 期望季节
        month: 月份
        tolerance: 容差
    """
    results = []
    temp_detector = SeasonClothingDetector(tolerance=tolerance)
    
    for idx, image_file in enumerate(images):
        try:
            contents = await image_file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                results.append({
                    "index": idx,
                    "filename": image_file.filename,
                    "success": False,
                    "error": "无法解析图片"
                })
                continue
            
            temp_path = f"/tmp/batch_{idx}_{datetime.now().timestamp()}.jpg"
            cv2.imwrite(temp_path, img)
            
            result = temp_detector.check_compliance(
                temp_path,
                expected_season=expected_season,
                current_month=month
            )
            
            Path(temp_path).unlink(missing_ok=True)
            
            results.append({
                "index": idx,
                "filename": image_file.filename,
                "success": True,
                "has_person": result['has_person'],
                "person_count": result.get('person_count', 0),
                "compliant_count": result.get('compliant_count', 0),
                "overall_compliant": result.get('overall_compliant'),
                "message": result.get('message')
            })
            
        except Exception as e:
            results.append({
                "index": idx,
                "filename": image_file.filename,
                "success": False,
                "error": str(e)
            })
    
    # 统计
    total = len(results)
    successful = sum(1 for r in results if r.get('success'))
    has_person = sum(1 for r in results if r.get('has_person'))
    compliant = sum(1 for r in results if r.get('overall_compliant'))
    
    return {
        "success": True,
        "summary": {
            "total": total,
            "successful": successful,
            "has_person": has_person,
            "compliant": compliant,
            "compliance_rate": compliant / has_person if has_person > 0 else 0
        },
        "results": results
    }


if __name__ == "__main__":
    import uvicorn
    
    print(f"""
╔══════════════════════════════════════════╗
║   服装季节检测 API 服务                  ║
║   Clothing Season Detection API          ║
╠══════════════════════════════════════════╣
║   版本: {DETECTOR_VERSION.upper():<32} ║
║   端口: 8000                             ║
║   文档: http://localhost:8000/docs       ║
╚══════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

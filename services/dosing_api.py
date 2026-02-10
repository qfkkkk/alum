# -*- encoding: utf-8 -*-
"""
投药优化API服务
功能：提供HTTP接口，供外部系统调用投药优化服务

路由列表：
    POST /alum_dosing/predict_turbidity   - 预测出水浊度
    POST /alum_dosing/optimize_dosing     - 获取最优投药量
    POST /alum_dosing/full_optimization   - 完整优化流程
    GET  /alum_dosing/health              - 健康检查
"""
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request

from .dosing_pipeline import DosingPipeline
from ..predictors import create_predictor
from ..optimizers import create_optimizer

# 复用data模块的数据读取和写入功能
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from data.data_factory import load_agg_data, upload_recommend_message
from utils.logger import Logger
from utils.read_config import get_config

# 全局配置
time_format = '%Y-%m-%d %H:%M:%S'
logger = Logger()
app = Flask(__name__)

# 模块级单例（延迟初始化）
_pipeline = None


def get_pipeline() -> DosingPipeline:
    """
    获取管道单例
    
    返回：
        DosingPipeline: 管道实例
    
    说明：
        使用单例模式避免重复加载模型
    """
    global _pipeline
    if _pipeline is None:
        _pipeline = DosingPipeline()
    return _pipeline


def data_read() -> pd.DataFrame:
    """
    读取投药相关数据点位
    
    返回：
        pd.DataFrame: 包含以下列的数据
            - datetime: 时间戳
            - turbidity_in: 进水浊度
            - turbidity_out: 出水浊度
            - flow: 进水流量
            - dosing_rate: 当前投药量
            - ph: pH值
            - temperature: 水温
    
    说明：
        从IoT平台读取最近30小时的数据用于预测
        复用 data.data_factory.load_agg_data
    """
    pass


@app.route('/alum_dosing/health', methods=['GET'])
def health_check():
    """
    健康检查接口
    
    返回：
        服务状态信息
    """
    return jsonify({
        'status': 'healthy',
        'service': 'alum_dosing',
        'timestamp': datetime.now().strftime(time_format)
    })


@app.route('/alum_dosing/predict_turbidity', methods=['POST'])
def predict_turbidity_api():
    """
    出水浊度预测API
    
    请求JSON格式：
        {
            "mode": "online" | "agent",
            "data": {...}  // mode=agent时需提供历史数据
        }
    
    返回JSON格式：
        {
            "status": "success",
            "predictions": [
                {"datetime": "2024-01-01 10:00:00", "turbidity_pred": 0.5},
                ...
            ],
            "count": 5,
            "timestamp": "2024-01-01 09:00:00"
        }
    """
    pass


@app.route('/alum_dosing/optimize_dosing', methods=['POST'])
def optimize_dosing_api():
    """
    投药量优化API（需提供预测结果）
    
    请求JSON格式：
        {
            "predicted_turbidity": [
                {"datetime": "...", "turbidity_pred": 0.5},
                ...
            ],
            "current_flow": 1000,
            "current_dosing": 15.0,  // 可选
            "constraints": {          // 可选
                "dosing_min": 5,
                "dosing_max": 50,
                "target_turbidity": 1.0
            }
        }
    
    返回JSON格式：
        {
            "status": "success",
            "optimal_dosing": 15.5,
            "efficiency_score": 0.92,
            "recommendations": [...],
            "timestamp": "2024-01-01 09:00:00"
        }
    """
    pass


@app.route('/alum_dosing/full_optimization', methods=['POST'])
def full_optimization_api():
    """
    完整优化流程API (预测浊度 + 计算最优投药量)
    
    请求JSON格式：
        {
            "mode": "online" | "agent",
            "data": {...},           // mode=agent时需提供
            "current_flow": 1000,
            "current_dosing": 15.0,  // 可选
            "constraints": {...}     // 可选
        }
    
    返回JSON格式：
        {
            "status": "success",
            "turbidity_predictions": [...],
            "optimal_dosing": 15.5,
            "recommendations": [...],
            "timestamp": "2024-01-01 09:00:00"
        }
    
    说明：
        这是主要的对外接口，一次调用完成：
        1. 读取/接收数据
        2. 预测未来5个时间点的浊度
        3. 计算最优投药量
        4. 返回完整结果
    """
    pass


def run_flask_app(host: str = '0.0.0.0', port: int = 5002):
    """
    运行Flask API服务（阻塞模式）
    
    参数：
        host: 监听地址，默认 0.0.0.0
        port: 监听端口，默认 5002
    
    说明：
        用于独立运行API服务（不含定时任务）
    """
    try:
        logger.info(f"启动投药优化API服务 @ {host}:{port}")
        app.run(host=host, port=port, debug=False, use_reloader=False)
    except Exception:
        error_msg = traceback.format_exc()
        logger.error(f"Flask服务运行异常：\n{error_msg}")


if __name__ == "__main__":
    run_flask_app()

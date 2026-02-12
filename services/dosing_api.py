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
from predictors import create_predictor
from optimizers import create_optimizer

# 复用data模块的数据读取和写入功能
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from dataio.data_factory import load_agg_data, upload_recommend_message
from utils.logger import Logger

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


import numpy as np

def data_read():
    """
    读取投药相关数据点位 (Mock 实现)
    
    返回：
        data_dict: {pool_name: np.ndarray}
        last_dt: datetime
    """
    # 模拟数据生成
    # 假设配置 seq_len=60, n_features=根据实际配置可能有差异，这里先假设为 15
    # 为了保证能跑通，我们最好从 pipeline 实例获取配置
    pipeline = get_pipeline()
    config = pipeline.predictor_manager.config
    seq_len = config.get('seq_len', 60)
    features = config.get('features', [])
    n_features = len(features) if features else 15
    
    # 构造 inputs
    # 假设启用了 4 个池子
    enabled_pools = ['pool_1', 'pool_2', 'pool_3', 'pool_4']
    
    input_data = np.random.rand(seq_len, n_features).astype(np.float32)
    
    data_dict = {
        pool_name: input_data 
        for pool_name in enabled_pools
    }
    
    last_dt = datetime.now()
    
    return data_dict, last_dt


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
    """
    try:
        pipeline = get_pipeline()
        
        # 1. 获取数据 (Mock)
        data_dict, last_dt = data_read()
        
        # 2. 调用预测
        predictions = pipeline.predict_only(data_dict, last_dt)
        
        # 3. 格式化输出
        # predictions 格式: {'pool_1': {'2024-01-01 10:00': 0.5, ...}}
        formatted_preds = []
        count = 0
        
        for pool_id, time_dict in predictions.items():
            pool_data = {
                'pool_id': pool_id,
                'forecast': []
            }
            # 排序
            sorted_items = sorted(time_dict.items(), key=lambda x: x[0])
            for dt_str, val in sorted_items:
                pool_data['forecast'].append({
                    'datetime': dt_str,
                    'turbidity_pred': round(float(val), 4)
                })
            
            formatted_preds.append(pool_data)
            count += len(time_dict)

        return jsonify({
            "status": "success",
            "predictions": formatted_preds,
            "count": count,
            "timestamp": last_dt.strftime(time_format)
        })
        
    except Exception as e:
        logger.error(f"预测接口异常: {traceback.format_exc()}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/alum_dosing/optimize_dosing', methods=['POST'])
def optimize_dosing_api():
    """
    投药量优化API（需提供预测结果）
    (本接口暂时保留，但逻辑尚未完全实现/测试，TODO)
    """
    return jsonify({
        "status": "development",
        "message": "Endpoint under construction"
    })


@app.route('/alum_dosing/full_optimization', methods=['POST'])
def full_optimization_api():
    """
    完整优化流程API (预测浊度 + 计算最优投药量)
    """
    try:
        pipeline = get_pipeline()
        
        # 1. 获取数据 (Mock)
        data_dict, last_dt = data_read()
        
        # 2. 调用完整流程
        # run 返回: {pool_id: {predictions: ..., recommendations: ...}}
        results = pipeline.run(data_dict, last_dt)
        
        # 3. 格式化输出
        formatted_results = []
        
        for pool_id, res in results.items():
            pool_data = {
                'pool_id': pool_id,
                'turbidity_predictions': [],
                'recommendations': []
            }
            
            # 处理预测值
            preds = res.get('predictions', {})
            if preds:
                sorted_preds = sorted(preds.items(), key=lambda x: x[0])
                pool_data['turbidity_predictions'] = [
                    {'datetime': k, 'value': round(float(v), 4)} 
                    for k, v in sorted_preds
                ]
                
            # 处理推荐值
            recs = res.get('recommendations', {})
            if recs:
                sorted_recs = sorted(recs.items(), key=lambda x: x[0])
                pool_data['recommendations'] = [
                    {'datetime': k, 'value': round(float(v), 2)} 
                    for k, v in sorted_recs
                ]
                
            formatted_results.append(pool_data)
            
        return jsonify({
            "status": "success",
            "results": formatted_results,
            "timestamp": last_dt.strftime(time_format)
        })

    except Exception as e:
        logger.error(f"全流程优化接口异常: {traceback.format_exc()}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


def run_flask_app(host: str = '0.0.0.0', port: int = 5002):
    """
    运行Flask API服务（阻塞模式）
    """
    try:
        logger.info(f"启动投药优化API服务 @ {host}:{port}")
        app.run(host=host, port=port, debug=False, use_reloader=False)
    except Exception:
        error_msg = traceback.format_exc()
        logger.error(f"Flask服务运行异常：\n{error_msg}")


if __name__ == "__main__":
    run_flask_app()

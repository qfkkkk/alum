# -*- encoding: utf-8 -*-
"""
预测器模块
提供浊度预测相关的预测器类
"""
from .base_predictor import BasePredictor
from .turbidity_predictor import TurbidityPredictor


def create_predictor(predictor_type: str, **kwargs):
    """
    预测器工厂函数
    
    参数：
        predictor_type: 预测器类型
            - 'turbidity': 浊度预测器
        **kwargs: 传递给预测器构造函数的参数
    
    返回：
        BasePredictor: 预测器实例
    
    使用示例：
        predictor = create_predictor('turbidity', model_path='model.pth')
    """
    predictors = {
        'turbidity': TurbidityPredictor,
    }
    
    if predictor_type not in predictors:
        raise ValueError(f"未知的预测器类型: {predictor_type}，可用类型: {list(predictors.keys())}")
    
    return predictors[predictor_type](**kwargs)

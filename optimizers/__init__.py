# -*- encoding: utf-8 -*-
"""
优化器模块
提供投药量优化相关的优化器类
"""
from .base_optimizer import BaseOptimizer
from .dosing_optimizer import DosingOptimizer


def create_optimizer(optimizer_type: str, **kwargs):
    """
    优化器工厂函数
    
    参数：
        optimizer_type: 优化器类型
            - 'dosing': 投药量优化器
        **kwargs: 传递给优化器构造函数的参数
    
    返回：
        BaseOptimizer: 优化器实例
    
    使用示例：
        optimizer = create_optimizer('dosing', efficiency_threshold=0.8)
    """
    optimizers = {
        'dosing': DosingOptimizer,
    }
    
    if optimizer_type not in optimizers:
        raise ValueError(f"未知的优化器类型: {optimizer_type}，可用类型: {list(optimizers.keys())}")
    
    return optimizers[optimizer_type](**kwargs)

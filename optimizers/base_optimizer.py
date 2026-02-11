# -*- coding: utf-8 -*-
"""
定义投药优化器的抽象基类。
任何具体的优化器实现（如 APC, MPC, PID 等）都应继承此类。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any

class BaseOptimizer(ABC):
    """
    优化器基类
    
    定义投药优化器的通用接口。
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}

    @abstractmethod
    def optimize(self, 
                 predictions: Dict[str, Dict[Any, float]], 
                 current_features: Dict[str, Dict[str, float]] = None) -> Dict[str, List[float]]:
        """
        执行投药优化计算
        
        参数：
            predictions: 各池子的浊度预测值 (带时间戳)
                格式: {'pool_1': {'2024-01-01 12:00': 0.5, ...}}
            current_features: 各池子的当前全量特征 (t时刻)
                格式: {'pool_1': {'ph': 7.2, 'flow': 2000, ...}, ...}
            
        返回：
            Dict: 推荐投药量
                格式: {'pool_1': [r_t+1, ..., r_t+5]}
        """
        pass

# -*- encoding: utf-8 -*-
"""
优化器抽象基类
定义所有优化器必须实现的接口规范

设计模式：策略模式 (Strategy Pattern)
作用：允许不同优化算法可互换使用
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional
from pathlib import Path
import pandas as pd


class BaseOptimizer(ABC):
    """
    优化器抽象基类
    
    功能说明：
        定义优化器的通用接口，所有具体优化器必须继承此类
    
    接口规范：
        - optimize(): 执行优化，返回最优解
        - get_top_n_solutions(): 返回前N个最优解
    
    使用示例：
        class MyOptimizer(BaseOptimizer):
            def optimize(self, predictions, constraints):
                # 实现优化逻辑
                pass
    """
    
    def __init__(self,
                 model_path: Union[str, Path] = None,
                 efficiency_threshold: float = 0.8):
        """
        初始化基类
        
        参数：
            model_path: 优化模型文件路径（可选）
            efficiency_threshold: 效率阈值，默认0.8
        """
        self.model_path = Path(model_path) if model_path else None
        self.efficiency_threshold = efficiency_threshold
        self.model = None
    
    @abstractmethod
    def optimize(self, 
                 predictions: pd.DataFrame,
                 current_value: float = None,
                 constraints: Dict = None) -> Dict:
        """
        执行优化（抽象方法，子类必须实现）
        
        参数：
            predictions: 预测结果DataFrame
            current_value: 当前值（用于计算切换成本）
            constraints: 约束条件字典
        
        返回：
            Dict: 优化结果
        """
        pass
    
    @abstractmethod
    def get_top_n_solutions(self, 
                            predictions: pd.DataFrame,
                            n: int = 5) -> List[Dict]:
        """
        获取前N个最优解（抽象方法，子类必须实现）
        
        参数：
            predictions: 预测结果
            n: 返回方案数量
        
        返回：
            List[Dict]: 方案列表
        """
        pass
    
    def calculate_switch_cost(self, 
                              current_value: float, 
                              new_value: float) -> float:
        """
        计算切换成本（通用方法）
        
        参数：
            current_value: 当前值
            new_value: 新值
        
        返回：
            float: 切换成本
        """
        if current_value is None:
            return 0.0
        return abs(new_value - current_value)

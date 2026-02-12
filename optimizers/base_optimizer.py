# -*- encoding: utf-8 -*-
"""
优化器抽象基类
定义所有投药优化器必须实现的接口规范

设计模式：模板方法模式
作用：optimize() 定义通用流程骨架，子类只需实现 _optimize_core()
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Any, Optional

import pandas as pd


class BaseOptimizer(ABC):
    """优化器抽象基类"""

    def __init__(self, config: dict, pool_id: str):
        """
        初始化优化器
        
        参数：
            config: 配置字典
            pool_id: 池子ID（字符串，如 'pool_1'）
        """
        self.pool_id = pool_id
        self.config = config
        self.control_horizon = config.get('control_horizon', 5)
        self.prediction_horizon = config.get('prediction_horizon', 6)
        self.lambda_du = config.get('lambda_du', 0.005)
        self.time_step = config.get('time_step', 5)  # 默认5分钟

    def optimize(
        self,
        predictions: Dict[str, Dict[str, float]],
        current_features: Dict[str, Dict[str, float]] = None,
        last_datetime: Optional[datetime] = None,
        **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """
        执行投药优化（模板方法）
        
        参数：
            predictions: {'pool_1': {'2024-01-01 12:00': 0.5, ...}, ...}
            current_features: {'pool_1': {'ph': 7.2, 'flow': 2000, ...}, ...}
            last_datetime: 最后一个数据点的时间（用于生成未来时间戳）
            **kwargs: 子类额外参数
            
        返回：
            {'pool_1': {'2024-01-01 12:05': 5.2, ...}, ...}
        """
        # 1. 验证输入
        self._validate_input(predictions, current_features, **kwargs)
        
        # 2. 数据准备
        prepared_data = self._prepare_input(predictions, current_features, **kwargs)
        
        # 3. 核心优化（子类实现）
        raw_result = self._optimize_core(prepared_data)
        
        # 4. 格式化输出（添加时间戳）
        return self._format_output(raw_result, last_datetime)

    @abstractmethod
    def _optimize_core(self, prepared_data: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        核心优化逻辑（子类必须实现）
        
        参数：
            prepared_data: 由 _prepare_input 返回的数据结构
            
        返回：
            {'pool_1': [dose_t+1, dose_t+2, ...], ...}
        """
        pass

    def _prepare_input(
        self,
        predictions: Dict[str, Dict[str, float]],
        current_features: Dict[str, Dict[str, float]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """准备优化器输入数据"""
        return {
            'predictions': predictions,
            'current_features': current_features,
            **kwargs
        }

    def _validate_input(
        self,
        predictions: Dict[str, Dict[str, float]],
        current_features: Dict[str, Dict[str, float]] = None,
        **kwargs
    ) -> bool:
        """验证输入数据"""
        has_predictions = predictions is not None and len(predictions) > 0
        has_features = current_features is not None and len(current_features) > 0
        has_kwargs = len(kwargs) > 0

        if not (has_predictions or has_features or has_kwargs):
            raise ValueError("必须提供 predictions、current_features 或 kwargs 之一")
        return True

    def _format_output(
        self,
        raw_result: Dict[str, List[float]],
        last_datetime: Optional[datetime] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        为投药量序列添加时间戳
        
        参数：
            raw_result: {'pool_1': [5.2, 5.3, 5.4], ...}
            last_datetime: 最后数据点时间，如果为 None 则返回无时间戳版本
            
        返回：
            {'pool_1': {'2024-01-01 12:05': 5.2, ...}, ...}
        """
        if last_datetime is None:
            # 如果没有提供时间戳，返回列表格式
            return raw_result
        
        result = {}
        for pool_key, dose_list in raw_result.items():
            result[pool_key] = {}
            for i, dose in enumerate(dose_list):
                future_dt = last_datetime + pd.Timedelta(minutes=self.time_step * (i + 1))
                dt_str = future_dt.strftime('%Y-%m-%d %H:%M:%S')
                result[pool_key][dt_str] = round(float(dose), 4)
        
        return result

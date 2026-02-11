# -*- encoding: utf-8 -*-
"""
预测器抽象基类
定义所有预测器必须实现的接口规范

设计模式：策略模式 
作用：允许不同预测算法可互换使用
"""
from abc import ABC, abstractmethod
from typing import Dict, Union, Optional
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


class BasePredictor(ABC):
    """
    预测器抽象基类

    功能说明：
        定义预测器的通用接口，所有具体预测器必须继承此类。
        一个 Predictor 实例对应一个池子（单池视角）。

    接口规范：
        - predict(): 执行预测，返回预测值数组
        - predict_with_timestamps(): 带时间戳的预测
        - _load_model(): 加载模型文件
    """

    def __init__(self, pool_id: int, config: dict):
        """
        初始化基类

        参数：
            pool_id: 池子编号 (1, 2, 3, 4)
            config: 从 predictor.yaml 读入的配置字典
        """
        self.pool_id = pool_id
        self.config = config
        self.model = None
        self.scaler = None
        self.model_config = None

    @abstractmethod
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        执行预测（抽象方法，子类必须实现）

        参数：
            input_data: 输入数据, shape [seq_len, n_features]

        返回：
            np.ndarray: 预测值, shape [pred_len]
        """
        pass

    @abstractmethod
    def _load_model(self):
        """
        加载模型（抽象方法，子类必须实现）
        """
        pass

    def predict_with_timestamps(self,
                                input_data: np.ndarray,
                                last_datetime: datetime,
                                time_interval_minutes: int = 5) -> Dict[str, float]:
        """
        带时间戳的预测

        参数：
            input_data: 输入数据, shape [seq_len, n_features]
            last_datetime: 输入数据最后一个时间点
            time_interval_minutes: 采样间隔（分钟）

        返回：
            Dict[str, float]: {datetime_str: predicted_value, ...}
        """
        predictions = self.predict(input_data)
        result = {}
        for i, val in enumerate(predictions):
            future_dt = last_datetime + pd.Timedelta(minutes=time_interval_minutes * (i + 1))
            dt_str = future_dt.strftime('%Y-%m-%d %H:%M:%S')
            result[dt_str] = round(float(val), 4)
        return result

    def validate_input(self, data: np.ndarray, expected_seq_len: int, expected_features: int) -> bool:
        """
        验证输入数据的形状

        参数：
            data: 输入数据
            expected_seq_len: 期望的时间步长
            expected_features: 期望的特征数

        异常：
            ValueError: 形状不匹配时抛出
        """
        if data.ndim != 2:
            raise ValueError(f"输入数据需要 2D 数组, 实际维度: {data.ndim}")
        if data.shape[0] != expected_seq_len:
            raise ValueError(
                f"输入数据时间步 {data.shape[0]} != 期望步长 {expected_seq_len}")
        if data.shape[1] != expected_features:
            raise ValueError(
                f"输入数据特征数 {data.shape[1]} != 期望特征数 {expected_features}")
        return True

# -*- encoding: utf-8 -*-
"""
预测器抽象基类
定义所有预测器必须实现的接口规范

设计模式：模板方法模式
作用：predict() 定义通用流程骨架，子类只需实现 _load_model() 和 _infer()
"""
from abc import ABC, abstractmethod
from typing import Dict
from datetime import datetime

import numpy as np
import pandas as pd


class BasePredictor(ABC):
    """
    预测器抽象基类

    功能说明：
        定义预测器的通用接口和通用预测流程。
        一个 Predictor 实例对应一个池子（单池视角）。

    通用流程（模板方法）：
        predict(): 验证 → 标准化 → _infer() → 反标准化

    子类需实现：
        - _load_model(): 加载模型文件
        - _infer(): 模型专属推理逻辑（接收标准化后的输入）
    """

    def __init__(self, pool_id: int, config: dict):
        """
        初始化基类

        参数：
            pool_id: 池子编号 (1, 2, 3, 4)
            config: 从 app.yaml 读入的配置字典
        """
        self.pool_id = pool_id
        self.config = config
        self.model = None
        self.scaler = None
        self.model_config = None

        self.seq_len = config.get('seq_len', 60)
        self.pred_len = config.get('pred_len', 6)
        self.features = config.get('features', [])
        self.target_idx = config.get('target_feature_idx', 1)

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        执行预测（模板方法，定义通用流程）

        流程：验证 → 标准化 → _infer() → 反标准化

        参数：
            input_data: 原始空间的输入数据, shape [seq_len, n_features]

        返回：
            np.ndarray: 原始空间的预测值, shape [pred_len]
        """
        n_features = len(self.features)
        self._validate_input(input_data, self.seq_len, n_features)

        # 标准化
        input_scaled = self.scaler.transform(input_data).astype(np.float32)

        # 子类推理（模型专属逻辑）
        pred_target_scaled = self._infer(input_scaled)

        # 反标准化
        mean_val = self.scaler.mean_[self.target_idx]
        std_val = self.scaler.scale_[self.target_idx]
        predictions = pred_target_scaled * std_val + mean_val

        return predictions

    @abstractmethod
    def _infer(self, scaled_input: np.ndarray) -> np.ndarray:
        """
        模型专属推理逻辑（抽象方法，子类必须实现）

        参数：
            scaled_input: 标准化后的输入数据, shape [seq_len, n_features]

        返回：
            np.ndarray: 标准化空间下的目标特征预测值, shape [pred_len]

        说明：
            子类在此处完成模型推理，以及模型专属的后处理（如反差分）。
            返回值仍在标准化空间，反标准化由基类的 predict() 统一完成。
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

    def _validate_input(self, data: np.ndarray, expected_seq_len: int, expected_features: int) -> bool:
        """验证输入数据的形状"""
        if data.ndim != 2:
            raise ValueError(f"输入数据需要 2D 数组, 实际维度: {data.ndim}")
        if data.shape[0] != expected_seq_len:
            raise ValueError(
                f"输入数据时间步 {data.shape[0]} != 期望步长 {expected_seq_len}")
        if data.shape[1] != expected_features:
            raise ValueError(
                f"输入数据特征数 {data.shape[1]} != 期望特征数 {expected_features}")
        return True

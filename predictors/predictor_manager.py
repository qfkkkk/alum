# -*- encoding: utf-8 -*-
"""
多池预测管理器
功能：管理多个 TurbidityPredictor 实例，统一执行预测并拼接结果
输入：各池的历史数据字典
输出：所有池预测结果的 JSON 格式
"""
import yaml
import logging
from typing import Dict, Optional
from pathlib import Path
from datetime import datetime

import numpy as np

from .turbidity_predictor import TurbidityPredictor

logger = logging.getLogger(__name__)


class TurbidityPredictorManager:
    """
    多池预测管理器

    功能说明：
        - 根据配置文件创建并管理多个 TurbidityPredictor 实例
        - 提供 predict_all() 接口一次性预测所有启用的池子
        - 输出统一的 JSON 格式供 DosingOptimizer 使用

    使用示例：
        manager = TurbidityPredictorManager()
        result = manager.predict_all(data_dict, last_datetime)
        # result = {"pool_1": {"2026-02-11 10:00:00": 0.32, ...}, ...}
    """

    def __init__(self, config_path: str = None):
        """
        初始化管理器

        参数：
            config_path: predictor.yaml 路径，默认为 configs/predictor.yaml
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'configs' / 'app.yaml'
        else:
            config_path = Path(config_path)

        self.config = self._load_config(config_path)
        self.predictors: Dict[str, TurbidityPredictor] = {}
        self.time_interval = self.config.get('time_interval_minutes', 5)

        self._init_predictors()

    def _load_config(self, config_path: Path) -> dict:
        """加载 YAML 配置"""
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _init_predictors(self):
        """根据配置初始化各池预测器"""
        pools_config = self.config.get('pools', {})

        for pool_name, pool_cfg in pools_config.items():
            if not pool_cfg.get('enabled', True):
                logger.info(f"跳过未启用的池子: {pool_name}")
                continue

            # 从 pool_name 中提取 pool_id，如 "pool_1" -> 1
            pool_id = int(pool_name.split('_')[1])

            try:
                predictor = TurbidityPredictor(pool_id=pool_id, config=self.config)
                self.predictors[pool_name] = predictor
                logger.info(f"已加载 {pool_name} 预测器")
            except Exception as e:
                logger.error(f"加载 {pool_name} 预测器失败: {e}")
                raise

    def predict_all(self,
                    data_dict: Dict[str, np.ndarray],
                    last_datetime: datetime) -> Dict[str, Dict[str, float]]:
        """
        预测所有启用的池子

        参数：
            data_dict: 各池的输入数据字典
                格式: {"pool_1": ndarray[60, 6], "pool_2": ndarray[60, 6], ...}
            last_datetime: 输入数据最后一个时间点

        返回：
            Dict: 所有池的预测结果
                {
                    "pool_1": {"2026-02-11 10:05:00": 0.32, "2026-02-11 10:10:00": 0.34, ...},
                    "pool_2": {"2026-02-11 10:05:00": 0.28, ...},
                    ...
                }
        """
        results = {}

        for pool_name, predictor in self.predictors.items():
            if pool_name not in data_dict:
                logger.warning(f"未提供 {pool_name} 的输入数据，跳过")
                continue

            input_data = data_dict[pool_name]

            try:
                pred = predictor.predict_with_timestamps(
                    input_data=input_data,
                    last_datetime=last_datetime,
                    time_interval_minutes=self.time_interval
                )
                results[pool_name] = pred
                logger.info(f"{pool_name} 预测完成, 共 {len(pred)} 步")
            except Exception as e:
                logger.error(f"{pool_name} 预测失败: {e}")
                raise

        return results

    def predict_single(self,
                       pool_name: str,
                       input_data: np.ndarray,
                       last_datetime: datetime) -> Dict[str, float]:
        """
        预测单个池子

        参数：
            pool_name: 池子名称，如 "pool_1"
            input_data: 输入数据, shape [seq_len, n_features]
            last_datetime: 输入数据最后一个时间点

        返回：
            Dict[str, float]: {"datetime_str": value, ...}
        """
        if pool_name not in self.predictors:
            raise ValueError(f"未找到池子 {pool_name}, 可用: {list(self.predictors.keys())}")

        return self.predictors[pool_name].predict_with_timestamps(
            input_data=input_data,
            last_datetime=last_datetime,
            time_interval_minutes=self.time_interval
        )

    @property
    def enabled_pools(self):
        """获取已启用的池子列表"""
        return list(self.predictors.keys())

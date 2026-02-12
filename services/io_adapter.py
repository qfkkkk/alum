# -*- encoding: utf-8 -*-
"""
IO 适配层（占位实现）。

用途：
    - read_data: 当前返回模拟输入数据
    - upload_recommend_message: 当前为 no-op，仅记录日志
"""

from datetime import datetime
from typing import Any, Dict, Tuple

import numpy as np

from utils.logger import Logger

logger = Logger()


def read_data(config: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], datetime]:
    """
    读取输入数据（占位实现）。

    参数:
        config: app.yaml 配置字典

    返回:
        data_dict: {pool_id: ndarray[seq_len, n_features]}
        last_dt: 当前时间
    """
    seq_len = config.get("seq_len", 60)
    features = config.get("features", [])
    n_features = len(features) if features else 1

    pools_cfg = config.get("pools", {})
    enabled_pools = [pid for pid, p_cfg in pools_cfg.items() if p_cfg.get("enabled", False)] or ["pool_1"]

    input_data = np.random.rand(seq_len, n_features).astype(np.float32)
    data_dict = {pool_id: input_data for pool_id in enabled_pools}
    return data_dict, datetime.now()


def upload_recommend_message(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    推荐结果上报（占位实现）。

    当前阶段不执行真实上报，仅返回跳过标记。
    """
    _ = payload
    logger.info("[IO Adapter] upload_recommend_message skipped (placeholder)")
    return {"success": True, "skipped": True}

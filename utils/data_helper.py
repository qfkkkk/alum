# -*- encoding: utf-8 -*-
"""
数据处理辅助函数
功能：提供数据验证、转换、重采样等通用功能
"""
import pandas as pd
import numpy as np
from typing import List, Optional


def validate_dataframe(df: pd.DataFrame, 
                       required_columns: List[str],
                       min_rows: int = 1) -> bool:
    """
    验证DataFrame是否满足要求
    
    参数：
        df: 待验证的DataFrame
        required_columns: 必需的列名列表
        min_rows: 最少行数，默认1
    
    返回：
        bool: 验证是否通过
    
    异常：
        ValueError: 验证失败时抛出，包含具体原因
    
    使用示例：
        validate_dataframe(data, ['datetime', 'turbidity'], min_rows=30)
    """
    if df is None:
        raise ValueError("数据为空 (None)")
    
    if df.empty:
        raise ValueError("数据为空 (empty DataFrame)")
    
    if len(df) < min_rows:
        raise ValueError(f"数据行数不足: 需要至少{min_rows}行, 实际{len(df)}行")
    
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"缺少必需列: {missing}")
    
    return True


def resample_data(df: pd.DataFrame, 
                  interval: str = '1H',
                  datetime_col: str = 'datetime',
                  method: str = 'mean') -> pd.DataFrame:
    """
    数据重采样
    
    参数：
        df: 输入数据
        interval: 采样间隔，如 '1H', '30T', '1D'
        datetime_col: 时间列名
        method: 聚合方法，'mean', 'last', 'first'
    
    返回：
        pd.DataFrame: 重采样后的数据
    
    使用示例：
        hourly_data = resample_data(raw_data, interval='1H')
    """
    pass


def fill_missing_values(df: pd.DataFrame,
                        method: str = 'interpolate',
                        limit: int = 3) -> pd.DataFrame:
    """
    缺失值填充
    
    参数：
        df: 输入数据
        method: 填充方法
            - 'interpolate': 插值
            - 'ffill': 前向填充
            - 'mean': 均值填充
        limit: 最大连续填充数量
    
    返回：
        pd.DataFrame: 填充后的数据
    """
    pass


def detect_anomalies(series: pd.Series, 
                     threshold: float = 3.0) -> pd.Series:
    """
    异常值检测（基于Z-score）
    
    参数：
        series: 输入序列
        threshold: Z-score阈值，默认3.0
    
    返回：
        pd.Series: 布尔序列，True表示异常值
    """
    pass

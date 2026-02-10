# -*- encoding: utf-8 -*-
"""
预测器抽象基类
定义所有预测器必须实现的接口规范

设计模式：策略模式 (Strategy Pattern)
作用：允许不同预测算法可互换使用
"""
from abc import ABC, abstractmethod
from typing import Dict, Union
from pathlib import Path
import pandas as pd


class BasePredictor(ABC):
    """
    预测器抽象基类
    
    功能说明：
        定义预测器的通用接口，所有具体预测器必须继承此类
    
    接口规范：
        - predict(): 执行预测，返回预测结果DataFrame
        - load_model(): 加载模型文件
    
    使用示例：
        class MyPredictor(BasePredictor):
            def predict(self, data):
                # 实现预测逻辑
                pass
    """
    
    def __init__(self, 
                 model_path: Union[str, Path] = None,
                 info_path: Union[str, Path] = None):
        """
        初始化基类
        
        参数：
            model_path: 模型权重文件路径
            info_path: 模型配置信息路径
        """
        self.model_path = Path(model_path) if model_path else None
        self.info_path = Path(info_path) if info_path else None
        self.model = None
        self.config = {}
    
    @abstractmethod
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        执行预测（抽象方法，子类必须实现）
        
        参数：
            data: 输入数据DataFrame
        
        返回：
            pd.DataFrame: 预测结果
        """
        pass
    
    @abstractmethod
    def _load_model(self):
        """
        加载模型（抽象方法，子类必须实现）
        """
        pass
    
    def validate_input(self, data: pd.DataFrame, required_columns: list) -> bool:
        """
        验证输入数据
        
        参数：
            data: 输入数据
            required_columns: 必需的列名列表
        
        返回：
            bool: 验证是否通过
        
        异常：
            ValueError: 缺少必需列时抛出
        """
        missing = set(required_columns) - set(data.columns)
        if missing:
            raise ValueError(f"输入数据缺少必需列: {missing}")
        return True

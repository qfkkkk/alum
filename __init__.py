# -*- encoding: utf-8 -*-
"""
投药优化模块
功能：基于出水浊度预测进行最佳投药量优化

架构说明：
    - predictors/  预测层（浊度预测算法）
    - optimizers/  优化层（投药优化算法）
    - services/    服务层（API和定时调度）
    - utils/       工具函数
    - models/      模型权重文件
"""
from .predictors import create_predictor, TurbidityPredictor
from .optimizers import create_optimizer, DosingOptimizer
from .services import DosingPipeline

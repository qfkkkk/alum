# -*- coding: utf-8 -*-
"""
投药优化管道的核心服务类。
负责协调多池预测管理器 (PredictorManager) 和投药优化器 (Optimizer)。
提供数据适配、预测、优化以及全流程执行的功能。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

from predictors import create_manager
from optimizers import create_optimizer

class DosingPipeline:
    """
    投药优化管道
    
    功能说明：
        封装完整流程：多池浊度预测 → 投药优化
        
    设计模式 (Adapter Pattern)：
        通过 `_adapt_input` 方法隔离外部数据格式变化。
        无论外部 IO 接口返回什么格式，都在 `_adapt_input` 中转换为 Pipeline 标准输入。
    """
    
    def __init__(self, config_path: str = None, optimizer_type: str = 'dummy', optimizer_config: Dict = None):
        """
        初始化管道
        """
        # 初始化预测管理器
        self.predictor_manager = create_manager(config_path)
        
        # 初始化优化器
        self.optimizer = create_optimizer(optimizer_type, optimizer_config)
        
    def _adapt_input(self, raw_data: Any) -> Dict[str, np.ndarray]:
        """
        核心适配层
        功能：将外部未知的 IO 数据格式，转换为 Pipeline 标准输入格式。
        
        当前逻辑：
            假设 raw_data 已经是 Dict[str, np.ndarray] 或 Dict[str, pd.DataFrame]
            如果是其他格式（例如 PLC 原始 json），在这里写转换逻辑。
        """
        # TODO: 未来对接真实 IO 接口时，只需修改此方法
        if isinstance(raw_data, dict):
            # 简单的透传或转换
            return raw_data
        else:
            raise ValueError(f"Unsupported data format: {type(raw_data)}")

    def _extract_last_features(self, input_data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        从输入数据中提取最后时刻 (t) 的所有特征值
        
        参数:
            input_data: 适配后的输入数据 (Dict[pool_id, DataFrame/Array])
            
        返回:
            Dict[pool_id, Dict[feature_name, value]]
        """
        features_dict = {}
        
        # 优先从 config 获取特征列表，这比从 predictor 实例获取更直接
        feature_names = self.predictor_manager.config.get('features', [])
        
        # 如果 config 里也是空的，尝试 fallback 到 predictor 实例
        if not feature_names and self.predictor_manager.predictors:
             first_predictor = next(iter(self.predictor_manager.predictors.values()))
             feature_names = first_predictor.features
        
        for pool_id, data in input_data.items():
            last_row = None
            
            if isinstance(data, pd.DataFrame):
                # 如果是 DataFrame，直接取最后一行转字典
                last_row = data.iloc[-1].to_dict()
                # 过滤掉非数值列（如果需要）
                # last_row = {k: float(v) for k, v in last_row.items() if isinstance(v, (int, float))}
                
            elif isinstance(data, np.ndarray):
                # 如果是 numpy array，需要结合 feature_names
                if feature_names and data.shape[1] == len(feature_names):
                    last_vals = data[-1, :]
                    last_row = {name: float(val) for name, val in zip(feature_names, last_vals)}
                else:
                    # 没有特征名，只能用索引 'feat_0', 'feat_1'...
                    last_vals = data[-1, :]
                    last_row = {f"feat_{i}": float(val) for i, val in enumerate(last_vals)}
            
            if last_row:
                features_dict[pool_id] = last_row
                
        return features_dict

    def predict_only(self, 
                     input_data: Dict[str, np.ndarray], 
                     last_dt: datetime) -> Dict[str, Dict[str, float]]:
        """
        纯预测功能
        输入: {"pool_1": ndarray[60, 6], "pool_2": ndarray[60, 6], ...}
        返回: {pool_id: {time: val}} (6个点)
        """
        # 调用预测管理器 (Predictor 内部会处理标准化等)
        return self.predictor_manager.predict_all(input_data, last_dt)

    def optimize_only(self, 
                      predictions: Dict[str, Dict[str, float]], 
                      current_features: Dict[str, Dict[str, float]] = None) -> Dict[str, Dict[str, float]]:
        """
        纯优化功能
        输入: 
            predictions: 预测结果 {pool_id: {time: val}} (6个点)
            current_features: 当前全量特征 {pool_id: {feat: val}}
            
        返回: 推荐结果 {pool_id: {time: val}} (5个点)
        """
        results = {}
        
        # 1. 准备时间戳映射 (用于后续将优化结果 List 映射回 Dict)
        pool_timestamps = {} 
        
        for pool_id, time_dict in predictions.items():
            # 排序确保时间顺序
            sorted_items = sorted(time_dict.items(), key=lambda x: x[0])
            timestamps = [item[0] for item in sorted_items]
            pool_timestamps[pool_id] = timestamps
            
        # 2. 执行优化 (直接传入带时间戳的 predictions)
        # BaseOptimizer 接口已更新为接收 Dict[str, Dict[Any, float]]
        opt_results = self.optimizer.optimize(
            predictions, 
            current_features=current_features
        )
        
        # 3. 结果映射回时间戳 (取前5个时间点)
        for pool_id, rec_values in opt_results.items():
            if pool_id in pool_timestamps:
                timestamps = pool_timestamps[pool_id]
                rec_dict = {}
                # 优化器返回 5 个点，对应 timestamps 的前 5 个
                for i, rec_val in enumerate(rec_values):
                    if i < len(timestamps):
                        rec_dict[timestamps[i]] = rec_val
                results[pool_id] = rec_dict
                
        return results

    def run(self, 
            raw_data: Any, 
            last_dt: datetime) -> Dict[str, Any]:
        """
        全流程执行：适配 -> 预测 + 提取特征 -> 优化
        """
        results = {}
        
        # 1. 数据适配 (Adapter)
        clean_input = self._adapt_input(raw_data)
        
        # 2. 执行预测
        predictions = self.predict_only(clean_input, last_dt)
        
        # 3. 提取当前特征 (用于优化器决策)
        current_features = self._extract_last_features(clean_input)
        
        # 4. 执行优化
        recommendations = self.optimize_only(
            predictions, 
            current_features=current_features
        )
        
        # 5. 组装最终结果
        for pool_id in predictions.keys():
            pool_res = {
                'status': 'success',
                'predictions': predictions[pool_id],
                'recommendations': recommendations.get(pool_id, {}),
                'generated_at': datetime.now().isoformat()
            }
            results[pool_id] = pool_res
            
        return results

if __name__ == "__main__":
    pass

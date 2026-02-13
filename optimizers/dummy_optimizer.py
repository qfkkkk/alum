# -*- encoding: utf-8 -*-
from typing import List, Dict, Any
import numpy as np
from .base_optimizer import BaseOptimizer

class DummyOptimizer(BaseOptimizer):
    """
    模拟优化器，仅用于测试流程畅通性
    (简单线性关系用于演示)
    """
    
    def _optimize_core(self, prepared_data: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        核心优化逻辑（实现抽象方法）
        
        参数：
            prepared_data: 由 _prepare_input 返回的数据结构
                - predictions: 预测数据字典
                - current_features: 当前特征字典
            
        返回：
            {'pool_1': [dose_t+1, dose_t+2, ...], ...}
        """
        predictions = prepared_data.get('predictions', {})
        current_features = prepared_data.get('current_features', {})
        
        results = {}
        
        # 打印收到的特征（调试用）
        if current_features:
            print(f"DummyOptimizer received features for {list(current_features.keys())}")
        
        # 默认系数，可通过 config 覆盖
        coeff = self.config.get('coefficient', 15.0) 
        
        for pool_id, pred_dict in predictions.items():
            # 将字典转为排序后的列表
            sorted_items = sorted(pred_dict.items(), key=lambda x: x[0])
            preds = [item[1] for item in sorted_items]
            
            # 确保输入长度 >= 6
            if len(preds) < 6:
                print(f"Warning: Pool {pool_id} prediction length {len(preds)} < 6")
                valid_preds = preds[:6] if preds else []
            else:
                valid_preds = preds[:6]
                
            # 输出前5个时间步的投药量（控制时域）
            # 逻辑：基于当前投药量 + 浊度变化调整
            # 简单模拟: new_dosing = base_dosing + (pred_turb * factor)
            dosing = []
            
            # 尝试从 current_features 获取当前投药量
            base_val = coeff
            if current_features and pool_id in current_features:
                # 尝试从特征中获取当前投药量
                current_dose = current_features[pool_id].get('current_dose')
                if current_dose is not None:
                    base_val = float(current_dose)
            
            # 生成控制序列（5步）
            for i in range(self.control_horizon):
                if i < len(valid_preds):
                    # 假定: 浊度每增加0.1，投药增加0.5
                    d_val = base_val + (valid_preds[i] * 5.0)
                    dosing.append(round(d_val, 2))
                else:
                    dosing.append(round(base_val, 2))
            
            results[pool_id] = dosing
            
        return results


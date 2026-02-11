# -*- encoding: utf-8 -*-
from typing import List, Dict, Any
import numpy as np
from .base_optimizer import BaseOptimizer

class DummyOptimizer(BaseOptimizer):
    """
    模拟优化器，仅用于测试流程畅通性
    (简单线性关系用于演示)
    """
    
    def optimize(self, 
                 predictions: Dict[str, Dict[Any, float]], 
                 current_features: Dict[str, Dict[str, float]] = None) -> Dict[str, List[float]]:
        results = {}
        
        # 打印收到的特征（调试用）
        if current_features:
            print(f"DummyOptimizer received features for {list(current_features.keys())}")
            # 示例：打印 pool_1 的 pH 值（如果存在）
            if 'pool_1' in current_features:
                feats = current_features['pool_1']
                # 假设特征名可能不一样，这里只打印 keys
                # print(f"  Pool 1 feature keys: {list(feats.keys())}")
        
        # 默认系数，可通过 config 覆盖
        coeff = self.config.get('coefficient', 15.0) 
        
        for pool_id, pred_dict in predictions.items():
            # 将字典转为排序后的列表
            sorted_items = sorted(pred_dict.items(), key=lambda x: x[0])
            preds = [item[1] for item in sorted_items]
            
            # 确保输入长度 >= 6
            if len(preds) < 6:
                print(f"Warning: Pool {pool_id} prediction length {len(preds)} < 6")
                valid_preds = preds[:6]
            else:
                valid_preds = preds[:6]
                
            # 输出前5个时间步的投药量
            # 逻辑：基于当前投药量 + 浊度变化调整
            # 简单模拟: new_dosing = current_dosing + (pred_turb * factor)
            dosing = []
            
            # 使用增量模拟
            # t+1 时刻基于 t 时刻投药量调整
            # t+2 ~ t+5 基于前一时刻调整
            # 由于通过参数移除了 current_dosing，我们尝试从 current_features 获取，
            # 或者直接使用默认值 coeff
            base_val = coeff
            if current_features and pool_id in current_features:
                # 尝试从特征中获取投药量 (假设特征里有 'dosing' 或类似的)
                # 这里为了简单，如果有特征就稍微调整下 base_val
                # base_val += 1.0 
                pass
            
            for i in range(5):
                if i < len(valid_preds):
                    # 假定: 浊度每增加0.1，投药增加0.5
                    d_val = base_val + (valid_preds[i] * 5.0)
                    dosing.append(round(d_val, 2))
                else:
                    dosing.append(round(base_val, 2))
            
            results[pool_id] = dosing
            
        return results

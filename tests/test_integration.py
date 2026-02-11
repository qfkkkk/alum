import unittest
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from predictors.predictor_manager import TurbidityPredictorManager
from utils.config_loader import load_config

class TestPredictorIntegration(unittest.TestCase):
    
    def setUp(self):
        self.last_dt = datetime(2026, 2, 11, 12, 0, 0)
        self.config = load_config() # 加载默认 configs/app.yaml
        self.seq_len = self.config.get('seq_len', 60)
        self.n_features = len(self.config.get('features', []))
        
        # 构造随机输入数据
        self.input_data = np.random.rand(self.seq_len, self.n_features).astype(np.float32)

    def test_real_model_loading(self):
        """测试真实模型加载与推理 (集成测试)"""
        print("\n=== 集成测试: 加载真实模型 ===")
        
        manager = TurbidityPredictorManager() # 使用默认配置
        
        # 打印已加载的池子
        enabled_pools = manager.enabled_pools
        print(f"已启用池子: {enabled_pools}")
        
        if not enabled_pools:
            print("警告: 没有启用的池子，跳过测试")
            return

        # 构造多池数据
        data_dict = {
            pool_name: self.input_data 
            for pool_name in enabled_pools
        }
        
        # 执行预测
        try:
            results = manager.predict_all(data_dict, self.last_dt)
            print(results)
            print(data_dict)
            # 验证结果
            for pool_name, res in results.items():
                self.assertTrue(len(res) > 0)
                first_key = list(res.keys())[0]
                first_val = res[first_key]
                print(f"{pool_name} 预测成功: {first_key} -> {first_val}")
                self.assertIsInstance(first_val, float)
                
            print("集成测试验证通过！")
            
        except Exception as e:
            self.fail(f"集成测试失败: {e}")

if __name__ == '__main__':
    unittest.main()
